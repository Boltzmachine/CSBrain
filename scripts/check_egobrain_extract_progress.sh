#!/usr/bin/env bash
# Show progress of the EgoBrain frame-extraction job.
#
#   ./scripts/check_egobrain_extract_progress.sh
#
# Reports, for the default extract cache dir:
#   - completed chapter-stitched subjects (and their h5 size + coverage)
#   - in-progress subjects (active .tmp files growing right now)
#   - stale .tmp files (likely from a worker that died)
#   - subjects with no file yet
#   - whether the extract Python process is still alive
#   - rough ETA based on completion rate
#
# Override the cache dir or subject set with:
#   CACHE_DIR=... SUBJECTS="P0001,P0002" ./scripts/check_egobrain_extract_progress.sh
#
# Read-only: never deletes or modifies files.

set -u

# -------- config --------
CACHE_DIR="${CACHE_DIR:-data/EgoBrain/cache_frames_facebook_dinov2-base_w1.0s1.0_e0.5_nw2_sz224}"
# How long a .tmp can sit without growing before we flag it as stale.
# The new chapter-sequential extractor accumulates frames in memory and
# bulk-writes at the very end, so .tmp stays at ~96 bytes for the
# ENTIRE decode (often 10-25 min per subject) and only jumps to its
# final size at flush. Default 30 min handles that; the actual "dead
# worker" signal in this case is "no python process alive" (see
# bottom of script).
ACTIVE_AGE_S="${ACTIVE_AGE_S:-1800}"
# Default to the 24 EgoBrain subjects that ship with video.
if [ -z "${SUBJECTS:-}" ]; then
  SUBJECTS=$(seq -f "P%04g" 1 24 | paste -sd,)
fi

# -------- helpers --------
human() {                                # bytes -> human-readable
  numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "$1"
}

# -------- pre-flight --------
if [ ! -d "$CACHE_DIR" ]; then
  echo "cache dir does not exist: $CACHE_DIR"
  echo "(set CACHE_DIR=... to override)"
  exit 0
fi

now=$(date +%s)
echo "EgoBrain frame extract progress  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "cache dir: $CACHE_DIR"
echo

# -------- 1. completed (.h5) — only mark each subject; filter to SUBJECTS later --------
declare -A done_sub
declare -A done_size
while IFS= read -r f; do
  [ -z "$f" ] && continue
  sub=$(basename "$f" .h5)
  done_sub["$sub"]=1
  done_size["$sub"]=$(stat -c '%s' "$f")
done < <(ls "$CACHE_DIR"/*.h5 2>/dev/null)

# -------- 2. tmp files: active vs stale --------
declare -A tmp_sub
declare -A tmp_mtime
declare -A tmp_size
active_count=0
stale_count=0
for f in "$CACHE_DIR"/*.tmp; do
  [ -f "$f" ] || continue
  sub=$(basename "$f" .h5.tmp)
  tmp_sub["$sub"]=1
  mtime=$(stat -c '%Y' "$f")
  size=$(stat -c '%s' "$f")
  tmp_mtime["$sub"]=$mtime
  tmp_size["$sub"]=$size
  age=$((now - mtime))
  if [ "$age" -le "$ACTIVE_AGE_S" ]; then
    active_count=$((active_count + 1))
  else
    stale_count=$((stale_count + 1))
  fi
done

# -------- 3. categorize the SUBJECTS list --------
declare -A all_sub
IFS=',' read -ra subs_arr <<<"$SUBJECTS"
total=${#subs_arr[@]}
filtered_done_count=0
filtered_done_size=0
filtered_active_count=0
filtered_stale_count=0
pending_count=0
for s in "${subs_arr[@]}"; do
  all_sub["$s"]=1
  if [ "${done_sub[$s]:-}" = "1" ]; then
    filtered_done_count=$((filtered_done_count + 1))
    filtered_done_size=$((filtered_done_size + ${done_size[$s]}))
    continue
  fi
  if [ "${tmp_sub[$s]:-}" = "1" ]; then
    age=$((now - tmp_mtime[$s]))
    if [ "$age" -le "$ACTIVE_AGE_S" ]; then
      filtered_active_count=$((filtered_active_count + 1))
    else
      filtered_stale_count=$((filtered_stale_count + 1))
    fi
    continue
  fi
  pending_count=$((pending_count + 1))
done

# -------- 4. headline summary --------
printf "  done:        %2d / %d  (%s)\n" "$filtered_done_count" "$total" "$(human "$filtered_done_size")"
printf "  in-progress: %2d  (.tmp updated within %ds)\n" "$filtered_active_count" "$ACTIVE_AGE_S"
printf "  stale:       %2d  (.tmp older than %ds — worker likely died)\n" "$filtered_stale_count" "$ACTIVE_AGE_S"
printf "  pending:     %2d  (no h5 or tmp yet)\n" "$pending_count"
echo

# -------- 5. detail listings (only print if any in that bucket) --------
print_bucket() {
  # $1 = label, rest = lines
  local label="$1"; shift
  [ "$#" -eq 0 ] && return 0
  echo "$label"
  printf "  %s\n" "$@"
  echo
}

done_lines=()
for s in "${subs_arr[@]}"; do
  [ "${done_sub[$s]:-}" = "1" ] || continue
  f="$CACHE_DIR/$s.h5"
  size=$(stat -c '%s' "$f")
  done_lines+=("$s  $(human "$size")")
done
print_bucket "Completed (.h5):" "${done_lines[@]}"

active_lines=()
stale_lines=()
for s in "${subs_arr[@]}"; do
  [ "${tmp_sub[$s]:-}" = "1" ] || continue
  m=${tmp_mtime[$s]}
  age=$((now - m))
  size=$(human "${tmp_size[$s]}")
  ts=$(date -d "@$m" '+%H:%M:%S')
  line=$(printf "%s  %-8s  last write %s  (%ds ago)" "$s" "$size" "$ts" "$age")
  if [ "$age" -le "$ACTIVE_AGE_S" ]; then
    active_lines+=("$line")
  else
    stale_lines+=("$line")
  fi
done
print_bucket "In-progress (.tmp active):" "${active_lines[@]}"
print_bucket "Stale (.tmp not growing — re-run will redo):" "${stale_lines[@]}"

pending_lines=()
for s in "${subs_arr[@]}"; do
  [ "${done_sub[$s]:-}" = "1" ] && continue
  [ "${tmp_sub[$s]:-}" = "1" ] && continue
  pending_lines+=("$s")
done
if [ "${#pending_lines[@]}" -gt 0 ]; then
  # Join into one comma-separated line so the bucket is not 20 lines tall.
  joined=$(IFS=','; echo "${pending_lines[*]}")
  print_bucket "Pending (no file yet):" "$joined"
fi

# -------- 6. worker process alive? --------
echo "Worker processes:"
worker_pids=$(pgrep -f "egobrain_extract_frames" 2>/dev/null | tr '\n' ' ')
if [ -z "$worker_pids" ]; then
  echo "  (no python egobrain_extract_frames process found — extract not running)"
else
  echo "  pids: $worker_pids"
  ps -o pid,etime,rss,%cpu,comm --no-headers -p $worker_pids 2>/dev/null | \
    awk '{printf "  pid=%-6s elapsed=%-8s rss=%-10s cpu=%-5s %s\n", $1, $2, $3"k", $4"%", $5}'
fi
echo

# -------- 7. ETA (very rough) --------
remaining=$((pending_count + filtered_active_count + filtered_stale_count))
if [ "$filtered_done_count" -gt 0 ] && [ "$remaining" -gt 0 ]; then
  # Find oldest completion mtime as a proxy for "start of this run".
  first_done_mtime=$(ls -t "$CACHE_DIR"/*.h5 2>/dev/null | tail -1 | xargs -r stat -c '%Y' 2>/dev/null)
  if [ -n "$first_done_mtime" ]; then
    elapsed=$((now - first_done_mtime))
    per_sub=$((elapsed / filtered_done_count))
    eta_s=$((per_sub * remaining / 2))      # 2 workers default
    printf "ETA (rough): ~%dm remaining  (rate %ds/subject, %d subjects left, 2 workers)\n" \
      "$((eta_s / 60))" "$per_sub" "$remaining"
  fi
fi
