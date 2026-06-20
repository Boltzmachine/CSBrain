#!/bin/bash
# Watcher: wait for the EgoBrain V-JEPA2 frame-cache job to finish, verify it
# completed cleanly, then submit ActionWorldModel training with the cbramod env.
# Auto-generated for an overnight unattended submit; safe to delete afterwards.
set -u
REPO=/gpfs/radev/pi/ying_rex/wq44/CSBrain
CACHE_JOB=2013443
CACHE="$REPO/data/EgoBrain/cache_frames_facebook_vjepa2-vitl-fpc64-256_w1.0s1.0_e0.5_nw2_sz256"
cd "$REPO" || exit 1

echo "[watch] waiting for cache job $CACHE_JOB to leave the queue ..."
while squeue -j "$CACHE_JOB" -h -o "%T" 2>/dev/null \
        | grep -qE "RUNNING|PENDING|COMPLETING|CONFIGURING"; do
    sleep 300
done

STATE=$(sacct -j "$CACHE_JOB" -n -o State 2>/dev/null | head -1 | tr -d ' ')
NDONE=$(ls "$CACHE"/*.h5 2>/dev/null | wc -l)
NTMP=$(ls "$CACHE"/*.h5.tmp 2>/dev/null | wc -l)
echo "[watch] cache job $CACHE_JOB ended: state='$STATE' .h5=$NDONE .tmp=$NTMP"

if echo "$STATE" | grep -q "COMPLETED" && [ "$NTMP" -eq 0 ] && [ "$NDONE" -ge 24 ]; then
    # Activate cbramod BEFORE sbatch so the training job inherits the env
    # (sh/ scripts don't self-activate; jobs die on numpy import otherwise).
    source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
    conda activate cbramod
    echo "[watch] env: python=$(which python)"
    NEWJOB=$(sbatch --parsable sh/pretrain_actionworldmodel.sh)
    echo "[watch] SUBMITTED training job: $NEWJOB"
    echo "$NEWJOB" > "$REPO/outputs/.action_wm_trainjob"
else
    echo "[watch] NOT submitting training: cache job not cleanly COMPLETED" \
         "(state='$STATE', .tmp=$NTMP, .h5=$NDONE). Investigate before resubmit."
fi
echo "[watch] done."
