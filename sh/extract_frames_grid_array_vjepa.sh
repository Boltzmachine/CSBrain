#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --job-name=ego_fgrid_vj
#SBATCH --output=outputs/slurms/%A_%a.out
#SBATCH --array=1-40%16

# Sharded grid frame extraction for V-JEPA 2 (256x256 crop) — same machinery as
# sh/extract_frames_grid_array.sh but --vision_encoder facebook/vjepa2-vitl-
# fpc64-256, so frames are resized shortest-edge 292 + center-crop 256 (the
# V-JEPA 2 video-processor geometry), NOT the DINOv2 224. Separate cache dir
# (cache_frames_grid_<enc>_g0.2_sz256/) — cannot reuse the 224 DINOv2 frames
# (different, larger center crop). 256-frame buffer is ~1.3x the 224 case
# -> mem 80G. One subject per task; decord pinned to the 8-CPU alloc.
#
# Array index k -> subject P<k:04d>. 1-40 covers all; the 16 no-video subjects
# (P0025-P0040) return 'no_video' instantly. Self-activates conda (zsh login).
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc '
  module load miniconda
  conda activate cbramod
  SUBJ=$(printf "P%04d" "$SLURM_ARRAY_TASK_ID")
  echo "[array task $SLURM_ARRAY_TASK_ID] subject=$SUBJ threads=$SLURM_CPUS_PER_TASK"
  exec python -m datasets.egobrain_extract_frames_grid \
      --data_dir data/EgoBrain \
      --subjects "$SUBJ" \
      --vision_encoder facebook/vjepa2-vitl-fpc64-256 \
      --grid_s 0.2 \
      --fs_out 200 \
      --num_workers 1 \
      --decord_threads "$SLURM_CPUS_PER_TASK"
'
