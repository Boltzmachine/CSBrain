#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=ego_fgrid
#SBATCH --output=outputs/slurms/%A_%a.out
#SBATCH --array=1-40%16

# Sharded grid frame extraction: ONE EgoBrain subject per array task, so the
# ~48 h of GoPro decode (decord must decode THROUGH each ~2 h recording to reach
# the 0.2 s grid frames, regardless of grid density) parallelizes across nodes
# instead of thrashing 12 readers on one node. Each task pins decord to its own
# 8-CPU allocation (--decord_threads), so no thread oversubscription.
#
# Array index k -> subject P<k:04d>. 1-40 covers all 40 subjects; the 16
# no-video ones (P0025-P0040) return 'no_video' in seconds (exit 0). %16 caps
# concurrency to keep GPFS I/O sane. Writes to the SAME shared cache dir
# (cache_frames_grid_<enc>_g0.2_sz224/<sub>.h5); per-subject files, no conflict.
#
# Self-activates conda in a zsh LOGIN shell (project_sbatch_env_inherit).
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc '
  module load miniconda
  conda activate cbramod
  SUBJ=$(printf "P%04d" "$SLURM_ARRAY_TASK_ID")
  echo "[array task $SLURM_ARRAY_TASK_ID] subject=$SUBJ threads=$SLURM_CPUS_PER_TASK"
  exec python -m datasets.egobrain_extract_frames_grid \
      --data_dir data/EgoBrain \
      --subjects "$SUBJ" \
      --vision_encoder facebook/dinov2-base \
      --grid_s 0.2 \
      --fs_out 200 \
      --num_workers 1 \
      --decord_threads "$SLURM_CPUS_PER_TASK"
'
