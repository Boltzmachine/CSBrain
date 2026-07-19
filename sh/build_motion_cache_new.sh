#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=motion_new
#SBATCH --output=outputs/slurms/%j.out

# Per-anchor motion scores for the NEWLY-VIDEOED EgoBrain subjects
# (P0025-P0040), written to <emb_grid_dir>/_motioncache/<sub>_<space>_<metric>_d<step>.npy
#
# Production (sh/pretrain_worldmodel.sh) uses:
#   --egobrain_motion_resample_space patch  --egobrain_motion_resample_metric cos
# and motion_step = (n_windows-1)*train_step = (max_horizon+1-1)*1 = 5  -> "d5".
# So the file production actually reads is <sub>_patch_cos_d5.npy. We build both
# spaces (patch,pixel) at d5 to match the coverage the existing 24 already have;
# each space run emits BOTH metrics (l1 + cos).
#
# The existing 24 subjects are passed over twice: an explicit --subjects list
# (not "all"), and the builder's own skip-if-exists guard. Nothing existing is
# rewritten. Cost is HDF5 decompression of the fp32 grid embeddings (~690 GB for
# the 16 new subjects) -> CPU/IO bound, scales with cores.
#
# Self-activates conda in a zsh LOGIN shell (project_sbatch_env_inherit).
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
SUBS=$(python3 -c "print(','.join('P%04d'%i for i in range(25,41)))")
exec zsh -lc "
  module load miniconda
  conda activate cbramod
  echo '[motion] subjects: $SUBS'
  exec python -m scripts.build_egobrain_motion_cache \
      --subjects '$SUBS' \
      --spaces patch,pixel \
      --step_slots 5 \
      --workers 16
"
