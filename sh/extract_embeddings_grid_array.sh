#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --job-name=emb_grid
#SBATCH --output=outputs/slurms/%A_%a.out
#SBATCH --qos=qos_nmi
#SBATCH --array=1-40%8

# Sharded grid-embedding extraction: ONE subject per array task, so the 24
# per-subject DINOv2 encode+write runs parallelize across GPUs instead of one
# H100 grinding through them sequentially. FLOAT32, UNCOMPRESSED (--compression
# none: fastest random reads — no decompress at all; fp32 features barely
# compress anyway). fp32 doubles the in-RAM grid buffers (~66 GB for the
# largest subject) -> mem=128G. ~1 TB on disk across 24 video subjects.
#
# Array index k -> subject P<k:04d>. 1-40 covers all: P0001 already done ->
# 'skip'; P0025-P0040 have no frame cache -> 'no_frames' (instant); P0002-P0024
# do the work. Reads the (complete) grid frame cache; writes per-subject
# embedding HDF5 to the SAME shared dir (no conflict, per-subject files).
#
# Self-activates conda in a zsh LOGIN shell (project_sbatch_env_inherit).
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc '
  module load miniconda
  conda activate cbramod
  SUBJ=$(printf "P%04d" "$SLURM_ARRAY_TASK_ID")
  echo "[array task $SLURM_ARRAY_TASK_ID] embedding subject=$SUBJ"
  exec python -m datasets.egobrain_extract_embeddings_grid \
      --data_dir data/EgoBrain \
      --subjects "$SUBJ" \
      --vision_encoder facebook/dinov2-base \
      --grid_s 0.2 \
      --dtype float32 \
      --compression none \
      --batch_size 256
'
