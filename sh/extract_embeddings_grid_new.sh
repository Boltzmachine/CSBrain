#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=emb_grid_new
#SBATCH --output=outputs/slurms/%A_%a.out
#SBATCH --qos=qos_nmi
#SBATCH --array=25-40%4

# Grid DINOv2 embeddings for the NEWLY-VIDEOED EgoBrain subjects (P0025-P0040),
# which gained GoPro MP4s after the first 24 were cached. One subject per array
# task (index k -> P<k:04d>).
#
# WHY A SEPARATE SCRIPT (do not just run sh/extract_embeddings_grid.sh):
#  * That script passes `--subjects all` + `--dtype float16`. The existing 24
#    subjects on disk are **float32**, and the extractor's skip check compares
#    dtype -- so `all` + fp16 would MISS the skip and RE-ENCODE/OVERWRITE all
#    985 GB of existing embeddings. Here we pass exactly ONE explicit subject
#    per task, so the existing files are never opened for writing.
#  * `--dtype float32` matches the existing 24 -> one homogeneous cache.
#    (fp32 also selects lzf compression by default in the extractor.)
#
# MEMORY: the extractor holds grid_full = (n_slots, 2, 256, 768) in RAM.
# In fp32 that is ~62 GB for a 39k-slot subject and ~69 GB for P0035 (43.9k
# slots), plus ~7 GB of uint8 frames. The 64G of extract_embeddings_grid.sh
# would OOM on the big ones -> 128G here.
#
# PREREQUISITES (in order):
#   1. clips.json video metadata refreshed for P0025-P0040 (egobrain_preprocess)
#   2. grid FRAME cache built for them (sbatch --array=25-40%16 sh/extract_frames_grid_array.sh)
#
# Self-activates conda in a zsh LOGIN shell (project_sbatch_env_inherit).
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc '
  module load miniconda
  conda activate cbramod
  SUBJ=$(printf "P%04d" "$SLURM_ARRAY_TASK_ID")
  echo "[array task $SLURM_ARRAY_TASK_ID] embedding subject=$SUBJ (fp32)"
  exec python -m datasets.egobrain_extract_embeddings_grid \
      --data_dir data/EgoBrain \
      --subjects "$SUBJ" \
      --vision_encoder facebook/dinov2-base \
      --grid_s 0.2 \
      --dtype float32 \
      --batch_size 256
'
