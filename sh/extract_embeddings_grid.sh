#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --job-name=emb_grid_extract
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Pre-compute frozen DINOv2 embeddings of the CONTINUOUS, time-keyed EgoBrain
# frame grid (datasets/egobrain_extract_embeddings_grid.py), so world-model
# pretraining with --egobrain_use_grid_embeddings loads cls/grid (+flip) per
# 0.2 s slot instead of running the encoder live. PREREQUISITE: the grid FRAME
# cache must exist first (sbatch sh/extract_frames_grid.sh) — this reads it and
# encodes one (slot x 2 orientations) forward per slot. ~460 GB float16 for
# dinov2-base @ grid_s=0.2 across 24 video subjects (the disk has ~11 TB free).
# The cache dir name encodes only vision_encoder/frame_size/grid_s.
#
# Self-activates conda inside a zsh LOGIN shell (see feedback_shell_and_remote /
# project_sbatch_env_inherit). grid_s + --vision_encoder MUST match the frame
# grid and the training run.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc 'module load miniconda; conda activate cbramod; exec python -m datasets.egobrain_extract_embeddings_grid \
    --data_dir data/EgoBrain \
    --subjects all \
    --vision_encoder facebook/dinov2-base \
    --grid_s 0.2 \
    --dtype float16 \
    --batch_size 256'
