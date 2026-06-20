#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --job-name=ego_vjepa_cache
#SBATCH --output=outputs/slurms/%j.out

# One-time prerequisite for sh/pretrain_actionworldmodel.sh: pre-decode the
# EgoBrain GoPro frames into per-subject HDF5 caches sized for V-JEPA 2
# (256x256), so training reads uint8 frames instead of live-decoding 4-5 GB
# MP4s (which OOM-kills DataLoader workers). CPU-only (no GPU needed); the
# extractor runs one worker per subject (peak ~3 GB/worker).
#
# Activate cbramod BEFORE sbatch so the env is inherited (sh/ scripts don't
# self-activate), e.g.:  conda activate cbramod && sbatch sh/build_egobrain_vjepa_cache.sh
#
# Window/stride/erp/n_windows MUST match pretrain_actionworldmodel.sh, or the
# dataset won't find the cache (the settings are encoded in the cache dir name).
python -m datasets.egobrain_extract_frames \
    --data_dir data/EgoBrain \
    --subjects all \
    --vision_encoder facebook/vjepa2-vitl-fpc64-256 \
    --window_s 1.0 \
    --stride_s 1.0 \
    --erp_latency_s 0.5 \
    --n_windows 2 \
    --num_workers 12
