#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --job-name=ego_frames_grid
#SBATCH --output=outputs/slurms/%j.out

# Pre-decode EgoBrain GoPro frames onto a CONTINUOUS, time-keyed grid (one
# frame every --grid_s seconds of EEG-clock time) into per-subject HDF5 —
# datasets/egobrain_extract_frames_grid.py. This is the knob-agnostic
# successor to sh/build_egobrain_vjepa_cache.sh / extract_frames: the cache
# does NOT encode window_s/stride_s/erp_latency_s/n_windows/clip_s, so the
# dataset (EgoBrainDataset(use_frame_grid=True)) can sample EEG windows at any
# 0.2 s-snapped offset across the whole recording and still find a frame.
# Only --vision_encoder (resize/crop + frame size) and --grid_s change the
# bytes, so they alone go in the dir name (cache_frames_grid_<enc>_g<g>_sz<sz>).
#
# CPU-only (no GPU); one worker per subject (peak ~3 GB/worker). Reads the
# existing EEG cache (cache_eeg_<fs>hz/*/clips.json) for n_clips/clip_s/video
# metadata; does NOT touch any existing processed file. ~66 GB for dinov2
# @ grid_s=0.2 across 24 video subjects.
#
# Self-activates conda inside a zsh LOGIN shell (the pattern proven to work in
# this environment; see feedback_shell_and_remote / project_sbatch_env_inherit).
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
exec zsh -lc 'module load miniconda; conda activate cbramod; exec python -m datasets.egobrain_extract_frames_grid \
    --data_dir data/EgoBrain \
    --subjects all \
    --vision_encoder facebook/dinov2-base \
    --grid_s 0.2 \
    --fs_out 200 \
    --num_workers 12'
