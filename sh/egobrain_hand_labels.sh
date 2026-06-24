#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
# WiLoR is a small model and this job is decode-bound, so a modest GPU is
# plenty (48 GB l40s) AND far less contended than the h100s — schedules sooner.
# a40 works equally well; swap the type here if l40s is busy.
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=hand_labels
#SBATCH --output=outputs/slurms/%j.out
# QOS must grant the requested GPU type. qos_nmi allots l40s=12 (also
# h100/h200/a100) but a40=0; qos_ying_rex allots a40=4 but l40s=0. So:
# l40s -> qos_nmi, a40 -> qos_ying_rex. Keep this in sync with --gres above.
#SBATCH --qos=qos_nmi

# Derive per-window hand-movement labels (0=left, 1=right, 2=both, -1=none —
# PhysioNet-MI aligned, "both feet"=3 excluded) + a continuous "drasticness"
# motion metric for EgoBrain, by running WiLoR on the egocentric GoPro frames.
# See datasets/egobrain_hand_labels.py. Install the backend FIRST on a login
# node: `bash sh/install_wilor.sh`.
#
# Window geometry MUST match the world-model run so the labels key-align with
# the EEG/frame caches — defaults below mirror sh/pretrain_worldmodel.sh
# (window_s 1.0 / stride_s 1.0 / erp_latency_s 0.5 / n_windows 2, clip_s 4.0).
# Output: data/EgoBrain/cache_hand_labels_wilor_w1.0s1.0_e0.5_nw2_k7_c4.0_fs200/
# (NEW dir; never overwrites the EEG cache, frame cache, or raw video. Reruns
# with a different config refuse to clobber unless --overwrite.)
#
# Only P0001-P0024 ship video, so only those get labels; P0025-P0040 report
# "no_video" and are skipped (matching egobrain_extract_frames).
#
# Calibration: --move_thresh (hand-lengths/s) sets the moving/still cut. The
# continuous metrics are always stored, so you can re-threshold offline with
# egobrain_hand_labels.recompute_labels WITHOUT rerunning this job. To tune,
# eyeball a handful of windows against the stored drasticness first.
#
# Per-subject parallelism: launch as an array (one subject per task), e.g.
#   sbatch --array=1-24 sh/egobrain_hand_labels.sh
# and the SLURM_ARRAY_TASK_ID below selects the subject.

# Uses the DEDICATED wilor env (built by sh/install_wilor.sh), NOT cbramod —
# WiLoR needs chumpy/numpy<1.24. The extractor only reads the EEG cache +
# video and writes HDF5, so it needs no cbramod training deps.
# sh/ scripts do NOT inherit the conda env from the submit shell — activate
# here or the job dies on the first import (and SLURM still marks it
# COMPLETED). See project_sbatch_env_inherit.
module load miniconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wilor

# Compute nodes are typically offline; model weights are already cached in the
# env (sh/install_wilor.sh warms them). Keep HF from trying the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Subject selection: array task -> single subject Pxxxx, else "all".
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SUBJECTS=$(printf 'P%04d' "${SLURM_ARRAY_TASK_ID}")
else
    SUBJECTS=all
fi

# Run as a DIRECT script (not `-m datasets.egobrain_hand_labels`): the wilor
# env lacks the training-only deps that datasets/__init__.py imports, and this
# module needs none of them.
python datasets/egobrain_hand_labels.py \
    --data_dir data/EgoBrain \
    --subjects "${SUBJECTS}" \
    --backend wilor \
    --window_s 1.0 \
    --stride_s 1.0 \
    --clip_s 4.0 \
    --fs_out 200 \
    --erp_latency_s 0.5 \
    --n_windows 2 \
    --frames_per_window 7 \
    --move_thresh 0.15 \
    --min_det_frac 0.4 \
    --handedness_source hybrid \
    --max_frame_width 1280 \
    --device cuda \
    --dtype float16
