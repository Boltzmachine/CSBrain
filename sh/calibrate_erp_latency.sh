#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=erp_calib
#SBATCH --output=outputs/slurms/%j.out

# Empirically calibrate --egobrain_erp_latency_s as a LAG-IDENTIFIABILITY study
# (scripts/calibrate_erp_latency.py). CPU-only, no GPU. Finds the physical lag
# Delta = frame_time - eeg_window_centre at which EEG best decodes the continuous
# hand-movement intensity, then either localises it (null-cleared, subject-
# bootstrapped CI) or proves it is NOT IDENTIFIABLE and refuses to move the knob.
#
# IMPORTANT (project memory: sbatch env inheritance): sh/ scripts do NOT self-
# activate conda, and a job that fails on numpy import still shows COMPLETED.
# ACTIVATE cbramod BEFORE `sbatch sh/calibrate_erp_latency.sh`:
#
#     module load miniconda && conda activate cbramod
#     sbatch sh/calibrate_erp_latency.sh
#
# Per-subject filtered-power cumsum arrays cache under --cache_dir so re-runs
# with a different lag grid are cheap. Outputs: outputs/eval_tables.md (appended
# dated verdict), outputs/calibrate_erp_latency_<date>.json, figs/.

set -euo pipefail

# 8 workers x 2 BLAS threads == the 16 requested CPUs (avoids oversubscription).
export EEG_CAL_BLAS_THREADS=2

python -u -m scripts.calibrate_erp_latency \
    --subjects all \
    --lag_step 0.05 \
    --slot_stride 1 \
    --window_variant sharp \
    --n_null 200 \
    --n_boot 2000 \
    --workers 8 \
    "$@"

# Smoke (exercise the whole path in ~1.5 min before the full run):
#   python -u -m scripts.calibrate_erp_latency \
#       --subjects P0001,P0002 --lag_step 0.1 --slot_stride 10
