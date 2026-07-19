#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=flipdiscrim
#SBATCH --output=outputs/slurms/%j.out

# Empirical judgement on diag_flip_discrim_acc: replace the lateral half of
# global_rep with an EEG-BLIND CONSTANT and see whether the diagnostic survives.
# NOTE: conda must be active BEFORE sbatch.

set -u
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u scripts/probe_flip_discrim_ablation.py \
    --ckpt outputs/wm-new-gradneg/epoch10_loss2.3018484115600586.pth \
    --tag wm-new-gradneg_ep10 --n_clips 3072

python -u scripts/probe_flip_discrim_ablation.py \
    --ckpt outputs/wm-new-biggradneg/epoch10_loss2.378716230392456.pth \
    --tag wm-new-biggradneg_ep10 --n_clips 3072

echo ALL DONE
