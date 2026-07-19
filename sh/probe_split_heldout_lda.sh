#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --job-name=heldout-lda
#SBATCH --output=outputs/slurms/%j.out

# The ONE downstream readout that is scale-free, out-of-sample and bias-free:
# LDA fitted on the 70 TRAIN subjects, scored on the 20 HELD-OUT test subjects.
# Null = permute the TRAIN labels within subject, refit, rescore the untouched test set.
# Plus the decisive control the earlier probes were missing: the SAME trained backbone
# with the learned gate REPLACED BY A CONSTANT SCALAR at its own mean (0.0725) -- this
# separates what the frame-averaging ARCHITECTURE buys from what the LEARNED GATE buys.
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for w in trained constgate randinit; do
    python -u scripts/probe_split_heldout_lda.py "$w"
done
