#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=dinov3-linear-probe
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

SCRIPT_DIR=$(dirname "$0")
LOG_DIR="log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(basename "$0" .sh).log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

VISION_ENCODER="facebook/dinov2-base"   # swap to DINOv3 id once access is granted
IMAGE_MODE="raw"                        # 'raw' matches the paper; 'spectrogram' = STFT grid
PROBE="logreg"                          # 'logreg' (sklearn, paper) or 'sgd' (nn.Linear)
CACHE_TAG=$(echo "$VISION_ENCODER" | tr '/' '_')

python scripts/linear_probe_dinov3_eeg.py \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --vision_encoder "$VISION_ENCODER" \
    --image_mode "$IMAGE_MODE" \
    --probe "$PROBE" \
    --logreg_Cs "[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]" \
    --batch_size 128 \
    --feature_cache "outputs/linear_probe_cache/${CACHE_TAG}_${IMAGE_MODE}_physionet.pt" \
    --results_csv outputs/linear_probe_results.csv

wait

echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"
