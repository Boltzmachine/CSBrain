#!/bin/bash
# 5-epoch diagnostic finetune used to characterise which layers move.
# Mirrors sh/finetune_CSBrain_PhysioNet.sh but with --grad_analysis and
# --epochs 5. Intended for `ssh r4516u05n01` direct execution, not sbatch.
set -euo pipefail

cd /gpfs/radev/pi/ying_rex/wq44/CSBrain

LOG_DIR="log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/finetune_CSBrain_PhysioNet_gradanalysis.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

FOUNDATION_DIR="outputs/worldmodel-mix-cinebrain-v8-moe/epoch4_loss0.8311042785644531.pth"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_gradanalysis"

GRAD_DIR="outputs/CSBrain/finetune_CSBrain_PhysioNet_gradanalysis/grad_analysis"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet_gradanalysis \
    --foundation_dir "$FOUNDATION_DIR" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --use_pretrained_weights \
    --use_initial_segment_only \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.00005 \
    --epochs 5 \
    --grad_analysis \
    --grad_analysis_dir "$GRAD_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"
