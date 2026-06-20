#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=csbrain-finetune-finemi
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)
LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

FOUNDATION_DIR="outputs/wm-mix-egobrain-nobands/epoch15_loss0.7349681854248047.pth"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="FineMI_${CKPT_NAME}_${EPOCH}"

python finetune_main.py \
    --model Align \
    --downstream_dataset FineMI \
    --datasets_dir data/preprocessed/finemi \
    --num_of_classes 8 \
    --model_dir outputs/CSBrain/finetune_CSBrain_FineMI \
    --foundation_dir "$FOUNDATION_DIR" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv outputs/finetune_results.csv \
    --use_pretrained_weights \
    --dropout 0.3 \
    --weight_decay 0.01 \
    --lr 0.00005

wait

echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"
