#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=csbrain-finetune
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_ying_rex
# Get the script directory
SCRIPT_DIR=$(dirname "$0")

# Create log directory if it doesn't exist
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# Get the script file name without the .sh extension
LOG_FILE_NAME=$(basename "$0" .sh)

# Set the log file path
LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

# Log the job start time
echo "Job started at $(date)" | tee -a "$LOG_FILE"

# Set CUDA device and run the Python fine-tuning script
# python finetune_main.py  \
#     --downstream_dataset BCIC-IV-2a \
#     --datasets_dir data/preprocessed/BCICIV2a \
#     --num_of_classes 4 \
#     --model_dir outputs/CSBrain/finetune_CSBrain_BCICIV2a \
#     --foundation_dir outputs/llm_vq/epoch21_loss0.03721848130226135.pth \
#     --model LLMVQ \
#     --use_pretrained_weights \
#     --dropout 0.3 \
#     --weight_decay  0.01 \
#     --lr 0.0001 

python finetune_main.py  \
    --downstream_dataset BCIC-IV-2a \
    --datasets_dir data/preprocessed/BCICIV2a \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_BCICIV2a \
    --foundation_dir outputs/ours_all_zerosync_16/epoch17_loss0.017739474773406982.pth \
    --model OurModel \
    --use_pretrained_weights \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.0001 

wait

# Log the task completion time
echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"
