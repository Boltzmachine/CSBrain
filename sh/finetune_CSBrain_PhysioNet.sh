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

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)

LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

python finetune_main.py  \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir outputs/ours_all_zerosync_16/epoch17_loss0.017739474773406982.pth \
    --model OurModel \
    --use_pretrained_weights \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.00005 

# python finetune_main.py  \
#     --downstream_dataset PhysioNet-MI \
#     --datasets_dir data/preprocessed/physionet_mi \
#     --num_of_classes 4 \
#     --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
#     --foundation_dir outputs/llm_vq/epoch21_loss0.03721848130226135.pth  \
#     --model LLMVQ \
#     --use_pretrained_weights \
#     --dropout 0.3 \
#     --weight_decay  0.01 \
#     --lr 0.00005 

wait

# 记录任务完成时间到日志文件
echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"


