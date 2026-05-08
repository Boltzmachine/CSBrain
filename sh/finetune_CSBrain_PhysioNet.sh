#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)

LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python finetune_main.py  \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir outputs/pretrain_CSBrain/epoch26_loss0.0015017741825431585.pth \
    --model CSBrain \
    --use_pretrained_weights \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.00005 

wait

# 记录任务完成时间到日志文件
echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"


