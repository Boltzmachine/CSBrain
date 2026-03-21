#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=csbrain-finetune
#SBATCH --output=outputs/slurms/%j.out

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)

LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"


python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir outputs/Align-all-add-global/epoch20_loss4.769124984741211.pth \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --use_pretrained_weights \
    --in_dim 20 \
    --out_dim 20 \
    --d_model 20 \
    --seq_len 300 \
    --nhead 4 \
    --n_layer 24 \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.00005 

# python finetune_main.py  \
#     --downstream_dataset PhysioNet-MI \
#     --datasets_dir data/preprocessed/physionet_mi \
#     --num_of_classes 4 \
#     --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
#     --foundation_dir outputs/CSBrain-deep/epoch19_loss0.03231504559516907.pth  \
#     --use_pretrained_weights \
#     --dropout 0.3 \
#     --weight_decay  0.01 \
#     --lr 0.00005 

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


