#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=csbrain-finetune
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

SCRIPT_DIR=$(dirname "$0")

LOG_DIR="log"
mkdir -p "$LOG_DIR"

LOG_FILE_NAME=$(basename "$0" .sh)

LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"


# python finetune_main.py \
#     --model Align \
#     --downstream_dataset PhysioNet-MI \
#     --datasets_dir data/preprocessed/physionet_mi \
#     --num_of_classes 4 \
#     --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
#     --foundation_dir outputs/Align-alljoined-mask/epoch22_loss4.793579578399658.pth \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --use_pretrained_weights \
#     --in_dim 40 \
#     --out_dim 40 \
#     --d_model 40 \
#     --seq_len 20 \
#     --nhead 4 \
#     --n_layer 24 \
#     --dropout 0.3 \
#     --weight_decay  0.01 \
#     --lr 0.00005 

FOUNDATION_DIR="outputs/worldmodel-mix-cinebrain-v4/epoch13_loss0.5890316367149353.pth"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv outputs/finetune_results.csv \
    --use_pretrained_weights \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.00005


# VISION_ENCODER="facebook/dinov2-base"   # swap to DINOv3 id once access is granted
# IMAGE_MODE="raw"                        # 'raw' matches the paper; 'spectrogram' = STFT grid
# WANDB_RUN_NAME="dinov3eeg_$(basename ${VISION_ENCODER})_${IMAGE_MODE}_lora"

# python finetune_main.py \
#     --model DINOv3EEG \
#     --downstream_dataset PhysioNet-MI \
#     --datasets_dir data/preprocessed/physionet_mi \
#     --num_of_classes 4 \
#     --model_dir outputs/CSBrain/finetune_DINOv3EEG_PhysioNet \
#     --vision_encoder "$VISION_ENCODER" \
#     --image_mode "$IMAGE_MODE" \
#     --use_lora \
#     --lora_rank 256 \
#     --lora_alpha 32 \
#     --wandb_run_name "$WANDB_RUN_NAME" \
#     --results_csv outputs/finetune_results.csv \
#     --dropout 0.1 \
#     --weight_decay 0.01 \
#     --lr 0.0001

wait

# 记录任务完成时间到日志文件
echo "All tasks completed at $(date)" | tee -a "$LOG_FILE"


