#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --job-name=csbrain-ft-forenzo-reg
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Forenzo2024 Continuous-Pursuit -- REGRESSION variant (2D cursor velocity
# x,y). Uses the dict-batch regression path (train_for_regression_dict) with
# MSE loss + corr/r2/rmse metrics. num_of_classes=2 == the 2 regression outputs.
# Same backbone/protocol as the classification variant.

FOUNDATION_DIR="${1:-outputs/wm-dino-dense/epoch10_loss2.1733548641204834.pth}"
SEED="${2:-42}"
NUM_WORKERS="${3:-4}"
SEGMENT_INDEX="${4:-0}"
HIGHPASS_HZ="${5:-0}"
EPOCHS="${6:-10}"   # ~401k windows -> reduced epochs (see cls variant)
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="Forenzo2024reg_${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}"

python finetune_main.py \
    --model Align \
    --downstream_dataset Forenzo2024Reg \
    --datasets_dir data/preprocessed/forenzo \
    --num_of_classes 2 \
    --model_dir outputs/CSBrain/finetune_CSBrain_Forenzo2024_reg \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --pre_cls_layernorm \
    --use_pretrained_weights \
    --use_initial_segment_only \
    --segment_index "$SEGMENT_INDEX" \
    --highpass_hz "$HIGHPASS_HZ" \
    --num_workers "$NUM_WORKERS" \
    --dropout 0.3 \
    --weight_decay  0.01 \
    --lr 0.00005

wait
echo "All tasks completed at $(date)"
