#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --job-name=csbrain-ft-weibo2014
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Weibo2014 (7-class simple+compound limb motor imagery) finetune, STANDARD
# protocol -- identical to the FineMI/PhysioNet-MI plain eval: world-model
# checkpoint loaded via load_pretrain_checkpoint + allowlist, frame-averaging
# pinned (flip prob 0), input cropped to the pretrained 1s window
# (--use_initial_segment_only), 7-class softmax. No equivariance/flip aug.

FOUNDATION_DIR="${1:-outputs/wm-dino-dense/epoch10_loss2.1733548641204834.pth}"
SEED="${2:-42}"
NUM_WORKERS="${3:-4}"
SEGMENT_INDEX="${4:-0}"
HIGHPASS_HZ="${5:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="Weibo2014_${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}"

python finetune_main.py \
    --model Align \
    --downstream_dataset Weibo2014 \
    --datasets_dir data/preprocessed/weibo2014 \
    --num_of_classes 7 \
    --model_dir outputs/CSBrain/finetune_CSBrain_Weibo2014 \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv outputs/finetune_results_weibo2014.csv \
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
