#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --job-name=csbrain-ft-kaya5f
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Kaya2018 5F (5-class five-finger MI) finetune, STANDARD protocol -- identical
# to FineMI/PhysioNet-MI plain eval: world-model checkpoint + allowlist,
# frame-averaging pinned, input cropped to the pretrained 1s window, 5-class
# softmax. Kaya trials are 1 s (n_win=1).

FOUNDATION_DIR="${1:-outputs/wm-dino-dense/epoch10_loss2.1733548641204834.pth}"
SEED="${2:-42}"
NUM_WORKERS="${3:-4}"
SEGMENT_INDEX="${4:-0}"
HIGHPASS_HZ="${5:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="Kaya5F_${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}"

python finetune_main.py \
    --model Align \
    --downstream_dataset Kaya5F \
    --datasets_dir data/preprocessed/kaya5f \
    --num_of_classes 5 \
    --model_dir outputs/CSBrain/finetune_CSBrain_Kaya5F \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv outputs/finetune_results_kaya5f.csv \
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
