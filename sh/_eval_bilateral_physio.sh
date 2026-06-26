#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-eval-bilat
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Bilateral two-bit head, NO augmentation. Same plain protocol as
# _eval_plain_physio.sh, but --bilateral_head swaps the 4-way softmax for two
# sigmoids [left_hand, right_hand] (left fist=[1,0], right=[0,1], both=[1,1],
# feet=[0,0]); BCE-trained, decoded back to 4 classes for metrics. Separate CSV.

FOUNDATION_DIR="${1:?foundation ckpt path required}"
SEED="${2:-42}"
NUM_WORKERS="${3:-4}"
SEGMENT_INDEX="${4:-0}"
HIGHPASS_HZ="${5:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}_hp${HIGHPASS_HZ}_bilat"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv outputs/finetune_results_bilateral.csv \
    --bilateral_head \
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
