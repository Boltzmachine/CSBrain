#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-noequi-seg
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# noequiv (no-equivariance baseline) at a chosen time segment. Plain protocol,
# 4-class softmax head (default), no augmentation, no TTA (a no-op anyway on a
# non-frame-averaging backbone). --segment_index selects which 1s window of the
# 4s PhysioNet trial is cropped (0=s1 .. 3=s4). Writes to a SEGMENT-TAGGED CSV.

FOUNDATION_DIR="${1:?foundation ckpt path required}"
SEED="${2:-42}"
SEGMENT_INDEX="${3:-1}"
NUM_WORKERS="${4:-4}"
HIGHPASS_HZ="${5:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}_hp${HIGHPASS_HZ}_noequi"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv "outputs/finetune_results_noequi_seg${SEGMENT_INDEX}.csv" \
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
