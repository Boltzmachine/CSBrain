#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-flipTT-jit
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Frame-flip equivariance TRAIN aug (--frame_flip_aug, 2x-batch concat) +
# matching TEST-time aug (--frame_flip_tta) + temporal-jitter train aug
# (--temporal_jitter N: random ±N-sample time shift of the crop window). 4-class
# softmax (default), seg0. The two flip views each get their own jitter draw;
# TTA at eval is jitter-free (nominal crop). Separate CSV.

FOUNDATION_DIR="${1:?foundation ckpt path required}"
SEED="${2:-42}"
JITTER="${3:-5}"
NUM_WORKERS="${4:-4}"
FLIP_PROB="${5:-0.5}"
SEGMENT_INDEX="${6:-0}"
HIGHPASS_HZ="${7:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}_hp${HIGHPASS_HZ}_flipTTjit${JITTER}"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv "outputs/finetune_results_flipTT_jit${JITTER}.csv" \
    --frame_flip_aug \
    --frame_flip_prob "$FLIP_PROB" \
    --frame_flip_tta \
    --temporal_jitter "$JITTER" \
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
