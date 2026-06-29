#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-flipreconTT-seg
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# fliprecon + TTA at a chosen time segment. Frame-native equivariance flip:
# TRAIN aug (--frame_flip_aug, 2x-batch concat) + matching TEST-time aug
# (--frame_flip_tta). 4-class softmax head (default). --segment_index selects
# which 1s window of the 4s PhysioNet trial is cropped (0=s1 .. 3=s4). Writes to
# a SEGMENT-TAGGED CSV so later-segment rows stay separable from the seg0 runs.

FOUNDATION_DIR="${1:?foundation ckpt path required}"
SEED="${2:-42}"
SEGMENT_INDEX="${3:-1}"
NUM_WORKERS="${4:-4}"
FLIP_PROB="${5:-0.5}"
HIGHPASS_HZ="${6:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}_hp${HIGHPASS_HZ}_flipreconTT"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv "outputs/finetune_results_fliprecon_tt_seg${SEGMENT_INDEX}.csv" \
    --frame_flip_aug \
    --frame_flip_prob "$FLIP_PROB" \
    --frame_flip_tta \
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
