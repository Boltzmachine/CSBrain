#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-eval-frameflip
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Frame-native equivariance flip-augmentation eval. Same plain protocol as
# _eval_plain_physio.sh but with --frame_flip_aug (training-only: present the
# batch mirrored via the frame-averaging forward + left<->right label remap).
# Tags the run name + writes to a separate CSV so it never collides with the
# no-aug rows. Only meaningful on a frame-averaging (equi) checkpoint.

FOUNDATION_DIR="${1:?foundation ckpt path required}"
SEED="${2:-42}"
FLIP_PROB="${3:-0.5}"
NUM_WORKERS="${4:-4}"
SEGMENT_INDEX="${5:-0}"
HIGHPASS_HZ="${6:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}_hp${HIGHPASS_HZ}_frameflip${FLIP_PROB}"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv outputs/finetune_results_frameflip.csv \
    --frame_flip_aug \
    --frame_flip_prob "$FLIP_PROB" \
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
