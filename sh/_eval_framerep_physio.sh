#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-eval-framerep
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Frame-averaging readout-ablation eval. Identical to _eval_plain_physio.sh
# (plain protocol, no augmentation) except it passes --frame_rep_mode so the
# classifier reads only the invariant ('inv') or equivariant ('eq') half of the
# equi backbone token h=[inv_half;eq_half] (global-token rows kept either way).
# Tags the run name + writes to a separate CSV so it never collides with the
# 'both' (default) plain-protocol rows.

FOUNDATION_DIR="${1:?foundation ckpt path required}"
SEED="${2:-42}"
FRAME_REP_MODE="${3:?frame_rep_mode required: inv|eq|both}"
NUM_WORKERS="${4:-4}"
SEGMENT_INDEX="${5:-0}"
HIGHPASS_HZ="${6:-0}"
CKPT_NAME=$(basename "$(dirname "$FOUNDATION_DIR")")
EPOCH=$(basename "$FOUNDATION_DIR" .pth | sed 's/_loss.*//')
WANDB_RUN_NAME="${CKPT_NAME}_${EPOCH}_seed${SEED}_seg${SEGMENT_INDEX}_hp${HIGHPASS_HZ}_rep${FRAME_REP_MODE}"

python finetune_main.py \
    --model Align \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir "$FOUNDATION_DIR" \
    --seed "$SEED" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --results_csv outputs/finetune_results_framerep.csv \
    --frame_rep_mode "$FRAME_REP_MODE" \
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
