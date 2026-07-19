#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --job-name=csbrain-smoke
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Reusable END-TO-END smoke test for a new downstream finetune wiring.
# Runs the REAL finetune_main path (Align backbone + world-model ckpt load +
# --use_initial_segment_only crop + softmax head) for a SINGLE epoch at the
# default batch size, so it exercises build+load+train+val+test without a full
# training run. Uses the conda env python by absolute path (no activation).
#   usage: sbatch sh/_smoke_finetune.sh <DOWNSTREAM> <DATASETS_DIR> <NUM_CLASSES>
set -u
DOWNSTREAM="$1"
DATASETS_DIR="$2"
NUM_CLASSES="$3"
PY=/gpfs/radev/home/wq44/.conda/envs/cbramod/bin/python

export WANDB_MODE=disabled
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "outputs/CSBrain/_smoke_${DOWNSTREAM}"

echo "SMOKE START $DOWNSTREAM $(date)"
"$PY" finetune_main.py \
    --model Align \
    --downstream_dataset "$DOWNSTREAM" \
    --datasets_dir "$DATASETS_DIR" \
    --num_of_classes "$NUM_CLASSES" \
    --model_dir "outputs/CSBrain/_smoke_${DOWNSTREAM}" \
    --foundation_dir outputs/wm-dino-dense/epoch10_loss2.1733548641204834.pth \
    --use_pretrained_weights \
    --use_initial_segment_only \
    --segment_index 0 \
    --num_workers 4 \
    --epochs 1 \
    --dropout 0.3 --weight_decay 0.01 --lr 0.00005
echo "SMOKE_DONE $DOWNSTREAM rc=$? $(date)"
