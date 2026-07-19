#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --job-name=csbrain-base-forenzo-reg
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# ORIGINAL CSBrain baseline on Forenzo2024 REGRESSION (2D cursor velocity).
# --model CSBrain, first-1s window (in_dim=200 -> 1 patch -> --seq_len 1), native
# recipe (lr 1e-4, dropout 0.1). num_workers=0 (401k LMDB segfault), reduced epochs.
SEED="${1:-42}"; NW="${2:-0}"; EPOCHS="${3:-10}"
python finetune_main.py \
    --model CSBrain \
    --downstream_dataset Forenzo2024Reg \
    --datasets_dir data/preprocessed/forenzo \
    --num_of_classes 2 \
    --model_dir outputs/CSBrain/finetune_Forenzo2024Reg_csbrain \
    --foundation_dir outputs/CSBrain.pth \
    --seed "$SEED" \
    --seq_len 1 \
    --epochs "$EPOCHS" \
    --use_initial_segment_only \
    --segment_index 0 \
    --wandb_run_name "CSBrain-base_Forenzo2024reg_seg0_seed${SEED}" \
    --pre_cls_layernorm \
    --use_pretrained_weights \
    --num_workers "$NW" \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr 0.0001
wait
echo "All tasks completed at $(date)"
