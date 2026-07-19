#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --job-name=csbrain-base-cls
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# ORIGINAL CSBrain baseline, generic classification. First-1s window: CSBrain
# in_dim=200 -> 1s = 1 patch (--seq_len 1 + --use_initial_segment_only). Native
# CSBrain recipe (lr 1e-4, dropout 0.1). Args: DATASET DDIR NCLS [SEED] [NW] [EPOCHS]
DATASET="$1"; DDIR="$2"; NCLS="$3"; SEED="${4:-42}"; NW="${5:-4}"; EPOCHS="${6:-80}"
python finetune_main.py \
    --model CSBrain \
    --downstream_dataset "$DATASET" \
    --datasets_dir "$DDIR" \
    --num_of_classes "$NCLS" \
    --model_dir "outputs/CSBrain/finetune_${DATASET}" \
    --foundation_dir outputs/CSBrain.pth \
    --seed "$SEED" \
    --seq_len 1 \
    --epochs "$EPOCHS" \
    --use_initial_segment_only \
    --segment_index 0 \
    --wandb_run_name "CSBrain-base_${DATASET}_seg0_seed${SEED}" \
    --results_csv "outputs/finetune_results_csbrain_base_${DATASET}.csv" \
    --use_pretrained_weights \
    --num_workers "$NW" \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr 0.0001
wait
echo "All tasks completed at $(date)"
