#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=04:00:00
#SBATCH --job-name=csbrain-base-finemi
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# ORIGINAL CSBrain baseline on FineMI, FIRST 1s window (same as our WM
# line). --model CSBrain, in_dim=200 so 1s = 1 patch -> --seq_len 1 +
# --use_initial_segment_only --segment_index 0. No aug. Native CSBrain recipe.
SEED="${1:-42}"
NUM_WORKERS="${2:-4}"
python finetune_main.py \
    --model CSBrain \
    --downstream_dataset FineMI \
    --datasets_dir data/preprocessed/finemi \
    --num_of_classes 8 \
    --model_dir outputs/CSBrain/finetune_CSBrain_FineMI \
    --foundation_dir outputs/CSBrain.pth \
    --seed "$SEED" \
    --seq_len 1 \
    --use_initial_segment_only \
    --segment_index 0 \
    --wandb_run_name "CSBrain-base_finemi_seg0_seed${SEED}" \
    --results_csv outputs/finetune_results_csbrain_base_finemi.csv \
    --use_pretrained_weights \
    --num_workers "$NUM_WORKERS" \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr 0.0001
wait
echo "All tasks completed at $(date)"
