#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=04:00:00
#SBATCH --job-name=csbrain-base-physio
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# ORIGINAL CSBrain baseline on PhysioNet-MI, FIRST 1s window (same as our WM
# line). --model CSBrain, in_dim=200 so 1s = 1 patch -> --seq_len 1 +
# --use_initial_segment_only --segment_index 0. No aug. Native CSBrain recipe.
SEED="${1:-42}"
NUM_WORKERS="${2:-4}"
python finetune_main.py \
    --model CSBrain \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_PhysioNet \
    --foundation_dir outputs/CSBrain.pth \
    --seed "$SEED" \
    --seq_len 1 \
    --use_initial_segment_only \
    --segment_index 0 \
    --wandb_run_name "CSBrain-base_physio_seg0_seed${SEED}" \
    --results_csv outputs/finetune_results_csbrain_base_physio.csv \
    --use_pretrained_weights \
    --num_workers "$NUM_WORKERS" \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr 0.0001
wait
echo "All tasks completed at $(date)"
