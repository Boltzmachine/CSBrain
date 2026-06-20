#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name=csbrain-orig-repro
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Faithful reproduction of the ORIGINAL CSBrain PhysioNet-MI finetune
# (yuchen2199/CSBrain) using the current codebase + outputs/CSBrain.pth.
# Uses --model CSBrain (not the local Align/world-model path) and the
# original hyperparameters (lr 5e-5, dropout 0.3, wd 0.01, 50 epochs).

SEED="${1:-42}"

python finetune_main.py \
    --model CSBrain \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir data/preprocessed/physionet_mi \
    --num_of_classes 4 \
    --model_dir outputs/CSBrain/finetune_CSBrain_orig_PhysioNet \
    --foundation_dir outputs/CSBrain.pth \
    --seed "$SEED" \
    --wandb_run_name "CSBrain_orig_repro_seed${SEED}" \
    --results_csv outputs/finetune_results.csv \
    --use_pretrained_weights \
    --num_workers 4 \
    --epochs 50 \
    --in_dim 200 \
    --out_dim 200 \
    --d_model 200 \
    --dim_feedforward 800 \
    --seq_len 30 \
    --n_layer 12 \
    --nhead 8 \
    --dropout 0.3 \
    --weight_decay 0.01 \
    --lr 0.00005

wait
