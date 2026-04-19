#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=worldmodel
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# EEG world-model / video-prediction pretraining on CineBrain.
# See plans/world_model.md for the design.
python pretrain_main.py \
    --model WorldModel \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --dataset_dir cinebrain \
    --cinebrain_root data/CineBrain \
    --cinebrain_subjects sub-0001,sub-0002,sub-0003,sub-0004,sub-0005,sub-0006 \
    --cinebrain_n_windows 1 \
    --cinebrain_window_s 1.0 \
    --cinebrain_stride_s 1.0 \
    --in_dim 40 \
    --out_dim 40 \
    --d_model 40 \
    --seq_len 5 \
    --n_layer 12 \
    --nhead 8 \
    --batch_size 128 \
    --epochs 40 \
    --mask_ratio 0.5 \
    --alignment_weight 0.1 \
    --latent_pred_weight 1.0 \
    --cls_pred_weight 0.1 \
    --max_horizon 0 \
    --pred_ramp_epochs 2 \
    --predictor_d_model 512 \
    --predictor_n_layers 4 \
    --model_dir outputs/ \
    --run_name worldmodel-cinebrain-v1
