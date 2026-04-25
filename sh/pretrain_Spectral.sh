#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=spectral
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

python pretrain_main.py \
    --model Spectral \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --model_dir outputs/ \
    --dataset_dir mix \
    --batch_size 128 \
    --n_layer 24 \
    --in_dim 40 \
    --out_dim 40 \
    --d_model 40 \
    --seq_len 20 \
    --nhead 4 \
    --saliency_rollout_skip_layers 2 \
    --vision_encoder facebook/dinov2-base \
    --run_name Spectral-4band-3level-40d-dinov2
