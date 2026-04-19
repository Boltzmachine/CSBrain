#!/bin/bash

#!/bin/zsh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=csbrain
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi


# python pretrain_main.py \
#     --model CSBrain \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --n_layer 24 \
#     --run_name CSBrain-deep

# python pretrain_main.py \
#     --model OurModel \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --run_name ours_all_zerosync_16

# ============================================================
# Stage 1: Pre-train source projector (reconstruction + decorrelation)
# ============================================================
# python pretrain_main.py \
#     --model SourceProjector \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --batch_size 256 \
#     --epochs 100 \
#     --in_dim 40 \
#     --seq_len 20 \
#     --num_sources 32 \
#     --decorr_weight 0.1 \
#     --need_mask False \
#     --run_name source_projector_pretrain2

# ============================================================
# Stage 2: Train full model with pre-trained source projector
# ============================================================
# python pretrain_main.py \
#     --model Align \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --batch_size 128 \
#     --n_layer 24 \
#     --in_dim 40 \
#     --out_dim 40 \
#     --d_model 40 \
#     --seq_len 20 \
#     --nhead 4 \
#     --run_name alljoined-projector-freeze

# ============================================================
# Adversarial session-agnostic pretraining
# ============================================================
python pretrain_main.py \
    --model Align \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --model_dir outputs/ \
    --dataset_dir mix \
    --n_layer 24 \
    --in_dim 40 \
    --out_dim 40 \
    --d_model 40 \
    --seq_len 20 \
    --nhead 4 \
    --samples_per_session 8 \
    --sessions_per_batch 16 \
    --run_name cinebrain


# ============================================================
# DINOv2 Self-Distillation Pretraining
# ============================================================
# python pretrain_main.py \
#     --model Align \
#     --dino_mode \
#     --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
#     --model_dir outputs/ \
#     --dataset_dir mix \
#     --n_layer 24 \
#     --in_dim 40 \
#     --out_dim 40 \
#     --d_model 40 \
#     --seq_len 20 \
#     --nhead 4 \
#     --mask_ratio 0.5 \
#     --n_prototypes 4096 \
#     --dino_head_hidden_dim 256 \
#     --dino_head_bottleneck_dim 256 \
#     --dino_head_n_layers 3 \
#     --student_temp 0.1 \
#     --teacher_temp_base 0.04 \
#     --teacher_temp_final 0.07 \
#     --teacher_temp_warmup_epochs 30 \
#     --ema_momentum_base 0.992 \
#     --ema_momentum_final 0.9995 \
#     --dino_loss_weight 1.0 \
#     --ibot_loss_weight 1.0 \
#     --koleo_loss_weight 0.1 \
#     --alignment_weight 0.0 \
#     --equivariance_weight 0.0 \
#     --last_layer_freeze_iters 1250 \
#     --lr_warmup_iters 5000 \
#     --use_freq_subband \
#     --n_local_crops 4 \
#     --local_crop_time_scale "(0.3, 0.7)" \
#     --local_crop_channel_scale "(0.5, 1.0)" \
#     --run_name dino-v2-multicrop-v2
