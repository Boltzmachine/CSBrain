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

# EEG world-model / video-prediction pretraining on Alljoined-1.6M +
# (CineBrain or EgoBrain). With max_horizon=0 the WorldModel wrapper
# degrades to plain CSBrainAlign (masked recon + image alignment); set
# max_horizon=1 to enable the future-window latent predictor. Alljoined
# samples (event-centered, no future stack) and CineBrain/EgoBrain
# windows ride in the same batch via collate_cached_with_future. See
# plans/world_model.md for the design.
#
# Switch between sources by flipping --dataset_dir:
#   * mix+cinebrain           — Alljoined + CineBrain (6 subj / ~36 h / 64ch)
#   * mix+egobrain            — Alljoined + EgoBrain  (40 subj / ~63 h / 32ch)
#   * mix+cinebrain+egobrain  — all three sources; EgoBrain's 32ch rows
#                               are zero-padded to 64ch in
#                               collate_cached_with_future, valid_channel_mask
#                               keeps padded channels out of the encoder
# Sampling weights: --mix_{alljoined,cinebrain,egobrain}_weight control
# the multinomial mix; weights for unused sources are ignored.
# Each source has its own --{cinebrain,egobrain}_* knobs below; the unused
# block is ignored by the active branch, so it's safe to leave them all set.
python pretrain_main.py \
    --model WorldModel \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --dataset_dir mix+egobrain \
    --mix_alljoined_weight 1.0 \
    --mix_cinebrain_weight 1.0 \
    --mix_egobrain_weight 1.0 \
    --cinebrain_root data/CineBrain \
    --cinebrain_subjects sub-0001,sub-0002,sub-0003,sub-0004,sub-0005,sub-0006 \
    --cinebrain_n_windows 1 \
    --cinebrain_window_s 1.0 \
    --cinebrain_stride_s 1.0 \
    --cinebrain_erp_latency_s 0.5 \
    --egobrain_root data/EgoBrain \
    --egobrain_subjects all \
    --egobrain_window_s 1.0 \
    --egobrain_stride_s 1.0 \
    --egobrain_clip_s 4.0 \
    --egobrain_erp_latency_s 0.5 \
    --egobrain_max_channels 32 \
    --in_dim 40 \
    --out_dim 40 \
    --d_model 40 \
    --seq_len 5 \
    --n_layer 12 \
    --nhead 8 \
    --batch_size 128 \
    --epochs 40 \
    --mask_ratio 0.5 \
    --clip_value 0.8 \
    --alignment_weight 0.1 \
    --latent_pred_weight 1.0 \
    --cls_pred_weight 0.1 \
    --max_horizon 1 \
    --pred_ramp_epochs 2 \
    --predictor_d_model 512 \
    --predictor_n_layers 4 \
    --model_dir outputs/ \
    --spectral_mode instantaneous \
    --run_name worldmodel-mix-egobrain
