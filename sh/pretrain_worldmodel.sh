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
#
# --lateralization_flip (bilateralization prior): a learned frontend splits the
# raw EEG additively into a bilateral stream x_bi and a lateral stream x_lat
# (x_bi + x_lat = x). Swapping only x_lat across homologous channels (C3<->C4)
# yields x_flip = x_bi + flip(x_lat); its encoding is aligned (InfoNCE, shared
# projector) to the HORIZONTALLY-FLIPPED frame while the un-flipped encoding
# aligns to the original frame — tying an EEG hemisphere swap to an image
# mirror. --flip_align_weight scales that loss; --lat_sparsity_weight keeps
# x_lat minimal so most signal stays bilateral. Needs frames (EgoBrain rows).
# The image flip target is a CENTERED column-band spatial descriptor of the
# patch grid (--flip_n_col_bands; 2 = left/right), NOT the global CLS — the CLS
# is ~horizontal-flip-invariant (cos~0.97) so it makes the loss vacuous, while
# the centered left/right descriptor is strongly flip-sensitive (cos~0.09 on
# motion frames). --flip_motion_ref>0 down-weights static frames (whose flip is
# near-vacuous) by the t->t+1 frame motion; 0 = uniform weighting.
#
# --egobrain_delta_whiten_g0 (ME->MI delta whitening): 1.0 = OFF. Set ~0.65-0.78
# to attenuate EgoBrain's motor-execution delta floor toward the motor-imagery
# (PhysioNet-MI) level. CALIBRATE g0 to the finetune LMDB MI delta (~46 montage /
# ~42 central, in percent) with a quick scipy.welch sweep; too-strong a gain
# overshoots past MI and HURTS. See project_me_mi_pretrain_manipulation.
python pretrain_main.py \
    --model WorldModel \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --dataset_dir egobrain \
    --egobrain_n_windows 2 \
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
    --lateralization_flip \
    --flip_align_weight 0.1 \
    --lat_sparsity_weight 0.0 \
    --flip_n_col_bands 2 \
    --flip_motion_ref 0.0 \
    --flip_pred_weight 1.0 \
    --latent_pred_weight 1.0 \
    --cls_pred_weight 0.1 \
    --max_horizon 1 \
    --pred_ramp_epochs 2 \
    --predictor_d_model 512 \
    --predictor_n_layers 4 \
    --model_dir outputs/ \
    --spectral_mode instantaneous \
    --aux_band_pred \
    --aux_delta_band "0.5,4" \
    --aux_power_bands "8,13;13,30" \
    --aux_phase_weight 1.0 \
    --aux_envelope_weight 0.005 \
    --run_name wm-bilat
