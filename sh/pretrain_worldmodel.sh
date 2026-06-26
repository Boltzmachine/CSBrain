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
# --frame_averaging (equivariant frame-averaging frontend; plans/eeg-wm.md):
# upgrades the bilateral/lateral split into a proper invariance/equivariance
# decomposition under the homologous-channel swap P (C3<->C4, P^2=I). A
# channel-independent frontend f (the CNN PatchEmbedding with its cross-channel
# conv positional encoding dropped) produces z = z_bi + z_lat in FEATURE space;
# P(z) = z_bi + flip(z_lat). The transformer T is then wrapped by frame
# averaging over the 2-element group {I, P}:
#   h_bi  = (T(z) + T(P z)) / 2     (P-invariant      -> bilateral half)
#   h_lat = (T(z) - P T(P z)) / 2   (P-anti-equivariant -> lateral half)
# Tokens split along the feature dim into a bilateral half and a lateral half.
# --frame_avg_flip_prob is the per-step probability of PRESENTING P(z) together
# with the HORIZONTALLY-MIRRORED frame; presenting P(z) only negates the lateral
# half, so the architecture is exactly equivariant (verified in
# tests/test_frame_averaging.py). On flip steps there is no ground-truth flipped
# timeseries, so the reconstruction self-supervises in frontend space
# (--frame_avg_recon_weight: f(recon) -> presented clean latent P(z_clean) on
# masked patches) and the world-model predictor predicts the flipped-future EEG
# latent from the flipped-current — the bilateral control signal that makes the
# decomposition non-trivial. Alignment has TWO terms: (1) present-orientation
# global token -> present frame CLS (--alignment_weight), and (2) a same-sample
# HARD-NEGATIVE on the global rep (--flip_align_weight): the presented vs
# opposite-orientation global rep must align to the presented vs MIRRORED frame's
# centered column-band descriptor (--flip_n_col_bands; 2=left/right, the CLS is
# ~flip-invariant so it can't serve here). Term (2) directly penalises a trivial
# (L-R symmetric) split, forcing z_lat to carry laterality. Needs frames
# (EgoBrain rows). Supersedes --lateralization_flip (do not set both).
#
# --egobrain_delta_whiten_g0 (ME->MI delta whitening): 1.0 = OFF. Set ~0.65-0.78
# to attenuate EgoBrain's motor-execution delta floor toward the motor-imagery
# (PhysioNet-MI) level. CALIBRATE g0 to the finetune LMDB MI delta (~46 montage /
# ~42 central, in percent) with a quick scipy.welch sweep; too-strong a gain
# overshoots past MI and HURTS. See project_me_mi_pretrain_manipulation.
#
# --aux_hand_pred (hand-movement decoding auxiliary; EgoBrain rows only): regress
# the CONTINUOUS per-window left/right hand-movement intensity (the WiLoR
# annotations in data/EgoBrain/cache_hand_labels_*; see
# datasets/egobrain_hand_labels.py) off each window's global rep with a small MLP
# head + masked SmoothL1 (per-column valid; undetected hand skipped). The
# --egobrain_hand_labels_dir slug (w1.0s1.0_e0.5_nw2_k7_c4.0_fs200) MUST match the
# egobrain window knobs above, or the (clip,window) keys misalign. On
# frame-averaging flip steps the left/right targets swap (mirrored scene). Watch
# hand_pred_loss / diag_hand_mae / diag_hand_valid_frac in wandb. Off without the flag.
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
    --wm_objective frame \
    --wm_frame_eeg_cond tokens \
    --run_name wm-frame-tokens-noequiv
    # --aux_hand_pred \
    # --aux_hand_weight 0.1 \
    # --egobrain_hand_labels_dir data/EgoBrain/cache_hand_labels_wilor_w1.0s1.0_e0.5_nw2_k7_c4.0_fs200 \
