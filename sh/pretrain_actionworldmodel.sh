#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=actionwm
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# Action-conditioned world model (EEG = action, frozen V-JEPA 2 = world).
# The INVERSE of pretrain_worldmodel.sh: there EEG *is* the world and we
# predict the next EEG latent; here the WORLD is the video (frozen V-JEPA 2
# spatial token grid s_t) and the EEG model decodes a "moving intention"
# action a_t that drives a predictor  ŝ_{t+1} = p(s_t, a_t)  against the
# frozen sg[VJEPA2(frame_{t+1})]. See plans/world_model.md (Action-Conditioned
# variant) and models/action_world_model.py.
#
# PREREQUISITE — build the V-JEPA 2 256px EgoBrain frame cache once (the
# existing cache is DINOv2-224). On a GPU node:
#   conda run -n cbramod python -m datasets.egobrain_extract_frames \
#       --data_dir data/EgoBrain --subjects all \
#       --vision_encoder facebook/vjepa2-vitl-fpc64-256 \
#       --window_s 1.0 --stride_s 1.0 --erp_latency_s 0.5 --n_windows 2 \
#       --num_workers 4
#
# The action is ALWAYS decoded from the full, unmasked window. Masked patch
# reconstruction is ON by default here as a co-objective (done as a SEPARATE
# masked EEG forward, so it never corrupts the action): --need_mask is left at
# its default (on), so recon is routed through the trainer's mask_loss
# (mask_ratio 0.5, weight 1.0 — same mechanism/weight as pretrain_worldmodel.sh).
#   * To DISABLE recon (prediction-only): add  --need_mask ''
#   * Alternative wrapper-internal route: --need_mask '' --recon_aux_weight W
# --eeg_ckpt optionally warm-starts the EEG backbone from a masked-recon ckpt.
#
# --egobrain_delta_whiten_g0 (ME->MI delta whitening): 1.0 = OFF. Set ~0.65-0.78
# to attenuate EgoBrain's motor-execution delta floor toward the motor-imagery
# (PhysioNet-MI) level. CALIBRATE g0 to the finetune LMDB MI delta (~46 montage /
# ~42 central, in percent) with a quick scipy.welch sweep; too-strong a gain
# overshoots past MI and HURTS. See project_me_mi_pretrain_manipulation.
python pretrain_main.py \
    --model ActionWorldModel \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --dataset_dir egobrain \
    --vision_encoder facebook/vjepa2-vitl-fpc64-256 \
    --egobrain_root data/EgoBrain \
    --egobrain_subjects all \
    --egobrain_n_windows 2 \
    --egobrain_window_s 1.0 \
    --egobrain_stride_s 1.0 \
    --egobrain_clip_s 4.0 \
    --egobrain_erp_latency_s 0.5 \
    --egobrain_max_channels 32 \
    --egobrain_load_frames 1 \
    --egobrain_delta_whiten_g0 1.0 \
    --egobrain_delta_whiten_cutoff_hz 8.0 \
    --in_dim 40 \
    --out_dim 40 \
    --d_model 40 \
    --seq_len 5 \
    --n_layer 12 \
    --nhead 8 \
    --recon_aux_weight 0.0 \
    --mask_ratio 0.5 \
    --max_horizon 1 \
    --action_dim 64 \
    --n_action_tokens 8 \
    --pred_residual \
    --pred_l1_weight 1.0 \
    --pred_cos_weight 0.1 \
    --predictor_d_model 512 \
    --predictor_n_layers 4 \
    --predictor_n_heads 8 \
    --batch_size 128 \
    --epochs 40 \
    --clip_value 0.8 \
    --model_dir outputs/ \
    --spectral_mode instantaneous \
    --run_name action_wm \
    # --eeg_ckpt outputs/<recon_run>/<ckpt>.pt
