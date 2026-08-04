#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=wm-scratch
#SBATCH --output=outputs/slurms/%j.out
#SBATCH --qos=qos_nmi

# LeWorldModel (LeWM) FROM-SCRATCH, jointly-trained world model.
# See plans/mighty-dazzling-lovelace.md and https://github.com/lucas-maes/le-wm.
#
# This is a MINIMAL DIFF of sh/pretrain_worldmodel.sh (the wm-subj-hard config):
# every hyperparameter is kept IDENTICAL, changing ONLY what training the video
# world model from scratch REQUIRES:
#   REMOVED  --egobrain_use_grid_embeddings   (the trainable ViT must run LIVE on
#            raw 224x224 frames every step; cached DINOv2 grids would bypass it.
#            --egobrain_use_frame_grid is KEPT, so raw frames flow from the
#            pre-decoded uint8 grid cache. --vision_encoder stays dinov2-base ONLY
#            to select that cache slug + ImageNet normalization; the MODEL ignores
#            the HF id under --wm_scratch_vit.)
#   ADDED    --wm_scratch_vit                 randomly-initialized TRAINABLE ViT-tiny
#            replaces the frozen DINOv2 EVERYWHERE (alignment + frame prediction).
#            Target = the ViT's OWN online embedding (NO stop-grad, NO EMA);
#            collapse is held off SOLELY by SIGReg.
#   ADDED    --wm_sigreg_weight 0.09          the single tunable LeWM loss weight.
#   ADDED    --scratch_vit_size tiny --scratch_vit_patch 16   (LeWM ViT-tiny, 192d).
#   ADDED    --amp                            bf16 autocast for the trainable ViT
#            forward ONLY (its output is cast back to fp32; the EEG encoder stays
#            fp32 — its FFT spectral ops don't support bf16). The live ViT over 6
#            frames x batch is heavy; bf16 keeps it on an H100. Drop it for strict
#            fp32 parity, and lower --batch_size if it OOMs.
#
# EVERYTHING ELSE is byte-for-byte the wm-subj-hard config: batch_size 128,
# --frame_averaging + flip knobs, cls_pred_weight 0.2, latent_pred_weight 2.0,
# max_horizon 5, predictor dims, the spectral aux, wm_frame_eeg_cond tokens, the
# motion_resample anchor sampler (patch-space; uses the DINOv2 embedding grid cache
# for SAMPLING only, never in the model/gradient path), the negative-EEG contrastive,
# and the subject-block sampler. Verified: the scratch path runs end-to-end with
# frame-averaging + contrastive + alignment all ON.
#
# Watch in W&B: sigreg_loss (kept small WITHOUT collapse); diag_frame_emb_std > 0
# and stable (0 == collapse); frame_pred_loss finite; diag_frame_eeg_gap > 0 and
# diag_frame_contrast_acc above chance (EEG is used); diag_frame_*_norm stable.
#
# --wm_predictor picks the world-model predictor (all three are dense-target,
# scratch-ViT, SIGReg-regularised; the frame ones share ALL the frame machinery):
#   frame        (default) dense multi-frame-from-current-frame; EEG injected as
#                CONCATENATED tokens in a query transformer.
#   frame_adaln  SAME dense multi-frame-from-current-frame objective, but the EEG
#                is injected by AdaLN-zero conditioning (LeWM ConditionalBlock) —
#                the EEG modulates each block (shift/scale/gate) instead of being
#                concatenated. Uses --predictor_* dims (incl. --predictor_n_layers).
#   ar           LeWM faithful CAUSAL ARPredictor (MSE next-embedding, EEG-as-action
#                per step, frame-CLS level); uses --wm_ar_* knobs. Watch diag_ar_*.
#
# PREDICTOR DEPTH (number of transformer layers), both set as active flags below:
#   --predictor_n_layers  frame / frame_adaln predictor depth
#   --wm_ar_depth         the LeWM AR predictor's depth (--wm_predictor ar)
# Only the one matching --wm_predictor takes effect; the other is ignored.
#
# For --wm_predictor ar the frame-contrast/motion knobs + --wm_frame_eeg_cond are
# ignored and the AR knobs apply (--wm_ar_depth above; --wm_ar_history 3
# --wm_ar_num_preds 1 --wm_ar_heads 16); needs egobrain_n_windows >=
# wm_ar_history + wm_ar_num_preds. Watch diag_ar_* + diag_ar_emb_std.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python pretrain_main.py \
    --model WorldModel \
    --TemEmbed_kernel_sizes "[(1,), (3,), (5,),]" \
    --dataset_dir egobrain \
    --egobrain_n_windows 6 \
    --mix_alljoined_weight 1.0 \
    --mix_cinebrain_weight 1.0 \
    --mix_egobrain_weight 1.0 \
    --egobrain_root data/EgoBrain \
    --egobrain_use_frame_grid \
    --egobrain_subjects all \
    --egobrain_window_s 1.0 \
    --egobrain_stride_s 0.2 \
    --egobrain_clip_s 4.0 \
    --egobrain_erp_latency_s -0.15 \
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
    --mask_weight 1.0 \
    --clip_value 0.8 \
    --alignment_weight 0.1 \
    --frame_averaging \
    --frame_avg_flip_prob 0.5 \
    --frame_avg_recon_weight 0.0 \
    --flip_align_weight 0.1 \
    --flip_n_col_bands 2 \
    --latent_pred_weight 2.0 \
    --cls_pred_weight 0.2 \
    --max_horizon 5 \
    --pred_ramp_epochs 0 \
    --predictor_d_model 512 \
    --predictor_n_layers 2 \
    --wm_ar_depth 3 \
    --model_dir outputs/ \
    --spectral_mode instantaneous \
    --aux_band_pred \
    --aux_delta_band "0.5,4" \
    --aux_power_bands "8,13;13,30" \
    --aux_phase_weight 1.0 \
    --aux_envelope_weight 0.005 \
    --wm_objective frame \
    --wm_frame_eeg_cond tokens \
    --vision_encoder facebook/dinov2-base \
    --wm_scratch_vit \
    --wm_predictor frame \
    --scratch_vit_size tiny \
    --scratch_vit_patch 16 \
    --wm_sigreg_weight 0.09 \
    --amp \
    --egobrain_motion_resample \
    --egobrain_motion_resample_space patch \
    --egobrain_motion_resample_metric cos \
    --egobrain_motion_resample_alpha 1.0 \
    --wm_frame_motion_alpha 0 \
    --wm_frame_contrast_weight 0.0 \
    --egobrain_subject_block 0 \
    --egobrain_block_window_s 0 \
    --wm_frame_contrast_n_neg 3 \
    --wm_frame_contrast_temp 0.1 \
    --wm_frame_contrast_grad_neg \
    --wm_frame_contrast_mode infonce \
    --run_name wm-scratch-frame
    # --- LeWM AR predictor variant: set --wm_predictor ar (--wm_ar_depth is already
    # an active flag above) and uncomment the rest ---
    # --wm_ar_history 3 \
    # --wm_ar_num_preds 1 \
    # --wm_ar_heads 16 \
    # --- Separate LR for the from-scratch ViT (LeWM used 5e-5) ---
    # --vision_lr 5e-5 \
