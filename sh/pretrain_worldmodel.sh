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
# the CONTINUOUS left/right hand-movement speed (the WiLoR annotations in
# data/EgoBrain/cache_hand_labels_grid_*) off each window's global rep with a
# small MLP head + masked SmoothL1 (per-column valid; undetected hand skipped).
# Under --egobrain_use_frame_grid (this run) pass --egobrain_hand_grid_dir: a
# TIME-KEYED cache whose slot k is the RAW hand speed over the forward 0.2 s pair
# at frame slot k, read at each window's anchor frame (no smoothing; grid_s must
# equal --egobrain_frame_grid_s, validated on load). The legacy clip-keyed
# --egobrain_hand_labels_dir (w1.0s1.0_e0.5_nw2_k7_c4.0_fs200) only works on the
# non-grid path and is REJECTED here — it is (clip,window)-keyed, so under the
# grid it would silently train on nothing. On frame-averaging flip steps the
# left/right targets swap (mirrored scene). Watch hand_pred_loss / diag_hand_mae /
# diag_hand_valid_frac in wandb. Off without the flag.
#
# --use_cached_embeddings: load PRE-COMPUTED frozen DINOv2 embeddings of the
# EgoBrain frames (cls/cls_flip + patch grid/grid_flip, both orientations) from
# data/EgoBrain/cache_embeddings_<enc>_w1.0s1.0_e0.5_nw2_sz224/ instead of
# running the encoder on the fly — removes the dominant per-step GPU cost and is
# numerically equivalent (both flip orientations are cached exactly; a feature-
# space flip is NOT, so the mirrored frames are stored). Build the cache once
# (needs the frame cache first) on an H100 via sh/extract_embeddings.sh, with
# the SAME window/stride/erp/n_windows/sz slug and --vision_encoder as this run.
# To enable: add a trailing `\` to the --run_name line and uncomment the flag in
# the optional block below. DINOv2-style encoders only for now.
#
# --egobrain_motion_resample (grid mode only): EgoBrain's egocentric video is
# mostly STATIC (the wearer sits still for long stretches), so the uniform anchor
# draw spends ~half the budget on frozen frames where the world-model's next-frame
# prediction is trivial (cos-dist median ~0.03, gini ~0.5; see the distribution
# study in outputs/eval_tables.md + outputs/motion/). This biases the anchor k
# toward visually DYNAMIC moments — genuine hand/object manipulation — by drawing
# k ∝ clip(motion(k),0,p_cap)^alpha mixed with a uniform floor, where motion(k) is
# the frame-to-frame distance across the prediction step. Only WITHIN-subject
# position is reweighted; cross-subject balance is unchanged. Knobs:
#   --egobrain_motion_resample_space  patch (DINOv2 grid tokens = the WM target,
#       V-JEPA-portable; default) | pixel (raw frames; cheaper, inflates screen/
#       lighting) | cls (DINOv2 global; not V-JEPA-portable)
#   --egobrain_motion_resample_metric cos (default) | l1   (rank-identical ~0.999)
#   --egobrain_motion_resample_alpha  0=uniform, 1=linear (default), 2=aggressive
#   --egobrain_motion_resample_cap_pct 99   --egobrain_motion_resample_floor_mix 0.1
# Per-subject motion scores cache lazily under the embedding cache _motioncache/;
# prebuild all subjects with `python -m scripts.build_egobrain_motion_cache`.
#
# --egobrain_video_subjects (BACKWARD COMPAT): EgoBrain later released GoPro video
# for P0025-P0040, which were originally EEG-only. has_image is derived purely from
# per-subject cache-FILE EXISTENCE, so once their frame/embedding caches are built
# `--egobrain_subjects all` silently pulls them into the image-alignment, flip-align,
# band-align and world-model frame objectives (and motion-weighted anchor sampling).
# To REPRODUCE pre-release results bit-for-bit while those caches sit on disk, add:
#     --egobrain_video_subjects legacy24
# which restricts video to P0001-P0024; the new subjects then fall back to EEG-only
# losses + uniform anchor draws, exactly as before their video existed. Accepts
# `all` (default = every subject with a cache), `legacy24`, or a subject list.
#
# --wm_frame_motion_alpha (--wm_objective frame only): per-PATCH motion weighting
# on the dense frame-prediction loss — the SPATIAL analog of motion_resample
# (which reweights WHICH frames), applied inside the loss over WHICH PATCHES. The
# next-0.2s DINOv2 grid is ~static, so the uniform L1 over patches is dominated by
# patches the anchor already explains; the predictor minimises it by COPYING the
# anchor and never uses the EEG (diag_frame_eeg_gap ~0.16% and shrinking — the
# 2026-07-04 ablation showed the predictor adds ~0). This weights each patch by
# how far its target moved from the anchor grid,
#   w = clip(motion/mean_motion, floor, inf)^alpha,  motion = mean_d|s_tgt - s_anchor|
# concentrating the loss on DYNAMIC patches — the only place the EEG can lower it —
# so the copy shortcut stops scoring well and gradient reaches the EEG. It is
# self-normalising (weighted mean by w), so latent_pred_weight needs no retuning,
# and adds NO parameters. Watch diag_frame_eeg_gap RISE and diag_frame_copy_l1
# climb (copy now scores badly on the weighted objective). The reference in the
# weight is the FIXED per-horizon EGOBRAIN_FRAME_MOTION_REF_PER_STEP constant
# (motion grows with the horizon, so it is per-step) baked into
# models/world_model.py — precompute it with scripts/compute_frame_motion_ref.py
# and recompute if you change the vision encoder / grid_s / stride_s / max_horizon.
# Knobs:
#   --wm_frame_motion_alpha 0=uniform/legacy (default), 1.0=motion-proportional
#       (recommended), 2.0=aggressive
#   --wm_frame_motion_floor 0.1   min per-patch weight (keeps static steps finite)
#   --wm_frame_motion_ref  <0 (default)=fixed baked-in per-horizon constant
#       (stable); 0=per-batch per-step mean (jitters); >0=explicit scalar override
# Best paired with the load-bearing frame-averaging (--frame_averaging), i.e. the
# full dino-dense config, not this nofa run. To enable, add the flags below.
#
# --wm_frame_clean_cond (--wm_objective frame only): asymmetric TWO-VIEW split.
# With mask_ratio=0.5 the masked forward feeds half mask-token reconstruction
# guesses + intermittent masking, so alignment/prediction learn to ignore the EEG
# (diag_frame_eeg_gap ~0 regardless of the motion weighting). With this on, the
# MASKED view is the reconstruction pretext ONLY (mask_loss + aux_phase/aux_env +
# frame_recon), and a second UNMASKED forward carries everything downstream-facing
# on the complete signal — image alignment, flip-align, hand-pred, and the frame
# predictor's EEG conditioning. Masked recon is untouched. Costs one extra (cheap)
# EEG-encoder forward. Watch diag_frame_eeg_gap: if it RISES, masking was the
# suppressor; if it stays ~0, the anchor confound (I(EEG;future|anchor)~0) is the
# real ceiling. Independent of --wm_frame_motion_alpha (can run with alpha=0).
#
# --wm_frame_contrast_weight (--wm_objective frame only): negative-EEG CONTRASTIVE
# term. The clean-cond + motion-weight ablations still leave diag_frame_eeg_gap ~0
# because both only change WHAT the copy shortcut is scored against — the predictor
# can still ignore the EEG. This term instead re-runs the predictor on the SAME
# anchor grid conditioned on OTHER rows' EEG (in-batch negatives) and penalises it
# for reproducing the true future from the WRONG EEG:
#   infonce: softmax over {pos, K negs} on -distance -> the true EEG must give the
#            smallest prediction distance (CE, self-normalising).
#   margin : softplus(margin + d_pos - d_neg) -> each negative's distance must
#            exceed the positive's by --wm_frame_contrast_margin.
# Holding the anchor fixed and swapping only the EEG isolates the EEG's motion
# contribution: the copy-the-anchor shortcut gives the SAME prediction for every
# EEG, so it can no longer make d_pos < d_neg — the only way to lower the loss is to
# read the motion out of the EEG. Negatives are drawn ONLY from rows sharing the
# anchor's frame-averaging flip, so under --frame_averaging the predictor can't win
# on orientation bookkeeping (a flip-mismatched negative) instead of motion; the
# realised negatives/row = min(n_neg, same-flip-group-size - 1). NOTE negatives are
# still DIFFERENT scenes, so scene identity remains a partial shortcut — pair with
# --wm_frame_motion_alpha>0. Negatives are detached from the EEG encoder by
# default (--wm_frame_contrast_detach_neg; use --wm_frame_contrast_grad_neg for
# standard InfoNCE) so only the predictor learns discrimination from them while the
# encoder is still pushed via the positive — keeps arbitrary (eeg_j, scene_i)
# pairings out of the downstream EEG rep. Costs n_neg extra predictor forwards/step.
# Watch diag_frame_contrast_acc (chance~1/(realised negs+1); strict tie-break so the
# EEG-ignoring copy mode reads chance, not 1) and diag_frame_contrast_gap
# (d_neg - d_pos, > 0 & growing = EEG used). BEST PAIRED with
# --wm_frame_motion_alpha>0 so the contrast focuses on DYNAMIC patches; with alpha=0
# it can be satisfied by encoding static SCENE IDENTITY into the EEG (which the
# anchor already carries) rather than motion. Knobs:
#   --wm_frame_contrast_weight 0=OFF (default) | ~0.5-2.0 to enable
#   --wm_frame_contrast_n_neg 1..K  (more = harder/stronger, K extra forwards)
#   --wm_frame_contrast_temp 0.1    (infonce; smaller amplifies tiny gaps)
#   --wm_frame_contrast_mode infonce | margin   --wm_frame_contrast_margin 0.1
#   --wm_frame_contrast_batched  run all n_neg negatives as ONE predictor forward
#       (n_neg*Bv rows) instead of an n_neg-step loop — identical result, fewer/
#       larger kernels but ~n_neg x peak attention memory. On = test GPU capacity;
#       off (default) = memory-safe loop. If it OOMs, drop it or lower n_neg.
# To enable, add a trailing `\` to the last active flag and uncomment the block
# in the optional section below.
#
# --gradnorm (GradNorm adaptive loss balancing; Chen et al. 2018): the loss terms
# (masked recon, frame prediction, image alignment, band phase/envelope aux, ...)
# have gradient norms at the shared frontend that differ by orders of magnitude,
# so the hand-tuned static weights are fragile. GradNorm learns a per-term weight
# online so every ENABLED term contributes a comparable gradient norm, scaled by
# its relative training rate. It SUPERSEDES the static weights (mask_weight /
# latent_pred_weight / aux_*_weight / alignment_weight) for the terms it balances —
# those now act only as ON/OFF gates (0 = drop the term entirely). Costs ~1 extra
# backward per balanced term/step (autograd.grad at patch_embedding). Knobs:
#   --gradnorm_alpha 1.5   asymmetry (0=equalise grad norms; larger respects rates)
#   --gradnorm_lr    0.025 lr of the per-term weight optimiser
# Watch gnw_<term> (learned weight) / gngrad_<term> (grad norm) in wandb — the
# gngrad_<term> values should converge together. To enable, add the flag below.
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
    --egobrain_use_grid_embeddings \
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
    --run_name wm-hand \
    --egobrain_motion_resample \
    --egobrain_motion_resample_space patch \
    --egobrain_motion_resample_metric cos \
    --egobrain_motion_resample_alpha 1.0 \
    --wm_frame_motion_alpha 0 \
    --wm_frame_contrast_weight 0.0 \
    --wm_frame_contrast_n_neg 3 \
    --wm_frame_contrast_temp 0.1 \
    --wm_frame_contrast_grad_neg \
    --wm_frame_contrast_mode infonce \
    --aux_hand_pred \
    --aux_hand_weight 0.5
    # Linear LR warmup over the first 5% of total steps (epochs * iters/epoch),
    # ramping each param group 0 -> base LR, then the cosine schedule takes over.
    # Auto-scales with epoch count / dataset size; set 0 to disable. Overrides
    # --lr_warmup_iters when > 0.
    # Per-patch motion weighting of the dense frame-pred loss (kills the copy
    # shortcut so the predictor uses the EEG). Move the trailing `\` up onto the
    # motion_resample_alpha line above and uncomment to enable:
    # --wm_frame_motion_floor 0.1 \
    # Unmasked EEG conditioning for the predictor (recon stays masked). Add a
    # trailing `\` to the last active flag above, then uncomment:
    # --wm_frame_clean_cond \
    # Negative-EEG contrastive term (forces the predictor to use the EEG; pair with
    # --wm_frame_motion_alpha>0). Uncomment:
    # --wm_frame_contrast_weight 1.0 \
    # --wm_frame_contrast_n_neg 4 \
    # --wm_frame_contrast_temp 0.1 \
    # --wm_frame_contrast_mode infonce \
    # CSBrain spatial-mixing residual (models/CSBrain.py): add
    # BrainEmbedEEGLayer(patch_emb)+patch_emb — a circular conv across EEG
    # channels — after the TemEmbed residual in every backbone layer. Channels
    # are region-GROUPED first (per-sample from ch_names, the EgoBrain analogue of
    # CSBrain's fixed x=x[:,sorted_indices]); area_config is None so this is the
    # only channel-order-dependent op, so grouping it locally leaves the coord PE /
    # masks / reconstruction target in native channel order. Add a trailing `\` to
    # the last active flag above (--wm_frame_contrast_mode infonce) and uncomment:
    # --use_brain_embed \
    # GradNorm adaptive loss balancing (learns the per-term weights online;
    # supersedes the static loss weights for balanced terms). Uncomment:
    # --gradnorm \
    # --gradnorm_alpha 1.5 \
    # --gradnorm_lr 0.025 \
    # --aux_hand_pred \
    # --aux_hand_weight 0.1 \
    # This run uses --egobrain_use_frame_grid, so the hand aux needs the
    # TIME-KEYED grid cache (build once: sbatch --array=1-40
    # sh/egobrain_hand_labels_grid.sh; grid_s MUST match --egobrain_frame_grid_s
    # above). Targets are the RAW per-slot hand speed over the forward 0.2 s pair
    # at the window's anchor frame (no smoothing). The legacy clip-keyed
    # --egobrain_hand_labels_dir is REJECTED under the frame grid (it would
    # silently train on nothing), as is the superseded smoothed *_r1.0_* cache.
    # --egobrain_hand_grid_dir data/EgoBrain/cache_hand_labels_grid_wilor_g0.2_raw_fs200 \
