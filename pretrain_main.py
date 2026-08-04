import argparse
import functools
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.pretraining_dataset import PretrainingDataset
from models.CSBrain import *
from models import get_model
from pretrain_trainer import Trainer
import wandb
import os
import logging
logger = logging.getLogger(__name__)

def _resolve_video_subjects(spec):
    """Parse --egobrain_video_subjects into a subject whitelist (or None = all).

    'all'/None -> None (every subject with a cache contributes video).
    'legacy24' -> P0001..P0024, the subjects that had GoPro video before
    EgoBrain released MP4s for P0025-P0040; use it to reproduce pre-release
    results bit-for-bit while the new subjects' caches sit on disk.
    Otherwise a comma-separated subject list.
    """
    if spec is None or spec.strip().lower() == 'all':
        return None
    if spec.strip().lower() == 'legacy24':
        return [f'P{i:04d}' for i in range(1, 25)]
    return [s.strip() for s in spec.split(',') if s.strip()]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='lr_scheduler')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Muon'],
                        help='optimizer; Muon uses SingleDeviceMuonWithAuxAdam (Muon for 2D hidden weights, AdamW for embeddings/heads/biases)')
    parser.add_argument('--muon_lr', type=float, default=0.02,
                        help='learning rate for the Muon parameter group (only used when --optimizer Muon); --lr is used for the AdamW group')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')
    parser.add_argument('--mask_weight', type=float, default=1.0,
                        help='coefficient on the masked-reconstruction (mask_loss) term in the total pretrain loss. mask_loss is the reference scale all other terms are tuned as a fraction of, so it defaults to 1.0 (byte-identical to legacy). Lower it to down-weight raw recon relative to the world-model / alignment / aux terms; raise it to do the opposite. Applies to the WorldModel patch-recon, frame-averaging non-flip recon, freq-mask recon, and the plain non-WorldModel path.')
    parser.add_argument('--gradnorm', action='store_true', default=False,
                        help='enable GradNorm adaptive loss balancing (Chen et al. 2018): learn a '
                             'per-term weight online so every ENABLED loss term (masked recon, frame '
                             'prediction, image alignment, band-phase/envelope aux, ...) contributes a '
                             'comparable gradient norm at the shared frontend, scaled by its relative '
                             'training rate. SUPERSEDES the hand-tuned static weights (mask_weight / '
                             'latent_pred_weight / aux_*_weight / alignment_weight) for terms it '
                             'balances — those weights now only act as ON/OFF gates (0 = drop the term). '
                             'Costs ~1 extra backward per balanced term/step (autograd.grad at the '
                             'shared layer). Watch gnw_<term> / gngrad_<term> in wandb.')
    parser.add_argument('--gradnorm_alpha', type=float, default=1.5,
                        help='GradNorm asymmetry: 0 = just equalise gradient norms; larger pulls '
                             'slow-training terms up harder (paper default 1.5).')
    parser.add_argument('--gradnorm_lr', type=float, default=0.025,
                        help='learning rate for the GradNorm loss-weight optimiser (Adam over the '
                             'per-term weights; separate from the model optimiser).')
    parser.add_argument('--freq_mask_prob', type=float, default=0.0,
                        help='probability of replacing patch-mask reconstruction with masked-frequency-band reconstruction on a given batch (0 = always patch-mask)')
    parser.add_argument('--freq_recon_n_bands', type=int, default=5,
                        help='number of equal-width frequency bands to split the rFFT axis into for masked-frequency-band reconstruction')
    parser.add_argument('--band_mask_ratio', type=float, default=None,
                        help='fraction of bands to mask per sample for masked-frequency-band reconstruction; k=max(1, round(freq_recon_n_bands * band_mask_ratio)). None (default) keeps the legacy behavior of masking exactly 1 band per sample.')
    parser.add_argument('--recon_band_split', action='store_true', default=False,
                        help='reconstruct only the LOW-frequency part of the masked patches (raw signal / time domain): rFFT bins with f >= --recon_phase_cutoff_hz are dropped, so delta-band waveform (phase informative) is reconstructed while induced mu/beta (unpredictable phase) is not penalised. Loss is MSE per retained DOF, so scale stays comparable and a large cutoff reproduces plain MSE exactly. Default off = legacy time-domain MSE. Bin width = --fs / --in_dim.')
    parser.add_argument('--recon_phase_cutoff_hz', type=float, default=8.0,
                        help='reconstruct frequencies strictly BELOW this (Hz) in --recon_band_split; a large value (>= Nyquist) or +inf reproduces the plain MSE bit-for-bit, smaller keeps less (only the slow/evoked band).')
    parser.add_argument('--recon_power_weight', type=float, default=1.0,
                        help='(unused in the current low-freq-only --recon_band_split; kept for back-compat).')
    parser.add_argument('--aux_band_pred', action='store_true', default=False,
                        help='ON TOP of the plain masked MSE, add Hilbert-derived band targets on the masked patches: instantaneous PHASE agreement for the delta band (--aux_delta_band, amplitude-weighted) and POWER-ENVELOPE matching for mu/beta (--aux_power_bands, scale-free relative MSE). Phase is informative at low freq; induced mu/beta phase is not, so only their band-power time course (ERD/ERS) is scored. Targets are computed on the model-completed waveform (true signal at visible patches, predictions at masked ones). See utils.util.band_phase_envelope_loss. Uses --fs for band edges.')
    parser.add_argument('--aux_delta_band', type=str, default='0.5,4',
                        help='lo,hi Hz of the delta band whose instantaneous phase is predicted in --aux_band_pred.')
    parser.add_argument('--aux_power_bands', type=str, default='8,13;13,30',
                        help='semicolon-separated lo,hi Hz bands (default mu 8-13 and beta 13-30) whose instantaneous power envelope is predicted in --aux_band_pred; the relative-MSE term is averaged across them.')
    parser.add_argument('--aux_phase_weight', type=float, default=1.0,
                        help='weight of the delta-phase auxiliary loss in --aux_band_pred. Raw phase loss is a bounded amplitude-weighted cosine distance and runs small (~0.01-0.05 once delta is reconstructed), so the default 1.0 makes its weighted contribution ~20%% of the plain mask_loss at convergence. Watch aux_phase_loss in wandb and retune so weight*aux_phase_loss sits ~10-30%% of mask_loss.')
    parser.add_argument('--aux_envelope_weight', type=float, default=0.005,
                        help='weight of the mu/beta power-envelope auxiliary loss in --aux_band_pred. Raw env loss is a log-power (dB-style) MSE and runs large (~1-10), so the default 0.005 makes its weighted contribution ~40%% of the plain mask_loss at convergence. Watch aux_env_loss in wandb and retune so weight*aux_env_loss sits ~10-40%% of mask_loss.')
    parser.add_argument('--aux_hand_pred', action='store_true', default=False,
                        help='Auxiliary objective: decode the CONTINUOUS EgoBrain hand-movement annotations (per-window left/right hand intensity in hand-lengths/s; see datasets/egobrain_hand_labels.py) from each EEG window global rep via a small MLP head + masked SmoothL1 regression, added to the pretrain loss as info["hand_pred_loss"]. Per-column masked to EgoBrain rows with a detected hand (undetected hand -> NaN -> skipped); non-EgoBrain rows contribute nothing. On frame-averaging flip steps the left/right targets are swapped to match the mirrored scene. Requires --egobrain_hand_labels_dir + an EgoBrain source. Watch hand_pred_loss / diag_hand_mae in wandb.')
    parser.add_argument('--aux_hand_weight', type=float, default=0.1,
                        help='weight of the --aux_hand_pred regression loss. Targets are O(0.05-4) hand-lengths/s (mostly <1) so SmoothL1 runs ~0.1-0.5; default 0.1 keeps weight*hand_pred_loss a small fraction of mask_loss. Retune so it sits ~5-20%% of mask_loss.')
    parser.add_argument('--use_brain_embed', action='store_true', default=False,
                        help="CSBrain spatial-mixing residual (models/CSBrain.py): add BrainEmbedEEGLayer(patch_emb)+patch_emb — a circular conv across EEG channels — after the TemEmbed residual in every encoder layer of the world-model backbone. Channels are region-GROUPED first (per-sample from ch_names, the EgoBrain analogue of CSBrain's fixed x=x[:,sorted_indices]); since area_config is None here, this is the only channel-order-dependent op so grouping it locally leaves the coord PE / masks / reconstruction target in native channel order. Off = current behavior (byte-identical).")
    parser.add_argument('--dataset_dir', type=str, default='path/to/dataset', help='dataset_dir')
    parser.add_argument('--model_dir', type=str, default='outputs', help='model_dir') # eg. 'CSBrain/pth'
    parser.add_argument('--TemEmbed_kernel_sizes', type=str, default="[(1,), (3,), (5,)]")
    parser.add_argument('--use_SmallerToken', type=bool, default=False, help='SmallerToken->dataset.py')
    parser.add_argument('--model', type=str, default='CSBrain', help='CSBrain')
    parser.add_argument('--vision_encoder', type=str, default='facebook/dinov2-base',
                        help='HF model id of the frozen vision encoder used for image alignment (e.g. facebook/dinov2-base or facebook/vjepa2-vitl-fpc64-256). Must match the preprocessing used to cache image_encoder_inputs; V-JEPA 2 expects pixel_values_videos at 256x256.')
    parser.add_argument('--image_pool_heads', type=int, default=4,
                        help='Heads for the learned-query attention pool over V-JEPA 2 patch tokens (Align model).')
    parser.add_argument('--use_saliency', action='store_true', default=False,
                        help='Enable the saliency-alignment branch on the Spectral model (KL between band-mixed saliency prediction and DINO attention rollout).')
    parser.add_argument('--saliency_rollout_skip_layers', type=int, default=2,
                        help='Number of initial vision-encoder layers to exclude from the saliency rollout target (Spectral model).')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--causal', action='store_true', default=False, help='use causal (next-patch prediction) instead of masked reconstruction')
    parser.add_argument('--project_to_source', action='store_true', default=False, help='project sensors to source space before transformer')
    parser.add_argument('--num_sources', type=int, default=32, help='number of brain sources')
    parser.add_argument('--decorr_weight', type=float, default=0.1, help='decorrelation loss weight for source projector pretraining')
    parser.add_argument('--source_projector_ckpt', type=str, default=None, help='path to pre-trained source projector checkpoint')
    parser.add_argument('--freeze_source_projector', action='store_true', default=False, help='freeze source projector weights during stage 2')
    parser.add_argument('--adversarial_weight', type=float, default=0.0, help='weight for adversarial session-agnostic loss (0 = disabled)')
    parser.add_argument('--samples_per_session', type=int, default=8, help='samples per session in session-grouped batching')
    parser.add_argument('--sessions_per_batch', type=int, default=16, help='number of distinct sessions per batch')
    parser.add_argument('--equivariance_weight', type=float, default=0.0, help='weight for hemispheric equivariance loss (0 = disabled)')
    parser.add_argument('--info_max_weight', type=float, default=0.0, help='weight for VICReg-style info-max regulariser on inv/eq subspaces (0 = disabled)')
    parser.add_argument('--alignment_weight', type=float, default=1.0, help='weight for EEG-image contrastive alignment loss (0 = disabled)')
    # --- Bilateralization prior (learned x = x_bi + x_lat split + flip-equivariant alignment) ---
    parser.add_argument('--lateralization_flip', action='store_true', default=False,
                        help='enable the bilateralization prior: a learned frontend splits the raw EEG into bilateral (x_bi) + lateral (x_lat) streams (x_bi + x_lat = x); flipping only x_lat across homologous channels (C3<->C4) builds x_flip whose encoding is aligned to the horizontally-flipped frame')
    parser.add_argument('--flip_align_weight', type=float, default=1.0,
                        help='weight for the flip-equivariant image-alignment loss (0 = disabled)')
    parser.add_argument('--lat_sparsity_weight', type=float, default=0.01,
                        help='weight for the lateral-gate minimality penalty (keeps x_lat small so most signal stays bilateral; 0 = disabled)')
    parser.add_argument('--flip_split_hidden', type=int, default=64,
                        help='hidden width of the LateralizationSplit gate network')
    parser.add_argument('--flip_pred_weight', type=float, default=1.0,
                        help='WorldModel: multiplier on the flipped-prediction terms (predict the flipped-future EEG latent from the flipped-current via x_bi+flip(x_lat)); 0 disables. Active only with --lateralization_flip and a predictor')
    parser.add_argument('--flip_n_col_bands', type=int, default=2,
                        help='number of vertical column bands for the centered spatial image descriptor (the flip target); 2 = left/right (strongest horizontal-flip signal), more = finer spatial detail; 1 = global pool (~flip-invariant CLS, ablation only)')
    parser.add_argument('--flip_motion_ref', type=float, default=0.0,
                        help='per-sample motion weighting of the flip loss: weight=clamp(motion/ref, flip_motion_min, 1) where motion=mean|frame_t+1 - frame_t|; 0 disables (uniform). EgoBrain static stretches have a near-vacuous flip, so ~15-20 down-weights them')
    parser.add_argument('--flip_motion_min', type=float, default=0.0,
                        help='floor on the per-sample flip motion weight so no row is fully zeroed')
    # --- Equivariant frame-averaging frontend (plans/eeg-wm.md) ---
    parser.add_argument('--frame_averaging', action='store_true', default=False,
                        help='upgrade the bilateral/lateral design into an invariance/equivariance decomposition under the homologous-channel swap P (P^2=I): a channel-independent frontend f produces z=z_bi+z_lat in feature space, then the transformer T is wrapped by frame averaging over {I,P} so h_bi=(T(z)+T(Pz))/2 is P-invariant and h_lat=(T(z)-P T(Pz))/2 is P-anti-equivariant; tokens split half-bilateral/half-lateral. A random per-step flip presents z or P(z) and the original/mirrored frame. Supersedes --lateralization_flip (mutually exclusive paths; frame_averaging takes precedence)')
    parser.add_argument('--frame_avg_flip_prob', type=float, default=0.5,
                        help='probability per training step of presenting the flipped orientation P(z) + horizontally-mirrored frame (frame averaging). 0.5 = balanced')
    parser.add_argument('--frame_avg_recon_weight', type=float, default=1.0,
                        help='weight of the flip-step reconstruction self-consistency loss: f(recon) matched to the presented clean latent P(z_clean) on masked patches (there is no ground-truth flipped timeseries). Non-flip steps use the standard masked MSE')
    parser.add_argument('--use_volume_conduction', action='store_true', default=False,
                        help='replace the sinusoidal spherical-coord channel positional encoding with a volume-conduction-aware encoding (arXiv 2601.06134): learnable exp(-d/tau) distance kernel over per-sample 3D electrode positions, row-normalised, smoothed positions projected to d_model and added per channel')
    parser.add_argument('--vc_tau_init', type=float, default=0.08,
                        help='initial volume-conduction decay length tau in metres (alpha is initialised so softplus(alpha) == vc_tau_init); ~8 cm matches typical inter-electrode spacing')

    # --- LLM-embedding VQ tokenization for contrastive path ---
    parser.add_argument('--use_llm_vq', action='store_true', default=False, help='tokenize the global token into K language tokens via a frozen LLM-embedding codebook before contrastive alignment')
    parser.add_argument('--num_language_tokens', type=int, default=8, help='number of LLM-vocab tokens produced from the global token (K)')
    parser.add_argument('--max_llm_codebook_size', type=int, default=4096, help='max codebook size (top-K most frequent LLM tokens kept)')
    parser.add_argument('--llm_vq_aux_weight', type=float, default=0.1, help='weight for the VQ commitment / entropy auxiliary loss')

    # --- Spectral branch inside the CNN patch embedder ---
    parser.add_argument('--spectral_mode', type=str, default='static', choices=['static', 'instantaneous'],
                        help='static: single FFT magnitude vector per patch (default). instantaneous: STFT-based spectrogram per patch processed by a Conv2d head.')
    parser.add_argument('--stft_n_fft', type=int, default=64, help='STFT n_fft for spectral_mode=instantaneous')
    parser.add_argument('--stft_hop', type=int, default=1, help='STFT hop_length for spectral_mode=instantaneous (hop=1 keeps the time axis equal to patch_size)')

    # --- Frequency-band MoE FFN (replaces dense FFN inside encoder layers) ---
    parser.add_argument('--use_moe', action='store_true', default=False,
                        help='replace the dense FFN in each encoder layer with a top-k MoE FFN whose gate is conditioned on the per-patch FFT magnitude; per-expert hidden = dim_feedforward // num_experts keeps the dense weight budget')
    parser.add_argument('--num_experts', type=int, default=4, help='number of MoE experts (must divide dim_feedforward)')
    parser.add_argument('--moe_top_k', type=int, default=2, help='top-k experts activated per token')
    parser.add_argument('--moe_gate_input_dim', type=int, default=32, help='dim of the spectral descriptor fed to the MoE gate')
    parser.add_argument('--moe_balance_weight', type=float, default=0.01, help='weight of the Switch-style load-balancing loss')
    parser.add_argument('--moe_band_prior_weight', type=float, default=0.1, help='weight of the KL(band-energy || gate) prior; set to 0 to disable the band-anchor warmup')
    parser.add_argument('--moe_z_loss_weight', type=float, default=1e-3, help='weight of the router z-loss (Zoph et al. ST-MoE) that keeps gate logits bounded; required to prevent late-training NaN')

    # --- Learnable spectral-band filterbank + cross-band + multi-level alignment ---
    parser.add_argument('--use_spectral_bands', action='store_true', default=False,
                        help='decompose raw EEG with a learnable SincNet-style filterbank into K ordered bands, add band-type embeddings, run them through the shared CSBrain backbone with cross-band attention, and align each band to multiple visual levels via a learned soft assignment (mutually exclusive with --use_moe)')
    parser.add_argument('--num_spectral_bands', type=int, default=4, help='number of learnable frequency bands K')
    parser.add_argument('--filterbank_kernel_size', type=int, default=101, help='SincNet bandpass FIR length (odd)')
    parser.add_argument('--filterbank_min_bw_hz', type=float, default=1.0, help='minimum per-band bandwidth in Hz (each band learns its own low/high cutoff independently and may overlap freely)')
    parser.add_argument('--use_cross_band_attn', action='store_true', default=True, help='enable cross-band attention (cross-frequency coupling) between encoder layers')
    parser.add_argument('--no_cross_band_attn', dest='use_cross_band_attn', action='store_false', help='disable cross-band attention (ablation)')
    parser.add_argument('--cross_band_every', type=int, default=1, help='apply cross-band attention every N encoder layers')
    parser.add_argument('--use_band_type_embedding', action='store_true', default=True, help='add a learned per-band type embedding')
    parser.add_argument('--no_band_type_embedding', dest='use_band_type_embedding', action='store_false', help='disable band-type embedding (ablation)')
    parser.add_argument('--num_visual_levels', type=int, default=3, help='number of image-encoder hidden levels (shallow/mid/deep) to align bands against')
    parser.add_argument('--band_decorr_weight', type=float, default=0.01, help='weight of the band-decorrelation regularizer on per-band aligned embeddings')

    # --- SSM/Mamba multi-frequency patch embedding ---
    parser.add_argument('--patch_embed_type', type=str, default='cnn', choices=['cnn', 'mamba'], help='patch embedder: CNN (default) or multi-frequency Mamba SSM')
    parser.add_argument('--mamba_band_periods', type=str, default=None, help='list of band sample periods, e.g. "[200,600,1200]"; default [in_dim, 3*in_dim, 6*in_dim]')
    parser.add_argument('--n_mamba_layers', type=int, default=2, help='number of stacked Mamba blocks in the patch embedder')
    parser.add_argument('--mamba_d_state', type=int, default=16, help='Mamba state dimension')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba depthwise conv width')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expansion factor')

    # --- DINOv2 self-distillation ---
    parser.add_argument('--dino_mode', action='store_true', default=False, help='enable DINOv2 self-distillation (replaces masked reconstruction)')
    parser.add_argument('--n_prototypes', type=int, default=4096, help='number of prototype vectors in DINO heads')
    parser.add_argument('--dino_head_hidden_dim', type=int, default=256, help='DINO head MLP hidden dimension')
    parser.add_argument('--dino_head_bottleneck_dim', type=int, default=256, help='DINO head bottleneck dimension before prototype layer')
    parser.add_argument('--dino_head_n_layers', type=int, default=3, help='number of MLP layers in DINO head')
    parser.add_argument('--student_temp', type=float, default=0.1, help='student softmax temperature')
    parser.add_argument('--teacher_temp_base', type=float, default=0.04, help='teacher temperature (initial)')
    parser.add_argument('--teacher_temp_final', type=float, default=0.07, help='teacher temperature (after warmup)')
    parser.add_argument('--teacher_temp_warmup_epochs', type=int, default=30, help='epochs to linearly warm up teacher temperature')
    parser.add_argument('--ema_momentum_base', type=float, default=0.992, help='EMA momentum for teacher (initial)')
    parser.add_argument('--ema_momentum_final', type=float, default=0.9995, help='EMA momentum for teacher (final)')
    parser.add_argument('--dino_loss_weight', type=float, default=1.0, help='weight for DINO CLS-token distillation loss')
    parser.add_argument('--ibot_loss_weight', type=float, default=1.0, help='weight for iBOT masked-patch distillation loss')
    parser.add_argument('--koleo_loss_weight', type=float, default=0.1, help='weight for KoLeo diversity loss')
    parser.add_argument('--use_freq_subband', action='store_true', default=False, help='enable frequency sub-band view augmentation for student')
    parser.add_argument('--freq_n_bands', type=int, default=5, help='number of frequency bands to divide the spectrum into')
    parser.add_argument('--freq_min_bands', type=int, default=1, help='minimum number of bands to keep in sub-band view')
    parser.add_argument('--freq_max_bands', type=int, default=None, help='maximum number of bands to keep (default: all)')
    parser.add_argument('--last_layer_freeze_iters', type=int, default=1250, help='freeze weight-norm magnitude of prototype layer for this many initial iterations (0 = never freeze)')
    parser.add_argument('--lr_warmup_iters', type=int, default=0, help='linear LR warmup iterations (0 = disabled)')
    parser.add_argument('--lr_warmup_frac', type=float, default=0.0,
                        help='linear LR warmup as a FRACTION of total training steps '
                             '(epochs * iters_per_epoch); 0 = disabled. Takes precedence '
                             'over --lr_warmup_iters when > 0, so it auto-scales with the '
                             'dataset size / epoch count.')
    # Multi-crop
    parser.add_argument('--n_local_crops', type=int, default=4, help='number of local EEG crops for DINO CLS loss (0 = single-view)')
    parser.add_argument('--local_crop_time_scale', type=str, default='(0.3, 0.7)', help='(min, max) fraction of time patches in local crops')
    parser.add_argument('--local_crop_channel_scale', type=str, default='(0.5, 1.0)', help='(min, max) fraction of channels in local crops')

    # --- World model (EEG video-prediction) ---
    parser.add_argument('--cinebrain_subjects', type=str, default='sub-0001', help='comma-separated list of CineBrain subject directories')
    parser.add_argument('--cinebrain_root', type=str, default='data/CineBrain', help='root directory for the CineBrain dataset')
    parser.add_argument('--cinebrain_n_windows', type=int, default=3, help='number of windows to emit per 4 s CineBrain clip')
    parser.add_argument('--cinebrain_window_s', type=float, default=2.0, help='EEG window length in seconds (must give window_samples divisible by in_dim)')
    parser.add_argument('--cinebrain_stride_s', type=float, default=1.0, help='stride between consecutive EEG windows in seconds')
    parser.add_argument('--cinebrain_erp_latency_s', type=float, default=0.0, help='time shift added to the frame-lookup timestamp (paper default: 0; sweep +/-0.15 to probe visual-ERP lag)')
    parser.add_argument('--cinebrain_load_frames', type=int, default=1, help='whether to decode raw video frames (0 disables the alignment term)')
    parser.add_argument('--predictor_d_model', type=int, default=512)
    parser.add_argument('--predictor_n_layers', type=int, default=4)
    parser.add_argument('--predictor_n_heads', type=int, default=8)
    parser.add_argument('--predictor_dim_feedforward', type=int, default=1024)
    parser.add_argument('--max_horizon', type=int, default=1, help='maximum prediction horizon k (in windows)')
    parser.add_argument('--wm_objective', type=str, default='eeg', choices=['eeg', 'frame'],
                        help="WorldModel predictor objective. 'eeg' (default): predict the next EEG "
                             "window's latent from the current EEG latent (original behavior). "
                             "'frame': predict the next video frame's per-patch embedding from the "
                             "current frame's per-patch embedding conditioned on the current EEG "
                             "embedding (target = frozen vision encoder; needs CineBrain/EgoBrain "
                             "frames and max_horizon>=1). Reuses --latent_pred_weight as the loss weight.")
    parser.add_argument('--wm_frame_eeg_cond', type=str, default='global', choices=['global', 'tokens'],
                        help="For --wm_objective frame: which EEG representation conditions the frame "
                             "predictor. 'global' (default): the window-level global rep (one vector). "
                             "'tokens': all EEG per-patch tokens (C*N tokens). Ignored for the eeg objective.")
    # ---- LeWM from-scratch, jointly-trained world model (plans/mighty-dazzling-lovelace.md) ----
    parser.add_argument('--wm_scratch_vit', action='store_true', default=False,
                        help="Train the video world model FROM SCRATCH (LeWorldModel) instead of using a "
                             "frozen pretrained DINOv2/V-JEPA. Replaces the vision encoder with a "
                             "randomly-initialized TRAINABLE ViT (models/scratch_vit.py) EVERYWHERE — "
                             "EEG<->frame alignment and the world-model frame prediction both run it live "
                             "and it receives gradient. The prediction target is the ViT's OWN online "
                             "embedding (no stop-grad, no EMA); collapse is held off SOLELY by SIGReg "
                             "(--wm_sigreg_weight). REQUIRES --egobrain_use_frame_grid and is INCOMPATIBLE "
                             "with --egobrain_use_grid_embeddings / --use_cached_embeddings (raw pixels "
                             "must flow every step). --wm_objective must be 'frame'.")
    parser.add_argument('--wm_predictor', type=str, default='frame',
                        choices=['frame', 'frame_adaln', 'ar'],
                        help="For --wm_scratch_vit: which predictor. 'frame' (default): dense "
                             "EEG-conditioned FramePredictor, query-transformer style (keeps "
                             "motion-weight / neg-EEG contrastive / flip / subject-block machinery). "
                             "'frame_adaln': SAME dense multi-frame-from-current-frame objective + all "
                             "that machinery, but the EEG is injected by AdaLN-zero conditioning (LeWM "
                             "ConditionalBlock) instead of concatenated tokens. 'ar': LeWM causal "
                             "ARPredictor over frame-CLS embeddings, conditioned per-step on the EEG.")
    parser.add_argument('--wm_sigreg_weight', type=float, default=0.09,
                        help="LeWM SIGReg regularizer weight (the single tunable loss weight). 0 disables "
                             "SIGReg (unsafe with a trainable ViT — collapse risk). Default 0.09 (LeWM).")
    parser.add_argument('--wm_sigreg_knots', type=int, default=17,
                        help="SIGReg Epps-Pulley quadrature knots (LeWM default 17).")
    parser.add_argument('--wm_sigreg_num_proj', type=int, default=1024,
                        help="SIGReg random 1-D projections per step (LeWM default 1024).")
    parser.add_argument('--scratch_vit_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help="From-scratch ViT preset (tiny=192d/12L/3H ~5.5M, LeWM default).")
    parser.add_argument('--scratch_vit_embed_dim', type=int, default=None,
                        help="Override the from-scratch ViT hidden size (must divide heads).")
    parser.add_argument('--scratch_vit_patch', type=int, default=16,
                        help="From-scratch ViT patch size (16 -> 14x14=196 patch grid at 224).")
    parser.add_argument('--scratch_vit_depth', type=int, default=None,
                        help="Override the from-scratch ViT depth (default: preset).")
    parser.add_argument('--scratch_vit_heads', type=int, default=None,
                        help="Override the from-scratch ViT attention heads (default: preset).")
    parser.add_argument('--wm_ar_history', type=int, default=3,
                        help="--wm_predictor ar: AR context length (LeWM history_size, default 3).")
    parser.add_argument('--wm_ar_num_preds', type=int, default=1,
                        help="--wm_predictor ar: prediction offset (LeWM num_preds, default 1).")
    parser.add_argument('--wm_ar_depth', type=int, default=6,
                        help="--wm_predictor ar: AR transformer depth (LeWM default 6).")
    parser.add_argument('--wm_ar_heads', type=int, default=16,
                        help="--wm_predictor ar: AR transformer heads (LeWM default 16).")
    parser.add_argument('--amp', action='store_true', default=False,
                        help="bf16 autocast for the trainable ViT forward ONLY (recommended with "
                             "--wm_scratch_vit — the live ViT over ~6 frames/step is heavy in fp32). "
                             "Scoped to the ViT (output cast back to fp32) so the EEG encoder's FFT "
                             "spectral ops stay fp32. No GradScaler needed for bf16.")
    parser.add_argument('--vision_lr', type=float, default=0.0,
                        help="Separate learning rate for the trainable ViT param group (--wm_scratch_vit). "
                             "0 (default) = use the single --lr for all params.")
    parser.add_argument('--wm_frame_motion_alpha', type=float, default=0.0,
                        help="For --wm_objective frame: per-patch motion weighting exponent on the dense "
                             "frame-prediction loss. 0 (default) = legacy UNIFORM mean over patches. >0 "
                             "weights each patch by how far its target grid moved from the anchor "
                             "(w=clip(motion/mean_motion, floor, inf)^alpha), concentrating the loss on "
                             "DYNAMIC patches — the only place the EEG can lower it — so the static "
                             "copy-the-anchor shortcut (which drives diag_frame_eeg_gap->0) stops scoring "
                             "well and gradient reaches the EEG. 1.0 = motion-proportional (recommended); "
                             "2.0 = aggressive. Self-normalising, so latent_pred_weight needs no retuning.")
    parser.add_argument('--wm_frame_motion_floor', type=float, default=0.1,
                        help="For --wm_objective frame with --wm_frame_motion_alpha>0: minimum per-patch "
                             "weight in [0,1] (uniform floor keeping fully-static steps finite and every "
                             "patch a little coverage). 0.1 = default.")
    parser.add_argument('--wm_frame_motion_ref', type=float, default=-1.0,
                        help="For --wm_objective frame with --wm_frame_motion_alpha>0: reference motion "
                             "for the weight normalisation clip(motion/ref, floor). <0 (default) uses the "
                             "FIXED baked-in per-horizon EGOBRAIN_FRAME_MOTION_REF_PER_STEP constant "
                             "(stable, lag-appropriate floor threshold; precomputed by "
                             "scripts/compute_frame_motion_ref.py). 0 = per-batch per-step mean (legacy, "
                             "jitters step-to-step). >0 = that explicit scalar for every step (override "
                             "for a different encoder/window config; recompute the constant instead if you "
                             "can).")
    parser.add_argument('--wm_frame_clean_cond', action='store_true', default=False,
                        help="For --wm_objective frame: asymmetric TWO-VIEW split. The masked forward "
                             "(needed for recon) feeds half mask-token reconstruction guesses + "
                             "intermittent masking, so alignment/prediction learn to ignore the EEG "
                             "(diag_frame_eeg_gap ~0). With this on, the MASKED view is the reconstruction "
                             "pretext ONLY (mask_loss + spectral aux + frame_recon), and a second UNMASKED "
                             "forward carries everything downstream-facing on the complete signal — image "
                             "alignment, flip-align, hand-pred, and the frame predictor's EEG conditioning. "
                             "Costs one extra (cheap) EEG-encoder forward. Off = legacy (single masked "
                             "forward carries all losses).")
    parser.add_argument('--wm_frame_contrast_weight', type=float, default=0.0,
                        help="For --wm_objective frame: weight of the negative-EEG CONTRASTIVE term. "
                             "The dense frame predictor learns to copy the anchor grid and ignore the "
                             "EEG (diag_frame_eeg_gap ~0). This re-runs the predictor on the SAME anchor "
                             "grid conditioned on OTHER rows' EEG (in-batch negatives) and penalises it "
                             "for reproducing the true future from the WRONG EEG — so the copy shortcut "
                             "can no longer win and the EEG conditioning becomes load-bearing. Negatives "
                             "are drawn only from rows sharing the anchor's frame-averaging flip (so "
                             "orientation can't be the discriminator under --frame_averaging). Watch "
                             "diag_frame_contrast_acc (chance~1/(realised negs+1); strict tie-break -> the "
                             "EEG-ignoring copy mode reads chance, not 1) and "
                             "diag_frame_contrast_gap (d_neg - d_pos; > 0 & growing). Costs n_neg extra "
                             "predictor forwards/step. 0 (default) = OFF (no extra forwards). BEST PAIRED "
                             "with --wm_frame_motion_alpha>0 so the contrast focuses on DYNAMIC patches "
                             "(else it can be satisfied by encoding static scene identity).")
    parser.add_argument('--wm_frame_contrast_n_neg', type=int, default=1,
                        help="For --wm_frame_contrast_weight>0: number of in-batch negative EEG samples "
                             "(distinct batch rolls) per row. More = harder task / stronger signal but "
                             "n_neg extra predictor forwards/step. Capped at batch_valid_rows-1.")
    parser.add_argument('--wm_frame_contrast_temp', type=float, default=0.1,
                        help="For --wm_frame_contrast_weight>0 with mode infonce: softmax temperature on "
                             "-distance logits. Smaller amplifies the (initially tiny) distance gaps.")
    parser.add_argument('--wm_frame_contrast_mode', type=str, default='infonce',
                        choices=['infonce', 'margin'],
                        help="For --wm_frame_contrast_weight>0: 'infonce' (softmax over {pos, negs}, the "
                             "true EEG must give the smallest prediction distance) or 'margin' "
                             "(softplus(margin + d_pos - d_neg), each negative must exceed the positive "
                             "distance by --wm_frame_contrast_margin).")
    parser.add_argument('--wm_frame_contrast_margin', type=float, default=0.1,
                        help="For --wm_frame_contrast_mode margin: required d_neg - d_pos margin.")
    parser.add_argument('--wm_frame_contrast_detach_neg', action='store_true', default=True,
                        help="For --wm_frame_contrast_weight>0: detach the negative EEG from the encoder "
                             "graph (default) so ONLY the predictor learns the discrimination from the "
                             "negatives; the EEG backbone is still pushed via the positive term. This "
                             "keeps the arbitrary (eeg_j, scene_i) pairings from injecting noise into the "
                             "downstream-facing EEG rep.")
    parser.add_argument('--wm_frame_contrast_grad_neg', dest='wm_frame_contrast_detach_neg',
                        action='store_false',
                        help="Let the negative EEG carry gradient to the EEG encoder too (standard "
                             "InfoNCE; overrides the default --wm_frame_contrast_detach_neg).")
    parser.add_argument('--wm_frame_contrast_batched', action='store_true', default=False,
                        help="For --wm_frame_contrast_weight>0: run all n_neg negative predictor "
                             "forwards as ONE batched call over n_neg*Bv rows instead of an n_neg-step "
                             "loop. Numerically identical (up to dropout RNG); fewer/larger kernels but "
                             "~n_neg x peak attention memory. Off (default) = the memory-safe loop; turn "
                             "ON to test whether the GPU has the capacity for the single-forward path.")
    parser.add_argument('--wm_frame_contrast_excl_s', type=float, default=-1.0,
                        help="For --wm_frame_contrast_weight>0 WITH --egobrain_subject_block>0: temporal "
                             "exclusion (seconds) between a subject-block row and its in-batch negatives. A "
                             "candidate whose window-0 start is within this of the anchor is masked out of "
                             "the contrast (it would share EEG samples with the positive = a near-duplicate "
                             "false negative). -1 (default) = auto = --egobrain_window_s, forbidding literal "
                             "EEG overlap; under temporal jitter it almost never fires. Ignored without "
                             "subject-block loading (there are no same-subject in-batch negatives to mask).")
    parser.add_argument('--latent_pred_weight', type=float, default=1.0)
    parser.add_argument('--cls_pred_weight', type=float, default=0.1)
    parser.add_argument('--pred_ramp_epochs', type=int, default=2, help='linearly ramp latent-prediction weight 0→1 over this many epochs')
    parser.add_argument('--target_momentum', type=float, default=0.998, help='EMA momentum for WorldModel target encoder (0 disables EMA — targets come from the online encoder)')
    parser.add_argument('--mix_alljoined_weight', type=float, default=1.0, help='sampling weight for Alljoined-1.6M in the mix+cinebrain dataset')
    parser.add_argument('--mix_cinebrain_weight', type=float, default=1.0, help='sampling weight for CineBrain in the mix+cinebrain dataset')

    # --- EgoBrain (motion-rich egocentric EEG + GoPro video) ---
    parser.add_argument('--egobrain_subjects', type=str, default='P0001', help='comma-separated EgoBrain subject ids (e.g. P0001,P0002) or "all"')
    parser.add_argument('--egobrain_root', type=str, default='data/EgoBrain', help='root directory for the EgoBrain dataset (after datasets.egobrain_preprocess)')
    parser.add_argument('--egobrain_n_windows', type=int, default=3, help='number of windows to emit per EgoBrain clip')
    parser.add_argument('--egobrain_window_s', type=float, default=2.0, help='EEG window length in seconds (window_samples must be divisible by in_dim)')
    parser.add_argument('--egobrain_stride_s', type=float, default=1.0, help='stride between consecutive EgoBrain windows in seconds')
    parser.add_argument('--egobrain_clip_s', type=float, default=4.0, help='preprocessed clip length in seconds (must match egobrain_preprocess --clip_s)')
    parser.add_argument('--egobrain_erp_latency_s', type=float, default=0.0, help='time shift added to the frame-lookup timestamp')
    parser.add_argument('--egobrain_load_frames', type=int, default=1, help='whether to decode raw video frames (0 disables the alignment term)')
    parser.add_argument('--egobrain_max_channels', type=int, default=32, help='cap on the EgoBrain channel count after 10-20 montage filtering')
    parser.add_argument('--egobrain_hand_labels_dir', type=str, default=None,
                        help='LEGACY clip-keyed hand-movement-annotation HDF5 dir (datasets/egobrain_hand_labels.py output, e.g. data/EgoBrain/cache_hand_labels_wilor_w1.0s1.0_e0.5_nw2_k7_c4.0_fs200). Enables --aux_hand_pred on the clip-keyed path only. The window slug MUST match the egobrain_window_s/stride_s/erp_latency_s/n_windows/clip_s/fs_out used here or the (clip,window) label keys misalign. INCOMPATIBLE with --egobrain_use_frame_grid (raises) — use --egobrain_hand_grid_dir there.')
    parser.add_argument('--egobrain_hand_grid_dir', type=str, default=None,
                        help='TIME-KEYED (grid) hand-movement-annotation HDF5 dir (datasets/egobrain_extract_hand_labels_grid.py output, e.g. data/EgoBrain/cache_hand_labels_grid_wilor_g0.2_raw_fs200). The --egobrain_use_frame_grid counterpart of --egobrain_hand_labels_dir: RAW per-0.2s-slot hand speed (forward pair, no smoothing) read at each window frame slot, so --aux_hand_pred works under the grid path. Its grid_s MUST equal --egobrain_frame_grid_s (validated at load). Requires --egobrain_use_frame_grid.')
    parser.add_argument('--mix_egobrain_weight', type=float, default=1.0, help='sampling weight for EgoBrain in the mix+egobrain dataset')
    parser.add_argument('--egobrain_delta_whiten_g0', type=float, default=1.0,
                        help='ME->MI delta-whitening DC gain: per-channel low-freq gain at 0 Hz, '
                             'raised-cosine ramp up to 1.0 at --egobrain_delta_whiten_cutoff_hz '
                             '(bins above the cutoff are untouched; per-channel RMS is preserved). '
                             '1.0 = OFF (no-op). ~0.60-0.78 (pipeline-dependent) attenuates the motor-execution delta '
                             'floor toward the motor-imagery level; calibrate to the finetune LMDB '
                             'MI delta level (~46 montage / ~42 central, in percent), do NOT '
                             'overshoot below it. See project_me_mi_pretrain_manipulation.')
    parser.add_argument('--egobrain_delta_whiten_cutoff_hz', type=float, default=8.0,
                        help='upper edge (Hz) of the ME->MI delta-whitening ramp.')
    parser.add_argument('--use_cached_embeddings', action='store_true', default=False,
                        help='Load PRE-COMPUTED frozen-vision-encoder embeddings of the EgoBrain '
                             'frames (cls/cls_flip + grid/grid_flip) from the cache_embeddings_<enc>_w.. '
                             'dir built by datasets.egobrain_extract_embeddings, instead of running '
                             'the encoder on the fly. Eliminates the dominant per-step GPU cost; '
                             'numerically equivalent to the live path (both frame orientations are '
                             'cached exactly). The cache slug must match the run window/stride/erp/'
                             'n_windows/sz and --vision_encoder (DINOv2-style only for now). '
                             'Default off = current behavior.')
    parser.add_argument('--egobrain_emb_cache_dir', type=str, default=None,
                        help='override the embedding-cache dir; default derives the '
                             'cache_embeddings_<enc>_w<ws>s<ss>_e<erp>_nw<nw>_sz<sz> slug.')
    parser.add_argument('--egobrain_use_frame_grid', action='store_true', default=False,
                        help='Use the CONTINUOUS, time-keyed frame grid (datasets.'
                             'egobrain_extract_frames_grid) instead of the clip-keyed '
                             'frame cache. EEG windows are sampled at any --egobrain_'
                             'frame_grid_s-snapped offset across the whole recording '
                             '(no 4 s clip boundary; recovers the discarded tail + adds '
                             'temporal variety) and paired with the nearest grid frame. '
                             'Cache is agnostic to window/stride/erp/n_windows. '
                             'Incompatible with --use_cached_embeddings. Default off = '
                             'current clip-keyed behavior (reproduces prior results).')
    parser.add_argument('--egobrain_frame_grid_dir', type=str, default=None,
                        help='override the grid frame-cache dir; default derives the '
                             'cache_frames_grid_<enc>_g<grid_s>_sz<sz> slug.')
    parser.add_argument('--egobrain_frame_grid_s', type=float, default=0.2,
                        help='time-grid spacing in seconds for --egobrain_use_frame_grid '
                             '(default 0.2 = the EEG patch size; window centres snap to it).')
    parser.add_argument('--egobrain_no_temporal_jitter', action='store_true', default=False,
                        help='with --egobrain_use_frame_grid, disable random anchor '
                             'jitter (use the deterministic clip-aligned offset). Off by '
                             'default = random offset augmentation each epoch.')
    parser.add_argument('--egobrain_subject_block', type=int, default=0,
                        help='lay each batch down as batch_size//N contiguous BLOCKS of N '
                             'same-subject clips (SubjectBlockBatchSampler) instead of a '
                             'plain shuffle. >0 enables (1) same-block negatives for the '
                             '--wm_frame_contrast term — same subject/scene, so the '
                             'predictor must read motor content not identity — and (2) '
                             'per-subject HDF5 read locality. batch_size must be divisible '
                             'by N (and by N*num_dp_gpus under DataParallel so blocks stay '
                             'whole per replica). 0 (default) = shuffled RandomSampler, '
                             'byte-identical to prior runs.')
    parser.add_argument('--egobrain_block_window_s', type=float, default=0.0,
                        help='with --egobrain_subject_block>0: confine each block to a random '
                             'CONTIGUOUS time window of ~this many seconds (HARD negatives) '
                             'instead of drawing block-mates uniformly across the ~90min '
                             'recording (~30min apart = easy). Switches the anchor draw to '
                             'clip-local jitter so contiguous clips map to a tight window; '
                             'block-mates then sit ~window_s/3 apart on average. When '
                             '--egobrain_motion_resample is also on, the window PLACEMENT is '
                             'motion-biased so proximity blocks still land on dynamic regions. '
                             '0 (default) = whole-recording blocks. Tighten the exclusion '
                             '(--wm_frame_contrast_excl_s) as the window shrinks toward the '
                             'movement autocorrelation (~1-2s) to avoid false negatives.')
    parser.add_argument('--egobrain_use_grid_embeddings', action='store_true', default=False,
                        help='with --egobrain_use_frame_grid, read PRE-COMPUTED frozen '
                             'DINOv2 embeddings (cls/grid + flips) from the time-keyed '
                             'cache_embeddings_grid_<enc>_g<g>_sz<sz> dir (built by '
                             'datasets.egobrain_extract_embeddings_grid) at each window '
                             'slot, so the encoder is skipped — encoder-skip speedup AND '
                             'continuous-offset variety. Only effective in the map-style '
                             '`egobrain` loader (the mix/iterable path runs the encoder '
                             'live regardless). Default off.')
    parser.add_argument('--egobrain_emb_grid_dir', type=str, default=None,
                        help='override the time-keyed embedding-grid cache dir; default '
                             'derives the cache_embeddings_grid_<enc>_g<g>_sz<sz> slug.')
    parser.add_argument('--egobrain_video_subjects', type=str, default=None,
                        help='Which EgoBrain subjects may contribute VIDEO (frames / '
                             'embeddings / hand-labels / motion resampling). Accepts '
                             '"all" (default: every subject that has a cache file), '
                             '"legacy24" (P0001-P0024 = the subjects that had video '
                             'before EgoBrain released MP4s for P0025-P0040), or a '
                             'comma-separated subject list. Blacklisted subjects keep '
                             'has_image=False and contribute EEG-only losses, exactly '
                             'as before their video existed. Use "legacy24" to '
                             'reproduce pre-release results bit-for-bit while the new '
                             'subjects\' caches sit on disk.')
    parser.add_argument('--egobrain_motion_resample', action='store_true', default=False,
                        help='with --egobrain_use_frame_grid, bias the random anchor draw '
                             'toward visually DYNAMIC moments instead of uniform-in-time. '
                             'EgoBrain video is mostly static, so uniform sampling wastes '
                             '~half the budget on frozen frames where next-frame prediction '
                             'is trivial; this draws the anchor k ∝ clip(motion(k),0,p_cap)^'
                             'alpha (mixed with a uniform floor), where motion(k) is the '
                             'frame-to-frame distance across the prediction step from the '
                             'time-keyed embedding cache. Only WITHIN-subject position is '
                             'reweighted (cross-subject balance unchanged). Per-subject '
                             'motion scores are cached under the embedding cache '
                             '_motioncache/ (built lazily or via '
                             'scripts.build_egobrain_motion_cache). Default off.')
    parser.add_argument('--egobrain_motion_resample_alpha', type=float, default=1.0,
                        help='exponent on motion in the anchor weight (0=uniform, 1=linear, '
                             '2=aggressive). See outputs/eval_tables.md for the budget '
                             'concentration each alpha induces.')
    parser.add_argument('--egobrain_motion_resample_space', type=str, default='patch',
                        choices=['patch', 'pixel', 'cls'],
                        help="representation the motion is measured in: 'patch' (DINOv2 grid "
                             'tokens = the world-model target, V-JEPA-portable; default), '
                             "'pixel' (raw frames; cheaper but inflates screen/lighting), or "
                             "'cls' (DINOv2 global; not V-JEPA-portable).")
    parser.add_argument('--egobrain_motion_resample_metric', type=str, default='cos',
                        choices=['cos', 'l1'],
                        help="distance metric for motion: 'cos' (1-cosine, clean static/"
                             "dynamic separation; default) or 'l1' (the loss metric). Both "
                             'rank anchors near-identically (Spearman ~0.999).')
    parser.add_argument('--egobrain_motion_resample_cap_pct', type=float, default=99.0,
                        help='winsorise motion at this per-subject percentile before the '
                             'exponent so a few extreme jumps do not swamp the budget.')
    parser.add_argument('--egobrain_motion_resample_floor_mix', type=float, default=0.1,
                        help='fraction of a uniform distribution mixed into the motion '
                             'weights so static regions keep a guaranteed coverage floor.')
    parser.add_argument('--egobrain_motion_emb_dir', type=str, default=None,
                        help='override the embedding cache dir the motion scores are read '
                             'from; default reuses the grid embedding cache slug.')

    # --- ActionWorldModel (EEG=action, frozen V-JEPA 2=world) ---
    parser.add_argument('--eeg_ckpt', type=str, default=None,
                        help='optional pretraining checkpoint (e.g. masked recon) to warm-start the ActionWorldModel EEG backbone')
    parser.add_argument('--action_dim', type=int, default=64,
                        help='dimension of each EEG-decoded action ("moving intention") token')
    parser.add_argument('--n_action_tokens', type=int, default=8,
                        help='number of action tokens the EEG action head emits')
    parser.add_argument('--action_pool_heads', type=int, default=4,
                        help='attention heads in the EEG action-pooling head')
    parser.add_argument('--pred_residual', action='store_true', default=False,
                        help='ActionPredictor predicts the residual s_t + delta (guards against the trivial copy solution)')
    parser.add_argument('--pred_l1_weight', type=float, default=1.0,
                        help='weight on the L1 next-world-state prediction loss')
    parser.add_argument('--pred_cos_weight', type=float, default=0.0,
                        help='weight on the (1-cosine) auxiliary prediction loss')
    parser.add_argument('--recon_aux_weight', type=float, default=0.0,
                        help='ActionWorldModel: weight on the OPTIONAL masked-patch reconstruction co-objective '
                             '(done as a separate masked EEG forward so the action stays decoded from the full '
                             'window). 0 disables it. Uses --mask_ratio. Alternatively leave --need_mask on to '
                             'route recon through the trainer mask_loss instead.')

    # --- Euclidean Alignment (per-subject whitening, arXiv:2601.17883 Eq 9-10) ---
    parser.add_argument('--use_euclidean_alignment', action='store_true', default=False,
                        help='apply per-subject Euclidean Alignment whitening before patching. '
                             'Requires precomputed sidecar files written by `python -m datasets.compute_ea {egobrain,cinebrain,alljoined,physio}`. '
                             'Subjects without a matrix in the sidecar pass through unchanged.')
    parser.add_argument('--ea_egobrain_path', type=str,
                        default='data/EgoBrain/cache_eeg_200hz/ea_subject.pt',
                        help='path to the EgoBrain EA sidecar (.pt)')
    parser.add_argument('--ea_cinebrain_path', type=str,
                        default='data/CineBrain/cache_eeg_200hz/ea_subject.pt',
                        help='path to the CineBrain EA sidecar (.pt)')
    parser.add_argument('--ea_alljoined_path', type=str,
                        default='data/cache/Alljoined-1.6M/ea_subject.pt',
                        help='path to the Alljoined-1.6M EA sidecar (.pt)')

    # --- Band-contrastive pretraining ---
    parser.add_argument('--contrastive_band', action='store_true', default=False,
                        help='enable band-contrastive pretraining (replaces masked reconstruction)')
    parser.add_argument('--fs', type=int, default=200, help='sampling rate (Hz) for band-contrastive bandpass split')
    parser.add_argument('--band_cutoffs', type=str, default='1,10',
                        help='comma-separated Hz cut frequencies; K-1 values define K bands. '
                             'Default "1,10" -> [<1 Hz, 1-10 Hz, >=10 Hz].')
    parser.add_argument('--contrastive_temp', type=float, default=0.1,
                        help='InfoNCE temperature for band-contrastive loss')
    parser.add_argument('--contrastive_proj_dim', type=int, default=64,
                        help='projection head output dim for band-contrastive loss')

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    params.model_dir = os.path.join(params.model_dir, params.run_name)

    if os.environ.get('DEBUG', '0') == '1':
        params.batch_size = 4

    # Resolve the subject-block frame-contrast temporal exclusion (seconds ->
    # fs_out=200 samples, matching EgoBrain's cached anchor_sample granularity).
    # Only meaningful with --egobrain_subject_block>0; auto (-1) defaults to the
    # EEG window span so negatives never share window-0 samples with the anchor.
    _excl_s = params.wm_frame_contrast_excl_s
    if _excl_s < 0:
        _excl_s = params.egobrain_window_s
    params.wm_frame_contrast_excl_samples = (
        int(round(_excl_s * 200)) if params.egobrain_subject_block > 0 else 0)
    # The temporal-proximity window rides on the subject-block sampler (it only
    # controls WHICH clips land in a block); it is meaningless without it.
    if params.egobrain_block_window_s > 0 and params.egobrain_subject_block <= 0:
        parser.error('--egobrain_block_window_s requires --egobrain_subject_block>0 '
                     '(the window confines the block sampler; there is no block to '
                     'confine otherwise).')

    # --- LeWM from-scratch world model guardrails ---
    # A trainable from-scratch ViT must run LIVE on raw pixels every step, so the
    # data path has to serve real 224² frames (not cached DINOv2 embeddings) and
    # the objective must be the frame predictor.
    if getattr(params, 'wm_scratch_vit', False):
        if params.wm_objective != 'frame':
            parser.error("--wm_scratch_vit requires --wm_objective frame.")
        if not params.egobrain_use_frame_grid:
            parser.error("--wm_scratch_vit requires --egobrain_use_frame_grid (the "
                         "trainable ViT reads raw 224² frames from the grid cache).")
        if params.egobrain_use_grid_embeddings or params.use_cached_embeddings:
            parser.error("--wm_scratch_vit is incompatible with cached embeddings "
                         "(--egobrain_use_grid_embeddings / --use_cached_embeddings): the "
                         "trainable ViT must run live on raw pixels every step.")

    # --- Euclidean Alignment sidecars (per-subject whitening matrices) ---
    # When --use_euclidean_alignment is on, load whichever sidecars the
    # active dataset branch will consume. Loaders that don't get a matrix
    # for a given subject silently pass that subject's data through
    # unchanged, so a missing sidecar degrades gracefully (with a warning).
    ea_egobrain = ea_cinebrain = ea_alljoined = None
    if params.use_euclidean_alignment:
        from datasets.euclidean_alignment import load_ea_matrices
        if 'egobrain' in params.dataset_dir:
            ea_egobrain = load_ea_matrices(params.ea_egobrain_path)
        if 'cinebrain' in params.dataset_dir:
            ea_cinebrain = load_ea_matrices(params.ea_cinebrain_path)
        if 'mix' in params.dataset_dir:               # mix mode always pulls Alljoined
            ea_alljoined = load_ea_matrices(params.ea_alljoined_path)

    if params.dataset_dir == 'mix':
        from datasets.cached_dataset import (
            get_webdataset, collate_cached, SessionGroupedLoader,
            set_active_vision_encoder,
        )
        set_active_vision_encoder(params.vision_encoder)
        from torch.utils.data import ConcatDataset
        dataset_names = [
            'tueg/*.tar',
            # 'siena_scalp/*.tar',
            # 'physionet_2018/*.tar',
            # 'raw_eeg/*.tar',
            # 'ds006171/*.tar', 
            # 'ds006317/*.tar',
            # 'ds006367/*.tar', 
            # 'ds006370/*.tar', 
            # 'ds006437/*.tar', 
            # 'ds006446/*.tar', 
            # 'ds006466/*.tar', 
            # 'ds006480/*.tar', 
            # 'ds006525/*.tar', 
            # 'ds006547/*.tar',
            # "Alljoined-1.6M/*.tar",
            # "things_eeg2/*.tar",
        ]
        n_samples_per_epoch = 1109545
        pretrained_dataset = get_webdataset(
            dataset_names,
            params,
            event_window_target_len=params.seq_len * params.in_dim,
        ) #resampled=True

        num_workers = 8 if os.environ.get('DEBUG', '0') == '0' else 0

        if params.adversarial_weight > 0:
            # Session-grouped batching.
            # WebDataset yields individual samples (no .batched());
            # multi-worker DataLoader interleaves samples from different
            # shards (different sessions); SessionGroupedLoader in the
            # main process collects them into session-balanced batches.
            effective_batch_size = params.samples_per_session * params.sessions_per_batch
            n_batches_per_epoch = n_samples_per_epoch // effective_batch_size
            samples_per_worker = math.ceil(n_samples_per_epoch / max(1, num_workers))

            pretrained_dataset = (
                pretrained_dataset
                .shuffle(2000)
                .with_epoch(samples_per_worker)
                .with_length(n_samples_per_epoch)
            )

            raw_loader = DataLoader(
                pretrained_dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=False,
            )
            data_loader = SessionGroupedLoader(
                raw_loader,
                samples_per_session=params.samples_per_session,
                sessions_per_batch=params.sessions_per_batch,
                collation_fn=collate_cached,
                batches_per_epoch=n_batches_per_epoch,
            )
        else:
            n_batches_per_epoch = (n_samples_per_epoch + params.batch_size - 1) // params.batch_size
            batches_per_worker = math.ceil(n_batches_per_epoch / max(1, num_workers))
            total_yielded_batches = batches_per_worker * max(1, num_workers)

            pretrained_dataset = (
                pretrained_dataset
                .shuffle(5000)
                .batched(params.batch_size, partial=True, collation_fn=collate_cached)
                .with_epoch(batches_per_worker)
                .with_length(total_yielded_batches)
            )

            data_loader = DataLoader(
                pretrained_dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=True,
            )
       
    elif params.dataset_dir == 'mix+cinebrain':
        from datasets.cached_dataset import (
            get_webdataset, collate_cached, collate_cached_with_future,
            set_active_vision_encoder,
        )
        from datasets.cinebrain_dataset import (
            CineBrainDataset, CineBrainIterableWrapper, WeightedSampleMix,
        )
        # Pin the live ``images`` collate path to the same encoder the
        # model uses; cached pixel_values for the no-``images`` path must
        # already match this encoder.
        set_active_vision_encoder(params.vision_encoder)

        # Crop the Alljoined event window to seq_len * in_dim so each
        # Alljoined sample lines up shape-for-shape with a CineBrain
        # window (C, seq_len, in_dim) and they can ride in the same batch.
        target_event_len = params.seq_len * params.in_dim

        alljoined_ds = get_webdataset(
            [
                "Alljoined-1.6M/*.tar",
                # "tueg/*.tar",
            ],
            params,
            event_window_target_len=target_event_len,
            ea_matrices=ea_alljoined,
        )

        # When the latent predictor is on we need at least max_horizon+1
        # CineBrain windows per clip; otherwise a single window suffices
        # and we skip the extra preprocessing.
        n_windows_cb = max(1, params.max_horizon + 1)
        subjects = [s.strip() for s in params.cinebrain_subjects.split(',')
                    if s.strip()]
        cinebrain_inner = CineBrainDataset(
            data_dir=params.cinebrain_root,
            subjects=subjects,
            in_dim=params.in_dim,
            n_windows=n_windows_cb,
            window_s=params.cinebrain_window_s,
            stride_s=params.cinebrain_stride_s,
            erp_latency_s=params.cinebrain_erp_latency_s,
            load_frames=bool(params.cinebrain_load_frames),
            vision_encoder=params.vision_encoder,
            ea_matrices=ea_cinebrain,
        )
        print('CineBrain clips:', len(cinebrain_inner))
        cinebrain_iter = CineBrainIterableWrapper(cinebrain_inner)

        # Match the 'mix' branch's epoch size for runtime/LR comparability.
        n_samples_per_epoch = 1109545
        pretrained_dataset = WeightedSampleMix(
            sources=[alljoined_ds, cinebrain_iter],
            weights=[params.mix_alljoined_weight,
                     params.mix_cinebrain_weight],
            samples_per_epoch=n_samples_per_epoch,
        )

        num_workers = 8 if os.environ.get('DEBUG', '0') == '0' else 0
        # max_horizon>0 requires the future-aware collate so CineBrain
        # samples' future stacks survive collation and the world-model
        # wrapper can route the prediction loss through cinebrain_idx.
        collate_fn = (
            collate_cached_with_future if params.max_horizon > 0
            else collate_cached
        )
        data_loader = DataLoader(
            pretrained_dataset,
            batch_size=params.batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    elif params.dataset_dir == 'mix+cinebrain+egobrain':
        from datasets.cached_dataset import (
            get_webdataset, collate_cached, collate_cached_with_future,
            set_active_vision_encoder,
        )
        from datasets.cinebrain_dataset import (
            CineBrainDataset, CineBrainIterableWrapper, WeightedSampleMix,
        )
        from datasets.egobrain_dataset import (
            EgoBrainDataset, EgoBrainIterableWrapper,
        )
        set_active_vision_encoder(params.vision_encoder)

        target_event_len = params.seq_len * params.in_dim
        alljoined_ds = get_webdataset(
            ["Alljoined-1.6M/*.tar"],
            params,
            event_window_target_len=target_event_len,
            ea_matrices=ea_alljoined,
        )

        # Both video-providing sources need max_horizon+1 windows when the
        # latent predictor is on; one window otherwise. CineBrain (64ch)
        # and EgoBrain (32ch) future stacks share a single batch via
        # collate_cached_with_future, which pads EgoBrain's channels up
        # to the batch max (64) before stacking.
        n_windows_fut = max(1, params.max_horizon + 1)

        cb_subjects = [s.strip() for s in params.cinebrain_subjects.split(',')
                       if s.strip()]
        cinebrain_inner = CineBrainDataset(
            data_dir=params.cinebrain_root,
            subjects=cb_subjects,
            in_dim=params.in_dim,
            n_windows=n_windows_fut,
            window_s=params.cinebrain_window_s,
            stride_s=params.cinebrain_stride_s,
            erp_latency_s=params.cinebrain_erp_latency_s,
            load_frames=bool(params.cinebrain_load_frames),
            vision_encoder=params.vision_encoder,
            ea_matrices=ea_cinebrain,
        )
        print('CineBrain clips:', len(cinebrain_inner))
        cinebrain_iter = CineBrainIterableWrapper(cinebrain_inner)

        if params.egobrain_subjects.lower() == 'all':
            import re as _re
            ego_subjects = sorted(
                d for d in os.listdir(params.egobrain_root)
                if _re.match(r'^P\d{4}$', d)
                and os.path.isdir(os.path.join(params.egobrain_root, d)))
        else:
            ego_subjects = [s.strip() for s in params.egobrain_subjects.split(',')
                            if s.strip()]
        egobrain_inner = EgoBrainDataset(
            data_dir=params.egobrain_root,
            subjects=ego_subjects,
            in_dim=params.in_dim,
            n_windows=n_windows_fut,
            window_s=params.egobrain_window_s,
            stride_s=params.egobrain_stride_s,
            clip_s=params.egobrain_clip_s,
            erp_latency_s=params.egobrain_erp_latency_s,
            load_frames=bool(params.egobrain_load_frames),
            vision_encoder=params.vision_encoder,
            max_channels=params.egobrain_max_channels,
            hand_labels_dir=params.egobrain_hand_labels_dir,
            hand_grid_dir=params.egobrain_hand_grid_dir,
            use_embeddings=params.use_cached_embeddings,
            emb_cache_dir=params.egobrain_emb_cache_dir,
            use_frame_grid=params.egobrain_use_frame_grid,
            frame_grid_dir=params.egobrain_frame_grid_dir,
            frame_grid_s=params.egobrain_frame_grid_s,
            temporal_jitter=not params.egobrain_no_temporal_jitter,
            use_grid_embeddings=params.egobrain_use_grid_embeddings,
            emb_grid_dir=params.egobrain_emb_grid_dir,
            video_subjects=_resolve_video_subjects(params.egobrain_video_subjects),
            ea_matrices=ea_egobrain,
            delta_whiten_g0=params.egobrain_delta_whiten_g0,
            delta_whiten_cutoff_hz=params.egobrain_delta_whiten_cutoff_hz,
            motion_resample=params.egobrain_motion_resample,
            motion_resample_alpha=params.egobrain_motion_resample_alpha,
            motion_resample_space=params.egobrain_motion_resample_space,
            motion_resample_metric=params.egobrain_motion_resample_metric,
            motion_resample_cap_pct=params.egobrain_motion_resample_cap_pct,
            motion_resample_floor_mix=params.egobrain_motion_resample_floor_mix,
            motion_emb_dir=params.egobrain_motion_emb_dir,
        )
        print('EgoBrain clips:', len(egobrain_inner))
        egobrain_iter = EgoBrainIterableWrapper(egobrain_inner)

        n_samples_per_epoch = 1109545
        pretrained_dataset = WeightedSampleMix(
            sources=[alljoined_ds, cinebrain_iter, egobrain_iter],
            weights=[params.mix_alljoined_weight,
                     params.mix_cinebrain_weight,
                     params.mix_egobrain_weight],
            samples_per_epoch=n_samples_per_epoch,
        )

        num_workers = 8 if os.environ.get('DEBUG', '0') == '0' else 0
        collate_fn = (
            collate_cached_with_future if params.max_horizon > 0
            else collate_cached
        )
        data_loader = DataLoader(
            pretrained_dataset,
            batch_size=params.batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    elif params.dataset_dir == 'mix+egobrain':
        from datasets.cached_dataset import (
            get_webdataset, collate_cached, collate_cached_with_future,
            set_active_vision_encoder,
        )
        from datasets.cinebrain_dataset import WeightedSampleMix
        from datasets.egobrain_dataset import (
            EgoBrainDataset, EgoBrainIterableWrapper,
        )
        set_active_vision_encoder(params.vision_encoder)

        target_event_len = params.seq_len * params.in_dim
        alljoined_ds = get_webdataset(
            ["Alljoined-1.6M/*.tar"],
            params,
            event_window_target_len=target_event_len,
            ea_matrices=ea_alljoined,
        )

        n_windows_ego = max(1, params.max_horizon + 1)
        if params.egobrain_subjects.lower() == 'all':
            import re as _re
            ego_subjects = sorted(
                d for d in os.listdir(params.egobrain_root)
                if _re.match(r'^P\d{4}$', d)
                and os.path.isdir(os.path.join(params.egobrain_root, d)))
        else:
            ego_subjects = [s.strip() for s in params.egobrain_subjects.split(',')
                            if s.strip()]
        egobrain_inner = EgoBrainDataset(
            data_dir=params.egobrain_root,
            subjects=ego_subjects,
            in_dim=params.in_dim,
            n_windows=n_windows_ego,
            window_s=params.egobrain_window_s,
            stride_s=params.egobrain_stride_s,
            clip_s=params.egobrain_clip_s,
            erp_latency_s=params.egobrain_erp_latency_s,
            load_frames=bool(params.egobrain_load_frames),
            vision_encoder=params.vision_encoder,
            max_channels=params.egobrain_max_channels,
            hand_labels_dir=params.egobrain_hand_labels_dir,
            hand_grid_dir=params.egobrain_hand_grid_dir,
            use_embeddings=params.use_cached_embeddings,
            emb_cache_dir=params.egobrain_emb_cache_dir,
            use_frame_grid=params.egobrain_use_frame_grid,
            frame_grid_dir=params.egobrain_frame_grid_dir,
            frame_grid_s=params.egobrain_frame_grid_s,
            temporal_jitter=not params.egobrain_no_temporal_jitter,
            use_grid_embeddings=params.egobrain_use_grid_embeddings,
            emb_grid_dir=params.egobrain_emb_grid_dir,
            video_subjects=_resolve_video_subjects(params.egobrain_video_subjects),
            ea_matrices=ea_egobrain,
            delta_whiten_g0=params.egobrain_delta_whiten_g0,
            delta_whiten_cutoff_hz=params.egobrain_delta_whiten_cutoff_hz,
            motion_resample=params.egobrain_motion_resample,
            motion_resample_alpha=params.egobrain_motion_resample_alpha,
            motion_resample_space=params.egobrain_motion_resample_space,
            motion_resample_metric=params.egobrain_motion_resample_metric,
            motion_resample_cap_pct=params.egobrain_motion_resample_cap_pct,
            motion_resample_floor_mix=params.egobrain_motion_resample_floor_mix,
            motion_emb_dir=params.egobrain_motion_emb_dir,
        )
        print('EgoBrain clips:', len(egobrain_inner))
        egobrain_iter = EgoBrainIterableWrapper(egobrain_inner)

        n_samples_per_epoch = 1109545
        pretrained_dataset = WeightedSampleMix(
            sources=[alljoined_ds, egobrain_iter],
            weights=[params.mix_alljoined_weight,
                     params.mix_egobrain_weight],
            samples_per_epoch=n_samples_per_epoch,
        )

        num_workers = 8 if os.environ.get('DEBUG', '0') == '0' else 0
        collate_fn = (
            collate_cached_with_future if params.max_horizon > 0
            else collate_cached
        )
        data_loader = DataLoader(
            pretrained_dataset,
            batch_size=params.batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    elif params.dataset_dir == 'egobrain':
        from datasets.egobrain_dataset import EgoBrainDataset, collate_egobrain
        if params.egobrain_subjects.lower() == 'all':
            import re as _re
            ego_subjects = sorted(
                d for d in os.listdir(params.egobrain_root)
                if _re.match(r'^P\d{4}$', d)
                and os.path.isdir(os.path.join(params.egobrain_root, d)))
        else:
            ego_subjects = [s.strip() for s in params.egobrain_subjects.split(',')
                            if s.strip()]
        pretrained_dataset = EgoBrainDataset(
            data_dir=params.egobrain_root,
            subjects=ego_subjects,
            in_dim=params.in_dim,
            n_windows=params.egobrain_n_windows,
            window_s=params.egobrain_window_s,
            stride_s=params.egobrain_stride_s,
            clip_s=params.egobrain_clip_s,
            erp_latency_s=params.egobrain_erp_latency_s,
            load_frames=bool(params.egobrain_load_frames),
            vision_encoder=params.vision_encoder,
            max_channels=params.egobrain_max_channels,
            hand_labels_dir=params.egobrain_hand_labels_dir,
            hand_grid_dir=params.egobrain_hand_grid_dir,
            use_embeddings=params.use_cached_embeddings,
            emb_cache_dir=params.egobrain_emb_cache_dir,
            use_frame_grid=params.egobrain_use_frame_grid,
            frame_grid_dir=params.egobrain_frame_grid_dir,
            frame_grid_s=params.egobrain_frame_grid_s,
            temporal_jitter=not params.egobrain_no_temporal_jitter,
            local_jitter=params.egobrain_block_window_s > 0,
            use_grid_embeddings=params.egobrain_use_grid_embeddings,
            emb_grid_dir=params.egobrain_emb_grid_dir,
            video_subjects=_resolve_video_subjects(params.egobrain_video_subjects),
            ea_matrices=ea_egobrain,
            delta_whiten_g0=params.egobrain_delta_whiten_g0,
            delta_whiten_cutoff_hz=params.egobrain_delta_whiten_cutoff_hz,
            motion_resample=params.egobrain_motion_resample,
            motion_resample_alpha=params.egobrain_motion_resample_alpha,
            motion_resample_space=params.egobrain_motion_resample_space,
            motion_resample_metric=params.egobrain_motion_resample_metric,
            motion_resample_cap_pct=params.egobrain_motion_resample_cap_pct,
            motion_resample_floor_mix=params.egobrain_motion_resample_floor_mix,
            motion_emb_dir=params.egobrain_motion_emb_dir,
        )
        print('EgoBrain clips:', len(pretrained_dataset))
        num_workers = 10 if os.environ.get('DEBUG', '0') == '0' else 0
        n_samples_per_epoch = 1109545
        collate_fn = functools.partial(
            collate_egobrain,
            frame_objective=(getattr(params, 'model', None) == 'WorldModel'
                             and getattr(params, 'wm_objective', 'eeg') == 'frame'),
            subject_block=params.egobrain_subject_block,
        )
        if params.egobrain_subject_block > 0:
            # Subject-block loading: each batch = batch_size//N blocks of N
            # same-subject clips. batch_sampler is mutually exclusive with
            # batch_size / sampler / drop_last, so pass it alone. num_batches
            # matches the RandomSampler epoch length it replaces.
            from datasets.egobrain_dataset import SubjectBlockBatchSampler
            # Temporal-proximity window (hard negatives): confine each block to a
            # contiguous run of ~block_window_s worth of clips, motion-biased so
            # windows still land on dynamic regions (reuses the per-clip motion
            # weights the dataset computed for motion_resample).
            window_clips = (
                max(1, int(math.ceil(params.egobrain_block_window_s
                                     / params.egobrain_clip_s)))
                if params.egobrain_block_window_s > 0 else 0)
            batch_sampler = SubjectBlockBatchSampler(
                pretrained_dataset._items,
                batch_size=params.batch_size,
                block_size=params.egobrain_subject_block,
                num_batches=n_samples_per_epoch // params.batch_size,
                seed=params.seed,
                window_clips=window_clips,
                clip_weights=(pretrained_dataset._clip_motion_w
                              if window_clips > 0 else None),
            )
            _win_msg = (
                f'window_clips={window_clips} '
                f'(~{window_clips * params.egobrain_clip_s:.0f}s proximity window)'
                if window_clips > 0 else 'whole-recording blocks')
            print(f'[EgoBrain] subject-block loading ON: '
                  f'block={params.egobrain_subject_block} '
                  f'blocks/batch={params.batch_size // params.egobrain_subject_block} '
                  f'excl_samples={params.wm_frame_contrast_excl_samples} {_win_msg}')
            data_loader = DataLoader(
                pretrained_dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=4,
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                pretrained_dataset,
                replacement=True,
                num_samples=n_samples_per_epoch,
            )
            data_loader = DataLoader(
                pretrained_dataset,
                batch_size=params.batch_size,
                num_workers=num_workers,
                sampler=sampler,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=4,
            )

    elif params.dataset_dir == 'cinebrain':
        from datasets.cinebrain_dataset import CineBrainDataset, collate_cinebrain
        subjects = [s.strip() for s in params.cinebrain_subjects.split(',') if s.strip()]
        pretrained_dataset = CineBrainDataset(
            data_dir=params.cinebrain_root,
            subjects=subjects,
            in_dim=params.in_dim,
            n_windows=params.cinebrain_n_windows,
            window_s=params.cinebrain_window_s,
            stride_s=params.cinebrain_stride_s,
            erp_latency_s=params.cinebrain_erp_latency_s,
            load_frames=bool(params.cinebrain_load_frames),
            vision_encoder=params.vision_encoder,
            ea_matrices=ea_cinebrain,
        )
        print('CineBrain clips:', len(pretrained_dataset))
        num_workers = 8 if os.environ.get('DEBUG', '0') == '0' else 0
        # Match the 'mix' branch's epoch size so runtime + LR schedules stay
        # comparable. CineBrain has far fewer unique clips, so we sample with
        # replacement — each clip is visited ~n_samples_per_epoch/len(ds)
        # times per epoch, which is fine since windows/frames are
        # deterministic per clip.
        n_samples_per_epoch = 1109545
        sampler = torch.utils.data.RandomSampler(
            pretrained_dataset,
            replacement=True,
            num_samples=n_samples_per_epoch,
        )
        data_loader = DataLoader(
            pretrained_dataset,
            batch_size=params.batch_size,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_cinebrain,
            pin_memory=True,
            drop_last=True,
        )
    else:
        pretrained_dataset = PretrainingDataset(
            dataset_dir=params.dataset_dir,
            SmallerToken=params.use_SmallerToken,
        )
        print(len(pretrained_dataset))
        data_loader = DataLoader(
            pretrained_dataset,
            batch_size=params.batch_size,
            num_workers=8,
            shuffle=True,
        )

    brain_regions = [
        0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1
    ]
    electrode_labels = [
        "FP1-REF", "FP2-REF", "F3-REF", "F4-REF",
        "C3-REF", "C4-REF", "P3-REF", "P4-REF",
        "O1-REF", "O2-REF", "F7-REF", "F8-REF",
        "T3-REF", "T4-REF", "T5-REF", "T6-REF",
        "FZ-REF", "CZ-REF", "PZ-REF"
    ]
    topology = {
        0: ["FP1-REF", "F7-REF", "F3-REF", "FZ-REF", "F4-REF", "F8-REF", "FP2-REF"], 
        4: ["C3-REF", "CZ-REF", "C4-REF"],
        1: ["P3-REF", "PZ-REF", "P4-REF"],
        3: ["O1-REF", "O2-REF"],
        2: ["T3-REF", "T5-REF", "T6-REF", "T4-REF"],
        -1: ["A1-REF"]
    }
    region_groups = {}
    for i, region in enumerate(brain_regions):
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append((i, electrode_labels[i]))
    sorted_indices = []
    for region in sorted(region_groups.keys()):
        region_electrodes = region_groups[region]
        sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
        sorted_indices.extend([e[0] for e in sorted_electrodes])

    print("Sorted Indices:", sorted_indices)

    model = get_model(params, brain_regions, sorted_indices)
  
    wandb.init(project="EEG", name=params.run_name, config=vars(params))
    wandb.watch(model, log="all", log_freq=500)
    trainer = Trainer(params, data_loader, model)
    trainer.train()

    if hasattr(pretrained_dataset, 'db'):
        pretrained_dataset.db.close()


if __name__ == '__main__':
    main()
