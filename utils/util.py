import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import random

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask


def apply_freq_band_mask(x, n_bands, mask_ratio=None):
    """rFFT the per-channel signal, zero ``k`` randomly chosen bands per
    sample, iFFT back. Returns ``(masked_x, band_mask)``:

    * ``masked_x``: (B, C, N, d) band-removed input.
    * ``band_mask``: (B, n_freq) float — ``1`` on the rFFT bins that were
      zeroed, ``0`` elsewhere — for restricting the recon loss to those bins.

    ``k = max(1, round(n_bands * mask_ratio))`` when ``mask_ratio`` is given;
    when ``mask_ratio is None`` (default), ``k=1`` (a single random band).

    x: (B, C, N, d) — per-patch timeseries; the band split is over the full
    per-channel length ``N*d``.
    """
    B, C, N, d = x.shape
    T = N * d
    ts = x.reshape(B, C, T)

    spec = torch.fft.rfft(ts, dim=-1)  # (B, C, n_freq)
    n_freq = spec.size(-1)

    edges = torch.linspace(0, n_freq, n_bands + 1, device=x.device).long()

    if mask_ratio is None:
        k = 1
    else:
        k = max(1, min(n_bands, int(round(n_bands * float(mask_ratio)))))

    # Sample k distinct band indices per row uniformly without replacement.
    picks = torch.rand(B, n_bands, device=x.device).topk(
        k, dim=-1, largest=False).indices  # (B, k)
    band_select = torch.zeros(
        B, n_bands, device=x.device, dtype=spec.real.dtype)
    band_select.scatter_(1, picks, 1.0)

    bin_idx = torch.arange(n_freq, device=x.device).unsqueeze(0)  # (1, n_freq)
    edge_lo = edges[:-1].unsqueeze(1)  # (n_bands, 1)
    edge_hi = edges[1:].unsqueeze(1)   # (n_bands, 1)
    band_of_bin = (
        (bin_idx >= edge_lo) & (bin_idx < edge_hi)
    ).to(spec.real.dtype)  # (n_bands, n_freq)
    band_mask = band_select @ band_of_bin  # (B, n_freq)

    masked_spec = spec * (1.0 - band_mask).unsqueeze(1)
    masked_ts = torch.fft.irfft(masked_spec, n=T, dim=-1)
    return masked_ts.reshape(B, C, N, d), band_mask


def bandpass_decompose(x, fs, cutoffs):
    """Split ``x`` into ``len(cutoffs) + 1`` frequency-band time-domain views.

    The split is done over the full per-channel signal ``N*d`` so band
    boundaries are crisp regardless of patch boundaries: rFFT, zero bins
    outside each band, irFFT.

    Args:
        x: (B, C, N, d) per-patch timeseries.
        fs: sampling rate in Hz.
        cutoffs: monotonically increasing iterable of K-1 cut frequencies in
            Hz, e.g. ``(1.0, 10.0)`` → 3 bands ``[0, 1) Hz``, ``[1, 10) Hz``,
            ``[10, +inf) Hz``.

    Returns:
        List of K tensors, each of shape (B, C, N, d).
    """
    B, C, N, d = x.shape
    T = N * d
    ts = x.reshape(B, C, T)
    spec = torch.fft.rfft(ts, dim=-1)  # (B, C, T//2+1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fs)).to(x.device)  # (T//2+1,)

    edges = [0.0] + [float(c) for c in cutoffs] + [float('inf')]
    views = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        bin_mask = ((freqs >= lo) & (freqs < hi)).to(spec.real.dtype)  # (n_freq,)
        spec_k = spec * bin_mask.view(1, 1, -1)
        ts_k = torch.fft.irfft(spec_k, n=T, dim=-1)
        views.append(ts_k.reshape(B, C, N, d))
    return views


def symmetric_band_infonce(z_per_band, temp, valid_channel_mask=None):
    """Per-position symmetric InfoNCE across K band views.

    For each (channel, patch) position, every sample's K band embeddings
    must agree (positives), while embeddings of other samples at the same
    position are negatives.

    Args:
        z_per_band: list of K tensors of shape (B, C, N, d_proj).
        temp: InfoNCE temperature.
        valid_channel_mask: optional (B, C) bool — False for padded
            channels. Padded (sample, channel) positions are excluded as
            both anchors and negatives.

    Returns:
        Scalar loss (mean over K*(K-1) ordered band pairs and over valid
        anchor positions).
    """
    K = len(z_per_band)
    if K < 2:
        return z_per_band[0].new_zeros(())
    B, C, N, d = z_per_band[0].shape
    P = C * N

    z = torch.stack(z_per_band, dim=0)               # (K, B, C, N, d)
    z = torch.nn.functional.normalize(z, dim=-1)
    z = z.permute(0, 2, 3, 1, 4).contiguous()        # (K, C, N, B, d)
    z = z.view(K, P, B, d)

    if valid_channel_mask is not None:
        ch_valid = valid_channel_mask.t().contiguous()       # (C, B)
        pos_valid = ch_valid.unsqueeze(1).expand(C, N, B)    # (C, N, B)
        pos_valid = pos_valid.reshape(P, B)                  # (P, B)
    else:
        pos_valid = torch.ones(P, B, dtype=torch.bool, device=z.device)

    NEG_INF = torch.finfo(z.dtype).min / 2
    targets = torch.arange(B, device=z.device).unsqueeze(0).expand(P, B)  # (P, B)

    loss = z.new_zeros(())
    pair_count = 0
    for k1 in range(K):
        for k2 in range(K):
            if k1 == k2:
                continue
            sim = torch.bmm(z[k1], z[k2].transpose(1, 2)) / temp     # (P, B, B)
            sim = sim.masked_fill(~pos_valid.unsqueeze(1), NEG_INF)  # mask cols
            log_probs = torch.nn.functional.log_softmax(sim, dim=-1)
            ce = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (P, B)
            ce = ce * pos_valid.float()
            denom = pos_valid.float().sum().clamp(min=1)
            loss = loss + ce.sum() / denom
            pair_count += 1
    return loss / max(pair_count, 1)


def to_tensor(array):
    return torch.from_numpy(array).float()


# Fields that control the encoder architecture and must match between
# pretraining and finetuning. Stored in the checkpoint so finetune scripts
# no longer need to re-specify them.
ARCH_PARAM_FIELDS = (
    'in_dim', 'out_dim', 'd_model', 'dim_feedforward',
    'seq_len', 'n_layer', 'nhead',
    'TemEmbed_kernel_sizes',
    'project_to_source', 'num_sources',
    'patch_embed_type', 'mamba_band_periods', 'n_mamba_layers',
    'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
    'use_llm_vq', 'num_language_tokens', 'max_llm_codebook_size',
    'spectral_mode', 'stft_n_fft', 'stft_hop',
    # Vision encoder determines image_feature_dim (768 for DINOv2-base,
    # 1024 for V-JEPA 2 ViT-L), which is baked into contrastive_proj and
    # llm_contrastive_proj weight shapes; mismatch -> checkpoint load fails.
    'vision_encoder', 'image_pool_heads',
    'use_volume_conduction', 'vc_tau_init'
)


def save_pretrain_checkpoint(path, model, params):
    torch.save({
        'state_dict': model.state_dict(),
        'params': vars(params),
    }, path)


def load_pretrain_checkpoint(path, map_location='cpu'):
    """Load a pretraining checkpoint, returning (state_dict, saved_params).

    Handles both the new format ({'state_dict', 'params'}) and legacy
    checkpoints that saved the bare state_dict; in the legacy case
    saved_params is None.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if (isinstance(obj, dict) and 'state_dict' in obj and 'params' in obj
            and isinstance(obj['params'], dict)):
        return obj['state_dict'], obj['params']
    return obj, None


def build_muon_optimizer(model, lr, weight_decay, muon_lr=None,
                         muon_momentum=0.95, muon_weight_decay=None,
                         extra_aux_keywords=()):
    """Wrap ``model``'s trainable params in a SingleDeviceMuonWithAuxAdam.

    2D hidden weights go to Muon; embeddings, classifier heads, prototypes,
    cls/mask tokens, and any 1D params (gains, biases, norms) go to the
    internal AdamW. Per the Muon paper, embedding and final-output layers
    must not be optimised by Muon.

    ``lr`` is the AdamW lr; ``muon_lr`` the Muon lr (defaults to 0.02 if
    unset, which matches the package default for hidden weights).
    """
    from muon import SingleDeviceMuonWithAuxAdam

    if muon_lr is None:
        muon_lr = 0.02
    if muon_weight_decay is None:
        muon_weight_decay = weight_decay

    # Per the Muon paper, lookup embeddings, the final output layer, and
    # all gains/biases must use AdamW. We match by both module type
    # (nn.Embedding) and by path component (head/classifier/prototype/
    # CLS/mask tokens etc).
    aux_components = {
        'head', 'classifier', 'last_layer', 'prototype', 'prototypes',
        'cls_token', 'mask_token', 'pos_emb', 'position_emb',
        'positional_emb', 'positional_encoding',
    } | set(extra_aux_keywords)

    import torch.nn as nn
    embedding_param_ids = set()
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            for p in mod.parameters(recurse=False):
                embedding_param_ids.add(id(p))

    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split('.')
        is_aux = (
            id(p) in embedding_param_ids
            or any(part in aux_components for part in parts)
        )
        if p.ndim >= 2 and not is_aux:
            muon_params.append(p)
        else:
            adam_params.append(p)

    param_groups = []
    if muon_params:
        param_groups.append(dict(
            params=muon_params, use_muon=True,
            lr=muon_lr, momentum=muon_momentum,
            weight_decay=muon_weight_decay,
        ))
    if adam_params:
        param_groups.append(dict(
            params=adam_params, use_muon=False,
            lr=lr, betas=(0.9, 0.95), eps=1e-10,
            weight_decay=weight_decay,
        ))
    def _safe_numel(ps):
        # Lazy modules may hold UninitializedParameters at this point;
        # .numel() raises on those, so just count tensors instead.
        try:
            return sum(p.numel() for p in ps)
        except ValueError:
            return f"{len(ps)} tensors"
    print(f"[Muon] {_safe_numel(muon_params)} params -> Muon "
          f"(lr={muon_lr}), {_safe_numel(adam_params)} -> AdamW "
          f"(lr={lr})")
    return SingleDeviceMuonWithAuxAdam(param_groups)


def apply_arch_params(params, saved_params):
    """Overlay encoder-architecture fields from a pretraining checkpoint.

    Wrapper flags (dino_mode, WorldModel) are intentionally not overlayed:
    finetuning always builds the plain encoder, and the state_dict loader
    strips wrapper prefixes. ``model`` is mapped WorldModel -> Align for
    the same reason.
    """
    if saved_params is None:
        return
    for field in ARCH_PARAM_FIELDS:
        if field in saved_params:
            setattr(params, field, saved_params[field])
    saved_model = saved_params.get('model')
    if saved_model == 'WorldModel':
        params.model = 'Align'
    elif saved_model is not None:
        params.model = saved_model


if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)