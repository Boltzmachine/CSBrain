import math
import os
import random
import signal

import numpy as np
import torch
import torch.nn.functional as F
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


def apply_delta_whiten(x, fs, g0, cutoff_hz, eps=1e-8):
    """Per-channel low-frequency (delta) attenuation toward a motor-imagery
    spectral profile — the ME->MI "rank-1 delta whitening".

    Each channel's rFFT is multiplied by a real, zero-phase raised-cosine gain
    that ramps from ``g0`` (<1) at DC up to ``1.0`` at ``cutoff_hz``; bins at or
    above ``cutoff_hz`` are untouched. This flattens the execution-heavy delta
    floor of motor-EXECUTION EEG (EgoBrain) toward the shallower delta level of
    motor-IMAGERY (PhysioNet-MI). Per-channel RMS is preserved after filtering
    so the downstream ``/100`` magnitude convention is unchanged — only the
    relative spectral SHAPE moves. ``g0 >= 1`` (or ``cutoff_hz <= 0``) is a
    no-op, so the default leaves the signal byte-for-byte identical.

    NOTE on the neuroscience/limits: the delta surplus this attenuates is a
    mix of execution-bound MRCP/reafferent slow potentials that imagery lacks
    AND motion/EOG artifact; the filter is non-selective and removes both. It
    is a calibrated domain-shift + denoise, not a clean reproduction of the MI
    neural state. See project_me_mi_pretrain_manipulation.

    Parameters
    ----------
    x : ndarray ``(C, T)`` — channels x time (uV).
    fs : float — sampling rate in Hz.
    g0 : float — DC gain of the ramp. Calibrate so the corpus' montage-avg
        relative delta lands on the *finetune target* (preprocessed PhysioNet-MI
        LMDB, ~46 montage / ~42 central, percent), NOT the raw-EDF literature
        value — too strong a gain overshoots past MI and grows the distance.
    cutoff_hz : float — upper edge of the ramp (bins >= it are left alone).

    Returns
    -------
    ndarray ``(C, T)`` — same shape and dtype as ``x``.
    """
    if g0 is None or g0 >= 1.0 or cutoff_hz is None or cutoff_hz <= 0:
        return x
    x = np.asarray(x)
    in_dtype = x.dtype
    xf = x.astype(np.float64, copy=False)
    T = xf.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / float(fs))            # (T//2+1,)
    gain = np.ones_like(freqs)
    ramp = freqs < cutoff_hz
    gain[ramp] = g0 + (1.0 - g0) * 0.5 * (
        1.0 - np.cos(np.pi * freqs[ramp] / cutoff_hz))       # g0 -> 1.0
    out = np.fft.irfft(np.fft.rfft(xf, axis=-1) * gain, n=T, axis=-1)
    # Preserve per-channel RMS: reshape the spectrum, not the amplitude.
    pre = np.sqrt((xf ** 2).mean(axis=-1, keepdims=True))
    post = np.sqrt((out ** 2).mean(axis=-1, keepdims=True))
    out = out * (pre / (post + eps))
    return out.astype(in_dtype, copy=False)


def band_split_recon_loss(pred, target, fs, phase_cutoff_hz, power_weight=1.0):
    """Low-frequency-only masked-patch reconstruction loss (raw signal space).

    Reconstructs only the LOW-frequency part of the waveform: both ``pred`` and
    ``target`` are conceptually low-pass filtered (rFFT bins with
    ``f >= phase_cutoff_hz`` dropped) and compared by time-domain MSE. The
    slow / evoked low-freq content (delta), whose phase is informative, is
    reconstructed in full (phase included); the induced high-freq content
    (mu/beta), whose phase is unpredictable, is simply not penalised — so we
    stop injecting noise gradients there instead of trying to nail it.

    Scale: computed as the mean squared error PER RETAINED real degree of
    freedom (DC and Nyquist carry 1 DOF, interior bins 2). This keeps the
    magnitude comparable to the plain MSE for any cutoff, and makes the loss
    EXACTLY the plain time-domain MSE when all bins are retained
    (``phase_cutoff_hz >= Nyquist``). Evaluated in the frequency domain via
    Parseval — numerically equivalent to time-domain MSE of the low-passed
    signals, but exact and without an inverse FFT.

    ``power_weight`` is accepted for arg/call back-compat but UNUSED here.

    No-op guarantee: returns the plain time-domain MSE bit-for-bit when
    ``phase_cutoff_hz`` is ``>= Nyquist`` or non-finite (all bins retained); the
    caller also skips this function entirely when ``--recon_band_split`` is off.
    ``phase_cutoff_hz <= 0`` retains nothing → zero loss (reconstruction off).

    Args:
        pred, target: (..., P) — per-masked-patch waveforms, ``P = in_dim``
            samples. Typically ``y[mask == 1]`` / ``x[mask == 1]``, i.e. (M, P).
        fs: patch sampling rate in Hz (rFFT bin width = ``fs / P``).
        phase_cutoff_hz: reconstruct frequencies strictly below this; a large
            value (``>= Nyquist``) or non-finite reproduces the plain MSE.
        power_weight: ignored (kept for back-compat).
    """
    P = pred.size(-1)
    F_ = P // 2 + 1
    bin_hz = fs / P
    nyquist = (F_ - 1) * bin_hz

    # All bins retained -> plain time-domain MSE, bit-identical to baseline.
    if not math.isfinite(float(phase_cutoff_hz)) or float(phase_cutoff_hz) > nyquist:
        return ((pred - target) ** 2).mean()

    # Per-bin real DOF == Parseval weight for an ortho rFFT of a length-P real
    # signal: DC (and Nyquist when P is even) carry 1, interior bins carry 2.
    freqs = torch.arange(F_, device=pred.device) * bin_hz
    pw = torch.full((F_,), 2.0, device=pred.device, dtype=pred.dtype)
    pw[0] = 1.0
    if P % 2 == 0:
        pw[F_ - 1] = 1.0
    pw = pw * (freqs < float(phase_cutoff_hz)).to(pw.dtype)   # keep low-freq bins

    dof = pw.sum()
    if float(dof) == 0.0:
        return (pred - target).pow(2).mean() * 0.0           # nothing retained

    D = torch.fft.rfft(pred - target, dim=-1, norm="ortho")  # (..., F)
    energy = D.real.pow(2) + D.imag.pow(2)                    # |D_k|^2
    return (energy * pw).sum(dim=-1).mean() / dof            # MSE per retained DOF


def _analytic_bandlimited(sig, fs, lo, hi):
    """Band-limited analytic (Hilbert) signal of a real tensor via FFT.

    Equivalent to ``scipy.signal.hilbert`` applied to the band-passed signal:
    full FFT, keep only the *positive* in-band frequencies (doubled), zero the
    DC/negative/out-of-band bins, inverse FFT to a complex analytic signal whose
    modulus is the instantaneous amplitude envelope and whose argument is the
    instantaneous phase. Differentiable w.r.t. ``sig``.

    Args:
        sig: (..., T) real tensor.
        fs: sampling rate in Hz.
        lo, hi: band edges in Hz; bins with ``lo <= f < hi`` are kept.

    Returns:
        Complex tensor of shape (..., T).
    """
    T = sig.shape[-1]
    Xf = torch.fft.fft(sig, dim=-1)
    freqs = torch.fft.fftfreq(T, d=1.0 / float(fs)).to(sig.device)
    nyq = float(fs) / 2.0
    H = torch.zeros(T, device=sig.device, dtype=sig.dtype)
    in_band_pos = (
        (freqs > 0) & (freqs < nyq) & (freqs >= float(lo)) & (freqs < float(hi))
    )
    H[in_band_pos] = 2.0                                     # positive freqs doubled
    if float(lo) <= 0.0 < float(hi):
        H[0] = 1.0                                          # DC (no doubling)
    if T % 2 == 0 and float(lo) <= nyq < float(hi):
        H[T // 2] = 1.0                                     # Nyquist (no doubling)
    return torch.fft.ifft(Xf * H.to(Xf.dtype), dim=-1)


def band_phase_envelope_loss(pred, target, mask, fs,
                             delta_band=(0.5, 4.0),
                             power_bands=((8.0, 13.0), (13.0, 30.0)),
                             complete=True, eps=1e-8):
    """Auxiliary masked-recon objectives layered ON TOP of the plain MSE.

    Adds two band-specific instantaneous targets, motivated by what each band
    can actually predict from masked context:

    * **delta (slow / evoked) -> instantaneous PHASE.** Low-frequency phase is
      informative, so we align the analytic-signal phase of the reconstruction
      with the target's. The per-sample term is ``1 - cos(Δφ)`` weighted by the
      *target* delta amplitude, so phase-meaningless silent moments don't
      dominate.
    * **mu / beta (induced) -> instantaneous POWER ENVELOPE** ``|analytic|^2``.
      Their phase is largely unpredictable, but the band-power time course
      (ERD/ERS) is — so we score the squared-magnitude envelope in the
      **log-power (dB-style) domain**, how band power / ERD is conventionally
      compared. ``log`` is scale-free (a ratio), robust to the ``/100``
      amplitude convention, and — unlike a relative MSE on raw ``|z|^2`` —
      bounded under large recon error (raw power MSE grows ~quartically in the
      noise and explodes); a relative floor at 1% of the band's mean target
      power keeps silent samples from blowing up the log.

    Both terms are restricted to the masked patches. With ``complete=True`` the
    Hilbert transform is taken on the model-completed waveform (true signal at
    visible patches, predictions at masked ones) so it sees a realistic
    continuous signal and gradients only flow through the masked region.

    Args:
        pred, target: (B, C, N, d) full reconstruction / ground-truth patches.
        mask: (B, C, N) with ``1`` on masked patches (loss support); ``None``
            applies the loss everywhere.
        fs: sampling rate in Hz.
        delta_band: (lo, hi) Hz for the phase term.
        power_bands: iterable of (lo, hi) Hz bands for the power-envelope term
            (averaged across bands).
        complete: take the Hilbert transform on the completed waveform.

    Returns:
        ``(phase_loss, envelope_loss)`` scalar tensors.
    """
    B, C, N, d = pred.shape
    T = N * d

    if mask is not None:
        m = mask.to(pred.dtype).unsqueeze(-1)                # (B, C, N, 1)
        if complete:
            pred = target * (1.0 - m) + pred * m             # fill masked patches
        loss_mask = m.expand(B, C, N, d).reshape(B, C, T)
    else:
        loss_mask = torch.ones(B, C, T, device=pred.device, dtype=pred.dtype)

    yp = pred.reshape(B, C, T)
    yt = target.reshape(B, C, T)

    # --- delta: instantaneous phase agreement (amplitude-weighted) ---
    zp = _analytic_bandlimited(yp, fs, delta_band[0], delta_band[1])
    zt = _analytic_bandlimited(yt, fs, delta_band[0], delta_band[1])
    ap, at = zp.abs(), zt.abs()
    cos_dphi = (zp.real * zt.real + zp.imag * zt.imag) / (ap * at + eps)
    w = at * loss_mask                                       # weight by target amp
    phase_loss = ((1.0 - cos_dphi) * w).sum() / w.sum().clamp(min=eps)

    # --- mu / beta: instantaneous power envelope (scale-free log-power MSE) ---
    denom = loss_mask.sum().clamp(min=eps)
    env_terms = []
    for (lo, hi) in power_bands:
        zpb = _analytic_bandlimited(yp, fs, lo, hi)
        ztb = _analytic_bandlimited(yt, fs, lo, hi)
        pp = zpb.real.pow(2) + zpb.imag.pow(2)               # |z|^2 power envelope
        pt = ztb.real.pow(2) + ztb.imag.pow(2)
        # Relative floor at 1% of the band's mean target power -> scale-free,
        # and keeps near-silent samples from exploding the log.
        floor = 0.01 * (pt * loss_mask).sum() / denom + eps
        lp = torch.log((pp + floor) / (pt + floor))          # dB-style log ratio
        env_terms.append((lp.pow(2) * loss_mask).sum() / denom)
    envelope_loss = torch.stack(env_terms).mean()

    return phase_loss, envelope_loss


def hand_regression_loss(pred, target, valid, flip=False):
    """Masked SmoothL1 regression of per-hand movement intensity.

    The continuous EgoBrain hand annotations (left/right intensity in
    hand-lengths/s, see datasets/egobrain_hand_labels.py) decoded from the EEG
    window representation as an auxiliary objective.

    ``pred``/``target``/``valid`` are all ``(B, D)`` with column 0 = left hand,
    column 1 = right hand (D is typically 2). ``valid`` is a PER-COLUMN mask:
    an undetected hand has a NaN intensity (already replaced by 0 in ``target``)
    AND ``valid=False`` for that column, so the loss never sees the NaN and never
    penalises a hand we did not observe. ``flip=True`` marks a frame-averaging
    flip step (the scene is horizontally mirrored, so left<->right swap): we swap
    columns 0,1 of BOTH target and valid so the rep that saw the mirrored scene is
    scored against the mirrored labels.

    Returns ``(loss, mae)`` — scalar tensors; both are 0 when no column is valid
    (so steps with no EgoBrain rows contribute nothing).
    """
    if flip and target.size(-1) >= 2:
        idx = list(range(target.size(-1)))
        idx[0], idx[1] = 1, 0
        target = target[:, idx]
        valid = valid[:, idx]
    valid = valid.to(pred.dtype)
    err = F.smooth_l1_loss(pred, target, reduction='none')      # (B, D)
    denom = valid.sum().clamp(min=1.0)
    loss = (err * valid).sum() / denom
    with torch.no_grad():
        mae = ((pred - target).abs() * valid).sum() / denom
    return loss, mae


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
    'use_volume_conduction', 'vc_tau_init',
    # Learnable spectral-band backbone: these change the module set / weight
    # shapes (filterbank, band-type embedding, cross-band attention, per-band
    # alignment heads, band->level logits), so finetuning must rebuild the
    # backbone with the *same* config the checkpoint was trained under.
    'use_spectral_bands', 'num_spectral_bands', 'fs',
    'filterbank_kernel_size', 'filterbank_min_bw_hz',
    'use_cross_band_attn', 'cross_band_every', 'use_band_type_embedding',
    'num_visual_levels',
    # Equivariant frame-averaging frontend (plans/eeg-wm.md). These change the
    # encoder's FORWARD (channel-independent patch embed via drop_pos_conv +
    # frame averaging over {I,P}) and add the frame_split / frame_flip_align_proj
    # modules, so finetuning MUST rebuild the backbone with frame_averaging=True
    # — otherwise the pretrained weights run in a different architecture
    # (untrained conv pos-enc injected + no frame averaging). flip_split_hidden /
    # flip_n_col_bands set those modules' weight shapes; flip_align_weight is
    # carried for config consistency.
    'frame_averaging', 'frame_avg_flip_prob', 'frame_avg_recon_weight',
    'flip_split_hidden', 'flip_n_col_bands', 'flip_align_weight',
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
    strips wrapper prefixes. ``model`` is mapped WorldModel / ActionWorldModel
    -> Align for the same reason (the EEG backbone is finetuned alone; the
    predictor / action head / V-JEPA 2 world encoder are discarded).
    """
    if saved_params is None:
        return
    for field in ARCH_PARAM_FIELDS:
        if field in saved_params:
            setattr(params, field, saved_params[field])
    saved_model = saved_params.get('model')
    if saved_model in ('WorldModel', 'ActionWorldModel'):
        params.model = 'Align'
    elif saved_model is not None:
        params.model = saved_model


if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)