"""Interpretability probe: is the frame-averaging BILATERALIZATION SPLIT working?

The split (models/alignment.py, ``frame_averaging`` a.k.a. equi-2.0) learns a gate
``g`` that additively cuts the patch embedding into a *bilateral* and a *lateral*
stream::

    z_lat = g * z          z_bi = (1 - g) * z          z = z_bi + z_lat
    P(z)  = z_bi + swap_homologous_channels(z_lat)      # C3<->C4, midline -> self

the backbone runs on both ``z`` and ``P(z)``, and the token feature dim is split
half P-invariant (``inv = (a+b)/2``, "bilateral") / half P-anti-equivariant
(``eq = (a - Pb)/2``, "lateral").

This script asks whether that split is doing anything REAL, using only a
checkpoint (no finetuning, no accuracy claims). Stages:

  weights  data-free. ``gate_head`` is LINEAR and the coord PE enters ADDITIVELY,
           so gate = sigmoid( W.GELU(sig_proj(x)) + [W.coord_proj(pe_c) + b] ).
           The bracket is a pure per-channel, data-independent logit offset
           L_coord(c) -> a gate topography straight out of the weights. Tests its
           magnitude and its LEFT-RIGHT MIRROR SYMMETRY (a consistent P demands
           L_coord(c) == L_coord(homolog(c))), against a norm-matched random-W null.

  antisym  THE KILLER TEST. The split's core claim is that (1-g)*z is symmetric
           under the hemispheric swap and g*z is antisymmetric. With
           R(u) = ||u - Pu||^2 / (||u - Pu||^2 + ||u + Pu||^2), a working split
           needs R(z_lat) > R(z) > R(z_bi). A CONSTANT gate gives
           R(z_lat) == R(z_bi) == R(z) EXACTLY, so every bit of separation is
           learned -- it cannot be explained by the gate's size or by the
           representation being good. Nulls: channel-shuffled gate, feature-
           shuffled gate, random-init model.

  eqconst  Falsifies the "orientation bookkeeping bit" degeneracy. At the global
           token eq_global = +-0.5*(a_g - b_g); a degenerate model emits a
           CONSTANT vector c for every input, which still scores
           diag_flip_discrim_acc ~ 1 while carrying zero EEG-laterality content.
           Measures kappa = ||mean(e)||^2 / mean(||e||^2), the participation
           ratio of cov(e), and how much flip-discrimination survives after the
           mean direction is projected out.

  mi       PhysioNet-MI content, subject-grouped CV. Regularized Fisher (LDA)
           separation of the lateral vs bilateral half along the ANTISYMMETRIC
           axis (left fist vs right fist) and the SYMMETRIC axes (both-fists vs
           both-feet; single vs both). Control: a random orthogonal rotation of
           the full rep before halving -> the LAT/BI ratio must collapse to ~1.

  swap     Behavioural anti-symmetry. Anatomically mirror the RAW input (swap the
           data of homologous channels, keep the electrode labels) and re-encode.
           A genuine laterality code must make the lateral half's left/right
           read-out INVERT while the bilateral half's symmetric read-out is
           preserved.

Usage:
    python scripts/probe_bilateral_split.py --ckpt outputs/wm-new-gradneg/epoch10_*.pth \
        --stages weights,antisym,eqconst,mi,swap --out outputs/split_probe/gradneg_ep10.json
"""
import argparse
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.util import load_pretrain_checkpoint, apply_arch_params  # noqa: E402
from datasets import physio_dataset  # noqa: E402
from models import model_for_physio  # noqa: E402
from models.alignment import (  # noqa: E402
    build_flip_perm_batch, _get_homologous_name, _normalize_ch_name,
)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# checkpoint / model
# ---------------------------------------------------------------------------
def build_params(ckpt, seed=42):
    p = Namespace(
        model='Align', downstream_dataset='PhysioNet-MI',
        datasets_dir=os.path.join(REPO, 'data/preprocessed/physionet_mi'),
        num_of_classes=4, foundation_dir=ckpt, seed=seed,
        use_pretrained_weights=True, use_initial_segment_only=True,
        segment_index=0, highpass_hz=0.0, batch_size=128, fs=200,
        frame_rep_mode='both', temporal_jitter=0, bilateral_head=False,
        frame_flip_aug=False, frame_flip_tta=False, lateralization_flip=False,
        flip_aug=False, flip_tta=False, symmetrize_aug=False,
        use_euclidean_alignment=False, dropout=0.1, linear_probe=False,
        multi_lr=False,
    )
    _, saved = load_pretrain_checkpoint(ckpt)
    apply_arch_params(p, saved)
    return p, saved


def build_backbone(ckpt, random_init=False, seed=42):
    """Load the pretrain ckpt into a frame-averaging backbone (encoder only)."""
    p, saved = build_params(ckpt, seed=seed)
    if random_init:
        p.use_pretrained_weights = False
        torch.manual_seed(1234)
    model = model_for_physio.Model(p).eval()
    bb = model.backbone
    if hasattr(bb, 'pretrained_image_encoder'):
        del bb.pretrained_image_encoder      # frozen DINOv2, unused encoder_only
    bb = bb.to(DEV).eval()
    if not getattr(bb, 'frame_averaging', False):
        raise SystemExit(f"{ckpt}: backbone.frame_averaging is False -- this "
                         "checkpoint has no bilateralization split.")
    # Guard: the gate must NOT be at its init (gate_head.weight is init'd to
    # EXACT zeros and the bias to a constant -2.0). If it is, either the ckpt
    # did not load or the split never trained -- every number below would be
    # measuring a random gate.
    gh_w = bb.frame_split.gate_head.weight.detach()
    gh_b = bb.frame_split.gate_head.bias.detach()
    loaded = (gh_w.norm().item() > 1e-6) and (gh_b.std().item() > 1e-6)
    info = dict(gate_head_w_norm=float(gh_w.norm()),
                gate_head_b_mean=float(gh_b.mean()),
                gate_head_b_std=float(gh_b.std()),
                gate_weights_loaded=bool(loaded))
    if not random_init and not loaded:
        raise SystemExit("frame_split is at INIT -- checkpoint did not load the "
                         f"gate. {info}")
    # The backbone does not keep in_dim/seq_len; the probes need them to carve
    # the 4 s trial into the seg0 window.
    bb.in_dim = p.in_dim
    bb.seq_len = p.seq_len
    return bb, p, info


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class KeyedPhysio(physio_dataset.CustomDataset):
    """PhysioNet-MI + the LMDB key, so we can recover the SUBJECT for grouped CV."""

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        d['key'] = self.keys[idx]
        return d

    def collate(self, batch):
        out = super().collate(batch)
        out['key'] = [b['key'] for b in batch]
        out['subject'] = [b['key'].split('R', 1)[0] for b in batch]
        return out


NUM_WORKERS = 4


def physio_loader(p, mode, batch_size=128, num_workers=None):
    ds = KeyedPhysio(p.datasets_dir, mode=mode, highpass_hz=0.0, fs=200)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=ds.collate,
        num_workers=NUM_WORKERS if num_workers is None else num_workers,
        shuffle=False)


def egobrain_loader(p, batch_size=64, n_clips=3000, seed=0):
    """The PRETRAINING domain. PhysioNet is transfer -- any claim about what the
    gate LEARNED has to hold here. Note the trainer divides by 100 before every
    encoder forward (pretrain_trainer.py:177-179) while the EgoBrain dataset
    emits raw uV; PhysioNet's dataset already bakes the /100 in. So this loader
    must apply it by hand or the encoder runs ~2 orders of magnitude off
    distribution."""
    import functools
    from datasets.egobrain_dataset import EgoBrainDataset, collate_egobrain

    ds = EgoBrainDataset(
        data_dir=os.path.join(REPO, 'data/EgoBrain'),
        subjects=[f'P{i:04d}' for i in range(1, 41)],
        in_dim=p.in_dim, n_windows=1, window_s=1.0, stride_s=0.2, clip_s=4.0,
        erp_latency_s=-0.15, fs_out=200, max_channels=32,
        load_frames=False, frame_size=1,
        use_frame_grid=True, frame_grid_s=0.2,
        temporal_jitter=False, motion_resample=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_clips, len(ds)), replace=False)
    sub = torch.utils.data.Subset(ds, idx.tolist())
    return torch.utils.data.DataLoader(
        sub, batch_size=batch_size, num_workers=0,
        collate_fn=functools.partial(collate_egobrain, frame_objective=True))


def unpack(batch, bb, domain):
    """-> (x, ch_coords, ch_names, vcm, vlm) in the encoder's expected scaling."""
    if domain == 'egobrain':
        x = (batch['timeseries'] / 100.0).to(DEV)     # trainer's /100
        return (x, batch['ch_coords'].to(DEV), batch['ch_names'],
                batch['valid_channel_mask'].to(DEV),
                batch['valid_length_mask'].to(DEV))
    x = batch['x'].to(DEV)                            # PhysioNet: /100 already applied
    B = x.size(0)
    x = x.reshape(B, x.size(1), -1, bb.in_dim)[:, :, :bb.seq_len].contiguous()
    return x, batch['ch_coords'].to(DEV), batch['ch_names'], None, None


# ---------------------------------------------------------------------------
# the frame-averaging forward, unrolled so we can see inside
# ---------------------------------------------------------------------------
@torch.no_grad()
def frame_forward(bb, x, ch_coords, ch_names, flip=None, vcm=None, vlm=None):
    """Re-implements alignment.py::_forward_frame_averaging up to the readout,
    returning every intermediate the probes need. Mirrors the model exactly."""
    B, C, N, _ = x.shape
    if flip is None:
        flip = torch.zeros(B, dtype=torch.bool, device=x.device)

    z = bb.patch_embedding(x, None)                       # (B,C,N,d)
    coord_pe, _ = bb._spherical_positional_encoding(ch_coords)
    perm = build_flip_perm_batch(ch_names, vcm).to(z.device)      # (B,C)
    Pz, gate = bb._build_Pz(z, coord_pe, perm, vcm)

    emb_z, vcm_g, vlm_g = bb._frame_add_context(z, ch_coords, vcm, vlm)
    emb_Pz, _, _ = bb._frame_add_context(Pz, ch_coords, vcm, vlm)
    a = bb._frame_run_layers(emb_z, bb._frame_backbone_layers, vcm_g, vlm_g)
    b = bb._frame_run_layers(emb_Pz, bb._frame_backbone_layers, vcm_g, vlm_g)
    off = 1 if bb.add_global else 0
    Pa = bb._swap_channels(a, perm, channel_offset=off)
    Pb = bb._swap_channels(b, perm, channel_offset=off)

    inv = 0.5 * (a + b)
    fr = flip.view(B, 1, 1, 1)
    eq = torch.where(fr, 0.5 * (b - Pa), 0.5 * (a - Pb))
    eq_opp = torch.where(fr, 0.5 * (a - Pb), 0.5 * (b - Pa))
    half = inv.size(-1) // 2
    h = torch.cat([inv[..., :half], eq[..., half:]], dim=-1)

    global_rep = h[:, 0, 0, :]
    opp_global = torch.cat([inv[:, 0, 0, :half], eq_opp[:, 0, 0, half:]], dim=-1)
    return dict(z=z, Pz=Pz, gate=gate, perm=perm, inv=inv, eq=eq, h=h,
                global_rep=global_rep, opp_global=opp_global,
                patch_tokens=h[:, 1:, 1:, :], half=half)


@torch.no_grad()
def verify_forward(bb, batch, domain='physio'):
    """frame_forward must reproduce the model's own encoder_only output EXACTLY.
    If it does not, every number in this file is measuring the wrong thing."""
    x, cc, cn, vcm, vlm = unpack(batch, bb, domain)
    B = x.size(0)
    ours = frame_forward(bb, x, cc, cn, vcm=vcm, vlm=vlm)
    bd = {'timeseries': x, 'ch_coords': cc, 'ch_names': cn,
          'flip': torch.zeros(B, dtype=torch.bool, device=DEV)}
    if vcm is not None:
        bd['valid_channel_mask'] = vcm
    if vlm is not None:
        bd['valid_length_mask'] = vlm
    _, info = bb(bd, encoder_only=True)
    dg = (ours['global_rep'] - info['global_rep']).abs().max().item()
    dp = (ours['patch_tokens'] - info['patch_tokens']).abs().max().item()
    scale = info['global_rep'].abs().max().item()
    ok = dg < 1e-4 * max(scale, 1.0) and dp < 1e-4 * max(scale, 1.0)
    return dict(max_abs_diff_global_rep=dg, max_abs_diff_patch_tokens=dp,
                global_rep_scale=scale, forward_matches_model=bool(ok))


def swap_ch(t, perm):
    """Homologous-channel swap on dim 1 of (B, C, ...) using perm (B, C)."""
    idx = perm.view(*perm.shape, *([1] * (t.dim() - 2))).expand_as(t)
    return torch.gather(t, 1, idx)


def antisym_fraction(u, perm, pair_mask=None):
    """R(u) = ||u - Pu||^2 / (||u - Pu||^2 + ||u + Pu||^2), over dims != 0.

    ``pair_mask`` (B, C) bool restricts the sum to PAIRED (non-midline) channels;
    midline channels have Pu == u there and only pad the symmetric term."""
    Pu = swap_ch(u, perm)
    d = u - Pu
    s = u + Pu
    if pair_mask is not None:
        m = pair_mask.view(*pair_mask.shape, *([1] * (u.dim() - 2))).to(u.dtype)
        d = d * m
        s = s * m
    num = (d ** 2).sum().item()
    den = num + (s ** 2).sum().item()
    return num / max(den, 1e-12)


# ---------------------------------------------------------------------------
# montage helpers
# ---------------------------------------------------------------------------
def montage_info(ch_names):
    """-> (is_midline (C,), homolog_idx (C,)) for one row's channel names."""
    norm = [_normalize_ch_name(n) for n in ch_names]
    idx = {n: i for i, n in enumerate(norm)}
    mid, hom = [], []
    for i, n in enumerate(norm):
        h = _get_homologous_name(n)
        j = idx.get(h, i)
        hom.append(j)
        mid.append(j == i)
    return np.array(mid), np.array(hom)


# ===========================================================================
# STAGE: weights -- data-free gate topography from the checkpoint weights
# ===========================================================================
@torch.no_grad()
def stage_weights(bb, ch_coords, ch_names, n_null=200, seed=0):
    """gate = sigmoid( W.GELU(sig_proj(x))  +  [W.coord_proj(pe_c) + b] ).

    The bracket is a pure per-channel, DATA-FREE logit offset. Compute it, and
    test whether it is LEFT-RIGHT MIRROR SYMMETRIC (which a consistent P
    operator requires: g(C3) must equal g(C4) or the swap is inconsistent)."""
    fs = bb.frame_split
    pe, _ = bb._spherical_positional_encoding(ch_coords[:1])   # (1, C, pe_dim)
    W = fs.gate_head.weight                                     # (d, hidden)
    bvec = fs.gate_head.bias                                    # (d,)
    L = F.linear(fs.coord_proj(pe[0]), W, bvec)                 # (C, d) logit offset

    mid, hom = montage_info(ch_names)
    paired = ~mid
    Lh = L[torch.as_tensor(hom, device=L.device)]               # mirrored copy

    def sym_stats(M):
        Mh = M[torch.as_tensor(hom, device=M.device)]
        P = torch.as_tensor(paired, device=M.device)
        a = 0.5 * (M - Mh)[P]                                   # antisymmetric part
        s = 0.5 * (M + Mh)[P]
        asym_frac = float((a ** 2).sum() / ((a ** 2).sum() + (s ** 2).sum() + 1e-12))
        Mc = M[P] - M[P].mean(0, keepdim=True)
        Mhc = Mh[P] - Mh[P].mean(0, keepdim=True)
        num = float((Mc * Mhc).sum())
        den = float(Mc.norm() * Mhc.norm() + 1e-12)
        return asym_frac, num / den

    asym_frac, mirror_corr = sym_stats(L)

    # Norm-matched random-W null: how mirror-symmetric would an ARBITRARY linear
    # read-out of the same coord PE be? (The PE itself is not mirror-symmetric --
    # mirroring negates theta -- so this null is genuinely asymmetric.)
    g = torch.Generator(device='cpu').manual_seed(seed)
    null_asym, null_corr = [], []
    for _ in range(n_null):
        Wr = torch.randn(W.shape, generator=g).to(W.device)
        Wr = Wr * (W.norm() / Wr.norm())
        Lr = F.linear(fs.coord_proj(pe[0]), Wr, bvec)
        a, c = sym_stats(Lr)
        null_asym.append(a)
        null_corr.append(c)
    null_asym = np.array(null_asym)
    null_corr = np.array(null_corr)

    # How much does the coord path modulate the gate at all? (0 at init: W == 0)
    coord_logit_std = float(L.std(dim=0).mean())
    per_ch_logit = L.mean(dim=1).cpu().numpy()                  # (C,) mean over feats

    return dict(
        coord_logit_std_across_channels=coord_logit_std,
        coord_logit_mean=float(L.mean()),
        mirror_antisym_fraction=asym_frac,
        mirror_antisym_fraction_null_mean=float(null_asym.mean()),
        mirror_antisym_fraction_null_std=float(null_asym.std()),
        mirror_antisym_z=float((asym_frac - null_asym.mean()) / (null_asym.std() + 1e-12)),
        mirror_antisym_p_lower=float((null_asym <= asym_frac).mean()),
        mirror_corr=mirror_corr,
        mirror_corr_null_mean=float(null_corr.mean()),
        mirror_corr_null_std=float(null_corr.std()),
        mirror_corr_p_upper=float((null_corr >= mirror_corr).mean()),
        n_channels=len(ch_names), n_midline=int(mid.sum()),
        per_channel_coord_logit={n: float(v) for n, v in zip(ch_names, per_ch_logit)},
    )


# ===========================================================================
# STAGE: antisym -- does the gate SELECT the antisymmetric signal components?
# ===========================================================================
@torch.no_grad()
def stage_antisym(bb, loader, max_batches=20, seed=0, gate_store=None,
                  domain='physio'):
    """R(z_lat) > R(z) > R(z_bi)  <=>  the gate routes the hemispherically
    ANTISYMMETRIC part of the signal into the lateral stream.

    A CONSTANT gate c gives R(c*z) == R((1-c)*z) == R(z) exactly, so any gap is
    entirely learned. Nulls re-run the same statistic with the gate's channel
    assignment (and, separately, its feature assignment) shuffled."""
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ('R_z', 'R_lat', 'R_bi', 'R_lat_chshuf', 'R_lat_ftshuf',
                           'R_lat_gsym', 'R_bi_gsym', 'gate_asym_frac',
                           'gate_mean', 'gate_mid', 'gate_lat')}
    per_ch_gate, per_ch_n = None, 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x, cc, cn, vcm, vlm = unpack(batch, bb, domain)
        B = x.size(0)
        out = frame_forward(bb, x, cc, cn, vcm=vcm, vlm=vlm)
        z, gate, perm = out['z'], out['gate'], out['perm']
        z_lat = gate * z
        z_bi = (1 - gate) * z

        mid, _ = montage_info(cn[0])
        pair_mask = torch.as_tensor(~mid, device=DEV).unsqueeze(0).expand(B, -1)

        acc['R_z'].append(antisym_fraction(z, perm, pair_mask))
        acc['R_lat'].append(antisym_fraction(z_lat, perm, pair_mask))
        acc['R_bi'].append(antisym_fraction(z_bi, perm, pair_mask))

        # NULL 1: shuffle the gate ACROSS CHANNELS (keeps its marginal
        # distribution and its feature structure; destroys its scalp placement).
        cperm = torch.as_tensor(rng.permutation(gate.size(1)), device=DEV)
        acc['R_lat_chshuf'].append(
            antisym_fraction(gate[:, cperm] * z, perm, pair_mask))
        # NULL 2: shuffle the gate ACROSS FEATURES (keeps its scalp map;
        # destroys which embedding dims it calls lateral).
        fperm = torch.as_tensor(rng.permutation(gate.size(-1)), device=DEV)
        acc['R_lat_ftshuf'].append(
            antisym_fraction(gate[..., fperm] * z, perm, pair_mask))

        # NULL 3 -- THE ONE THAT MATTERS. Write S for the homologous swap and
        # split the gate into its own symmetric / antisymmetric parts
        #     gbar = (g + Sg)/2 ,  delta = (g - Sg)/2 .
        # Then
        #     g*z - S(g*z) = gbar*(z - Sz) + delta*(z + Sz)
        # and the SECOND term is the trap: the symmetric signal (z + Sz) carries
        # ~13x the energy of the antisymmetric one here (R_z ~ 0.07), so a gate
        # that is merely hemispherically ASYMMETRIC manufactures antisymmetry out
        # of perfectly bilateral signal. On a signal with ZERO antisymmetric
        # energy the closed form is
        #     sel = delta^2/(delta^2 + gbar^2) - delta^2/(delta^2 + (1-gbar)^2)
        # which is POSITIVE for every gbar < 0.5 -- i.e. for every gate in this
        # family. So a positive ``selectivity`` is NOT evidence of anything.
        # (The random-init control gives exactly 0 only because its gate is a
        # global SCALAR -- gate_head.weight is initialised to zeros -- which is
        # the one gate class where delta == 0 identically. That control is
        # circular and proves nothing about a trained gate.)
        # The honest question is whether the gate selects antisymmetric signal
        # BEYOND what its own asymmetry mechanically produces. Symmetrise the
        # gate -- keeping its full channel topography, feature structure and data
        # dependence, removing only its power to fabricate asymmetry -- and ask
        # again.
        gate_sym = 0.5 * (gate + swap_ch(gate, perm))
        acc['R_lat_gsym'].append(
            antisym_fraction(gate_sym * z, perm, pair_mask))
        acc['R_bi_gsym'].append(
            antisym_fraction((1 - gate_sym) * z, perm, pair_mask))
        acc['gate_asym_frac'].append(float(
            ((gate - swap_ch(gate, perm)) ** 2).sum()
            / (((gate - swap_ch(gate, perm)) ** 2).sum()
               + ((gate + swap_ch(gate, perm)) ** 2).sum()).clamp(min=1e-12)))

        gm = gate.mean(dim=(0, 2, 3))                          # (C,)
        per_ch_gate = gm if per_ch_gate is None else per_ch_gate + gm
        per_ch_n += 1
        acc['gate_mean'].append(float(gate.mean()))
        acc['gate_mid'].append(float(gate[:, torch.as_tensor(mid, device=DEV)].mean()))
        acc['gate_lat'].append(float(gate[:, torch.as_tensor(~mid, device=DEV)].mean()))

    res = {k: float(np.mean(v)) for k, v in acc.items() if v}
    res['R_lat_minus_R_z'] = res['R_lat'] - res['R_z']
    res['R_z_minus_R_bi'] = res['R_z'] - res['R_bi']
    res['selectivity'] = res['R_lat'] - res['R_bi']
    # Per-batch spread of the headline statistic. R is INVARIANT to a scalar
    # rescale of the gate (numerator and denominator both scale by c^2), so
    # ``selectivity`` is gauge-free -- unlike the gate mean, which the backbone
    # can trade against its own gain. Its analytic null is EXACTLY 0 for any
    # constant gate, so the SE below is the only thing standing between a
    # reported effect and noise.
    sel = np.array(acc['R_lat']) - np.array(acc['R_bi'])
    res['selectivity_per_batch_sd'] = float(sel.std(ddof=1)) if len(sel) > 1 else float('nan')
    res['selectivity_se'] = float(sel.std(ddof=1) / np.sqrt(len(sel))) if len(sel) > 1 else float('nan')
    res['selectivity_t'] = float(res['selectivity'] / (res['selectivity_se'] + 1e-12))
    res['n_batches'] = min(max_batches, bi + 1)

    # THE HEADLINE. selectivity_gsym is the raw selectivity recomputed with the
    # gate's own hemispheric asymmetry removed. Only this version answers "does
    # the gate select LATERALIZED SIGNAL?" -- the raw one also answers "is the
    # gate itself lopsided?", and the second effect dominates (see NULL 3).
    if acc['R_lat_gsym']:
        selg = np.array(acc['R_lat_gsym']) - np.array(acc['R_bi_gsym'])
        res['selectivity_gsym'] = float(selg.mean())
        res['selectivity_gsym_se'] = float(
            selg.std(ddof=1) / np.sqrt(len(selg))) if len(selg) > 1 else float('nan')
        res['enrichment_gsym'] = float(np.mean(acc['R_lat_gsym']) / res['R_z'])
        res['gate_asym_frac'] = float(np.mean(acc['gate_asym_frac']))
    if gate_store is not None and per_ch_gate is not None:
        gate_store['per_channel_gate'] = (per_ch_gate / per_ch_n).cpu().numpy()
        gate_store['ch_names'] = cn[0]
    return res


# ===========================================================================
# STAGE: eqconst -- is eq_global a constant orientation bit?
# ===========================================================================
@torch.no_grad()
def stage_eqconst(bb, loader, max_batches=20, domain='physio'):
    E, Bi = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, cc, cn, vcm, vlm = unpack(batch, bb, domain)
        out = frame_forward(bb, x, cc, cn, vcm=vcm, vlm=vlm)
        h = out['half']
        E.append(out['global_rep'][:, h:].float().cpu().numpy())    # lateral half
        Bi.append(out['global_rep'][:, :h].float().cpu().numpy())   # bilateral half
    E = np.concatenate(E)
    Bi = np.concatenate(Bi)

    def stats(M, tag):
        mu = M.mean(0)
        kappa = float((mu ** 2).sum() / (M ** 2).sum(1).mean())     # constant fraction
        Mc = M - mu
        C = np.cov(Mc, rowvar=False)
        ev = np.linalg.eigvalsh(C)
        ev = np.clip(ev, 0, None)
        pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-30))     # participation ratio
        return {f'{tag}_kappa_constant_fraction': kappa,
                f'{tag}_participation_ratio': pr,
                f'{tag}_dim': int(M.shape[1]),
                f'{tag}_mean_norm': float(np.linalg.norm(mu)),
                f'{tag}_rms_norm': float(np.sqrt((M ** 2).sum(1).mean()))}

    res = {}
    res.update(stats(E, 'lat'))
    res.update(stats(Bi, 'bi'))
    # sign-discriminability of the lateral half BEFORE vs AFTER removing its mean
    # direction: how much of the "I know which way I'm facing" signal is carried
    # by a constant vector?
    mu = E.mean(0)
    mu_hat = mu / (np.linalg.norm(mu) + 1e-12)
    proj = E @ mu_hat
    res['lat_mean_dir_snr'] = float(proj.mean() / (proj.std() + 1e-12))
    res['lat_var_along_mean_dir_frac'] = float(
        proj.var() / (E.var(0).sum() + 1e-12))
    res['n_samples'] = int(E.shape[0])
    # variance fraction of the lateral half within the full global_rep
    res['lat_var_frac_of_global_rep'] = float(
        E.var(0).sum() / (E.var(0).sum() + Bi.var(0).sum() + 1e-12))
    return res


# ===========================================================================
# encode a whole split (for the mi / swap stages)
# ===========================================================================
@torch.no_grad()
def encode_split(bb, loader, anat_flip=False, max_batches=None):
    G, Y, S = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch['x'].to(DEV)
        B = x.size(0)
        x = x.reshape(B, x.size(1), -1, bb.in_dim)[:, :, :bb.seq_len].contiguous()
        cn = batch['ch_names']
        if anat_flip:
            # Mirror the SUBJECT: the data at C3 moves to C4. Electrode labels and
            # coordinates stay put, so channel i now carries its homolog's signal.
            perm = build_flip_perm_batch(cn, None).to(DEV)
            x = swap_ch(x, perm)
        out = frame_forward(bb, x, batch['ch_coords'].to(DEV), cn)
        G.append(out['global_rep'].float().cpu().numpy())
        Y.append(np.asarray(batch['y']))
        S.extend(batch['subject'])
    return np.concatenate(G), np.concatenate(Y), np.array(S)


def fisher(X, y, A, B, eps_frac=1e-2):
    ia, ib = np.isin(y, A), np.isin(y, B)
    Xa, Xb = X[ia], X[ib]
    if len(Xa) < 5 or len(Xb) < 5:
        return float('nan')
    d = Xa.mean(0) - Xb.mean(0)
    na, nb = len(Xa), len(Xb)
    Sw = ((na - 1) * np.cov(Xa, rowvar=False) + (nb - 1) * np.cov(Xb, rowvar=False)) \
        / (na + nb - 2)
    Sw = np.atleast_2d(Sw)
    Sw += eps_frac * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    try:
        return float(d @ np.linalg.solve(Sw, d))
    except np.linalg.LinAlgError:
        return float('nan')


AXES = {
    'LR_0v1_antisym': ([0], [1]),
    'handfeet_2v3_sym': ([2], [3]),
    'extent_01v2_sym': ([0, 1], [2]),
}


def stage_mi(bb, p, n_rot_null=50, seed=0, max_batches=None):
    """Fisher separation in the lateral vs bilateral half, with SUBJECT-GROUPED
    folds (the ratio is computed per held-out subject group, then averaged) and a
    random-rotation null that destroys the split while keeping the rep."""
    tr = physio_loader(p, 'train')
    te = physio_loader(p, 'test')
    Xtr, ytr, str_ = encode_split(bb, tr, max_batches=max_batches)
    Xte, yte, ste = encode_split(bb, te, max_batches=max_batches)
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    s = np.concatenate([str_, ste])
    d = X.shape[1]
    half = d // 2

    subs = np.unique(s)
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    folds = np.array_split(subs, 5)

    res = {'n_samples': int(len(y)), 'n_subjects': int(len(subs)), 'dim': int(d),
           'lat_var_frac': float(X[:, half:].var(0).sum()
                                 / (X.var(0).sum() + 1e-12))}
    for axis, (A, B) in AXES.items():
        jb_f, jl_f, ratio_f = [], [], []
        for f in folds:
            m = np.isin(s, f)
            if m.sum() < 40:
                continue
            jb = fisher(X[m][:, :half], y[m], A, B)
            jl = fisher(X[m][:, half:], y[m], A, B)
            jb_f.append(jb)
            jl_f.append(jl)
            ratio_f.append(jl / jb if jb > 1e-9 else np.nan)
        res[f'{axis}_bi'] = float(np.nanmean(jb_f))
        res[f'{axis}_lat'] = float(np.nanmean(jl_f))
        res[f'{axis}_ratio'] = float(np.nanmean(ratio_f))
        res[f'{axis}_ratio_sd'] = float(np.nanstd(ratio_f))
        res[f'{axis}_ratio_folds'] = [float(v) for v in ratio_f]

        # NULL: random orthogonal rotation of the FULL rep, then split in half.
        # Keeps every bit of information in X and destroys only the split's basis.
        null = []
        for r in range(n_rot_null):
            Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
            Xr = X @ Q
            rr = []
            for f in folds:
                m = np.isin(s, f)
                if m.sum() < 40:
                    continue
                jb = fisher(Xr[m][:, :half], y[m], A, B)
                jl = fisher(Xr[m][:, half:], y[m], A, B)
                rr.append(jl / jb if jb > 1e-9 else np.nan)
            null.append(np.nanmean(rr))
        null = np.array(null)
        res[f'{axis}_ratio_null_mean'] = float(np.nanmean(null))
        res[f'{axis}_ratio_null_sd'] = float(np.nanstd(null))
        res[f'{axis}_ratio_p_upper'] = float(np.nanmean(null >= res[f'{axis}_ratio']))
    return res


def stage_swap(bb, p, seed=0, max_batches=None):
    """Fit an LDA left-vs-right direction on the LATERAL half (train subjects),
    score held-out subjects, then score the SAME trials with the input
    anatomically mirrored. A real laterality code must INVERT (acc -> 1 - acc);
    the bilateral half's symmetric read-out must be PRESERVED."""
    tr = physio_loader(p, 'train')
    te = physio_loader(p, 'test')
    Xtr, ytr, _ = encode_split(bb, tr, max_batches=max_batches)
    Xte, yte, _ = encode_split(bb, te, max_batches=max_batches)
    Xte_f, yte_f, _ = encode_split(bb, te, anat_flip=True, max_batches=max_batches)
    half = Xtr.shape[1] // 2

    def lda_dir(X, y, A, B):
        ia, ib = np.isin(y, A), np.isin(y, B)
        Xa, Xb = X[ia], X[ib]
        d = Xa.mean(0) - Xb.mean(0)
        na, nb = len(Xa), len(Xb)
        Sw = ((na - 1) * np.cov(Xa, rowvar=False)
              + (nb - 1) * np.cov(Xb, rowvar=False)) / (na + nb - 2)
        Sw = np.atleast_2d(Sw)
        Sw += 1e-2 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
        w = np.linalg.solve(Sw, d)
        thr = 0.5 * (Xa @ w).mean() + 0.5 * (Xb @ w).mean()
        return w, thr

    res = {}
    for tag, sl in (('lat', slice(half, None)), ('bi', slice(0, half))):
        for axis, (A, B) in (('LR_0v1', ([0], [1])), ('handfeet_2v3', ([2], [3]))):
            w, thr = lda_dir(Xtr[:, sl], ytr, A, B)

            def acc(X, y):
                m = np.isin(y, A + B)
                pred = np.where(X[m][:, sl] @ w > thr, A[0], B[0])
                truth = np.where(np.isin(y[m], A), A[0], B[0])
                return float((pred == truth).mean())

            a0 = acc(Xte, yte)
            a1 = acc(Xte_f, yte_f)
            res[f'{tag}_{axis}_acc'] = a0
            res[f'{tag}_{axis}_acc_anatflip'] = a1
            res[f'{tag}_{axis}_inversion'] = float(a0 - a1)   # ~ +2*(a0-0.5) if it inverts
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--stages', default='weights,antisym,eqconst,mi,swap')
    ap.add_argument('--out', default=None)
    ap.add_argument('--random_init', action='store_true')
    ap.add_argument('--max_batches', type=int, default=20)
    ap.add_argument('--mi_max_batches', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--domain', default='physio', choices=['physio', 'egobrain'],
                    help="which data the weights/antisym/eqconst stages see. "
                         "'egobrain' is the PRETRAINING domain (the only place a "
                         "claim about what the gate LEARNED can be tested); "
                         "'physio' is transfer. mi/swap are PhysioNet-only.")
    ap.add_argument('--ego_clips', type=int, default=3000)
    args = ap.parse_args()

    global NUM_WORKERS
    NUM_WORKERS = args.num_workers

    tag = args.tag or (os.path.basename(os.path.dirname(args.ckpt)) + '_'
                       + os.path.basename(args.ckpt).split('_loss')[0])
    if args.random_init:
        tag += '_RANDINIT'
    t0 = time.time()
    bb, p, load_info = build_backbone(args.ckpt, random_init=args.random_init,
                                      seed=args.seed)
    print(f"[{tag}] loaded. d_model={p.d_model} seq_len={p.seq_len} "
          f"in_dim={p.in_dim} n_layer={p.n_layer}", flush=True)
    print(f"[{tag}] gate weights: {load_info}", flush=True)

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    res = {'tag': tag, 'ckpt': args.ckpt, 'random_init': args.random_init,
           'load_info': load_info,
           'arch': {k: getattr(p, k) for k in
                    ('d_model', 'seq_len', 'in_dim', 'n_layer', 'nhead')}}

    dom = args.domain
    res['domain'] = dom

    def make_loader():
        return (egobrain_loader(p, n_clips=args.ego_clips, seed=args.seed)
                if dom == 'egobrain' else physio_loader(p, 'test'))

    b0 = next(iter(make_loader()))
    cn0 = b0['ch_names'][0]
    cc0 = (b0['ch_coords'][:1]).to(DEV)

    res['verify'] = verify_forward(bb, b0, dom)
    print(f"[{tag}/{dom}] verify: {res['verify']}", flush=True)
    if not res['verify']['forward_matches_model']:
        raise SystemExit("frame_forward does NOT reproduce the model's encoder "
                         f"output: {res['verify']}")

    if 'weights' in stages:
        print(f"[{tag}/{dom}] stage weights ...", flush=True)
        res['weights'] = stage_weights(bb, cc0, cn0, seed=args.seed)
    if 'antisym' in stages:
        print(f"[{tag}/{dom}] stage antisym ...", flush=True)
        gs = {}
        res['antisym'] = stage_antisym(bb, make_loader(),
                                       max_batches=args.max_batches,
                                       seed=args.seed, gate_store=gs, domain=dom)
        if 'per_channel_gate' in gs:
            res['antisym']['per_channel_gate'] = {
                n: float(v) for n, v in zip(gs['ch_names'], gs['per_channel_gate'])}
    if 'eqconst' in stages:
        print(f"[{tag}/{dom}] stage eqconst ...", flush=True)
        res['eqconst'] = stage_eqconst(bb, make_loader(),
                                       max_batches=args.max_batches, domain=dom)
    if 'mi' in stages:
        print(f"[{tag}] stage mi ...", flush=True)
        res['mi'] = stage_mi(bb, p, seed=args.seed, max_batches=args.mi_max_batches)
    if 'swap' in stages:
        print(f"[{tag}] stage swap ...", flush=True)
        res['swap'] = stage_swap(bb, p, seed=args.seed,
                                 max_batches=args.mi_max_batches)

    res['elapsed_s'] = time.time() - t0
    out = args.out or os.path.join(REPO, 'outputs', 'split_probe', f'{tag}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"[{tag}] wrote {out}  ({res['elapsed_s']:.0f}s)", flush=True)
    print(json.dumps({k: v for k, v in res.items()
                      if k in ('weights', 'antisym', 'eqconst', 'mi', 'swap')},
                     indent=2, default=str)[:4000])


if __name__ == '__main__':
    main()
