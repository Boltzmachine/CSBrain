"""Per-loss gradient-magnitude diagnostic for the WorldModel pretraining.

Answers: *how hard does each loss term push the weights?* The total pretrain
objective is a weighted sum ``L = Σ coef_i · loss_i``; wandb already logs each
term's scalar VALUE, but not the size of the gradient it contributes. This
script loads a pretraining checkpoint, runs the EXACT training-step loss
assembly on a few real batches, and for every term reports the L2 norm of the
gradient that term alone contributes to (a) the shared EEG encoder and (b) all
trainable params.

Method
------
Each weighted term ``coef_i · loss_i`` is back-propagated SEPARATELY with
``torch.autograd.grad(..., retain_graph=True)`` and its grad norm taken. Because
gradients are linear, the vector sum of the per-term grads equals the total
gradient (the thing clipped at ``clip_value``) — but the NORMS do not add. If
``Σ_i ‖g_i‖  ≫  ‖Σ_i g_i‖`` the terms are pulling in different directions
(gradient conflict); the script reports that ratio. ``allow_unused=True`` is
required because e.g. ``frame_pred_loss`` only touches the predictor + the EEG
conditioning path, not every parameter.

We also report the RAW (unweighted) grad norm, ``‖g_i‖ / |coef_i|`` — the
intrinsic gradient scale of each loss before its coefficient — which is what you
want when retuning the weights.

The model is left in ``train()`` mode (dropout on, per-sample frame-averaging
flip sampled) so the numbers reflect real training-step gradients; results are
averaged over ``--n_batches`` to average out that stochasticity.

Faithfulness
------------
This reconstructs the model + dataloader from the checkpoint's saved params and
re-implements the ``pretrain_trainer.Trainer.train`` ``need_mask`` branch
verbatim (masked MSE / band-split recon + the frame-averaging
``skip_external_recon`` non-flip-row handling + the aux band phase/envelope
terms). It supports the ``--dataset_dir egobrain`` map-style loader used by
sh/pretrain_worldmodel.sh. The trainer is NOT modified.

Usage
-----
    conda run -n cbramod python diag_grad_loss_norms.py \
        --ckpt outputs/wm-dino-dense/epoch20_loss....pth \
        --n_batches 8 --batch_size 32
"""

from __future__ import annotations

import argparse
import functools
import re
from collections import defaultdict

import numpy as np
import torch

from utils.util import (
    load_pretrain_checkpoint, generate_mask,
    band_split_recon_loss, band_phase_envelope_loss,
)
from models import get_model
# Reuse the trainer's own helpers so the loss assembly is byte-faithful.
from pretrain_trainer import to_device, _parse_band, _parse_bands


# The fixed montage constants pretrain_main.main() builds before get_model.
# The WorldModel encoder is created with brain_regions=None and never touches
# sorted_indices, but get_model's signature requires them, so pass the real
# values (harmless if unused).
_BRAIN_REGIONS = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]
_ELECTRODE_LABELS = [
    "FP1-REF", "FP2-REF", "F3-REF", "F4-REF", "C3-REF", "C4-REF", "P3-REF",
    "P4-REF", "O1-REF", "O2-REF", "F7-REF", "F8-REF", "T3-REF", "T4-REF",
    "T5-REF", "T6-REF", "FZ-REF", "CZ-REF", "PZ-REF",
]
_TOPOLOGY = {
    0: ["FP1-REF", "F7-REF", "F3-REF", "FZ-REF", "F4-REF", "F8-REF", "FP2-REF"],
    4: ["C3-REF", "CZ-REF", "C4-REF"],
    1: ["P3-REF", "PZ-REF", "P4-REF"],
    3: ["O1-REF", "O2-REF"],
    2: ["T3-REF", "T5-REF", "T6-REF", "T4-REF"],
    -1: ["A1-REF"],
}


def _sorted_indices():
    groups = defaultdict(list)
    for i, region in enumerate(_BRAIN_REGIONS):
        groups[region].append((i, _ELECTRODE_LABELS[i]))
    out = []
    for region in sorted(groups.keys()):
        elecs = sorted(groups[region], key=lambda x: _TOPOLOGY[region].index(x[1]))
        out.extend(e[0] for e in elecs)
    return out


def build_params(saved, overrides):
    """argparse.Namespace from the checkpoint's saved param dict + overrides."""
    if saved is None:
        raise SystemExit(
            "Checkpoint has no saved params (legacy bare state_dict). This "
            "script needs a checkpoint written by save_pretrain_checkpoint.")
    ns = argparse.Namespace(**dict(saved))
    for k, v in overrides.items():
        if v is not None:
            setattr(ns, k, v)
    return ns


def build_egobrain_loader(params, num_workers, n_batches):
    """Replicate pretrain_main's ``dataset_dir == 'egobrain'`` branch."""
    import os
    from datasets.egobrain_dataset import EgoBrainDataset, collate_egobrain

    if params.egobrain_subjects.lower() == 'all':
        ego_subjects = sorted(
            d for d in os.listdir(params.egobrain_root)
            if re.match(r'^P\d{4}$', d)
            and os.path.isdir(os.path.join(params.egobrain_root, d)))
    else:
        ego_subjects = [s.strip() for s in params.egobrain_subjects.split(',')
                        if s.strip()]

    ds = EgoBrainDataset(
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
        hand_labels_dir=getattr(params, 'egobrain_hand_labels_dir', None),
        hand_grid_dir=getattr(params, 'egobrain_hand_grid_dir', None),
        use_embeddings=params.use_cached_embeddings,
        emb_cache_dir=getattr(params, 'egobrain_emb_cache_dir', None),
        use_frame_grid=params.egobrain_use_frame_grid,
        frame_grid_dir=getattr(params, 'egobrain_frame_grid_dir', None),
        frame_grid_s=params.egobrain_frame_grid_s,
        temporal_jitter=not params.egobrain_no_temporal_jitter,
        use_grid_embeddings=params.egobrain_use_grid_embeddings,
        emb_grid_dir=getattr(params, 'egobrain_emb_grid_dir', None),
        ea_matrices=None,
        delta_whiten_g0=params.egobrain_delta_whiten_g0,
        delta_whiten_cutoff_hz=params.egobrain_delta_whiten_cutoff_hz,
        motion_resample=params.egobrain_motion_resample,
        motion_resample_alpha=params.egobrain_motion_resample_alpha,
        motion_resample_space=params.egobrain_motion_resample_space,
        motion_resample_metric=params.egobrain_motion_resample_metric,
        motion_resample_cap_pct=params.egobrain_motion_resample_cap_pct,
        motion_resample_floor_mix=params.egobrain_motion_resample_floor_mix,
        motion_emb_dir=getattr(params, 'egobrain_motion_emb_dir', None),
    )
    print(f"EgoBrain clips: {len(ds)}  (subjects: {len(ego_subjects)})")
    sampler = torch.utils.data.RandomSampler(
        ds, replacement=True, num_samples=params.batch_size * n_batches)
    frame_obj = (getattr(params, 'model', None) == 'WorldModel'
                 and getattr(params, 'wm_objective', 'eeg') == 'frame')
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=params.batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=functools.partial(collate_egobrain, frame_objective=frame_obj),
        pin_memory=True,
        drop_last=True,
    )
    return loader


def compute_loss_terms(model, batch, params, device, criterion):
    """Run one training step and return ``{name: (coef, raw_loss_tensor)}``.

    Mirrors pretrain_trainer.Trainer.train's ``need_mask`` branch (the path
    sh/pretrain_worldmodel.sh takes: masked recon + frame-averaging
    skip_external_recon + aux band terms). The graph is retained by the caller.
    """
    mask_w = getattr(params, 'mask_weight', 1.0)
    batch = to_device(batch, device)
    x = batch['timeseries'] / 100
    batch['timeseries'] = x

    bz, ch_num, patch_num, patch_size = x.shape
    mask = generate_mask(bz, ch_num, patch_num,
                         mask_ratio=params.mask_ratio, device=device)
    out = model.training_step(batch, mask=mask)
    assert isinstance(out, tuple), "expected (y, info) from training_step"
    y, info = out

    terms = {}       # name -> (coef, raw_loss)  ; coef may be a 0-dim tensor
    raw_vals = {}    # name -> float scalar value (for the report)

    # (1) Terms the model emits into info as (coef, tensor). Match the trainer:
    #     add only when coef != 0 (so frame_recon_loss at weight 0 is dropped).
    for key, value in info.items():
        if 'loss' in key:
            coef, lss = value
            raw_vals[key] = float(lss.detach())
            keep = bool(coef != 0) if not torch.is_tensor(coef) else bool((coef != 0).any())
            if keep:
                terms[key] = (coef, lss)

    # (2) Trainer-level reconstruction + aux terms (NOT in info).
    if info.get('skip_external_recon', False):
        # Frame-averaging per-sample flip: raw-space recon scored on NON-flip
        # rows only, scaled so their gradient matches the full-batch masked
        # count (ratio_nf). Identical to pretrain_trainer.py:287-343.
        flip_row = info.get('flip_row')
        if flip_row is not None:
            nf = (~flip_row.bool()).view(mask.size(0), *([1] * (mask.dim() - 1)))
            eff = (mask == 1) & nf
        else:
            eff = (mask == 1)
        if eff.any():
            ratio_nf = (eff.sum().float()
                        / (mask == 1).sum().clamp(min=1).float())
            masked_x, masked_y = x[eff], y[eff]
            if getattr(params, 'recon_band_split', False):
                mask_loss = band_split_recon_loss(
                    masked_y, masked_x, fs=float(getattr(params, 'fs', 200)),
                    phase_cutoff_hz=params.recon_phase_cutoff_hz,
                    power_weight=getattr(params, 'recon_power_weight', 1.0))
            else:
                mask_loss = criterion(masked_y, masked_x)
            terms['mask_loss'] = (ratio_nf * mask_w, mask_loss)
            raw_vals['mask_loss'] = float(mask_loss.detach())

            if getattr(params, 'aux_band_pred', False):
                aux_mask = mask if flip_row is None else mask * nf.to(mask.dtype)
                phase_loss, env_loss = band_phase_envelope_loss(
                    y, x, aux_mask, fs=float(getattr(params, 'fs', 200)),
                    delta_band=_parse_band(getattr(params, 'aux_delta_band', '0.5,4')),
                    power_bands=_parse_bands(getattr(params, 'aux_power_bands', '8,13;13,30')))
                raw_vals['aux_phase_loss'] = float(phase_loss.detach())
                raw_vals['aux_env_loss'] = float(env_loss.detach())
                if params.aux_phase_weight != 0:
                    terms['aux_phase_loss'] = (ratio_nf * params.aux_phase_weight, phase_loss)
                if params.aux_envelope_weight != 0:
                    terms['aux_env_loss'] = (ratio_nf * params.aux_envelope_weight, env_loss)
    else:
        # Plain masked recon (no frame-averaging flip this step).
        masked_x, masked_y = x[mask == 1], y[mask == 1]
        if getattr(params, 'recon_band_split', False):
            mask_loss = band_split_recon_loss(
                masked_y, masked_x, fs=float(getattr(params, 'fs', 200)),
                phase_cutoff_hz=params.recon_phase_cutoff_hz,
                power_weight=getattr(params, 'recon_power_weight', 1.0))
        else:
            mask_loss = criterion(masked_y, masked_x)
        terms['mask_loss'] = (mask_w, mask_loss)
        raw_vals['mask_loss'] = float(mask_loss.detach())

        if getattr(params, 'aux_band_pred', False):
            phase_loss, env_loss = band_phase_envelope_loss(
                y, x, mask, fs=float(getattr(params, 'fs', 200)),
                delta_band=_parse_band(getattr(params, 'aux_delta_band', '0.5,4')),
                power_bands=_parse_bands(getattr(params, 'aux_power_bands', '8,13;13,30')))
            raw_vals['aux_phase_loss'] = float(phase_loss.detach())
            raw_vals['aux_env_loss'] = float(env_loss.detach())
            if params.aux_phase_weight != 0:
                terms['aux_phase_loss'] = (params.aux_phase_weight, phase_loss)
            if params.aux_envelope_weight != 0:
                terms['aux_env_loss'] = (params.aux_envelope_weight, env_loss)

    return terms, raw_vals


def grad_norms_for_terms(terms, params_all, is_enc):
    """Per-term (weighted) grad norm over all params, encoder params, and the
    rest (predictor). Returns {name: dict(all, enc, pred, coef)} plus the total
    over the summed loss."""
    def _split_sq(grads):
        s_all = torch.zeros((), device=grads_dev)
        s_enc = torch.zeros((), device=grads_dev)
        for g, e in zip(grads, is_enc):
            if g is None:
                continue
            sq = g.detach().float().pow(2).sum()
            s_all = s_all + sq
            if e:
                s_enc = s_enc + sq
        return float(s_all.sqrt()), float((s_all - s_enc).clamp(min=0).sqrt()), float(s_enc.sqrt())

    grads_dev = params_all[0].device
    out = {}
    weighted_terms = []
    for name, (coef, raw) in terms.items():
        if not (torch.is_tensor(raw) and raw.requires_grad):
            continue
        wt = coef * raw
        weighted_terms.append(wt)
        grads = torch.autograd.grad(wt, params_all, retain_graph=True,
                                    allow_unused=True)
        n_all, n_pred, n_enc = _split_sq(grads)
        out[name] = dict(all=n_all, enc=n_enc, pred=n_pred,
                         coef=float(coef.detach()) if torch.is_tensor(coef) else float(coef))

    # Total gradient (vector sum of the terms) — frees the graph.
    total_all = float('nan')
    if weighted_terms:
        total = sum(weighted_terms)
        grads = torch.autograd.grad(total, params_all, retain_graph=False,
                                    allow_unused=True)
        s = torch.zeros((), device=grads_dev)
        for g in grads:
            if g is not None:
                s = s + g.detach().float().pow(2).sum()
        total_all = float(s.sqrt())
    return out, total_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='pretraining checkpoint .pth')
    ap.add_argument('--n_batches', type=int, default=8,
                    help='batches to average the grad norms over')
    ap.add_argument('--batch_size', type=int, default=32,
                    help='override the saved batch size (smaller = faster/less mem)')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--epoch', type=int, default=None,
                    help='current_epoch for the pred-weight ramp (default: infer '
                         'from ckpt filename, else 39)')
    ap.add_argument('--subjects', type=str, default=None,
                    help='override egobrain_subjects (e.g. P0001 for a fast run)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {args.ckpt}")
    state_dict, saved = load_pretrain_checkpoint(args.ckpt, map_location='cpu')
    params = build_params(saved, {
        'batch_size': args.batch_size,
        'egobrain_subjects': args.subjects,
    })
    print(f"model={params.model}  dataset_dir={params.dataset_dir}  "
          f"wm_objective={getattr(params, 'wm_objective', 'eeg')}  "
          f"frame_averaging={getattr(params, 'frame_averaging', False)}")
    if params.dataset_dir != 'egobrain':
        raise SystemExit(
            f"This script currently supports --dataset_dir egobrain "
            f"(map-style loader); checkpoint has {params.dataset_dir!r}. Extend "
            f"build_egobrain_loader() to replicate that branch of pretrain_main.")

    model = get_model(params, _BRAIN_REGIONS, _sorted_indices())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model = model.to(device)
    model.train()  # training-step gradients: dropout on, flip sampled

    # Pred-weight ramp epoch (no-op when pred_ramp_epochs == 0).
    if hasattr(model, 'current_epoch'):
        ep = args.epoch
        if ep is None:
            m = re.search(r'epoch(\d+)', args.ckpt)
            ep = int(m.group(1)) if m else 39
        model.current_epoch.fill_(float(ep))
        print(f"current_epoch set to {ep} (pred_ramp_epochs="
              f"{getattr(params, 'pred_ramp_epochs', 0)})")

    criterion = torch.nn.MSELoss(reduction='mean').to(device)

    # Trainable-param bookkeeping: which of the trainable params live in the
    # shared EEG encoder (vs the predictor / other heads).
    params_all = [p for p in model.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in model.encoder.parameters()}
    is_enc = [id(p) in enc_ids for p in params_all]
    n_enc = sum(is_enc)
    print(f"Trainable tensors: {len(params_all)}  (encoder {n_enc}, "
          f"other/predictor {len(params_all) - n_enc})")

    loader = build_egobrain_loader(params, args.num_workers, args.n_batches)

    # Accumulate per-term grad norms over the batches.
    acc_all = defaultdict(list)
    acc_enc = defaultdict(list)
    acc_pred = defaultdict(list)
    acc_raw = defaultdict(list)
    coefs = {}
    total_norms, sum_norms = [], []

    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        model.zero_grad(set_to_none=True)
        terms, raw_vals = compute_loss_terms(model, batch, params, device, criterion)
        per_term, total_all = grad_norms_for_terms(terms, params_all, is_enc)
        for name, d in per_term.items():
            acc_all[name].append(d['all'])
            acc_enc[name].append(d['enc'])
            acc_pred[name].append(d['pred'])
            coefs[name] = d['coef']
        for name, v in raw_vals.items():
            acc_raw[name].append(v)
        total_norms.append(total_all)
        sum_norms.append(sum(d['all'] for d in per_term.values()))
        print(f"  batch {bi + 1}/{args.n_batches}: "
              f"total‖g‖={total_all:.4g}  terms={list(per_term)}")

    if not acc_all:
        raise SystemExit("No loss terms captured — check the batch has frames.")

    # --- Report ---------------------------------------------------------
    names = sorted(acc_all, key=lambda n: -np.mean(acc_all[n]))
    mean_total = float(np.nanmean(total_norms))

    def ms(xs):
        a = np.asarray(xs, float)
        return np.mean(a), (np.std(a) if len(a) > 1 else 0.0)

    print("\n" + "=" * 100)
    print(f"PER-LOSS GRADIENT MAGNITUDE  —  {args.ckpt}")
    print(f"averaged over {len(total_norms)} batches of size {params.batch_size}"
          f"   |   model in train() mode")
    print("=" * 100)
    hdr = (f"{'loss term':<20}{'weight':>9}{'raw val':>10}"
           f"{'‖g‖ all':>11}{'‖g‖ enc':>11}{'‖g‖ pred':>11}"
           f"{'raw ‖g‖':>11}{'% enc grad':>11}")
    print(hdr)
    print("-" * 100)
    for n in names:
        w = coefs.get(n, float('nan'))
        gv, _ = ms(acc_all[n])
        ge, _ = ms(acc_enc[n])
        gp, _ = ms(acc_pred[n])
        rv = float(np.mean(acc_raw[n])) if n in acc_raw else float('nan')
        raw_g = gv / abs(w) if w else float('nan')
        pct_enc = 100.0 * ge / gv if gv else 0.0
        print(f"{n:<20}{w:>9.4g}{rv:>10.4g}{gv:>11.4g}{ge:>11.4g}{gp:>11.4g}"
              f"{raw_g:>11.4g}{pct_enc:>10.1f}%")
    print("-" * 100)
    mean_sum = float(np.mean(sum_norms))
    print(f"{'TOTAL (Σ terms)':<20}{'':>9}{'':>10}{mean_total:>11.4g}")
    print(f"\nΣ‖g_i‖ / ‖Σ g_i‖ = {mean_sum:.4g} / {mean_total:.4g} = "
          f"{mean_sum / mean_total:.2f}x   "
          f"(1.0 = aligned; >>1 = terms fighting / gradient conflict)")

    # Encoder-only shares (which loss shapes the shared representation).
    enc_tot = {n: np.mean(acc_enc[n]) for n in names}
    denom = sum(enc_tot.values()) or 1.0
    print("\nShare of ENCODER gradient magnitude (sum-of-norms basis):")
    for n in sorted(enc_tot, key=lambda k: -enc_tot[k]):
        print(f"  {n:<20}{100 * enc_tot[n] / denom:>6.1f}%")
    print("=" * 100)


if __name__ == '__main__':
    main()
