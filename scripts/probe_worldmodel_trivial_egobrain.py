"""Multi-epoch trivial-identity probe for an EgoBrain WorldModel run.

Same diagnostic idea as ``probe_worldmodel_trivial.py`` (which is hardwired
to CineBrain), but:

  * loads pure-EgoBrain checkpoints (``dataset_dir == 'egobrain'``);
  * caches ONE fixed set of EgoBrain batches and replays them against every
    epoch's checkpoint, so the only thing that changes across epochs is the
    network — the inputs are byte-identical;
  * sweeps a list of epochs and prints a trend table.

For each batch it encodes window t with the ONLINE encoder (no mask, like a
clean JEPA probe), encodes windows t and t+1 with the EMA TARGET encoder (the
training regression target), runs the predictor, and reports — in the
patch-token space (B, C, N, D):

  (a) cos(s_t,  s_{t+1})  — persistence / identity baseline.
  (b) cos(pred, s_{t+1})  — what the predictor actually achieves (==
                            diag_latent_pred_cos at train time).
  (c) cos(pred, s_t)      — the trivial-copy signature. ~1 => f(s_t) ~ s_t.

  gain = (b) - (a): information the predictor adds over just copying s_t.

L1 distances mirror these. If (c) ~ 1 and gain ~ 0 the predictor collapsed to
copying the current window. If (a) is itself ~1, copying is near-optimal and
the prediction task is trivial *by construction* (consecutive 1 s windows are
nearly identical in latent space) — a separate finding worth flagging.

Usage:
  conda run -n cbramod python scripts/probe_worldmodel_trivial_egobrain.py \
      --ckpt_dir outputs/wm-bandaux2 \
      --epochs 1,3,5,7,9,10,13,16,19,22,25 \
      --n_subjects 4 --n_batches 6 --batch_size 32
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from models import get_model  # noqa: E402
from utils.util import load_pretrain_checkpoint  # noqa: E402
from datasets.egobrain_dataset import EgoBrainDataset, collate_egobrain  # noqa: E402


# --------------------------- metric helpers --------------------------------

def _patch_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = F.normalize(a.flatten(end_dim=-2), dim=-1)
    bf = F.normalize(b.flatten(end_dim=-2), dim=-1)
    return (af * bf).sum(-1).mean().item()


def _patch_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.l1_loss(a, b).item()


def _token_norm(a: torch.Tensor) -> float:
    return a.flatten(end_dim=-2).norm(dim=-1).mean().item()


def _feature_std(a: torch.Tensor) -> float:
    return a.flatten(end_dim=-2).std(dim=0).mean().item()


def _build_model_from_params(params_dict: dict):
    defaults = {
        'spectral_mode': 'static',
        'stft_n_fft': 64,
        'stft_hop': 1,
        'use_moe': False,
        'num_experts': 4,
        'moe_top_k': 2,
        'moe_gate_input_dim': 32,
        'moe_balance_weight': 0.01,
        'moe_band_prior_weight': 0.1,
        'moe_z_loss_weight': 1e-3,
        'image_pool_heads': 4,
        'vision_encoder': 'facebook/dinov2-base',
        'patch_embed_type': 'cnn',
    }
    merged = {**defaults, **params_dict}
    ns = SimpleNamespace(**merged)
    return get_model(ns, brain_regions=None, sorted_indices=None), ns


def _epoch_of(path: str) -> int:
    m = re.search(r'epoch(\d+)_loss', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _ckpt_for_epoch(ckpt_dir: str, epoch: int) -> str | None:
    hits = glob.glob(os.path.join(ckpt_dir, f'epoch{epoch}_loss*.pth'))
    return hits[0] if hits else None


@torch.no_grad()
def _cache_batches(args, saved_params) -> list[dict]:
    """Build EgoBrain once and pull a fixed set of CPU batches."""
    if saved_params.get('egobrain_subjects', 'all').lower() == 'all':
        subs = sorted(
            d for d in os.listdir(args.egobrain_root)
            if re.match(r'^P\d{4}$', d)
            and os.path.isdir(os.path.join(args.egobrain_root, d)))
    else:
        subs = [s.strip() for s in saved_params['egobrain_subjects'].split(',')
                if s.strip()]
    subs = subs[:args.n_subjects]

    ds = EgoBrainDataset(
        data_dir=args.egobrain_root,
        subjects=subs,
        in_dim=saved_params['in_dim'],
        n_windows=2,                         # need x_t and x_{t+1}
        window_s=saved_params.get('egobrain_window_s', 1.0),
        stride_s=saved_params.get('egobrain_stride_s', 1.0),
        clip_s=saved_params.get('egobrain_clip_s', 4.0),
        erp_latency_s=saved_params.get('egobrain_erp_latency_s', 0.5),
        load_frames=False,                   # EEG-only probe, no video decode
        vision_encoder=saved_params.get('vision_encoder', 'facebook/dinov2-base'),
        max_channels=saved_params.get('egobrain_max_channels', 32),
        ea_matrices=None,                    # run had use_euclidean_alignment=False
        delta_whiten_g0=saved_params.get('egobrain_delta_whiten_g0', 1.0),
        delta_whiten_cutoff_hz=saved_params.get('egobrain_delta_whiten_cutoff_hz', 8.0),
    )
    print(f'[probe] EgoBrain clips: {len(ds)} over {len(subs)} subjects {subs}')

    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, generator=g,
        collate_fn=collate_egobrain, num_workers=0, drop_last=True,
    )
    batches = []
    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        batches.append(batch)
    print(f'[probe] cached {len(batches)} batches of {args.batch_size} clips')
    return batches


@torch.no_grad()
def probe_epoch(ckpt_path, batches, device):
    state_dict, saved_params = load_pretrain_checkpoint(ckpt_path, map_location='cpu')
    model, _ = _build_model_from_params(saved_params)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    encoder = model.encoder
    target_encoder = model.target_encoder
    predictor = model.predictor
    if predictor is None:
        raise SystemExit('checkpoint has no predictor (max_horizon=0)')

    def _enc(enc, b):
        _, info = enc({**b}, encoder_only=True)
        return info['patch_tokens']

    acc = {k: 0.0 for k in [
        'cos_st_stp1', 'cos_pred_stp1', 'cos_pred_st', 'cos_st_stp1_tt',
        'l1_pred_stp1', 'l1_pred_st', 'l1_st_stp1',
        'norm_st', 'norm_stp1', 'norm_pred',
        'std_st', 'std_stp1', 'std_pred',
    ]}
    n = 0
    for batch in batches:
        b = {}
        for k, v in batch.items():
            b[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        b['timeseries'] = b['timeseries'] / 100.0      # trainer's /100 rescale

        batch_t = model._build_alignment_batch(b, window_idx=0)
        cb_idx = torch.arange(b['timeseries_future'].size(0), device=device)
        batch_tp1 = model._build_future_subbatch(b, cb_idx, window_idx=1)

        s_t_online = _enc(encoder, batch_t)            # predictor's input
        s_t_target = _enc(target_encoder, batch_t)
        s_tp1_target = _enc(target_encoder, batch_tp1)  # regression target
        pred, _ = predictor(s_t_online, horizon=1)

        acc['cos_st_stp1']    += _patch_cos(s_t_online, s_tp1_target)
        acc['cos_pred_stp1']  += _patch_cos(pred,       s_tp1_target)
        acc['cos_pred_st']    += _patch_cos(pred,       s_t_online)
        acc['cos_st_stp1_tt'] += _patch_cos(s_t_target, s_tp1_target)
        acc['l1_pred_stp1']   += _patch_l1(pred,        s_tp1_target)
        acc['l1_pred_st']     += _patch_l1(pred,        s_t_online)
        acc['l1_st_stp1']     += _patch_l1(s_t_online,  s_tp1_target)
        acc['norm_st']   += _token_norm(s_t_online)
        acc['norm_stp1'] += _token_norm(s_tp1_target)
        acc['norm_pred'] += _token_norm(pred)
        acc['std_st']   += _feature_std(s_t_online)
        acc['std_stp1'] += _feature_std(s_tp1_target)
        acc['std_pred'] += _feature_std(pred)
        n += 1

    return {k: v / max(n, 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default='outputs/wm-bandaux2')
    ap.add_argument('--epochs', default='', help='comma list; empty = all found')
    ap.add_argument('--egobrain_root', default='data/EgoBrain')
    ap.add_argument('--n_subjects', type=int, default=4)
    ap.add_argument('--n_batches', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.epochs.strip():
        epochs = [int(e) for e in args.epochs.split(',') if e.strip()]
    else:
        found = [_epoch_of(p) for p in glob.glob(os.path.join(args.ckpt_dir, 'epoch*_loss*.pth'))]
        epochs = sorted(e for e in found if e > 0)
    print(f'[probe] epochs to probe: {epochs}')

    # Build the batch cache once, using the first available checkpoint's params.
    first_ck = next((_ckpt_for_epoch(args.ckpt_dir, e) for e in epochs
                     if _ckpt_for_epoch(args.ckpt_dir, e)), None)
    assert first_ck, f'no checkpoints found in {args.ckpt_dir}'
    _, saved_params = load_pretrain_checkpoint(first_ck, map_location='cpu')
    print(f"[probe] run={saved_params.get('run_name')} dataset={saved_params.get('dataset_dir')} "
          f"max_horizon={saved_params.get('max_horizon')} pred_ramp_epochs={saved_params.get('pred_ramp_epochs')}")
    batches = _cache_batches(args, saved_params)

    rows = []
    for e in epochs:
        ck = _ckpt_for_epoch(args.ckpt_dir, e)
        if not ck:
            print(f'[probe] epoch {e}: no checkpoint, skipping')
            continue
        m = probe_epoch(ck, batches, device)
        gain = m['cos_pred_stp1'] - m['cos_st_stp1']
        rows.append((e, m, gain))
        print(f'[probe] epoch {e:2d} done  '
              f"cos(pred,t+1)={m['cos_pred_stp1']:.4f} cos(t,t+1)={m['cos_st_stp1']:.4f} "
              f"cos(pred,t)={m['cos_pred_st']:.4f} gain={gain:+.4f}")

    # ---- trend table ----
    print()
    print('=== Trivial-identity trend over epochs (patch-token cosine) ===')
    hdr = (f"{'ep':>3} | {'cos(pred,t+1)':>13} {'cos(t,t+1)':>11} {'gain':>7} | "
           f"{'cos(pred,t)':>11} | {'L1(pred,t+1)':>12} {'L1(pred,t)':>10} {'L1(t,t+1)':>9} | "
           f"{'|pred|':>6} {'|s_t|':>6} {'std_pred':>8} {'std_t':>6}")
    print(hdr)
    print('-' * len(hdr))
    for e, m, gain in rows:
        print(f"{e:>3} | {m['cos_pred_stp1']:>13.4f} {m['cos_st_stp1']:>11.4f} {gain:>+7.4f} | "
              f"{m['cos_pred_st']:>11.4f} | {m['l1_pred_stp1']:>12.4f} {m['l1_pred_st']:>10.4f} "
              f"{m['l1_st_stp1']:>9.4f} | {m['norm_pred']:>6.2f} {m['norm_st']:>6.2f} "
              f"{m['std_pred']:>8.4f} {m['std_st']:>6.4f}")

    print()
    print('Read: cos(pred,t)~1 AND gain~0 => predictor copies the current window.')
    print('      cos(t,t+1)~1 => consecutive windows already near-identical (task trivial by construction).')


if __name__ == '__main__':
    main()
