"""Diagnose whether a WorldModel checkpoint learned a trivial-identity predictor.

Loads the EMA target encoder, online encoder, and predictor from a saved
WorldModel checkpoint, then on a small CineBrain batch computes three cosine
diagnostics in the patch-token space:

  (a) cos(s_t,  s_{t+1})   — identity baseline. If high, predicting identity
                              is already a good solution.
  (b) cos(pred, s_{t+1})   — what the predictor actually achieves
                              (this is the diag_latent_pred_cos logged at train time).
  (c) cos(pred, s_t)       — the trivial-identity signature. If ≈ 1 then
                              the predictor learned f(s_t) ≈ s_t.

  Information gain over identity = (b) − (a). If this is ~0 and (c) is high,
  the predictor is trivial.

Also reports L1 distances, token-wise norms, and a feature-rank proxy
(per-token feature std) to flag representation collapse on the encoder side.

Usage:
  conda run -n cbramod python scripts/probe_worldmodel_trivial.py \
      --ckpt outputs/worldmodel-mix-cinebrain-v6/epoch26_loss0.5639974474906921.pth \
      --n_batches 4 --batch_size 16
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Repo root on sys.path so the local packages import as in training.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from models import get_model  # noqa: E402
from utils.util import load_pretrain_checkpoint  # noqa: E402
from datasets.cinebrain_dataset import CineBrainDataset, collate_cinebrain  # noqa: E402


def _patch_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token cosine between two (B, C, N, D) tensors, averaged."""
    af = F.normalize(a.flatten(end_dim=-2), dim=-1)
    bf = F.normalize(b.flatten(end_dim=-2), dim=-1)
    return (af * bf).sum(-1).mean()


def _patch_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(a, b)


def _feature_std(a: torch.Tensor) -> torch.Tensor:
    """Mean per-feature std across all tokens — low → representation collapse."""
    flat = a.flatten(end_dim=-2)  # (N_total, D)
    return flat.std(dim=0).mean()


def _token_norm(a: torch.Tensor) -> torch.Tensor:
    return a.flatten(end_dim=-2).norm(dim=-1).mean()


def _build_model_from_params(params_dict: dict):
    # The training params dict already carries everything models/__init__ reads
    # via getattr(params, ...). Wrap in a SimpleNamespace and add defaults
    # for any newer fields that older checkpoints might lack.
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


@torch.no_grad()
def probe(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[probe] loading {args.ckpt}')
    state_dict, saved_params = load_pretrain_checkpoint(args.ckpt, map_location='cpu')
    assert saved_params is not None, 'expected new-format checkpoint with params'
    print(f"[probe] saved run: model={saved_params.get('model')} "
          f"max_horizon={saved_params.get('max_horizon')} "
          f"latent_pred_weight={saved_params.get('latent_pred_weight')} "
          f"pred_ramp_epochs={saved_params.get('pred_ramp_epochs')}")

    model, ns = _build_model_from_params(saved_params)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[probe] missing keys (first 5): {missing[:5]} (total {len(missing)})')
    if unexpected:
        print(f'[probe] unexpected keys (first 5): {unexpected[:5]} (total {len(unexpected)})')

    model = model.to(device).eval()

    # Sanity: is the predictor actually trained (vs. at init)?
    if getattr(model, 'predictor', None) is not None:
        w = model.predictor.out_proj_patch.weight.detach()
        print(f'[probe] predictor.out_proj_patch.weight: '
              f'norm={w.norm().item():.4f} mean={w.mean().item():.4e} '
              f'std={w.std().item():.4e}')
    else:
        print('[probe] no predictor in this wrapper (max_horizon=0)')
        return

    # CineBrain dataset with n_windows=2 so we have x_t and x_{t+1}.
    subjects = [s.strip() for s in saved_params.get(
        'cinebrain_subjects', 'sub-0001').split(',') if s.strip()]
    ds = CineBrainDataset(
        data_dir=args.cinebrain_root,
        subjects=subjects[:args.n_subjects],
        in_dim=saved_params['in_dim'],
        n_windows=2,
        window_s=saved_params.get('cinebrain_window_s', 1.0),
        stride_s=saved_params.get('cinebrain_stride_s', 1.0),
        erp_latency_s=saved_params.get('cinebrain_erp_latency_s', 0.5),
        load_frames=False,
        vision_encoder=saved_params.get('vision_encoder', 'facebook/dinov2-base'),
    )
    print(f'[probe] cinebrain clips: {len(ds)}')

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_cinebrain, num_workers=2, drop_last=True,
    )

    encoder = model.encoder
    target_encoder = model.target_encoder
    predictor = model.predictor

    def _encode_window(enc, batch_w0_or_w1: dict):
        # Reuse the wrapper's batch builder so the encoder sees the same
        # /100 scaling and metadata layout as during training.
        _, info = enc({**batch_w0_or_w1}, encoder_only=True)
        return info['patch_tokens'], info['global_rep']

    sums = {k: 0.0 for k in [
        'cos_st_stp1', 'cos_pred_stp1', 'cos_pred_st',
        'cos_pred_stp1_target', 'cos_st_stp1_target',
        'l1_pred_stp1', 'l1_pred_st', 'l1_st_stp1',
        'norm_st', 'norm_stp1', 'norm_pred',
        'std_st', 'std_stp1', 'std_pred',
    ]}
    n_seen = 0

    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        # Move to device.
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        # Trainer rescales the primary 'timeseries' by /100; replicate.
        batch['timeseries'] = batch['timeseries'] / 100.0

        # Build the inputs the wrapper would.
        batch_t = model._build_alignment_batch(batch, window_idx=0)
        cb_idx = torch.arange(batch['timeseries_future'].size(0), device=device)
        batch_tp1 = model._build_future_subbatch(batch, cb_idx, window_idx=1)

        # Online encoder at t (this is what the predictor consumes).
        s_t_patch_online, _ = _encode_window(encoder, batch_t)

        # Target encoder at t and t+1 (this is the regression target during training).
        if target_encoder is not None:
            s_t_patch_target, _ = _encode_window(target_encoder, batch_t)
            s_tp1_patch_target, _ = _encode_window(target_encoder, batch_tp1)
        else:
            s_t_patch_target = s_t_patch_online
            s_tp1_patch_target, _ = _encode_window(encoder, batch_tp1)

        pred_patch, _ = predictor(s_t_patch_online, horizon=1)

        # Diagnostics — same shapes (B, C, N, D).
        sums['cos_st_stp1']        += _patch_cos(s_t_patch_online, s_tp1_patch_target).item()
        sums['cos_pred_stp1']      += _patch_cos(pred_patch,        s_tp1_patch_target).item()
        sums['cos_pred_st']        += _patch_cos(pred_patch,        s_t_patch_online).item()
        sums['cos_pred_stp1_target'] += _patch_cos(pred_patch,      s_tp1_patch_target).item()
        sums['cos_st_stp1_target'] += _patch_cos(s_t_patch_target,  s_tp1_patch_target).item()

        sums['l1_pred_stp1'] += _patch_l1(pred_patch, s_tp1_patch_target).item()
        sums['l1_pred_st']   += _patch_l1(pred_patch, s_t_patch_online).item()
        sums['l1_st_stp1']   += _patch_l1(s_t_patch_online, s_tp1_patch_target).item()

        sums['norm_st']   += _token_norm(s_t_patch_online).item()
        sums['norm_stp1'] += _token_norm(s_tp1_patch_target).item()
        sums['norm_pred'] += _token_norm(pred_patch).item()

        sums['std_st']   += _feature_std(s_t_patch_online).item()
        sums['std_stp1'] += _feature_std(s_tp1_patch_target).item()
        sums['std_pred'] += _feature_std(pred_patch).item()

        n_seen += 1

    if n_seen == 0:
        print('[probe] no batches ran — check that CineBrain data is reachable.')
        return

    avg = {k: v / n_seen for k, v in sums.items()}
    print()
    print(f'=== Averaged over {n_seen} batches of {args.batch_size} clips ===')
    print()
    print('[ identity vs predictor ]')
    print(f"  cos(s_t,  s_{{t+1}})   (identity baseline, online→target)  = {avg['cos_st_stp1']:.4f}")
    print(f"  cos(pred, s_{{t+1}})   (predictor quality vs target)       = {avg['cos_pred_stp1']:.4f}")
    print(f"  cos(pred, s_t)        (trivial-identity signature)        = {avg['cos_pred_st']:.4f}")
    print(f"  cos(s_t,  s_{{t+1}})   (target→target, for reference)      = {avg['cos_st_stp1_target']:.4f}")
    gain = avg['cos_pred_stp1'] - avg['cos_st_stp1']
    print()
    print(f"  Information gain over identity (cos pred-vs-t+1 − cos t-vs-t+1) = {gain:+.4f}")
    if gain <= 0.005:
        print('  -> Gain ≈ 0: predictor is NOT outperforming the identity baseline.')
    else:
        print('  -> Predictor adds measurable information over identity.')
    if avg['cos_pred_st'] > 0.97:
        print('  -> cos(pred, s_t) > 0.97: predictor output is essentially a copy of s_t.')
    print()
    print('[ L1 distances ]')
    print(f"  L1(pred, s_{{t+1}})  = {avg['l1_pred_stp1']:.4f}  (training target)")
    print(f"  L1(pred, s_t)      = {avg['l1_pred_st']:.4f}    (would be ≈0 if trivial)")
    print(f"  L1(s_t,  s_{{t+1}})  = {avg['l1_st_stp1']:.4f}   (irreducible from identity)")
    print()
    print('[ representation norms / spread ]')
    print(f"  ||s_t||,  ||s_{{t+1}}||,  ||pred||  = "
          f"{avg['norm_st']:.3f}, {avg['norm_stp1']:.3f}, {avg['norm_pred']:.3f}")
    print(f"  per-feature std s_t, s_{{t+1}}, pred = "
          f"{avg['std_st']:.4f}, {avg['std_stp1']:.4f}, {avg['std_pred']:.4f}")
    if min(avg['std_st'], avg['std_stp1']) < 1e-3:
        print('  -> very low encoder-side feature std: possible representation collapse.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cinebrain_root', default='data/CineBrain')
    ap.add_argument('--n_batches', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--n_subjects', type=int, default=1,
                    help='use first N subjects from the training set for the probe')
    args = ap.parse_args()
    probe(args)


if __name__ == '__main__':
    main()
