"""Plot the sampled-anchor motion distribution BEFORE (uniform) vs AFTER the
motion-weighted resampling, using the EXACT per-subject CDF the dataset builds.
Encoder-agnostic (DINOv2 or V-JEPA 2) via --vision_encoder + cache dirs.

  conda run -n cbramod python -m scripts.plot_motion_before_after \
      --vision_encoder facebook/vjepa2-vitl-fpc64-256 \
      --subjects P0001,P0005,P0015,P0017 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _frozen_thresh(metric):
    return 0.05 if metric == 'cos' else None      # cos: <0.05 ~ frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vision_encoder', default='facebook/dinov2-base')
    ap.add_argument('--data_dir', default='data/EgoBrain')
    ap.add_argument('--subjects', default='P0001,P0005,P0015,P0017')
    ap.add_argument('--space', default='patch', choices=['patch', 'pixel', 'cls'])
    ap.add_argument('--metric', default='cos', choices=['cos', 'l1'])
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--cap_pct', type=float, default=99.0)
    ap.add_argument('--floor_mix', type=float, default=0.1)
    ap.add_argument('--frozen', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datasets.egobrain_dataset import EgoBrainDataset, _frame_size_for
    from datasets.egobrain_motion import load_or_compute_motion

    subs = [s.strip() for s in args.subjects.split(',') if s.strip()]
    fs = _frame_size_for(args.vision_encoder)
    enc_slug = args.vision_encoder.replace('/', '_')
    emb_dir = os.path.join(args.data_dir,
                           f'cache_embeddings_grid_{enc_slug}_g0.2_sz{fs}')
    frm_dir = os.path.join(args.data_dir,
                           f'cache_frames_grid_{enc_slug}_g0.2_sz{fs}')

    ds = EgoBrainDataset(
        data_dir=args.data_dir, subjects=subs, in_dim=200, n_windows=2,
        window_s=1.0, stride_s=1.0, clip_s=4.0, erp_latency_s=-0.15,
        max_channels=32, vision_encoder=args.vision_encoder, load_frames=True,
        use_frame_grid=True, use_grid_embeddings=False, frame_grid_dir=frm_dir,
        motion_resample=True, motion_resample_alpha=args.alpha,
        motion_resample_space=args.space, motion_resample_metric=args.metric,
        motion_resample_cap_pct=args.cap_pct,
        motion_resample_floor_mix=args.floor_mix, motion_emb_dir=emb_dir)

    n = len(subs)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    frmd = frm_dir if args.space == 'pixel' else None
    for ax, sub in zip(axes, subs):
        mot = load_or_compute_motion(emb_dir, sub, 5, args.metric, args.space,
                                     frames_grid_dir=frmd)
        k_min, k_max = ds._anchor_k_bounds(sub)
        ent = ds._anchor_cdf.get(sub)
        if ent is None:
            ax.set_title(f'{sub}: no motion weights'); continue
        k0, cdf = ent
        p_after = np.diff(np.concatenate([[0.0], cdf]))
        ks = np.arange(k_min, k_max + 1)
        m = mot[ks]; fin = np.isfinite(m); mv = m[fin]
        wb = np.full(mv.shape, 1.0); wb /= wb.sum()
        wa = p_after[fin].copy(); wa /= wa.sum()
        hi = np.nanpercentile(mot, 99.5)
        bins = np.linspace(0, hi, 70)
        ax.hist(mv, bins=bins, weights=wb, alpha=0.55, color='gray',
                label='before (uniform)')
        ax.hist(mv, bins=bins, weights=wa, alpha=0.55, color='crimson',
                label=f'after (motion^{args.alpha:g})')

        def wmed(x, w):
            o = np.argsort(x); xc = x[o]
            wc = np.cumsum(w[o]) / w.sum()
            return xc[np.searchsorted(wc, 0.5)]
        mb, ma = wmed(mv, wb), wmed(mv, wa)
        fb = wb[mv < args.frozen].sum(); fa = wa[mv < args.frozen].sum()
        ax.axvline(mb, color='k', ls='--', lw=1)
        ax.axvline(ma, color='crimson', ls='--', lw=1)
        ax.set_title(f'{sub}  median {mb:.3f}->{ma:.3f}   '
                     f'frozen(<{args.frozen:g}) {fb*100:.0f}%->{fa*100:.0f}%',
                     fontsize=10)
        ax.set_xlabel(f'{args.space}/{args.metric} motion (1s step)')
        ax.set_ylabel('sampling probability'); ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(f'{args.vision_encoder}: sampled-anchor motion BEFORE vs AFTER '
                 f'(alpha={args.alpha:g}, floor={args.floor_mix:g}, '
                 f'cap p{args.cap_pct:g})', fontsize=12)
    fig.tight_layout()
    out = args.out or (f'outputs/motion/before_after_{enc_slug}_'
                       f'{args.space}_{args.metric}.png')
    fig.savefig(out, dpi=115); print('saved', out)


if __name__ == '__main__':
    main()
