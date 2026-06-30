"""What does each motion SPACE actually capture? Pixel motion folds in head
egomotion + lighting; patch (DINOv2) motion is more semantic (hand/object).
This montage shows, for one subject, anchors where the two DISAGREE:

  * row A: high pixel, low patch  -> camera/light change, scene semantics steady
  * row B: high patch, low pixel  -> semantic/object change w/o big pixel shift
  * row C: high both              -> agreed strong motion
  * row D: low both               -> agreed static

Each cell shows frame[k] and frame[k+step]. Uses standardized (per-subject
z-scored) scores so "high/low" is comparable across the two spaces.

  conda run -n cbramod python -m scripts.motion_disagreement_montage --sub P0001
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from datasets.egobrain_motion import load_or_compute_motion  # noqa: E402

EMB = 'data/EgoBrain/cache_embeddings_grid_facebook_dinov2-base_g0.2_sz224'
FRM = 'data/EgoBrain/cache_frames_grid_facebook_dinov2-base_g0.2_sz224'


def _z(x):
    m = np.isfinite(x); z = np.full_like(x, np.nan)
    z[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-8)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub', default='P0001')
    ap.add_argument('--metric', default='cos', choices=['cos', 'l1'])
    ap.add_argument('--step_slots', type=int, default=5)
    ap.add_argument('--ncol', type=int, default=6)
    args = ap.parse_args()

    pa = load_or_compute_motion(EMB, args.sub, args.step_slots, args.metric, 'patch')
    px = load_or_compute_motion(EMB, args.sub, args.step_slots, args.metric, 'pixel',
                                frames_grid_dir=FRM)
    za, zx = _z(pa), _z(px)
    fin = np.isfinite(za) & np.isfinite(zx)
    idx = np.where(fin)[0]
    diff = zx - za                       # >0 pixel-heavy, <0 patch-heavy
    both = za + zx
    rows = {
        'A: pixel>>patch (egomotion/light)': idx[np.argsort(diff[idx])[-args.ncol:]],
        'B: patch>>pixel (semantic/object)': idx[np.argsort(diff[idx])[:args.ncol]],
        'C: high both (agreed motion)': idx[np.argsort(both[idx])[-args.ncol:]],
        'D: low both (agreed static)': idx[np.argsort(both[idx])[:args.ncol]],
    }
    import h5py
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = h5py.File(os.path.join(FRM, f'{args.sub}.h5'), 'r'); F = h['frames']
    nr, nc = 4, args.ncol * 2
    fig, axes = plt.subplots(nr, nc, figsize=(2 * nc, 2 * nr + 1))
    for r, (name, ks) in enumerate(rows.items()):
        for c, k in enumerate(ks):
            k = int(k)
            axes[r, 2*c].imshow(np.asarray(F[k])); axes[r, 2*c].axis('off')
            axes[r, 2*c].set_title(f'patch z={za[k]:+.1f}\npix z={zx[k]:+.1f}',
                                   fontsize=6)
            axes[r, 2*c+1].imshow(np.asarray(F[k+args.step_slots]))
            axes[r, 2*c+1].axis('off')
            axes[r, 2*c+1].set_title('+step', fontsize=6)
        axes[r, 0].text(-0.6, 0.5, name, transform=axes[r, 0].transAxes,
                        rotation=90, va='center', fontsize=9)
    h.close()
    fig.suptitle(f'{args.sub}  motion-space disagreement ({args.metric})',
                 fontsize=12)
    fig.tight_layout()
    out = f'outputs/motion/disagreement_{args.sub}_{args.metric}.png'
    fig.savefig(out, dpi=95); print(f'saved {out}')


if __name__ == '__main__':
    main()
