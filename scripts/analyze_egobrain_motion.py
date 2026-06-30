"""Characterise the per-anchor visual MOTION distribution of EgoBrain across
representation spaces (patch grid tokens + raw pixels) and metrics (l1 + cos).

For each of the 4 signals it reports how concentrated motion is (percentiles,
gini) and how much of a uniform sampler's budget lands on near-static frames.
It then measures CROSS-SIGNAL agreement (rank correlation on the same anchors)
to answer whether pixel motion — which also captures head egomotion + lighting
— ranks dynamic moments the same way the semantic patch tokens do. Reads the
precomputed _motioncache (scripts/build_egobrain_motion_cache.py).

  conda run -n cbramod python -m scripts.analyze_egobrain_motion --step_slots 5
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from datasets.egobrain_motion import (  # noqa: E402
    load_or_compute_motion, motion_cache_path)

EMB_DEFAULT = ('data/EgoBrain/cache_embeddings_grid_'
               'facebook_dinov2-base_g0.2_sz224')
FRM_DEFAULT = ('data/EgoBrain/cache_frames_grid_'
               'facebook_dinov2-base_g0.2_sz224')
SIGNALS = [('patch', 'l1'), ('patch', 'cos'), ('pixel', 'l1'), ('pixel', 'cos')]


def _gini(x):
    x = np.sort(x[np.isfinite(x)].astype(np.float64))
    x = x - min(x.min(), 0.0)
    n = x.size
    if n == 0 or x.sum() == 0:
        return float('nan')
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def _budget_row(allv, alphas=(0.0, 0.5, 1.0, 2.0), tops=(50, 25, 10, 5, 1)):
    order = np.argsort(allv); ranks = np.empty_like(order)
    ranks[order] = np.arange(allv.size)
    topfrac = 1.0 - ranks / allv.size
    out = {}
    for a in alphas:
        w = np.power(np.clip(allv, 0, None), a); w = w / w.sum()
        out[a] = [w[topfrac <= t / 100.0].sum() * 100 for t in tops]
    return out, tops


def _spearman(a, b):
    """Rank correlation on the common-finite entries (subsampled to 200k)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size > 200_000:
        idx = np.linspace(0, a.size - 1, 200_000).astype(int)
        a, b = a[idx], b[idx]
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb_grid_dir', default=EMB_DEFAULT)
    ap.add_argument('--frames_grid_dir', default=FRM_DEFAULT)
    ap.add_argument('--step_slots', type=int, default=5)
    ap.add_argument('--subjects', default='all')
    ap.add_argument('--corr_subjects', type=int, default=8,
                    help='# subjects pooled for cross-signal rank correlation')
    ap.add_argument('--out_dir', default='outputs/motion')
    args = ap.parse_args()

    if args.subjects.lower() == 'all':
        subs = sorted(re.match(r'^(P\d{4})\.h5$', f).group(1)
                      for f in os.listdir(args.emb_grid_dir)
                      if re.match(r'^P\d{4}\.h5$', f))
    else:
        subs = [s.strip() for s in args.subjects.split(',') if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    # Only analyse subjects whose cache for ALL signals exists (others still
    # building) — report which are skipped.
    def ready(sub):
        return all(os.path.exists(motion_cache_path(
            args.emb_grid_dir, sub, sp, mt, args.step_slots))
            for sp, mt in SIGNALS)
    ready_subs = [s for s in subs if ready(s)]
    print(f'{len(ready_subs)}/{len(subs)} subjects have all 4 signals cached: '
          f'{",".join(ready_subs)}')
    if not ready_subs:
        print('nothing cached yet; run scripts.build_egobrain_motion_cache first')
        return

    pooled = {}
    for sp, mt in SIGNALS:
        vals = []
        for sub in ready_subs:
            m = load_or_compute_motion(args.emb_grid_dir, sub, args.step_slots,
                                       mt, sp, args.frames_grid_dir)
            if m is not None:
                vals.append(m[np.isfinite(m)])
        pooled[(sp, mt)] = np.concatenate(vals)

    print('\n=== per-signal distribution (pooled) ===')
    qs = [10, 25, 50, 75, 90, 95, 99]
    hdr = f"{'signal':>10} {'n':>10} " + ' '.join(f'p{q}'.rjust(8) for q in qs) \
          + f" {'gini':>6}"
    print(hdr)
    for sp, mt in SIGNALS:
        v = pooled[(sp, mt)]
        print(f"{sp+'/'+mt:>10} {v.size:>10,} " +
              ' '.join(f'{np.percentile(v,q):8.3f}' for q in qs) +
              f" {_gini(v):>6.3f}")

    print('\n=== budget concentration: P(draw in top-q% dynamic), motion^alpha ===')
    for sp, mt in SIGNALS:
        rows, tops = _budget_row(pooled[(sp, mt)])
        print(f'-- {sp}/{mt} --   ' + 'top: ' +
              ' '.join(f'{t}%'.rjust(6) for t in tops))
        for a, r in rows.items():
            print(f'   alpha={a:>3}   ' + ' '.join(f'{x:6.1f}' for x in r))

    # ---- cross-signal rank correlation on a common subject subset ----
    corr_subs = ready_subs[:args.corr_subjects]
    print(f'\n=== cross-signal Spearman (pooled over {len(corr_subs)} subjects, '
          f'common anchors) ===')
    arrs = {}
    for sp, mt in SIGNALS:
        per = [load_or_compute_motion(args.emb_grid_dir, s, args.step_slots, mt,
                                      sp, args.frames_grid_dir)
               for s in corr_subs]
        arrs[(sp, mt)] = np.concatenate(per)
    keys = SIGNALS
    M = np.zeros((4, 4))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = 1.0 if i == j else _spearman(arrs[ki], arrs[kj])
    names = [f'{sp[:2]}/{mt}' for sp, mt in keys]
    print('            ' + ' '.join(n.rjust(9) for n in names))
    for i, n in enumerate(names):
        print(f'{n:>10}  ' + ' '.join(f'{M[i,j]:9.3f}' for j in range(4)))

    # ---- figure: 4 histograms + correlation heatmap ----
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 5, figsize=(24, 4))
        for c, (sp, mt) in enumerate(SIGNALS):
            v = pooled[(sp, mt)]; hi = np.percentile(v, 99.5)
            ax[c].hist(np.clip(v, 0, hi), bins=100, color='steelblue')
            ax[c].axvline(np.median(v), color='k', ls='--', lw=1)
            ax[c].set_title(f'{sp}/{mt}  gini={_gini(v):.2f}\nmed={np.median(v):.3g}')
            ax[c].set_xlabel('motion'); ax[c].set_ylabel('# anchors')
        im = ax[4].imshow(M, vmin=0, vmax=1, cmap='viridis')
        ax[4].set_xticks(range(4)); ax[4].set_xticklabels(names, rotation=45)
        ax[4].set_yticks(range(4)); ax[4].set_yticklabels(names)
        for i in range(4):
            for j in range(4):
                ax[4].text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                           color='w' if M[i, j] < 0.6 else 'k', fontsize=8)
        ax[4].set_title('cross-signal Spearman')
        plt.colorbar(im, ax=ax[4], fraction=0.046)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f'motion_multi_d{args.step_slots}.png')
        fig.savefig(out, dpi=110); print(f'\nsaved {out}')
    except Exception as e:                       # noqa: BLE001
        print(f'[plot skipped] {e}')


if __name__ == '__main__':
    main()
