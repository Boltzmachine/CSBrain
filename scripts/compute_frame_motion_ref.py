"""Precompute the DATASET-LEVEL mean per-patch motion used to normalise the
per-patch weighting of the dense frame-prediction loss (``--wm_frame_motion_alpha``;
see models/world_model.py ``_motion_weight``).

Why hard-code it: the loss weight is ``w = clip(motion/ref, floor, inf)^alpha`` with
``motion[b,τ,p] = mean_d |s_tgt[b,τ,p] - s_anchor[b,p]|``. If ``ref`` is the PER-BATCH
mean motion it jitters step-to-step (batch composition + ``--egobrain_motion_resample``
both move it), so the floor threshold ``floor*ref`` — which decides what counts as a
"static" patch — drifts during training. Replacing it with a fixed dataset constant
makes the weighting deterministic. This script measures that constant.

What it measures (matches the training loss exactly):
  * anchor  = window-0 frame grid = grid slot ``k``   (orientation 0)
  * targets = windows 1..H         = slots ``k + τ·train_step``, τ=1..H
    with ``train_step = round(stride_s / frame_grid_s)`` (=1 for the 0.2s/0.2s run)
  * motion(k, τ) = mean over (patch, feature) of ``|grid[k+τ·train_step] - grid[k]|``
    == ``datasets.egobrain_motion._pair_dist`` L1 (per-token mean-abs), i.e. the
    per-patch ``motion`` averaged over patches.
The reference is the mean of ``motion(k, τ)`` over all valid anchors ``k``, all
horizons ``τ=1..H`` and all subjects — the UNCONDITIONAL mean (independent of the
resample knobs, so the constant stays valid if you retune resample alpha).

Default is a Monte-Carlo estimate (random anchors per subject) — the reference only
sets a floor threshold, so a few thousand samples give a stable mean. Pass
``--samples_per_subject 0`` for the exact full-stream pass (reads every slot; a
cluster-scale job over the ~1.4 TB dinov2 cache).

  conda run -n cbramod python -m scripts.compute_frame_motion_ref \
      --emb_grid_dir data/EgoBrain/cache_embeddings_grid_facebook_dinov2-base_g0.2_sz224 \
      --stride_s 0.2 --frame_grid_s 0.2 --max_horizon 5 --samples_per_subject 400
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

EMB_DEFAULT = ('data/EgoBrain/cache_embeddings_grid_'
               'facebook_dinov2-base_g0.2_sz224')


def _subject_stats(arg):
    """Per-subject (sum, count, per-step sum/count, sampled values) for the
    L1 motion between anchor slot k and slots k+step for each step in ``steps``.
    Returns a dict; a random subset of motion values is kept for percentiles."""
    sub, path, steps, n_samples, seed, keep = arg
    import h5py
    maxstep = max(steps)
    rng = np.random.default_rng(seed)
    tot_sum = 0.0
    tot_cnt = 0
    step_sum = {s: 0.0 for s in steps}
    step_cnt = {s: 0 for s in steps}
    vals = []
    t = time.time()
    with h5py.File(path, 'r') as h:
        ds = h['grid']                       # (n, 2, P, d)
        n = ds.shape[0]
        has = np.asarray(h['has_image'][:]).astype(bool)
        k_hi = n - maxstep - 1               # last anchor whose farthest target exists
        if k_hi < 0:
            return {'sub': sub, 'status': 'too_short', 'dt': 0.0,
                    'tot_sum': 0.0, 'tot_cnt': 0,
                    'step_sum': step_sum, 'step_cnt': step_cnt, 'vals': []}
        if n_samples and n_samples > 0:
            anchors = rng.integers(0, k_hi + 1, size=int(n_samples))
        else:
            anchors = np.arange(0, k_hi + 1)          # exact: every anchor
        for k in anchors:
            k = int(k)
            # ONE contiguous read of slots [k, k+maxstep] (orientation 0), so all
            # H horizons are computed from a single decompressed block.
            buf = np.asarray(ds[k:k + maxstep + 1, 0], dtype=np.float32)  # (maxstep+1,P,d)
            a = buf[0]
            for s in steps:
                if not (has[k] and has[k + s]):
                    continue
                d = float(np.abs(buf[s] - a).mean())   # mean over (P, d) == _pair_dist L1
                tot_sum += d; tot_cnt += 1
                step_sum[s] += d; step_cnt[s] += 1
                if len(vals) < keep and rng.random() < 0.3:
                    vals.append(d)
    return {'sub': sub, 'status': 'ok', 'dt': time.time() - t,
            'tot_sum': tot_sum, 'tot_cnt': tot_cnt,
            'step_sum': step_sum, 'step_cnt': step_cnt, 'vals': vals}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb_grid_dir', default=EMB_DEFAULT)
    p.add_argument('--subjects', default='all')
    p.add_argument('--stride_s', type=float, default=0.2,
                   help='EEG window stride (must match the training run)')
    p.add_argument('--frame_grid_s', type=float, default=0.2,
                   help='frame grid snap in seconds (the g<..> in the cache dir name)')
    p.add_argument('--max_horizon', type=int, default=5,
                   help='H: number of future frame steps (== wrapper max_horizon)')
    p.add_argument('--samples_per_subject', type=int, default=400,
                   help='random anchors per subject; 0 = exact (every anchor)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    train_step = int(round(args.stride_s / args.frame_grid_s))
    assert train_step >= 1, (
        f"stride_s/frame_grid_s = {args.stride_s}/{args.frame_grid_s} -> "
        f"train_step={train_step} < 1")
    steps = [tau * train_step for tau in range(1, args.max_horizon + 1)]

    if args.subjects.lower() == 'all':
        subs = sorted(re.match(r'^(P\d{4})\.h5$', f).group(1)
                      for f in os.listdir(args.emb_grid_dir)
                      if re.match(r'^P\d{4}\.h5$', f))
    else:
        subs = [s.strip() for s in args.subjects.split(',') if s.strip()]

    print(f"emb_grid_dir : {args.emb_grid_dir}", flush=True)
    print(f"subjects     : {len(subs)}  ({subs[0]}..{subs[-1]})", flush=True)
    print(f"train_step   : {train_step} slots  (stride {args.stride_s}s / "
          f"grid {args.frame_grid_s}s)", flush=True)
    print(f"horizons     : τ=1..{args.max_horizon} -> slot steps {steps}", flush=True)
    print(f"sampling     : {'EXACT (all anchors)' if args.samples_per_subject == 0 else str(args.samples_per_subject)+' anchors/subject'}", flush=True)

    tasks = [(sub, os.path.join(args.emb_grid_dir, f'{sub}.h5'), steps,
              args.samples_per_subject, args.seed + i, 4000)
             for i, sub in enumerate(subs)]
    tot_sum = 0.0; tot_cnt = 0
    step_sum = {s: 0.0 for s in steps}; step_cnt = {s: 0 for s in steps}
    all_vals = []
    t0 = time.time()
    with Pool(min(args.workers, len(tasks))) as pool:
        for i, r in enumerate(pool.imap_unordered(_subject_stats, tasks)):
            tot_sum += r['tot_sum']; tot_cnt += r['tot_cnt']
            for s in steps:
                step_sum[s] += r['step_sum'][s]; step_cnt[s] += r['step_cnt'][s]
            all_vals.extend(r['vals'])
            print(f"[{i+1}/{len(tasks)}] {r['sub']}: {r['status']} "
                  f"n={r['tot_cnt']} ({r['dt']:.0f}s)", flush=True)

    if tot_cnt == 0:
        print("no valid (anchor, step) pairs — check the cache dir / config",
              flush=True)
        sys.exit(1)

    ref = tot_sum / tot_cnt
    per_step = [step_sum[s] / step_cnt[s] if step_cnt[s] else float('nan')
                for s in steps]
    print("\n" + "=" * 64, flush=True)
    print(f"pairs sampled            : {tot_cnt:,}", flush=True)
    print(f"per-horizon mean L1 motion (slots -> mean):", flush=True)
    for s, m in zip(steps, per_step):
        print(f"    step {s:2d} ({s*args.frame_grid_s:.1f}s): {m:.6f}   (n={step_cnt[s]:,})",
              flush=True)
    if all_vals:
        v = np.asarray(all_vals)
        pcts = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"value percentiles (5/25/50/75/95): "
              f"{'/'.join(f'{x:.4f}' for x in pcts)}", flush=True)
    print("=" * 64, flush=True)
    # The loss normalises each horizon by its OWN mean (motion grows with the
    # horizon), so the constant is PER-STEP. Index by τ=1..H.
    tup = ', '.join(f'{m:.4f}' for m in per_step)
    print(f"\nPER-HORIZON MOTION REFERENCE (paste into models/world_model.py):",
          flush=True)
    print(f"    EGOBRAIN_FRAME_MOTION_REF_PER_STEP = ({tup})", flush=True)
    print(f"\n(scalar mean over all steps = {ref:.4f}; computed in "
          f"{time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
