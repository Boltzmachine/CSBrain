"""Precompute per-anchor motion scores for every EgoBrain video subject, in
both representation spaces (patch grid tokens + raw pixels) and both metrics
(l1 + cos), caching to <emb_grid_dir>/_motioncache/. Parallel over
(subject, space) tasks — the cost is HDF5 gzip decompression (CPU-bound), so it
scales with cores.

  conda run -n cbramod python -m scripts.build_egobrain_motion_cache \
      --step_slots 5 --spaces patch,pixel --workers 12
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

from datasets.egobrain_motion import (  # noqa: E402
    compute_subject_motion_both, motion_cache_dir, motion_cache_path)

EMB_DEFAULT = ('data/EgoBrain/cache_embeddings_grid_'
               'facebook_dinov2-base_g0.2_sz224')
FRM_DEFAULT = ('data/EgoBrain/cache_frames_grid_'
               'facebook_dinov2-base_g0.2_sz224')


def _task(arg):
    sub, space, emb_dir, frm_dir, step, block, overwrite = arg
    src = (os.path.join(frm_dir, f'{sub}.h5') if space == 'pixel'
           else os.path.join(emb_dir, f'{sub}.h5'))
    if not os.path.exists(src):
        return (sub, space, 'no_src', 0.0)
    paths = {m: motion_cache_path(emb_dir, sub, space, m, step)
             for m in ('l1', 'cos')}
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        return (sub, space, 'skip', 0.0)
    t = time.time()
    res = compute_subject_motion_both(src, step, space, block=block)
    os.makedirs(motion_cache_dir(emb_dir), exist_ok=True)
    for m, p in paths.items():
        tmp = p + '.tmp.npy'
        np.save(tmp, res[m])
        os.replace(tmp, p)
    return (sub, space, 'ok', time.time() - t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb_grid_dir', default=EMB_DEFAULT)
    p.add_argument('--frames_grid_dir', default=FRM_DEFAULT)
    p.add_argument('--subjects', default='all')
    p.add_argument('--spaces', default='patch,pixel')
    p.add_argument('--step_slots', type=int, default=5)
    p.add_argument('--block', type=int, default=3000)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.subjects.lower() == 'all':
        subs = sorted(re.match(r'^(P\d{4})\.h5$', f).group(1)
                      for f in os.listdir(args.emb_grid_dir)
                      if re.match(r'^P\d{4}\.h5$', f))
    else:
        subs = [s.strip() for s in args.subjects.split(',') if s.strip()]
    spaces = [s.strip() for s in args.spaces.split(',') if s.strip()]

    tasks = [(sub, sp, args.emb_grid_dir, args.frames_grid_dir, args.step_slots,
              args.block, args.overwrite)
             for sp in spaces for sub in subs]
    print(f'{len(tasks)} tasks ({len(subs)} subjects x {len(spaces)} spaces), '
          f'{args.workers} workers, step={args.step_slots}', flush=True)
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for sub, sp, status, dt in pool.imap_unordered(_task, tasks):
            done += 1
            print(f'[{done}/{len(tasks)}] {sub} {sp}: {status} ({dt:.0f}s)',
                  flush=True)
    print(f'all done in {time.time() - t0:.0f}s -> '
          f'{motion_cache_dir(args.emb_grid_dir)}', flush=True)


if __name__ == '__main__':
    main()
