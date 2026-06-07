"""Compute and persist per-subject Euclidean Alignment matrices.

The output is a single ``.pt`` sidecar file per dataset, stored beside the
existing EEG cache but with a distinct name (``ea_subject.pt``) so the
existing cache is never modified. To roll back EA, set
``--use_euclidean_alignment 0`` in the training script, or delete the
sidecar file.

Run one of:

    conda run -n cbramod python -m datasets.compute_ea egobrain
    conda run -n cbramod python -m datasets.compute_ea cinebrain
    conda run -n cbramod python -m datasets.compute_ea alljoined
    conda run -n cbramod python -m datasets.compute_ea physio

Each command iterates ALL subjects of that dataset and writes one ``(C, C)``
whitening matrix per subject. Per-subject trial counts are printed on
stderr so you can spot subjects with too few trials for a stable estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import Iterator, Optional

import numpy as np

from datasets.euclidean_alignment import (
    compute_ea_matrix, finalize_ea_matrix, save_ea_matrices,
)


# ---------------------------------------------------------------------------
# EgoBrain
# ---------------------------------------------------------------------------


def _egobrain_trials_for_subject(cache_dir: str, sub: str,
                                 keep_mask: np.ndarray,
                                 ) -> Iterator[np.ndarray]:
    sub_dir = os.path.join(cache_dir, sub)
    meta_path = os.path.join(sub_dir, 'clips.json')
    with open(meta_path) as f:
        meta = json.load(f)
    n_clips = meta['n_clips']
    for c in range(n_clips):
        arr = np.load(os.path.join(sub_dir, f'{c}.npy'))
        if keep_mask.shape[0] != arr.shape[0]:
            raise RuntimeError(
                f'{sub}: cached channels {arr.shape[0]} '
                f'vs keep_mask {keep_mask.shape[0]} mismatch')
        yield arr[keep_mask]


def _egobrain_compute(args: argparse.Namespace) -> None:
    from datasets.egobrain_dataset import _resolve_ch_coords
    cache_dir = args.cache_dir
    subjects = _scan_subjects(cache_dir, r'^P\d{4}$')
    print(f'[ea] EgoBrain: {len(subjects)} subjects, cache={cache_dir}')
    matrices = {}
    for sub in subjects:
        meta_path = os.path.join(cache_dir, sub, 'clips.json')
        with open(meta_path) as f:
            meta = json.load(f)
        kept_names, keep_mask, _ = _resolve_ch_coords(meta['ch_names'])
        n_ch = len(kept_names)
        if args.max_channels is not None and n_ch > args.max_channels:
            # Match the EgoBrainDataset's truncation logic.
            idx = np.where(keep_mask)[0][:args.max_channels]
            keep_mask = np.zeros_like(keep_mask)
            keep_mask[idx] = True
            n_ch = args.max_channels
        trials = _egobrain_trials_for_subject(cache_dir, sub, keep_mask)
        R, n, s = compute_ea_matrix(trials, n_ch, eps=args.eps,
                                    rescale=not args.no_rescale)
        matrices[sub] = R
        print(f'  {sub}: n_trials={n:>5}  s_raw={s:.2f}  '
              f'cond(R^-1/2)={np.linalg.cond(R):.2e}', file=sys.stderr)
    save_ea_matrices(matrices, args.out, meta={
        'dataset': 'egobrain', 'n_subjects': len(subjects),
        'n_channels': int(matrices[subjects[0]].shape[0]),
        'rescale_to_raw_rms': not args.no_rescale, 'eps': args.eps,
        'cache_dir': cache_dir,
    })
    print(f'[ea] wrote {args.out}')


# ---------------------------------------------------------------------------
# CineBrain
# ---------------------------------------------------------------------------


def _cinebrain_trials_for_subject(cache_dir: str, sub: str
                                  ) -> Iterator[np.ndarray]:
    sub_dir = os.path.join(cache_dir, sub)
    # CineBrain cache layout: <sub>/<clip_idx>.npy, no per-subject json.
    files = sorted(
        f for f in os.listdir(sub_dir)
        if f.endswith('.npy') and f[:-4].isdigit())
    for f in files:
        yield np.load(os.path.join(sub_dir, f))


def _cinebrain_compute(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir
    subjects = _scan_subjects(cache_dir, r'^sub-\d{4}$')
    print(f'[ea] CineBrain: {len(subjects)} subjects, cache={cache_dir}')
    n_ch = args.n_channels                              # 64 for biosemi64
    matrices = {}
    for sub in subjects:
        trials = _cinebrain_trials_for_subject(cache_dir, sub)
        R, n, s = compute_ea_matrix(trials, n_ch, eps=args.eps,
                                    rescale=not args.no_rescale)
        matrices[sub] = R
        print(f'  {sub}: n_trials={n:>5}  s_raw={s:.2f}',
              file=sys.stderr)
    save_ea_matrices(matrices, args.out, meta={
        'dataset': 'cinebrain', 'n_subjects': len(subjects),
        'n_channels': n_ch, 'rescale_to_raw_rms': not args.no_rescale,
        'eps': args.eps, 'cache_dir': cache_dir,
    })
    print(f'[ea] wrote {args.out}')


# ---------------------------------------------------------------------------
# Alljoined-1.6M — scan webdataset shards once, accumulate per subject.
# ---------------------------------------------------------------------------


def _alljoined_subject_from_key(key: str) -> Optional[str]:
    # Sample key shape: 'sub-01_sess-01_block-02_evt_00000000'
    m = re.match(r'(sub-\d+)', key)
    return m.group(1) if m else None


def _alljoined_compute(args: argparse.Namespace) -> None:
    import webdataset as wds
    shards = sorted(glob(args.shards))
    if not shards:
        raise FileNotFoundError(f'no shards matched {args.shards}')
    print(f'[ea] Alljoined: {len(shards)} shards, glob={args.shards}')
    n_ch = args.n_channels
    # Running per-subject accumulators (instead of caching trials, which
    # would dominate RAM on a 1.6M-sample corpus).
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    sq_sums: dict[str, float] = {}
    el_counts: dict[str, int] = {}
    ds = wds.WebDataset(shards, shardshuffle=False).decode()
    for i, s in enumerate(ds):
        sub = _alljoined_subject_from_key(s.get('__key__', ''))
        if sub is None:
            continue
        ts = np.asarray(s.get('timeseries.npy'), dtype=np.float64)
        if ts.ndim != 2 or ts.shape[0] != n_ch:
            continue
        if sub not in sums:
            sums[sub] = np.zeros((n_ch, n_ch), dtype=np.float64)
            counts[sub] = 0
            sq_sums[sub] = 0.0
            el_counts[sub] = 0
        sums[sub] += ts @ ts.T
        counts[sub] += 1
        sq_sums[sub] += float((ts * ts).sum())
        el_counts[sub] += ts.size
        if (i + 1) % args.print_every == 0:
            print(f'  scanned {i+1} samples across {len(sums)} subjects',
                  file=sys.stderr)
        if args.max_samples and (i + 1) >= args.max_samples:
            break
    print(f'[ea] scanned {i+1} samples total', file=sys.stderr)
    matrices: dict[str, np.ndarray] = {}
    for sub in sorted(sums.keys()):
        R_inv_sqrt, s_raw = finalize_ea_matrix(
            sums[sub], counts[sub], sq_sums[sub], el_counts[sub],
            n_channels=n_ch, eps=args.eps, rescale=not args.no_rescale)
        matrices[sub] = R_inv_sqrt
        print(f'  {sub}: n_trials={counts[sub]:>6}  s_raw={s_raw:.2f}',
              file=sys.stderr)
    save_ea_matrices(matrices, args.out, meta={
        'dataset': 'alljoined', 'n_subjects': len(matrices),
        'n_channels': n_ch, 'rescale_to_raw_rms': not args.no_rescale,
        'eps': args.eps, 'shards_glob': args.shards,
    })
    print(f'[ea] wrote {args.out}')


# ---------------------------------------------------------------------------
# PhysioNet (downstream)
# ---------------------------------------------------------------------------


def _physio_subject_from_key(key: str) -> Optional[str]:
    # Sample key shape: 'S001R04-1'
    m = re.match(r'(S\d{3})', key)
    return m.group(1) if m else None


def _physio_compute(args: argparse.Namespace) -> None:
    import lmdb, pickle
    env = lmdb.open(args.lmdb_dir, readonly=True, lock=False, readahead=True)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    print(f'[ea] PhysioNet: {args.lmdb_dir} (splits={splits})')
    with env.begin(write=False) as txn:
        keys_blob = pickle.loads(txn.get(b'__keys__'))
        all_keys = []
        for split in splits:
            all_keys.extend(keys_blob.get(split, []))
        if not all_keys:
            raise RuntimeError(
                f'no keys found in splits={splits}; available splits: '
                f'{list(keys_blob.keys())}')
        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        sq_sums: dict[str, float] = {}
        el_counts: dict[str, int] = {}
        n_ch_first = None
        for i, key in enumerate(all_keys):
            sub = _physio_subject_from_key(key)
            if sub is None:
                continue
            s = pickle.loads(txn.get(key.encode()))
            x = np.asarray(s['sample'], dtype=np.float64)
            # PhysioNet samples are (C, N_windows, T) — flatten to (C, T_total).
            if x.ndim == 3:
                x = x.reshape(x.shape[0], -1)
            elif x.ndim != 2:
                continue
            C = x.shape[0]
            if n_ch_first is None:
                n_ch_first = C
            if C != n_ch_first:
                continue
            if sub not in sums:
                sums[sub] = np.zeros((C, C), dtype=np.float64)
                counts[sub] = 0
                sq_sums[sub] = 0.0
                el_counts[sub] = 0
            sums[sub] += x @ x.T
            counts[sub] += 1
            sq_sums[sub] += float((x * x).sum())
            el_counts[sub] += x.size
            if (i + 1) % args.print_every == 0:
                print(f'  scanned {i+1}/{len(all_keys)} samples ',
                      f'across {len(sums)} subjects', file=sys.stderr)
    matrices: dict[str, np.ndarray] = {}
    for sub in sorted(sums.keys()):
        R_inv_sqrt, s_raw = finalize_ea_matrix(
            sums[sub], counts[sub], sq_sums[sub], el_counts[sub],
            n_channels=n_ch_first, eps=args.eps,
            rescale=not args.no_rescale)
        matrices[sub] = R_inv_sqrt
        print(f'  {sub}: n_trials={counts[sub]:>4}  s_raw={s_raw:.2f}',
              file=sys.stderr)
    save_ea_matrices(matrices, args.out, meta={
        'dataset': 'physio', 'n_subjects': len(matrices),
        'n_channels': n_ch_first, 'rescale_to_raw_rms': not args.no_rescale,
        'eps': args.eps, 'lmdb_dir': args.lmdb_dir,
    })
    print(f'[ea] wrote {args.out}')


# ---------------------------------------------------------------------------


def _scan_subjects(cache_dir: str, pattern: str) -> list[str]:
    rgx = re.compile(pattern)
    out = sorted(d for d in os.listdir(cache_dir)
                 if rgx.match(d) and
                 os.path.isdir(os.path.join(cache_dir, d)))
    if not out:
        raise FileNotFoundError(
            f'no subdirs match {pattern!r} under {cache_dir}')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='dataset', required=True)

    ego = sub.add_parser('egobrain')
    ego.add_argument('--cache_dir', default='data/EgoBrain/cache_eeg_200hz')
    ego.add_argument('--out',
                     default='data/EgoBrain/cache_eeg_200hz/ea_subject.pt')
    ego.add_argument('--max_channels', type=int, default=32)
    ego.add_argument('--eps', type=float, default=1e-6)
    ego.add_argument('--no_rescale', action='store_true')
    ego.set_defaults(func=_egobrain_compute)

    cine = sub.add_parser('cinebrain')
    cine.add_argument('--cache_dir', default='data/CineBrain/cache_eeg_200hz')
    cine.add_argument('--out',
                      default='data/CineBrain/cache_eeg_200hz/ea_subject.pt')
    cine.add_argument('--n_channels', type=int, default=64)
    cine.add_argument('--eps', type=float, default=1e-6)
    cine.add_argument('--no_rescale', action='store_true')
    cine.set_defaults(func=_cinebrain_compute)

    allj = sub.add_parser('alljoined')
    allj.add_argument('--shards',
                      default='data/cache/Alljoined-1.6M/*.tar')
    allj.add_argument('--out',
                      default='data/cache/Alljoined-1.6M/ea_subject.pt')
    allj.add_argument('--n_channels', type=int, default=32)
    allj.add_argument('--eps', type=float, default=1e-6)
    allj.add_argument('--no_rescale', action='store_true')
    allj.add_argument('--max_samples', type=int, default=0,
                      help='stop after this many samples (0 = full scan)')
    allj.add_argument('--print_every', type=int, default=50000)
    allj.set_defaults(func=_alljoined_compute)

    phys = sub.add_parser('physio')
    phys.add_argument('--lmdb_dir', default='data/preprocessed/physionet_mi')
    phys.add_argument('--out',
                      default='data/preprocessed/physionet_mi/ea_subject.pt')
    phys.add_argument('--splits', type=str, default='train,val,test',
                      help='comma-separated splits whose samples contribute '
                           'to each subject\'s EA estimate. Default uses ALL '
                           'splits because PhysioNet\'s split is cross-subject '
                           '(train: S001-S070, val/test: S071-S109): a '
                           'train-only matrix would leave val/test subjects '
                           'without any EA, creating a train/test distribution '
                           'shift WORSE than no EA. Per He & Wu 2020, this '
                           'second-moment-only "leak" of test EEG is standard '
                           'EA practice. Pass --splits train if you want to '
                           'restrict to a within-subject train/test split.')
    phys.add_argument('--eps', type=float, default=1e-6)
    phys.add_argument('--no_rescale', action='store_true')
    phys.add_argument('--print_every', type=int, default=2000)
    phys.set_defaults(func=_physio_compute)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
