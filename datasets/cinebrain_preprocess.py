"""Offline preprocessor for the CineBrain EEG dataset.

Running this once converts the per-segment 1000 Hz ``.npy`` files into
per-clip, already-filtered/resampled/clipped tensors. The training loader
(:class:`datasets.cinebrain_dataset.CineBrainDataset`) reads these cached
files directly, avoiding the MNE filter/notch/resample cost — which is
the CPU bottleneck that pushes an epoch from ~30 min to 6+ h.

Cache layout (deterministic from ``fs_out``):
    <cache_dir>/<subject>/<clip_idx>.npy     # (C, fs_out*4) float32 µV

Usage (one-time; idempotent — existing cache files are skipped):
    conda run -n cbramod python -m datasets.cinebrain_preprocess \
        --data_dir data/CineBrain \
        --subjects sub-0001,sub-0002,sub-0003,sub-0004,sub-0005,sub-0006 \
        --num_workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from typing import Sequence

import numpy as np
from tqdm import tqdm

from datasets.cinebrain_dataset import preprocess_segment


# Worker state initialised by each process before the pool starts.
_WORKER_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    _WORKER_CFG.update(cfg)


def _preprocess_one(task) -> tuple[str, int, str]:
    """Preprocess a single (subject, clip_idx) and write to cache.

    Returns (subject, clip_idx, status) — status is 'ok', 'skip', or an
    error string. Errors are returned rather than raised so a single
    corrupt clip doesn't abort the whole subject.
    """
    sub, c = task
    cfg = _WORKER_CFG
    cache_file = os.path.join(cfg['cache_dir'], sub, f'{c}.npy')
    if os.path.exists(cache_file) and not cfg['overwrite']:
        return sub, c, 'skip'

    eeg_dir = os.path.join(cfg['data_dir'], sub, cfg['eeg_subdir'])
    chunks = []
    for j in range(5):
        p = os.path.join(eeg_dir, f'{c * 5 + j}.npy')
        if not os.path.exists(p):
            return sub, c, f'missing {p}'
        arr = np.load(p)
        chunks.append(arr[:cfg['n_eeg_channels']])
    raw = np.concatenate(chunks, axis=-1)                # (C, fs_in*4)
    ts = preprocess_segment(raw, cfg['fs_in'], cfg['fs_out'])
    ts = ts * 1e6 if np.abs(ts).max() < 1.0 else ts
    np.clip(ts, -300, 300, out=ts)
    ts = ts.astype(np.float32)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    # Write to a tmp file with an open handle (np.save auto-appends '.npy'
    # to string paths, which breaks the rename trick) then atomically
    # rename so a crash mid-write can't leave a truncated cache file.
    tmp = cache_file + '.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, ts, allow_pickle=False)
    os.replace(tmp, cache_file)
    return sub, c, 'ok'


def _enumerate_clips(data_dir: str, subjects: Sequence[str],
                     eeg_subdir: str) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for sub in subjects:
        eeg_dir = os.path.join(data_dir, sub, eeg_subdir)
        if not os.path.isdir(eeg_dir):
            print(f'[skip] {eeg_dir} does not exist', file=sys.stderr)
            continue
        files = [f for f in os.listdir(eeg_dir)
                 if f.endswith('.npy') and f[:-4].isdigit()]
        if not files:
            print(f'[skip] no .npy files under {eeg_dir}', file=sys.stderr)
            continue
        idx_max = max(int(f[:-4]) for f in files)
        n_clips = (idx_max + 1) // 5
        for c in range(n_clips):
            if all(os.path.exists(os.path.join(
                    eeg_dir, f'{c * 5 + j}.npy')) for j in range(5)):
                tasks.append((sub, c))
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default='data/CineBrain')
    p.add_argument('--cache_dir', default=None,
                   help='defaults to <data_dir>/cache_eeg_<fs_out>hz')
    p.add_argument('--subjects', required=True,
                   help='comma-separated (e.g. sub-0001,sub-0002)')
    p.add_argument('--eeg_subdir', default='eeg_02')
    p.add_argument('--n_eeg_channels', type=int, default=64)
    p.add_argument('--fs_in', type=int, default=1000)
    p.add_argument('--fs_out', type=int, default=200)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--overwrite', action='store_true',
                   help='re-compute and replace existing cache files')
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    cache_dir = (args.cache_dir
                 or os.path.join(args.data_dir, f'cache_eeg_{args.fs_out}hz'))

    tasks = _enumerate_clips(args.data_dir, subjects, args.eeg_subdir)
    print(f'found {len(tasks)} clips across {len(subjects)} subject(s); '
          f'cache → {cache_dir}')

    cfg = dict(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        eeg_subdir=args.eeg_subdir,
        n_eeg_channels=args.n_eeg_channels,
        fs_in=args.fs_in,
        fs_out=args.fs_out,
        overwrite=args.overwrite,
    )

    counts = {'ok': 0, 'skip': 0, 'err': 0}
    if args.num_workers <= 1:
        _init_worker(cfg)
        for t in tqdm(tasks, mininterval=5):
            _, _, status = _preprocess_one(t)
            counts['ok' if status == 'ok' else
                   'skip' if status == 'skip' else 'err'] += 1
    else:
        # `maxtasksperchild` limits memory growth from MNE's filter cache.
        with mp.get_context('spawn').Pool(
                args.num_workers, initializer=_init_worker,
                initargs=(cfg,), maxtasksperchild=200) as pool:
            for sub, c, status in tqdm(
                    pool.imap_unordered(_preprocess_one, tasks, chunksize=8),
                    total=len(tasks), mininterval=5):
                if status == 'ok':
                    counts['ok'] += 1
                elif status == 'skip':
                    counts['skip'] += 1
                else:
                    counts['err'] += 1
                    print(f'[err] {sub} clip {c}: {status}', file=sys.stderr)

    print(f'done: ok={counts["ok"]} skip={counts["skip"]} err={counts["err"]}')


if __name__ == '__main__':
    main()
