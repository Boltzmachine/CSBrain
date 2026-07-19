"""Reusable integrity checker for a preprocessed downstream LMDB.

Verifies the invariants shared by every CSBrain finetune dataset:
  * __keys__ has train/val/test and they are non-empty + key-disjoint
  * cross-subject: no subject-prefix appears in more than one split
  * every sample is (C, n_win, 200) float32, finite, plausible uV/100 range
  * every label is an int in [0, n_classes)
  * ch_names length == C, ch_coords is (C, 3); reports NaN-coord channels
  * every record shares the same channel set

Usage:
  python scripts/verify_lmdb.py <lmdb_dir> --channels 60 --classes 7 --windows 4
"""
import argparse
import pickle
import re
from collections import Counter

import lmdb
import numpy as np


def subject_of(key):
    # 'subject7-12' -> 'subject7'; 'A08T-3-5' -> 'A08T'; 'S001R04-1' -> 'S001'
    m = re.match(r'(subject\d+|S\d+|A\d+[ET]?)', key)
    return m.group(1) if m else key.split('-')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lmdb_dir')
    ap.add_argument('--channels', type=int, required=True)
    ap.add_argument('--classes', type=int, required=True)
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--sample_n', type=int, default=200,
                    help='how many records to deep-check per split')
    args = ap.parse_args()

    db = lmdb.open(args.lmdb_dir, readonly=True, lock=False)
    with db.begin() as txn:
        keys = pickle.loads(txn.get(b'__keys__'))

    ok = True
    print(f'== {args.lmdb_dir} ==')
    print('splits:', {k: len(v) for k, v in keys.items()})

    # 1. non-empty + disjoint keys
    all_keys = [k for v in keys.values() for k in v]
    if len(all_keys) != len(set(all_keys)):
        print('  FAIL: duplicate keys across splits'); ok = False
    for s in ('train', 'val', 'test'):
        if not keys.get(s):
            print(f'  FAIL: split {s} empty'); ok = False

    # 2. cross-subject disjointness
    subj_by_split = {s: set(subject_of(k) for k in keys[s]) for s in keys}
    print('subjects/split:', {s: sorted(v) for s, v in subj_by_split.items()})
    for a in keys:
        for b in keys:
            if a < b and subj_by_split[a] & subj_by_split[b]:
                print(f'  FAIL: subject leak {a}&{b}: {subj_by_split[a] & subj_by_split[b]}')
                ok = False

    # 3/4/5. deep-check a sample of records per split
    ref_ch = None
    global_min, global_max = np.inf, -np.inf
    for s in keys:
        labels = []
        idxs = np.linspace(0, len(keys[s]) - 1, min(args.sample_n, len(keys[s]))).astype(int)
        with db.begin() as txn:
            for i in idxs:
                rec = pickle.loads(txn.get(keys[s][i].encode()))
                x, y = rec['sample'], rec['label']
                if x.shape != (args.channels, args.windows, 200):
                    print(f'  FAIL: {keys[s][i]} shape {x.shape} != {(args.channels, args.windows, 200)}'); ok = False
                if x.dtype != np.float32:
                    print(f'  FAIL: {keys[s][i]} dtype {x.dtype}'); ok = False
                if not np.isfinite(x).all():
                    print(f'  FAIL: {keys[s][i]} has non-finite samples'); ok = False
                if not (isinstance(y, (int, np.integer)) and 0 <= y < args.classes):
                    print(f'  FAIL: {keys[s][i]} label {y} out of [0,{args.classes})'); ok = False
                if len(rec['ch_names']) != args.channels:
                    print(f'  FAIL: {keys[s][i]} ch_names len {len(rec["ch_names"])}'); ok = False
                if np.asarray(rec['ch_coords']).shape != (args.channels, 3):
                    print(f'  FAIL: {keys[s][i]} ch_coords shape {np.asarray(rec["ch_coords"]).shape}'); ok = False
                if ref_ch is None:
                    ref_ch = list(rec['ch_names'])
                    nan_ch = [c for c, co in zip(ref_ch, np.asarray(rec['ch_coords'])) if not np.isfinite(co).all()]
                    print('NaN-coord channels:', nan_ch or 'none')
                elif list(rec['ch_names']) != ref_ch:
                    print(f'  FAIL: {keys[s][i]} channel set differs'); ok = False
                labels.append(int(y))
                xr = x / 100.0
                global_min = min(global_min, float(xr.min()))
                global_max = max(global_max, float(xr.max()))
        print(f'  [{s}] label dist (sampled):', dict(sorted(Counter(labels).items())))

    print(f'sample/100 range: [{global_min:.3f}, {global_max:.3f}]  (expect ~O(0.1-10))')
    print('RESULT:', 'PASS' if ok else 'FAIL')
    db.close()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
