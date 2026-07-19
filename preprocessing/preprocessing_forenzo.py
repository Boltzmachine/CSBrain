"""Preprocess Forenzo2024 (Continuous Pursuit MI, He lab / KiltHub 25360300)
into a PhysioNet-MI-consistent LMDB with BOTH a classification and a regression
target per window.

The task is CONTINUOUS: subjects do hand MI to steer a 2D cursor toward a
randomly moving target for 60 s per trial. We window each trial into 1-second
EEG segments and attach two targets:
  * regression: mean 2D cursor VELOCITY (cursorvel x,y) over the window
  * classification: 4-way intended DIRECTION derived from mean (target-cursor)
    -> 0 right / 1 up / 2 left / 3 down (dominant axis + sign)

Conventions mirror preprocessing_finemi.py:
  * resample 1000 Hz -> 200 Hz; 1-second (200-sample) windows -> (62, 1, 200)
  * common-average reference across the 62 EEG channels
  * standard_1005 coords (CB1/CB2 -> NaN, exactly like FineMI's 62-ch montage)
  * cross-subject split; micro-volt EEG storage so the loader /100 fits
Records hold BOTH 'label' (int 0..3) and 'vel' (float32 [2]); the classification
and regression loaders read the respective field from the same LMDB.

.mat files are MATLAB v7.3 (HDF5) -> parsed with h5py. Chance-level runs are
skipped. Heavy dataset (~130 GB): per-subject download -> parse -> delete,
with resumable incremental __keys__.
  usage: python preprocessing/preprocessing_forenzo.py --subjects all --delete-raw
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time
import urllib.request

os.environ.setdefault('MNE_DATA',
                      '/gpfs/radev/pi/ying_rex/wq44/CSBrain/data/raw/mne_data')
import pickle

import h5py
import lmdb
import numpy as np
from scipy import signal
import mne
mne.set_log_level('ERROR')

raw_dir = './data/raw/forenzo'
out_dir = './data/preprocessed/forenzo'
ARTICLE_ID = 25360300
SRC_FS = 1000
DST_FS = 200
WIN_MS = 1000                             # 1-second windows
N_CLASSES = 4                             # right/up/left/down
ARTIFACT_UV = 800.0                       # drop windows with max |amplitude|
                                          # above this (electrode pops/movement
                                          # in the continuous recording; ~17%)

# 62-channel Neuroscan montage (identical order to Forenzo channellabels ==
# FineMI). CB1/CB2 have no standard_1005 coords (-> NaN), like FineMI.
SELECTED_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2',
]

# Cross-subject split over 28 subjects (train 20 / val 4 / test 4). Process
# order front-loads one subject per split so the LMDB is smoke-testable early.
SPLIT = {
    'train': [f'S{n:02d}' for n in range(1, 21)],
    'val':   ['S21', 'S22', 'S23', 'S24'],
    'test':  ['S25', 'S26', 'S27', 'S28'],
}
SUBJECT_TO_SPLIT = {s: sp for sp, subs in SPLIT.items() for s in subs}
PROC_ORDER = (['S01', 'S21', 'S25']
              + [s for s in SUBJECT_TO_SPLIT if s not in ('S01', 'S21', 'S25')])


def _retry(fn, tries=8, base=15):
    """Retry fn on transient errors (figshare 503, network hiccups)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as ex:
            if i == tries - 1:
                raise
            wait = min(base * (2 ** i), 300)
            print(f'  download failed ({type(ex).__name__}: {ex}); '
                  f'retry {i + 1}/{tries} in {wait}s')
            time.sleep(wait)


def _refstr(h, ref):
    return ''.join(chr(c) for c in h[ref][()].ravel())


def channel_coords(ch_names):
    ch_config = 'standard_1005'
    montage = mne.channels.make_standard_montage(ch_config)
    pos = {k.lower(): v for k, v in montage.get_positions()['ch_pos'].items()}
    coords, unknown = [], []
    for ch in ch_names:
        key = ch.lower()
        if key in pos:
            coords.append(pos[key])
        else:
            coords.append((np.nan, np.nan, np.nan))
            unknown.append(ch)
    if unknown:
        print('  unrecognized channels (NaN coords):', unknown)
    return np.stack(coords, axis=0).astype(np.float32), ch_config


def _file_urls():
    a = json.loads(urllib.request.urlopen(
        f'https://api.figshare.com/v2/articles/{ARTICLE_ID}', timeout=60).read())
    return {f['name']: f['download_url'] for f in a['files']}


def parse_run(path):
    """Window one run .mat into (X (n,62,200) uV, y_cls (n,), y_vel (n,2), ch)."""
    with h5py.File(path, 'r') as h:
        e = h['eeg']
        data = np.asarray(e['data']).astype(np.float64)       # (nS, 62) uV
        ch = [_refstr(h, r) for r in np.asarray(e['channellabels']).ravel()]
        times = np.asarray(e['times']).ravel()                # ms, 1 kHz
        post = np.asarray(e['postimes']).ravel()              # ms, 25 Hz
        cvx = np.asarray(e['cursorvel']['x']).ravel()
        cvy = np.asarray(e['cursorvel']['y']).ravel()
        cpx = np.asarray(e['cursorpos']['x']).ravel()
        cpy = np.asarray(e['cursorpos']['y']).ravel()
        tpx = np.asarray(e['targetpos']['x']).ravel()
        tpy = np.asarray(e['targetpos']['y']).ravel()
        ev = e['event']
        lat = [float(h[r][()].ravel()[0]) for r in np.asarray(ev['latency']).ravel()]
        typ = [_refstr(h, r) for r in np.asarray(ev['type']).ravel()]

    # pair TrialStart -> TrialEnd (events alternate in order)
    starts = [lat[i] for i in range(len(typ)) if typ[i] == 'TrialStart']
    ends = [lat[i] for i in range(len(typ)) if typ[i] == 'TrialEnd']
    win = SRC_FS                                              # 1000 samples = 1 s

    X, ycls, yvel = [], [], []
    for st, en in zip(starts, ends):
        si = int(np.searchsorted(times, st))
        ei = int(np.searchsorted(times, en))
        w = si
        while w + win <= ei:
            seg = data[w:w + win, :]                          # (1000, 62)
            seg = signal.resample(seg, DST_FS, axis=0)        # (200, 62)
            seg = seg - seg.mean(axis=1, keepdims=True)       # common-avg ref
            if np.abs(seg).max() > ARTIFACT_UV:               # artifact reject
                w += win
                continue
            t0, t1 = times[w], times[w + win - 1]
            mask = (post >= t0) & (post <= t1)
            if mask.sum() >= 1:
                vx, vy = float(cvx[mask].mean()), float(cvy[mask].mean())
                dx = float((tpx[mask] - cpx[mask]).mean())
                dy = float((tpy[mask] - cpy[mask]).mean())
            else:
                vx = vy = dx = dy = 0.0
            cls = (0 if dx >= 0 else 2) if abs(dx) >= abs(dy) else (1 if dy >= 0 else 3)
            X.append(seg.T.astype(np.float32))                # (62, 200)
            ycls.append(cls)
            yvel.append([vx, vy])
            w += win                                          # non-overlapping
    if not X:
        return None
    return (np.stack(X), np.array(ycls, dtype=int),
            np.asarray(yvel, dtype=np.float32), ch)


def process_subject(subj, urls, db, dataset, delete_raw, ref):
    zip_path = os.path.join(raw_dir, f'{subj}.zip')
    if not os.path.exists(zip_path):
        print(f'  downloading {subj}.zip...')
        _retry(lambda: urllib.request.urlretrieve(urls[f'{subj}.zip'], zip_path))
    ext_dir = os.path.join(raw_dir, f'_{subj}')
    subprocess.run(['unzip', '-o', '-q', zip_path, '-d', ext_dir], check=True)
    mats = sorted(glob.glob(os.path.join(ext_dir, '**', '*.mat'), recursive=True))
    split = SUBJECT_TO_SPLIT[subj]
    n = 0
    for mp in mats:
        if 'chance' in os.path.basename(mp).lower():          # skip chance runs
            continue
        parsed = parse_run(mp)
        if parsed is None:
            continue
        X, ycls, yvel, ch = parsed
        if ref['ch'] is None:
            up = [c.upper() for c in ch]
            assert up == SELECTED_CHANNELS, f'channel mismatch {up[:5]}'
            ref['ch'] = SELECTED_CHANNELS
            ref['coords'], ref['config'] = channel_coords(SELECTED_CHANNELS)
        X = X.reshape(X.shape[0], 62, 1, DST_FS)
        base = os.path.splitext(os.path.basename(mp))[0]
        for i in range(X.shape[0]):
            key = f'{subj}-{base}-{i}'
            rec = {'sample': X[i], 'label': int(ycls[i]),
                   'vel': yvel[i], 'ch_names': ref['ch'],
                   'ch_coords': ref['coords'], 'ch_config': ref['config']}
            with db.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps(rec))
            dataset[split].append(key)
            n += 1
    print(f'subject {subj} [{split}] -> {n} windows from {len(mats)} runs')
    shutil.rmtree(ext_dir, ignore_errors=True)
    if delete_raw:
        os.remove(zip_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='all')
    ap.add_argument('--delete-raw', action='store_true')
    args = ap.parse_args()

    subjects = PROC_ORDER if args.subjects == 'all' else args.subjects.split(',')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    db = lmdb.open(out_dir, map_size=64 * 1024 ** 3)

    with db.begin() as txn:
        rk = txn.get(b'__keys__')
    dataset = pickle.loads(rk) if rk else {'train': [], 'val': [], 'test': []}
    done = {k.split('-')[0] for v in dataset.values() for k in v}

    urls = _retry(_file_urls)
    ref = {'ch': None, 'coords': None, 'config': None}
    for subj in subjects:
        if subj not in SUBJECT_TO_SPLIT or subj in done:
            print(f'skip {subj}'); continue
        process_subject(subj, urls, db, dataset, args.delete_raw, ref)
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(dataset))

    db.close()
    print('train/val/test sizes:',
          len(dataset['train']), len(dataset['val']), len(dataset['test']))


if __name__ == '__main__':
    main()
