"""Preprocess the Kaya et al. 2018 **5F** (five-finger MI) dataset into a
PhysioNet-MI-consistent LMDB.

NOTE: MOABB's ``Kaya2018`` loader exposes only the CLA (3-class) paradigm, NOT
5F. We therefore download the 5F ``.mat`` files directly from the figshare
collection (c.3917698) and parse Kaya's ``o`` struct ourselves.

Mirrors ``preprocessing_finemi.py`` conventions:
  * resample to 200 Hz (the HFREQ sessions are 1000 Hz)
  * common-average reference across the 19 scalp EEG channels
  * 1-second (200-sample) windows -> sample shape (19, 1, 200)
  * standard_1005 montage coordinates
  * cross-subject train/val/test split; labels stored 0-indexed
  * micro-volt storage (Kaya binsuV==1 -> data already in uV) so /100 fits

5F classes: marker 1..5 = thumb / index / middle / ring / little finger MI,
stored as marker-1 (0..4). Markers 0/91/92/99 (rest, session/pause) are ignored.
Channels: Kaya records 22 (21 EEG + X5 sync); A1/A2 (earlobe refs) and X5 are
dropped, leaving 19 scalp EEG. Old T3/T4/T5/T6 are renamed to T7/T8/P7/P8.
"""
import argparse
import json
import os
import urllib.request

os.environ.setdefault('MNE_DATA',
                      '/gpfs/radev/pi/ying_rex/wq44/CSBrain/data/raw/mne_data')
import pickle

import lmdb
import numpy as np
import scipy.io as sio
from scipy import signal
import mne
mne.set_log_level('ERROR')

raw_dir = './data/raw/kaya5f'
out_dir = './data/preprocessed/kaya5f'
DST_FS = 200
WIN = DST_FS                              # 1-second trial window (200 samples)
N_CLASSES = 5

# figshare article id -> subject letter (19 sessions across 8 subjects A..I,
# no D). HFREQ sessions are 1000 Hz, the rest 200 Hz.
ARTICLE_SUBJECT = {
    6818732: 'A', 6818672: 'A',
    6818777: 'B', 6818771: 'B', 6818687: 'B', 6818681: 'B',
    6818774: 'C', 6818684: 'C',
    6818744: 'E', 6818765: 'E', 6818741: 'E',
    6818780: 'F', 6818678: 'F', 6818675: 'F',
    6818783: 'G', 6818753: 'G',
    6818789: 'H',
    6818786: 'I', 6818762: 'I',
}

# Cross-subject split (train 4 / val 2 / test 2 subjects).
SPLIT = {'train': ['A', 'B', 'C', 'E'], 'val': ['F', 'G'], 'test': ['H', 'I']}
SUBJECT_TO_SPLIT = {s: sp for sp, subs in SPLIT.items() for s in subs}

RENAME = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
DROP = {'A1', 'A2', 'X5'}


def download_article(aid):
    """Download the single .mat of a figshare article into raw_dir; return path."""
    os.makedirs(raw_dir, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(
        f'https://api.figshare.com/v2/articles/{aid}', timeout=60).read())
    f = meta['files'][0]
    path = os.path.join(raw_dir, f['name'])
    if os.path.exists(path) and os.path.getsize(path) == f['size']:
        return path
    print(f'  downloading {f["name"]} ({f["size"]/1e6:.0f} MB)...')
    urllib.request.urlretrieve(f['download_url'], path)
    return path


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


def parse_mat(path):
    """Return (X (n,19,200) float32 uV, y (n,) 0..4, ch_names)."""
    o = sio.loadmat(path)['o'][0, 0]
    fs = int(o['sampFreq'].ravel()[0])
    data = o['data'].astype(np.float64)                 # (nS, 22) uV
    mk = o['marker'].ravel().astype(int)
    chn = [str(c[0]) for c in o['chnames'].ravel()]
    eeg_idx = [i for i, c in enumerate(chn) if c not in DROP]
    assert len(eeg_idx) == 19, (len(eeg_idx), chn)
    ch_names = [RENAME.get(chn[i], chn[i]) for i in eeg_idx]

    X = data[:, eeg_idx]
    X = X - X.mean(axis=1, keepdims=True)               # common-average ref
    win = fs                                             # 1-second at native fs
    trials, labels = [], []
    i, n = 0, len(mk)
    while i < n:
        k = mk[i]
        if k in (1, 2, 3, 4, 5):
            j = i
            while j < n and mk[j] == k:
                j += 1
            if j - i >= win:
                seg = X[i:i + win]                       # (fs, 19)
                if fs != DST_FS:
                    seg = signal.resample(seg, DST_FS, axis=0)
                trials.append(seg.T.astype(np.float32))  # (19, 200)
                labels.append(k - 1)
            i = j
        else:
            i += 1
    return np.stack(trials, 0), np.array(labels, dtype=int), ch_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep-raw', action='store_true',
                    help='keep downloaded .mat files (default: delete after use)')
    args = ap.parse_args()

    os.makedirs(out_dir, exist_ok=True)
    db = lmdb.open(out_dir, map_size=8 * 1024 ** 3)
    dataset = {'train': [], 'val': [], 'test': []}

    # group articles by subject
    by_subject = {}
    for aid, subj in ARTICLE_SUBJECT.items():
        by_subject.setdefault(subj, []).append(aid)

    ref_ch = ch_coords = ch_config = None
    for subj in sorted(SUBJECT_TO_SPLIT):
        split = SUBJECT_TO_SPLIT[subj]
        tcount = 0
        for aid in sorted(by_subject[subj]):
            path = download_article(aid)
            X, y, ch_names = parse_mat(path)
            if ref_ch is None:
                ref_ch = ch_names
                ch_coords, ch_config = channel_coords(ch_names)
                assert len(ref_ch) == 19, len(ref_ch)
            assert ch_names == ref_ch, f'{aid} channel mismatch {ch_names}'
            X = X.reshape(X.shape[0], 19, 1, DST_FS)     # n_win == 1
            for i in range(X.shape[0]):
                key = f'{subj}-{aid}-{i}'
                rec = {'sample': X[i], 'label': int(y[i]), 'ch_names': ref_ch,
                       'ch_coords': ch_coords, 'ch_config': ch_config}
                with db.begin(write=True) as txn:
                    txn.put(key.encode(), pickle.dumps(rec))
                dataset[split].append(key)
            tcount += X.shape[0]
            if not args.keep_raw:
                os.remove(path)
        print(f'subject {subj} [{split}] -> {tcount} trials')
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(dataset))

    db.close()
    print('train/val/test sizes:',
          len(dataset['train']), len(dataset['val']), len(dataset['test']))


if __name__ == '__main__':
    main()
