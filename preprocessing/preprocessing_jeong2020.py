"""Preprocess Jeong2020 (intuitive upper-limb MI, 11 classes) into a
PhysioNet-MI-consistent LMDB.

Mirrors ``preprocessing_finemi.py`` conventions:
  * resample 1000 Hz -> 200 Hz
  * common-average reference across the 60 scalp EEG channels
  * 1-second (200-sample) windows -> sample shape (60, 4, 200); interval [0,4]s
  * standard_1005 montage coordinates (NaN for unknown)
  * cross-subject train/val/test split; labels stored 0-indexed
  * micro-volt storage so the loader's ``/100`` lands in the usual range

Source: MOABB ``Jeong2020`` (condition='MI' is the default -> pure motor
IMAGERY, no realMove/ME). Each subject = 3 sessions x 3 runs; run 0 = 6 reach
directions, run 1 = 3 grasps, run 2 = 2 wrist twists -> 11 classes. MOABB types
the 4 EOG + 7 EMG channels as 'eeg', so they are excluded by name, leaving 60
scalp EEG channels.

Classes (canonical MOABB event_id, stored as id-1):
  0 reach_forward 1 reach_backward 2 reach_left 3 reach_right 4 reach_up
  5 reach_down 6 grasp_cup 7 grasp_ball 8 grasp_card 9 twist_pronation
  10 twist_supination

Heavy dataset (~9.8 GB/subject). Runs subject-by-subject, writes __keys__
incrementally (resumable: already-done subjects are skipped), and with
--delete-raw removes each subject's raw cache after preprocessing so peak disk
stays ~1 subject.
  usage: python preprocessing/preprocessing_jeong2020.py --subjects 1,2,3
         python preprocessing/preprocessing_jeong2020.py --subjects all --delete-raw
"""
import argparse
import os
import shutil
import time

os.environ.setdefault('MNE_DATA',
                      '/gpfs/radev/pi/ying_rex/wq44/CSBrain/data/raw/mne_data')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import pickle

import lmdb
import numpy as np
import mne
mne.set_log_level('ERROR')
from moabb.datasets import Jeong2020

out_dir = './data/preprocessed/jeong2020'
CACHE_DIR = os.path.join(os.environ['MNE_DATA'], 'MNE-jeong2020-data')
DST_FS = 200
TMIN, TMAX = 0.0, 4.0
N_WIN = int(round(TMAX - TMIN))          # 4 one-second windows
N_DST = N_WIN * DST_FS                    # 800

EVENT_ID = {'reach_forward': 1, 'reach_backward': 2, 'reach_left': 3,
            'reach_right': 4, 'reach_up': 5, 'reach_down': 6, 'grasp_cup': 7,
            'grasp_ball': 8, 'grasp_card': 9, 'twist_pronation': 10,
            'twist_supination': 11}
N_CLASSES = len(EVENT_ID)

# Non-scalp channels MOABB mistypes as EEG -> drop to leave the 60 scalp EEG.
EXCLUDE = {'hEOG_L', 'hEOG_R', 'vEOG_U', 'vEOG_D',
           'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6', 'EMG_ref'}

# Cross-subject split over subjects 1..15 (11/2/2). Order 1,2,3,... populates
# all three splits after the first 3 subjects so the LMDB is smoke-testable
# early even while the rest download.
SPLIT = {
    'train': [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'val':   [2, 14],
    'test':  [3, 15],
}
SUBJECT_TO_SPLIT = {s: split for split, subs in SPLIT.items() for s in subs}


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


def _retry(fn, tries=8, base=15):
    """Retry fn on transient errors (Zenodo/figshare 503, network hiccups)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as ex:
            if i == tries - 1:
                raise
            wait = min(base * (2 ** i), 300)
            print(f'  download/parse failed ({type(ex).__name__}: {ex}); '
                  f'retry {i + 1}/{tries} in {wait}s')
            time.sleep(wait)


def load_subject_epochs(ds, s):
    """Return (X_uV (n, 60, 800) float32, y (n,) 0..10, ch_names)."""
    data = ds.get_data(subjects=[s])[s]
    Xs, ys, ref_ch = [], [], None
    for sess, runs in data.items():
        for run, raw in runs.items():
            raw.load_data()
            keep = [c for c in raw.ch_names if c not in EXCLUDE]
            raw.pick(keep)
            assert raw.info['nchan'] == 60, (s, sess, run, raw.info['nchan'])
            raw.set_eeg_reference('average', projection=False)
            if raw.info['sfreq'] != DST_FS:
                raw.resample(DST_FS)
            events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID)
            if len(events) == 0:
                continue
            # Each run holds only its subset of the 11 classes (reach / grasp /
            # twist), so tolerate event ids with no matching events this run.
            epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN,
                                tmax=TMAX - 1.0 / DST_FS, baseline=None,
                                preload=True, on_missing='ignore')
            X = epochs.get_data(units='uV')
            assert X.shape[-1] == N_DST, X.shape
            Xs.append(X.astype(np.float32))
            ys.append(epochs.events[:, 2] - 1)
            if ref_ch is None:
                ref_ch = list(epochs.ch_names)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(int)
    return X, y, ref_ch


def delete_subject_cache(s):
    sub_dir = os.path.join(CACHE_DIR, f'sub{s}')
    if os.path.isdir(sub_dir):
        shutil.rmtree(sub_dir, ignore_errors=True)
    zen = os.path.join(CACHE_DIR, 'zenodo')
    if os.path.isdir(zen):                     # clear used archives
        for f in os.listdir(zen):
            shutil.rmtree(os.path.join(zen, f), ignore_errors=True)
    print(f'  deleted raw cache for subject {s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='1,2,3',
                    help="comma list, or 'all' for the full SPLIT")
    ap.add_argument('--delete-raw', action='store_true',
                    help='remove each subject raw cache after preprocessing')
    args = ap.parse_args()

    if args.subjects == 'all':
        subjects = sorted(SUBJECT_TO_SPLIT)
    else:
        subjects = [int(x) for x in args.subjects.split(',')]

    os.makedirs(out_dir, exist_ok=True)
    ds = Jeong2020(condition='MI')
    db = lmdb.open(out_dir, map_size=16 * 1024 ** 3)

    # Resume: load existing split index + which subjects are already stored.
    with db.begin() as txn:
        raw_keys = txn.get(b'__keys__')
    dataset = pickle.loads(raw_keys) if raw_keys else {'train': [], 'val': [], 'test': []}
    done = set()
    for v in dataset.values():
        for k in v:
            done.add(int(k.split('-')[0][3:]))   # 'sub7-12' -> 7

    ref_ch = ch_coords = ch_config = None
    for s in subjects:
        if s not in SUBJECT_TO_SPLIT:
            print(f'subject {s} not in SPLIT, skipping'); continue
        if s in done:
            print(f'subject {s} already stored, skipping'); continue
        split = SUBJECT_TO_SPLIT[s]
        X, y, ch_names = _retry(lambda: load_subject_epochs(ds, s))
        if ref_ch is None:
            ref_ch = ch_names
            ch_coords, ch_config = channel_coords(ch_names)
            assert len(ref_ch) == 60, len(ref_ch)
        assert ch_names == ref_ch, f'subject {s} channel-order mismatch'

        X = X.reshape(X.shape[0], 60, N_WIN, DST_FS).astype(np.float32)
        print(f'subject{s} [{split}] -> {X.shape}, labels {np.unique(y)} '
              f'counts {np.bincount(y, minlength=N_CLASSES).tolist()}')
        for i in range(X.shape[0]):
            key = f'sub{s}-{i}'
            rec = {'sample': X[i], 'label': int(y[i]), 'ch_names': ref_ch,
                   'ch_coords': ch_coords, 'ch_config': ch_config}
            with db.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps(rec))
            dataset[split].append(key)
        # incremental __keys__ so a partial LMDB is usable/resumable
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(dataset))
        done.add(s)
        if args.delete_raw:
            delete_subject_cache(s)

    db.close()
    print('train/val/test sizes:',
          len(dataset['train']), len(dataset['val']), len(dataset['test']))


if __name__ == '__main__':
    main()
