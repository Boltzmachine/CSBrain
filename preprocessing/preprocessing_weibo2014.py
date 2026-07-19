"""Preprocess Weibo2014 (Yi et al. 2014) -- simple + compound limb motor
imagery, 7 classes -- into a PhysioNet-MI-consistent LMDB.

Mirrors ``preprocessing_finemi.py`` conventions exactly:
  * 200 Hz sampling (MOABB already delivers 200 Hz; asserted, not resampled)
  * common-average reference across the EEG channels
  * 1-second (200-sample) windows -> sample shape (C, n_win, 200)
  * standard_1005 montage coordinates (NaN for unknown; none here)
  * cross-subject train/val/test split; labels stored 0-indexed
  * micro-volt storage so the loader's ``/100`` lands in the usual range

Source: MOABB ``Weibo2014``. The recording is a Neuroscan 64-ch Quik-Cap; MOABB
delivers 65 channels of which 60 are EEG (CB1/CB2 are typed ``misc``, VEO/HEO
``eog``, STIM014 ``stim`` and are dropped). The MI epoch window is
``ds.interval`` == [3, 7] s == 4 s -> 4 one-second windows (800 samples).

Classes (canonical MOABB event_id, stored as id-1):
  0 left_hand   1 right_hand   2 hands (both hands)   3 feet
  4 left_hand_right_foot   5 right_hand_left_foot   6 rest
"""
import os
os.environ.setdefault('MNE_DATA',
                      '/gpfs/radev/pi/ying_rex/wq44/CSBrain/data/raw/mne_data')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import pickle

import lmdb
import numpy as np
import mne
mne.set_log_level('ERROR')
from moabb.datasets import Weibo2014

out_dir = './data/preprocessed/weibo2014'
DST_FS = 200
TMIN, TMAX = 3.0, 7.0                    # MOABB ds.interval
N_WIN = int(round(TMAX - TMIN))          # 4 one-second windows
N_DST = N_WIN * DST_FS                    # 800 samples

# Deterministic canonical label map (== MOABB ds.event_id); stored as id-1.
EVENT_ID = {'left_hand': 1, 'right_hand': 2, 'hands': 3, 'feet': 4,
            'left_hand_right_foot': 5, 'right_hand_left_foot': 6, 'rest': 7}
N_CLASSES = len(EVENT_ID)

# Cross-subject split over the 10 subjects (mirrors FineMI's 12/3/3 -> 6/2/2).
files_dict = {'train': [1, 2, 3, 4, 5, 6], 'val': [7, 8], 'test': [9, 10]}


def channel_coords(ch_names):
    """standard_1005 spherical/cartesian coords; NaN for unknown electrodes."""
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


def load_subject_epochs(ds, s):
    """Return (X_uV (n, C, 800) float32, y (n,) 0..6, ch_names)."""
    sess = ds.get_data(subjects=[s])[s]
    raws = [raw for runs in sess.values() for raw in runs.values()]
    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
    raw.load_data()
    raw.pick('eeg')                                     # 60 EEG channels
    assert raw.info['sfreq'] == DST_FS, raw.info['sfreq']
    raw.set_eeg_reference('average', projection=False)  # common-average ref
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID)
    epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN,
                        tmax=TMAX - 1.0 / DST_FS, baseline=None,
                        preload=True, picks='eeg')
    X = epochs.get_data(units='uV')                     # (n, C, 800)
    assert X.shape[-1] == N_DST, X.shape
    y = epochs.events[:, 2] - 1                          # canonical id -> 0..6
    return X.astype(np.float32), y.astype(int), list(raw.ch_names)


def main():
    os.makedirs(out_dir, exist_ok=True)
    ds = Weibo2014()
    dataset = {'train': [], 'val': [], 'test': []}
    db = lmdb.open(out_dir, map_size=8 * 1024 ** 3)

    ref_ch = ch_coords = ch_config = None
    for split, subs in files_dict.items():
        for s in subs:
            X, y, ch_names = load_subject_epochs(ds, s)
            if ref_ch is None:
                ref_ch = ch_names
                ch_coords, ch_config = channel_coords(ch_names)
                assert len(ref_ch) == 60, len(ref_ch)
            assert ch_names == ref_ch, f'subject {s} channel-order mismatch'

            C = X.shape[1]
            X = X.reshape(X.shape[0], C, N_WIN, DST_FS).astype(np.float32)
            print(f'subject{s} [{split}] -> {X.shape}, labels {np.unique(y)} '
                  f'counts {np.bincount(y, minlength=N_CLASSES).tolist()}')

            for i in range(X.shape[0]):
                key = f'subject{s}-{i}'
                data_dict = {
                    'sample': X[i],
                    'label': int(y[i]),
                    'ch_names': ref_ch,
                    'ch_coords': ch_coords,
                    'ch_config': ch_config,
                }
                with db.begin(write=True) as txn:
                    txn.put(key.encode(), pickle.dumps(data_dict))
                dataset[split].append(key)

    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(dataset))
    db.close()

    print('train/val/test sizes:',
          len(dataset['train']), len(dataset['val']), len(dataset['test']))


if __name__ == '__main__':
    main()
