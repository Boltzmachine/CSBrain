"""Preprocess the FineMI dataset (Scientific Data, s41597-025-05286-0).

FineMI is a 64-channel Neuroscan SynAmps2 EEG (+fNIRS) recording of fine
motor imagery of 8 upper-limb joint movements. The figshare release ships
convenient per-subject ``subjectN_data_label.mat`` files holding the already
epoched, 4-40 Hz band-passed, 250 Hz EEG:

    data  : (320 trials, 62 channels, 1000 samples)   float64, units mV
    label : (1, 320)                                   int64, values 1..8

The 62 channels are the scalp EEG of the standard Neuroscan 64-channel
Quik-Cap (62 EEG + HEO + VEO; HEO/VEO are dropped from these .mat files).
M1 was the recording reference; the authors already applied a common-average
reference.

This script mirrors ``../CBraMod/preprocessing/preprocessing_physio.py``:
it resamples to 200 Hz, re-applies an average reference, slices each trial
into 4 one-second patches of 200 samples, attaches montage coordinates, and
writes everything to an LMDB keyed by a train/val/test split over subjects.

The model (CSBrain) divides the stored sample by 100 at load time, so we
store the signal in micro-volts (the .mat is in mV -> x1000) to land in the
same numeric range as the other downstream datasets.
"""

import os
import pickle

import lmdb
import numpy as np
import scipy.io as sio
from scipy import signal
import mne

# --- paths -------------------------------------------------------------------
root_dir = './data/raw/finemi'
out_dir = './data/preprocessed/finemi'

# --- acquisition constants ---------------------------------------------------
SRC_FS = 250          # sampling rate of the released .mat
DST_FS = 200          # CSBrain token rate (1 patch == 1 s == 200 samples)
N_SUBJECTS = 18
TRIAL_SECONDS = 4     # 1000 samples @ 250 Hz

# Standard Neuroscan 64-channel Quik-Cap scalp order (62 EEG channels; the
# HEO/VEO bipolar EOG pair is excluded in the released .mat). This is the
# acquisition order, i.e. the channel order of ``data`` in the .mat files.
selected_channels = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2',
]
assert len(selected_channels) == 62, len(selected_channels)

# Cross-subject split (mirrors the per-subject split style of physionet_mi).
files_dict = {
    'train': list(range(1, 13)),    # subjects 1-12
    'val':   list(range(13, 16)),   # subjects 13-15
    'test':  list(range(16, 19)),   # subjects 16-18
}
print(files_dict)


def channel_coords(ch_names):
    """Spherical/cartesian montage coordinates, NaN for unknown electrodes.

    Mirrors preprocessing_physio.py: build a standard_1005 montage and look
    up each channel (case-insensitive). CB1/CB2 are cerebellar electrodes not
    present in the standard montage and get NaN coordinates, exactly like
    physionet's IZ/T9/T10 handling downstream.
    """
    ch_config = 'standard_1005'
    montage = mne.channels.make_standard_montage(ch_config)
    pos = montage.get_positions()['ch_pos']
    pos = {k.lower(): v for k, v in pos.items()}
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


def main():
    os.makedirs(out_dir, exist_ok=True)
    ch_coords, ch_config = channel_coords(selected_channels)

    dataset = {'train': [], 'val': [], 'test': []}
    db = lmdb.open(out_dir, map_size=6 * 1024 ** 3)

    n_dst = TRIAL_SECONDS * DST_FS  # 800 samples

    for split, subjects in files_dict.items():
        for s in subjects:
            mat_path = os.path.join(root_dir, f'subject{s}_data_label.mat')
            mat = sio.loadmat(mat_path)
            data = mat['data'].astype(np.float64)      # (320, 62, 1000) mV
            labels = mat['label'].ravel().astype(int)  # (320,) values 1..8
            assert data.shape[1] == 62, data.shape

            # mV -> uV so that the dataset's /100 lands in the usual range.
            data = data * 1000.0

            # Re-apply common-average reference across the 62 scalp channels.
            data = data - data.mean(axis=1, keepdims=True)

            # Resample 250 Hz -> 200 Hz along the time axis (1000 -> 800).
            data = signal.resample(data, n_dst, axis=2)

            # (trials, ch, 4, 200)
            data = data.reshape(data.shape[0], data.shape[1], TRIAL_SECONDS, DST_FS)
            data = data.astype(np.float32)

            print(f'subject{s} [{split}] -> {data.shape}, '
                  f'labels {np.unique(labels)}')

            for i, (sample, label) in enumerate(zip(data, labels)):
                key = f'subject{s}-{i}'
                data_dict = {
                    'sample': sample,
                    'label': int(label) - 1,        # 1..8 -> 0..7
                    'ch_names': selected_channels,
                    'ch_coords': ch_coords,
                    'ch_config': ch_config,
                }
                txn = db.begin(write=True)
                txn.put(key=key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[split].append(key)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
    txn.commit()
    db.close()

    print('train/val/test sizes:',
          len(dataset['train']), len(dataset['val']), len(dataset['test']))


if __name__ == '__main__':
    main()
