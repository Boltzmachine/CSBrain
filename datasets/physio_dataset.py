import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from .cached_dataset import _to_spherical

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            flip_aug=False,
            ea_matrices=None,
            highpass_hz=0.0,
            fs=200,
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        # Optional per-subject Euclidean Alignment matrices (dict keyed by
        # 'SXXX' subject id, parsed from the LMDB key prefix). When set,
        # whitening is applied in __getitem__ before the /100 rescale.
        self.ea_matrices = ea_matrices

        # Optional zero-phase high-pass to strip sub-``highpass_hz`` content
        # (e.g. 7 Hz to remove delta/theta + slow cue-locked evoked drift and
        # keep mu/beta motor-imagery rhythms). Coefficients are precomputed
        # once; the filter is applied along the flattened time axis in
        # __getitem__. 0/None disables it.
        self.highpass_hz = float(highpass_hz or 0.0)
        self.fs = fs
        self._hp_sos = None
        if self.highpass_hz > 0:
            from scipy.signal import butter
            self._hp_sos = butter(4, self.highpass_hz / (fs / 2.0),
                                  btype='high', output='sos')

        # Hemisphere-flip augmentation (training only).
        # Swaps homologous electrode pairs and relabels left↔right fist.
        self.flip_aug = flip_aug and mode == 'train'
        if self.flip_aug:
            from models.alignment import build_flip_perm_batch
            with self.db.begin(write=False) as txn:
                pair = pickle.loads(txn.get(self.keys[0].encode()))
            self.flip_perm = build_flip_perm_batch(
                [pair['ch_names']])[0].numpy()
            # PhysioNet-MI labels: 0=left fist, 1=right fist,
            #                      2=both fists, 3=both feet
            self._label_flip = {0: 1, 1: 0}

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']

        # --- high-pass filter (optional) ---
        # Zero-phase Butterworth along the continuous time axis. Sample is
        # (C, n_windows, T); flatten windows*T so the filter sees the trial
        # as one contiguous series, then restore the shape.
        if self._hp_sos is not None:
            from scipy.signal import sosfiltfilt
            C, W, T = data.shape
            filtered = sosfiltfilt(self._hp_sos, data.reshape(C, W * T), axis=-1)
            data = np.ascontiguousarray(filtered, dtype=data.dtype).reshape(C, W, T)

        # --- hemisphere-flip augmentation ---
        if self.flip_aug and random.random() < 0.5:
            data = data[self.flip_perm]
            label = self._label_flip.get(label, label)

        # --- Euclidean Alignment (optional) ---
        # Sample shape is (C, n_windows, T). EA whitens along the channel
        # axis; we reshape to (C, n_windows*T), apply, then reshape back.
        if self.ea_matrices is not None:
            sub_id = key.split('R', 1)[0]                  # 'S001R04-1' -> 'S001'
            R = self.ea_matrices.get(sub_id)
            if R is not None:
                from datasets.euclidean_alignment import apply_ea
                C, W, T = data.shape
                data = apply_ea(data.reshape(C, W * T), R).reshape(C, W, T)

        ch_coords = _to_spherical(pair["ch_coords"])
        ch_names = pair["ch_names"]
        return {
            'x': data/100,
            'y': label,
            'ch_coords': ch_coords,
            'ch_names': ch_names,
        }

    def collate(self, batch):
        x_data = np.array([x['x'] for x in batch])
        y_label = np.array([x['y'] for x in batch])
        ch_coords = np.array([x['ch_coords'] for x in batch])
        return {
            'x': to_tensor(x_data),
            'y': to_tensor(y_label).long(),
            'ch_coords': to_tensor(ch_coords),
        }


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.flip_aug = getattr(params, 'hemisphere_flip_aug', False)
        # Lazy-load the EA matrices sidecar only when the flag is set, so
        # default behaviour is unchanged. Sidecar path defaults to
        # ``<datasets_dir>/ea_subject.pt`` (next to the LMDB) but can be
        # overridden via params.ea_path.
        self.ea_matrices = None
        if getattr(params, 'use_euclidean_alignment', False):
            from datasets.euclidean_alignment import load_ea_matrices
            ea_path = getattr(params, 'ea_path', None) or os.path.join(
                self.datasets_dir, 'ea_subject.pt')
            self.ea_matrices = load_ea_matrices(ea_path)

        self.highpass_hz = getattr(params, 'highpass_hz', 0.0)
        self.fs = getattr(params, 'fs', 200)

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train',
                                  flip_aug=self.flip_aug,
                                  ea_matrices=self.ea_matrices,
                                  highpass_hz=self.highpass_hz, fs=self.fs)
        val_set = CustomDataset(self.datasets_dir, mode='val',
                                ea_matrices=self.ea_matrices,
                                highpass_hz=self.highpass_hz, fs=self.fs)
        test_set = CustomDataset(self.datasets_dir, mode='test',
                                 ea_matrices=self.ea_matrices,
                                 highpass_hz=self.highpass_hz, fs=self.fs)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                num_workers=4,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                num_workers=4,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                num_workers=4,
                shuffle=False,
            ),
        }
        return data_loader
