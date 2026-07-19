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
    """Forenzo2024 Continuous-Pursuit dataset -- CLASSIFICATION view.

    Samples are stored as (62, 1, 200) float32 in micro-volts; the model
    consumes ``x/100``. Labels are the 4-way intended DIRECTION derived from
    the target-relative vector: 0 right / 1 up / 2 left / 3 down.
    """

    def __init__(self, data_dir, mode='train'):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False,
                            readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        return {
            'x': pair['sample'] / 100,
            'y': pair['label'],
            'ch_coords': _to_spherical(pair['ch_coords']),
            'ch_names': pair['ch_names'],
        }

    def collate(self, batch):
        x_data = np.array([x['x'] for x in batch])
        y_label = np.array([x['y'] for x in batch])
        ch_coords = np.array([x['ch_coords'] for x in batch])
        return {
            'x': to_tensor(x_data),
            'y': to_tensor(y_label).long(),
            'ch_coords': to_tensor(ch_coords),
            'ch_names': [x['ch_names'] for x in batch],
        }


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        # Honor --num_workers (0 avoids the LMDB-fork segfault that the ~401k-entry
        # Forenzo env triggers with worker processes; the smaller datasets got
        # lucky with 4).
        nw = getattr(self.params, 'num_workers', 4)
        data_loader = {
            'train': DataLoader(train_set, batch_size=self.params.batch_size,
                                collate_fn=train_set.collate, num_workers=nw, shuffle=True),
            'val': DataLoader(val_set, batch_size=self.params.batch_size,
                              collate_fn=val_set.collate, num_workers=nw, shuffle=False),
            'test': DataLoader(test_set, batch_size=self.params.batch_size,
                               collate_fn=test_set.collate, num_workers=nw, shuffle=False),
        }
        return data_loader
