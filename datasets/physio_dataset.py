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
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

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

        # --- hemisphere-flip augmentation ---
        if self.flip_aug and random.random() < 0.5:
            data = data[self.flip_perm]
            label = self._label_flip.get(label, label)

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

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train',
                                  flip_aug=self.flip_aug)
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
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
