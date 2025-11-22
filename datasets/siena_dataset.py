import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import lmdb
import pickle
import json
from scipy.signal import resample
from scipy import signal


class CustomDataset(Dataset):
    def __init__(self, root, new_sr, if_reshape, ts):
        self.root = root  # Path to JSON file
        self.files = json.load(open(root, "r"))
        self.old_sr = self.files['dataset_info']['sampling_rate']  # Original sampling rate
        self.channel_name = self.files['dataset_info']['ch_names']  # Channel names
        self.data = self.files['subject_data']  # EEG data
        self.new_sr = new_sr  # New sampling rate
        self.ts = ts  # Signal length
        self.if_reshape = if_reshape

    def __len__(self):
        return len(self.data)

    def get_ch_names(self):
        return self.channel_name

    def normalize(self, X):
        X = X * 10000  # Normalization factor
        return X

    def resample_data(self, data):
        if self.old_sr == self.new_sr:
            return data  # No resampling needed if sampling rates are the same
        else:
            number_of_samples = int(data.shape[-1] * self.new_sr / self.old_sr)
            return signal.resample(data, number_of_samples, axis=-1)

    def __getitem__(self, index):
        trial = self.data[index]
        file_path = trial['file']
        sample = pickle.load(open(file_path, "rb"))
        X = sample["X"]
        X = self.resample_data(X)  # Resample the data if necessary
        X = self.normalize(X)  # Normalize the data
        X = torch.FloatTensor(X)
        Y = int(sample["Y"])
        if self.if_reshape:
            X = X.reshape(len(self.channel_name), self.ts, self.new_sr)  # Reshape if required
        return X, Y

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return torch.FloatTensor(x_data), torch.FloatTensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir + '/train.json', 200, if_reshape=True, ts=10)
        val_set = CustomDataset(self.datasets_dir + '/val.json', 200, if_reshape=True, ts=10)
        test_set = CustomDataset(self.datasets_dir + '/test.json', 200, if_reshape=True, ts=10)
        print(len(train_set), len(val_set), len(test_set))

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
