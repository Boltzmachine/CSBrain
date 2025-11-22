import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import json
import pickle
from scipy import signal
import pandas as pd


class CustomDataLoader(Dataset):
    def __init__(self, root, new_sr, normalize_method='z_score', ems_factor=0.001, factor=100, dim=0, cross=False, subject_id=0):
        self.root = root
        self.files = json.load(open(root, "r"))
        self.old_sr = self.files['dataset_info']['sampling_rate']
        self.channel_name = self.files['dataset_info']['ch_names']
        self.mean_value = self.files['dataset_info']['mean']
        self.std_value = self.files['dataset_info']['std']
        self.max = self.files['dataset_info']['max']
        self.min = self.files['dataset_info']['min']
        self.data = self.files['subject_data']
        self.subject_data = [entry for entry in self.data if entry["subject_id"] == subject_id]
        self.cross = cross
        self.normalize_method = normalize_method
        self.dim = dim
        self.factor = factor
        self.new_sr = new_sr
        self.ems_factor = ems_factor

    def __len__(self):
        return len(self.subject_data) if self.cross else len(self.data)

    def get_ch_names(self):
        return self.channel_name

    def normalize(self, X):
        if self.normalize_method == 'z_score':
            mean_value, std_value = np.array(self.mean_value), np.array(self.std_value)
            mu, sigma = np.expand_dims(mean_value, axis=1), np.expand_dims(std_value, axis=1)
            X = (X - mu) / (sigma + 1e-8)
        elif self.normalize_method == 'min_max':
            X = (X - self.max) / (self.max - self.min)
        elif self.normalize_method == 'ems':
            X = self.exponential_moving_standardize(X)
        elif self.normalize_method == "0.1mv":
            X = X / self.factor
        elif self.normalize_method == '95':
            X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        return X

    def exponential_moving_standardize(self, X, eps=1e-4):
        X = X.T
        df = pd.DataFrame(X)
        meaned = df.ewm(alpha=self.ems_factor).mean()
        demeaned = df - meaned
        squared = demeaned * demeaned
        square_ewmed = squared.ewm(alpha=self.ems_factor).mean()
        standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
        return standardized.T

    def resample_data(self, data):
        if self.old_sr != self.new_sr:
            number_of_samples = int(data.shape[-1] * self.new_sr / self.old_sr)
            return signal.resample(data, number_of_samples, axis=-1)
        return data

    def __getitem__(self, index):
        trial = self.subject_data[index] if self.cross else self.data[index]
        file_path = trial['file']
        sample = pickle.load(open(file_path, "rb"))
        X = sample["X"]
        X = self.resample_data(X)
        X = self.normalize(X)
        X = torch.FloatTensor(X)
        X = X.view(X.shape[0], -1, 200)
        Y = int(sample["Y"])
        return X, Y

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


def get_data_loader(args):
    dataset_train = CustomDataLoader(args.datasets_dir + '/train.json', 200)
    dataset_test = CustomDataLoader(args.datasets_dir + '/test.json', 200)
    dataset_val = CustomDataLoader(args.datasets_dir + '/val.json', 200)

    data_loader = {
        'train': DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=dataset_train.collate,
            shuffle=True,
        ),
        'val': DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=dataset_val.collate,
            shuffle=False,
        ),
        'test': DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            collate_fn=dataset_test.collate,
            shuffle=False,
        ),
    }

    return data_loader
