import pickle
import lmdb
from torch.utils.data import Dataset
from utils.util import to_tensor
import numpy as np
from scipy.signal import resample

class PretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            SmallerToken
    ):
        super(PretrainingDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.SmallerToken = SmallerToken
        self.db = None
        env = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))
        env.close()

    def _init_db(self):
        self.db = lmdb.open(
            self.dataset_dir, readonly=True, lock=False,
            readahead=False, meminit=False,
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._init_db()
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))

        if self.SmallerToken:
            cha,temstep,point = patch.shape
            patch = patch.reshape(cha,temstep*2,point//2)
        patch = to_tensor(patch)

        return patch



