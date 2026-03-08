import os
import pickle
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import resample
import mne
from glob import glob
from scipy.io import loadmat
import webdataset as wds
import json
import io
from tqdm import tqdm

def preprocessing_recording(raw):
    # raw.resample(200) #already resampled to 200Hz
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((60))
    raw.pick('eeg')
    return raw

class Physionet2018Dataset(Dataset):
    def __init__(
            self,
            dataset_dir,
    ):
        super().__init__()
        # search for all .mat, excluding -arousal.mat
        self.files = glob(dataset_dir + '/**/*.mat', recursive=True)
        self.files = [f for f in self.files if not f.endswith('-arousal.mat')]
        self.dataset_dir = dataset_dir
        self.channel_config = 'standard_1020'

    def __len__(self):
        return len(self.files)

    def cache_item(self, idx):
        file = self.files[idx]
        mat = loadmat(file)
        ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

        # construct mne raw object
        data = mat['val'][:len(ch_names), :]
        raw = mne.io.RawArray(data * 1e-6, mne.create_info(ch_names, 200, ch_types='eeg'))
        raw = preprocessing_recording(raw)

        montage = mne.channels.make_standard_montage(self.channel_config)
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        ch_coords_dict = {k: v for k, v in raw.get_montage().get_positions()['ch_pos'].items()}
        
        ch_coords = []
        for ch in raw.ch_names:
            if ch not in ch_coords_dict:
                ch_coords.append((np.nan, np.nan, np.nan))
            else:
                ch_coords.append(ch_coords_dict[ch])
        ch_coords = np.stack(ch_coords, axis=0).astype(np.float32)
        timeseries = (raw.get_data() * 1e6).astype(np.float32)
        return {
            "file_name": file.replace(self.dataset_dir, '').lstrip('/'),
            "timeseries": timeseries,
            "ch_names": raw.ch_names,
            "ch_coords": ch_coords,
            "ch_config": self.channel_config if self.channel_config is not None else 'auto',
        }

# def preprocess():
#     from matplotlib import pyplot as plt
#     import h5py
#     from tqdm import tqdm
#     dataset = Physionet2018Dataset('./data/challenge-2018/training')
#     # dataset.cache_item(0)
#     # exit(0)
#     file = h5py.File(f'data/cache/physionet_2018.h5', 'w')
#     for idx in tqdm(range(len(dataset)), desc=f"Caching {dataset}"):
#         item = dataset.cache_item(idx)
#         grp = file.create_group(item['file_name'].replace('/', '_'))
#         grp.create_dataset('timeseries', data=item['timeseries'])
#         grp.create_dataset('ch_names', data=np.array(item['ch_names'], dtype='S'))
#         grp.create_dataset('ch_coords', data=item['ch_coords'])
#         grp.attrs['ch_config'] = item['ch_config']
#     file.close()

def preprocess():
    # dataset = SienaScalpDataset('./data/siena-scalp-eeg/')
    dataset = Physionet2018Dataset('./data/challenge-2018/training')

    output_dir = 'data/cache/physionet_2018'
    os.makedirs(output_dir, exist_ok=True)

    def to_npy_bytes(arr):
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    pattern = os.path.join(output_dir, "train-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=256) as sink:
        for idx in tqdm(range(len(dataset)), desc="Writing WebDataset"):
            item = dataset.cache_item(idx)
            key = os.path.splitext(item["file_name"])[0].replace("/", "_")
            assert "." not in key, f"Key {key} contains dot, which may cause issues in WebDataset"
            sink.write({
                "__key__": key,
                "timeseries.npy": to_npy_bytes(item["timeseries"]),
                "ch_coords.npy": to_npy_bytes(item["ch_coords"]),
                "ch_names.json": json.dumps(item["ch_names"]),
                "ch_config.txt": item["ch_config"],
                "file_name.txt": item["file_name"],
                "source.txt": "physionet_2018",
            })

def plot():
    from matplotlib import pyplot as plt
    dataset = Physionet2018Dataset('./data/challenge-2018/training')
    item = dataset.cache_item(0)
    plt.figure(figsize=(12, 6))
    for i in range(min(10, item['timeseries'].shape[0])):
        plt.plot(item['timeseries'][i], label=item['ch_names'][i])
    plt.title(f"Physionet2018 - {item['file_name']}")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (µV)')
    plt.ylim(-300, 300)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"figs/physionet_2018.png")
    plt.close()


if __name__ == "__main__":
    preprocess()
    # plot()