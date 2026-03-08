import pickle
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import resample
import mne
from glob import glob
import os
import io
import json
import webdataset as wds
from tqdm import tqdm

def preprocessing_recording(raw):
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((60))
    raw.pick('eeg')
    return raw

class RawEEGDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = glob(dataset_dir + '/**/*.bdf', recursive=True)
        self.channel_config = 'standard_1005'

    def __len__(self):
        return len(self.files)

    def cache_item(self, idx):
        file = self.files[idx]
        raw = mne.io.read_raw_bdf(file, preload=True)
        raw = preprocessing_recording(raw)
        montage = mne.channels.make_standard_montage(self.channel_config)
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        ch_coords_dict = {k.lower(): v for k, v in raw.get_montage().get_positions()['ch_pos'].items()}
        
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
#     import matplotlib.pyplot as plt
#     import h5py
#     from tqdm import tqdm
#     dataset = RawEEGDataset('./data/RawEEG')
#     file = h5py.File(f'data/cache/raw_eeg.h5', 'w')
#     for idx in tqdm(range(len(dataset)), desc=f"Caching {dataset}"):
#         item = dataset.cache_item(idx)
#         grp = file.create_group(item['file_name'].replace('/', '_'))
#         grp.create_dataset('timeseries', data=item['timeseries'])
#         grp.create_dataset('ch_names', data=np.array(item['ch_names'], dtype='S'))
#         grp.create_dataset('ch_coords', data=item['ch_coords'])
#         grp.attrs['ch_config'] = item['ch_config']
#     file.close()
#     # psd = raw.compute_psd(fmin=1, fmax=100)
#     # fig = psd.plot()
#     # fig.savefig(f"raweeg.png", dpi=300)
#     # plt.close(fig)

def preprocess():
    dataset = RawEEGDataset('./data/RawEEG')
    output_dir = 'data/cache/raw_eeg'
    os.makedirs(output_dir, exist_ok=True)

    def to_npy_bytes(arr):
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    pattern = os.path.join(output_dir, "raw_eeg-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=256) as sink:
        for idx in tqdm(range(len(dataset)), desc="Writing WebDataset"):
            item = dataset.cache_item(idx)
            key = os.path.splitext(item["file_name"])[0].replace("/", "_").replace(".", "_")
            sink.write({
                "__key__": key,
                "timeseries.npy": to_npy_bytes(item["timeseries"]),
                "ch_coords.npy": to_npy_bytes(item["ch_coords"]),
                "ch_names.json": json.dumps(item["ch_names"]),
                "ch_config.txt": item["ch_config"],
                "file_name.txt": item["file_name"],
                "source.txt": "raw-eeg",
            })

def plot():
    from matplotlib import pyplot as plt
    dataset = RawEEGDataset('./data/RawEEG')
    item = dataset.cache_item(0)

    plt.figure(figsize=(15, 10))
    for i in range(min(10, item['timeseries'].shape[0])):
        plt.plot(item['timeseries'][i], label=item['ch_names'][i])  # Offset each channel for visibility
    plt.title(f"RawEEG - {item['file_name']}")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (µV)')
    plt.ylim(-300, 300)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"figs/raw_eeg.png")
    plt.close()

if __name__ == "__main__":
    preprocess()
    # plot()