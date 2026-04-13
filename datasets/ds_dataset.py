import io

from torch.utils.data import Dataset
from glob import glob
import os
import mne
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import webdataset as wds
import json

def DsDataset(path):
    if 'ds006171' in path:
        return BDFDataset(path, 50, channel_config='biosemi128')
    elif 'ds006317' in path:
        return EDFDataset(path, 50, channel_config='standard_1005')
    elif 'ds006367' in path:
        return EpochEEGLabDataset(path, 50)
    elif 'ds006370' in path:   
        return EpochEEGLabDataset(path, 50)
    elif 'ds006437' in path:   
        return EEGLabDataset(path, 60)
    elif 'ds006446' in path:
        return EDFDataset(path, 60, channel_config='standard_1005')
    elif 'ds006466' in path:
        return EEGLabDataset(path, 60)
    elif 'ds006480' in path:
        return EEGLabDataset(path, 60)
    elif 'ds006525' in path:
        return EEGLabDataset(path, 60)
    elif 'ds006547' in path:
        return BrainVisionDataset(path, 60)
    else:
        raise ValueError(f"Unknown dataset for path={path!r}")

def to_raw(obj):
    # If Epochs (including EpochsEEGLAB), convert to Raw by randomly selecting one epoch and treating it as a continuous recording
    if not isinstance(obj, mne.io.BaseRaw):
        epochs = obj.get_data()  # shape (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = epochs.shape
        random_epoch_idx = random.randint(0, n_epochs - 1)
        epoch_data = epochs[random_epoch_idx]  # shape (n_channels, n_times)
        info = obj.info.copy()
        raw = mne.io.RawArray(epoch_data, info)
        return raw
    return obj

def preprocessing_recording(raw, notch_freq):
    raw = to_raw(raw)
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((notch_freq))
    # eeg_array = raw.to_data_frame().values
    # print(raw.info)
    # points, chs = eeg_array.shape
    # eeg_array = eeg_array[:, 1:]
    # if points < 300 * 200:
    #     return raw, None
    # a = points % (30 * 200)
    # eeg_array = eeg_array[60 * 200:-(a+60 * 200), :]
    # # print(eeg_array.shape)
    # eeg_array = eeg_array.reshape(-1, 30, 200, chs)
    # eeg_array = eeg_array.transpose(0, 3, 1, 2)
    # Filter to keep only EEG channels
    raw.pick('eeg')
    return raw


def find_file(path, pattern):
    files = glob(os.path.join(path, f'*{pattern}*.*'))
    if len(files) == 0:
        return None
    assert len(files) == 1, f"Multiple files found in {path!r} with pattern {pattern!r}: {files}"
    return files[0]


class ExtDataset(Dataset):
    def __init__(
        self, 
        path, 
        ext, 
        notch_freq, 
        read_func,
        window_size=30,
        channel_config = None
    ):
        self.path = path
        self.files = sorted(glob(path + f'/**/*.{ext}', recursive=True))
        self.read_func = read_func
        self.notch_freq = notch_freq
        self.window_size = window_size
        self.channel_config = channel_config

    def __len__(self):
        return len(self.files)

    def cache_item(self, idx, return_raw=False):
        file_path = self.files[idx]
        try:
            raw = self.read_func(file_path, preload=True)
        except:
            raw = self.read_func(file_path)
        if return_raw:
            return raw
        raw = preprocessing_recording(raw, self.notch_freq)
        # case 1: use montage if available
        if (montage := raw.info.get_montage()) is not None:
            ch_coords_dict = montage.get_positions()['ch_pos']
        elif self.channel_config is not None:
            montage = mne.channels.make_standard_montage(self.channel_config)
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            ch_coords_dict = raw.get_montage().get_positions()['ch_pos']
        # case 2: look for electrodes file
        elif (electrode_file := find_file(os.path.dirname(file_path), 'electrodes')) is not None:
            df = pd.read_csv(electrode_file, sep='\t' if electrode_file.endswith('.tsv') else ',')
            ch_coords_dict = {}
            for _, row in df.iterrows():
                ch_coords_dict[row['name']] = (row['x'], row['y'], row['z'])
        else:
            raise ValueError(f"No montage found for {file_path!r}")
        
        ch_coords_dict = {k.lower(): v for k, v in ch_coords_dict.items()}
        ch_coords = []
        unrecognized_chs = []
        for ch in raw.ch_names:
            ch = ch.lower()
            if ch not in ch_coords_dict:
                ch_coords.append((np.nan, np.nan, np.nan))
                unrecognized_chs.append(ch)
            else:
                ch_coords.append(ch_coords_dict[ch])
        ch_coords = np.stack(ch_coords, axis=0).astype(np.float32)
        timeseries = (raw.get_data() * 1e6).astype(np.float32)
        assert timeseries.shape[0] == ch_coords.shape[0] == len(raw.ch_names), f"Channel count mismatch: timeseries has {timeseries.shape[0]} channels, but found {ch_coords.shape[0]} coordinates and {len(raw.ch_names)} channel names. Unrecognized channels: {unrecognized_chs}"

        return {
            "file_name": file_path.replace(self.path, '').lstrip('/'),
            "timeseries": timeseries,
            "ch_names": raw.ch_names,
            "ch_coords": ch_coords,
            "ch_config": self.channel_config if self.channel_config is not None else 'auto',
        }

class BDFDataset(ExtDataset):
    def __init__(self, path, notch_freq, **kwargs):
        super().__init__(path, 'bdf', notch_freq, mne.io.read_raw_bdf, **kwargs)

class EDFDataset(ExtDataset):
    def __init__(self, path, notch_freq, **kwargs):
        super().__init__(path, 'edf', notch_freq, mne.io.read_raw_edf, **kwargs)
    
class EEGLabDataset(ExtDataset):
    def __init__(self, path, notch_freq, **kwargs):
        super().__init__(path, 'set', notch_freq, mne.io.read_raw_eeglab, **kwargs)

class EpochEEGLabDataset(ExtDataset):
    def __init__(self, path, notch_freq, **kwargs):
        super().__init__(path, 'set', notch_freq, mne.io.read_epochs_eeglab, **kwargs)    

class BrainVisionDataset(ExtDataset):
    def __init__(self, path, notch_freq, **kwargs):
        super().__init__(path, 'vhdr', notch_freq, mne.io.read_raw_brainvision, **kwargs)


# def preprocess():
#     import h5py
#     import sys
#     sys.path.append('.')
#     from utils.electrode import Electrode
#     import matplotlib.pyplot as plt
#     from tqdm import tqdm
#     for dataset in [
#         'ds006171', 
#         'ds006317',
#         'ds006367', 
#         'ds006370', 
#         'ds006437', 
#         'ds006446', 
#         'ds006466', 
#         'ds006480', 
#         'ds006525', 
#         'ds006547'
#     ]:
#         ds = DsDataset(f'./data/{dataset}')
#         # ds.cache_item(0)
#         # continue
        
#         file = h5py.File(f'data/cache/{dataset}.h5', 'w')
#         for idx in tqdm(range(len(ds)), desc=f"Caching {dataset}"):
#             item = ds.cache_item(idx)
#             grp = file.create_group(item['file_name'].replace('/', '_'))
#             grp.create_dataset('timeseries', data=item['timeseries'])
#             grp.create_dataset('ch_names', data=np.array(item['ch_names'], dtype='S'))
#             grp.create_dataset('ch_coords', data=item['ch_coords'])
#             grp.attrs['ch_config'] = item['ch_config']
#         file.close()

def preprocess():
    for ds in [
        'ds006171', 
        'ds006317',
        'ds006367', 
        'ds006370', 
        'ds006437', 
        'ds006446', 
        'ds006466', 
        'ds006480', 
        'ds006525', 
        'ds006547'
    ]:
        dataset = DsDataset(f'./data/{ds}')
        output_dir = f'data/cache/{ds}'
        os.makedirs(output_dir, exist_ok=True)

        def to_npy_bytes(arr):
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            return buf.getvalue()

        pattern = os.path.join(output_dir, f"{ds}-%06d.tar")
        with wds.ShardWriter(pattern, maxcount=256) as sink:
            for idx in tqdm(range(len(dataset)), desc="Writing WebDataset"):
                item = dataset.cache_item(idx)
                key = os.path.splitext(item["file_name"])[0].replace("/", "_")
                sink.write({
                    "__key__": key,
                    "timeseries.npy": to_npy_bytes(item["timeseries"]),
                    "ch_coords.npy": to_npy_bytes(item["ch_coords"]),
                    "ch_names.json": json.dumps(item["ch_names"]),
                    "ch_config.txt": item["ch_config"],
                    "file_name.txt": item["file_name"],
                    "source.txt": ds,
                    "session_id.txt": f"{ds}/{item['file_name']}",
                })

def plot():
    from matplotlib import pyplot as plt
    for dataset in [
        'ds006171', 
        'ds006317',
        'ds006367', 
        'ds006370', 
        'ds006437', 
        'ds006446', 
        'ds006466', 
        'ds006480', 
        'ds006525', 
        'ds006547'
    ]:
        ds = DsDataset(f'./data/{dataset}')
        item = ds.cache_item(0)
        # plot and save
        plt.figure(figsize=(12, 6))
        for i in range(min(10, item['timeseries'].shape[0])):
            plt.plot(item['timeseries'][i], label=item['ch_names'][i])
        plt.title(f"{dataset} - {item['file_name']}")
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude (µV)')
        plt.ylim(-300, 300)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"figs/{dataset}.png")
        plt.close()
        

if __name__ == '__main__':
    # plot()
    preprocess()