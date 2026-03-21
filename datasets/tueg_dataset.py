import pickle
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import resample
import mne
from glob import glob

selected_channels = {
    '01_tcp_ar': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ],
    '02_tcp_le': [
            'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
            'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
            'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'
    ],
    '03_tcp_ar_a': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ]
}


def preprocessing_recording(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    if '02_tcp_le' in file_path:
        for ch in selected_channels['02_tcp_le']:
            if ch not in raw.info['ch_names']:
                return
        raw.pick_channels(selected_channels['02_tcp_le'], ordered=True)
    elif '01_tcp_ar' in file_path:
        for ch in selected_channels['01_tcp_ar']:
            if ch not in raw.info['ch_names']:
                return
        raw.pick_channels(selected_channels['01_tcp_ar'], ordered=True)
    elif '03_tcp_ar_a' in file_path:
        for ch in selected_channels['03_tcp_ar_a']:
            if ch not in raw.info['ch_names']:
                return
        raw.pick_channels(selected_channels['03_tcp_ar_a'], ordered=True)
    else:
        return
    # print(raw.info)
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((60))

    return raw

    eeg_array = raw.to_data_frame().values
    # print(raw.info)
    eeg_array = eeg_array[:, 1:]
    points, chs = eeg_array.shape
    if points < 300 * 200:
        return
    a = points % (30 * 200)
    eeg_array = eeg_array[60 * 200:-(a+60 * 200), :]
    # print(eeg_array.shape)
    eeg_array = eeg_array.reshape(-1, 30, 200, chs)
    eeg_array = eeg_array.transpose(0, 3, 1, 2)


class TUEGDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = glob(dataset_dir + '/**/*.edf', recursive=True)
        self.channel_config = 'standard_1020'

    def __len__(self):
        return len(self.files)

    def cache_item(self, idx):
        file = self.files[idx]
        raw = preprocessing_recording(file)

        if raw is None:
            return

        rename_dict = {}
        for ch in raw.ch_names:
            rename_dict[ch] = ch.replace('EEG ', '').replace('-LE', '').replace('-REF', '')
            
        raw.rename_channels(rename_dict)
        montage = mne.channels.make_standard_montage(self.channel_config)
        raw.set_montage(montage, match_case=False)
        ch_coords_dict = {k.lower(): v for k, v in raw.get_montage().get_positions()['ch_pos'].items()}
        
        ch_coords = []
        for ch in raw.ch_names:
            ch = ch.lower()
            if ch not in ch_coords_dict:
                raise ValueError(f"Channel {ch!r} not found in montage")
            ch_coords.append(ch_coords_dict[ch])
        ch_coords = np.stack(ch_coords, axis=0).astype(np.float32)

        return {
            "file_name": file.replace(self.dataset_dir, '').lstrip('/'),
            "timeseries": (raw.get_data() * 1e6).astype(np.float32),
            "ch_names": raw.ch_names,
            "ch_coords": ch_coords,
            "ch_config": self.channel_config if self.channel_config is not None else 'auto',
        }

if __name__ == "__main__":
    import os
    import h5py
    from tqdm import tqdm
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    import webdataset as wds
    import json
    from io import BytesIO

    dataset = TUEGDataset("./data/TUEG/v2.0.1/edf")
    # dataset.cache_item(0)
    # exit(0)
    os.makedirs("data/cache", exist_ok=True)

    # I/O-heavy EDF loading usually benefits from more threads
    num_workers = min(8, (os.cpu_count() or 1) * 4)
    print(f"Caching with {num_workers} workers...")

    def load_item(idx):
        item = dataset.cache_item(idx)
        if item is None:
            return
        key = item["file_name"].replace("/", "_")
        return key, item

    # with h5py.File("data/cache/tueg.h5", "w") as f:
    #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         next_idx = 0
    #         in_flight = set()
    #         max_in_flight = max(1, num_workers)

    #         while next_idx < len(dataset) and len(in_flight) < max_in_flight:
    #             in_flight.add(executor.submit(load_item, next_idx))
    #             next_idx += 1

    #         with tqdm(total=len(dataset), desc=f"Caching {dataset}") as pbar:
    #             while in_flight:
    #                 done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

    #                 for future in done:
    #                     result = future.result()

    #                     if result is not None:
    #                         key, item = result
    #                         if key in f:
    #                             raise ValueError(f"Duplicate key {key!r} found while caching")

    #                         grp = f.create_group(key)
    #                         grp.create_dataset("timeseries", data=item["timeseries"], dtype=np.float32)
    #                         grp.create_dataset("ch_names", data=np.array(item["ch_names"], dtype="S"))
    #                         grp.create_dataset("ch_coords", data=item["ch_coords"], dtype=np.float32)
    #                         grp.attrs["ch_config"] = item["ch_config"]

    #                     pbar.update(1)

    #                 while next_idx < len(dataset) and len(in_flight) < max_in_flight:
    #                     in_flight.add(executor.submit(load_item, next_idx))
    #                     next_idx += 1
    # Save to WebDataset format

    def to_npy_bytes(array: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()

    shard_pattern = "data/cache/tueg/tueg-%06d.tar"
    max_samples_per_shard = 1000
    maxsize = 500 * 1024 * 1024  # 500 MB
    seen_keys = set()

    with wds.ShardWriter(shard_pattern, maxcount=max_samples_per_shard, maxsize=maxsize) as sink:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            next_idx = 0
            in_flight = set()
            max_in_flight = max(1, num_workers)

            while next_idx < len(dataset) and len(in_flight) < max_in_flight:
                in_flight.add(executor.submit(load_item, next_idx))
                next_idx += 1

            with tqdm(total=len(dataset), desc="Caching WebDataset") as pbar:
                while in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

                    for future in done:
                        result = future.result()

                        if result is not None:
                            key, item = result
                            if key in seen_keys:
                                raise ValueError(f"Duplicate key {key!r} found while caching")
                            seen_keys.add(key)

                            
                            sample = {
                                "__key__": key.replace(".", "_"),
                                "timeseries.npy": to_npy_bytes(item["timeseries"]),
                                "ch_coords.npy": to_npy_bytes(item["ch_coords"]),
                                "ch_names.json": json.dumps(item["ch_names"]),
                                "ch_config.txt": item["ch_config"],
                                "file_name.txt": item["file_name"],
                            }
                            sink.write(sample)

                        pbar.update(1)

                    while next_idx < len(dataset) and len(in_flight) < max_in_flight:
                        in_flight.add(executor.submit(load_item, next_idx))
                        next_idx += 1