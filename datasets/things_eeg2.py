from glob import glob
from io import BytesIO
import json
import numpy as np
import os
import mne
import webdataset as wds
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import argparse


def to_npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def preprocessing_recording(raw):
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((50))

    return raw


class THINGSEEG2Dataset:
    def __init__(self, dataset_dir='data/THINGS-EEG2', split='train'):
        self.dataset_dir = dataset_dir
        self.split = split
        self.paths = []
        for sub in range(1, 11):
            path = os.path.join(dataset_dir, f'sub-{sub:02d}')
            for ses_dir in glob(os.path.join(path, 'ses-*')):
                for file in glob(os.path.join(ses_dir, "eeg", f'*.vhdr')):
                    if self.split in file:
                        self.paths.append(file)
        assert len(self.paths) > 0, f"No files found for split {self.split} in {dataset_dir}"
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def cache_item(self, idx):
        file = self.paths[idx]
        raw = mne.io.read_raw_brainvision(file, preload=True)
        raw = preprocessing_recording(raw)

        ch_coords_dict = raw.get_montage().get_positions()['ch_pos']
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
        preprocessed_data = {
            "timeseries": timeseries,
            "ch_names": raw.ch_names,
            "ch_coords": ch_coords,
            "split": self.split,
            "file_name": file.replace(self.dataset_dir, '').lstrip('/')
        }


        beh_path = file.replace('.vhdr', '.mat').replace('eeg', 'beh')
        if os.path.exists(beh_path):
            events_samples0, _ = mne.events_from_annotations(raw)
            events_samples = events_samples0[:,0]
 
            beh_data = loadmat(beh_path)['data']
            events_beh = np.asarray(beh_data[0][0][2]['tot_img_number'][0], dtype=int)
            events_beh[np.where(events_beh == 0)[0]] = 99999
            idx = 0
            image_id_list = []
            image_ids = np.zeros((timeseries.shape[1]), dtype=int)
            for s in range(timeseries.shape[1]):
                if idx < len(events_beh):
                    if events_samples[idx] == s:
                        image_ids[s] = events_beh[idx]
                        image_id_list.append(events_beh[idx])
                        idx += 1
            image_id_list = np.array(image_id_list, dtype=int)
            preprocessed_data['image_ids'] = image_id_list[image_id_list != 99999]
            preprocessed_data['events'] = events_samples0[image_id_list != 99999]
        else:
            assert 'rest' in file, f"Behavioral data not found for {file}"

        return preprocessed_data

def _extract_event_windows(item, pre=40, post=200):
    ds = item["timeseries"]
    if "events" not in item or "image_ids" not in item:
        raise ValueError("Item must contain 'events' and 'image_ids' to extract event windows")

    events = item["events"][:, 0].astype(np.int64)
    image_ids = item["image_ids"].astype(np.int64)
    windows = []
    for i, t in enumerate(events):
        window_start = int(t) - pre
        window_end = int(t) + post

        read_start = max(0, window_start)
        read_end = min(ds.shape[1], window_end)
        timeseries = ds[:, read_start:read_end]

        pad_left = max(0, -window_start)
        pad_right = max(0, window_end - ds.shape[1])
        if pad_left > 0 or pad_right > 0:
            timeseries = np.pad(timeseries, ((0, 0), (pad_left, pad_right)))

        if timeseries.shape[1] != pre + post:
            raise ValueError(f"Unexpected window length: expected {pre + post}, but got {timeseries.shape[1]} for event at time {t} with image_id {image_ids[i]} in file {item['file_name']}")

        windows.append({
            "timeseries": timeseries,
            "image_id": int(image_ids[i]),
        })
    return windows


def preprocess(output_dir='data/cache/things_eeg2', event_windows=True):
    os.makedirs(output_dir, exist_ok=True)
    target_dtype = np.float32
    def to_npy_bytes(arr):
        buf = BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    for split in ['train', 'test', 'rest']:
        dataset = THINGSEEG2Dataset(split=split)
        shard_maxcount = 128 if split == 'rest' else 2048

        pattern = os.path.join(output_dir, f"things-eeg2-{split}-%06d.tar")
        with wds.ShardWriter(pattern, maxcount=shard_maxcount) as sink:
            for idx in tqdm(range(len(dataset)), desc="Writing WebDataset"):
                item = dataset.cache_item(idx)
                key = os.path.splitext(item["file_name"])[0].replace("/", "_").replace(".", "_")

                if event_windows and split != 'rest' and 'events' in item and 'image_ids' in item:
                    windows = _extract_event_windows(item)
                    for w_idx, w in enumerate(windows):
                        to_write = {
                            "__key__": f"{key}_evt_{w_idx:05d}",
                            "timeseries.npy": to_npy_bytes(w["timeseries"].astype(target_dtype)),
                            "ch_coords.npy": to_npy_bytes(item["ch_coords"]),
                            "ch_names.json": json.dumps(item["ch_names"]),
                            "file_name.txt": item["file_name"],
                            "split.txt": item["split"],
                            "source.txt": "things-eeg2",
                            "events.npy": to_npy_bytes(np.array([[40, 0, 0]], dtype=np.int32)),
                            "image_ids.npy": to_npy_bytes(np.array([w["image_id"]], dtype=np.int32)),
                        }
                        sink.write(to_write)
                else:
                    for seg_idx, seg in enumerate(range(0, item["timeseries"].shape[1], 60 * 200)):
                        to_write = {
                            "__key__": f"{key}_seg_{seg_idx:05d}",
                            "timeseries.npy": to_npy_bytes(item["timeseries"][:, seg:seg + 60 * 200].astype(target_dtype)),
                            "ch_coords.npy": to_npy_bytes(item["ch_coords"]),
                            "ch_names.json": json.dumps(item["ch_names"]),
                            "file_name.txt": item["file_name"],
                            "split.txt": item["split"],
                            "source.txt": "things-eeg2",
                        }
                        if 'events' in item:
                            to_write["events.npy"] = to_npy_bytes(item["events"])
                        if 'image_ids' in item:
                            to_write["image_ids.npy"] = to_npy_bytes(item["image_ids"])
                        sink.write(to_write)


def plot():
    dataset = THINGSEEG2Dataset()
    item = dataset.cache_item(0)
    timeseries = item['timeseries']
    ch_names = item['ch_names']
    ch_coords = item['ch_coords']

    # psd = raw.compute_psd(fmin=1, fmax=100)
    # fig = psd.plot()
    # fig.savefig(f"things-eeg2.png", dpi=300)
    # plt.close(fig)

    plt.figure(figsize=(12, 6))
    for i in range(min(10, timeseries.shape[0])):  # Plot up to 10 channels
        plt.plot(timeseries[i], label=ch_names[i])  # Offset each channel for visibility
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (µV) + offset')
    plt.ylim(-300, 300)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"figs/things-eeg2.png")
    plt.close()

if __name__ == "__main__":
    # plot()
    preprocess()