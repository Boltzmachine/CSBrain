import os

from matplotlib import image
from torch.utils.data import Dataset
import h5py
import random
import numpy as np
import torch
import webdataset as wds
import json
from glob import glob
from transformers import AutoImageProcessor


# model_name = "facebook/dinov2-base"
# image_processor = AutoImageProcessor.from_pretrained(model_name)

# @profile
def collate_cached(batch):
    # pad timeseries to the same channel, also pad ch_coords and ch_names accordingly
    max_chs = max(x['timeseries'].shape[0] for x in batch)
    for x in batch:
        chs = x['timeseries'].shape[0]
        if chs < max_chs:
            padding_chs = max_chs - chs
            x['timeseries'] = torch.cat([x['timeseries'], torch.zeros(padding_chs, x['timeseries'].shape[1], x['timeseries'].shape[2])], dim=0)
            x['ch_coords'] = torch.cat([x['ch_coords'], torch.zeros(padding_chs, 3)], dim=0)
            x['ch_names'] = list(x['ch_names']) + ['pad'] * padding_chs
        x['valid_channel_mask'] = torch.tensor([1] * chs + [0] * (max_chs - chs), dtype=torch.bool)

    image_pixel_values = None
    if 'images' in batch[0]:
        all_images = []
        for x in batch:
            all_images.extend(x['images'])
        image_encoder_inputs = image_processor(images=all_images, return_tensors="pt")
        for key in image_encoder_inputs:
            if key == 'pixel_values':
                image_pixel_values = image_encoder_inputs[key].view(len(batch), -1, *image_encoder_inputs[key].shape[1:])
            else:
                raise ValueError(f"Unexpected key '{key}' in image_encoder_inputs")
            
    if "pixel_values" in batch[0]:
        image_pixel_values = torch.stack([x['pixel_values'] for x in batch], dim=0)
        
    return {
        'timeseries': torch.stack([x['timeseries'] for x in batch], dim=0),
        'ch_coords': torch.stack([x['ch_coords'] for x in batch], dim=0),
        'ch_names': [x['ch_names'] for x in batch],
        'valid_channel_mask': torch.stack([x['valid_channel_mask'] for x in batch], dim=0),
        'source': [x['source'] for x in batch],
        'image_hidden_states': torch.stack([x['image_hidden_states'] for x in batch if 'image_hidden_states' in x], dim=0) if 'image_hidden_states' in batch[0] else None,
        'events': torch.stack([x['events'] for x in batch if 'events' in x], dim=0) if 'events' in batch[0] else None,
        'sfreq': torch.stack([x['sfreq'] for x in batch], dim=0) if 'sfreq' in batch[0] else None,
        "image_encoder_inputs": {
            'pixel_values': image_pixel_values
        }
    }

class H5Dataset(Dataset):
    def __init__(self, h5_file, do_strip=False):
        super().__init__()
        self.file_path = h5_file
        self.do_strip = do_strip
        with h5py.File(h5_file, 'r') as f:
            self.keys = list(f.keys())
                    
        # 2. Initialize the file handle as None. 
        # It will be instantiated by each worker separately.
        self.file = None


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.file is None:
            # Note: swmr=True (Single Writer, Multiple Reader) is highly recommended 
            # for parallel reading to prevent locking issues.
            self.file = h5py.File(
                self.file_path, 
                'r', 
                swmr=True,
                rdcc_nbytes=512 * 1024 * 1024, 
                rdcc_nslots=1_000_003
            )

        data = self.file[self.keys[idx]]
        if self.do_strip:
            ds = data['timeseries']
        else:
            ds = data['timeseries']
        
        full_len = ds.shape[1]
        strip_len = 60 * 200 if self.do_strip else 0
        valid_len = full_len - (2 * strip_len)
        target_len = 30 * 200
        
        if valid_len > target_len:
            start_offset = strip_len + random.randint(0, valid_len - target_len)
            # ONLY read the 30-second slice from disk
            timeseries = ds[:, start_offset : start_offset + target_len]
        else:
            # For short files, read what we have and pad
            timeseries = ds[:, strip_len : full_len - strip_len]
            padding = target_len - timeseries.shape[1]
            timeseries = np.pad(timeseries, ((0, 0), (0, padding)))
            timeseries = timeseries.reshape(timeseries.shape[0], 30, 200)
        
        np.clip(timeseries, -300, 300, out=timeseries)
        timeseries = timeseries.reshape(timeseries.shape[0], 30, 200)

        ch_coords = torch.tensor(data['ch_coords'][:], dtype=torch.float32)
        ch_coords_spherical = torch.zeros_like(ch_coords)
        x, y, z = ch_coords[:, 0], ch_coords[:, 1], ch_coords[:, 2]
        ch_coords_spherical[:, 0] = torch.sqrt(x**2 + y**2 + z**2)  # r
        ch_coords_spherical[:, 1] = torch.atan2(y, x)  # theta
        ch_coords_spherical[:, 2] = torch.acos(z / (ch_coords_spherical[:, 0] + 1e-8))  # phi

        return {
            'timeseries': torch.tensor(timeseries, dtype=torch.float32),
            'ch_coords': ch_coords_spherical,
            'ch_names': data['ch_names'][:],
            "source": self.file_path,
        }
    
    def __del__(self):
        if self.file is not None:
            self.file.close()


def _to_spherical(ch_coords):
    ch_coords = torch.tensor(ch_coords, dtype=torch.float32)
    assert ch_coords.size(-1) == 3, "Expected Cartesian coordinates with shape (..., 3)"
    ch_coords_spherical = torch.zeros_like(ch_coords)
    x, y, z = ch_coords[:, 0], ch_coords[:, 1], ch_coords[:, 2]
    ch_coords_spherical[:, 0] = torch.sqrt(x**2 + y**2 + z**2)  # r
    ch_coords_spherical[:, 1] = torch.atan2(y, x)  # theta
    ch_coords_spherical[:, 2] = torch.acos(z / (ch_coords_spherical[:, 0] + 1e-8))  # phi
    return ch_coords_spherical


# @profile
class _process_webdataset_sample:
    def __init__(self):
        self.image_file = None


    def __call__(self, sample):
        if self.image_file is None:
            self.image_file = h5py.File("data/Alljoined-1.6M/pixel_values.h5", 'r')
        s_freq = 200
        do_strip = 'tueg' in str(sample.get("source", "")) or 'tueg' in sample.get("__url__", "")
        ds = sample["timeseries.npy"]

        full_len = ds.shape[1]
        strip_len = 60 * 200 if do_strip else 0
        valid_len = full_len - (2 * strip_len)
        target_len = 30 * 200

        if valid_len > target_len:
            start_offset = strip_len + random.randint(0, valid_len - target_len)
            timeseries = ds[:, start_offset : start_offset + target_len]
        else:
            timeseries = ds[:, strip_len : full_len - strip_len]
            padding = target_len - timeseries.shape[1]
            if padding > 0:
                timeseries = np.pad(timeseries, ((0, 0), (0, padding)))
            timeseries = timeseries.reshape(timeseries.shape[0], 30, 200)

        np.clip(timeseries, -300, 300, out=timeseries)

        timeseries = timeseries.reshape(timeseries.shape[0], 30, 200).astype(np.float32)

        ch_names_raw = sample.get("ch_names.json", "[]")
        if isinstance(ch_names_raw, bytes):
            ch_names_raw = ch_names_raw.decode("utf-8")
        ch_names = json.loads(ch_names_raw) if isinstance(ch_names_raw, str) else ch_names_raw

        source = sample.get("file_name.txt")
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        if source is None:
            source = sample.get("__url__", "webdataset")


        data = {
            "timeseries": torch.from_numpy(timeseries),
            "ch_coords": _to_spherical(sample["ch_coords.npy"]),
            "ch_names": ch_names,
            "source": source,
        }
        if "events.npy" in sample:
            # tmin = 0.2
            # tmax = 1.0
            # pre_step = int(tmin * sfreq)
            # post_step = int(tmax * sfreq)
            events = sample["events.npy"][:, 0]
            window_start = start_offset if "start_offset" in locals() else strip_len
            window_end = window_start + target_len

            keep = (events > window_start) & (events < window_end)
            events = events[keep] - window_start
            selected_indices = np.random.choice(np.arange(events.shape[0]), size=min(5, events.shape[0]), replace=False)
            data['events'] = torch.tensor(events[selected_indices], dtype=torch.long)

            if "images.npy" in sample:
                images = list(sample['images.npy'][keep][selected_indices])
                data['images'] = images
                # inputs = image_processor(images=images, return_tensors="pt")
                # data['image_encoder_inputs'] = inputs
            else:
                image_ids = sample['image_ids.npy'][keep][selected_indices]
                sort_order = np.argsort(image_ids, kind="stable")
                sorted_ids = image_ids[sort_order]

                unique_ids, inverse = np.unique(sorted_ids, return_inverse=True)
                sorted_images = self.image_file['pixel_values'][unique_ids]
                sorted_images = sorted_images[inverse]

                restore_order = np.argsort(sort_order, kind="stable")
                images = sorted_images[restore_order]
                data['pixel_values'] = torch.from_numpy(images)

        data['sfreq'] = torch.tensor(s_freq, dtype=torch.float32)
        return data


import tarfile
import io
from PIL import Image
import tempfile
def make_dummy_wds_shard(
    out_tar_path: str,
    n: int = 3,
    image_size=(32, 32),
) -> str:
    """Create a tiny WebDataset shard (tar) with {__key__, jpg, cls, json} per sample."""
    W, H = image_size

    with tarfile.open(out_tar_path, "w") as tar:
        for i in range(n):
            key = f"{i:06d}"

            # --- create a tiny RGB image in memory ---
            arr = (np.random.rand(H, W, 3) * 255).astype("uint8")
            img = Image.fromarray(arr, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            jpg_bytes = buf.getvalue()

            # --- simple label + metadata ---
            cls_bytes = str(i % 2).encode("utf-8")
            meta_bytes = json.dumps({"id": i, "split": "test"}).encode("utf-8")

            def add_bytes(name: str, data: bytes):
                ti = tarfile.TarInfo(name=name)
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))

            # WebDataset grouping is by common prefix key
            add_bytes(f"{key}.jpg", jpg_bytes)
            add_bytes(f"{key}.cls", cls_bytes)
            add_bytes(f"{key}.json", meta_bytes)

    return out_tar_path

# @profile
def get_webdataset(
    dataset_names,
):
    
    # tmp = tempfile.mkdtemp()
    # shard = os.path.join(tmp, "dummy-000000.tar")
    # make_dummy_wds_shard(shard, n=5)
    # return wds.WebDataset(shard, resampled=True).decode()

    shards = []
    for name in dataset_names:
        files = glob(f"data/cache/{name}")
        if len(files) == 0:
            raise ValueError(f"No shards found for dataset '{name}' in 'data/cache/{name}/*.tar'")
        shards.extend(files)
    dataset = wds.WebDataset(shards, resampled=True).decode()
    dataset = dataset.map(_process_webdataset_sample())
    return dataset
