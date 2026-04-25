import os
import re
from collections import defaultdict, deque

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
    max_lens = max(x['timeseries'].shape[1] for x in batch)
    for x in batch:
        chs = x['timeseries'].shape[0]
        lens = x['timeseries'].shape[1]
        if chs < max_chs:
            padding_chs = max_chs - chs
            x['timeseries'] = torch.cat([x['timeseries'], torch.zeros(padding_chs, x['timeseries'].shape[1], x['timeseries'].shape[2])], dim=0)
            x['ch_coords'] = torch.cat([x['ch_coords'], torch.zeros(padding_chs, 3)], dim=0)
            x['ch_names'] = list(x['ch_names']) + ['pad'] * padding_chs
        if lens < max_lens:
            padding_len = max_lens - lens
            x['timeseries'] = torch.cat([x['timeseries'], torch.zeros(x['timeseries'].shape[0], padding_len, x['timeseries'].shape[2])], dim=1)
        x['valid_channel_mask'] = torch.tensor([1] * chs + [0] * (max_chs - chs), dtype=torch.bool)
        x['valid_length_mask'] = torch.tensor([1] * lens + [0] * (max_lens - lens), dtype=torch.bool)

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

    all_pixel_values = []
    has_image = []
    for item in batch:
        if 'pixel_values' in item:
            all_pixel_values.append(item['pixel_values'])
            has_image.append(True)
        else:
            all_pixel_values.append(torch.zeros((3, 224, 224), dtype=torch.float32))
            has_image.append(False)
    image_pixel_values = torch.stack(all_pixel_values, dim=0)
    has_image = torch.tensor(has_image, dtype=torch.bool)

    text_embedding = None
    text_sample = next((x['text_embedding'] for x in batch if 'text_embedding' in x), None)
    if text_sample is not None:
        text_dim = text_sample.shape[-1]
        text_embedding = torch.stack([
            x['text_embedding'] if 'text_embedding' in x
            else torch.zeros(text_dim, dtype=text_sample.dtype)
            for x in batch
        ], dim=0)

    return {
        'timeseries': torch.stack([x['timeseries'] for x in batch], dim=0),
        'ch_coords': torch.stack([x['ch_coords'] for x in batch], dim=0),
        'ch_names': [x['ch_names'] for x in batch],
        'valid_channel_mask': torch.stack([x['valid_channel_mask'] for x in batch], dim=0),
        "valid_length_mask": torch.stack([x['valid_length_mask'] for x in batch], dim=0),
        'source': [x['source'] for x in batch],
        'image_hidden_states': torch.stack([x['image_hidden_states'] for x in batch if 'image_hidden_states' in x], dim=0) if 'image_hidden_states' in batch[0] else None,
        'events': torch.stack([x['events'] for x in batch if 'events' in x], dim=0) if 'events' in batch[0] else None,
        'sfreq': torch.stack([x['sfreq'] for x in batch], dim=0) if 'sfreq' in batch[0] else None,
        "image_encoder_inputs": {
            'pixel_values': image_pixel_values
        },
        "text_embedding": text_embedding,
        "has_image": has_image,
        "source": [x['source'] for x in batch],
        "session_id": [x.get('session_id', 'unknown') for x in batch],
    }


def collate_cached_with_future(batch):
    """Like :func:`collate_cached` but also stacks per-sample future-window
    fields when present (CineBrain in mix-mode training).

    Adds these keys when at least one sample carries a future stack:

    * ``cinebrain_idx``        — (M,) long, row indices of those samples
    * ``timeseries_future``    — (M, W, C, N, d)
    * ``pixel_values_future``  — (M, W, 3, H, W)
    * ``has_image_future``     — (M, W) bool

    The world-model wrapper uses ``cinebrain_idx`` to restrict its
    latent-prediction loss to rows that actually have a future stack;
    rows without one (Alljoined) only contribute to masked recon and
    image alignment.

    Assumes the future stack's channel count equals the batch's max
    channel count (true today: CineBrain has 64ch, Alljoined has ≤64).
    """
    out = collate_cached(batch)
    cb_indices = [i for i, b in enumerate(batch) if 'timeseries_future' in b]
    if not cb_indices:
        return out
    ts_future_stack = torch.stack(
        [batch[i]['timeseries_future'] for i in cb_indices], dim=0)
    max_chs = out['timeseries'].size(1)
    assert ts_future_stack.size(2) == max_chs, (
        f"future-window channels ({ts_future_stack.size(2)}) must match "
        f"the batch's max channel count ({max_chs}); add channel padding "
        f"to collate_cached_with_future if a future-providing dataset has "
        f"fewer channels than another mixed source"
    )
    out['cinebrain_idx'] = torch.tensor(cb_indices, dtype=torch.long)
    out['timeseries_future'] = ts_future_stack
    out['pixel_values_future'] = torch.stack(
        [batch[i]['pixel_values_future'] for i in cb_indices], dim=0)
    out['has_image_future'] = torch.stack(
        [batch[i]['has_image_future'] for i in cb_indices], dim=0)
    return out


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
    def __init__(self, params, event_window_target_len=None):
        self.image_file = None
        self.things_image_file = None
        self.text_embed_file = None
        self.params = params
        # When set, the event-centered crop is `event_window_target_len`
        # samples drawn from a random offset within [t-40, t+s_freq); when
        # None we keep the historical fixed 240-sample window so existing
        # configs (e.g. Spectral pretraining) are untouched.
        self.event_window_target_len = event_window_target_len

    def __call__(self, sample):
        if self.image_file is None:
            self.image_file = h5py.File("data/Alljoined-1.6M/pixel_values.h5", 'r')
            self.things_image_file = h5py.File("data/THINGS-EEG2/pixel_values.h5", 'r')
            text_embed_path = "data/Alljoined-1.6M/text_embeddings.h5"
            if os.path.exists(text_embed_path):
                self.text_embed_file = h5py.File(text_embed_path, 'r')
        s_freq = 200
        do_strip = 'tueg' in str(sample.get("source", "")) or 'tueg' in sample.get("__url__", "") or 'tueg' in sample.get("source.txt", "")
        ds = sample["timeseries.npy"]

        full_len = ds.shape[1]
        strip_len = 60 * s_freq if do_strip else 0
        valid_len = full_len - (2 * strip_len)

        selected_event_idx = None
        if "events.npy" in sample and sample["events.npy"].shape[0] > 0:
            events = sample["events.npy"][:, 0].astype(np.int64)
            selected_event_idx = random.randint(0, events.shape[0] - 1)
            t = int(events[selected_event_idx])

            # Event window: [t - 40, t + s_freq), 240 samples by default.
            event_window_start = t - 40
            event_window_end = t + s_freq
            event_window_len = event_window_end - event_window_start

            if self.event_window_target_len is not None:
                target_len = self.event_window_target_len
                assert target_len <= event_window_len, (
                    f"event_window_target_len={target_len} cannot exceed "
                    f"event_window_len={event_window_len}"
                )
                offset = random.randint(0, event_window_len - target_len)
                window_start = event_window_start + offset
                window_end = window_start + target_len
            else:
                target_len = event_window_len
                window_start = event_window_start
                window_end = event_window_end

            read_start = max(0, window_start)
            read_end = min(full_len, window_end)
            timeseries = ds[:, read_start:read_end]

            pad_left = max(0, -window_start)
            pad_right = max(0, window_end - full_len)
            if pad_left > 0 or pad_right > 0:
                timeseries = np.pad(timeseries, ((0, 0), (pad_left, pad_right)))

            assert timeseries.shape[1] == target_len, f"Expected event-window crop of length {target_len}, but got {timeseries.shape[1]}"
        elif sample.get("source.txt", "") == "things-eeg2":
            target_len = 12 * self.params.in_dim
            if valid_len > target_len:
                start_offset = strip_len + random.randint(0, valid_len - target_len)
                timeseries = ds[:, start_offset : start_offset + target_len]
            else:
                timeseries = ds[:, strip_len : full_len - strip_len]
                padding = target_len - timeseries.shape[1]
                if padding > 0:
                    timeseries = np.pad(timeseries, ((0, 0), (0, padding)))
        else:
            target_len = self.params.seq_len * self.params.in_dim
            if valid_len > target_len:
                start_offset = strip_len + random.randint(0, valid_len - target_len)
                timeseries = ds[:, start_offset : start_offset + target_len]
            else:
                timeseries = ds[:, strip_len : full_len - strip_len]
                padding = target_len - timeseries.shape[1]
                if padding > 0:
                    timeseries = np.pad(timeseries, ((0, 0), (0, padding)))

        np.clip(timeseries, -300, 300, out=timeseries)

        timeseries = timeseries.reshape(timeseries.shape[0], -1, self.params.in_dim).astype(np.float32)

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
        if selected_event_idx is not None:
            if "images.npy" in sample:
                raise
                selected_image = sample["images.npy"][selected_event_idx]
                data["pixel_values"] = torch.from_numpy(np.asarray(selected_image))
            elif "image_ids.npy" in sample:
                image_id = int(sample["image_ids.npy"][selected_event_idx])
                if sample.get('source.txt') == "things-eeg2":
                    split = sample['split.txt']
                    data['pixel_values'] = torch.from_numpy(self.things_image_file[f"{split}/pixel_values"][image_id - 1])
                elif sample['source.txt'] == "Alljoined-1.6M":
                    data["pixel_values"] = torch.from_numpy(self.image_file["pixel_values"][image_id])
                    if self.text_embed_file is not None:
                        data["text_embedding"] = torch.from_numpy(
                            self.text_embed_file["text_embeddings"][image_id]
                        )
                else:
                    raise ValueError(f"Unexpected source '{sample.get('source.txt')}' for image retrieval")
        # if "events.npy" in sample:
        #     # tmin = 0.2
        #     # tmax = 1.0
        #     # pre_step = int(tmin * sfreq)
        #     # post_step = int(tmax * sfreq)
        #     events = sample["events.npy"][:, 0]
        #     window_start = start_offset if "start_offset" in locals() else strip_len
        #     window_end = window_start + target_len

        #     keep = (events > window_start) & (events < window_end)
        #     events = events[keep] - window_start
        #     selected_indices = np.random.choice(np.arange(events.shape[0]), size=min(5, events.shape[0]), replace=False)
        #     data['events'] = torch.tensor(events[selected_indices], dtype=torch.long)

        #     if "images.npy" in sample:
        #         images = list(sample['images.npy'][keep][selected_indices])
        #         data['images'] = images
        #         # inputs = image_processor(images=images, return_tensors="pt")
        #         # data['image_encoder_inputs'] = inputs
        #     else:
        #         image_ids = sample['image_ids.npy'][keep][selected_indices]
        #         sort_order = np.argsort(image_ids, kind="stable")
        #         sorted_ids = image_ids[sort_order]

        #         unique_ids, inverse = np.unique(sorted_ids, return_inverse=True)
        #         sorted_images = self.image_file['pixel_values'][unique_ids]
        #         sorted_images = sorted_images[inverse]

        #         restore_order = np.argsort(sort_order, kind="stable")
        #         images = sorted_images[restore_order]
        #         data['pixel_values'] = torch.from_numpy(images)

        data['sfreq'] = torch.tensor(s_freq, dtype=torch.float32)
        data['source'] = sample.get('source.txt', None) or sample.get("source", None) or "unknown"
        data['session_id'] = self._extract_session_id(sample)
        return data

    @staticmethod
    def _extract_session_id(sample):
        # Explicit field (written by updated preprocessing)
        if 'session_id.txt' in sample:
            sid = sample['session_id.txt']
            return sid.decode('utf-8') if isinstance(sid, bytes) else sid
        # Parse from Alljoined-style key: sub-XX_sess-XX_block-XX_evt_XXXXXXXX
        key = sample.get('__key__', '')
        m = re.match(r'(sub-\d+_sess-\d+)', key)
        if m:
            return m.group(1)
        # Fall back to file_name (each recording file ≈ one session)
        file_name = sample.get('file_name.txt', key)
        if isinstance(file_name, bytes):
            file_name = file_name.decode('utf-8')
        source = sample.get('source.txt', 'unknown')
        if isinstance(source, bytes):
            source = source.decode('utf-8')
        return f"{source}/{file_name}"


class SessionGroupedLoader:
    """Wraps a DataLoader that yields *individual samples* and re-batches
    them into session-grouped batches in the main process.

    Why this exists
    ---------------
    WebDataset reads tar shards sequentially — each shard contains ~2 k
    samples from **one** session.  A per-worker session batcher would have
    to wait for one shard to finish before seeing the next session.

    By moving the grouping into the main process we benefit from the
    DataLoader's multi-worker interleaving: with ``num_workers=8``, eight
    different shards (≈ eight different sessions) are read concurrently and
    their samples arrive interleaved, so the grouper fills up quickly.

    Parameters
    ----------
    loader : DataLoader
        Must yield individual sample dicts (``batch_size=None`` with a
        WebDataset that has *no* ``.batched()`` stage).
    samples_per_session : int
        How many samples to draw from each session in one batch.
    sessions_per_batch : int
        Target number of distinct sessions per batch.
        Effective batch size = samples_per_session × sessions_per_batch.
    collation_fn : callable
        ``collate_cached`` or equivalent.
    min_sessions : int
        Minimum sessions required to emit a partial batch when the stream
        is temporarily stuck on one session.
    batches_per_epoch : int | None
        If set, stop iteration after this many batches (replaces
        ``with_epoch``).
    """

    def __init__(self, loader, samples_per_session, sessions_per_batch,
                 collation_fn, min_sessions=2, batches_per_epoch=None):
        self.loader = loader
        self.samples_per_session = samples_per_session
        self.sessions_per_batch = sessions_per_batch
        self.collation_fn = collation_fn
        self.min_sessions = min_sessions
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        if self.batches_per_epoch:
            return self.batches_per_epoch
        # rough estimate
        batch_sz = self.samples_per_session * self.sessions_per_batch
        return max(1, len(self.loader) // batch_sz)

    # ------------------------------------------------------------------ #

    def __iter__(self):
        session_buffers = defaultdict(list)
        ready = deque()
        ready_set = set()
        n_batches = 0
        n_skipped = 0

        for sample in self.loader:
            sid = sample.get('session_id', 'unknown')

            # Cap per-session buffer — drop excess from already-full sessions
            if sid in ready_set:
                n_skipped += 1
            else:
                session_buffers[sid].append(sample)
                n_skipped = 0
                if len(session_buffers[sid]) >= self.samples_per_session:
                    ready.append(sid)
                    ready_set.add(sid)

            # Primary: emit a full batch
            while len(ready) >= self.sessions_per_batch:
                yield self._pop_batch(session_buffers, ready, ready_set,
                                      self.sessions_per_batch)
                n_batches += 1
                n_skipped = 0
                if self.batches_per_epoch and n_batches >= self.batches_per_epoch:
                    return

            # Fallback: stuck in a same-session burst — emit partial batch
            if (n_skipped > self.samples_per_session * 4
                    and len(ready) >= self.min_sessions):
                n_emit = min(len(ready), self.sessions_per_batch)
                yield self._pop_batch(session_buffers, ready, ready_set,
                                      n_emit)
                n_batches += 1
                n_skipped = 0
                if self.batches_per_epoch and n_batches >= self.batches_per_epoch:
                    return

    def _pop_batch(self, session_buffers, ready, ready_set, n_sessions):
        batch_samples = []
        for _ in range(n_sessions):
            sid = ready.popleft()
            ready_set.discard(sid)
            batch_samples.extend(session_buffers[sid][:self.samples_per_session])
            remaining = session_buffers[sid][self.samples_per_session:]
            if remaining:
                session_buffers[sid] = remaining
                if len(remaining) >= self.samples_per_session:
                    ready.append(sid)
                    ready_set.add(sid)
            else:
                del session_buffers[sid]
        random.shuffle(batch_samples)
        return self.collation_fn(batch_samples)


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
    params,
    event_window_target_len=None,
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
    dataset = dataset.map(_process_webdataset_sample(
        params, event_window_target_len=event_window_target_len))
    return dataset
