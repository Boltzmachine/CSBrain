"""CineBrain EEG + video dataset for the world-model extension.

Per the official README, EEG is recorded at 1000 Hz on 64 channels (plus ECG
as the last channel). Recording is segmented into non-overlapping 0.8 s
windows; a 4 s video clip corresponds to 5 consecutive EEG segments. Each
subject has ≈27000 EEG segments (6 h) covering 5400 video clips.

This loader groups the 5 EEG segments that share a clip, resamples the
concatenated 4 s recording to 200 Hz, applies the standard band-pass/notch,
slices it into overlapping 2 s windows (stride 1 s), and pairs the first
window with a later window as (x_t, x_{t+k}).

Mapping from a subject's local clip index to a global clip file (under
``data/CineBrain/clips``) is not documented in the released files. We expose
a ``clip_id_fn`` hook so the caller can supply the mapping when it becomes
known; by default we assume the local index equals the global clip index
(wrapped modulo the number of clips on disk). The prediction loss is
invariant to this mapping; the cross-modal alignment loss is not and will
degrade if the mapping is wrong.
"""

from __future__ import annotations

import json
import math
import os
import random
from glob import glob
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Preprocessing helpers — use MNE's filter/notch/resample so the CineBrain
# pipeline matches whatever MNE-based preprocessing the rest of the project
# (and the original dataset README) already assumes.
# ---------------------------------------------------------------------------

def preprocess_segment(x: np.ndarray, fs_in: int = 1000,
                       fs_out: int = 200,
                       l_freq: float = 0.3, h_freq: float = 75.0,
                       notch_hz: float = 60.0) -> np.ndarray:
    """Band-pass, notch, and resample a (C, T) EEG recording via MNE."""
    import mne
    # MNE expects float64 and channels-first — already our layout.
    x = np.ascontiguousarray(x, dtype=np.float64)
    x = mne.filter.filter_data(
        x, sfreq=fs_in, l_freq=l_freq, h_freq=h_freq, verbose='ERROR')
    x = mne.filter.notch_filter(
        x, Fs=fs_in, freqs=notch_hz, verbose='ERROR')
    if fs_in != fs_out:
        g = math.gcd(int(fs_in), int(fs_out))
        x = mne.filter.resample(
            x, up=fs_out // g, down=fs_in // g, verbose='ERROR')
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Channel coordinates — biosemi64 is the closest documented 64-ch montage to
# the one used by CineBrain. We build a deterministic spherical-coordinate
# array for the first 64 channels; ECG/aux channels are dropped.
# ---------------------------------------------------------------------------

def _biosemi64_coords_spherical() -> tuple[list[str], np.ndarray]:
    # MNE is already a hard dependency of this file via ``preprocess_segment``;
    # failing to build the montage means something is genuinely wrong, so
    # let the exception surface instead of falling back to zero coords
    # (which would silently destroy any coord-conditioned computation).
    import mne
    mon = mne.channels.make_standard_montage('biosemi64')
    ch_names = mon.ch_names
    pos = mon.get_positions()['ch_pos']
    xyz = np.stack([pos[n] for n in ch_names], axis=0).astype(np.float32)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / (r + 1e-8))
    sph = np.stack([r, theta, phi], axis=-1).astype(np.float32)
    return ch_names, sph


_BIOSEMI64_NAMES, _BIOSEMI64_COORDS = _biosemi64_coords_spherical()


# ---------------------------------------------------------------------------
# Video frame loader — decord is the only backend that actually installs
# cleanly in the `cbramod` env; we import lazily to keep the test suite
# runnable in environments without video support.
# ---------------------------------------------------------------------------

_DECORD_BRIDGE_SET = False
_VISION_PROCESSOR_CACHE: dict = {}

# Per-process LRU cache of decord.VideoReader instances, keyed by file path.
# Opening a multi-GB GoPro MP4 takes ~2 s on the first access (header parse +
# index build); reusing the open reader drops random-access frame load to
# ~0.5 s per frame. The cache is per-DataLoader-worker because each worker
# is a separate process (fork); cache size is bounded so we don't keep more
# than a handful of large file handles open.
from collections import OrderedDict as _OrderedDict
_VIDEO_READER_CACHE: '_OrderedDict[str, object]' = _OrderedDict()
_VIDEO_READER_CACHE_MAX = 4


def _encoder_kind(vision_encoder: str) -> str:
    """Cheap heuristic: model id contains 'vjepa' => video encoder, else image."""
    return 'vjepa2' if 'vjepa' in vision_encoder.lower() else 'image'


def _frame_size_for(vision_encoder: str) -> int:
    """Default crop size matched to the processor (V-JEPA 2 = 256, DINOv2 = 224)."""
    return 256 if _encoder_kind(vision_encoder) == 'vjepa2' else 224


def _get_vision_processor(vision_encoder: str):
    """Return (and cache) the right HF processor for ``vision_encoder``."""
    if vision_encoder not in _VISION_PROCESSOR_CACHE:
        if _encoder_kind(vision_encoder) == 'vjepa2':
            from transformers import AutoVideoProcessor
            proc = AutoVideoProcessor.from_pretrained(vision_encoder)
        else:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(vision_encoder)
        _VISION_PROCESSOR_CACHE[vision_encoder] = proc
    return _VISION_PROCESSOR_CACHE[vision_encoder]


def _get_video_reader(clip_path: str):
    """Return a (potentially cached) decord.VideoReader for ``clip_path``.

    On cache miss, opens a new reader and evicts the least-recently-used
    entry once the cache reaches ``_VIDEO_READER_CACHE_MAX``. On cache hit,
    promotes the entry to most-recently-used so a hot file stays warm.
    """
    global _DECORD_BRIDGE_SET
    import decord
    if not _DECORD_BRIDGE_SET:
        decord.bridge.set_bridge('native')
        _DECORD_BRIDGE_SET = True
    cached = _VIDEO_READER_CACHE.get(clip_path)
    if cached is not None:
        _VIDEO_READER_CACHE.move_to_end(clip_path)
        return cached
    vr = decord.VideoReader(clip_path)
    _VIDEO_READER_CACHE[clip_path] = vr
    if len(_VIDEO_READER_CACHE) > _VIDEO_READER_CACHE_MAX:
        _VIDEO_READER_CACHE.popitem(last=False)
    return vr


def _load_frame(clip_path: str, t_seconds: float,
                vision_encoder: str = 'facebook/dinov2-base') -> torch.Tensor:
    """Return a vision-encoder-ready pixel_values tensor (3, H, W).

    Dispatches on ``vision_encoder``: V-JEPA 2 ids go through
    ``AutoVideoProcessor`` (256x256, V-JEPA 2 mean/std); other ids go through
    ``AutoImageProcessor`` (e.g. 224x224 ImageNet for DINOv2-base).
    """
    vr = _get_video_reader(clip_path)
    fps = vr.get_avg_fps()
    # CineBrain clips are 24 fps per the paper; if decord can't recover
    # the fps header we'd rather crash than silently mis-index frames.
    assert fps and fps > 0, f"decord could not read fps from {clip_path}"
    idx = int(round(t_seconds * fps))
    idx = max(0, min(len(vr) - 1, idx))
    frame = vr[idx].asnumpy()  # (H, W, C), uint8
    proc = _get_vision_processor(vision_encoder)
    if _encoder_kind(vision_encoder) == 'vjepa2':
        # Wrap as a single-frame video: list of videos, each a list of frames.
        processed = proc(videos=[[frame]], return_tensors='pt')
        return processed['pixel_values_videos'][0, 0]
    processed = proc(images=frame, return_tensors='pt')
    return processed['pixel_values'][0]


def _load_frames_batch(clip_path: str,
                       t_seconds_list: Sequence[float],
                       vision_encoder: str = 'facebook/dinov2-base'
                       ) -> torch.Tensor:
    """Batched variant of :func:`_load_frame` for multi-window clips.

    Decodes all requested frames with a single ``vr.get_batch`` (sequential
    GOP read, much cheaper than per-call ``vr[idx]`` seeks when frames are
    close in time) and runs the HF processor on the batch in one shot.
    Returns ``(K, 3, H, W)`` aligned to ``t_seconds_list`` order.
    """
    vr = _get_video_reader(clip_path)
    fps = vr.get_avg_fps()
    assert fps and fps > 0, f"decord could not read fps from {clip_path}"
    n = len(vr)
    indices = [max(0, min(n - 1, int(round(t * fps)))) for t in t_seconds_list]
    # ``get_batch`` requires monotonically-non-decreasing indices on most
    # decord builds; sort + unscatter to preserve caller order.
    order = sorted(range(len(indices)), key=lambda i: indices[i])
    sorted_idx = [indices[i] for i in order]
    frames_native = vr.get_batch(sorted_idx).asnumpy()  # (K, H, W, C) uint8
    # Undo the sort: out[order[k]] = frames_native[k]
    unsorted = [None] * len(indices)
    for k, src in enumerate(order):
        unsorted[src] = frames_native[k]
    proc = _get_vision_processor(vision_encoder)
    if _encoder_kind(vision_encoder) == 'vjepa2':
        processed = proc(videos=[[f] for f in unsorted], return_tensors='pt')
        return processed['pixel_values_videos'][:, 0]
    processed = proc(images=list(unsorted), return_tensors='pt')
    return processed['pixel_values']


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CineBrainDataset(Dataset):
    """EEG + next-video-frame pairs for world-model pretraining.

    Each ``__getitem__`` returns a 4 s continuous EEG recording (after
    filter + resample) sliced into ``n_windows`` 2 s windows (stride 1 s),
    plus DINOv2-ready pixel tensors for the frames aligned to each window.

    The caller pairs windows at training time: given horizon ``k``, the
    ``x_t`` target is window ``i`` and the ``x_{t+k}`` target is window
    ``i+k``. We return the whole per-clip stack once (3 windows / 3 frames
    by default) so both halves come from the same preprocessing pass.
    """

    def __init__(
        self,
        data_dir: str = 'data/CineBrain',
        subjects: Sequence[str] = ('sub-0001',),
        in_dim: int = 200,
        n_windows: int = 3,
        window_s: float = 2.0,
        stride_s: float = 1.0,
        fs_out: int = 200,
        fs_in: int = 1000,
        eeg_subdir: str = 'eeg_02',
        clips_subdir: str = 'clips',
        n_eeg_channels: int = 64,
        frame_size: Optional[int] = None,
        erp_latency_s: float = 0.0,
        clip_id_fn: Optional[Callable[[str, int], int]] = None,
        load_frames: bool = True,
        cache_dir: Optional[str] = None,
        vision_encoder: str = 'facebook/dinov2-base',
        ea_matrices: Optional[dict] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.subjects = list(subjects)
        # Per-subject EA whitening matrices (see datasets/euclidean_alignment.py).
        # Applied in _load_preprocessed when present.
        self.ea_matrices = ea_matrices
        self.in_dim = in_dim
        self.n_windows = n_windows
        self.window_samples = int(round(window_s * fs_out))
        self.stride_samples = int(round(stride_s * fs_out))
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.eeg_subdir = eeg_subdir
        self.clips_subdir = clips_subdir
        self.n_eeg_channels = n_eeg_channels
        self.vision_encoder = vision_encoder
        # Default crop size depends on the encoder (V-JEPA 2 = 256, DINOv2 = 224).
        self.frame_size = frame_size if frame_size is not None else _frame_size_for(vision_encoder)
        self.erp_latency_s = erp_latency_s
        self.clip_id_fn = clip_id_fn
        self.load_frames = load_frames
        # Resolved cache root. The offline preprocessor (see
        # ``datasets/cinebrain_preprocess.py``) writes one (C, fs_out*4)
        # float32 ``.npy`` per clip into this layout:
        #   <cache_dir>/<subject>/<clip_idx>.npy
        # When a cache hit occurs, the expensive MNE filter/notch/resample
        # step is skipped entirely.
        self.cache_dir = (
            cache_dir or os.path.join(data_dir, f'cache_eeg_{fs_out}hz'))

        # Enforce window compatibility with the encoder's patch layout:
        # each window of length window_samples must be divisible by in_dim
        # so it reshapes cleanly into (N, in_dim) patches.
        assert self.window_samples % in_dim == 0, (
            f"window_samples={self.window_samples} must be a multiple of "
            f"in_dim={in_dim}")
        self.n_patches_per_window = self.window_samples // in_dim

        # Build the flat list of (subject, local_clip_idx) samples.
        self._items: List[tuple[str, int]] = []
        for sub in self.subjects:
            print(f'Indexing subject {sub}...')
            eeg_dir = os.path.join(data_dir, sub, eeg_subdir)
            files = [f for f in os.listdir(eeg_dir)
                     if f.endswith('.npy') and f[:-4].isdigit()]
            idx_max = max(int(f[:-4]) for f in files)
            n_clips = (idx_max + 1) // 5  # 5 EEG segments per 4 s clip
            for c in range(n_clips):
                # Require all 5 segments to be present (subjects with
                # occasional gaps are skipped here — rare in practice).
                if all(os.path.exists(os.path.join(
                        eeg_dir, f'{c * 5 + j}.npy')) for j in range(5)):
                    self._items.append((sub, c))

        # Discover available clip files once.
        clips_dir = os.path.join(data_dir, clips_subdir)
        clip_files = sorted(glob(os.path.join(clips_dir, '*.mp4')))
        self._clip_files = clip_files
        self._n_global_clips = len(clip_files)
        if self._n_global_clips == 0 and load_frames:
            raise RuntimeError(
                f"No video clips found in {clips_dir}; set load_frames=False"
                " or provide the clip directory.")

        # Preload ch coords + names once; all subjects share them.
        self._ch_names = _BIOSEMI64_NAMES[:n_eeg_channels]
        self._ch_coords = torch.from_numpy(
            _BIOSEMI64_COORDS[:n_eeg_channels]).float()

    # --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def _load_concat_recording(self, sub: str, c: int) -> np.ndarray:
        """Concatenate 5 consecutive 0.8 s segments → (C, 4000) @ 1000 Hz."""
        eeg_dir = os.path.join(self.data_dir, sub, self.eeg_subdir)
        chunks = []
        for j in range(5):
            arr = np.load(os.path.join(eeg_dir, f'{c * 5 + j}.npy'))
            # Each segment: (69, 800). Keep only EEG channels.
            chunks.append(arr[:self.n_eeg_channels])
        return np.concatenate(chunks, axis=-1)

    def _cache_path(self, sub: str, c: int) -> str:
        return os.path.join(self.cache_dir, sub, f'{c}.npy')

    def _load_preprocessed(self, sub: str, c: int) -> np.ndarray:
        """Return (C, fs_out*4) float32 µV-scaled, ±300 clipped.

        Prefers the offline cache; falls back to on-the-fly filter/notch/
        resample via MNE. The fallback is 2–3 orders of magnitude slower,
        so training runs should ALWAYS populate the cache first (see
        ``datasets/cinebrain_preprocess.py``).

        When ``self.ea_matrices`` is set, the subject's Euclidean Alignment
        whitening is applied as a final post-cache step.
        """
        cache = self._cache_path(sub, c)
        if os.path.exists(cache):
            ts = np.load(cache)
        else:
            raw = self._load_concat_recording(sub, c)            # (C, 4000)
            ts = preprocess_segment(raw, self.fs_in, self.fs_out)  # (C, 800)
            ts = ts * 1e6 if np.abs(ts).max() < 1.0 else ts
            np.clip(ts, -300, 300, out=ts)
            ts = ts.astype(np.float32)
        if self.ea_matrices is not None and sub in self.ea_matrices:
            from datasets.euclidean_alignment import apply_ea
            ts = apply_ea(ts, self.ea_matrices[sub])
        return ts

    def _resolve_clip_path(self, sub: str, c: int) -> str:
        if self.clip_id_fn is not None:
            gid = self.clip_id_fn(sub, c)
        else:
            gid = c
        gid = int(gid) % self._n_global_clips
        return self._clip_files[gid]

    def __getitem__(self, idx: int) -> dict:
        sub, c = self._items[idx]
        ts = self._load_preprocessed(sub, c)               # (C, fs_out*4)

        # Slice into overlapping windows: (n_windows, C, window_samples)
        windows = []
        for i in range(self.n_windows):
            s = i * self.stride_samples
            e = s + self.window_samples
            if e > ts.shape[-1]:
                pad = e - ts.shape[-1]
                w = np.pad(ts[:, s:], ((0, 0), (0, pad)))
            else:
                w = ts[:, s:e]
            windows.append(w)
        windows_np = np.stack(windows, axis=0)             # (W, C, T)

        # Reshape each window to (C, N, in_dim) patches.
        W, C, T = windows_np.shape
        windows_patched = windows_np.reshape(
            W, C, self.n_patches_per_window, self.in_dim).astype(np.float32)
        timeseries = torch.from_numpy(windows_patched)     # (W, C, N, d)

        # Frames aligned to each window centre (+ ERP latency).
        pixel_values = torch.zeros(
            self.n_windows, 3, self.frame_size, self.frame_size,
            dtype=torch.float32)
        has_image = torch.zeros(self.n_windows, dtype=torch.bool)
        if self.load_frames:
            clip_path = self._resolve_clip_path(sub, c)
            t_videos = [
                (i * self.stride_samples + self.window_samples / 2)
                / self.fs_out + self.erp_latency_s
                for i in range(self.n_windows)
            ]
            try:
                # Batched decode + HF preprocess. CineBrain clips are short
                # (~4 s, 24 fps) so the windows live in one or two GOPs;
                # batch + sort gives near-sequential read.
                frames = _load_frames_batch(
                    clip_path, t_videos, self.vision_encoder)
                pixel_values = frames
                has_image[:] = True
            except (OSError, RuntimeError) as _e:
                # Tolerate decode/IO errors — alignment loss skips this
                # sample via has_image. Other exceptions bubble up so we
                # don't silently mask real bugs.
                print(f'[cinebrain] batch frame load failed for {clip_path} '
                      f'@ {t_videos}: {_e}')

        return {
            'timeseries': timeseries,                      # (W, C, N, d)
            'ch_coords': self._ch_coords,                  # (C, 3)
            'ch_names': list(self._ch_names),
            'pixel_values': pixel_values,                  # (W, 3, H, W)
            'has_image': has_image,                        # (W,)
            'source': 'cinebrain',
            'session_id': sub,
            'subject': sub,
            'local_clip_idx': c,
        }


# ---------------------------------------------------------------------------
# Collation — build the stacked tensors and the valid_*_mask fields the
# CSBrainAlign encoder expects. Returns a batch where the first-window half
# lives at key ``timeseries`` (so the existing alignment/recon loss fires)
# and the remaining windows live at ``timeseries_future``.
# ---------------------------------------------------------------------------

def collate_cinebrain(batch):
    # Expected per-item shape: timeseries (W, C, N, d)
    B = len(batch)
    W, C, N, d = batch[0]['timeseries'].shape

    ts_all = torch.stack([b['timeseries'] for b in batch], dim=0)  # (B, W, C, N, d)
    ch_coords = torch.stack([b['ch_coords'] for b in batch], dim=0)
    pixel_values_all = torch.stack([b['pixel_values'] for b in batch], dim=0)
    has_image_all = torch.stack([b['has_image'] for b in batch], dim=0)

    valid_channel_mask = torch.ones(B, C, dtype=torch.bool)
    valid_length_mask = torch.ones(B, N, dtype=torch.bool)

    return {
        # Primary window = window 0.
        'timeseries': ts_all[:, 0],
        'ch_coords': ch_coords,
        'ch_names': [b['ch_names'] for b in batch],
        'valid_channel_mask': valid_channel_mask,
        'valid_length_mask': valid_length_mask,
        # Future windows stacked along a new W axis: (B, W, C, N, d)
        'timeseries_future': ts_all,
        'pixel_values_future': pixel_values_all,     # (B, W, 3, H, W)
        'has_image_future': has_image_all,           # (B, W)
        'image_encoder_inputs': {
            'pixel_values': pixel_values_all[:, 0],  # align loss uses window 0
        },
        'has_image': has_image_all[:, 0],
        'source': [b['source'] for b in batch],
        'session_id': [b.get('session_id', 'unknown') for b in batch],
    }


# ---------------------------------------------------------------------------
# Iterable adapter — yield single-window CineBrain samples in the dict format
# consumed by ``datasets.cached_dataset.collate_cached``. Used to mix
# CineBrain with the Alljoined webdataset under a single dataloader.
#
# Requires ``n_windows=1`` upstream so we don't waste preprocessing on windows
# that would just be discarded; mixed batches don't carry a future stack.
# ---------------------------------------------------------------------------

class CineBrainIterableWrapper(IterableDataset):
    """Adapt :class:`CineBrainDataset` to an IterableDataset whose items
    match the schema expected by ``collate_cached``.

    When the underlying dataset has ``n_windows > 1`` we additionally
    emit per-sample future-window fields (``timeseries_future``,
    ``pixel_values_future``, ``has_image_future``) so a future-aware
    collate can route them to the world-model predictor without
    touching the Alljoined samples in the same batch.

    Iteration is infinite — the outer mixer / DataLoader bounds the epoch.
    Each worker draws independent random indices from its own RNG.
    """

    def __init__(self, dataset: CineBrainDataset, seed: Optional[int] = None):
        super().__init__()
        assert dataset.n_windows >= 1
        self.dataset = dataset
        self.seed = seed
        self.has_future = dataset.n_windows > 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.seed is not None:
            base_seed = self.seed + (
                worker_info.id if worker_info is not None else 0)
        elif worker_info is not None:
            base_seed = worker_info.seed
        else:
            base_seed = None
        rng = np.random.default_rng(base_seed)
        n = len(self.dataset)
        sfreq = torch.tensor(self.dataset.fs_out, dtype=torch.float32)
        while True:
            idx = int(rng.integers(0, n))
            sample = self.dataset[idx]
            out = {
                'timeseries': sample['timeseries'][0],   # (C, N, d)
                'ch_coords': sample['ch_coords'],        # (C, 3)
                'ch_names': sample['ch_names'],          # list[str]
                'source': sample['source'],
                'session_id': sample['session_id'],
                'sfreq': sfreq,
            }
            if bool(sample['has_image'][0].item()):
                out['pixel_values'] = sample['pixel_values'][0]  # (3, H, W)
            if self.has_future:
                out['timeseries_future'] = sample['timeseries']      # (W, C, N, d)
                out['pixel_values_future'] = sample['pixel_values']  # (W, 3, H, W)
                out['has_image_future'] = sample['has_image']        # (W,)
            yield out


class WeightedSampleMix(IterableDataset):
    """Sample from multiple iterable sources by fixed probabilities.

    The mixer yields raw per-sample dicts; downstream batching and collation
    happen in ``DataLoader``. Sources are expected to be infinite (or to
    restart cleanly on ``StopIteration``); the mixer bounds the epoch via
    ``samples_per_epoch`` so ``DataLoader`` knows when to stop.
    """

    def __init__(self, sources: Sequence,
                 weights: Sequence[float],
                 samples_per_epoch: Optional[int] = None):
        super().__init__()
        assert len(sources) == len(weights), (
            "sources and weights must have the same length")
        self.sources = list(sources)
        w = np.asarray(weights, dtype=np.float64)
        assert (w > 0).all(), "all mix weights must be > 0"
        self.probs = (w / w.sum()).tolist()
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        if self.samples_per_epoch is None:
            raise TypeError(
                "WeightedSampleMix has no length unless samples_per_epoch "
                "is set")
        return self.samples_per_epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = (worker_info.seed
                if worker_info is not None else None)
        rng = np.random.default_rng(seed)
        iters = [iter(s) for s in self.sources]
        if self.samples_per_epoch is None:
            n_per_worker = None
        elif worker_info is not None:
            n_per_worker = math.ceil(
                self.samples_per_epoch / worker_info.num_workers)
        else:
            n_per_worker = self.samples_per_epoch
        count = 0
        while n_per_worker is None or count < n_per_worker:
            i = int(rng.choice(len(iters), p=self.probs))
            try:
                yield next(iters[i])
            except StopIteration:
                iters[i] = iter(self.sources[i])
                yield next(iters[i])
            count += 1
