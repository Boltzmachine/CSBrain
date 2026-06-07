"""EgoBrain EEG + egocentric-video dataset for the world-model extension.

Mirrors :mod:`datasets.cinebrain_dataset` so the same ``WorldModelWrapper``
loss path consumes both datasets. Key differences:

* EgoBrain has 32 channels @ 256 Hz on a 10-20 montage (not 64-ch biosemi).
* One EDF per subject covers the whole ~6 h session, so the preprocessor
  slices it into fixed-length clips up front; see
  :mod:`datasets.egobrain_preprocess`.
* The egocentric video lives in one or more GoPro chapter files; the
  preprocessor picks a canonical one and records the EEG → video offset
  in the per-subject ``clips.json``.

The dataset surfaces every clip as one ``(W, C, N, d)`` window stack so a
single ``__getitem__`` covers both halves of a ``(x_t, x_{t+k})`` pair.
The collation and iterable-wrapper APIs are intentionally identical to
the CineBrain side so the existing mix-mode wiring just works.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from datasets.cinebrain_dataset import (
    preprocess_segment,
    _load_frame,
    _load_frames_batch,
    _frame_size_for,
    _get_vision_processor,
    _encoder_kind,
)


# Per-worker cache of opened HDF5 frame-cache file handles. Each DataLoader
# worker is a separate process (fork), so each gets its own cache.
_H5_FRAME_HANDLES: dict = {}


def _get_h5_frame_handle(path: str):
    """Return (and cache) an open h5py File for the per-subject frame cache."""
    h = _H5_FRAME_HANDLES.get(path)
    if h is None:
        import h5py
        h = h5py.File(path, 'r')
        _H5_FRAME_HANDLES[path] = h
    return h


# Cache the (mean, std) tensors for the active vision encoder so the fast
# path can do tensor normalization in one call without re-allocating.
_NORMALIZE_CACHE: dict = {}


def _get_normalize_params(vision_encoder: str) -> tuple[torch.Tensor, torch.Tensor]:
    cached = _NORMALIZE_CACHE.get(vision_encoder)
    if cached is not None:
        return cached
    proc = _get_vision_processor(vision_encoder)
    mean = torch.tensor(getattr(proc, 'image_mean', [0.485, 0.456, 0.406]),
                        dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(getattr(proc, 'image_std', [0.229, 0.224, 0.225]),
                       dtype=torch.float32).view(1, 3, 1, 1)
    _NORMALIZE_CACHE[vision_encoder] = (mean, std)
    return mean, std


# ---------------------------------------------------------------------------
# Channel coordinates — EgoBrain uses the 10-20 system. We look up the
# spherical coordinate for every channel name reported in the per-subject
# clips.json, dropping (and warning about) any name not found.
# ---------------------------------------------------------------------------

_10_20_CACHE: Optional[dict] = None


def _load_1020_montage() -> dict:
    global _10_20_CACHE
    if _10_20_CACHE is None:
        import mne
        mon = mne.channels.make_standard_montage('standard_1020')
        pos = mon.get_positions()['ch_pos']
        # Normalise names to upper-case for case-insensitive lookup.
        _10_20_CACHE = {n.upper(): np.asarray(p, dtype=np.float32)
                        for n, p in pos.items()}
    return _10_20_CACHE


def _cart_to_spherical(xyz: np.ndarray) -> np.ndarray:
    """(C, 3) Cartesian → (C, 3) spherical (r, theta, phi)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / (r + 1e-8))
    return np.stack([r, theta, phi], axis=-1).astype(np.float32)


def _resolve_ch_coords(ch_names: Sequence[str]
                       ) -> tuple[List[str], np.ndarray, np.ndarray]:
    """Look up 10-20 coordinates for ``ch_names``.

    Returns ``(kept_names, keep_mask, coords)`` where:
    * ``kept_names`` lists only channels with a known coordinate
    * ``keep_mask`` is a length-len(ch_names) bool array selecting them
    * ``coords`` is (n_kept, 3) spherical
    """
    mon = _load_1020_montage()
    keep_idx, keep_names = [], []
    xyz = []
    for i, name in enumerate(ch_names):
        key = name.upper().strip()
        # EmoTiv sometimes prefixes electrode names with "EEG " or appends
        # "-REF"; strip both so the 10-20 lookup still hits.
        for prefix in ('EEG ', 'EEG.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key.endswith('-REF'):
            key = key[:-4]
        if key in mon:
            keep_idx.append(i)
            keep_names.append(name)
            xyz.append(mon[key])
    if not keep_idx:
        raise RuntimeError(
            f"No EgoBrain channels matched the 10-20 montage; got {ch_names}")
    keep_mask = np.zeros(len(ch_names), dtype=bool)
    keep_mask[keep_idx] = True
    coords = _cart_to_spherical(np.stack(xyz, axis=0))
    return keep_names, keep_mask, coords


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EgoBrainDataset(Dataset):
    """EEG + egocentric-frame pairs for world-model pretraining.

    Each ``__getitem__`` returns ``n_windows`` consecutive 2 s EEG windows
    (stride 1 s by default) drawn from one preprocessed clip, plus the
    vision-encoder-ready frame stack aligned to each window's centre time.
    The schema matches :class:`CineBrainDataset` so the same
    ``collate_cinebrain`` and world-model wrapper apply.
    """

    def __init__(
        self,
        data_dir: str = 'data/EgoBrain',
        subjects: Sequence[str] = ('P0001',),
        in_dim: int = 200,
        n_windows: int = 3,
        window_s: float = 2.0,
        stride_s: float = 1.0,
        fs_out: int = 200,
        cache_dir: Optional[str] = None,
        frame_size: Optional[int] = None,
        erp_latency_s: float = 0.0,
        load_frames: bool = True,
        vision_encoder: str = 'facebook/dinov2-base',
        clip_s: float = 4.0,
        max_channels: Optional[int] = None,
        frames_cache_dir: Optional[str] = None,
        ea_matrices: Optional[dict] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.subjects = list(subjects)
        self.in_dim = in_dim
        self.n_windows = n_windows
        self.window_samples = int(round(window_s * fs_out))
        self.stride_samples = int(round(stride_s * fs_out))
        self.fs_out = fs_out
        self.clip_s = clip_s
        self.clip_samples = int(round(clip_s * fs_out))
        self.cache_dir = (cache_dir
                          or os.path.join(data_dir, f'cache_eeg_{fs_out}hz'))
        self.erp_latency_s = erp_latency_s
        self.load_frames = load_frames
        self.vision_encoder = vision_encoder
        self.frame_size = (frame_size if frame_size is not None
                           else _frame_size_for(vision_encoder))
        self.max_channels = max_channels
        # Optional per-subject Euclidean Alignment whitening matrices. When
        # provided, _load_clip applies R̄_s^{-1/2} after the keep_mask
        # selection and before windowing. Sidecar file lives next to the
        # EEG cache; see datasets/compute_ea.py + euclidean_alignment.py.
        self.ea_matrices = ea_matrices

        # If a per-subject HDF5 frame cache exists with attrs matching the
        # current configuration, use the fast path in __getitem__ that
        # reads pre-decoded uint8 frames instead of seeking into the
        # GoPro MP4 every call. The path is encoded with the cache-
        # invalidating settings; see datasets/egobrain_extract_frames.py.
        if frames_cache_dir is None:
            enc_slug = vision_encoder.replace('/', '_')
            frames_cache_dir = os.path.join(
                data_dir,
                f'cache_frames_{enc_slug}'
                f'_w{window_s}s{stride_s}'
                f'_e{erp_latency_s}_nw{n_windows}'
                f'_sz{self.frame_size}')
        self.frames_cache_dir = frames_cache_dir
        self.use_frames_cache = (load_frames and
                                 os.path.isdir(frames_cache_dir))

        assert self.window_samples % in_dim == 0, (
            f"window_samples={self.window_samples} must be a multiple of "
            f"in_dim={in_dim}")
        self.n_patches_per_window = self.window_samples // in_dim

        # Stride enough windows fit; the last window must end inside the
        # clip. ``__getitem__`` zero-pads any tail that runs past the end.
        max_n_windows = 1 + max(
            0, (self.clip_samples - self.window_samples) // self.stride_samples)
        if n_windows > max_n_windows:
            raise ValueError(
                f"clip_s={clip_s}s @ fs_out={fs_out} (window={window_s}s, "
                f"stride={stride_s}s) only supports n_windows<={max_n_windows} "
                f"without padding, got {n_windows}")

        # Per-subject metadata: ch_coords + clip count + video sync.
        self._subject_meta: dict[str, dict] = {}
        self._items: List[tuple[str, int]] = []
        for sub in self.subjects:
            meta_path = os.path.join(self.cache_dir, sub, 'clips.json')
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"missing {meta_path}; run "
                    f"`python -m datasets.egobrain_preprocess "
                    f"--data_dir {data_dir} --subjects {sub}` first")
            with open(meta_path) as f:
                meta = json.load(f)
            kept_names, keep_mask, coords = _resolve_ch_coords(
                meta['ch_names'])
            if max_channels is not None and len(kept_names) > max_channels:
                kept_names = kept_names[:max_channels]
                keep_mask_idx = np.where(keep_mask)[0][:max_channels]
                keep_mask = np.zeros_like(keep_mask)
                keep_mask[keep_mask_idx] = True
                coords = coords[:max_channels]

            self._subject_meta[sub] = {
                'ch_names': kept_names,
                'ch_coords': torch.from_numpy(coords).float(),
                'keep_mask': keep_mask,
                'n_clips': int(meta['n_clips']),
                'video': meta.get('video'),
                'clip_s_cache': float(meta['clip_s']),
                'fs_out_cache': int(meta['fs_out']),
            }
            assert meta['fs_out'] == fs_out, (
                f'cache fs_out={meta["fs_out"]} != dataset fs_out={fs_out}')
            assert abs(meta['clip_s'] - clip_s) < 1e-6, (
                f"cache clip_s={meta['clip_s']} != requested clip_s={clip_s}; "
                f"rebuild cache with the new clip length")
            for c in range(meta['n_clips']):
                self._items.append((sub, c))
        if not self._items:
            raise RuntimeError(
                f"EgoBrainDataset({self.cache_dir}) contains zero clips")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def _clip_path(self, sub: str, c: int) -> str:
        return os.path.join(self.cache_dir, sub, f'{c}.npy')

    def _load_clip(self, sub: str, c: int) -> np.ndarray:
        """(C_kept, fs_out*clip_s) float32 µV — already filtered/notched.

        When ``self.ea_matrices`` is set, applies the subject's Euclidean
        Alignment whitening matrix after channel selection. Falls back to
        the raw signal if no matrix is registered for this subject (e.g.
        the EA sidecar was built before this subject was downloaded).
        """
        arr = np.load(self._clip_path(sub, c))
        keep = self._subject_meta[sub]['keep_mask']
        if keep.shape[0] != arr.shape[0]:
            # The keep_mask was built from the cached ch_names list, so any
            # mismatch is a stale cache; rebuild it.
            raise RuntimeError(
                f"cached clip {self._clip_path(sub, c)} has C={arr.shape[0]} "
                f"but clips.json lists {keep.shape[0]} channels; rebuild cache")
        arr = arr[keep]
        if self.ea_matrices is not None and sub in self.ea_matrices:
            from datasets.euclidean_alignment import apply_ea
            arr = apply_ea(arr, self.ea_matrices[sub])
        return arr

    def _video_chapters(self, sub: str) -> list[dict]:
        """Return the per-subject ordered chapter list (with abs paths +
        cumulative time offsets), or [] when the subject ships no video.

        Memoised: the cumulative-offset computation runs once per subject
        on first access. Supports both the new ``video.chapters`` schema
        and a one-off legacy ``video.path`` shape (single chapter), which
        is treated as a one-element list so old preprocess caches keep
        working without re-running.
        """
        cache = self._subject_meta[sub].get('_chapters_cached')
        if cache is not None:
            return cache
        v = self._subject_meta[sub].get('video')
        if v is None:
            self._subject_meta[sub]['_chapters_cached'] = []
            return []
        if 'chapters' in v:
            chapters = list(v['chapters'])
        else:                                                # legacy fallback
            chapters = [{'path': v['path']}]
        out: list[dict] = []
        cum = 0.0
        for ch in chapters:
            entry = {
                'path': os.path.join(self.data_dir, ch['path']),
                'start_s': cum,
                'duration_s': ch.get('duration_s'),
                'fps': ch.get('fps'),
                'n_frames': ch.get('n_frames'),
            }
            out.append(entry)
            if entry['duration_s'] is not None:
                cum += float(entry['duration_s'])
            else:
                # Unknown duration — leave cum unchanged; chapter routing
                # below will treat the chapter as covering [start_s, +inf)
                # so a missing-duration legacy entry still serves as a
                # fallback for clip 0..N (matching pre-stitching behavior).
                cum = float('inf')
        self._subject_meta[sub]['_chapters_cached'] = out
        return out

    def _video_offset_s(self, sub: str) -> float:
        v = self._subject_meta[sub].get('video') or {}
        return float(v.get('video_offset_s', 0.0))

    def _resolve_chapter(self, sub: str, eeg_t: float
                         ) -> Optional[tuple[dict, float]]:
        """Return (chapter_entry, t_in_chapter) for an EEG timestamp.

        Returns ``None`` when ``eeg_t`` falls past the last chapter
        (or before t=0). The chapter list is sorted in time, so a linear
        scan is fast enough — n_chapters is at most a few dozen.
        """
        t_video = eeg_t - self._video_offset_s(sub)
        if t_video < 0:
            return None
        chapters = self._video_chapters(sub)
        if not chapters:
            return None
        for ch in chapters:
            dur = ch.get('duration_s')
            if dur is None or t_video < ch['start_s'] + dur:
                return ch, t_video - ch['start_s']
        return None

    def __getitem__(self, idx: int) -> dict:
        sub, c = self._items[idx]
        ts = self._load_clip(sub, c)                          # (C, T)
        C = ts.shape[0]

        # Window-slice the clip.
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
        windows_np = np.stack(windows, axis=0)               # (W, C, T)
        timeseries = torch.from_numpy(
            windows_np.reshape(self.n_windows, C,
                               self.n_patches_per_window, self.in_dim)
            .astype(np.float32))                              # (W, C, N, d)

        ch_coords = self._subject_meta[sub]['ch_coords']      # (C, 3)
        ch_names = list(self._subject_meta[sub]['ch_names'])  # length C

        # Frames aligned to window centres (+ optional ERP latency shift).
        pixel_values = torch.zeros(
            self.n_windows, 3, self.frame_size, self.frame_size,
            dtype=torch.float32)
        has_image = torch.zeros(self.n_windows, dtype=torch.bool)
        used_cache = False
        if self.load_frames and self.use_frames_cache:
            # Fast path: read pre-decoded uint8 frames from the per-subject
            # HDF5 cache. Frames were stored after HF-equivalent resize +
            # center-crop, so all that's left here is rescale to [0,1]
            # and ImageNet mean/std normalization — no PIL, no HF
            # processor call. ~15-25 ms/sample vs ~500 ms live decode.
            h5_path = os.path.join(self.frames_cache_dir, f'{sub}.h5')
            if os.path.exists(h5_path):
                h = _get_h5_frame_handle(h5_path)
                frames_uint8 = h['frames'][c]            # (W, H, W_, 3) uint8
                ok = h['has_image'][c]                   # (W,) bool
                mean, std = _get_normalize_params(self.vision_encoder)
                x = torch.from_numpy(np.ascontiguousarray(frames_uint8))
                x = x.float().div_(255.0).permute(0, 3, 1, 2)  # (W,3,H,W_)
                x = (x - mean) / std
                pixel_values = x
                has_image = torch.from_numpy(np.asarray(ok)).bool()
                used_cache = True
        if self.load_frames and not used_cache:
            chapters = self._video_chapters(sub)
            if chapters:
                # For each window, find which GoPro chapter holds the
                # corresponding video timestamp. GoPro chapters cap at
                # ~11 min, so a 2 h session needs all 13 chapters to be
                # routed correctly.
                t0 = c * self.clip_s
                per_chapter: dict[str, list[tuple[int, float]]] = {}
                for i in range(self.n_windows):
                    win_center = (i * self.stride_samples
                                  + self.window_samples / 2) / self.fs_out
                    t_eeg = t0 + win_center + self.erp_latency_s
                    resolved = self._resolve_chapter(sub, t_eeg)
                    if resolved is None:
                        continue
                    ch, t_in_ch = resolved
                    per_chapter.setdefault(ch['path'], []).append((i, t_in_ch))
                for ch_path, wanted in per_chapter.items():
                    if not os.path.exists(ch_path):
                        continue
                    slots = [w[0] for w in wanted]
                    t_videos = [w[1] for w in wanted]
                    try:
                        # Batched decode + HF preprocess in one pass — windows
                        # within the same chapter usually map to nearby GOPs.
                        frames = _load_frames_batch(
                            ch_path, t_videos, self.vision_encoder)
                        for k, i in enumerate(slots):
                            pixel_values[i] = frames[k]
                            has_image[i] = True
                    except (OSError, RuntimeError) as _e:
                        # Out-of-range timestamps and decode errors leave
                        # has_image=False so the alignment loss skips this
                        # window; never silently ignore other exceptions.
                        print(f'[egobrain] batch frame load failed for '
                              f'{ch_path} @ {t_videos}: {_e}')

        return {
            'timeseries': timeseries,
            'ch_coords': ch_coords,
            'ch_names': ch_names,
            'pixel_values': pixel_values,
            'has_image': has_image,
            'source': 'egobrain',
            'session_id': sub,
            'subject': sub,
            'local_clip_idx': c,
        }


# ---------------------------------------------------------------------------
# Collation — identical schema to ``collate_cinebrain`` so the world-model
# loss path can consume either dataset interchangeably.
# ---------------------------------------------------------------------------


def collate_egobrain(batch):
    B = len(batch)
    # EgoBrain subjects can have slightly different kept-channel counts
    # (e.g. when a subject had a noisy electrode dropped); pad to the
    # batch max so the stack succeeds.
    max_C = max(b['timeseries'].shape[1] for b in batch)
    W, _, N, d = batch[0]['timeseries'].shape

    ts_all = torch.zeros(B, W, max_C, N, d, dtype=torch.float32)
    ch_coords = torch.zeros(B, max_C, 3, dtype=torch.float32)
    valid_channel_mask = torch.zeros(B, max_C, dtype=torch.bool)
    for i, b in enumerate(batch):
        c = b['timeseries'].shape[1]
        ts_all[i, :, :c] = b['timeseries']
        ch_coords[i, :c] = b['ch_coords']
        valid_channel_mask[i, :c] = True

    pixel_values_all = torch.stack(
        [b['pixel_values'] for b in batch], dim=0)
    has_image_all = torch.stack([b['has_image'] for b in batch], dim=0)
    valid_length_mask = torch.ones(B, N, dtype=torch.bool)

    return {
        'timeseries': ts_all[:, 0],
        'ch_coords': ch_coords,
        'ch_names': [b['ch_names'] for b in batch],
        'valid_channel_mask': valid_channel_mask,
        'valid_length_mask': valid_length_mask,
        'timeseries_future': ts_all,
        'pixel_values_future': pixel_values_all,
        'has_image_future': has_image_all,
        'image_encoder_inputs': {
            'pixel_values': pixel_values_all[:, 0],
        },
        'has_image': has_image_all[:, 0],
        'source': [b['source'] for b in batch],
        'session_id': [b.get('session_id', 'unknown') for b in batch],
    }


# ---------------------------------------------------------------------------
# Iterable wrapper for mix-mode training (same contract as
# ``CineBrainIterableWrapper``).
# ---------------------------------------------------------------------------


class EgoBrainIterableWrapper(IterableDataset):
    def __init__(self, dataset: EgoBrainDataset, seed: Optional[int] = None):
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
                'timeseries': sample['timeseries'][0],
                'ch_coords': sample['ch_coords'],
                'ch_names': sample['ch_names'],
                'source': sample['source'],
                'session_id': sample['session_id'],
                'sfreq': sfreq,
            }
            if bool(sample['has_image'][0].item()):
                out['pixel_values'] = sample['pixel_values'][0]
            if self.has_future:
                out['timeseries_future'] = sample['timeseries']
                out['pixel_values_future'] = sample['pixel_values']
                out['has_image_future'] = sample['has_image']
            yield out
