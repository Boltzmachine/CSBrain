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
from torch.utils.data import Dataset


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
_DINOV2_PROCESSOR = None


def _get_dinov2_processor():
    # Cached so repeated frame loads don't reparse the processor config.
    global _DINOV2_PROCESSOR
    if _DINOV2_PROCESSOR is None:
        from transformers import AutoImageProcessor
        _DINOV2_PROCESSOR = AutoImageProcessor.from_pretrained(
            'facebook/dinov2-base')
    return _DINOV2_PROCESSOR


def _load_frame(clip_path: str, t_seconds: float) -> torch.Tensor:
    """Return a DINOv2-ready pixel_values tensor (3, H, W).

    We defer resize + ImageNet normalisation to
    ``AutoImageProcessor.from_pretrained('facebook/dinov2-base')`` so the
    tensor matches exactly what the encoder expects, including any future
    config changes (rescale factor, interpolation, crop size).
    """
    global _DECORD_BRIDGE_SET
    import decord
    if not _DECORD_BRIDGE_SET:
        decord.bridge.set_bridge('native')
        _DECORD_BRIDGE_SET = True
    vr = decord.VideoReader(clip_path)
    fps = vr.get_avg_fps()
    # CineBrain clips are 24 fps per the paper; if decord can't recover
    # the fps header we'd rather crash than silently mis-index frames.
    assert fps and fps > 0, f"decord could not read fps from {clip_path}"
    idx = int(round(t_seconds * fps))
    idx = max(0, min(len(vr) - 1, idx))
    frame = vr[idx].asnumpy()  # (H, W, C), uint8
    processed = _get_dinov2_processor()(images=frame, return_tensors='pt')
    return processed['pixel_values'][0]  # (3, H, W)


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
        frame_size: int = 224,
        erp_latency_s: float = 0.0,
        clip_id_fn: Optional[Callable[[str, int], int]] = None,
        load_frames: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.subjects = list(subjects)
        self.in_dim = in_dim
        self.n_windows = n_windows
        self.window_samples = int(round(window_s * fs_out))
        self.stride_samples = int(round(stride_s * fs_out))
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.eeg_subdir = eeg_subdir
        self.clips_subdir = clips_subdir
        self.n_eeg_channels = n_eeg_channels
        self.frame_size = frame_size
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
        """
        cache = self._cache_path(sub, c)
        if os.path.exists(cache):
            return np.load(cache)
        raw = self._load_concat_recording(sub, c)            # (C, 4000)
        ts = preprocess_segment(raw, self.fs_in, self.fs_out)  # (C, 800)
        ts = ts * 1e6 if np.abs(ts).max() < 1.0 else ts
        np.clip(ts, -300, 300, out=ts)
        return ts.astype(np.float32)

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
            for i in range(self.n_windows):
                window_center_s = (
                    i * self.stride_samples + self.window_samples / 2
                ) / self.fs_out + self.erp_latency_s
                try:
                    pixel_values[i] = _load_frame(clip_path, window_center_s)
                    has_image[i] = True
                except (OSError, RuntimeError) as _e:
                    # Tolerate decode/IO errors — alignment loss skips
                    # this sample via has_image. Other exceptions bubble
                    # up so we don't silently mask real bugs (that's how
                    # the earlier TypeError went unnoticed).
                    print(f'[cinebrain] frame load failed for {clip_path} '
                          f'@ {window_center_s:.2f}s: {_e}')

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
