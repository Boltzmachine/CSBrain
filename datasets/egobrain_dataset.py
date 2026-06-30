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


# Per-worker cache of opened HDF5 hand-label-cache file handles (same fork
# rationale as the frame handles above).
_H5_HAND_HANDLES: dict = {}


# Per-worker cache of opened HDF5 embedding-cache file handles (datasets/
# egobrain_extract_embeddings.py output: cls/cls_flip/grid/grid_flip).
_H5_EMB_HANDLES: dict = {}


def _get_h5_emb_handle(path: str):
    """Return (and cache) an open h5py File for the per-subject embedding cache."""
    h = _H5_EMB_HANDLES.get(path)
    if h is None:
        import h5py
        h = h5py.File(path, 'r')
        _H5_EMB_HANDLES[path] = h
    return h


def _get_h5_hand_handle(path: str):
    """Return (and cache) an open h5py File for the per-subject hand-label cache
    (datasets/egobrain_hand_labels.py output: arrays (n_clips, n_windows))."""
    h = _H5_HAND_HANDLES.get(path)
    if h is None:
        import h5py
        h = h5py.File(path, 'r')
        _H5_HAND_HANDLES[path] = h
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
        emb_cache_dir: Optional[str] = None,
        use_embeddings: bool = False,
        hand_labels_dir: Optional[str] = None,
        frame_grid_dir: Optional[str] = None,
        use_frame_grid: bool = False,
        frame_grid_s: float = 0.2,
        temporal_jitter: bool = True,
        jitter_seed: Optional[int] = None,
        use_grid_embeddings: bool = False,
        emb_grid_dir: Optional[str] = None,
        ea_matrices: Optional[dict] = None,
        delta_whiten_g0: float = 1.0,
        delta_whiten_cutoff_hz: float = 8.0,
        motion_resample: bool = False,
        motion_resample_alpha: float = 1.0,
        motion_resample_space: str = 'patch',
        motion_resample_metric: str = 'cos',
        motion_resample_cap_pct: float = 99.0,
        motion_resample_floor_mix: float = 0.1,
        motion_emb_dir: Optional[str] = None,
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

        # ME->MI "rank-1 delta whitening": a per-channel low-frequency gain
        # (g0 at DC, raised-cosine ramp to 1.0 at delta_whiten_cutoff_hz) that
        # attenuates EgoBrain's motor-execution delta floor toward the shallower
        # motor-imagery (PhysioNet-MI) level. g0>=1.0 is a no-op (default off).
        # Applied in _load_clip after channel selection. See utils.util
        # .apply_delta_whiten and project_me_mi_pretrain_manipulation. NOTE: do
        # not combine with EA without recomputing the EA matrices on the
        # whitened signal (whitening changes the channel covariance).
        self.delta_whiten_g0 = delta_whiten_g0
        self.delta_whiten_cutoff_hz = delta_whiten_cutoff_hz

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

        # The clip-keyed embedding cache and the time-keyed frame grid index
        # frames differently (clip,window vs absolute-time slot); they cannot
        # be combined. Check before either block so the error is unambiguous.
        if use_frame_grid and use_embeddings:
            raise ValueError(
                "[EgoBrain] use_frame_grid is incompatible with "
                "use_embeddings: the embedding cache is keyed by (clip, "
                "window); the grid path samples arbitrary offsets. Pick one.")

        # Optional per-subject HDF5 vision-encoder EMBEDDING cache
        # (datasets/egobrain_extract_embeddings.py): pre-computed cls/cls_flip
        # (B,d) + grid/grid_flip (B,P,d) for the frozen DINOv2 encoder, so the
        # model skips the on-the-fly forward. Opt-in via use_embeddings; the dir
        # slug encodes the same cache-invalidating knobs as the frame cache.
        if emb_cache_dir is None:
            enc_slug = vision_encoder.replace('/', '_')
            emb_cache_dir = os.path.join(
                data_dir,
                f'cache_embeddings_{enc_slug}'
                f'_w{window_s}s{stride_s}'
                f'_e{erp_latency_s}_nw{n_windows}'
                f'_sz{self.frame_size}')
        self.emb_cache_dir = emb_cache_dir
        self.use_emb_cache = (use_embeddings and load_frames
                              and os.path.isdir(emb_cache_dir))
        if use_embeddings and load_frames and not self.use_emb_cache:
            raise FileNotFoundError(
                f"[EgoBrain] --use_cached_embeddings set but embedding cache "
                f"not found at '{emb_cache_dir}'.\n"
                f"  Build it once with:\n"
                f"    conda run -n cbramod python -m datasets.egobrain_extract_embeddings \\\n"
                f"      --data_dir {data_dir} --subjects all \\\n"
                f"      --vision_encoder {vision_encoder} \\\n"
                f"      --window_s {window_s} --stride_s {stride_s} "
                f"--erp_latency_s {erp_latency_s} --n_windows {n_windows}\n"
                f"  (window/stride/erp/n_windows must match this run, or the "
                f"cache dir name won't match.)")

        # ------------------------------------------------------------------
        # Continuous, time-keyed frame grid (datasets/
        # egobrain_extract_frames_grid.py): the knob-agnostic successor to the
        # clip-keyed frame cache. When enabled, __getitem__ samples an EEG
        # window at an arbitrary frame_grid_s-snapped offset across the WHOLE
        # continuous recording (no 4 s clip boundary) and looks up the aligned
        # frame by ABSOLUTE EEG-clock time (slot = round((window_centre + erp)
        # / frame_grid_s)). erp/window/stride are applied here at lookup, not
        # baked into the cache. Off by default -> the existing clip-keyed path
        # below is used verbatim (so prior results reproduce bit-for-bit).
        self.use_frame_grid = bool(use_frame_grid)
        self.frame_grid_s = float(frame_grid_s)
        self.grid_samples = int(round(frame_grid_s * fs_out))
        self.erp_samples = int(round(erp_latency_s * fs_out))
        self.temporal_jitter = bool(temporal_jitter)
        self.jitter_seed = jitter_seed
        self._anchor_rng = None
        self.frame_grid_dir = frame_grid_dir
        # Optional time-keyed DINOv2 embedding cache (datasets/
        # egobrain_extract_embeddings_grid.py): the grid counterpart of the
        # clip-keyed embedding cache. When on, __getitem__ reads cls/grid (+flip)
        # at the slot for each window instead of pixels, so the model skips the
        # live encoder — encoder-skip speedup AND continuous-offset variety.
        self.use_grid_embeddings = bool(use_grid_embeddings)
        self.emb_grid_dir = emb_grid_dir
        if self.use_grid_embeddings and not self.use_frame_grid:
            raise ValueError(
                "[EgoBrain] use_grid_embeddings requires use_frame_grid=True "
                "(it is the time-keyed embedding cache for the grid path).")
        if self.use_frame_grid:
            if self.frame_grid_dir is None:
                enc_slug = vision_encoder.replace('/', '_')
                self.frame_grid_dir = os.path.join(
                    data_dir,
                    f'cache_frames_grid_{enc_slug}'
                    f'_g{frame_grid_s}_sz{self.frame_size}')
            if self.use_grid_embeddings:
                # Reading embeddings; the uint8 frame grid is not needed at
                # train time (it was only the extractor's input).
                if self.emb_grid_dir is None:
                    enc_slug = vision_encoder.replace('/', '_')
                    self.emb_grid_dir = os.path.join(
                        data_dir,
                        f'cache_embeddings_grid_{enc_slug}'
                        f'_g{frame_grid_s}_sz{self.frame_size}')
                if load_frames and not os.path.isdir(self.emb_grid_dir):
                    raise FileNotFoundError(
                        f"[EgoBrain] use_grid_embeddings set but grid embedding "
                        f"cache not found at '{self.emb_grid_dir}'.\n"
                        f"  Build it once (after the frame grid) with:\n"
                        f"    conda run -n cbramod python -m "
                        f"datasets.egobrain_extract_embeddings_grid \\\n"
                        f"      --data_dir {data_dir} --subjects all \\\n"
                        f"      --vision_encoder {vision_encoder} "
                        f"--grid_s {frame_grid_s}\n"
                        f"  (only vision_encoder/frame_size/grid_s affect the "
                        f"dir name.)")
            elif load_frames and not os.path.isdir(self.frame_grid_dir):
                raise FileNotFoundError(
                    f"[EgoBrain] use_frame_grid set but grid frame cache not "
                    f"found at '{self.frame_grid_dir}'.\n"
                    f"  Build it once with:\n"
                    f"    conda run -n cbramod python -m "
                    f"datasets.egobrain_extract_frames_grid \\\n"
                    f"      --data_dir {data_dir} --subjects all \\\n"
                    f"      --vision_encoder {vision_encoder} "
                    f"--grid_s {frame_grid_s}\n"
                    f"  (only vision_encoder/frame_size/grid_s affect the dir "
                    f"name; window/stride/erp/n_windows do NOT.)")
            # The time-keyed grid supersedes the clip-keyed frame cache.
            self.use_frames_cache = False

        # Optional per-subject HDF5 hand-movement-annotation cache
        # (datasets/egobrain_hand_labels.py): per-window continuous
        # left/right intensities surfaced as 'hand_targets'/'hand_valid' for the
        # auxiliary regression objective. Off (zeros + all-False mask) when the
        # dir is absent. The cache's window slug MUST match this dataset's
        # window_s/stride_s/erp_latency_s/n_windows/clip_s/fs_out or the
        # (clip, window) keys silently misalign — same contract as the frames.
        self.hand_labels_dir = hand_labels_dir
        self.use_hand_labels = (hand_labels_dir is not None
                                and os.path.isdir(hand_labels_dir))
        if load_frames and not self.use_frames_cache and not self.use_frame_grid:
            # Live-decoding 4-5 GB GoPro MP4s on the fly is slow AND
            # memory-heavy — it OOM-kills DataLoader workers at scale. The
            # cache is mandatory; fail fast instead of silently falling back.
            raise FileNotFoundError(
                f"[EgoBrain] frame cache not found at '{frames_cache_dir}'.\n"
                f"  Live GoPro MP4 decode is disabled (too slow / OOMs at "
                f"scale). Build the cache once with:\n"
                f"    conda run -n cbramod python -m datasets.egobrain_extract_frames \\\n"
                f"      --data_dir {data_dir} --subjects all \\\n"
                f"      --vision_encoder {vision_encoder} \\\n"
                f"      --window_s {window_s} --stride_s {stride_s} "
                f"--erp_latency_s {erp_latency_s} --n_windows {n_windows} "
                f"--num_workers 12\n"
                f"  (window/stride/erp/n_windows must match this run, or the "
                f"cache dir name won't match.)")

        assert self.window_samples % in_dim == 0, (
            f"window_samples={self.window_samples} must be a multiple of "
            f"in_dim={in_dim}")
        self.n_patches_per_window = self.window_samples // in_dim

        # Stride enough windows fit; the last window must end inside the
        # clip. ``__getitem__`` zero-pads any tail that runs past the end.
        max_n_windows = 1 + max(
            0, (self.clip_samples - self.window_samples) // self.stride_samples)
        if n_windows > max_n_windows and not self.use_frame_grid:
            # In grid mode windows are sliced from the CONTINUOUS recording
            # (clips stitched), so the per-clip limit doesn't apply; the true
            # bound is the subject length, enforced in _sample_base_slot.
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

        # Surface a PARTIAL embedding cache loudly. A video subject whose
        # per-subject embedding HDF5 is missing still reports has_image=True
        # from the frame cache, so __getitem__ yields no frame_* tensors and
        # collate zero-fills that (still-valid) image row — training then sees
        # all-zero CLS/grid "positives". We don't change that behaviour here,
        # but warn so a half-built cache isn't mistaken for a complete one.
        if self.use_emb_cache:
            missing_emb = [
                s for s in self.subjects
                if self._subject_meta[s].get('video') is not None
                and not os.path.exists(
                    os.path.join(self.emb_cache_dir, f'{s}.h5'))]
            if missing_emb:
                import warnings
                warnings.warn(
                    f"[EgoBrain] use_embeddings is on but the embedding cache "
                    f"'{self.emb_cache_dir}' is MISSING {len(missing_emb)} "
                    f"video subject(s): {','.join(missing_emb)}. Their image "
                    f"rows keep has_image=True (from the frame cache) yet carry "
                    f"no cached embeddings, so collate zero-fills them and the "
                    f"alignment/frame objectives train on bogus all-zero "
                    f"targets. Re-run datasets.egobrain_extract_embeddings for "
                    f"those subjects before training.", stacklevel=2)

        # ------------------------------------------------------------------
        # Motion-weighted anchor resampling (grid mode only). EgoBrain's video
        # is mostly static, so the uniform anchor draw in ``_sample_base_slot``
        # spends ~half the budget on frozen frames where the world-model's
        # next-frame prediction is trivial. When enabled, the anchor ``k`` is
        # drawn ∝ ``clip(motion(k), 0, p_cap)^alpha`` (mixed with a uniform
        # floor), biasing toward visually dynamic moments — genuine hand/object
        # manipulation; see datasets/egobrain_motion.py + the distribution study
        # in outputs/eval_tables.md. Only WITHIN-subject position is reweighted
        # (cross-subject balance is unchanged). Off by default so prior runs
        # reproduce bit-for-bit. The per-subject sampling CDF is precomputed once
        # here (main process) so the DataLoader-worker forks inherit it.
        self.motion_resample = bool(motion_resample)
        self.motion_resample_alpha = float(motion_resample_alpha)
        self.motion_resample_space = motion_resample_space
        self.motion_resample_metric = motion_resample_metric
        self.motion_resample_cap_pct = float(motion_resample_cap_pct)
        self.motion_resample_floor_mix = float(motion_resample_floor_mix)
        self.motion_emb_dir = motion_emb_dir
        self._anchor_cdf: dict[str, Optional[tuple]] = {}
        if self.motion_resample:
            self._init_motion_resample()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def _clip_path(self, sub: str, c: int) -> str:
        return os.path.join(self.cache_dir, sub, f'{c}.npy')

    def _select_and_whiten(self, arr: np.ndarray, sub: str) -> np.ndarray:
        """Channel-select (keep_mask) -> optional EA whitening -> optional
        delta-whitening. Shared verbatim by the clip-keyed (``_load_clip``)
        and the continuous grid (``_load_continuous_raw``) paths so both apply
        the identical transform in the identical order.

        When ``self.ea_matrices`` is set, applies the subject's Euclidean
        Alignment whitening matrix after channel selection. Falls back to
        the raw signal if no matrix is registered for this subject (e.g.
        the EA sidecar was built before this subject was downloaded).
        """
        keep = self._subject_meta[sub]['keep_mask']
        if keep.shape[0] != arr.shape[0]:
            # The keep_mask was built from the cached ch_names list, so any
            # mismatch is a stale cache; rebuild it.
            raise RuntimeError(
                f"cached EEG for {sub} has C={arr.shape[0]} but clips.json "
                f"lists {keep.shape[0]} channels; rebuild cache")
        arr = arr[keep]
        if self.ea_matrices is not None and sub in self.ea_matrices:
            from datasets.euclidean_alignment import apply_ea
            arr = apply_ea(arr, self.ea_matrices[sub])
        if self.delta_whiten_g0 is not None and self.delta_whiten_g0 < 1.0:
            from utils.util import apply_delta_whiten
            arr = apply_delta_whiten(
                arr, self.fs_out, self.delta_whiten_g0,
                self.delta_whiten_cutoff_hz)
        return arr

    def _load_clip(self, sub: str, c: int) -> np.ndarray:
        """(C_kept, fs_out*clip_s) float32 µV — already filtered/notched."""
        return self._select_and_whiten(np.load(self._clip_path(sub, c)), sub)

    # ------------------------------------------------------------------
    # Continuous (grid-mode) helpers
    # ------------------------------------------------------------------

    def _get_anchor_rng(self):
        """Lazily build a per-worker RNG for temporal-jitter anchor sampling.

        Seeded from the DataLoader worker seed (which PyTorch re-rolls per
        epoch) so each epoch sees fresh offsets; ``jitter_seed`` overrides for
        reproducible runs/tests.
        """
        if self._anchor_rng is None:
            if self.jitter_seed is not None:
                seed = int(self.jitter_seed)
                wi = torch.utils.data.get_worker_info()
                if wi is not None:
                    seed += 100003 * int(wi.id)
            else:
                wi = torch.utils.data.get_worker_info()
                seed = int(wi.seed) if wi is not None else os.getpid()
            self._anchor_rng = np.random.default_rng(seed)
        return self._anchor_rng

    def _anchor_k_bounds(self, sub: str) -> tuple[int, int]:
        """Inclusive valid range ``[k_min, k_max]`` of the window-0 frame slot
        for ``sub``: ``eeg_start_0 = k*grid - erp`` must lie in ``[0, total -
        span]``. Shared by ``_sample_base_slot`` and the resample precompute so
        both index the anchor axis identically."""
        n_clips = self._subject_meta[sub]['n_clips']
        total = n_clips * self.clip_samples
        span = (self.n_windows - 1) * self.stride_samples + self.window_samples
        grid = self.grid_samples
        erp = self.erp_samples
        k_min = (erp + grid - 1) // grid                 # ceil(erp/grid)
        k_max = (total - span + erp) // grid             # floor
        if k_max < k_min:
            k_max = k_min
        return k_min, k_max

    def _sample_base_slot(self, sub: str, c: int) -> int:
        """Base frame slot ``k`` for window 0. The FRAME is the discrete side
        (pre-stored on the 0.2 s grid) so we pin it to slot ``k`` and slide the
        CONTINUOUS EEG to match: window-0 START = ``k*grid_samples - erp_samples``
        so the frame at slot ``k`` sits exactly at ``start + erp``. Then every
        window's frame slot is an exact integer for ANY erp (no rounding) — see
        ``_getitem_grid``. ``temporal_jitter`` picks ``k`` over the valid range —
        uniformly, or (when ``motion_resample`` is on and this subject has a
        sampling CDF) biased toward visually dynamic anchors; otherwise the
        deterministic ``k`` puts window-0's start at ~clip ``c`` (reproducible
        eval — never reweighted)."""
        k_min, k_max = self._anchor_k_bounds(sub)
        if self.temporal_jitter:
            if self.motion_resample:
                ent = self._anchor_cdf.get(sub)
                if ent is not None:
                    k0, cdf = ent
                    u = float(self._get_anchor_rng().random())
                    off = int(np.searchsorted(cdf, u, side='right'))
                    return k0 + min(off, cdf.shape[0] - 1)
            return int(self._get_anchor_rng().integers(k_min, k_max + 1))
        k = int(round((c * self.clip_samples + self.erp_samples)
                      / self.grid_samples))
        return min(max(k, k_min), k_max)

    def _init_motion_resample(self) -> None:
        """Precompute the per-subject motion-weighted anchor sampling CDF (main
        process; worker forks inherit it). Each subject's CDF spans its
        ``[k_min, k_max]`` and draws ``k`` ∝ ``clip(anchor_motion, 0, p_cap)^
        alpha`` mixed with ``floor_mix`` uniform. Subjects without a usable
        motion cache (no video, or all-static) map to ``None`` -> uniform
        fallback in ``_sample_base_slot``."""
        if not self.use_frame_grid:
            raise ValueError(
                "[EgoBrain] motion_resample requires use_frame_grid=True (it "
                "reweights the continuous grid anchor draw).")
        from datasets.egobrain_motion import (
            load_or_compute_motion, anchor_motion, build_anchor_weights,
            build_anchor_cdf)
        space = self.motion_resample_space
        # The motion (cls/patch) source is the time-keyed embedding cache; pixel
        # motion reads the frame grid. Resolve the embedding dir even when the
        # run itself feeds raw frames (use_frame_grid without grid embeddings).
        emb_dir = (self.motion_emb_dir or self.emb_grid_dir)
        if emb_dir is None:
            enc_slug = self.vision_encoder.replace('/', '_')
            emb_dir = os.path.join(
                self.data_dir,
                f'cache_embeddings_grid_{enc_slug}'
                f'_g{self.frame_grid_s}_sz{self.frame_size}')
        if space in ('patch', 'cls') and not os.path.isdir(emb_dir):
            raise FileNotFoundError(
                f"[EgoBrain] motion_resample space='{space}' needs the time-keyed "
                f"embedding cache at '{emb_dir}' (build it with "
                f"datasets.egobrain_extract_embeddings_grid), or pass "
                f"motion_emb_dir, or use space='pixel'.")
        if self.stride_samples % self.grid_samples != 0:
            import warnings
            warnings.warn(
                f"[EgoBrain] motion_resample: stride_samples={self.stride_samples}"
                f" is not a multiple of grid_samples={self.grid_samples}; the "
                f"per-window frame step is rounded for the motion lookup.",
                stacklevel=2)
        step = int(round(self.stride_samples / self.grid_samples))
        frames_dir = self.frame_grid_dir
        n_weighted = 0
        ess_fracs = []
        for sub in self.subjects:
            mot = load_or_compute_motion(
                emb_dir, sub, step, self.motion_resample_metric, space,
                frames_grid_dir=frames_dir)
            if mot is None:
                self._anchor_cdf[sub] = None
                continue
            am = anchor_motion(mot, self.n_windows, step)
            w = build_anchor_weights(am, self.motion_resample_alpha,
                                     self.motion_resample_cap_pct)
            k_min, k_max = self._anchor_k_bounds(sub)
            k_max = min(k_max, w.shape[0] - 1)           # never index past motion
            cdf = build_anchor_cdf(w, k_min, k_max, self.motion_resample_floor_mix)
            if cdf is None:
                self._anchor_cdf[sub] = None
            else:
                self._anchor_cdf[sub] = (k_min, cdf)
                n_weighted += 1
                # effective sample size fraction = how concentrated the draw is
                p = np.diff(np.concatenate([[0.0], cdf]))
                ess_fracs.append(1.0 / (np.sum(p * p) * p.shape[0] + 1e-12))
        ess = float(np.mean(ess_fracs)) if ess_fracs else float('nan')
        print(f"[EgoBrain] motion_resample ON: space={space} "
              f"metric={self.motion_resample_metric} alpha="
              f"{self.motion_resample_alpha} cap_p{self.motion_resample_cap_pct} "
              f"floor_mix={self.motion_resample_floor_mix} step={step} | "
              f"{n_weighted}/{len(self.subjects)} subjects weighted "
              f"(mean effective-sample-size fraction {ess:.2f})")

    def _load_continuous_raw(self, sub: str, start: int, length: int
                             ) -> np.ndarray:
        """Raw (C_all, length) µV segment of the continuous recording, stitched
        across the per-clip ``.npy`` files (a window can straddle a former 4 s
        boundary). Zero-pads if the request runs past the last clip."""
        cs = self.clip_samples
        n_clips = self._subject_meta[sub]['n_clips']
        first = start // cs
        last = min((start + length - 1) // cs, n_clips - 1)
        pieces = [np.load(self._clip_path(sub, cl)) for cl in range(first, last + 1)]
        full = np.concatenate(pieces, axis=1)            # (C_all, k*cs)
        local = start - first * cs
        seg = full[:, local:local + length]
        if seg.shape[1] < length:
            seg = np.pad(seg, ((0, 0), (0, length - seg.shape[1])))
        return np.ascontiguousarray(seg)

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
        if self.use_frame_grid:
            return self._getitem_grid(idx)
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

        # Pre-computed vision-encoder embeddings (datasets/
        # egobrain_extract_embeddings.py): when enabled, surface cls/cls_flip
        # (W,d) + grid/grid_flip (W,P,d) so the model skips the frozen-encoder
        # forward. has_image is taken from the EMBEDDING cache so the alignment
        # masks match exactly what was embedded. pixel_values are still returned
        # below (harmless; the model prefers the cached tensors when present).
        frame_emb = None
        if self.use_emb_cache:
            emb_path = os.path.join(self.emb_cache_dir, f'{sub}.h5')
            if os.path.exists(emb_path):
                he = _get_h5_emb_handle(emb_path)
                frame_emb = {
                    'frame_cls': torch.from_numpy(
                        np.asarray(he['cls'][c], dtype=np.float32)),        # (W,d)
                    'frame_cls_flip': torch.from_numpy(
                        np.asarray(he['cls_flip'][c], dtype=np.float32)),
                    'frame_grid': torch.from_numpy(
                        np.asarray(he['grid'][c], dtype=np.float32)),       # (W,P,d)
                    'frame_grid_flip': torch.from_numpy(
                        np.asarray(he['grid_flip'][c], dtype=np.float32)),
                }
                has_image = torch.from_numpy(
                    np.asarray(he['has_image'][c])).bool()

        # Auxiliary continuous hand-movement targets (W, 2) = [left, right]
        # intensity, with a per-column valid mask (undetected hand -> NaN -> 0 +
        # invalid, so the regression loss skips it). Zeros + all-invalid when no
        # hand-label cache is configured (then the aux loss masks this row out).
        if self.use_hand_labels:
            hand_targets, hand_valid = self._load_hand_labels(sub, c)
        else:
            hand_targets = torch.zeros(self.n_windows, 2, dtype=torch.float32)
            hand_valid = torch.zeros(self.n_windows, 2, dtype=torch.bool)

        out = {
            'timeseries': timeseries,
            'ch_coords': ch_coords,
            'ch_names': ch_names,
            'pixel_values': pixel_values,
            'has_image': has_image,
            'hand_targets': hand_targets,
            'hand_valid': hand_valid,
            'source': 'egobrain',
            'session_id': sub,
            'subject': sub,
            'local_clip_idx': c,
        }
        if frame_emb is not None:
            out.update(frame_emb)
        return out

    def _load_hand_labels(self, sub, c):
        """Per-window continuous hand targets + per-column valid mask for clip
        ``c`` of ``sub``. Returns (hand_targets (W,2) float32, hand_valid (W,2)
        bool). Mirrors the frame-cache read: index the (n_clips, n_windows)
        arrays at [c]. Column 0 = left, 1 = right intensity; a hand is valid iff
        the window has video AND its intensity is finite (undetected -> NaN)."""
        W = self.n_windows
        path = os.path.join(self.hand_labels_dir, f'{sub}.h5')
        if not os.path.exists(path):
            return (torch.zeros(W, 2, dtype=torch.float32),
                    torch.zeros(W, 2, dtype=torch.bool))
        h = _get_h5_hand_handle(path)
        li = np.asarray(h['left_intensity'][c], dtype=np.float32)    # (nw,)
        ri = np.asarray(h['right_intensity'][c], dtype=np.float32)
        hv = np.asarray(h['has_video'][c]).astype(bool)
        tgt = np.stack([np.nan_to_num(li), np.nan_to_num(ri)], axis=-1)  # (nw,2)
        val = np.stack([hv & np.isfinite(li), hv & np.isfinite(ri)], axis=-1)
        nw = tgt.shape[0]
        if nw < W:                                  # pad (cache nw should match)
            tgt = np.concatenate([tgt, np.zeros((W - nw, 2), np.float32)], 0)
            val = np.concatenate([val, np.zeros((W - nw, 2), bool)], 0)
        elif nw > W:
            tgt, val = tgt[:W], val[:W]
        return torch.from_numpy(tgt).float(), torch.from_numpy(val).bool()

    def _getitem_grid(self, idx: int) -> dict:
        """Continuous / 0.2 s grid path (use_frame_grid=True).

        Pins the (discrete, pre-stored) FRAME to its 0.2 s slot and slides the
        (continuous) EEG window to match: window-0 START = ``k*grid_samples -
        erp_samples`` so its frame sits exactly at ``start + erp`` (slot ``k``),
        making every window's frame slot an exact integer for ANY erp. The
        window is anchored on its START point. Returns the SAME dict schema as
        ``__getitem__``. Hand labels are clip-keyed and not supported here
        (returned as zeros / all-invalid)."""
        sub, c = self._items[idx]
        k = self._sample_base_slot(sub, c)
        eeg_start = k * self.grid_samples - self.erp_samples   # window-0 START
        span_total = ((self.n_windows - 1) * self.stride_samples
                      + self.window_samples)
        seg = self._select_and_whiten(
            self._load_continuous_raw(sub, eeg_start, span_total), sub)  # (C, T)
        C = seg.shape[0]

        windows = [seg[:, i * self.stride_samples:
                       i * self.stride_samples + self.window_samples]
                   for i in range(self.n_windows)]
        windows_np = np.stack(windows, axis=0)                # (W, C, T)
        timeseries = torch.from_numpy(
            windows_np.reshape(self.n_windows, C,
                               self.n_patches_per_window, self.in_dim)
            .astype(np.float32))                              # (W, C, N, d)

        ch_coords = self._subject_meta[sub]['ch_coords']
        ch_names = list(self._subject_meta[sub]['ch_names'])

        # Frame slot for window i = (window-i START + erp) / grid
        # = (k*grid + i*stride) / grid = k + i*stride/grid. Exact integer for
        # any erp when stride is a multiple of the grid (round() only guards a
        # non-multiple stride). START-anchored, not centre-anchored.
        frame_slots = [
            int(round((k * self.grid_samples + i * self.stride_samples)
                      / self.grid_samples))
            for i in range(self.n_windows)]

        pixel_values = torch.zeros(
            self.n_windows, 3, self.frame_size, self.frame_size,
            dtype=torch.float32)
        has_image = torch.zeros(self.n_windows, dtype=torch.bool)
        frame_emb = None
        if self.load_frames and self.use_grid_embeddings:
            # Cached DINOv2 embeddings: read cls/grid (+flip) per slot and let
            # the model skip the live encoder. pixel_values stay zero (the model
            # prefers the cached tensors); has_image comes from the emb cache.
            frame_emb = self._read_grid_embeddings(sub, frame_slots, has_image)
        elif self.load_frames and self.use_frame_grid:
            # Frames from the time-keyed frame grid (live encode downstream).
            gpath = os.path.join(self.frame_grid_dir, f'{sub}.h5')
            if os.path.exists(gpath):
                h = _get_h5_frame_handle(gpath)
                frames_ds = h['frames']
                has_ds = h['has_image']
                n_slots = frames_ds.shape[0]
                mean, std = _get_normalize_params(self.vision_encoder)
                for i, slot in enumerate(frame_slots):
                    if 0 <= slot < n_slots and bool(has_ds[slot]):
                        fr = np.ascontiguousarray(frames_ds[slot])   # (H,W,3)
                        x = (torch.from_numpy(fr).float().div_(255.0)
                             .permute(2, 0, 1).unsqueeze(0))          # (1,3,H,W)
                        pixel_values[i] = ((x - mean) / std)[0]
                        has_image[i] = True

        # Hand labels are (clip, window)-keyed; not aligned to arbitrary
        # offsets, so the grid path leaves them off (the aux loss masks them).
        hand_targets = torch.zeros(self.n_windows, 2, dtype=torch.float32)
        hand_valid = torch.zeros(self.n_windows, 2, dtype=torch.bool)

        out = {
            'timeseries': timeseries,
            'ch_coords': ch_coords,
            'ch_names': ch_names,
            'pixel_values': pixel_values,
            'has_image': has_image,
            'hand_targets': hand_targets,
            'hand_valid': hand_valid,
            'source': 'egobrain',
            'session_id': sub,
            'subject': sub,
            'local_clip_idx': c,
            'anchor_sample': int(eeg_start),     # window-0 START sample
            'base_slot': int(k),                 # window-0 frame slot
        }
        if frame_emb is not None:
            out.update(frame_emb)
        return out

    def _read_grid_embeddings(self, sub, frame_slots, has_image):
        """Read grid (+ cls for DINOv2-style; grid-ONLY for V-JEPA 2) from the
        time-keyed embedding cache at the given per-window slots, updating
        ``has_image`` in place. Orientations are interleaved on axis 1
        (``grid`` (n_slots,2,P,d), ``cls`` (n_slots,2,d) when present; [:,0]=orig,
        [:,1]=h-flip), so each slot's BOTH orientations come in ONE chunk read.

        V-JEPA 2 caches have NO ``cls`` (its alignment rep is a trainable pool
        over the grid, computed in the model), so only ``frame_grid``/
        ``frame_grid_flip`` are returned. Returns None when the subject has no
        cache (e.g. no-video), leaving has_image False."""
        epath = os.path.join(self.emb_grid_dir, f'{sub}.h5')
        if not os.path.exists(epath):
            return None
        he = _get_h5_emb_handle(epath)
        grid_ds, has_ds = he['grid'], he['has_image']
        has_cls = 'cls' in he                       # DINOv2 yes; V-JEPA 2 no
        cls_ds = he['cls'] if has_cls else None
        n_slots, _, P, d = grid_ds.shape             # (n_slots, 2, P, d)
        W = self.n_windows
        f_grid = torch.zeros(W, P, d, dtype=torch.float32)
        f_gridf = torch.zeros(W, P, d, dtype=torch.float32)
        if has_cls:
            d_cls = cls_ds.shape[2]
            f_cls = torch.zeros(W, d_cls, dtype=torch.float32)
            f_clsf = torch.zeros(W, d_cls, dtype=torch.float32)
        for i, slot in enumerate(frame_slots):
            if 0 <= slot < n_slots and bool(has_ds[slot]):
                gpair = np.asarray(grid_ds[slot], np.float32)   # (2,P,d) 1 read
                f_grid[i] = torch.from_numpy(gpair[0])
                f_gridf[i] = torch.from_numpy(gpair[1])
                if has_cls:
                    cpair = np.asarray(cls_ds[slot], np.float32)  # (2,d) 1 read
                    f_cls[i] = torch.from_numpy(cpair[0])
                    f_clsf[i] = torch.from_numpy(cpair[1])
                has_image[i] = True
        out = {'frame_grid': f_grid, 'frame_grid_flip': f_gridf}
        if has_cls:
            out['frame_cls'] = f_cls
            out['frame_cls_flip'] = f_clsf
        return out


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

    out = {
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
        # Window-0 hand-movement regression targets + per-column valid mask.
        # The encoder's masked-recon / global_rep is computed on window 0, so the
        # aux target is window 0's (B, 2) = [left, right] intensity.
        'hand_targets': torch.stack([b['hand_targets'][0] for b in batch]),
        'hand_valid': torch.stack([b['hand_valid'][0] for b in batch]),
        'source': [b['source'] for b in batch],
        'session_id': [b.get('session_id', 'unknown') for b in batch],
    }

    # Cached vision-encoder embeddings (when EgoBrainDataset.use_emb_cache): the
    # model reads these instead of running the frozen encoder. Window-0 tensors
    # (B,...) feed the image alignment + flip-align descriptor; the future stacks
    # (B,W,...) feed the world-model frame objective. Row order is the batch
    # order, identical to pixel_values_future, so the per-row flip mask lines up.
    #
    # Emit whenever ANY item carries them, zero-filling items that don't — those
    # are the no-video subjects (P0025-P0040) whose has_image is all-False, so
    # the alignment / flip-align / frame-objective masks drop their (zero) rows
    # anyway. Requiring EVERY item to carry them would instead disable the cache
    # for any batch that happens to include a no-video row (almost all of them).
    emb_items = [b for b in batch if 'frame_grid' in b]
    if emb_items:
        ref = emb_items[0]
        Wn, P, d_img = ref['frame_grid'].shape
        zg = torch.zeros(Wn, P, d_img)
        grid = [b.get('frame_grid', zg) for b in batch]
        grid_f = [b.get('frame_grid_flip', zg) for b in batch]
        out['frame_grid'] = torch.stack([t[0] for t in grid])           # (B,P,d)
        out['frame_grid_flip'] = torch.stack([t[0] for t in grid_f])
        out['frame_grid_future'] = torch.stack(grid)                    # (B,W,P,d)
        out['frame_grid_flip_future'] = torch.stack(grid_f)
        # cls only for DINOv2-style caches. V-JEPA 2 is grid-only (its alignment
        # rep is a trainable pool over the grid, recomputed in the model), so
        # no item carries frame_cls -> omit it entirely.
        cls_items = [b for b in emb_items if 'frame_cls' in b]
        if cls_items:
            d_cls = cls_items[0]['frame_cls'].shape[-1]
            zc = torch.zeros(Wn, d_cls)
            cls = [b.get('frame_cls', zc) for b in batch]
            cls_f = [b.get('frame_cls_flip', zc) for b in batch]
            out['frame_cls'] = torch.stack([t[0] for t in cls])          # (B,d)
            out['frame_cls_flip'] = torch.stack([t[0] for t in cls_f])
    return out


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
        # The iterable (mix-mode) path yields only pixels/futures — it never
        # forwards the cached frame_cls/frame_grid tensors — so a dataset built
        # with use_embeddings runs the frozen encoder LIVE here, silently
        # negating the cache. Warn rather than fail (pixels still train fine).
        if (getattr(dataset, 'use_emb_cache', False)
                or getattr(dataset, 'use_grid_embeddings', False)):
            import warnings
            warnings.warn(
                "[EgoBrain] cached embeddings (clip-keyed or grid) are NOT used "
                "by the iterable / mix-mode path (EgoBrainIterableWrapper yields "
                "pixels only); the frozen vision encoder will run live, so the "
                "embedding cache has no speed effect here. Cached embeddings "
                "only apply to the map-style `egobrain` loader "
                "(collate_egobrain).", stacklevel=2)

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
                'hand_targets': sample['hand_targets'][0],   # (2,) window 0
                'hand_valid': sample['hand_valid'][0],       # (2,) window 0
            }
            if bool(sample['has_image'][0].item()):
                out['pixel_values'] = sample['pixel_values'][0]
            if self.has_future:
                out['timeseries_future'] = sample['timeseries']
                out['pixel_values_future'] = sample['pixel_values']
                out['has_image_future'] = sample['has_image']
            yield out
