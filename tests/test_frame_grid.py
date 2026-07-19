"""Tests for the continuous, time-keyed 0.2 s-snap EgoBrain path.

Covers datasets/egobrain_extract_frames_grid.py (slug + chapter routing) and
EgoBrainDataset(use_frame_grid=True): anchor snapping, frame-slot alignment,
cross-(former-clip)-boundary EEG stitching, jitter reproducibility — plus a
back-compat guard that use_frame_grid=False reproduces the legacy clip-keyed
path. Builds tiny synthetic caches so no real EgoBrain data is needed.

Run:
    conda run -n cbramod python -m pytest tests/test_frame_grid.py -q
"""

import json
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('h5py')
pytest.importorskip('mne')                       # _resolve_ch_coords needs it
import h5py                                       # noqa: E402

from datasets.egobrain_dataset import (           # noqa: E402
    EgoBrainDataset, collate_egobrain, _NORMALIZE_CACHE,
)
from datasets.egobrain_extract_frames_grid import (  # noqa: E402
    _grid_cache_dir, _route_chapter,
)
from datasets.egobrain_extract_hand_labels_grid import (  # noqa: E402
    default_grid_out_dir,
)

@pytest.fixture(autouse=True)
def _restore_normalize_cache():
    """These tests seed the module-global _NORMALIZE_CACHE with an identity
    transform; snapshot + restore it so the pollution can't leak into other
    test files (e.g. test_egobrain_dataset's normalization assertions)."""
    import datasets.egobrain_dataset as eg
    saved = dict(eg._NORMALIZE_CACHE)
    try:
        yield
    finally:
        eg._NORMALIZE_CACHE.clear()
        eg._NORMALIZE_CACHE.update(saved)


ENC = 'facebook/dinov2-base'
FS = 200
CLIP_S = 4.0
CLIP_SAMPLES = int(CLIP_S * FS)                   # 800
N_CLIPS = 3
CH_NAMES = ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2']   # all in standard_1020
C = len(CH_NAMES)
SZ = 8


def _seed_identity_normalize():
    """Make _get_normalize_params a no-op (mean 0 / std 1) so pixel_values ==
    frame/255 and we can read back which slot was fetched — and skip the HF
    processor download."""
    mean = torch.zeros(1, 3, 1, 1)
    std = torch.ones(1, 3, 1, 1)
    _NORMALIZE_CACHE[ENC] = (mean, std)


def _make_eeg_cache(root, sub, abs_index_fill=True):
    """Per-subject EEG clip cache. When abs_index_fill, clip c sample t holds
    the ABSOLUTE sample index (c*CLIP_SAMPLES + t) in every channel, so a
    stitched window's values reveal exactly which absolute samples were read."""
    sub_dir = os.path.join(root, f'cache_eeg_{FS}hz', sub)
    os.makedirs(sub_dir, exist_ok=True)
    for c in range(N_CLIPS):
        if abs_index_fill:
            base = np.arange(c * CLIP_SAMPLES, (c + 1) * CLIP_SAMPLES,
                             dtype=np.float32)
            arr = np.broadcast_to(base, (C, CLIP_SAMPLES)).astype(np.float32)
        else:
            arr = np.zeros((C, CLIP_SAMPLES), dtype=np.float32)
        np.save(os.path.join(sub_dir, f'{c}.npy'), arr)
    meta = {
        'fs_in': 256, 'fs_out': FS, 'clip_s': CLIP_S, 'n_clips': N_CLIPS,
        'n_channels': C, 'ch_names': CH_NAMES, 'edf_path': 'x.edf',
        'video': None, 'events': [],
        'preprocess': {'l_freq': 0.3, 'h_freq': 75.0, 'notch_hz': 50.0},
    }
    with open(os.path.join(sub_dir, 'clips.json'), 'w') as f:
        json.dump(meta, f)


def _make_grid_frame_cache(root, sub, grid_s=0.2, n_slots=80):
    """Time-keyed grid frame cache: frame at slot k is filled with value
    (k % 256) so the dataset's fetched slot is recoverable from pixels."""
    d = _grid_cache_dir(root, ENC, grid_s, SZ)
    os.makedirs(d, exist_ok=True)
    frames = np.zeros((n_slots, SZ, SZ, 3), dtype=np.uint8)
    for k in range(n_slots):
        frames[k] = k % 256
    with h5py.File(os.path.join(d, f'{sub}.h5'), 'w') as h:
        h.create_dataset('frames', data=frames)
        h.create_dataset('has_image', data=np.ones((n_slots,), dtype=bool))
        h.attrs['grid_s'] = grid_s
        h.attrs['frame_size'] = SZ
        h.attrs['vision_encoder'] = ENC
    return d


def _make_grid_emb_cache(root, sub, grid_s=0.2, n_slots=80, d=16, P=4,
                         with_cls=True):
    """Time-keyed embedding cache, INTERLEAVED layout (format_version 2):
    grid (n_slots,2,P,d) (+ cls (n_slots,2,d) for DINOv2-style; OMITTED for
    V-JEPA 2 grid-only when ``with_cls=False``), axis 1 = [orig, h-flip]. Slot k
    filled with value k (orig) / k+1000 (flip) so the fetched slot+orientation
    is recoverable from the returned tensor."""
    from datasets.egobrain_extract_embeddings_grid import emb_grid_cache_dir
    dr = emb_grid_cache_dir(root, ENC, grid_s, SZ)
    os.makedirs(dr, exist_ok=True)
    grid = np.zeros((n_slots, 2, P, d), np.float16)
    for k in range(n_slots):
        grid[k, 0] = k;      grid[k, 1] = k + 1000      # [orig, flip]
    with h5py.File(os.path.join(dr, f'{sub}.h5'), 'w') as h:
        h.create_dataset('grid', data=grid)
        if with_cls:
            cls = np.zeros((n_slots, 2, d), np.float16)
            for k in range(n_slots):
                cls[k, 0] = k;   cls[k, 1] = k + 1000
            h.create_dataset('cls', data=cls)
        h.create_dataset('has_image', data=np.ones((n_slots,), bool))
        h.attrs['grid_s'] = grid_s
        h.attrs['d_img'] = d
        h.attrs['n_patches'] = P
        h.attrs['orient_axis'] = 1
        h.attrs['has_cls'] = bool(with_cls)
    return dr


def _make_old_frame_cache(root, sub, n_windows=2):
    """Legacy clip-keyed frame cache (n_clips, n_windows, ...)."""
    d = os.path.join(root, f'cache_frames_{ENC.replace("/", "_")}'
                     f'_w1.0s1.0_e0.5_nw{n_windows}_sz{SZ}')
    os.makedirs(d, exist_ok=True)
    frames = np.zeros((N_CLIPS, n_windows, SZ, SZ, 3), dtype=np.uint8)
    for c in range(N_CLIPS):
        for w in range(n_windows):
            frames[c, w] = (c * 10 + w) % 256
    with h5py.File(os.path.join(d, f'{sub}.h5'), 'w') as h:
        h.create_dataset('frames', data=frames)
        h.create_dataset('has_image', data=np.ones((N_CLIPS, n_windows), bool))
    return d


def _grid_dataset(root, **kw):
    _make_eeg_cache(root, 'P0001')
    _make_grid_frame_cache(root, 'P0001')
    _seed_identity_normalize()
    params = dict(
        data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
        window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
        erp_latency_s=0.5, vision_encoder=ENC, frame_size=SZ,
        use_frame_grid=True, frame_grid_s=0.2, max_channels=32,
    )
    params.update(kw)
    return EgoBrainDataset(**params)


# --------------------------------------------------------------------------
# Extractor pure helpers
# --------------------------------------------------------------------------

def test_grid_slug_is_knob_agnostic():
    d = _grid_cache_dir('data/EgoBrain', ENC, 0.2, 224)
    assert d.endswith('cache_frames_grid_facebook_dinov2-base_g0.2_sz224')
    # The slug must NOT mention window/stride/erp/n_windows/clip.
    for tok in ('_w', 's1.0', '_e0', '_nw', '_c4'):
        assert tok not in os.path.basename(d)


def test_route_chapter_maps_into_chapter_local_time():
    chapters = [
        {'path': 'a.mp4', 'start_s': 0.0, 'duration_s': 10.0},
        {'path': 'b.mp4', 'start_s': 10.0, 'duration_s': 10.0},
    ]
    assert _route_chapter(chapters, -1.0) is None
    ch, t = _route_chapter(chapters, 3.0)
    assert ch['path'] == 'a.mp4' and abs(t - 3.0) < 1e-9
    ch, t = _route_chapter(chapters, 12.5)
    assert ch['path'] == 'b.mp4' and abs(t - 2.5) < 1e-9
    assert _route_chapter(chapters, 25.0) is None


# --------------------------------------------------------------------------
# Grid loading path
# --------------------------------------------------------------------------

def test_grid_sample_shapes_and_schema():
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_dataset(root, temporal_jitter=False)
        assert len(ds) == N_CLIPS
        s = ds[0]
        assert s['timeseries'].shape == (2, C, 5, 40)   # (W,C,N,d)
        assert s['pixel_values'].shape == (2, 3, SZ, SZ)
        assert s['has_image'].shape == (2,)
        assert s['source'] == 'egobrain'
        # schema parity with the legacy path so collate works unchanged
        for k in ('ch_coords', 'ch_names', 'hand_targets', 'hand_valid'):
            assert k in s


def test_grid_frame_slot_exact_and_start_anchored():
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_dataset(root, temporal_jitter=True, jitter_seed=0)
        grid, erp, step = (ds.grid_samples, ds.erp_samples,
                           ds.stride_samples // ds.grid_samples)
        for ic in range(len(ds)):
            s = ds[ic]
            k = s['base_slot']
            # START-anchored: window-0 start = k*grid - erp (frame pinned to k)
            assert s['anchor_sample'] == k * grid - erp
            # frame slots are EXACT integers k + i*step (no rounding)
            for i in range(2):
                want = k + i * step
                assert bool(s['has_image'][i])
                # identity normalize -> pixel == frame/255, frame == slot%256
                assert round(float(s['pixel_values'][i].mean() * 255.0)) == want % 256


def test_grid_frame_slot_exact_for_any_erp():
    # The redesign's point: pinning the frame + sliding the continuous EEG gives
    # exact integer slots even for erp values that the old centre-based round()
    # would have split half-a-slot (e.g. erp=0.0 with window_s=1.0).
    for erp in (0.0, 0.2, 0.5):
        with tempfile.TemporaryDirectory() as root:
            ds = _grid_dataset(root, temporal_jitter=True, jitter_seed=1,
                               erp_latency_s=erp)
            grid, erp_s, step = (ds.grid_samples, ds.erp_samples,
                                 ds.stride_samples // ds.grid_samples)
            for ic in range(len(ds)):
                s = ds[ic]
                k = s['base_slot']
                assert s['anchor_sample'] == k * grid - erp_s
                for i in range(2):
                    want = (k + i * step) % 256
                    assert round(float(s['pixel_values'][i].mean() * 255.0)) == want


def test_grid_stitches_across_former_clip_boundary():
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_dataset(root, temporal_jitter=False)
        # Read 20 samples straddling the 800-sample clip-0/clip-1 boundary.
        seg = ds._load_continuous_raw('P0001', start=790, length=20)
        assert seg.shape == (C, 20)
        # abs_index_fill: value == absolute sample index, continuous across
        # the boundary (789.. wait start 790 -> 790..809).
        np.testing.assert_array_equal(
            seg[0], np.arange(790, 810, dtype=np.float32))


def test_grid_base_slot_valid_and_eeg_fits():
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_dataset(root, temporal_jitter=True, jitter_seed=0)
        grid, erp = ds.grid_samples, ds.erp_samples
        span = (ds.n_windows - 1) * ds.stride_samples + ds.window_samples
        total = N_CLIPS * CLIP_SAMPLES
        for _ in range(50):
            k = ds._sample_base_slot('P0001', 0)
            eeg_start = k * grid - erp                 # window-0 START (continuous)
            assert eeg_start >= 0                       # fits at the front
            assert eeg_start + span <= total            # ...and the back


def test_grid_jitter_reproducible_with_seed():
    with tempfile.TemporaryDirectory() as root:
        ds1 = _grid_dataset(root, temporal_jitter=True, jitter_seed=123)
        a1 = [ds1[i]['anchor_sample'] for i in range(N_CLIPS)]
    with tempfile.TemporaryDirectory() as root:
        ds2 = _grid_dataset(root, temporal_jitter=True, jitter_seed=123)
        a2 = [ds2[i]['anchor_sample'] for i in range(N_CLIPS)]
    assert a1 == a2
    # ...and a different seed gives a different sequence (very likely).
    with tempfile.TemporaryDirectory() as root:
        ds3 = _grid_dataset(root, temporal_jitter=True, jitter_seed=999)
        a3 = [ds3[i]['anchor_sample'] for i in range(N_CLIPS)]
    assert a3 != a1


def test_grid_collate_runs():
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_dataset(root, temporal_jitter=False)
        batch = collate_egobrain([ds[0], ds[1]])
        assert batch['timeseries'].shape == (2, C, 5, 40)
        assert batch['pixel_values_future'].shape == (2, 2, 3, SZ, SZ)
        assert batch['source'] == ['egobrain', 'egobrain']


def _grid_emb_dataset(root, **kw):
    _make_eeg_cache(root, 'P0001')
    _make_grid_frame_cache(root, 'P0001')          # extractor input (not read at train)
    _make_grid_emb_cache(root, 'P0001')
    params = dict(
        data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
        window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
        erp_latency_s=0.5, vision_encoder=ENC, frame_size=SZ,
        use_frame_grid=True, use_grid_embeddings=True, frame_grid_s=0.2,
        max_channels=32, temporal_jitter=False,
    )
    params.update(kw)
    return EgoBrainDataset(**params)


def test_grid_embeddings_surface_frame_tensors_at_right_slots():
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_emb_dataset(root, temporal_jitter=True, jitter_seed=0)
        s = ds[0]
        k = s['base_slot']
        step = ds.stride_samples // ds.grid_samples
        # cached-embedding schema (same keys as the clip-keyed path)
        for key in ('frame_cls', 'frame_cls_flip', 'frame_grid', 'frame_grid_flip'):
            assert key in s, key
        assert s['frame_cls'].shape == (2, 16)       # (W, d)
        assert s['frame_grid'].shape == (2, 4, 16)   # (W, P, d)
        # grid is read for EVERY window; cls only for the ANCHOR (window 0) —
        # collate keeps frame_cls[0] and drops the rest, so the loader no longer
        # reads the future-window cls (dead I/O removed).
        for i in range(2):
            slot = k + i * step
            assert bool(s['has_image'][i])
            assert round(float(s['frame_grid'][i].mean())) == slot
            assert round(float(s['frame_grid_flip'][i].mean())) == slot + 1000
        assert round(float(s['frame_cls'][0].mean())) == k
        assert round(float(s['frame_cls_flip'][0].mean())) == k + 1000
        assert torch.count_nonzero(s['frame_cls'][1:]) == 0        # non-anchor cls unread
        # pixels are a 1x1 placeholder — grid embeddings mean the encoder never
        # runs on raw frames, so no full-res frame tensor is materialized.
        assert torch.count_nonzero(s['pixel_values']) == 0
        assert s['pixel_values'].shape[-2:] == (1, 1)


def test_grid_embeddings_collate_emits_future_stacks():
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_emb_dataset(root)
        batch = collate_egobrain([ds[0], ds[1]])
        assert batch['frame_cls'].shape == (2, 16)          # (B, d) window-0
        assert batch['frame_grid'].shape == (2, 4, 16)      # (B, P, d) window-0
        assert batch['frame_grid_future'].shape == (2, 2, 4, 16)  # (B, W, P, d)
        assert batch['frame_grid_flip_future'].shape == (2, 2, 4, 16)


def test_grid_frame_objective_trims_future_eeg():
    # --wm_objective frame never encodes the future EEG windows, so collate
    # trims timeseries_future to window 0 (presence still drives cb_idx). The
    # frame TARGET stack (frame_grid_future) is left at its full W windows.
    with tempfile.TemporaryDirectory() as root:
        ds = _grid_emb_dataset(root)
        items = [ds[0], ds[1]]
        full = collate_egobrain(items)
        assert full['timeseries_future'].shape[1] == ds.n_windows
        trimmed = collate_egobrain(items, frame_objective=True)
        assert trimmed['timeseries_future'].shape[1] == 1
        assert torch.allclose(
            trimmed['timeseries_future'][:, 0], full['timeseries_future'][:, 0])
        assert trimmed['frame_grid_future'].shape[1] == ds.n_windows


def test_grid_embeddings_vjepa_grid_only():
    # V-JEPA 2 cache has NO cls -> dataset surfaces frame_grid only; collate
    # omits frame_cls (the alignment pool is computed in the model from grid).
    with tempfile.TemporaryDirectory() as root:
        _make_eeg_cache(root, 'P0001')
        _make_grid_frame_cache(root, 'P0001')
        _make_grid_emb_cache(root, 'P0001', with_cls=False)        # grid-only
        ds = EgoBrainDataset(
            data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
            window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS, erp_latency_s=0.5,
            vision_encoder=ENC, frame_size=SZ, use_frame_grid=True,
            use_grid_embeddings=True, frame_grid_s=0.2, max_channels=32,
            temporal_jitter=True, jitter_seed=0)
        s = ds[0]
        k = s['base_slot']
        step = ds.stride_samples // ds.grid_samples
        assert 'frame_grid' in s and 'frame_grid_flip' in s
        assert 'frame_cls' not in s and 'frame_cls_flip' not in s   # grid-only
        assert s['frame_grid'].shape == (2, 4, 16)
        for i in range(2):
            slot = k + i * step
            assert round(float(s['frame_grid'][i].mean())) == slot
            assert round(float(s['frame_grid_flip'][i].mean())) == slot + 1000
        b = collate_egobrain([ds[0], ds[1]])
        assert b['frame_grid_future'].shape == (2, 2, 4, 16)
        assert 'frame_cls' not in b and 'frame_cls_flip' not in b   # collate omits cls


def test_grid_embeddings_require_frame_grid():
    with tempfile.TemporaryDirectory() as root:
        _make_eeg_cache(root, 'P0001')
        with pytest.raises(ValueError):
            EgoBrainDataset(
                data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
                window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
                vision_encoder=ENC, frame_size=SZ,
                use_frame_grid=False, use_grid_embeddings=True)


def test_video_subjects_whitelist_blocks_frames():
    """A blacklisted subject must behave exactly as if it had no video: zeros +
    has_image all-False, even though its frame cache file exists on disk. This
    is the backward-compat lever for EgoBrain's newly-released P0025-P0040."""
    with tempfile.TemporaryDirectory() as root:
        # allowed: reads the grid frame cache -> has_image True
        ds_on = _grid_dataset(root, temporal_jitter=False)
        s_on = ds_on[0]
        assert bool(s_on['has_image'].any())
        assert torch.count_nonzero(s_on['pixel_values']) > 0

        # blacklisted (whitelist excludes P0001) -> no frames despite cache file
        _seed_identity_normalize()
        ds_off = EgoBrainDataset(
            data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
            window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
            erp_latency_s=0.5, vision_encoder=ENC, frame_size=SZ,
            use_frame_grid=True, frame_grid_s=0.2, max_channels=32,
            temporal_jitter=False, video_subjects=['P9999'])
        s_off = ds_off[0]
        assert not bool(s_off['has_image'].any()), 'blacklisted subject leaked video'
        assert torch.count_nonzero(s_off['pixel_values']) == 0
        # EEG is unaffected by the whitelist
        assert torch.equal(s_on['timeseries'], s_off['timeseries'])


def test_video_subjects_whitelist_blocks_grid_embeddings():
    """Same guarantee on the grid-embedding path: no frame_* tensors emitted."""
    with tempfile.TemporaryDirectory() as root:
        ds_on = _grid_emb_dataset(root)
        assert 'frame_grid' in ds_on[0] and bool(ds_on[0]['has_image'].any())
    with tempfile.TemporaryDirectory() as root:
        ds_off = _grid_emb_dataset(root, video_subjects=['P9999'])
        s = ds_off[0]
        assert 'frame_grid' not in s, 'blacklisted subject emitted cached embeddings'
        assert not bool(s['has_image'].any())


def test_whitelist_mixed_batch_collates():
    """A batch mixing an allowed and a blacklisted subject must still stack:
    pixel_values shape is decided by the MODE (1x1 placeholder under
    use_grid_embeddings), never per-subject. Guards the exact regression where
    the blacklisted row fell into the full-size branch."""
    with tempfile.TemporaryDirectory() as root:
        for s in ('P0001', 'P0002'):
            _make_eeg_cache(root, s)
            _make_grid_frame_cache(root, s)
            _make_grid_emb_cache(root, s)
        ds = EgoBrainDataset(
            data_dir=root, subjects=['P0001', 'P0002'], in_dim=40, n_windows=2,
            window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
            erp_latency_s=0.5, vision_encoder=ENC, frame_size=SZ,
            use_frame_grid=True, use_grid_embeddings=True, frame_grid_s=0.2,
            max_channels=32, temporal_jitter=False,
            video_subjects=['P0001'])          # P0002 blacklisted
        a, b = ds[0], ds[N_CLIPS]              # P0001 clip0, P0002 clip0
        assert a['subject'] == 'P0001' and b['subject'] == 'P0002'
        assert a['pixel_values'].shape == b['pixel_values'].shape  # <- the bug
        assert bool(a['has_image'].any()) and not bool(b['has_image'].any())
        assert 'frame_grid' in a and 'frame_grid' not in b
        batch = collate_egobrain([a, b])       # must not raise
        assert batch['frame_grid'].shape[0] == 2
        # blacklisted row is zero-filled by collate
        assert torch.count_nonzero(batch['frame_grid'][1]) == 0
        assert torch.count_nonzero(batch['frame_grid'][0]) > 0


def test_grid_incompatible_with_embeddings():
    with tempfile.TemporaryDirectory() as root:
        _make_eeg_cache(root, 'P0001')
        _make_grid_frame_cache(root, 'P0001')
        with pytest.raises(ValueError):
            EgoBrainDataset(
                data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
                window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
                vision_encoder=ENC, frame_size=SZ,
                use_frame_grid=True, use_embeddings=True)


# --------------------------------------------------------------------------
# Back-compat: legacy clip-keyed path untouched
# --------------------------------------------------------------------------

def test_legacy_path_unchanged_when_grid_off():
    with tempfile.TemporaryDirectory() as root:
        _make_eeg_cache(root, 'P0001')
        old_dir = _make_old_frame_cache(root, 'P0001', n_windows=2)
        _seed_identity_normalize()
        ds = EgoBrainDataset(
            data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
            window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
            erp_latency_s=0.5, vision_encoder=ENC, frame_size=SZ,
            frames_cache_dir=old_dir, use_frame_grid=False, max_channels=32)
        assert not ds.use_frame_grid
        assert ds.use_frames_cache
        s = ds[0]
        # legacy fixed grid: window 0 of clip 0 starts at sample 0
        np.testing.assert_array_equal(
            s['timeseries'].reshape(2, C, -1)[0, 0].numpy()[:5],
            np.arange(0, 5, dtype=np.float32))
        # legacy clip-keyed frames: clip 0 window w -> value (0*10+w)
        for w in range(2):
            val = float(s['pixel_values'][w].mean() * 255.0)
            assert round(val) == (0 * 10 + w) % 256


# --------------------------------------------------------------------------
# Time-keyed grid HAND labels (datasets/egobrain_extract_hand_labels_grid.py)
# --------------------------------------------------------------------------

def _make_grid_hand_cache(root, sub, grid_s=0.2, n_slots=80, fs=FS,
                          format_version=2):
    """Time-keyed grid hand cache (format 2 = RAW single-pair forward speed):
    slot k holds left_intensity=k, right_intensity=k+1000 so the fetched slot is
    recoverable per column, plus an all-True has_video. Written under the real
    default_grid_out_dir slug."""
    d = default_grid_out_dir(root, 'wilor', grid_s, fs)
    os.makedirs(d, exist_ok=True)
    li = np.arange(n_slots, dtype=np.float32)
    ri = np.arange(n_slots, dtype=np.float32) + 1000.0
    with h5py.File(os.path.join(d, f'{sub}.h5'), 'w') as h:
        h.create_dataset('left_intensity', data=li)
        h.create_dataset('right_intensity', data=ri)
        h.create_dataset('left_det', data=np.ones(n_slots, bool))
        h.create_dataset('right_det', data=np.ones(n_slots, bool))
        h.create_dataset('has_video', data=np.ones(n_slots, bool))
        h.attrs['grid_s'] = grid_s
        h.attrs['format_version'] = format_version
        h.attrs['speed_interval'] = 'forward'
        h.attrs['smoothing'] = 'none'
    return d


def test_grid_hand_labels_read_at_frame_slots():
    with tempfile.TemporaryDirectory() as root:
        hg = _make_grid_hand_cache(root, 'P0001')
        ds = _grid_dataset(root, temporal_jitter=False, hand_grid_dir=hg)
        assert ds.use_hand_grid
        step = ds.stride_samples // ds.grid_samples          # 5
        for ic in range(len(ds)):
            s = ds[ic]
            k = s['base_slot']
            assert s['hand_targets'].shape == (2, 2)
            for i in range(2):
                slot = k + i * step                          # frame slot of window i
                # left col == slot, right col == slot + 1000 (recoverable)
                assert float(s['hand_targets'][i, 0]) == float(slot)
                assert float(s['hand_targets'][i, 1]) == float(slot) + 1000.0
                assert bool(s['hand_valid'][i, 0])
                assert bool(s['hand_valid'][i, 1])


def test_grid_hand_labels_nan_and_oob_masked():
    with tempfile.TemporaryDirectory() as root:
        hg = _make_grid_hand_cache(root, 'P0001')
        # Poison one slot: NaN intensity + has_video False -> must mask out.
        with h5py.File(os.path.join(hg, 'P0001.h5'), 'r+') as h:
            h['left_intensity'][3] = np.nan
            h['has_video'][3] = False
        ds = _grid_dataset(root, temporal_jitter=False, hand_grid_dir=hg)
        # Force window-0 frame slot to the poisoned slot 3 by finding a clip
        # whose deterministic base_slot hits it; otherwise assert the general
        # rule on whatever slot each window lands on.
        for ic in range(len(ds)):
            s = ds[ic]
            k = s['base_slot']
            step = ds.stride_samples // ds.grid_samples
            for i in range(2):
                slot = k + i * step
                if slot == 3:
                    assert not bool(s['hand_valid'][i, 0])   # NaN + no video
                    assert float(s['hand_targets'][i, 0]) == 0.0  # nan_to_num


def test_grid_hand_labels_absent_subject_zeros():
    # No grid hand h5 for the subject -> zeros / all-invalid (aux masks it).
    with tempfile.TemporaryDirectory() as root:
        hg = default_grid_out_dir(root, 'wilor', 0.2, FS)
        os.makedirs(hg, exist_ok=True)                       # empty dir (isdir True)
        ds = _grid_dataset(root, temporal_jitter=False, hand_grid_dir=hg)
        s = ds[0]
        assert not s['hand_valid'].any()
        assert float(s['hand_targets'].abs().sum()) == 0.0


def test_clip_hand_labels_rejected_under_frame_grid():
    # The core bug guard: clip-keyed hand labels + frame grid must FAIL loud,
    # not silently train on nothing.
    with tempfile.TemporaryDirectory() as root:
        _make_eeg_cache(root, 'P0001')
        _make_grid_frame_cache(root, 'P0001')
        _seed_identity_normalize()
        clip_hand = os.path.join(root, 'cache_hand_labels_wilor_legacy')
        os.makedirs(clip_hand, exist_ok=True)
        with pytest.raises(ValueError, match='clip-keyed hand labels'):
            EgoBrainDataset(
                data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
                window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
                erp_latency_s=0.5, vision_encoder=ENC, frame_size=SZ,
                use_frame_grid=True, frame_grid_s=0.2, max_channels=32,
                hand_labels_dir=clip_hand)


def test_grid_hand_dir_requires_frame_grid():
    with tempfile.TemporaryDirectory() as root:
        _make_eeg_cache(root, 'P0001')
        old_dir = _make_old_frame_cache(root, 'P0001', n_windows=2)
        _seed_identity_normalize()
        hg = _make_grid_hand_cache(root, 'P0001')
        with pytest.raises(ValueError, match='requires use_frame_grid'):
            EgoBrainDataset(
                data_dir=root, subjects=['P0001'], in_dim=40, n_windows=2,
                window_s=1.0, stride_s=1.0, clip_s=CLIP_S, fs_out=FS,
                erp_latency_s=0.5, vision_encoder=ENC, frame_size=SZ,
                frames_cache_dir=old_dir, use_frame_grid=False,
                max_channels=32, hand_grid_dir=hg)


def test_grid_hand_grid_s_mismatch_raises():
    # A wrong grid_s shifts every slot index -> misaligned labels; fail loud.
    with tempfile.TemporaryDirectory() as root:
        hg = _make_grid_hand_cache(root, 'P0001', grid_s=0.1)  # cache says 0.1
        with pytest.raises(ValueError, match='grid_s'):
            _grid_dataset(root, temporal_jitter=False, hand_grid_dir=hg,
                          frame_grid_s=0.2)                      # dataset wants 0.2


def test_legacy_smoothed_v1_hand_cache_rejected():
    # v1 stored a hand_ref_s WINDOW-AVERAGED speed; the reader expects the raw
    # single-pair forward speed. Must fail loud, not silently read wrong values.
    with tempfile.TemporaryDirectory() as root:
        hg = _make_grid_hand_cache(root, 'P0001', format_version=1)
        with pytest.raises(ValueError, match='format_version'):
            _grid_dataset(root, temporal_jitter=False, hand_grid_dir=hg)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
