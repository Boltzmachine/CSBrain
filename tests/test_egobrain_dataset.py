"""Smoke tests for the EgoBrain dataset + world-model wiring.

The real dataset is gated; this test builds a tiny synthetic cache that
matches the layout written by :mod:`datasets.egobrain_preprocess`, then
verifies that:

1. ``EgoBrainDataset`` constructs and yields samples with the expected
   shapes and dtypes.
2. ``collate_egobrain`` produces a batch matching the keys the existing
   ``WorldModelWrapper`` consumes.
3. ``EgoBrainIterableWrapper`` yields the per-sample dict schema that
   ``collate_cached_with_future`` expects (mix-mode training).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _make_fake_cache(root: str, sub: str = 'P0001', n_clips: int = 4,
                     fs_out: int = 200, clip_s: float = 4.0):
    """Write a fake preprocessed cache mirroring the real layout.

    EgoBrain's documented channel set is the 10-20 standard; we pick a
    valid subset that the dataset's montage lookup must accept.
    """
    sub_dir = os.path.join(root, sub)
    os.makedirs(sub_dir, exist_ok=True)
    ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4',
                'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                'Pz', 'P4', 'P8', 'PO7', 'PO3', 'PO4', 'PO8', 'Oz']
    C = len(ch_names)
    clip_samples = int(round(clip_s * fs_out))
    rng = np.random.default_rng(0)
    for c in range(n_clips):
        arr = rng.standard_normal((C, clip_samples)).astype(np.float32)
        np.save(os.path.join(sub_dir, f'{c}.npy'), arr, allow_pickle=False)
    meta = {
        'fs_in': 256, 'fs_out': fs_out, 'clip_s': clip_s,
        'n_clips': n_clips, 'n_channels': C,
        'ch_names': ch_names,
        'edf_path': f'{sub}/EmoTiv_FLEX_2_Gel/Phase_2/fake.edf',
        'video': None,
        'events': [],
        'preprocess': {'l_freq': 0.3, 'h_freq': 75.0, 'notch_hz': 50.0},
    }
    with open(os.path.join(sub_dir, 'clips.json'), 'w') as f:
        json.dump(meta, f)


class TestEgoBrainDataset(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='egobrain_test_')
        self.cache = os.path.join(self.tmpdir, 'cache')
        _make_fake_cache(self.cache, sub='P0001', n_clips=4)
        _make_fake_cache(self.cache, sub='P0002', n_clips=4)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_ds(self, n_windows=3, in_dim=40):
        from datasets.egobrain_dataset import EgoBrainDataset
        return EgoBrainDataset(
            data_dir=self.tmpdir,
            subjects=('P0001', 'P0002'),
            cache_dir=self.cache,
            in_dim=in_dim,
            n_windows=n_windows,
            window_s=2.0,
            stride_s=1.0,
            clip_s=4.0,
            load_frames=False,
            max_channels=32,
        )

    def test_dataset_shapes(self):
        ds = self._make_ds(n_windows=3, in_dim=40)
        self.assertEqual(len(ds), 8)
        sample = ds[0]
        self.assertEqual(
            sample['timeseries'].shape,
            (3, 32, 2 * 200 // 40, 40),
            f"got {tuple(sample['timeseries'].shape)}")
        self.assertEqual(sample['ch_coords'].shape, (32, 3))
        self.assertEqual(len(sample['ch_names']), 32)
        self.assertEqual(sample['pixel_values'].shape[0], 3)
        self.assertEqual(sample['has_image'].dtype, torch.bool)
        self.assertFalse(sample['has_image'].any())

    def test_collate_keys(self):
        from datasets.egobrain_dataset import collate_egobrain
        ds = self._make_ds(n_windows=2, in_dim=40)
        batch = collate_egobrain([ds[0], ds[1]])
        for k in ('timeseries', 'ch_coords', 'valid_channel_mask',
                  'valid_length_mask', 'timeseries_future',
                  'pixel_values_future', 'has_image_future',
                  'image_encoder_inputs', 'has_image'):
            self.assertIn(k, batch, f'missing key {k}')
        self.assertEqual(batch['timeseries'].shape[0], 2)
        self.assertEqual(batch['timeseries_future'].shape[1], 2)
        self.assertTrue(batch['valid_channel_mask'].all())

    def test_iterable_wrapper(self):
        from datasets.egobrain_dataset import EgoBrainIterableWrapper
        ds = self._make_ds(n_windows=2, in_dim=40)
        it = iter(EgoBrainIterableWrapper(ds, seed=123))
        sample = next(it)
        for k in ('timeseries', 'ch_coords', 'ch_names', 'source',
                  'session_id', 'sfreq', 'timeseries_future',
                  'pixel_values_future', 'has_image_future'):
            self.assertIn(k, sample, f'missing key {k}')
        self.assertEqual(sample['source'], 'egobrain')
        self.assertEqual(sample['timeseries'].shape, (32, 10, 40))


class TestWorldModelIntegration(unittest.TestCase):
    """Run a single forward pass of WorldModel on an EgoBrain batch."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='egobrain_wm_test_')
        self.cache = os.path.join(self.tmpdir, 'cache')
        _make_fake_cache(self.cache, sub='P0001', n_clips=4)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_forward_runs(self):
        # Build a tiny CSBrainAlign + LatentPredictor, then run forward on a
        # collated EgoBrain batch. The point is to confirm the dataset's
        # batch schema is consumable by WorldModelWrapper without errors.
        from datasets.egobrain_dataset import (
            EgoBrainDataset, collate_egobrain,
        )
        from models.alignment import CSBrainAlign
        from models.world_model import LatentPredictor, WorldModelWrapper

        in_dim = 40
        ds = EgoBrainDataset(
            data_dir=self.tmpdir,
            subjects=('P0001',),
            cache_dir=self.cache,
            in_dim=in_dim,
            n_windows=2,
            window_s=2.0,
            stride_s=1.0,
            clip_s=4.0,
            load_frames=False,
            max_channels=32,
        )
        batch = collate_egobrain([ds[0], ds[1]])

        enc = CSBrainAlign(
            in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
            dim_feedforward=4 * in_dim,
            seq_len=batch['timeseries'].shape[2],
            n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
            brain_regions=None, sorted_indices=[], causal=False,
            alignment_weight=0.0,
            equivariance_weight=0.0,
            patch_embed_type='cnn',
        )
        predictor = LatentPredictor(
            d_model=in_dim, predictor_d_model=64, n_layers=2, n_heads=4,
            dim_feedforward=128, max_horizon=4)
        wm = WorldModelWrapper(
            encoder=enc, predictor=predictor,
            latent_pred_weight=1.0, cls_pred_weight=0.0,
            max_horizon=1, ramp_epochs=0)
        wm.train()
        n_ch = batch['timeseries'].shape[1]
        n_patches = batch['timeseries'].shape[2]
        mask = torch.zeros(
            batch['timeseries'].shape[0], n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = wm.training_step(batch, mask=mask)
        self.assertEqual(out.shape, batch['timeseries'].shape)
        self.assertIn('latent_pred_loss', info)


class TestCachedFramesFastPath(unittest.TestCase):
    """A fake HDF5 frame cache must short-circuit the live-decode path
    and produce a correctly-normalized pixel_values tensor.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='egobrain_h5_test_')
        self.cache = os.path.join(self.tmpdir, 'cache')
        _make_fake_cache(self.cache, sub='P0001', n_clips=4)
        # Build a minimal cache_frames dir that the dataset can auto-discover.
        self.frames_dir = os.path.join(
            self.tmpdir,
            'cache_frames_facebook_dinov2-base_w2.0s1.0_e0.0_nw2_sz224')
        os.makedirs(self.frames_dir)
        import h5py
        rng = np.random.default_rng(0)
        with h5py.File(os.path.join(self.frames_dir, 'P0001.h5'), 'w') as h:
            frames = rng.integers(
                0, 256, size=(4, 2, 224, 224, 3), dtype=np.uint8)
            h.create_dataset('frames', data=frames)
            has = np.ones((4, 2), dtype=bool)
            h.create_dataset('has_image', data=has)
            h.attrs['vision_encoder'] = 'facebook/dinov2-base'
            h.attrs['frame_size'] = 224
            h.attrs['window_s'] = 2.0
            h.attrs['stride_s'] = 1.0
            h.attrs['erp_latency_s'] = 0.0
            h.attrs['clip_s'] = 4.0
            h.attrs['n_windows'] = 2

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_path_takes_over_and_normalizes(self):
        from datasets.egobrain_dataset import EgoBrainDataset
        ds = EgoBrainDataset(
            data_dir=self.tmpdir,
            subjects=('P0001',),
            cache_dir=self.cache,
            frames_cache_dir=self.frames_dir,
            in_dim=40,
            n_windows=2,
            window_s=2.0,
            stride_s=1.0,
            clip_s=4.0,
            load_frames=True,
            max_channels=32,
        )
        self.assertTrue(ds.use_frames_cache)
        sample = ds[0]
        # Cache marks every window as has_image; pixel_values should be
        # normalized float32 in roughly ImageNet's post-norm range.
        self.assertTrue(sample['has_image'].all())
        self.assertEqual(sample['pixel_values'].shape, (2, 3, 224, 224))
        self.assertEqual(sample['pixel_values'].dtype, torch.float32)
        pv = sample['pixel_values']
        # ImageNet-normalized uint8 ranges land in roughly [-2.2, +2.7].
        self.assertGreater(pv.max().item(), 0.5)
        self.assertLess(pv.min().item(), -0.5)
        self.assertLess(pv.abs().max().item(), 3.0)


class TestChapterRouting(unittest.TestCase):
    """Multi-chapter GoPro recordings: a clip whose timestamp falls past
    chapter 1's end must route into chapter 2 (not get clamped to
    chapter 1's last frame). Uses a synthetic clips.json with two
    fake chapters and asserts ``_resolve_chapter`` returns the right
    chapter index + in-chapter time.
    """

    def _make_cache_with_chapters(self, root):
        sub_dir = os.path.join(root, 'P0001')
        os.makedirs(sub_dir, exist_ok=True)
        # Match the schema written by _make_fake_cache but with the new
        # video.chapters block. EEG content is irrelevant for this test.
        ch_names = ['Fp1', 'Fp2', 'Cz', 'Pz']
        n_clips = 200                                       # 200 * 4s = 800s
        rng = np.random.default_rng(0)
        for c in range(n_clips):
            np.save(os.path.join(sub_dir, f'{c}.npy'),
                    rng.standard_normal(
                        (len(ch_names), 800)).astype(np.float32),
                    allow_pickle=False)
        meta = {
            'fs_in': 256, 'fs_out': 200, 'clip_s': 4.0,
            'n_clips': n_clips, 'n_channels': len(ch_names),
            'ch_names': ch_names,
            'edf_path': 'fake.edf',
            'video': {
                'chapters': [
                    # Chapter 1 covers 0..400s, chapter 2 covers 400..900s.
                    {'path': 'P0001/fake/GX010001.MP4',
                     'chapter': 1, 'group': 1,
                     'duration_s': 400.0, 'fps': 30.0, 'n_frames': 12000},
                    {'path': 'P0001/fake/GX020001.MP4',
                     'chapter': 2, 'group': 1,
                     'duration_s': 500.0, 'fps': 30.0, 'n_frames': 15000},
                ],
                'video_offset_s': 0.0,
                'total_duration_s': 900.0,
            },
            'events': [],
            'preprocess': {'l_freq': 0.3, 'h_freq': 75.0, 'notch_hz': 50.0},
        }
        with open(os.path.join(sub_dir, 'clips.json'), 'w') as f:
            json.dump(meta, f)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='egobrain_chapter_test_')
        self.cache = os.path.join(self.tmpdir, 'cache')
        self._make_cache_with_chapters(self.cache)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resolve_chapter_picks_the_right_chapter(self):
        from datasets.egobrain_dataset import EgoBrainDataset
        ds = EgoBrainDataset(
            data_dir=self.tmpdir,
            subjects=('P0001',),
            cache_dir=self.cache,
            in_dim=40,
            n_windows=2,
            window_s=1.0,
            stride_s=1.0,
            clip_s=4.0,
            load_frames=False,                              # no real video
            max_channels=32,
        )
        # 100 s → chapter 1, time-in-chapter 100 s.
        ch, t_in = ds._resolve_chapter('P0001', 100.0)
        self.assertEqual(ch['path'].endswith('GX010001.MP4'), True)
        self.assertAlmostEqual(t_in, 100.0)

        # 400 s exactly → chapter 2, t_in = 0 (chapter-1 spans [0, 400)).
        ch, t_in = ds._resolve_chapter('P0001', 400.0)
        self.assertEqual(ch['path'].endswith('GX020001.MP4'), True)
        self.assertAlmostEqual(t_in, 0.0)

        # 500 s → chapter 2, t_in 100 s.
        ch, t_in = ds._resolve_chapter('P0001', 500.0)
        self.assertEqual(ch['path'].endswith('GX020001.MP4'), True)
        self.assertAlmostEqual(t_in, 100.0)

        # Past all chapters → None.
        self.assertIsNone(ds._resolve_chapter('P0001', 900.5))

        # Chapter list is non-empty and ordered.
        chs = ds._video_chapters('P0001')
        self.assertEqual(len(chs), 2)
        self.assertEqual(chs[0]['start_s'], 0.0)
        self.assertEqual(chs[1]['start_s'], 400.0)

    def test_legacy_single_path_video_meta_still_loads(self):
        """A clips.json from the pre-stitching era used ``video.path``
        (single chapter). The dataset should still load it as a one-
        element chapter list with unknown duration — matching the
        original behaviour where only one chapter is reachable.
        """
        from datasets.egobrain_dataset import EgoBrainDataset
        # Rewrite the meta to the legacy shape.
        meta_path = os.path.join(self.cache, 'P0001', 'clips.json')
        with open(meta_path) as f:
            meta = json.load(f)
        meta['video'] = {
            'path': 'P0001/fake/GX010001.MP4',
            'video_offset_s': 0.0,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        ds = EgoBrainDataset(
            data_dir=self.tmpdir,
            subjects=('P0001',),
            cache_dir=self.cache,
            in_dim=40, n_windows=2, window_s=1.0, stride_s=1.0,
            clip_s=4.0, load_frames=False, max_channels=32,
        )
        chs = ds._video_chapters('P0001')
        self.assertEqual(len(chs), 1)
        self.assertIsNone(chs[0]['duration_s'])
        # Legacy fallback treats t_video < +inf as in-chapter.
        ch, t_in = ds._resolve_chapter('P0001', 1234.0)
        self.assertAlmostEqual(t_in, 1234.0)


class TestMixModeChannelPadding(unittest.TestCase):
    """Mixed-batch with 32ch EgoBrain + 64ch Alljoined-like rows must pad
    EgoBrain's future-stack channels up to the batch's max channel count so
    ``WorldModelWrapper`` can slice ``ch_coords`` and the encoder masks
    padding via ``valid_channel_mask``.
    """

    def test_collate_cached_with_future_pads_egobrain_channels(self):
        from datasets.cached_dataset import (
            collate_cached_with_future, set_active_vision_encoder,
        )
        set_active_vision_encoder('facebook/dinov2-base')

        N, d = 5, 40

        def alljoined_like(C=64):
            return {
                'timeseries': torch.randn(C, N, d),
                'ch_coords': torch.randn(C, 3),
                'ch_names': [f'A{i}' for i in range(C)],
                'source': 'alljoined',
                'sfreq': torch.tensor(200.0),
            }

        def egobrain_like(C=32, W=2):
            return {
                'timeseries': torch.randn(C, N, d),
                'ch_coords': torch.randn(C, 3),
                'ch_names': [f'E{i}' for i in range(C)],
                'source': 'egobrain',
                'sfreq': torch.tensor(200.0),
                'timeseries_future': torch.randn(W, C, N, d),
                'pixel_values_future': torch.zeros(W, 3, 224, 224),
                'has_image_future': torch.zeros(W, dtype=torch.bool),
            }

        batch = [alljoined_like(), egobrain_like(),
                 alljoined_like(), egobrain_like()]
        out = collate_cached_with_future(batch)

        # Primary stack padded to 64 by collate_cached's per-row pad logic.
        self.assertEqual(out['timeseries'].shape, (4, 64, N, d))
        # Future stack uses only the M=2 EgoBrain rows, padded 32 → 64.
        self.assertEqual(out['timeseries_future'].shape, (2, 2, 64, N, d))
        # cinebrain_idx points to the EgoBrain rows in the full batch.
        self.assertEqual(out['cinebrain_idx'].tolist(), [1, 3])
        # The padding zone must be all-zero so the encoder can mask it via
        # valid_channel_mask without polluting the predictor target.
        self.assertEqual(
            out['timeseries_future'][:, :, 32:].abs().max().item(), 0.0)
        # valid_channel_mask should mark 64 valid Alljoined ch and 32 EgoBrain.
        self.assertEqual(
            out['valid_channel_mask'].sum(dim=-1).tolist(), [64, 32, 64, 32])

    def test_collate_cached_with_future_three_way_mix(self):
        """Three sources (Alljoined 64ch / CineBrain 64ch / EgoBrain 32ch)
        sharing one collation must produce a single future-stack tensor
        whose channel axis matches the batch's max (64) — EgoBrain rows
        are zero-padded; CineBrain rows pass through unchanged.
        """
        from datasets.cached_dataset import (
            collate_cached_with_future, set_active_vision_encoder,
        )
        set_active_vision_encoder('facebook/dinov2-base')

        N, d, W = 5, 40, 2

        def alljoined_like(C=64):
            return {
                'timeseries': torch.randn(C, N, d),
                'ch_coords': torch.randn(C, 3),
                'ch_names': [f'A{i}' for i in range(C)],
                'source': 'alljoined',
                'sfreq': torch.tensor(200.0),
            }

        def with_future(source, C):
            return {
                'timeseries': torch.randn(C, N, d),
                'ch_coords': torch.randn(C, 3),
                'ch_names': [f'{source[0].upper()}{i}' for i in range(C)],
                'source': source,
                'sfreq': torch.tensor(200.0),
                'timeseries_future': torch.randn(W, C, N, d),
                'pixel_values_future': torch.zeros(W, 3, 224, 224),
                'has_image_future': torch.zeros(W, dtype=torch.bool),
            }

        batch = [
            alljoined_like(),               # row 0  64ch  no future
            with_future('cinebrain', 64),   # row 1  64ch  future
            with_future('egobrain', 32),    # row 2  32ch  future, will pad
            with_future('cinebrain', 64),   # row 3  64ch  future
            with_future('egobrain', 32),    # row 4  32ch  future, will pad
        ]
        out = collate_cached_with_future(batch)

        self.assertEqual(out['timeseries'].shape, (5, 64, N, d))
        # 4 future-providing rows, all stacked at max_chs=64.
        self.assertEqual(out['timeseries_future'].shape, (4, W, 64, N, d))
        self.assertEqual(out['cinebrain_idx'].tolist(), [1, 2, 3, 4])
        # cb rows (positions 0, 2 in the future stack) keep all 64ch nonzero;
        # ego rows (positions 1, 3) have channels 32:64 zero-padded.
        cb_rows = out['timeseries_future'][[0, 2]]
        ego_rows = out['timeseries_future'][[1, 3]]
        self.assertGreater(cb_rows[:, :, 32:].abs().sum().item(), 0.0)
        self.assertEqual(ego_rows[:, :, 32:].abs().max().item(), 0.0)
        # valid_channel_mask reflects per-row real channel counts.
        self.assertEqual(
            out['valid_channel_mask'].sum(dim=-1).tolist(),
            [64, 64, 32, 64, 32])


if __name__ == '__main__':
    unittest.main()
