"""Tests for the hand-movement auxiliary regression objective:
the masked loss (utils.util.hand_regression_loss), the per-window label loader
(EgoBrainDataset._load_hand_labels, NaN-safe + per-column valid), and the
window-0 threading through collate_egobrain.
"""
import os
import tempfile
import unittest

import numpy as np
import torch

from utils.util import hand_regression_loss


class TestHandRegressionLoss(unittest.TestCase):
    def test_masks_invalid_columns(self):
        pred = torch.tensor([[1.0, 1.0]])
        target = torch.tensor([[0.0, 5.0]])           # left target 0, right 5
        # only LEFT valid -> loss sees only the (1-0) error, ignores right's huge err
        valid = torch.tensor([[1.0, 0.0]])
        loss, mae = hand_regression_loss(pred, target, valid)
        self.assertAlmostEqual(mae.item(), 1.0, places=5)        # |1-0| only
        self.assertGreater(loss.item(), 0.0)
        # right-only mask would give a much larger error
        loss_r, mae_r = hand_regression_loss(pred, target, torch.tensor([[0.0, 1.0]]))
        self.assertAlmostEqual(mae_r.item(), 4.0, places=5)      # |1-5|

    def test_no_valid_is_zero(self):
        pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        valid = torch.zeros(4, 2)
        loss, mae = hand_regression_loss(pred, target, valid)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(mae.item(), 0.0)

    def test_flip_swaps_left_right(self):
        pred = torch.tensor([[2.0, 9.0]])             # predicts left=2, right=9
        target = torch.tensor([[9.0, 2.0]])           # true left=9, right=2
        valid = torch.tensor([[1.0, 1.0]])
        # no flip: pred(2,9) vs target(9,2) -> big error
        loss_noflip, _ = hand_regression_loss(pred, target, valid, flip=False)
        # flip swaps target -> (2,9), matching pred -> ~0 error
        loss_flip, mae_flip = hand_regression_loss(pred, target, valid, flip=True)
        self.assertAlmostEqual(mae_flip.item(), 0.0, places=5)
        self.assertGreater(loss_noflip.item(), loss_flip.item())

    def test_flip_swaps_valid_too(self):
        # left invalid, right valid; after flip the *valid* column should move to
        # column 0 along with its target.
        pred = torch.tensor([[3.0, 0.0]])
        target = torch.tensor([[0.0, 3.0]])           # right target 3
        valid = torch.tensor([[0.0, 1.0]])            # right valid
        loss, mae = hand_regression_loss(pred, target, valid, flip=True)
        # flip -> target (3,0), valid (1,0): pred left 3 vs target 3 -> 0 error
        self.assertAlmostEqual(mae.item(), 0.0, places=5)

    def test_nan_in_unmasked_does_not_leak(self):
        # Targets are pre-sanitised (NaN->0) by the loader, but make sure a 0 in
        # an invalid column never contributes regardless.
        pred = torch.tensor([[1.0, 1.0]])
        target = torch.tensor([[0.0, 0.0]])
        valid = torch.tensor([[1.0, 0.0]])
        loss, _ = hand_regression_loss(pred, target, valid)
        self.assertTrue(torch.isfinite(loss).all())


class TestLoadHandLabels(unittest.TestCase):
    def _write_cache(self, d):
        import h5py
        # 3 clips x 2 windows. clip0: both finite+video; clip1: left NaN;
        # clip2: no video.
        li = np.array([[0.5, 0.6], [np.nan, 0.2], [0.1, 0.1]], np.float32)
        ri = np.array([[0.3, 0.4], [0.9, 0.8], [0.2, 0.2]], np.float32)
        hv = np.array([[True, True], [True, True], [False, False]])
        with h5py.File(os.path.join(d, 'P0001.h5'), 'w') as h:
            h.create_dataset('left_intensity', data=li)
            h.create_dataset('right_intensity', data=ri)
            h.create_dataset('has_video', data=hv)

    def test_loader(self):
        from datasets.egobrain_dataset import EgoBrainDataset, _H5_HAND_HANDLES
        with tempfile.TemporaryDirectory() as d:
            self._write_cache(d)
            fake = EgoBrainDataset.__new__(EgoBrainDataset)
            fake.hand_labels_dir = d
            fake.n_windows = 2

            t0, v0 = fake._load_hand_labels('P0001', 0)
            self.assertEqual(tuple(t0.shape), (2, 2))
            self.assertTrue(torch.allclose(t0[0], torch.tensor([0.5, 0.3])))
            self.assertTrue(v0.all())                      # all finite + video

            t1, v1 = fake._load_hand_labels('P0001', 1)    # left NaN
            self.assertEqual(t1[0, 0].item(), 0.0)         # NaN -> 0
            self.assertFalse(bool(v1[0, 0]))               # left invalid
            self.assertTrue(bool(v1[0, 1]))                # right valid
            self.assertTrue(torch.isfinite(t1).all())

            t2, v2 = fake._load_hand_labels('P0001', 2)    # no video
            self.assertFalse(v2.any())                     # all invalid
            _H5_HAND_HANDLES.clear()

    def test_missing_file_returns_zeros(self):
        from datasets.egobrain_dataset import EgoBrainDataset
        with tempfile.TemporaryDirectory() as d:
            fake = EgoBrainDataset.__new__(EgoBrainDataset)
            fake.hand_labels_dir = d
            fake.n_windows = 2
            t, v = fake._load_hand_labels('P9999', 0)
            self.assertEqual(tuple(t.shape), (2, 2))
            self.assertFalse(v.any())


class TestCollateThreading(unittest.TestCase):
    def _item(self, lt):
        W, C, N, d = 2, 2, 5, 4
        return {
            'timeseries': torch.zeros(W, C, N, d),
            'ch_coords': torch.zeros(C, 3),
            'ch_names': ['Cz', 'C3'],
            'pixel_values': torch.zeros(W, 3, 8, 8),
            'has_image': torch.zeros(W, dtype=torch.bool),
            'hand_targets': torch.tensor(lt, dtype=torch.float32),   # (W,2)
            'hand_valid': torch.ones(W, 2, dtype=torch.bool),
            'source': 'egobrain',
            'session_id': 'P0001',
        }

    def test_collate_egobrain_takes_window0(self):
        from datasets.egobrain_dataset import collate_egobrain
        batch = [self._item([[0.1, 0.2], [9.0, 9.0]]),
                 self._item([[0.3, 0.4], [9.0, 9.0]])]
        out = collate_egobrain(batch)
        self.assertEqual(tuple(out['hand_targets'].shape), (2, 2))
        # window 0 only
        self.assertTrue(torch.allclose(
            out['hand_targets'], torch.tensor([[0.1, 0.2], [0.3, 0.4]])))
        self.assertEqual(tuple(out['hand_valid'].shape), (2, 2))


class TestMixCollateDefaultFill(unittest.TestCase):
    """collate_cached (mix mode) must default-fill non-EgoBrain rows so the
    masked loss skips them, and return None when no row carries labels."""

    def _sample(self, hand):
        d = {'timeseries': torch.zeros(2, 5, 4), 'ch_coords': torch.zeros(2, 3),
             'ch_names': ['C3', 'C4'], 'source': 'x',
             'pixel_values': torch.zeros(3, 8, 8)}
        if hand:
            d['hand_targets'] = torch.tensor([0.5, 0.3])
            d['hand_valid'] = torch.tensor([True, True])
        return d

    def test_mixed_batch_defaults_absent_rows(self):
        from datasets.cached_dataset import collate_cached
        out = collate_cached([self._sample(True), self._sample(False)])
        self.assertEqual(tuple(out['hand_targets'].shape), (2, 2))
        self.assertEqual(out['hand_targets'][1].tolist(), [0.0, 0.0])  # absent -> 0
        self.assertTrue(out['hand_valid'][0].all())
        self.assertFalse(out['hand_valid'][1].any())                  # absent -> masked

    def test_no_labels_returns_none(self):
        from datasets.cached_dataset import collate_cached
        out = collate_cached([self._sample(False), self._sample(False)])
        self.assertIsNone(out['hand_targets'])    # model guard skips on None


class TestModelIntegration(unittest.TestCase):
    """End-to-end: the head runs inside CSBrainAlign.forward and emits the loss
    into the info dict (which the trainer auto-sums)."""

    def _enc(self, frame_averaging, aux=True):
        from models.alignment import CSBrainAlign
        return CSBrainAlign(
            in_dim=40, out_dim=40, d_model=40, dim_feedforward=160,
            seq_len=3, n_layer=2, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
            brain_regions=None, sorted_indices=[], causal=False,
            alignment_weight=0.0, frame_averaging=frame_averaging,
            flip_split_hidden=32, frame_avg_recon_weight=0.0,
            aux_hand_pred=aux, aux_hand_weight=0.1)

    def _batch(self, B=4, flip=None, targets=True):
        names = ['C3', 'C4', 'F3', 'F4']
        C, N = len(names), 3
        b = {'timeseries': torch.randn(B, C, N, 40) / 100.0,
             'ch_coords': torch.randn(B, C, 3).abs() + 0.1,
             'ch_names': [list(names) for _ in range(B)],
             'valid_channel_mask': torch.ones(B, C, dtype=torch.bool),
             'hand_targets': torch.rand(B, 2) if targets else None,
             'hand_valid': torch.ones(B, 2, dtype=torch.bool) if targets else None}
        if flip is not None:
            b['flip'] = flip
        return b

    def test_frameavg_emits_loss_and_backprops(self):
        enc = self._enc(frame_averaging=True); enc.train()
        _, info = enc(self._batch(flip=False), mask=None)
        self.assertIn('hand_pred_loss', info)
        w, loss = info['hand_pred_loss']
        self.assertEqual(w, 0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('diag_hand_mae', info)
        loss.backward()
        self.assertIsNotNone(enc.hand_pred_head[0].weight.grad)

    def test_standard_path_emits_loss(self):
        enc = self._enc(frame_averaging=False); enc.train()
        _, info = enc(self._batch(), mask=None)
        self.assertIn('hand_pred_loss', info)
        self.assertTrue(torch.isfinite(info['hand_pred_loss'][1]))

    def test_flip_step_runs(self):
        enc = self._enc(frame_averaging=True); enc.train()
        _, info = enc(self._batch(flip=True), mask=None)
        self.assertIn('hand_pred_loss', info)        # swap handled internally

    def test_disabled_emits_nothing(self):
        enc = self._enc(frame_averaging=True, aux=False); enc.train()
        _, info = enc(self._batch(flip=False), mask=None)
        self.assertNotIn('hand_pred_loss', info)
        self.assertFalse(hasattr(enc, 'hand_pred_head'))

    def test_no_targets_emits_nothing(self):
        enc = self._enc(frame_averaging=True); enc.train()
        _, info = enc(self._batch(flip=False, targets=False), mask=None)
        self.assertNotIn('hand_pred_loss', info)     # None targets -> skipped


if __name__ == '__main__':
    unittest.main()
