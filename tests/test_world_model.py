"""Smoke tests for the CineBrain + world-model pipeline.

Runs four checks end-to-end:
1. ``LatentPredictor`` accepts the expected shapes and returns matching ones.
2. ``CineBrainDataset`` / ``collate_cinebrain`` produce a batch with the
   keys the ``WorldModelWrapper`` expects (requires data on disk).
3. ``WorldModelWrapper`` forward produces a loss dict with every declared
   term and the shapes the trainer expects for masked reconstruction.
4. The assembled loss backprops to the encoder's parameters.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.alignment import CSBrainAlign
from models.world_model import LatentPredictor, WorldModelWrapper


def _tiny_encoder(in_dim=40, n_ch=8, n_patches=2):
    return CSBrainAlign(
        in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
        dim_feedforward=4 * in_dim, seq_len=n_patches,
        # CSBrainAlign.forward hard-codes the semantic-branch hook at
        # layer (num_layers - 3); anything smaller than 3 layers leaves
        # the branch_embs local uninitialised.
        n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
        brain_regions=None, sorted_indices=[], causal=False,
        alignment_weight=0.0,        # skip contrastive path
        equivariance_weight=0.0,
        patch_embed_type='cnn',
    )


class TestLatentPredictor(unittest.TestCase):
    def test_shapes(self):
        B, C, N, D = 2, 6, 3, 40
        p = LatentPredictor(d_model=D, predictor_d_model=64, n_layers=2,
                            n_heads=4, dim_feedforward=128, max_horizon=4)
        s = torch.randn(B, C, N, D)
        x = torch.randn(B, C, N, D)
        pred, cls = p(s, x, horizon=2)
        self.assertEqual(pred.shape, (B, C, N, D))
        self.assertEqual(cls.shape, (B, D))


class TestWorldModelWrapper(unittest.TestCase):
    def test_forward_and_backward(self):
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        pred = LatentPredictor(d_model=in_dim, predictor_d_model=64,
                               n_layers=2, n_heads=4, dim_feedforward=128,
                               max_horizon=4)
        wrapper = WorldModelWrapper(encoder=enc, predictor=pred,
                                    latent_pred_weight=1.0, cls_pred_weight=0.1,
                                    max_horizon=2, ramp_epochs=0)
        wrapper.train()

        B, W = 3, 3
        ts = torch.randn(B, W, n_ch, n_patches, in_dim)
        pv = torch.zeros(B, W, 3, 224, 224)
        has_img = torch.zeros(B, W, dtype=torch.bool)

        # Build coords from biosemi64 subset
        from datasets.cinebrain_dataset import _BIOSEMI64_COORDS
        coords = torch.from_numpy(_BIOSEMI64_COORDS[:n_ch]).unsqueeze(0).expand(B, -1, -1).contiguous()

        batch = {
            'timeseries': ts[:, 0] / 100.0,
            'timeseries_future': ts,
            'pixel_values_future': pv,
            'has_image_future': has_img,
            'ch_coords': coords,
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': pv[:, 0]},
            'has_image': has_img[:, 0],
            'source': ['cinebrain'] * B,
            'session_id': ['sub-0001'] * B,
        }

        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1  # mask first time patch on every channel

        out, info = wrapper.training_step(batch, mask=mask)

        # Reconstruction output must line up with the input shape so the
        # trainer's masked-reconstruction loss works.
        self.assertEqual(out.shape, batch['timeseries'].shape)

        # Every loss term declared in the plan must be present.
        for key in ('latent_pred_loss', 'latent_cls_loss'):
            self.assertIn(key, info)
            coef, tensor = info[key]
            self.assertTrue(torch.is_tensor(tensor))
            self.assertTrue(tensor.ndim == 0)

        # Backward step — verify at least one encoder param has a gradient.
        loss_terms = [v[0] * v[1] for k, v in info.items()
                      if isinstance(v, tuple) and 'loss' in k]
        mask_loss = (out[mask == 1] - batch['timeseries'][mask == 1]).pow(2).mean()
        total = mask_loss + sum(loss_terms)
        total.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in enc.patch_embedding.parameters())
        self.assertTrue(has_grad, "encoder.patch_embedding received no gradient")


class TestWorldModelZeroHorizon(unittest.TestCase):
    """``max_horizon=0`` should reduce the wrapper to plain CSBrainAlign.

    Verifies no predictor params are registered and the wrapper still
    returns the reconstruction output with no latent-prediction losses.
    """

    def test_reduces_to_plain_align(self):
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        wrapper = WorldModelWrapper(encoder=enc, predictor=None,
                                    max_horizon=0, ramp_epochs=0)

        # No predictor → no extra params beyond the encoder's.
        self.assertIs(wrapper.predictor, None)
        n_wrapper = sum(p.numel() for p in wrapper.parameters())
        n_encoder = sum(p.numel() for p in enc.parameters())
        self.assertEqual(n_wrapper, n_encoder)

        B = 2
        ts = torch.randn(B, n_ch, n_patches, in_dim) / 100.0
        from datasets.cinebrain_dataset import _BIOSEMI64_COORDS
        coords = torch.from_numpy(_BIOSEMI64_COORDS[:n_ch]).unsqueeze(0).expand(B, -1, -1).contiguous()
        batch = {
            'timeseries': ts,
            'ch_coords': coords,
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': torch.zeros(B, 3, 224, 224)},
            'has_image': torch.zeros(B, dtype=torch.bool),
            'source': ['cinebrain'] * B,
        }
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = wrapper.training_step(batch, mask=mask)
        self.assertEqual(out.shape, batch['timeseries'].shape)
        self.assertNotIn('latent_pred_loss', info)
        self.assertNotIn('latent_cls_loss', info)


class TestCineBrainDataset(unittest.TestCase):
    def setUp(self):
        # The dataset requires real data on disk; skip if it isn't there.
        self.root = 'data/CineBrain'
        if not os.path.isdir(self.root):
            self.skipTest(f"{self.root} not available")
        if not os.path.isdir(os.path.join(self.root, 'sub-0001', 'eeg_02')):
            self.skipTest('sub-0001 EEG not available')

    def test_getitem_and_collate(self):
        from datasets.cinebrain_dataset import (
            CineBrainDataset, collate_cinebrain)
        ds = CineBrainDataset(
            data_dir=self.root,
            subjects=['sub-0001'],
            in_dim=200,
            n_windows=3,
            load_frames=False,  # decord may be unavailable in CI
        )
        self.assertGreater(len(ds), 10)
        item = ds[0]
        self.assertEqual(item['timeseries'].shape[0], 3)   # W
        self.assertEqual(item['timeseries'].shape[-1], 200)  # in_dim

        batch = collate_cinebrain([ds[0], ds[1]])
        self.assertEqual(batch['timeseries'].shape[0], 2)  # B
        self.assertIn('timeseries_future', batch)
        self.assertEqual(batch['timeseries_future'].shape[1], 3)  # W


class TestWorldModelMixMode(unittest.TestCase):
    """Simulate ``collate_cached_with_future``: batch of size B, but future
    stacks only contain the M<B CineBrain rows. Before the fix, the wrapper
    would index ``patch_emb`` (B rows) with ``has_image_future[:, 0]`` (M
    rows), crashing at alignment.py:784.
    """

    def test_mix_mode_future_is_M_rows(self):
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        pred = LatentPredictor(d_model=in_dim, predictor_d_model=64,
                               n_layers=2, n_heads=4, dim_feedforward=128,
                               max_horizon=4)
        wrapper = WorldModelWrapper(encoder=enc, predictor=pred,
                                    latent_pred_weight=1.0,
                                    cls_pred_weight=0.1,
                                    max_horizon=2, ramp_epochs=0)
        wrapper.train()

        B, M, W = 5, 2, 3  # mix batch: 5 rows total, only 2 from CineBrain
        cb_idx = torch.tensor([1, 3], dtype=torch.long)

        from datasets.cinebrain_dataset import _BIOSEMI64_COORDS
        coords = torch.from_numpy(
            _BIOSEMI64_COORDS[:n_ch]
        ).unsqueeze(0).expand(B, -1, -1).contiguous()

        ts_full = torch.randn(B, n_ch, n_patches, in_dim) / 100.0
        # Future stacks only carry the M CineBrain rows — mirror what
        # collate_cached_with_future produces.
        ts_future_M = torch.randn(M, W, n_ch, n_patches, in_dim)
        pv_future_M = torch.zeros(M, W, 3, 224, 224)
        has_image_future_M = torch.zeros(M, W, dtype=torch.bool)

        # Full-batch image fields at B rows (from collate_cached).
        pv_B = torch.zeros(B, 3, 224, 224)
        has_image_B = torch.zeros(B, dtype=torch.bool)

        batch = {
            'timeseries': ts_full,
            'ch_coords': coords,
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': pv_B},
            'has_image': has_image_B,
            'cinebrain_idx': cb_idx,
            'timeseries_future': ts_future_M,
            'pixel_values_future': pv_future_M,
            'has_image_future': has_image_future_M,
            'source': ['alljoined', 'cinebrain', 'alljoined',
                       'cinebrain', 'alljoined'],
            'session_id': [f'sess-{i}' for i in range(B)],
        }
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1

        out, info = wrapper.training_step(batch, mask=mask)
        self.assertEqual(out.shape, batch['timeseries'].shape)
        # Prediction branch should fire and use the M-row future stacks.
        self.assertIn('latent_pred_loss', info)
        self.assertIn('latent_cls_loss', info)
        # Sanity: loss is finite and scalar
        for key in ('latent_pred_loss', 'latent_cls_loss'):
            coef, val = info[key]
            self.assertTrue(torch.isfinite(val).item(),
                            f'{key} not finite: {val}')

        # Backward works end-to-end
        loss_terms = [v[0] * v[1] for k, v in info.items()
                      if isinstance(v, tuple) and 'loss' in k]
        mask_loss = (
            out[mask == 1] - batch['timeseries'][mask == 1]
        ).pow(2).mean()
        (mask_loss + sum(loss_terms)).backward()

    def test_real_collate_cached_with_future(self):
        """End-to-end: feed the real ``collate_cached_with_future`` a
        mixed list of Alljoined-like + CineBrain-like per-sample dicts and
        push the result through the wrapper. Pins down the collate
        output shapes the wrapper relies on.
        """
        in_dim, n_ch, n_patches = 40, 8, 2
        from datasets.cached_dataset import collate_cached_with_future
        from datasets.cinebrain_dataset import _BIOSEMI64_COORDS

        coords = torch.from_numpy(_BIOSEMI64_COORDS[:n_ch]).float()
        sfreq = torch.tensor(200.0, dtype=torch.float32)

        def _aj(with_image: bool):
            d = {
                'timeseries': torch.randn(n_ch, n_patches, in_dim),
                'ch_coords': coords.clone(),
                'ch_names': ['pad'] * n_ch,
                'source': 'alljoined',
                'session_id': 'sess-aj',
                'sfreq': sfreq,
            }
            if with_image:
                d['pixel_values'] = torch.zeros(3, 224, 224)
            return d

        def _cb(W: int):
            d = _aj(with_image=True)
            d['source'] = 'cinebrain'
            d['session_id'] = 'sub-0001'
            d['timeseries_future'] = torch.randn(W, n_ch, n_patches, in_dim)
            d['pixel_values_future'] = torch.zeros(W, 3, 224, 224)
            d['has_image_future'] = torch.zeros(W, dtype=torch.bool)
            return d

        W = 3
        raw_batch = [_aj(False), _cb(W), _aj(True), _cb(W), _aj(False)]
        batch = collate_cached_with_future(raw_batch)

        B = len(raw_batch)
        self.assertEqual(batch['timeseries'].shape[0], B)
        self.assertEqual(batch['has_image'].shape[0], B)
        # Future stacks at M=2, not B=5
        self.assertEqual(batch['cinebrain_idx'].tolist(), [1, 3])
        self.assertEqual(batch['timeseries_future'].shape[0], 2)
        self.assertEqual(batch['pixel_values_future'].shape[0], 2)
        self.assertEqual(batch['has_image_future'].shape[0], 2)

        # Trainer normally divides by 100 before calling the wrapper;
        # mirror that here.
        batch['timeseries'] = batch['timeseries'] / 100.0

        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        pred = LatentPredictor(d_model=in_dim, predictor_d_model=64,
                               n_layers=2, n_heads=4, dim_feedforward=128,
                               max_horizon=4)
        wrapper = WorldModelWrapper(encoder=enc, predictor=pred,
                                    max_horizon=2, ramp_epochs=0)
        wrapper.train()

        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = wrapper.training_step(batch, mask=mask)
        self.assertEqual(out.shape, batch['timeseries'].shape)
        self.assertIn('latent_pred_loss', info)

    def test_mix_mode_no_cinebrain_rows_this_step(self):
        """If the mix sampler drew only Alljoined rows, the future-aware
        collate returns no future keys and no ``cinebrain_idx``. The
        wrapper should fall back to plain CSBrainAlign (no prediction
        losses) without crashing."""
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        pred = LatentPredictor(d_model=in_dim, predictor_d_model=64,
                               n_layers=2, n_heads=4, dim_feedforward=128,
                               max_horizon=4)
        wrapper = WorldModelWrapper(encoder=enc, predictor=pred,
                                    max_horizon=2, ramp_epochs=0)
        wrapper.train()

        B = 3
        from datasets.cinebrain_dataset import _BIOSEMI64_COORDS
        coords = torch.from_numpy(
            _BIOSEMI64_COORDS[:n_ch]
        ).unsqueeze(0).expand(B, -1, -1).contiguous()
        batch = {
            'timeseries': torch.randn(B, n_ch, n_patches, in_dim) / 100.0,
            'ch_coords': coords,
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {
                'pixel_values': torch.zeros(B, 3, 224, 224)},
            'has_image': torch.zeros(B, dtype=torch.bool),
            'source': ['alljoined'] * B,
        }
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = wrapper.training_step(batch, mask=mask)
        self.assertEqual(out.shape, batch['timeseries'].shape)
        self.assertNotIn('latent_pred_loss', info)


if __name__ == '__main__':
    unittest.main(verbosity=2)
