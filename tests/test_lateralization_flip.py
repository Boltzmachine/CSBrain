"""Tests for the bilateralization prior (learned x = x_bi + x_lat split +
flip-equivariant image alignment).

1. ``LateralizationSplit`` produces an *exact* additive decomposition
   ``x_bi + x_lat == x`` with a bounded gate, and starts near-identity.
2. ``build_flip_perm_batch`` swaps homologous channels (C3<->C4), fixes
   midline channels, and is an involution.
3. The full ``CSBrainAlign`` forward with ``lateralization_flip=True`` emits a
   finite ``flip_align_loss`` / ``lat_sparsity_loss`` and backprops into the
   split + projection heads (and the backbone).
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.alignment import (
    CSBrainAlign, LateralizationSplit, build_flip_perm_batch,
)
from models.world_model import LatentPredictor, WorldModelWrapper


class TestWorldModelFlipPrediction(unittest.TestCase):
    """The world model's prediction loss is applied to the flipped stream:
    predict the flipped-future EEG latent from the flipped-current, with the
    flipped future built by the same x_bi+flip(x_lat) split. The frozen-ish
    split must receive a temporal (dynamics) gradient — not just image
    alignment."""

    NAMES = ['C3', 'C4', 'CZ', 'FC3', 'FC4', 'F3', 'F4', 'PZ']

    def test_flip_prediction_terms_and_grad(self):
        in_dim, n_ch, n_patches = 40, len(self.NAMES), 2
        enc = CSBrainAlign(
            in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
            dim_feedforward=4 * in_dim, seq_len=n_patches,
            n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
            brain_regions=None, sorted_indices=[], causal=False,
            alignment_weight=0.0, lateralization_flip=True,
            flip_split_hidden=64, flip_n_col_bands=2,
        )
        pred = LatentPredictor(d_model=in_dim, predictor_d_model=64,
                               n_layers=2, n_heads=4, dim_feedforward=128,
                               max_horizon=4)
        wrapper = WorldModelWrapper(encoder=enc, predictor=pred,
                                    latent_pred_weight=1.0, cls_pred_weight=0.1,
                                    max_horizon=2, ramp_epochs=0,
                                    flip_pred_weight=1.0)
        wrapper.train()

        B, M, W = 5, 2, 3
        cb_idx = torch.tensor([1, 3], dtype=torch.long)
        coords = (torch.randn(B, n_ch, 3).abs() + 0.1)
        batch = {
            'timeseries': torch.randn(B, n_ch, n_patches, in_dim) / 100.0,
            'ch_coords': coords,
            'ch_names': [list(self.NAMES) for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': torch.zeros(B, 3, 224, 224)},
            'has_image': torch.zeros(B, dtype=torch.bool),   # no image alignment
            'cinebrain_idx': cb_idx,
            'timeseries_future': torch.randn(M, W, n_ch, n_patches, in_dim),
            'pixel_values_future': torch.zeros(M, W, 3, 224, 224),
            'has_image_future': torch.zeros(M, W, dtype=torch.bool),
            'source': ['alljoined', 'egobrain', 'alljoined', 'egobrain', 'alljoined'],
        }
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1

        out, info = wrapper.training_step(batch, mask=mask)
        for key in ('latent_pred_loss', 'flip_latent_pred_loss',
                    'flip_latent_cls_loss'):
            self.assertIn(key, info)
            coef, val = info[key]
            self.assertTrue(torch.is_tensor(val) and val.ndim == 0)
            self.assertTrue(torch.isfinite(val).item(), f'{key}={val}')
        self.assertIn('diag_flip_latent_pred_cos', info)

        # The flipped prediction must train the learned split (dynamics grad),
        # even with image alignment off (has_image all False).
        loss = sum(v[0] * v[1] for k, v in info.items()
                   if isinstance(v, tuple) and 'loss' in k)
        loss.backward()
        split_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in enc.lateralization_split.parameters())
        self.assertTrue(split_grad,
                        "lateralization_split got no gradient from flip prediction")

    def test_flip_pred_weight_zero_disables(self):
        in_dim, n_ch, n_patches = 40, len(self.NAMES), 2
        enc = CSBrainAlign(
            in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
            dim_feedforward=4 * in_dim, seq_len=n_patches,
            n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
            brain_regions=None, sorted_indices=[], causal=False,
            alignment_weight=0.0, lateralization_flip=True, flip_split_hidden=64,
        )
        pred = LatentPredictor(d_model=in_dim, predictor_d_model=64,
                               n_layers=2, n_heads=4, dim_feedforward=128,
                               max_horizon=4)
        wrapper = WorldModelWrapper(encoder=enc, predictor=pred,
                                    max_horizon=2, ramp_epochs=0,
                                    flip_pred_weight=0.0)
        wrapper.train()
        B, M, W = 4, 2, 3
        batch = {
            'timeseries': torch.randn(B, n_ch, n_patches, in_dim) / 100.0,
            'ch_coords': torch.randn(B, n_ch, 3).abs() + 0.1,
            'ch_names': [list(self.NAMES) for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': torch.zeros(B, 3, 224, 224)},
            'has_image': torch.zeros(B, dtype=torch.bool),
            'cinebrain_idx': torch.tensor([0, 2], dtype=torch.long),
            'timeseries_future': torch.randn(M, W, n_ch, n_patches, in_dim),
            'pixel_values_future': torch.zeros(M, W, 3, 224, 224),
            'has_image_future': torch.zeros(M, W, dtype=torch.bool),
            'source': ['egobrain'] * B,
        }
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        _, info = wrapper.training_step(batch, mask=mask)
        self.assertNotIn('flip_latent_pred_loss', info)


class TestLateralizationSplit(unittest.TestCase):
    def test_additive_decomposition_is_exact(self):
        B, C, N, d = 2, 6, 3, 40
        coord_pe_dim = 192
        split = LateralizationSplit(in_dim=d, coord_pe_dim=coord_pe_dim, hidden=32)
        x = torch.randn(B, C, N, d)
        coord_pe = torch.randn(B, C, coord_pe_dim)
        x_lat, gate = split(x, coord_pe)
        x_bi = x - x_lat
        # x_bi + x_lat == x by construction.
        self.assertTrue(torch.allclose(x_bi + x_lat, x, atol=1e-6))
        # Gate is a valid (0, 1) fraction.
        self.assertGreaterEqual(gate.min().item(), 0.0)
        self.assertLessEqual(gate.max().item(), 1.0)

    def test_near_identity_init(self):
        # With the near-zero gate init, x_lat starts small relative to x.
        B, C, N, d = 4, 8, 3, 40
        split = LateralizationSplit(in_dim=d, coord_pe_dim=192, hidden=32)
        x = torch.randn(B, C, N, d)
        coord_pe = torch.zeros(B, C, 192)
        x_lat, gate = split(x, coord_pe)
        # gate_init_bias=-2 -> sigmoid ~= 0.12
        self.assertLess(gate.mean().item(), 0.2)

    def test_valid_channel_mask_zeros_lateral(self):
        B, C, N, d = 2, 5, 3, 40
        split = LateralizationSplit(in_dim=d, coord_pe_dim=192, hidden=16)
        x = torch.randn(B, C, N, d)
        coord_pe = torch.randn(B, C, 192)
        vcm = torch.ones(B, C, dtype=torch.bool)
        vcm[:, -2:] = False  # last two channels padded/invalid
        x_lat, gate = split(x, coord_pe, valid_channel_mask=vcm)
        self.assertTrue(torch.all(x_lat[:, -2:] == 0))
        self.assertTrue(torch.all(gate[:, -2:] == 0))


class TestFlipPerm(unittest.TestCase):
    def test_swaps_homologues_and_fixes_midline(self):
        names = [['C3', 'C4', 'F3', 'F4', 'Cz', 'Pz', 'O1', 'O2']]
        perm = build_flip_perm_batch(names)[0].tolist()
        # C3(0)<->C4(1), F3(2)<->F4(3), Cz(4)->Cz, Pz(5)->Pz, O1(6)<->O2(7)
        self.assertEqual(perm, [1, 0, 3, 2, 4, 5, 7, 6])

    def test_is_involution(self):
        names = [['C3', 'C4', 'F3', 'F4', 'Cz', 'Pz', 'O1', 'O2']]
        perm = build_flip_perm_batch(names)[0]
        # Applying the permutation twice returns the identity.
        self.assertTrue(torch.equal(perm[perm], torch.arange(len(perm))))

    def test_flip_construction_swaps_channel_data(self):
        # x_flip = x_bi + flip(x_lat); when x_bi=0 and x_lat=x, x_flip = S(x).
        names = [['C3', 'C4', 'Cz']]
        B, C, N, d = 1, 3, 2, 4
        perm = build_flip_perm_batch(names).expand(B, -1)
        x_lat = torch.randn(B, C, N, d)
        perm_exp = perm[:, :C].view(B, C, 1, 1).expand(B, C, N, d)
        x_lat_flip = torch.gather(x_lat, 1, perm_exp)
        # C3 and C4 channels swap; Cz unchanged.
        self.assertTrue(torch.equal(x_lat_flip[:, 0], x_lat[:, 1]))
        self.assertTrue(torch.equal(x_lat_flip[:, 1], x_lat[:, 0]))
        self.assertTrue(torch.equal(x_lat_flip[:, 2], x_lat[:, 2]))


class TestFlipAlignmentForward(unittest.TestCase):
    """Full encoder smoke: forward emits the flip losses and they backprop."""

    def _encoder(self, in_dim=40, n_patches=2, n_bands=2, flip_motion_ref=0.0):
        return CSBrainAlign(
            in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
            dim_feedforward=4 * in_dim, seq_len=n_patches,
            n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
            brain_regions=None, sorted_indices=[], causal=False,
            alignment_weight=0.0,          # isolate the flip path
            equivariance_weight=0.0,
            patch_embed_type='cnn',
            lateralization_flip=True,
            flip_align_weight=1.0,
            lat_sparsity_weight=0.05,
            flip_n_col_bands=n_bands,
            flip_motion_ref=flip_motion_ref,
        )

    def test_forward_backward(self):
        in_dim, n_patches = 40, 2
        names = ['C3', 'C4', 'F3', 'F4', 'Cz', 'Pz', 'O1', 'O2']
        n_ch = len(names)
        enc = self._encoder(in_dim=in_dim, n_patches=n_patches)
        enc.train()

        B = 4
        x = torch.randn(B, n_ch, n_patches, in_dim) / 100.0
        # Finite spherical-ish coords; exact geometry is irrelevant here.
        coords = torch.randn(B, n_ch, 3).abs() + 0.1
        # Random (non-zero) frames so flip(image) != image under DINOv2.
        pv = torch.randn(B, 3, 224, 224)
        has_image = torch.tensor([True, True, True, False])

        batch = {
            'timeseries': x,
            'ch_coords': coords,
            'ch_names': [list(names) for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': pv},
            'has_image': has_image,
            'source': ['egobrain'] * B,
        }
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1

        out, info = enc.training_step(batch, mask=mask)
        self.assertEqual(out.shape, batch['timeseries'].shape)

        # Flip-path terms present, finite, scalar.
        for key in ('flip_align_loss', 'lat_sparsity_loss'):
            self.assertIn(key, info)
            coef, val = info[key]
            self.assertTrue(torch.is_tensor(val) and val.ndim == 0)
            self.assertTrue(torch.isfinite(val).item(), f'{key}={val}')
        self.assertIn('flip_align_acc', info)
        self.assertIn('diag_flip_discrim_acc', info)
        self.assertIn('diag_lat_gate_mean', info)

        # Backprop the flip terms; the split + projection heads must receive
        # gradient (they are reachable ONLY through the flip path).
        loss = sum(v[0] * v[1] for k, v in info.items()
                   if isinstance(v, tuple) and 'loss' in k)
        loss.backward()

        def _has_grad(module):
            return any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in module.parameters())

        self.assertTrue(_has_grad(enc.lateralization_split),
                        'LateralizationSplit received no gradient')
        self.assertTrue(_has_grad(enc.flip_align_proj),
                        'flip_align_proj received no gradient')
        self.assertTrue(_has_grad(enc.patch_embedding),
                        'backbone patch_embedding received no gradient')

    def test_image_descriptor_shape_and_flip_sensitivity(self):
        enc = self._encoder(n_bands=4)
        self.assertEqual(enc.flip_align_proj[-1].out_features,
                         4 * enc.image_feature_dim)
        pv = torch.randn(3, 3, 224, 224)
        desc = enc._image_lateral_descriptor(pv)
        self.assertEqual(desc.shape, (3, 4 * enc.image_feature_dim))
        desc_flip = enc._image_lateral_descriptor(torch.flip(pv, dims=[-1]))
        # Centered spatial descriptor must change under a horizontal flip.
        self.assertGreater((desc - desc_flip).abs().mean().item(), 0.0)
        # Target carries no gradient (image encoder is frozen).
        self.assertFalse(desc.requires_grad)

    def test_motion_weighting_through_wrapper(self):
        from models.world_model import WorldModelWrapper
        in_dim, n_patches = 40, 2
        names = ['C3', 'C4', 'F3', 'F4', 'Cz', 'Pz', 'O1', 'O2']
        n_ch = len(names)
        enc = self._encoder(in_dim=in_dim, n_patches=n_patches,
                            n_bands=2, flip_motion_ref=10.0)
        wrapper = WorldModelWrapper(encoder=enc, predictor=None,
                                    max_horizon=0, ramp_epochs=0)
        wrapper.train()

        B, W = 4, 2
        ts = torch.randn(B, W, n_ch, n_patches, in_dim) / 100.0
        # Future frames differ from window-0 frames -> nonzero motion.
        pv0 = torch.randn(B, 3, 224, 224)
        pvf = torch.stack([pv0, pv0 + torch.randn_like(pv0)], dim=1)  # (B,W,3,H,W)
        has_img = torch.tensor([True, True, True, False])
        batch = {
            'timeseries': ts[:, 0],
            'timeseries_future': ts,
            'pixel_values_future': pvf,
            'has_image_future': torch.stack([has_img, has_img], dim=1),
            'ch_coords': torch.randn(B, n_ch, 3).abs() + 0.1,
            'ch_names': [list(names) for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': pv0},
            'has_image': has_img,
            'source': ['egobrain'] * B,
        }
        # The wrapper should compute a (B,) motion vector with real values on
        # the future rows.
        motion = wrapper._compute_flip_motion(batch)
        self.assertEqual(tuple(motion.shape), (B,))
        self.assertTrue((motion >= 0).all().item())

        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = wrapper.training_step(batch, mask=mask)
        self.assertIn('flip_align_loss', info)
        self.assertIn('diag_flip_motion_weight', info)
        coef, val = info['flip_align_loss']
        self.assertTrue(torch.isfinite(val).item())

    def test_disabled_by_default(self):
        # Without the flag the loss keys must not appear (and no extra params).
        in_dim, n_patches = 40, 2
        enc = CSBrainAlign(
            in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
            dim_feedforward=4 * in_dim, seq_len=n_patches,
            n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
            brain_regions=None, sorted_indices=[], causal=False,
            alignment_weight=0.0, equivariance_weight=0.0,
            patch_embed_type='cnn',
        )
        self.assertFalse(hasattr(enc, 'lateralization_split'))
        self.assertFalse(enc.lateralization_flip)


if __name__ == '__main__':
    unittest.main(verbosity=2)
