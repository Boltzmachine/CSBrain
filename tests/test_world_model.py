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
from models.world_model import (
    LatentPredictor, FramePredictor, WorldModelWrapper,
    EGOBRAIN_FRAME_MOTION_REF_PER_STEP)


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
        pred, cls = p(s, horizon=2)
        self.assertEqual(pred.shape, (B, C, N, D))
        self.assertEqual(cls.shape, (B, D))


class TestFramePredictor(unittest.TestCase):
    def test_shapes(self):
        B, P, frame_dim, eeg_dim, H = 2, 16, 32, 40, 4
        p = FramePredictor(frame_dim=frame_dim, eeg_dim=eeg_dim,
                           predictor_d_model=64, n_layers=2, n_heads=4,
                           dim_feedforward=128, max_horizon=H)
        s = torch.randn(B, P, frame_dim)
        eeg = torch.randn(B, eeg_dim)
        pred = p(s, eeg)
        # Dense: one grid per 0.2 s step -> (B, H, P, frame_dim).
        self.assertEqual(pred.shape, (B, H, P, frame_dim))

    def test_shapes_token_conditioning(self):
        """The predictor also accepts the full EEG token set (B, M, eeg_dim)."""
        B, P, frame_dim, eeg_dim, M, H = 2, 16, 32, 40, 7, 4
        p = FramePredictor(frame_dim=frame_dim, eeg_dim=eeg_dim,
                           predictor_d_model=64, n_layers=2, n_heads=4,
                           dim_feedforward=128, max_horizon=H)
        s = torch.randn(B, P, frame_dim)
        eeg = torch.randn(B, M, eeg_dim)
        pred = p(s, eeg)
        self.assertEqual(pred.shape, (B, H, P, frame_dim))

    def test_eeg_conditioning_changes_output(self):
        """Zeroing the EEG embedding must change the prediction — otherwise the
        predictor ignores the conditioning and the objective is vacuous."""
        B, P, frame_dim, eeg_dim = 2, 16, 32, 40
        p = FramePredictor(frame_dim=frame_dim, eeg_dim=eeg_dim,
                           predictor_d_model=64, n_layers=2, n_heads=4,
                           dim_feedforward=128, max_horizon=4)
        s = torch.randn(B, P, frame_dim)
        eeg = torch.randn(B, eeg_dim)
        a = p(s, eeg)
        b = p(s, torch.zeros_like(eeg))
        self.assertGreater((a - b).abs().mean().item(), 1e-6)

    def test_padding_mask_invariance(self):
        """Masked EEG tokens must not influence the frame prediction at all
        (excluded as attention keys)."""
        B, P, frame_dim, eeg_dim, M = 2, 8, 16, 12, 5
        p = FramePredictor(frame_dim=frame_dim, eeg_dim=eeg_dim,
                           predictor_d_model=64, n_layers=2, n_heads=4,
                           dim_feedforward=128, max_horizon=4)
        p.eval()  # deterministic (no dropout)
        s = torch.randn(B, P, frame_dim)
        eeg = torch.randn(B, M, eeg_dim)
        kpm = torch.zeros(B, M, dtype=torch.bool)
        kpm[:, 3:] = True  # mark the last two EEG tokens as padding

        with torch.no_grad():
            out1 = p(s, eeg, eeg_key_padding_mask=kpm)
            eeg2 = eeg.clone()
            eeg2[:, 3:] = torch.randn(B, M - 3, eeg_dim)  # perturb only masked
            out2 = p(s, eeg2, eeg_key_padding_mask=kpm)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5),
                        "masked EEG tokens changed the prediction")


class TestWorldModelFrameObjective(unittest.TestCase):
    """``objective='frame'`` swaps the EEG-latent predictor for the video-frame
    predictor. The frozen vision encoder is monkeypatched so the test never runs
    the heavy DINOv2 forward; we only validate the wrapper plumbing + backward.
    """

    @staticmethod
    def _patch_grid(enc, s_grid=4):
        d_img = enc.image_feature_dim
        return lambda pv: torch.randn(pv.shape[0], s_grid, s_grid, d_img)

    def _run(self, batch, enc, max_horizon=2, frame_eeg_cond='global'):
        # Predictor's dense output width must match the wrapper's max_horizon
        # (= number of 0.2 s target frames = W - 1).
        pred = FramePredictor(frame_dim=enc.image_feature_dim, eeg_dim=enc.d_model,
                              predictor_d_model=64, n_layers=2, n_heads=4,
                              dim_feedforward=128, max_horizon=max_horizon)
        wrapper = WorldModelWrapper(encoder=enc, predictor=pred,
                                    latent_pred_weight=1.0, max_horizon=max_horizon,
                                    ramp_epochs=0, objective='frame',
                                    frame_eeg_cond=frame_eeg_cond)
        wrapper.train()
        # No EMA target encoder for the frame objective (frozen vision target).
        self.assertIsNone(wrapper.target_encoder)
        enc._image_patch_grid = self._patch_grid(enc)
        return wrapper.training_step(batch, mask=batch.pop('_mask'))

    @staticmethod
    def _pure_batch(n_ch, n_patches, in_dim, B=3, W=3, pad_last_ch=False):
        ts = torch.randn(B, W, n_ch, n_patches, in_dim)
        pv = torch.zeros(B, W, 3, 224, 224)
        # Future stack has frames at every window; the encoder's own image path
        # is disabled (has_image=False) so only the frame predictor uses frames.
        has_future = torch.ones(B, W, dtype=torch.bool)
        from datasets.cinebrain_dataset import _BIOSEMI64_COORDS
        coords = torch.from_numpy(
            _BIOSEMI64_COORDS[:n_ch]).unsqueeze(0).expand(B, -1, -1).contiguous()
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        vcm = torch.ones(B, n_ch, dtype=torch.bool)
        if pad_last_ch:
            # Mark the last channel as padding so the predictor's EEG-token
            # padding mask path is exercised end-to-end (tokens conditioning).
            vcm[:, -1] = False
            mask[:, -1, :] = 0  # don't score recon on the padded channel
        return {
            'timeseries': ts[:, 0] / 100.0,
            'timeseries_future': ts,
            'pixel_values_future': pv,
            'has_image_future': has_future,
            'ch_coords': coords,
            'valid_channel_mask': vcm,
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': pv[:, 0]},
            'has_image': torch.zeros(B, dtype=torch.bool),
            'source': ['egobrain'] * B,
            '_mask': mask,
        }

    def _check_pure(self, enc, batch, frame_eeg_cond):
        mask = batch['_mask']
        out, info = self._run(batch, enc, frame_eeg_cond=frame_eeg_cond)
        self.assertEqual(out.shape, batch['timeseries'].shape)
        # Frame objective emits frame_pred_loss; the EEG-latent terms are gone.
        self.assertIn('frame_pred_loss', info)
        self.assertNotIn('latent_pred_loss', info)
        coef, val = info['frame_pred_loss']
        self.assertTrue(torch.isfinite(val).item())
        self.assertIn('diag_frame_eeg_gap', info)

        loss_terms = [v[0] * v[1] for k, v in info.items()
                      if isinstance(v, tuple) and 'loss' in k]
        mask_loss = (out[mask == 1] - batch['timeseries'][mask == 1]).pow(2).mean()
        (mask_loss + sum(loss_terms)).backward()
        # Gradient must reach the EEG backbone through the EEG conditioning.
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in enc.patch_embedding.parameters())
        self.assertTrue(has_grad, "encoder.patch_embedding received no gradient")

    def test_pure_mode_global_cond(self):
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        batch = self._pure_batch(n_ch, n_patches, in_dim)
        self._check_pure(enc, batch, frame_eeg_cond='global')

    def test_pure_mode_tokens_cond(self):
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        # Include a padded channel so the EEG-token key-padding mask is built.
        batch = self._pure_batch(n_ch, n_patches, in_dim, pad_last_ch=True)
        self._check_pure(enc, batch, frame_eeg_cond='tokens')

    def test_mix_mode_cb_idx_alignment(self):
        """Mix mode: future stacks carry only the M<B frame rows; the frame
        objective must map them back via cb_idx and filter by per-row validity.
        """
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)

        B, M, W = 5, 2, 3
        cb_idx = torch.tensor([1, 3], dtype=torch.long)
        from datasets.cinebrain_dataset import _BIOSEMI64_COORDS
        coords = torch.from_numpy(
            _BIOSEMI64_COORDS[:n_ch]).unsqueeze(0).expand(B, -1, -1).contiguous()

        # Row 0 valid at both windows; row 1 missing the future frame -> dropped.
        has_future = torch.ones(M, W, dtype=torch.bool)
        has_future[1, 2] = False

        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        batch = {
            'timeseries': torch.randn(B, n_ch, n_patches, in_dim) / 100.0,
            'ch_coords': coords,
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': torch.zeros(B, 3, 224, 224)},
            'has_image': torch.zeros(B, dtype=torch.bool),
            'cinebrain_idx': cb_idx,
            'timeseries_future': torch.randn(M, W, n_ch, n_patches, in_dim),
            'pixel_values_future': torch.zeros(M, W, 3, 224, 224),
            'has_image_future': has_future,
            'source': ['alljoined', 'egobrain', 'alljoined', 'egobrain', 'alljoined'],
            '_mask': mask,
        }

        out, info = self._run(batch, enc, max_horizon=2)
        self.assertIn('frame_pred_loss', info)
        coef, val = info['frame_pred_loss']
        self.assertTrue(torch.isfinite(val).item())
        loss_terms = [v[0] * v[1] for k, v in info.items()
                      if isinstance(v, tuple) and 'loss' in k]
        mask_loss = (out[mask == 1] - batch['timeseries'][mask == 1]).pow(2).mean()
        (mask_loss + sum(loss_terms)).backward()


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


class TestFrameMotionWeighting(unittest.TestCase):
    """Unit tests for the per-patch motion weighting of the dense frame
    objective (``WorldModelWrapper._motion_weight`` / ``_reduce_over_patches``).
    These are pure tensor ops, so no encoder / vision weights are needed.
    """

    def _tgt(self, moving_patch_delta=1.0):
        # (Bv, H, P, d): P=3 patches. Anchor is a fixed random grid; the target
        # equals the anchor EXCEPT patch 0, which moves by `moving_patch_delta`.
        torch.manual_seed(0)
        Bv, H, P, d = 2, 2, 3, 4
        s_anchor = torch.randn(Bv, P, d)
        s_tgt = s_anchor.unsqueeze(1).expand(Bv, H, P, d).clone()
        s_tgt[:, :, 0, :] += moving_patch_delta          # only patch 0 moves
        tgt_valid = torch.ones(Bv, H)
        return s_anchor, s_tgt, tgt_valid

    def test_alpha_zero_is_uniform_mean(self):
        # alpha<=0 -> no weight, reducer is a plain mean (legacy path).
        s_anchor, s_tgt, tgt_valid = self._tgt()
        w, ref = WorldModelWrapper._motion_weight(
            s_anchor, s_tgt, tgt_valid, alpha=0.0, floor=0.1)
        self.assertIsNone(w)
        self.assertIsNone(ref)
        x = torch.randn(2, 2, 3)
        self.assertTrue(torch.allclose(
            WorldModelWrapper._reduce_over_patches(x, None), x.mean(dim=-1)))

    def test_weight_concentrates_on_moving_patch(self):
        # With alpha>0 the moving patch (patch 0) must get a strictly larger
        # weight than the static patches, so a per-patch error that is large only
        # on patch 0 produces a HIGHER weighted loss than the uniform mean.
        s_anchor, s_tgt, tgt_valid = self._tgt(moving_patch_delta=2.0)
        w, ref = WorldModelWrapper._motion_weight(
            s_anchor, s_tgt, tgt_valid, alpha=1.0, floor=0.1)
        self.assertEqual(w.shape, (2, 2, 3))
        self.assertEqual(ref.shape, (2,))                # per-horizon reference
        self.assertTrue((ref > 0).all())
        # Patch 0 (moved) weighted above the static patches (floored).
        self.assertTrue((w[..., 0] > w[..., 1]).all())
        self.assertTrue(torch.allclose(w[..., 1], w[..., 2]))          # both static
        # Error that lives only on the moving patch: weighted >> uniform.
        err = torch.zeros(2, 2, 3)
        err[..., 0] = 1.0
        weighted = WorldModelWrapper._reduce_over_patches(err, w)
        uniform = WorldModelWrapper._reduce_over_patches(err, None)
        self.assertTrue((weighted > uniform).all())

    def test_constant_error_scale_preserved(self):
        # A constant per-patch error reduces to that constant under BOTH the
        # uniform and the weighted reducer -> the weighting does not rescale the
        # loss (so latent_pred_weight needs no retuning).
        s_anchor, s_tgt, tgt_valid = self._tgt()
        w, _ = WorldModelWrapper._motion_weight(
            s_anchor, s_tgt, tgt_valid, alpha=1.0, floor=0.1)
        err = torch.full((2, 2, 3), 0.7)
        self.assertTrue(torch.allclose(
            WorldModelWrapper._reduce_over_patches(err, w),
            torch.full((2, 2), 0.7), atol=1e-6))

    def test_all_static_no_nan(self):
        # motion == 0 everywhere: per-step ref==0, weights fall back to the floor,
        # reducer is finite and equals the uniform mean (nothing to concentrate on).
        s_anchor, s_tgt, tgt_valid = self._tgt(moving_patch_delta=0.0)
        w, ref = WorldModelWrapper._motion_weight(
            s_anchor, s_tgt, tgt_valid, alpha=1.0, floor=0.1)
        self.assertTrue(torch.isfinite(w).all())
        self.assertTrue((ref == 0).all())
        err = torch.randn(2, 2, 3)
        red = WorldModelWrapper._reduce_over_patches(err, w)
        self.assertTrue(torch.isfinite(red).all())
        self.assertTrue(torch.allclose(red, err.mean(dim=-1), atol=1e-5))

    def test_invalid_steps_excluded_from_reference(self):
        # An invalid (row, step) whose target is garbage must not skew the motion
        # reference for the OTHER steps (per-batch per-step ref).
        s_anchor, s_tgt, tgt_valid = self._tgt(moving_patch_delta=1.0)
        s_tgt[:, 1] += 100.0                       # huge garbage on step 1
        tgt_valid[:, 1] = 0.0                      # ...but step 1 is invalid
        w_masked, ref_masked = WorldModelWrapper._motion_weight(
            s_anchor, s_tgt, tgt_valid, alpha=1.0, floor=0.1)
        # step-0 ref computed only over the valid step 0 -> unaffected by garbage.
        s_a2, s_t2, v2 = self._tgt(moving_patch_delta=1.0)
        _, ref_clean = WorldModelWrapper._motion_weight(
            s_a2, s_t2[:, :1], v2[:, :1], alpha=1.0, floor=0.1)
        self.assertTrue(torch.allclose(ref_masked[:1], ref_clean, atol=1e-5))
        self.assertEqual(ref_masked[1].item(), 0.0)   # invalid step -> ref 0

    def test_fixed_per_step_ref_is_batch_independent(self):
        # A supplied per-step reference is used verbatim, regardless of the
        # batch's own motion — the whole point of hard-coding it.
        ref_vec = (0.5, 0.8)                        # H=2
        sa, st, tv = self._tgt(moving_patch_delta=2.0)
        _, ref_a = WorldModelWrapper._motion_weight(
            sa, st, tv, alpha=1.0, floor=0.1, ref=ref_vec)
        sa2, st2, tv2 = self._tgt(moving_patch_delta=0.3)   # different scale
        _, ref_b = WorldModelWrapper._motion_weight(
            sa2, st2, tv2, alpha=1.0, floor=0.1, ref=ref_vec)
        self.assertTrue(torch.allclose(ref_a, torch.tensor([0.5, 0.8])))
        self.assertTrue(torch.allclose(ref_b, torch.tensor([0.5, 0.8])))

    def test_scalar_ref_broadcasts_to_all_steps(self):
        sa, st, tv = self._tgt(moving_patch_delta=1.0)
        _, ref = WorldModelWrapper._motion_weight(
            sa, st, tv, alpha=1.0, floor=0.1, ref=0.5)
        self.assertTrue(torch.allclose(ref, torch.tensor([0.5, 0.5])))

    def test_ref_shorter_than_horizon_asserts(self):
        sa, st, tv = self._tgt(moving_patch_delta=1.0)     # H=2
        with self.assertRaises(AssertionError):
            WorldModelWrapper._motion_weight(
                sa, st, tv, alpha=1.0, floor=0.1, ref=(0.5,))   # only 1 < H=2

    def test_resolved_motion_ref(self):
        enc = _tiny_encoder()
        neg = WorldModelWrapper(encoder=enc, predictor=None, max_horizon=0,
                                objective='frame', frame_motion_ref=-1.0)
        self.assertEqual(neg._resolved_motion_ref(),
                         EGOBRAIN_FRAME_MOTION_REF_PER_STEP)
        zero = WorldModelWrapper(encoder=enc, predictor=None, max_horizon=0,
                                 objective='frame', frame_motion_ref=0.0)
        self.assertIsNone(zero._resolved_motion_ref())
        pos = WorldModelWrapper(encoder=enc, predictor=None, max_horizon=0,
                                objective='frame', frame_motion_ref=2.5)
        self.assertEqual(pos._resolved_motion_ref(), 2.5)

    def test_wrapper_validates_params(self):
        enc = _tiny_encoder()
        with self.assertRaises(AssertionError):
            WorldModelWrapper(encoder=enc, predictor=None, max_horizon=0,
                              objective='frame', frame_motion_alpha=-1.0)
        with self.assertRaises(AssertionError):
            WorldModelWrapper(encoder=enc, predictor=None, max_horizon=0,
                              objective='frame', frame_motion_floor=1.5)


class TestFrameCleanCond(unittest.TestCase):
    """``frame_clean_cond`` two-view split: the masked view is recon-only; the
    clean (unmasked) view carries prediction/alignment/hand. Uses synthetic cached
    grids so no vision weights are needed."""

    def _wrapper(self, clean_cond):
        in_dim, n_ch, n_patches, fdim, H = 40, 8, 4, 12, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        pred = FramePredictor(frame_dim=fdim, eeg_dim=in_dim, predictor_d_model=48,
                              n_layers=2, n_heads=4, dim_feedforward=96, max_horizon=H)
        w = WorldModelWrapper(
            encoder=enc, predictor=pred, latent_pred_weight=1.0, cls_pred_weight=0.0,
            max_horizon=H, ramp_epochs=0, objective='frame', frame_eeg_cond='tokens',
            frame_clean_cond=clean_cond)
        w.train()
        return w, n_ch, n_patches, fdim, H

    def _batch(self, B, n_ch, n_patches, fdim, H):
        torch.manual_seed(3)
        W = H + 1
        grid_f = torch.randn(B, W, 9, fdim)
        return {
            'timeseries': torch.randn(B, n_ch, n_patches, 40) / 100.0,
            'ch_coords': torch.randn(B, n_ch, 3).abs() + 0.1,
            'ch_names': [['pad'] * n_ch for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'has_image': torch.zeros(B, dtype=torch.bool),
            'timeseries_future': torch.randn(B, W, n_ch, n_patches, 40),
            'pixel_values_future': torch.zeros(B, W, 1, 1, 1),
            'has_image_future': torch.ones(B, W, dtype=torch.bool),
            'frame_grid_future': grid_f,
            'frame_grid_flip_future': grid_f.clone(),
            'source': ['egobrain'] * B,
        }

    def _spy_forwards(self, w, batch, mask):
        # Record (mask_is_None, 'has_image' in batch) at each encoder forward.
        orig = w.encoder.forward
        calls = []

        def spy(fwd_batch, *a, **k):
            calls.append((k.get('mask', a[0] if a else None) is None,
                          'has_image' in fwd_batch))
            return orig(fwd_batch, *a, **k)

        w.encoder.forward = spy
        try:
            _, info = w.training_step(batch, mask=mask)
        finally:
            w.encoder.forward = orig
        return calls, info

    def test_two_view_split_structure(self):
        B = 4
        wf, n_ch, n_patches, fdim, H = self._wrapper(clean_cond=False)
        wt, *_ = self._wrapper(clean_cond=True)
        batch = self._batch(B, n_ch, n_patches, fdim, H)
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1                                   # mask ~half the patches

        calls_off, info_off = self._spy_forwards(wf, batch, mask)
        calls_on, info_on = self._spy_forwards(wt, batch, mask)

        # Legacy: a single masked forward carrying the downstream inputs.
        self.assertEqual(len(calls_off), 1)
        self.assertEqual(calls_off[0], (False, True))       # mask set, has_image kept

        # Two-view: one extra forward. Exactly one unmasked (clean) pass, and the
        # MASKED (recon) pass has the downstream inputs stripped (no has_image).
        self.assertEqual(len(calls_on), 2)
        self.assertEqual(sum(1 for is_none, _ in calls_on if is_none), 1)   # 1 clean
        masked = [c for c in calls_on if not c[0]]
        clean = [c for c in calls_on if c[0]]
        self.assertEqual(len(masked), 1)
        self.assertFalse(masked[0][1])                      # recon batch stripped
        self.assertTrue(clean[0][1])                        # clean batch keeps it

        # frame_pred_loss present & finite both ways, and differs (masked vs clean
        # conditioning under frame_eeg_cond='tokens' with patches masked).
        for info in (info_off, info_on):
            self.assertIn('frame_pred_loss', info)
            self.assertTrue(torch.isfinite(info['frame_pred_loss'][1]).item())
        self.assertFalse(torch.allclose(
            info_off['frame_pred_loss'][1], info_on['frame_pred_loss'][1]))

    def test_two_view_out_is_masked_view(self):
        # The returned ``out`` (which drives the trainer's mask_loss) must be the
        # MASKED forward's reconstruction, not the clean one.
        B = 4
        w, n_ch, n_patches, fdim, H = self._wrapper(clean_cond=True)
        batch = self._batch(B, n_ch, n_patches, fdim, H)
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = w.training_step(batch, mask=mask)
        self.assertEqual(out.shape, batch['timeseries'].shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_two_view_backward(self):
        B = 4
        w, n_ch, n_patches, fdim, H = self._wrapper(clean_cond=True)
        batch = self._batch(B, n_ch, n_patches, fdim, H)
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = w.training_step(batch, mask=mask)
        loss = sum(v[0] * v[1] for k, v in info.items()
                   if isinstance(v, tuple) and 'loss' in k)
        # include a recon term on the masked out so both views get gradient
        loss = loss + (out[mask == 1] ** 2).mean()
        loss.backward()
        g = [p.grad for p in w.encoder.patch_embedding.parameters()
             if p.grad is not None]
        self.assertTrue(len(g) > 0 and any(torch.isfinite(x).all() for x in g))


class TestFrameContrast(unittest.TestCase):
    """Negative-EEG contrastive term (``WorldModelWrapper._add_frame_contrast``).

    Re-runs the frame predictor on the same anchor grid with OTHER rows' EEG and
    penalises reproducing the true future from the wrong EEG. Uses synthetic
    cached grids so no vision weights are needed.
    """

    def _wrapper(self, weight, mode='infonce', n_neg=2, detach_neg=True,
                 motion_alpha=0.0):
        in_dim, n_ch, n_patches, fdim, H = 40, 8, 4, 12, 2
        enc = _tiny_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        pred = FramePredictor(frame_dim=fdim, eeg_dim=in_dim, predictor_d_model=48,
                              n_layers=2, n_heads=4, dim_feedforward=96, max_horizon=H)
        w = WorldModelWrapper(
            encoder=enc, predictor=pred, latent_pred_weight=1.0, cls_pred_weight=0.0,
            max_horizon=H, ramp_epochs=0, objective='frame', frame_eeg_cond='tokens',
            frame_motion_alpha=motion_alpha,
            frame_contrast_weight=weight, frame_contrast_mode=mode,
            frame_contrast_n_neg=n_neg, frame_contrast_detach_neg=detach_neg)
        w.train()
        return w, n_ch, n_patches, fdim, H

    def _batch(self, B, n_ch, n_patches, fdim, H):
        torch.manual_seed(5)
        W = H + 1
        grid_f = torch.randn(B, W, 9, fdim)
        return {
            'timeseries': torch.randn(B, n_ch, n_patches, 40) / 100.0,
            'ch_coords': torch.randn(B, n_ch, 3).abs() + 0.1,
            'ch_names': [['pad'] * n_ch for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'has_image': torch.zeros(B, dtype=torch.bool),
            'timeseries_future': torch.randn(B, W, n_ch, n_patches, 40),
            'pixel_values_future': torch.zeros(B, W, 1, 1, 1),
            'has_image_future': torch.ones(B, W, dtype=torch.bool),
            'frame_grid_future': grid_f,
            'frame_grid_flip_future': grid_f.clone(),
            'source': ['egobrain'] * B,
        }

    def _run(self, w, B, n_ch, n_patches, fdim, H):
        batch = self._batch(B, n_ch, n_patches, fdim, H)
        mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
        mask[:, :, 0] = 1
        return w.training_step(batch, mask=mask), batch, mask

    def test_off_by_default_no_key(self):
        # weight=0 -> the contrastive block (and its extra forwards) is skipped.
        w, n_ch, n_patches, fdim, H = self._wrapper(weight=0.0)
        (out, info), *_ = self._run(w, 4, n_ch, n_patches, fdim, H)
        self.assertIn('frame_pred_loss', info)
        self.assertNotIn('frame_contrast_loss', info)
        self.assertNotIn('diag_frame_contrast_acc', info)

    def _check_on(self, mode, detach_neg, motion_alpha):
        w, n_ch, n_patches, fdim, H = self._wrapper(
            weight=1.0, mode=mode, n_neg=2, detach_neg=detach_neg,
            motion_alpha=motion_alpha)
        (out, info), batch, mask = self._run(w, 5, n_ch, n_patches, fdim, H)
        self.assertIn('frame_contrast_loss', info)
        coef, val = info['frame_contrast_loss']
        self.assertGreater(coef, 0.0)
        self.assertTrue(torch.isfinite(val).item())
        self.assertGreaterEqual(val.item(), 0.0)
        # Diagnostics present and in range.
        acc = info['diag_frame_contrast_acc']
        self.assertTrue(0.0 <= acc.item() <= 1.0)
        self.assertTrue(torch.isfinite(info['diag_frame_contrast_gap']).item())
        self.assertEqual(info['diag_frame_contrast_n_neg'].item(), 2.0)

        # Backward reaches the predictor (always) and the EEG backbone (via the
        # positive term, even when negatives are detached).
        loss = sum(v[0] * v[1] for k, v in info.items()
                   if isinstance(v, tuple) and 'loss' in k)
        loss = loss + (out[mask == 1] ** 2).mean()
        loss.backward()
        pred_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in w.predictor.parameters())
        self.assertTrue(pred_grad, 'predictor received no gradient')
        enc_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in w.encoder.patch_embedding.parameters())
        self.assertTrue(enc_grad, 'EEG backbone received no gradient')

    def test_infonce_detached(self):
        self._check_on(mode='infonce', detach_neg=True, motion_alpha=0.0)

    def test_infonce_grad_neg_with_motion(self):
        self._check_on(mode='infonce', detach_neg=False, motion_alpha=1.0)

    def test_margin_mode(self):
        self._check_on(mode='margin', detach_neg=True, motion_alpha=0.0)

    def test_single_row_skips_contrast(self):
        # With one usable row there is no negative to form -> no contrast term,
        # and the rest of the frame objective still runs.
        w, n_ch, n_patches, fdim, H = self._wrapper(weight=1.0, n_neg=4)
        (out, info), *_ = self._run(w, 1, n_ch, n_patches, fdim, H)
        self.assertIn('frame_pred_loss', info)
        self.assertNotIn('frame_contrast_loss', info)

    def test_n_neg_capped_to_batch(self):
        # n_neg larger than (valid_rows - 1) is capped, not an error.
        w, n_ch, n_patches, fdim, H = self._wrapper(weight=1.0, n_neg=99)
        (out, info), *_ = self._run(w, 3, n_ch, n_patches, fdim, H)
        self.assertIn('frame_contrast_loss', info)
        self.assertEqual(info['diag_frame_contrast_n_neg'].item(), 2.0)  # 3-1

    def test_detach_neg_controls_negative_grad_path(self):
        # The contrastive term ALWAYS reaches the encoder via the positive d_pos;
        # ``detach_neg`` only controls whether the NEGATIVE predictor calls also
        # feed the encoder. Spy on the predictor's EEG argument: the first call is
        # the positive (always requires_grad), calls 1..K are the negatives, whose
        # requires_grad must equal ``not detach_neg``.
        for detach in (True, False):
            w, n_ch, n_patches, fdim, H = self._wrapper(
                weight=1.0, mode='infonce', n_neg=2, detach_neg=detach)
            orig = w.predictor.forward
            seen = []

            def spy(s_anchor, eeg_emb, *a, **k):
                seen.append(bool(eeg_emb.requires_grad))
                return orig(s_anchor, eeg_emb, *a, **k)

            w.predictor.forward = spy
            try:
                self._run(w, 5, n_ch, n_patches, fdim, H)
            finally:
                w.predictor.forward = orig
            self.assertTrue(seen[0], 'positive predictor call must carry grad')
            negatives = seen[1:3]  # K=2 negative calls follow the positive
            self.assertEqual(
                negatives, [not detach, not detach],
                f'detach_neg={detach}: negative eeg requires_grad={negatives}')

    def test_same_flip_neg_indices(self):
        W = WorldModelWrapper
        # All same flip group (frame averaging off): every row gets K distinct
        # valid negatives, none of which is the row itself.
        flip = torch.zeros(5, dtype=torch.bool)
        idx, val = W._same_flip_neg_indices(flip, K=2)
        self.assertEqual(idx.shape, (5, 2))
        self.assertTrue(val.all())
        for i in range(5):
            for m in range(2):
                self.assertNotEqual(idx[i, m].item(), i)
        # Two balanced groups of 3: negatives stay within the same flip group, and
        # a group of size 3 yields only 2 valid negatives (slot 3 is masked).
        flip = torch.tensor([False, True, False, True, False, True])
        idx, val = W._same_flip_neg_indices(flip, K=3)
        for i in range(6):
            for m in range(3):
                if val[i, m]:
                    self.assertEqual(flip[idx[i, m]].item(), flip[i].item())
        self.assertTrue((val.sum(dim=1) == 2).all())
        # Singleton group -> that row has no valid negative.
        flip = torch.tensor([False, False, False, True])
        idx, val = W._same_flip_neg_indices(flip, K=2)
        self.assertFalse(val[3].any())
        self.assertTrue(val[:3].any(dim=1).all())

    def test_neg_indices_backcompat_none(self):
        # block_id=None + anchor=None must reproduce the pure same-flip result.
        W = WorldModelWrapper
        flip = torch.tensor([False, True, False, True, False, True])
        idx0, val0 = W._same_flip_neg_indices(flip, K=3)
        idx1, val1 = W._same_flip_neg_indices(flip, K=3, block_id=None,
                                              anchor=None, excl_samples=0)
        self.assertTrue(torch.equal(idx0, idx1))
        self.assertTrue(torch.equal(val0, val1))

    def test_neg_indices_block_gating(self):
        # Two blocks of 4 (all same flip): every valid negative shares the block,
        # and a same-flip block of 4 yields exactly 3 valid negatives per row.
        W = WorldModelWrapper
        flip = torch.zeros(8, dtype=torch.bool)
        block = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        idx, val = W._same_flip_neg_indices(flip, K=3, block_id=block)
        for i in range(8):
            for m in range(3):
                if val[i, m]:
                    self.assertEqual(block[idx[i, m]].item(), block[i].item())
        self.assertTrue((val.sum(dim=1) == 3).all())

    def test_neg_indices_block_and_flip_jointly(self):
        # One block split 2/2 by flip -> each row has exactly one same-(flip,block)
        # partner; negatives must match BOTH keys.
        W = WorldModelWrapper
        flip = torch.tensor([False, True, False, True])
        block = torch.zeros(4, dtype=torch.long)
        idx, val = W._same_flip_neg_indices(flip, K=3, block_id=block)
        for i in range(4):
            for m in range(3):
                if val[i, m]:
                    self.assertEqual(flip[idx[i, m]].item(), flip[i].item())
                    self.assertEqual(block[idx[i, m]].item(), block[i].item())
        self.assertTrue((val.sum(dim=1) == 1).all())

    def test_neg_indices_temporal_exclusion(self):
        # A candidate within excl_samples of the anchor is masked out.
        W = WorldModelWrapper
        flip = torch.zeros(4, dtype=torch.bool)
        block = torch.zeros(4, dtype=torch.long)
        anchor = torch.tensor([0, 10, 5000, 9000], dtype=torch.long)
        idx, val = W._same_flip_neg_indices(
            flip, K=3, block_id=block, anchor=anchor, excl_samples=100)
        for i in range(4):
            for m in range(3):
                if val[i, m]:
                    self.assertGreaterEqual(
                        abs(int(anchor[i]) - int(anchor[idx[i, m]])), 100)
        # Rows 0 and 1 are 10 samples apart -> each loses that one candidate.
        self.assertLessEqual(int(val[0].sum()), 2)
        self.assertLessEqual(int(val[1].sum()), 2)

    def test_block_gating_confines_negatives_in_loss(self):
        # End-to-end through _add_frame_contrast: 2 blocks of 3, negatives never
        # cross a block, so the realised per-row negative count is 2 (block-1).
        Bv, H, P, d = 6, 2, 3, 4
        w, *_ = self._wrapper(weight=1.0, mode='infonce', n_neg=5)
        w.predictor = self._CopyPred(H)
        w.frame_contrast_excl_samples = 0
        torch.manual_seed(7)
        s_anchor = torch.randn(Bv, P, d)
        s_tgt = torch.randn(Bv, H, P, d)
        eeg_emb = torch.randn(Bv, d)
        tgt_valid = torch.ones(Bv, H)
        copy = s_anchor.unsqueeze(1).expand(-1, H, -1, -1)
        err = torch.nn.functional.l1_loss(copy, s_tgt, reduction='none').mean(-1)
        per_step_pos = WorldModelWrapper._reduce_over_patches(err, None)
        info = {}
        w._add_frame_contrast(
            info, per_step_pos, s_anchor, s_tgt, eeg_emb, None, None, tgt_valid,
            scale=1.0, flip_valid=torch.zeros(Bv, dtype=torch.bool),
            block_id=torch.tensor([0, 0, 0, 1, 1, 1]))
        self.assertIn('frame_contrast_loss', info)
        self.assertAlmostEqual(
            float(info['diag_frame_contrast_n_neg']), 2.0, places=5)

    class _CopyPred(torch.nn.Module):
        """Predictor stub that IGNORES the EEG and copies the anchor grid across
        all H horizon steps — the copy-the-anchor failure mode the term targets."""
        def __init__(self, H):
            super().__init__()
            self.H = H

        def forward(self, s_anchor, eeg_emb, eeg_key_padding_mask=None):
            return s_anchor.unsqueeze(1).expand(-1, self.H, -1, -1).clone()

    def _copy_case(self, mode, flip_valid, n_neg):
        Bv, H, P, d = flip_valid.numel(), 2, 3, 4
        w, *_ = self._wrapper(weight=1.0, mode=mode, n_neg=n_neg)
        w.predictor = self._CopyPred(H)
        torch.manual_seed(4)
        s_anchor = torch.randn(Bv, P, d)
        s_tgt = torch.randn(Bv, H, P, d)
        eeg_emb = torch.randn(Bv, d)  # ignored by the stub
        tgt_valid = torch.ones(Bv, H)
        copy = s_anchor.unsqueeze(1).expand(-1, H, -1, -1)
        err = torch.nn.functional.l1_loss(copy, s_tgt, reduction='none').mean(-1)
        per_step_pos = WorldModelWrapper._reduce_over_patches(err, None)
        info = {}
        w._add_frame_contrast(info, per_step_pos, s_anchor, s_tgt, eeg_emb, None,
                              None, tgt_valid, scale=1.0, flip_valid=flip_valid)
        return info

    def test_strict_accuracy_reports_chance_when_predictor_ignores_eeg(self):
        # Copy-the-anchor => d_pos == every d_neg. Strict '<' must count that as
        # INCORRECT (acc 0.0), NOT the misleading 1.0 an argmin tie-break gives.
        info = self._copy_case('infonce', torch.zeros(6, dtype=torch.bool), n_neg=3)
        self.assertIn('frame_contrast_loss', info)
        self.assertEqual(info['diag_frame_contrast_acc'].item(), 0.0)

    def test_flip_valid_excludes_singleton_group(self):
        # 4 rows in one flip group + 1 lone row of the other flip. The lone row
        # has no same-flip negative, so it is dropped from the reduction (n_rows=4)
        # and the loss stays finite.
        flip_valid = torch.tensor([False, False, False, False, True])
        info = self._copy_case('margin', flip_valid, n_neg=3)
        self.assertEqual(info['diag_frame_contrast_n_rows'].item(), 4.0)
        self.assertTrue(torch.isfinite(info['frame_contrast_loss'][1]).item())
        # Realised negatives per usable row = min(n_neg, group_size-1) = min(3,3)=3.
        self.assertEqual(info['diag_frame_contrast_n_neg'].item(), 3.0)

    def _equiv_case(self, cond, flip_valid, n_neg):
        # Real predictor in eval() (no dropout) => the loop and batched negative
        # forwards must produce identical distances. Only d_neg differs between the
        # two paths (d_pos is the same shared per_step_pos), so this isolates the
        # batched _frame_neg_distances against the reference loop.
        Bv = flip_valid.numel()
        H, P, fdim, edim, M = 2, 4, 12, 40, 5
        w, *_ = self._wrapper(weight=1.0, mode='infonce', n_neg=n_neg)
        w.eval()
        torch.manual_seed(7)
        s_anchor = torch.randn(Bv, P, fdim)
        s_tgt = torch.randn(Bv, H, P, fdim)
        if cond == 'global':
            eeg, kpm = torch.randn(Bv, edim), None
        else:
            eeg = torch.randn(Bv, M, edim)
            kpm = torch.zeros(Bv, M, dtype=torch.bool)
            kpm[:, -1] = True                       # a padded token per row
        tv = torch.ones(Bv, H)
        with torch.no_grad():
            pos = w.predictor(s_anchor, eeg, eeg_key_padding_mask=kpm)
            err = torch.nn.functional.l1_loss(pos, s_tgt, reduction='none').mean(-1)
            per_step_pos = WorldModelWrapper._reduce_over_patches(err, None)
        outs = {}
        for batched in (False, True):
            w.frame_contrast_batched = batched
            info = {}
            with torch.no_grad():
                w._add_frame_contrast(info, per_step_pos, s_anchor, s_tgt, eeg,
                                      kpm, None, tv, 1.0, flip_valid)
            outs[batched] = info
        return outs[False], outs[True]

    def test_batched_matches_loop(self):
        # Balanced flip split so same-flip selection + masked slots are exercised
        # in both paths, for global and token conditioning.
        for cond in ('global', 'tokens'):
            flip = torch.tensor([False, True, False, True, False, True])
            loop, batch = self._equiv_case(cond, flip, n_neg=4)
            self.assertTrue(torch.allclose(
                loop['frame_contrast_loss'][1], batch['frame_contrast_loss'][1],
                atol=1e-5), f'{cond}: contrast loss loop vs batched differ')
            self.assertTrue(torch.allclose(
                loop['diag_frame_contrast_gap'], batch['diag_frame_contrast_gap'],
                atol=1e-5), f'{cond}: gap differs')
            self.assertEqual(loop['diag_frame_contrast_acc'].item(),
                             batch['diag_frame_contrast_acc'].item())
            self.assertEqual(loop['diag_frame_contrast_n_neg'].item(),
                             batch['diag_frame_contrast_n_neg'].item())


class TestSubjectBlockBatchSampler(unittest.TestCase):
    """``datasets.egobrain_dataset.SubjectBlockBatchSampler`` — batches laid down
    as contiguous same-subject blocks for in-batch same-block negatives + I/O
    locality."""

    def _sampler(self, **kw):
        from datasets.egobrain_dataset import SubjectBlockBatchSampler
        return SubjectBlockBatchSampler(**kw)

    def _items(self):
        # A: 10 clips, B: 6, C: 3 (C is smaller than block_size on purpose).
        return ([('A', c) for c in range(10)]
                + [('B', c) for c in range(6)]
                + [('C', c) for c in range(3)])

    def test_block_structure(self):
        items = self._items()
        s = self._sampler(items=items, batch_size=12, block_size=4,
                          num_batches=5, seed=0)
        self.assertEqual(len(s), 5)
        batches = list(s)
        self.assertEqual(len(batches), 5)
        for b in batches:
            self.assertEqual(len(b), 12)
            for bl in range(3):                         # 3 blocks of 4
                block = b[bl * 4:(bl + 1) * 4]
                subs = {items[i][0] for i in block}
                self.assertEqual(len(subs), 1, 'each block must be one subject')

    def test_distinct_clips_when_subject_large_enough(self):
        # A block from a subject with >= block_size clips draws DISTINCT clips.
        s = self._sampler(items=[('A', c) for c in range(10)],
                          batch_size=4, block_size=4, num_batches=20, seed=1)
        for b in list(s):
            self.assertEqual(len(set(b)), 4)

    def test_small_subject_falls_back_to_replacement(self):
        # C has 3 clips < block_size=4 -> the block still fills (with replacement)
        # and stays a single subject.
        s = self._sampler(items=[('C', c) for c in range(3)],
                          batch_size=4, block_size=4, num_batches=5, seed=2)
        for b in list(s):
            self.assertEqual(len(b), 4)

    def test_divisibility_guard(self):
        with self.assertRaises(ValueError):
            self._sampler(items=[('A', 0)], batch_size=10, block_size=4,
                          num_batches=1)

    def test_epoch_reseed_varies(self):
        items = [('A', c) for c in range(10)] + [('B', c) for c in range(10)]
        s = self._sampler(items=items, batch_size=8, block_size=4,
                          num_batches=3, seed=0)
        self.assertNotEqual(list(s), list(s), 'consecutive epochs should differ')

    def test_fixed_seed_reproducible(self):
        items = self._items()
        a = list(self._sampler(items=items, batch_size=8, block_size=4,
                               num_batches=3, seed=0))
        b = list(self._sampler(items=items, batch_size=8, block_size=4,
                               num_batches=3, seed=0))
        self.assertEqual(a, b, 'same seed -> same first epoch')


if __name__ == '__main__':
    unittest.main(verbosity=2)
