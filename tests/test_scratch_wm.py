"""Smoke tests for the LeWM from-scratch, jointly-trained world model.

Covers the pieces added in plans/mighty-dazzling-lovelace.md:
1. ``SIGReg`` separates an isotropic-Gaussian batch from a collapsed one.
2. ``build_scratch_vit`` produces a trainable ViT with the expected shapes, and
   the gradient-carrying encode methods (``_scratch_patch_grid`` / ``_scratch_cls``)
   actually flow gradient into the ViT.
3. Wrapper FRAME path (``wm_predictor='frame'``, scratch ViT): emits
   ``frame_pred_loss`` + ``sigreg_loss``; the target is NOT detached (faithful
   LeWM) so backward reaches both the ViT and the EEG backbone.
4. Wrapper AR path (``wm_predictor='ar'``): emits ``ar_pred_loss`` + ``sigreg_loss``
   + ``diag_ar_eeg_gap``; backward reaches the ViT and the EEG backbone.
5. Regression: with ``scratch_vit`` off the wrapper is unchanged (no SIGReg,
   frozen vision path).

A small-depth ViT (image 224) is used so the real forward runs on CPU in
seconds while still exercising genuine gradient flow.
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
from models.world_model import (WorldModelWrapper, FramePredictor, ARWorldModel,
                                 FrameAdaLNPredictor)
from models.lewm_modules import SIGReg
from models.scratch_vit import build_scratch_vit
from datasets.cinebrain_dataset import _BIOSEMI64_COORDS


def _scratch_encoder(in_dim=40, n_ch=8, n_patches=2, depth=2):
    """Tiny CSBrainAlign whose vision encoder is a from-scratch trainable ViT."""
    return CSBrainAlign(
        in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
        dim_feedforward=4 * in_dim, seq_len=n_patches,
        n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
        brain_regions=None, sorted_indices=[], causal=False,
        alignment_weight=0.0, equivariance_weight=0.0, patch_embed_type='cnn',
        scratch_vit=True, scratch_vit_depth=depth, scratch_vit_heads=3,
    )


def _coords(B, n_ch):
    return torch.from_numpy(
        _BIOSEMI64_COORDS[:n_ch]).unsqueeze(0).expand(B, -1, -1).contiguous()


def _frame_batch(B, n_ch, n_patches, in_dim, W):
    """Pure-egobrain batch with REAL 224² random pixels (so the ViT sees signal)."""
    ts = torch.randn(B, W, n_ch, n_patches, in_dim)
    pv = torch.randn(B, W, 3, 224, 224)
    has_future = torch.ones(B, W, dtype=torch.bool)
    mask = torch.zeros(B, n_ch, n_patches, dtype=torch.long)
    mask[:, :, 0] = 1
    return {
        'timeseries': ts[:, 0] / 100.0,
        'timeseries_future': ts,
        'pixel_values_future': pv,
        'has_image_future': has_future,
        'ch_coords': _coords(B, n_ch),
        'ch_names': [['pad'] * n_ch for _ in range(B)],
        'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
        'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
        'image_encoder_inputs': {'pixel_values': pv[:, 0]},
        'has_image': torch.zeros(B, dtype=torch.bool),
        'source': ['egobrain'] * B,
        '_mask': mask,
    }


class TestSIGReg(unittest.TestCase):
    def test_separates_gaussian_from_collapse(self):
        torch.manual_seed(0)
        sig = SIGReg(knots=17, num_proj=256)
        gaussian = torch.randn(4, 128, 32)          # (T, B, D) isotropic
        collapsed = torch.ones(4, 128, 32) * 0.3     # constant -> collapsed
        g = float(sig(gaussian))
        c = float(sig(collapsed))
        self.assertTrue(torch.isfinite(torch.tensor(g)))
        self.assertGreater(c, 5.0 * max(g, 1e-6),
                           f"SIGReg failed to flag collapse (gaussian={g}, collapsed={c})")

    def test_grad_flows(self):
        sig = SIGReg(knots=9, num_proj=64)
        x = torch.randn(3, 64, 16, requires_grad=True)
        sig(x).backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


class TestScratchViT(unittest.TestCase):
    def test_build_shapes_and_trainable(self):
        vit = build_scratch_vit(size='tiny', patch_size=16, image_size=224)
        self.assertEqual(vit.config.hidden_size, 192)
        self.assertEqual(vit.config.model_type, 'vit')
        self.assertTrue(all(p.requires_grad for p in vit.parameters()))
        hs = vit(pixel_values=torch.randn(2, 3, 224, 224)).last_hidden_state
        self.assertEqual(hs.shape, (2, 197, 192))     # CLS + 14x14 patches

    def test_grad_capable_encode_methods(self):
        enc = _scratch_encoder()
        self.assertTrue(enc.vision_trainable)
        self.assertEqual(enc.image_feature_dim, 192)
        pv = torch.randn(2, 3, 224, 224)
        grid = enc._scratch_patch_grid(pv)            # (2, 14, 14, 192)
        cls = enc._scratch_cls(pv)                    # (2, 1, 192)
        self.assertEqual(grid.shape, (2, 14, 14, 192))
        self.assertEqual(cls.shape, (2, 1, 192))
        self.assertTrue(grid.requires_grad)
        self.assertTrue(cls.requires_grad)
        # Gradient reaches the ViT.
        (grid.mean() + cls.mean()).backward()
        vit_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in enc.pretrained_image_encoder.parameters())
        self.assertTrue(vit_grad, "trainable ViT received no gradient")


class TestScratchFramePath(unittest.TestCase):
    def _wrapper(self, sigreg_weight=0.09):
        in_dim, n_ch, n_patches, W = 40, 8, 2, 3
        enc = _scratch_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        H = W - 1
        pred = FramePredictor(frame_dim=enc.image_feature_dim, eeg_dim=enc.d_model,
                              predictor_d_model=48, n_layers=2, n_heads=4,
                              dim_feedforward=96, max_horizon=H)
        w = WorldModelWrapper(
            encoder=enc, predictor=pred, latent_pred_weight=1.0, max_horizon=H,
            ramp_epochs=0, objective='frame', frame_eeg_cond='global',
            scratch_vit=True, sigreg_weight=sigreg_weight, wm_predictor='frame')
        w.train()
        return w, enc, n_ch, n_patches, in_dim, W

    def test_frame_path_losses_and_grad(self):
        w, enc, n_ch, n_patches, in_dim, W = self._wrapper()
        # No EMA target (frozen/faithful target, not EMA).
        self.assertIsNone(w.target_encoder)
        batch = _frame_batch(3, n_ch, n_patches, in_dim, W)
        mask = batch.pop('_mask')
        out, info = w.training_step(batch, mask=mask)
        self.assertIn('frame_pred_loss', info)
        self.assertIn('sigreg_loss', info)
        self.assertTrue(torch.isfinite(info['frame_pred_loss'][1]).item())
        self.assertTrue(torch.isfinite(info['sigreg_loss'][1]).item())

        loss_terms = [v[0] * v[1] for k, v in info.items()
                      if isinstance(v, tuple) and 'loss' in k]
        mask_loss = (out[mask == 1] - batch['timeseries'][mask == 1]).pow(2).mean()
        (mask_loss + sum(loss_terms)).backward()
        # Faithful target (not detached) + SIGReg => the ViT trains.
        vit_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in enc.pretrained_image_encoder.parameters())
        self.assertTrue(vit_grad, "trainable ViT received no gradient (frame path)")
        # EEG backbone trains via the frame-prediction conditioning.
        eeg_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in enc.patch_embedding.parameters())
        self.assertTrue(eeg_grad, "EEG backbone received no gradient (frame path)")

    def test_sigreg_alone_reaches_vit(self):
        # SIGReg is computed on the ViT's own grid embeddings, so its gradient
        # alone must reach the ViT (proves the target embeddings are not detached).
        w, enc, n_ch, n_patches, in_dim, W = self._wrapper()
        batch = _frame_batch(3, n_ch, n_patches, in_dim, W)
        batch.pop('_mask')
        _, info = w.training_step(batch, mask=None)
        info['sigreg_loss'][1].backward()
        vit_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in enc.pretrained_image_encoder.parameters())
        self.assertTrue(vit_grad, "SIGReg gradient did not reach the ViT")


class TestScratchARPath(unittest.TestCase):
    def _wrapper(self, history=3, num_preds=1):
        in_dim, n_ch, n_patches = 40, 8, 2
        enc = _scratch_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        pred = ARWorldModel(
            frame_dim=enc.image_feature_dim, eeg_dim=enc.d_model,
            embed_dim=enc.image_feature_dim, history=history, num_preds=num_preds,
            depth=2, heads=4, mlp_dim=256)
        w = WorldModelWrapper(
            encoder=enc, predictor=pred, latent_pred_weight=1.0, max_horizon=4,
            ramp_epochs=0, objective='frame', scratch_vit=True,
            sigreg_weight=0.09, wm_predictor='ar')
        w.train()
        return w, enc, n_ch, n_patches, in_dim

    def test_ar_path_losses_and_grad(self):
        w, enc, n_ch, n_patches, in_dim = self._wrapper(history=3, num_preds=1)
        W = 5  # >= history + num_preds
        batch = _frame_batch(3, n_ch, n_patches, in_dim, W)
        mask = batch.pop('_mask')
        out, info = w.training_step(batch, mask=mask)
        self.assertIn('ar_pred_loss', info)
        self.assertIn('sigreg_loss', info)
        self.assertIn('diag_ar_eeg_gap', info)
        self.assertTrue(torch.isfinite(info['ar_pred_loss'][1]).item())

        loss_terms = [v[0] * v[1] for k, v in info.items()
                      if isinstance(v, tuple) and 'loss' in k]
        mask_loss = (out[mask == 1] - batch['timeseries'][mask == 1]).pow(2).mean()
        (mask_loss + sum(loss_terms)).backward()
        vit_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in enc.pretrained_image_encoder.parameters())
        self.assertTrue(vit_grad, "trainable ViT received no gradient (ar path)")
        # EEG backbone trains via the per-window 'action' conditioning.
        eeg_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in enc.patch_embedding.parameters())
        self.assertTrue(eeg_grad, "EEG backbone received no gradient (ar path)")


class TestFrameAdaLNPredictor(unittest.TestCase):
    """The AdaLN-conditioned dense frame predictor (wm_predictor='frame_adaln')."""

    def _p(self, frame_dim=32, eeg_dim=40, H=4):
        return FrameAdaLNPredictor(frame_dim=frame_dim, eeg_dim=eeg_dim,
                                   predictor_d_model=64, n_layers=2, n_heads=4,
                                   dim_feedforward=128, max_horizon=H)

    def test_shapes_global_and_tokens(self):
        B, P, fd, ed, H = 2, 16, 32, 40, 4
        p = self._p(fd, ed, H)
        s = torch.randn(B, P, fd)
        # Same output contract as FramePredictor: (B, H, P, frame_dim).
        self.assertEqual(p(s, torch.randn(B, ed)).shape, (B, H, P, fd))          # global
        self.assertEqual(p(s, torch.randn(B, 7, ed)).shape, (B, H, P, fd))       # tokens

    def test_eeg_conditioning_changes_output(self):
        # AdaLN-zero is INERT at init (gate=0), so the EEG only matters once the
        # gate opens during training. Simulate a trained state by perturbing the
        # AdaLN modulation off zero, then verify the EEG changes the prediction.
        B, P, fd, ed = 2, 16, 32, 40
        p = self._p(fd, ed); p.eval()
        for blk in p.blocks:
            torch.nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.1)
            torch.nn.init.normal_(blk.adaLN_modulation[-1].bias, std=0.1)
        s, eeg = torch.randn(B, P, fd), torch.randn(B, ed)
        self.assertGreater((p(s, eeg) - p(s, torch.zeros_like(eeg))).abs().mean().item(),
                           1e-6)

    def test_padding_mask_respected(self):
        # Masked EEG tokens must not affect the masked-mean conditioning.
        B, P, fd, ed, M = 2, 8, 16, 12, 5
        p = self._p(fd, ed, H=3); p.eval()
        s, eeg = torch.randn(B, P, fd), torch.randn(B, M, ed)
        kpm = torch.zeros(B, M, dtype=torch.bool); kpm[:, 3:] = True
        o1 = p(s, eeg, eeg_key_padding_mask=kpm)
        eeg2 = eeg.clone(); eeg2[:, 3:] = torch.randn(B, M - 3, ed)
        o2 = p(s, eeg2, eeg_key_padding_mask=kpm)
        self.assertTrue(torch.allclose(o1, o2, atol=1e-5),
                        "masked EEG tokens changed the AdaLN conditioning")


class TestScratchFrameAdaLN(unittest.TestCase):
    def test_frame_adaln_scratch_path(self):
        in_dim, n_ch, n_patches, W = 40, 8, 2, 3
        enc = _scratch_encoder(in_dim=in_dim, n_ch=n_ch, n_patches=n_patches)
        H = W - 1
        pred = FrameAdaLNPredictor(frame_dim=enc.image_feature_dim, eeg_dim=enc.d_model,
                                   predictor_d_model=48, n_layers=2, n_heads=4,
                                   dim_feedforward=96, max_horizon=H)
        w = WorldModelWrapper(
            encoder=enc, predictor=pred, latent_pred_weight=1.0, max_horizon=H,
            ramp_epochs=0, objective='frame', frame_eeg_cond='tokens',
            scratch_vit=True, sigreg_weight=0.09, wm_predictor='frame_adaln')
        w.train()
        batch = _frame_batch(3, n_ch, n_patches, in_dim, W)
        mask = batch.pop('_mask')
        out, info = w.training_step(batch, mask=mask)
        # Reuses the dense frame objective -> same loss keys as 'frame'.
        self.assertIn('frame_pred_loss', info)
        self.assertIn('sigreg_loss', info)
        self.assertIn('diag_frame_emb_std', info)
        loss_terms = [v[0] * v[1] for k, v in info.items()
                      if isinstance(v, tuple) and 'loss' in k]
        mask_loss = (out[mask == 1] - batch['timeseries'][mask == 1]).pow(2).mean()
        (mask_loss + sum(loss_terms)).backward()
        vit = any(p.grad is not None and p.grad.abs().sum().item() > 0
                  for p in enc.pretrained_image_encoder.parameters())
        eeg = any(p.grad is not None and p.grad.abs().sum().item() > 0
                  for p in enc.patch_embedding.parameters())
        self.assertTrue(vit, "trainable ViT received no gradient (frame_adaln)")
        self.assertTrue(eeg, "EEG backbone received no gradient (frame_adaln)")


class TestScratchRegressionOff(unittest.TestCase):
    """With scratch_vit off, the wrapper/encoder must be unchanged."""

    def test_defaults_no_scratch(self):
        enc = CSBrainAlign(
            in_dim=40, out_dim=40, d_model=40, dim_feedforward=160, seq_len=2,
            n_layer=3, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
            brain_regions=None, sorted_indices=[], causal=False,
            alignment_weight=0.0, patch_embed_type='cnn',
            vision_encoder='facebook/dinov2-base')
        # Default = frozen pretrained encoder, not trainable.
        self.assertFalse(getattr(enc, 'vision_trainable', False))
        self.assertFalse(any(p.requires_grad
                             for p in enc.pretrained_image_encoder.parameters()))
        w = WorldModelWrapper(encoder=enc, predictor=None, max_horizon=0,
                              ramp_epochs=0)
        self.assertFalse(w.scratch_vit)
        self.assertIsNone(w.sigreg)


if __name__ == '__main__':
    unittest.main()
