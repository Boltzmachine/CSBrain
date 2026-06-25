"""Tests for the equivariant frame-averaging frontend (plans/eeg-wm.md).

The frontend ``f`` splits the patch embedding additively into z = z_bi + z_lat
(feature space), defines the homologous-channel swap P(z) = z_bi + flip(z_lat)
(P^2 = I), and wraps the transformer T by frame averaging over {I, P}:

    h_bi  = (T(z) + T(P z)) / 2      -> P-invariant      (bilateral half)
    h_lat = (T(z) - P T(P z)) / 2    -> P-anti-equivariant (lateral half)

The token feature dim is split half-bilateral / half-lateral. A random per-step
``flip`` presents z or P(z) (and downstream the original / mirrored frame);
presenting P(z) only negates the lateral half. These tests verify:

1. The architecture is *exactly* P-invariant (bi half) and P-anti-equivariant
   (lat half) -- the plan's "Important tests: check equivariance/invariance".
2. The channel-independent frontend f commutes with the raw-channel swap (the
   conv positional encoding is dropped; GroupNorm stats are permutation-invariant
   so they do not break it).
3. The full world-model step trains on flip and non-flip steps: the flip step
   self-supervises reconstruction (no GT flipped series) and feeds a gradient
   into the learned split.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.alignment import CSBrainAlign, build_flip_perm_batch
from models.world_model import LatentPredictor, WorldModelWrapper


NAMES = ['C3', 'C4', 'CZ', 'FC3', 'FC4', 'F3', 'F4', 'PZ']


def _make_encoder(n_patches=3, n_layer=3, alignment_weight=0.0):
    in_dim = 40
    return CSBrainAlign(
        in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
        dim_feedforward=4 * in_dim, seq_len=n_patches,
        n_layer=n_layer, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
        brain_regions=None, sorted_indices=[], causal=False,
        alignment_weight=alignment_weight, frame_averaging=True,
        flip_split_hidden=32,
    )


def _swap_real_channels(t, perm):
    """Channel swap on (B, C, N, d) real-channel tokens via ``perm`` (B, C)."""
    B, C, N, d = t.shape
    idx = perm.view(B, C, 1, 1).expand(B, C, N, d)
    return torch.gather(t, 1, idx)


class TestFrameAveragingEquivariance(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.enc = _make_encoder()
        self.enc.eval()
        # Push signal into the lateral stream so the equivariance test is not
        # vacuous (near-identity init leaves z_lat ~ 0).
        with torch.no_grad():
            self.enc.frame_split.gate_head.bias.fill_(0.6)
        self.B, self.C, self.N = 4, len(NAMES), 3
        self.x = torch.randn(self.B, self.C, self.N, 40)
        self.coords = torch.randn(self.B, self.C, 3)
        self.names = [list(NAMES) for _ in range(self.B)]

    def _tokens(self, flip):
        batch = {'timeseries': self.x, 'ch_coords': self.coords,
                 'ch_names': self.names, 'flip': flip}
        with torch.no_grad():
            _, info = self.enc(batch, encoder_only=True)
        return info['patch_tokens']

    def test_bi_invariant_lat_antiequivariant(self):
        pt0 = self._tokens(flip=False)
        pt1 = self._tokens(flip=True)
        half = pt0.size(-1) // 2
        perm = build_flip_perm_batch(self.names)
        pt0_swapped = _swap_real_channels(pt0, perm)

        bi_err = (pt1[..., :half] - pt0[..., :half]).abs().max().item()
        lat_err = (pt1[..., half:] + pt0_swapped[..., half:]).abs().max().item()
        lat_mag = pt0[..., half:].abs().mean().item()

        self.assertLess(bi_err, 1e-4, f'bilateral half not P-invariant: {bi_err}')
        self.assertLess(lat_err, 1e-4,
                        f'lateral half not P-anti-equivariant: {lat_err}')
        # The decomposition must be non-trivial (lateral half carries signal).
        self.assertGreater(lat_mag, 1e-2, 'lateral half collapsed to ~0')

    def test_P_is_involution(self):
        perm = build_flip_perm_batch(self.names)
        twice = torch.gather(perm, 1, perm)
        ref = torch.arange(self.C).unsqueeze(0).expand(self.B, -1)
        self.assertTrue(torch.equal(twice, ref))

    def test_frontend_commutes_with_channel_swap(self):
        # f(P_raw x) == P f(x): the channel-independent frontend (conv pos-enc
        # dropped) commutes with the homologous-channel swap. GroupNorm pools its
        # stats over the whole channel set, which is permutation-invariant, so it
        # does NOT break the commutation.
        self.assertTrue(self.enc.patch_embedding.drop_pos_conv)
        perm = build_flip_perm_batch(self.names)
        with torch.no_grad():
            z = self.enc.patch_embedding(self.x, None)        # f(x)
            x_swapped = _swap_real_channels(self.x, perm)      # P_raw x
            z_from_swapped = self.enc.patch_embedding(x_swapped, None)
            z_swapped = _swap_real_channels(z, perm)           # P f(x)
        err = (z_from_swapped - z_swapped).abs().max().item()
        self.assertLess(err, 1e-4, f'frontend does not commute with P: {err}')

    def test_drop_pos_conv_changes_embedding(self):
        # Sanity: the flag actually disables the conv positional encoding.
        self.enc.patch_embedding.drop_pos_conv = False
        with torch.no_grad():
            z_with = self.enc.patch_embedding(self.x, None)
            self.enc.patch_embedding.drop_pos_conv = True
            z_without = self.enc.patch_embedding(self.x, None)
        self.assertGreater((z_with - z_without).abs().max().item(), 1e-6)


class TestFrameAveragingForward(unittest.TestCase):
    """Full (loss-computing) forward: alignment + flip-recon self-consistency."""

    def setUp(self):
        torch.manual_seed(0)
        self.enc = _make_encoder(alignment_weight=0.1)
        self.enc.train()
        self.B, self.C, self.N = 4, len(NAMES), 3
        self.batch = {
            'timeseries': torch.randn(self.B, self.C, self.N, 40) / 100.0,
            'ch_coords': torch.randn(self.B, self.C, 3).abs() + 0.1,
            'ch_names': [list(NAMES) for _ in range(self.B)],
            'image_encoder_inputs': {'pixel_values': torch.randn(self.B, 3, 224, 224)},
            'has_image': torch.tensor([True, True, False, True]),
            'source': ['egobrain'] * self.B,
        }
        self.mask = torch.zeros(self.B, self.C, self.N, dtype=torch.long)
        self.mask[:, :, 0] = 1

    def test_flip_step_self_supervises_recon(self):
        out, info = self.enc({**self.batch, 'flip': True}, mask=self.mask)
        self.assertTrue(info.get('skip_external_recon', False))
        self.assertIn('frame_recon_loss', info)
        coef, val = info['frame_recon_loss']
        self.assertTrue(torch.isfinite(val).item())
        self.assertEqual(out.shape, (self.B, self.C, self.N, 40))
        # Alignment present (has_image has True rows).
        self.assertIn('contrastive_loss_0', info)

    def test_nonflip_step_defers_recon_to_trainer(self):
        out, info = self.enc({**self.batch, 'flip': False}, mask=self.mask)
        self.assertNotIn('skip_external_recon', info)
        self.assertNotIn('frame_recon_loss', info)

    def test_hard_negative_alignment_present_both_orientations(self):
        # The same-sample hard-negative (present vs opposite global rep ->
        # present vs mirrored frame) must fire on flip AND non-flip steps and
        # train both the projector and the split.
        for flip in (False, True):
            self.enc.zero_grad()
            _, info = self.enc({**self.batch, 'flip': flip}, mask=self.mask)
            self.assertIn('flip_align_loss', info, f'flip={flip}')
            coef, val = info['flip_align_loss']
            self.assertTrue(torch.isfinite(val).item())
            self.assertIn('diag_flip_discrim_acc', info)
            (coef * val).backward()
            proj_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                            for p in self.enc.frame_flip_align_proj.parameters())
            split_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                             for p in self.enc.frame_split.parameters())
            self.assertTrue(proj_grad, f'proj got no grad (flip={flip})')
            self.assertTrue(split_grad, f'split got no grad (flip={flip})')

    def test_recon_weight_zero_emits_one_key_with_zero_weight(self):
        # frame_avg_recon_weight=0 -> recon is STILL emitted under the single key
        # 'frame_recon_loss' (always logged), but with weight 0 so the trainer
        # does not add it to the loss. The external MSE(out, x) is still skipped.
        enc = CSBrainAlign(
            in_dim=40, out_dim=40, d_model=40, dim_feedforward=160,
            seq_len=self.N, n_layer=3, nhead=4,
            TemEmbed_kernel_sizes=[(1,), (3,)], brain_regions=None,
            sorted_indices=[], causal=False, alignment_weight=0.1,
            frame_averaging=True, flip_split_hidden=32, frame_avg_recon_weight=0.0)
        enc.train()
        _, info = enc({**self.batch, 'flip': True}, mask=self.mask)
        self.assertTrue(info.get('skip_external_recon', False))
        self.assertIn('frame_recon_loss', info)            # one consistent key
        self.assertNotIn('diag_frame_recon', info)         # no separate diag key
        coef, val = info['frame_recon_loss']
        self.assertEqual(coef, 0.0)                        # -> trainer won't add it
        self.assertTrue(torch.isfinite(val).item())

    def test_flip_recon_trains_split(self):
        _, info = self.enc({**self.batch, 'flip': True}, mask=self.mask)
        loss = sum(v[0] * v[1] for k, v in info.items()
                   if isinstance(v, tuple) and 'loss' in k)
        loss.backward()
        split_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in self.enc.frame_split.parameters())
        self.assertTrue(split_grad, 'frame_split got no gradient')


class TestFrameAveragingWorldModel(unittest.TestCase):
    def _wrapper(self, flip_prob):
        enc = _make_encoder(n_patches=3, alignment_weight=0.0)
        enc.frame_avg_flip_prob = flip_prob
        pred = LatentPredictor(d_model=40, predictor_d_model=64, n_layers=2,
                               n_heads=4, dim_feedforward=128, max_horizon=4)
        w = WorldModelWrapper(encoder=enc, predictor=pred, latent_pred_weight=1.0,
                              cls_pred_weight=0.1, max_horizon=2, ramp_epochs=0)
        w.train()
        return enc, w

    def _batch(self):
        in_dim, n_ch, n_patches = 40, len(NAMES), 3
        B, M, W = 5, 2, 3
        return {
            'timeseries': torch.randn(B, n_ch, n_patches, in_dim) / 100.0,
            'ch_coords': torch.randn(B, n_ch, 3).abs() + 0.1,
            'ch_names': [list(NAMES) for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': torch.zeros(B, 3, 224, 224)},
            'has_image': torch.zeros(B, dtype=torch.bool),
            'cinebrain_idx': torch.tensor([1, 3], dtype=torch.long),
            'timeseries_future': torch.randn(M, W, n_ch, n_patches, in_dim),
            'pixel_values_future': torch.zeros(M, W, 3, 224, 224),
            'has_image_future': torch.zeros(M, W, dtype=torch.bool),
            'source': ['alljoined', 'egobrain', 'alljoined', 'egobrain', 'alljoined'],
        }

    def test_flip_step_predicts_and_self_supervises(self):
        torch.manual_seed(0)
        enc, wrapper = self._wrapper(flip_prob=1.0)   # always flip
        batch = self._batch()
        mask = torch.zeros(5, len(NAMES), 3, dtype=torch.long)
        mask[:, :, 0] = 1
        out, info = wrapper.training_step(batch, mask=mask)
        # Standard prediction loss IS the flipped prediction on a flip step.
        self.assertIn('latent_pred_loss', info)
        self.assertTrue(torch.isfinite(info['latent_pred_loss'][1]).item())
        # The legacy raw-space flip branch must NOT fire under frame averaging.
        self.assertNotIn('flip_latent_pred_loss', info)
        # Flip step self-supervises reconstruction.
        self.assertTrue(info.get('skip_external_recon', False))
        self.assertIn('frame_recon_loss', info)
        # Trains the split.
        loss = sum(v[0] * v[1] for k, v in info.items()
                   if isinstance(v, tuple) and 'loss' in k)
        loss.backward()
        split_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                         for p in enc.frame_split.parameters())
        self.assertTrue(split_grad)

    def test_nonflip_step(self):
        torch.manual_seed(0)
        _, wrapper = self._wrapper(flip_prob=0.0)     # never flip
        batch = self._batch()
        mask = torch.zeros(5, len(NAMES), 3, dtype=torch.long)
        mask[:, :, 0] = 1
        _, info = wrapper.training_step(batch, mask=mask)
        self.assertIn('latent_pred_loss', info)
        self.assertNotIn('skip_external_recon', info)
        self.assertNotIn('frame_recon_loss', info)


if __name__ == '__main__':
    unittest.main()
