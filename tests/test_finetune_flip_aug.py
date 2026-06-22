"""Finetune bilateralization-prior flip augmentation (model_for_physio).

Exercises the real ``Model.flip_augment`` method: x_flip = x_bi + flip(x_lat)
via the learned split, plus the left<->right label remap. Built lightweight
(no vision encoder) by attaching a frozen LateralizationSplit and a minimal
spherical-PE stub, so it stays fast and memory-safe.
"""
import math
import types
import unittest

import torch
import torch.nn as nn

from models.alignment import (
    CSBrainAlign, LateralizationSplit, build_flip_perm_batch,
)
from models.model_for_physio import Model


class _PEStub(nn.Module):
    """Minimal carrier of ``CSBrainAlign._spherical_positional_encoding``."""

    def __init__(self):
        super().__init__()
        self.spherical_r_scale = 0.1
        self.spherical_num_freqs = 32
        self.spherical_pe_base = 20.0
        inv = torch.exp(-math.log(20.0)
                        * torch.arange(32, dtype=torch.float32) / 32)
        self.register_buffer('spherical_inv_freq', inv, persistent=False)

    _spherical_positional_encoding = CSBrainAlign._spherical_positional_encoding


def _make_model(in_dim=40, prob=1.0, label_map=(1, 0, 2, 3), enabled=True):
    """Build a Model shell (bypassing the heavy __init__) wired for
    flip_augment: a frozen split + PE stub + label LUT."""
    m = Model.__new__(Model)
    nn.Module.__init__(m)
    m.param = types.SimpleNamespace(in_dim=in_dim, linear_probe=False)
    m.lateral_flip_aug = enabled
    m.lateral_flip_tta = False
    m._needs_split = enabled
    m.lateral_flip_prob = prob
    m.flip_perm = None
    m.backbone = _PEStub()
    if enabled:
        split = LateralizationSplit(in_dim=in_dim, coord_pe_dim=3 * 2 * 32,
                                    hidden=64)
        with torch.no_grad():
            split.gate_head.bias.fill_(0.5)   # constant, non-zero gate
        m.lateral_split = split
        m.register_buffer('flip_label_lut',
                          torch.tensor(label_map, dtype=torch.long))
    m.train()
    return m


def _make_sym_model(prob=1.0, target=2, src=(0, 1)):
    """Minimal Model shell wired for symmetrize_augment (思路1). No split /
    backbone needed — the symmetrization uses a raw full channel swap."""
    m = Model.__new__(Model)
    nn.Module.__init__(m)
    m.param = types.SimpleNamespace(in_dim=40, linear_probe=False)
    m.symmetrize_aug = True
    m.symmetrize_aug_prob = prob
    m.symmetrize_target_label = target
    m.register_buffer('symmetrize_src_labels', torch.tensor(list(src)))
    m.flip_perm = None
    m.lateral_flip_aug = False
    m.lateral_flip_tta = False
    m._needs_split = False
    m.train()
    return m


class TestSymmetrizeAug(unittest.TestCase):
    NAMES = ['C3', 'C4', 'CZ', 'FC3', 'FC4', 'F3', 'F4', 'PZ']

    def _batch(self, y):
        C = len(self.NAMES)
        x = torch.randn(len(y), C, 2, 40)
        return {'x': x.clone(), 'y': y.clone(),
                'ch_names': [list(self.NAMES) for _ in range(len(y))]}, x

    def test_single_hand_becomes_symmetric_both_fists(self):
        m = _make_sym_model(prob=1.0)
        y = torch.tensor([0, 1, 2, 3, 0, 1])
        batch, x = self._batch(y)
        y_out = m.symmetrize_augment(batch, y.clone())

        # single-hand (0,1) -> 2; bilateral classes (2,3) untouched.
        self.assertEqual(y_out.tolist(), [2, 2, 2, 3, 2, 2])

        perm = build_flip_perm_batch([self.NAMES])[0]
        for i in range(len(y)):
            if y[i].item() in (0, 1):
                exp = 0.5 * (x[i] + x[i].index_select(0, perm))
                self.assertTrue(torch.allclose(batch['x'][i], exp, atol=1e-6))
                # x_sym is invariant under the channel swap (truly bilateral).
                self.assertTrue(torch.allclose(
                    batch['x'][i].index_select(0, perm), batch['x'][i], atol=1e-6))
            else:
                self.assertTrue(torch.allclose(batch['x'][i], x[i]))

    def test_prob_zero_and_disabled_are_noop(self):
        for m in (_make_sym_model(prob=0.0), _make_sym_model(prob=1.0)):
            if m.symmetrize_aug_prob == 1.0:
                m.symmetrize_aug = False  # disabled flag
            y = torch.tensor([0, 1, 2, 3])
            batch, x = self._batch(y)
            y_out = m.symmetrize_augment(batch, y.clone())
            self.assertTrue(torch.equal(y_out, y))
            self.assertTrue(torch.allclose(batch['x'], x))

    def test_eval_mode_is_noop(self):
        m = _make_sym_model(prob=1.0)
        m.eval()
        y = torch.tensor([0, 1])
        batch, x = self._batch(y)
        y_out = m.symmetrize_augment(batch, y.clone())
        self.assertTrue(torch.equal(y_out, y))
        self.assertTrue(torch.allclose(batch['x'], x))


class TestFinetuneFlipAug(unittest.TestCase):
    NAMES = ['C3', 'C4', 'CZ', 'FC3', 'FC4', 'F3', 'F4', 'PZ']

    def _batch(self, B=6):
        C = len(self.NAMES)
        x = torch.randn(B, C, 2, 40)
        y = torch.tensor([0, 1, 2, 3, 0, 3])[:B]
        return {
            'x': x.clone(), 'y': y.clone(),
            'ch_coords': torch.randn(B, C, 3).abs() + 0.1,
            'ch_names': [list(self.NAMES) for _ in range(B)],
        }, x, y

    def test_flip_and_label_remap(self):
        m = _make_model(prob=1.0)
        batch, x, y = self._batch()
        y_out = m.flip_augment(batch)

        # Every row flipped (prob=1) -> labels remapped via the LUT.
        lut = torch.tensor([1, 0, 2, 3])
        self.assertTrue(torch.equal(y_out, lut[y]))

        # Signal changed, shape preserved.
        self.assertEqual(batch['x'].shape, x.shape)
        self.assertFalse(torch.allclose(batch['x'], x))

        # x_flip == x - x_lat + flip(x_lat).
        perm = build_flip_perm_batch([self.NAMES])[0]
        coord_pe = m.backbone._spherical_positional_encoding(batch['ch_coords'])[0]
        with torch.no_grad():
            x_lat, _ = m.lateral_split(x, coord_pe)
        expected = x - x_lat + x_lat.index_select(1, perm)
        self.assertTrue(torch.allclose(batch['x'], expected, atol=1e-5))

        # Channel-swap sanity: C3<->C4, midline maps to itself.
        self.assertEqual(perm[self.NAMES.index('C3')].item(),
                         self.NAMES.index('C4'))
        self.assertEqual(perm[self.NAMES.index('CZ')].item(),
                         self.NAMES.index('CZ'))

    def test_label_lut_values(self):
        # left<->right swapped; bilateral classes fixed.
        m = _make_model(prob=1.0)
        batch, x, y = self._batch()
        y_out = m.flip_augment(batch)
        self.assertEqual(y_out.tolist(), [1, 0, 2, 3, 1, 3])

    def test_prob_zero_is_noop(self):
        m = _make_model(prob=0.0)
        batch, x, y = self._batch()
        y_out = m.flip_augment(batch)
        self.assertTrue(torch.equal(y_out, y))
        self.assertTrue(torch.allclose(batch['x'], x))

    def test_disabled_is_noop(self):
        m = _make_model(enabled=False)
        batch, x, y = self._batch()
        y_out = m.flip_augment(batch)
        self.assertTrue(torch.equal(y_out, y))
        self.assertTrue(torch.allclose(batch['x'], x))

    def test_eval_mode_is_noop(self):
        m = _make_model(prob=1.0)
        m.eval()
        batch, x, y = self._batch()
        y_out = m.flip_augment(batch)
        self.assertTrue(torch.equal(y_out, y))
        self.assertTrue(torch.allclose(batch['x'], x))

    def test_tta_combines_flipped_logits(self):
        # Stub the backbone+classifier core with preset logits so we can check
        # the TTA combination math (flip computation still runs through the
        # real split + permutation + label-aligned average).
        m = _make_model(enabled=True)
        m.lateral_flip_tta = True
        m.eval()
        preset = [torch.tensor([[2.0, 1.0, 0.0, 0.0]]),    # logits(x)
                  torch.tensor([[0.0, 3.0, 0.0, 0.0]])]    # logits(x_flip)
        calls = {'n': 0}

        def stub(b):
            out = preset[calls['n']]
            calls['n'] += 1
            self.assertIn('x', b)            # each call gets its own 'x'
            return out
        m._forward_core = stub

        C = len(self.NAMES)
        batch = {
            'x': torch.randn(1, C, 2, 40),
            'y': torch.tensor([0]),
            'ch_coords': torch.randn(1, C, 3).abs() + 0.1,
            'ch_names': [list(self.NAMES)],
        }
        out = m.forward(batch)
        # flipped logits gathered by lut=[1,0,2,3] -> [3,0,0,0]; averaged with
        # [2,1,0,0] -> [2.5, 0.5, 0, 0]. The flipped 'right' (col 1) reinforces
        # the original 'left' (col 0).
        self.assertEqual(calls['n'], 2)
        self.assertTrue(torch.allclose(out, torch.tensor([[2.5, 0.5, 0.0, 0.0]])))
        self.assertEqual(out.argmax(1).item(), 0)


if __name__ == '__main__':
    unittest.main()
