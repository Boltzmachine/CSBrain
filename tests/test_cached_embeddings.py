"""Tests for cached frozen-vision-encoder embeddings (datasets/
egobrain_extract_embeddings.py + the model substitution).

Two layers:

* ``TestCachedCollateAndReshape`` — fast, no encoder: the ``collate_egobrain``
  stacking/row-ordering of the cached keys and the ``_image_patch_grid(grid=...)``
  reshape. Runs everywhere.
* ``TestCachedEmbeddingParity`` — loads ``facebook/dinov2-base`` (skipped if the
  weights can't be obtained) and asserts the cached path is numerically
  equivalent to the live encoder path for all three consumers: image alignment
  CLS, the flip-align lateral descriptor, and the world-model frame objective.
  The cache here is built in-memory by the SAME ``encode_frame_embeddings`` the
  extractor uses, so this byte-checks extraction + substitution together.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from datasets.egobrain_dataset import collate_egobrain
from datasets.egobrain_extract_embeddings import encode_frame_embeddings
from models.alignment import CSBrainAlign
from models.world_model import FramePredictor, WorldModelWrapper

NAMES = ['C3', 'C4', 'CZ', 'FC3', 'FC4', 'F3', 'F4', 'PZ']


# ---------------------------------------------------------------------------
# Fast unit tests — no vision encoder.
# ---------------------------------------------------------------------------
class TestCachedCollateAndReshape(unittest.TestCase):
    def _item(self, i, W=2, C=4, N=2, d=8, d_img=6, P=4, sz=8, with_emb=True):
        it = {
            'timeseries': torch.randn(W, C, N, d),
            'ch_coords': torch.randn(C, 3),
            'ch_names': list(NAMES[:C]),
            'pixel_values': torch.randn(W, 3, sz, sz),
            'has_image': torch.ones(W, dtype=torch.bool),
            'hand_targets': torch.zeros(W, 2),
            'hand_valid': torch.zeros(W, 2, dtype=torch.bool),
            'source': 'egobrain',
            'session_id': f'P{i:04d}',
        }
        if with_emb:
            # Tag every tensor with the item index so row order is checkable.
            it['frame_cls'] = torch.full((W, d_img), float(i))
            it['frame_cls_flip'] = torch.full((W, d_img), float(i) + 0.5)
            it['frame_grid'] = torch.full((W, P, d_img), float(i))
            it['frame_grid_flip'] = torch.full((W, P, d_img), float(i) + 0.5)
        return it

    def test_collate_stacks_cached_keys_in_row_order(self):
        B, W, d_img, P = 3, 2, 6, 4
        batch = [self._item(i) for i in range(B)]
        out = collate_egobrain(batch)
        self.assertEqual(out['frame_cls'].shape, (B, d_img))
        self.assertEqual(out['frame_cls_flip'].shape, (B, d_img))
        self.assertEqual(out['frame_grid'].shape, (B, P, d_img))
        self.assertEqual(out['frame_grid_flip'].shape, (B, P, d_img))
        self.assertEqual(out['frame_grid_future'].shape, (B, W, P, d_img))
        self.assertEqual(out['frame_grid_flip_future'].shape, (B, W, P, d_img))
        # Window-0 tensors are item i's window-0 (== i); futures preserve order.
        for i in range(B):
            self.assertTrue(torch.all(out['frame_cls'][i] == i))
            self.assertTrue(torch.all(out['frame_grid'][i] == i))
            self.assertTrue(torch.all(out['frame_grid_future'][i] == i))
            self.assertTrue(torch.all(out['frame_grid_flip_future'][i] == i + 0.5))

    def test_collate_zerofills_items_lacking_cached_keys(self):
        # A no-video item (no frame_* keys, has_image all-False) must not disable
        # the cache for the whole batch: the keys are emitted, the missing item's
        # rows are zero-filled (the model masks them out via has_image anyway).
        batch = [self._item(3), self._item(1, with_emb=False)]
        out = collate_egobrain(batch)
        self.assertIn('frame_grid', out)
        self.assertIn('frame_grid_future', out)
        self.assertTrue(torch.all(out['frame_grid'][0] == 3))    # present item -> value 3
        self.assertTrue(torch.all(out['frame_grid'][1] == 0))    # missing item -> zero-fill
        self.assertTrue(torch.all(out['frame_grid_flip'][1] == 0))
        self.assertTrue(torch.all(out['frame_grid_future'][1] == 0))

    def test_collate_omits_cached_keys_when_no_item_has_them(self):
        batch = [self._item(0, with_emb=False), self._item(1, with_emb=False)]
        out = collate_egobrain(batch)
        for k in ('frame_cls', 'frame_grid', 'frame_grid_future'):
            self.assertNotIn(k, out)
        self.assertIn('pixel_values_future', out)
        self.assertIn('image_encoder_inputs', out)

    def test_image_patch_grid_accepts_precomputed_grid(self):
        # The grid path of _image_patch_grid touches no instance state, so call
        # it unbound with self=None: a flat (B,P,d) grid must reshape to the
        # square (B,s,s,d), and an already-(B,s,s,d) grid passes through.
        B, s, d = 2, 4, 6
        flat = torch.randn(B, s * s, d)
        out = CSBrainAlign._image_patch_grid(None, grid=flat)
        self.assertEqual(out.shape, (B, s, s, d))
        self.assertTrue(torch.equal(out, flat.reshape(B, s, s, d)))
        already = torch.randn(B, s, s, d)
        self.assertTrue(torch.equal(
            CSBrainAlign._image_patch_grid(None, grid=already), already))


# ---------------------------------------------------------------------------
# Parity tests — real DINOv2.
# ---------------------------------------------------------------------------
def _make_encoder(alignment_weight=0.1, n_patches=3, n_layer=3):
    in_dim = 40
    return CSBrainAlign(
        in_dim=in_dim, out_dim=in_dim, d_model=in_dim,
        dim_feedforward=4 * in_dim, seq_len=n_patches,
        n_layer=n_layer, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
        brain_regions=None, sorted_indices=[], causal=False,
        alignment_weight=alignment_weight, frame_averaging=True,
        flip_split_hidden=32)


class TestCachedEmbeddingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        # Load lazily here (not in a class decorator) so the fast unit tests
        # above never trigger the dinov2-base download/load.
        try:
            cls.enc = _make_encoder().eval()
        except Exception as e:                              # noqa: BLE001
            raise unittest.SkipTest(
                f'facebook/dinov2-base weights unavailable: {e}')
        cls.model = cls.enc.pretrained_image_encoder
        cls.B = 4
        cls.pv = torch.randn(cls.B, 3, 224, 224)
        # float32 in-memory cache (extraction slices). The model upcasts the
        # real fp16 cache to float32 at read, so a float32 cache is the right
        # parity reference; a separate fp16-rounded check bounds the storage error.
        cls.emb = encode_frame_embeddings(cls.model, cls.pv)

    def test_grid_extraction_matches_image_patch_grid(self):
        live = self.enc._image_patch_grid(pixel_values=self.pv).reshape(
            self.B, -1, self.emb['grid'].size(-1))
        cached = self.enc._image_patch_grid(grid=self.emb['grid']).reshape(
            self.B, -1, self.emb['grid'].size(-1))
        self.assertTrue(torch.allclose(live, cached, atol=1e-4),
                        f'grid mismatch: {(live - cached).abs().max().item()}')

    def test_cls_extraction_matches_dinov2_cls_token(self):
        cls_live = self.enc._dinov2_cls_token({'pixel_values': self.pv})[:, 0]
        self.assertTrue(torch.allclose(cls_live, self.emb['cls'], atol=1e-4))
        flip_live = self.enc._dinov2_cls_token(
            {'pixel_values': torch.flip(self.pv, dims=[-1])})[:, 0]
        self.assertTrue(torch.allclose(flip_live, self.emb['cls_flip'], atol=1e-4))

    def test_lateral_descriptor_matches_on_mirrored_frame(self):
        # The flip-align target: cached grid_flip must reproduce the descriptor of
        # the re-encoded mirrored frame (NOT a feature-space flip).
        live = self.enc._image_lateral_descriptor(
            pixel_values=torch.flip(self.pv, dims=[-1]))
        cached = self.enc._image_lateral_descriptor(grid=self.emb['grid_flip'])
        self.assertTrue(torch.allclose(live, cached, atol=1e-4),
                        f'descriptor mismatch: {(live - cached).abs().max().item()}')

    def test_fp16_storage_within_tolerance(self):
        # The default on-disk cache is float32 (bit-exact, covered by the grid
        # test above); this bounds the round-trip error of the OPTIONAL
        # --dtype float16 mode on the grid.
        g16 = self.emb['grid'].half().float()
        live = self.enc._image_patch_grid(pixel_values=self.pv).reshape(
            self.B, -1, g16.size(-1))
        cached = self.enc._image_patch_grid(grid=g16).reshape(
            self.B, -1, g16.size(-1))
        self.assertTrue(torch.allclose(live, cached, atol=2e-2))

    def _align_batch(self, flip):
        return {
            'timeseries': torch.randn(self.B, len(NAMES), 3, 40) / 100.0,
            'ch_coords': torch.randn(self.B, len(NAMES), 3).abs() + 0.1,
            'ch_names': [list(NAMES) for _ in range(self.B)],
            'image_encoder_inputs': {'pixel_values': self.pv},
            'has_image': torch.tensor([True, True, False, True]),
            'source': ['egobrain'] * self.B,
            'flip': flip,
        }

    def test_forward_alignment_and_flipalign_parity(self):
        # Sites A (alignment CLS) + B (flip-align descriptor) end-to-end through
        # the frame-averaging encoder forward. Build the per-window-0 cache from
        # the same pixel_values and assert the loss values match.
        mask = torch.zeros(self.B, len(NAMES), 3, dtype=torch.long)
        mask[:, :, 0] = 1
        flip = torch.tensor([True, False, True, False])
        base = self._align_batch(flip)
        cached = {
            **base,
            'frame_cls': self.emb['cls'], 'frame_cls_flip': self.emb['cls_flip'],
            'frame_grid': self.emb['grid'], 'frame_grid_flip': self.emb['grid_flip'],
        }
        with torch.no_grad():
            torch.manual_seed(1)
            _, info_live = self.enc(base, mask=mask)
            torch.manual_seed(1)
            _, info_cached = self.enc(cached, mask=mask)
        for key in ('contrastive_loss_0', 'flip_align_loss'):
            self.assertIn(key, info_live)
            self.assertIn(key, info_cached)
            lv, cv = info_live[key][1], info_cached[key][1]
            self.assertTrue(torch.allclose(lv, cv, atol=1e-4),
                            f'{key}: live {lv.item()} vs cached {cv.item()}')

    def _frame_wrapper(self, flip_prob):
        enc = _make_encoder(alignment_weight=0.0).eval()
        enc.frame_avg_flip_prob = flip_prob
        pred = FramePredictor(frame_dim=self.emb['grid'].size(-1), eeg_dim=40,
                              predictor_d_model=64, n_layers=2, n_heads=4,
                              dim_feedforward=128, max_horizon=1)
        w = WorldModelWrapper(
            encoder=enc, predictor=pred, latent_pred_weight=1.0,
            cls_pred_weight=0.0, max_horizon=1, ramp_epochs=0,
            objective='frame', frame_eeg_cond='tokens').eval()
        return enc, w

    def _frame_batch(self, enc):
        in_dim, n_ch, n_patches = 40, len(NAMES), 3
        B, W = self.B, 2
        # All B rows are EgoBrain rows that carry a future stack (cb_idx=arange).
        pvf = torch.randn(B, W, 3, 224, 224)
        emb_f = [encode_frame_embeddings(enc.pretrained_image_encoder, pvf[:, w])
                 for w in range(W)]
        grid_f = torch.stack([e['grid'] for e in emb_f], dim=1)        # (B,W,P,d)
        grid_ff = torch.stack([e['grid_flip'] for e in emb_f], dim=1)
        batch = {
            'timeseries': torch.randn(B, n_ch, n_patches, in_dim) / 100.0,
            'ch_coords': torch.randn(B, n_ch, 3).abs() + 0.1,
            'ch_names': [list(NAMES) for _ in range(B)],
            'valid_channel_mask': torch.ones(B, n_ch, dtype=torch.bool),
            'valid_length_mask': torch.ones(B, n_patches, dtype=torch.bool),
            'image_encoder_inputs': {'pixel_values': pvf[:, 0]},
            'has_image': torch.ones(B, dtype=torch.bool),
            'timeseries_future': torch.randn(B, W, n_ch, n_patches, in_dim),
            'pixel_values_future': pvf,
            'has_image_future': torch.ones(B, W, dtype=torch.bool),
            'source': ['egobrain'] * B,
        }
        cached_extra = {'frame_grid_future': grid_f,
                        'frame_grid_flip_future': grid_ff}
        return batch, cached_extra

    def test_frame_objective_parity_both_orientations(self):
        # Site C: the world-model frame objective. flip_prob in {0,1} forces all
        # rows to use grid / grid_flip respectively (max_horizon=1 => k=1), so the
        # cached per-row selection is deterministically exercised both ways.
        mask = torch.zeros(self.B, len(NAMES), 3, dtype=torch.long)
        mask[:, :, 0] = 1
        for flip_prob in (0.0, 1.0):
            enc, wrapper = self._frame_wrapper(flip_prob)
            batch, cached_extra = self._frame_batch(enc)
            with torch.no_grad():
                torch.manual_seed(7)
                _, info_live = wrapper.training_step(batch, mask=mask)
                torch.manual_seed(7)
                _, info_cached = wrapper.training_step(
                    {**batch, **cached_extra}, mask=mask)
            self.assertIn('frame_pred_loss', info_live)
            lv = info_live['frame_pred_loss'][1]
            cv = info_cached['frame_pred_loss'][1]
            self.assertTrue(torch.allclose(lv, cv, atol=1e-4),
                            f'flip_prob={flip_prob}: frame_pred_loss live '
                            f'{lv.item()} vs cached {cv.item()}')


if __name__ == '__main__':
    unittest.main()
