"""d_model is a free hyper-parameter, independent of the input patch size.

Historically the cnn ``PatchEmbedding`` pinned ``d_model == in_dim``: its first
conv was written ``Conv2d(1, d_model, kernel_size=(1, d_model))``, so the kernel
that is meant to span one raw patch (width ``in_dim``) was sized by the hidden
width instead. The two happened to be equal (40), which hid the conflation --
until you tried to widen the transformer, at which point the conv raised
"Kernel size can't be greater than actual input size".

Two properties are locked in here:

1. BACK-COMPAT -- at ``d_model == in_dim`` every parameter shape is exactly what
   it was before the decoupling, so existing checkpoints still load.
2. DECOUPLING -- ``d_model != in_dim`` builds and trains, reconstructing back to
   the ``out_dim``-wide raw-sample space.

Note ``out_dim`` must stay equal to ``in_dim``: ``proj_out`` is the masked-recon
decoder back to sample space, so its width is the patch size, not the hidden size.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.alignment import CSBrainAlign
from models.world_model import LatentPredictor


NAMES = ['C3', 'C4', 'CZ', 'FC3', 'FC4', 'F3', 'F4', 'PZ']
IN_DIM = 40


def _make_encoder(d_model, in_dim=IN_DIM, spectral_mode='instantaneous',
                  frame_averaging=True, n_patches=3, n_layer=2, **kw):
    return CSBrainAlign(
        in_dim=in_dim, out_dim=in_dim, d_model=d_model,
        dim_feedforward=4 * d_model, seq_len=n_patches,
        n_layer=n_layer, nhead=4, TemEmbed_kernel_sizes=[(1,), (3,)],
        brain_regions=None, sorted_indices=[], causal=False,
        alignment_weight=0.0, spectral_mode=spectral_mode,
        frame_averaging=frame_averaging, flip_split_hidden=32,
        **kw,
    )


def _batch(B=2, N=3, in_dim=IN_DIM, seed=7):
    """A fresh dict holding the SAME data every call.

    Fresh dict because ``forward`` appends the add-global row to
    ``valid_channel_mask`` in place; same data so a loss is comparable across
    optimizer steps.
    """
    C = len(NAMES)
    g = torch.Generator().manual_seed(seed)
    batch = {
        'timeseries': torch.randn(B, C, N, in_dim, generator=g),
        'ch_names': [NAMES] * B,
        'ch_coords': torch.randn(B, C, 3, generator=g),
        'valid_channel_mask': torch.ones(B, C, dtype=torch.bool),
    }
    mask = (torch.rand(B, C, N, generator=g) < 0.5).long()
    return batch, mask


class TestBackCompat(unittest.TestCase):
    """At d_model == in_dim nothing moved: same shapes, so old ckpts still load."""

    def test_patch_embed_shapes_unchanged(self):
        for mode, spectral_in in (('static', IN_DIM // 2 + 1), ('instantaneous', None)):
            with self.subTest(spectral_mode=mode):
                torch.manual_seed(0)
                enc = _make_encoder(IN_DIM, spectral_mode=mode)
                pe = enc.patch_embedding
                # Conv kernel spans one raw patch and emits d_model channels.
                self.assertEqual(tuple(pe.proj_in[0].weight.shape),
                                 (IN_DIM, 1, 1, IN_DIM))
                if spectral_in is not None:
                    # static: fed the rFFT magnitude -> in_dim // 2 + 1 bins.
                    self.assertEqual(tuple(pe.spectral_proj[0].weight.shape),
                                     (IN_DIM, spectral_in))
                self.assertEqual(tuple(enc.proj_out[0].weight.shape),
                                 (IN_DIM, IN_DIM))

    def test_temembed_runs_at_token_width(self):
        """TemEmbed is a residual on the d_model-wide tokens, so it is sized by
        d_model -- identical to the legacy in_dim sizing when the two agree."""
        torch.manual_seed(0)
        enc = _make_encoder(IN_DIM)
        self.assertEqual(enc.TemEmbedEEGLayer.convs[0].weight.shape[1], IN_DIM)


class TestDecoupled(unittest.TestCase):
    """d_model may differ from the patch size."""

    def test_forward_and_train_step(self):
        d_model = 80
        for mode in ('instantaneous', 'static'):
            with self.subTest(spectral_mode=mode):
                torch.manual_seed(0)
                enc = _make_encoder(d_model, spectral_mode=mode)

                # The conv kernel follows the PATCH, the channels follow d_model.
                self.assertEqual(tuple(enc.patch_embedding.proj_in[0].weight.shape),
                                 (d_model, 1, 1, IN_DIM))
                # TemEmbed rebuilt at the token width, not the patch width.
                self.assertEqual(enc.TemEmbedEEGLayer.convs[0].weight.shape[1], d_model)
                # The recon decoder still lands in raw-sample space.
                self.assertEqual(tuple(enc.proj_out[0].weight.shape), (IN_DIM, d_model))

                opt = torch.optim.AdamW(
                    [p for p in enc.parameters() if p.requires_grad], lr=1e-4)
                losses = []
                for _ in range(2):
                    batch, mask = _batch()
                    out, aux = enc(batch, mask=mask)
                    self.assertEqual(out.shape[-1], IN_DIM)      # recon: patch width
                    self.assertEqual(aux['global_rep'].shape[-1], d_model)  # rep: hidden
                    loss = F.mse_loss(out[mask == 1], batch['timeseries'][mask == 1])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    self.assertTrue(torch.isfinite(loss))
                    losses.append(loss.item())
                self.assertLess(losses[-1], losses[0])

    def test_odd_width_builds(self):
        """Nothing special about 80 -- any d_model divisible by nhead and by the
        GroupNorm group count (5) works."""
        torch.manual_seed(0)
        enc = _make_encoder(120)
        batch, mask = _batch()
        out, aux = enc(batch, mask=mask)
        self.assertEqual(out.shape[-1], IN_DIM)
        self.assertEqual(aux['global_rep'].shape[-1], 120)

    def test_world_model_predictor_follows_d_model(self):
        """The WorldModel predictor takes its EEG width from d_model already."""
        lp = LatentPredictor(d_model=80, predictor_d_model=64, n_layers=1,
                             n_heads=4, dim_feedforward=128, max_horizon=2)
        self.assertEqual(tuple(lp.in_proj_latent.weight.shape), (64, 80))
        self.assertEqual(tuple(lp.out_proj_patch.weight.shape), (80, 64))


class TestSourceProjectorGuard(unittest.TestCase):
    """SourceProjector is built at raw-sample width but consumes d_model-wide
    tokens, so it cannot be decoupled -- it must refuse loudly, not die in a matmul."""

    def test_raises_when_decoupled(self):
        with self.assertRaises(ValueError) as ctx:
            _make_encoder(80, project_to_source=True, frame_averaging=False)
        self.assertIn('project_to_source', str(ctx.exception))

    def test_allowed_when_coupled(self):
        # frame_averaging is independently incompatible with project_to_source
        # (asserted in the frame-averaging path), so it is off here.
        torch.manual_seed(0)
        enc = _make_encoder(IN_DIM, project_to_source=True, frame_averaging=False)
        self.assertTrue(enc.project_to_source)


if __name__ == '__main__':
    unittest.main()
