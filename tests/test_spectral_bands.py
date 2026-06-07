"""Tests for the learnable spectral-band feature in ``CSBrainAlign``.

Covers: filterbank ordering/bounds, that the feature is shape-transparent
(bands OFF == baseline external shapes; bands ON keeps the same external
shapes so masked-recon / encoder_only callers are unaffected), the
multi-level alignment loss, a backward smoke test, and the ablation toggles.

The frozen vision encoder (``facebook/dinov2-base``) is loaded once and must
be available in the local HF cache.
"""
import pytest
import torch
import torch.nn.functional as F

from models.alignment import CSBrainAlign, LearnableFilterbank

# in_dim must equal d_model (patch_size == d_model convention in PatchEmbedding);
# d_model must be divisible by the GroupNorm groups (5 and 8) -> 40.
B, C, N, D = 2, 6, 8, 40
VENC = "facebook/dinov2-base"


def _model(use_bands, **kw):
    return CSBrainAlign(
        in_dim=D, out_dim=D, d_model=D, dim_feedforward=64, seq_len=N,
        n_layer=2, nhead=4, vision_encoder=VENC,
        use_spectral_bands=use_bands, num_spectral_bands=4, fs=200,
        filterbank_kernel_size=31, num_visual_levels=3, **kw,
    )


def _batch(with_image):
    batch = {
        "timeseries": torch.randn(B, C, N, D),
        "ch_coords": torch.randn(B, C, 3),
        "valid_channel_mask": torch.ones(B, C, dtype=torch.bool),
        "valid_length_mask": torch.ones(B, N, dtype=torch.bool),
        "source": ["s"] * B,
    }
    if with_image:
        batch["has_image"] = torch.ones(B, dtype=torch.bool)
        batch["image_encoder_inputs"] = {"pixel_values": torch.randn(B, 3, 224, 224)}
    return batch


def test_filterbank_independent_bounds():
    fb = LearnableFilterbank(num_bands=5, fs=200, kernel_size=31, min_bw_hz=1.0)
    # Each band has its own learned low/high; bounds stay valid for ANY logits.
    with torch.no_grad():
        fb.low_logits.copy_(torch.randn(5) * 3)
        fb.width_logits.copy_(torch.randn(5) * 3)
    low, high = fb.band_bounds()
    assert low.shape == (5,) and high.shape == (5,)
    assert torch.all(low >= 1.0 - 1e-4), low            # >= f_min
    assert torch.all(high <= 100.0 + 1e-4), high        # <= nyquist
    assert torch.all(high >= low + 1.0 - 1e-4), (low, high)  # >= min_bw wide
    out = fb(torch.randn(B, C, N, D))
    assert out.shape == (B, 5, C, N, D)


def test_filterbank_bands_distinct_at_init():
    # Equal-width tiling init must break the K-fold symmetry (bands not identical).
    fb = LearnableFilterbank(num_bands=4, fs=200, kernel_size=31)
    low, high = fb.band_bounds()
    centers = 0.5 * (low + high)
    assert centers.unique().numel() == 4, centers       # all distinct


@pytest.mark.parametrize("use_bands", [False, True])
def test_external_shapes_transparent(use_bands):
    m = _model(use_bands).eval()
    mask = (torch.rand(B, C, N) > 0.5).long()
    with torch.no_grad():
        out, info = m(_batch(False), mask=mask)
        assert out.shape == (B, C, N, D)
        assert info["global_rep"].shape == (B, D)
        assert info["patch_tokens"].shape == (B, C, N, D)
        _ = out[mask == 1]  # masked-recon indexing must line up
        _, enc = m(_batch(False), mask=None, encoder_only=True)
        assert enc["global_rep"].shape == (B, D)
        assert enc["patch_tokens"].shape == (B, C, N, D)


def test_multilevel_alignment_loss():
    m = _model(True).eval()
    mask = (torch.rand(B, C, N) > 0.5).long()
    with torch.no_grad():
        _, info = m(_batch(True), mask=mask)
    assert "band_align_loss" in info
    assert info["diag_band_level_assignment"].shape == (4, 3)
    for k, v in info.items():
        if isinstance(v, tuple) and "loss" in k:
            assert torch.isfinite(v[1]).all(), k


def test_backward_grads_reach_band_params():
    m = _model(True).train()
    mask = (torch.rand(B, C, N) > 0.5).long()
    out, info = m(_batch(True), mask=mask)
    loss = sum(v[1] for k, v in info.items() if isinstance(v, tuple) and "loss" in k)
    loss = loss + F.mse_loss(out[mask == 1], torch.zeros_like(out[mask == 1]))
    loss.backward()
    params = dict(m.named_parameters())
    for name in ["filterbank.low_logits", "filterbank.width_logits",
                 "band_type_embed", "band_level_logits"]:
        g = params[name].grad
        assert g is not None and torch.isfinite(g).all(), name


def test_moe_mutual_exclusion():
    with pytest.raises(ValueError):
        _model(True, use_moe=True, num_experts=4)


@pytest.mark.parametrize("kw", [
    {"use_cross_band_attn": False},   # cross-band ablation
    {"use_band_type_embedding": False},  # band-type ablation
    {"cross_band_every": 2},          # cross-band at a subset of layers
])
def test_ablations_run(kw):
    m = _model(True, **kw).eval()
    mask = (torch.rand(B, C, N) > 0.5).long()
    with torch.no_grad():
        out, info = m(_batch(True), mask=mask)
    assert out.shape == (B, C, N, D)
    assert "band_align_loss" in info
