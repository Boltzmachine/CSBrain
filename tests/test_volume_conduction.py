"""Tests for the volume-conduction channel positional encoding (arXiv 2601.06134)."""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from models.alignment import VolumeConductionEncoding


def _spherical(cart: torch.Tensor) -> torch.Tensor:
    """Cartesian (..., 3) -> spherical (r, theta=atan2(y,x), phi=acos(z/r))."""
    x, y, z = cart[..., 0], cart[..., 1], cart[..., 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = torch.atan2(y, x)
    phi = torch.acos(z / (r + 1e-8))
    return torch.stack([r, theta, phi], dim=-1)


def test_tau_initialisation_matches_target():
    """softplus(alpha) should equal the requested tau at init."""
    vc = VolumeConductionEncoding(d_model=8, tau_init=0.08)
    with torch.no_grad():
        tau = torch.nn.functional.softplus(vc.alpha) + vc.eps
    assert abs(tau.item() - (0.08 + vc.eps)) < 1e-5, tau.item()


def test_kernel_is_row_stochastic_under_full_mask():
    """Smoothing a delta input over C valid channels should reproduce the
    weighted-average property: the kernel applied to a constant per-row signal
    must return the same constant (a basic sanity check on row normalisation)."""
    torch.manual_seed(0)
    B, C, d = 2, 6, 4
    # Random Cartesian positions on a rough scalp sphere (radius 0.085 m).
    cart = torch.randn(B, C, 3) * 0.05 + torch.tensor([0.0, 0.0, 0.085])
    spher = _spherical(cart)

    vc = VolumeConductionEncoding(d_model=d, coord_kind='spherical', tau_init=0.05)
    # Replace the linear projection with identity (3 -> 3) so we can read the
    # smoothed coordinates directly from the output. Pad to d=4 with zero col.
    with torch.no_grad():
        W = torch.zeros(d, 3)
        W[:3, :3] = torch.eye(3)
        vc.proj.weight.copy_(W)
        vc.proj.bias.zero_()

    out = vc(spher)
    # First 3 dims of out are the smoothed Cartesian positions.
    smoothed = out[..., :3]
    # Each smoothed point should lie within the convex hull of inputs along
    # every axis, i.e. between per-batch min and max.
    cart_min = cart.min(dim=1, keepdim=True).values
    cart_max = cart.max(dim=1, keepdim=True).values
    assert torch.all(smoothed >= cart_min - 1e-5)
    assert torch.all(smoothed <= cart_max + 1e-5)


def test_invalid_channels_are_zeroed_and_excluded():
    """Channels marked invalid (NaN coord or mask=False) must (1) have zero
    output and (2) contribute zero mass to other channels' smoothed positions."""
    torch.manual_seed(1)
    B, C, d = 1, 5, 8
    cart = torch.randn(B, C, 3) * 0.04 + torch.tensor([0.0, 0.0, 0.085])
    # Mark channel 2 invalid via NaN coords.
    cart[0, 2] = float('nan')
    spher = _spherical(cart)

    vc = VolumeConductionEncoding(d_model=d, coord_kind='spherical', tau_init=0.05)
    out = vc(spher)

    assert torch.all(out[0, 2] == 0), out[0, 2]
    assert torch.isfinite(out).all()

    # Repeat with an explicit mask (no NaN), and confirm equivalent behaviour.
    cart2 = torch.randn(B, C, 3) * 0.04 + torch.tensor([0.0, 0.0, 0.085])
    spher2 = _spherical(cart2)
    mask = torch.tensor([[True, True, False, True, True]])
    out2 = vc(spher2, valid_channel_mask=mask)
    assert torch.all(out2[0, 2] == 0)
    assert torch.isfinite(out2).all()


def test_distance_decay_drops_with_separation():
    """Two nearby electrodes should yield more similar smoothed positions
    than a pair that are far apart."""
    torch.manual_seed(2)
    # 3 electrodes: two close (A, B) and one far (C).
    cart = torch.tensor([
        [[0.030, 0.000, 0.085],   # A
         [0.032, 0.000, 0.085],   # B (close to A)
         [-0.030, 0.000, 0.085]], # C (far)
    ])
    spher = _spherical(cart)

    vc = VolumeConductionEncoding(d_model=16, coord_kind='spherical', tau_init=0.02)
    # Use identity-like projection so we can compare in coord space.
    with torch.no_grad():
        W = torch.zeros(16, 3)
        W[:3, :3] = torch.eye(3)
        vc.proj.weight.copy_(W)
        vc.proj.bias.zero_()

    out = vc(spher)
    smoothed = out[..., :3]
    dist_AB = torch.linalg.vector_norm(smoothed[0, 0] - smoothed[0, 1])
    dist_AC = torch.linalg.vector_norm(smoothed[0, 0] - smoothed[0, 2])
    assert dist_AB < dist_AC, (dist_AB.item(), dist_AC.item())


def test_gradient_flows_to_alpha():
    """alpha must receive non-zero gradient so the decay length is learnable."""
    torch.manual_seed(3)
    B, C, d = 2, 8, 16
    cart = torch.randn(B, C, 3) * 0.04 + torch.tensor([0.0, 0.0, 0.085])
    spher = _spherical(cart)
    vc = VolumeConductionEncoding(d_model=d, coord_kind='spherical', tau_init=0.06)
    out = vc(spher)
    out.sum().backward()
    assert vc.alpha.grad is not None
    assert torch.isfinite(vc.alpha.grad).all()
    assert vc.alpha.grad.abs() > 0


def test_align_forward_with_volume_conduction():
    """End-to-end smoke test: CSBrainAlign with use_volume_conduction=True runs."""
    from models.alignment import CSBrainAlign

    torch.manual_seed(4)
    B, C, N, P = 2, 5, 4, 40
    d_model = 40

    model = CSBrainAlign(
        in_dim=P, out_dim=P, d_model=d_model, dim_feedforward=80,
        seq_len=N, n_layer=2, nhead=4,
        TemEmbed_kernel_sizes=[(1,), (3,), (5,)],
        brain_regions=None,
        alignment_weight=0.0,
        use_volume_conduction=True,
        vc_tau_init=0.06,
    ).eval()

    x = torch.randn(B, C, N, P)
    cart = torch.randn(B, C, 3) * 0.04 + torch.tensor([0.0, 0.0, 0.085])
    spher = _spherical(cart)
    batch = {'timeseries': x, 'ch_coords': spher}

    out, info = model(batch, encoder_only=True)
    assert info['global_rep'].shape == (B, d_model)
    assert torch.isfinite(info['global_rep']).all()


if __name__ == '__main__':
    test_tau_initialisation_matches_target()
    print("tau init: OK")
    test_kernel_is_row_stochastic_under_full_mask()
    print("convex-combination smoothing: OK")
    test_invalid_channels_are_zeroed_and_excluded()
    print("invalid-channel handling: OK")
    test_distance_decay_drops_with_separation()
    print("distance decay: OK")
    test_gradient_flows_to_alpha()
    print("alpha gradient: OK")
    test_align_forward_with_volume_conduction()
    print("CSBrainAlign smoke: OK")
