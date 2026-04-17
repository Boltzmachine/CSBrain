"""
Verify that CSBrainAlign in causal mode is truly causal:
  output[:, :, t, :] must NOT depend on input[:, :, t', :] for any t' > t.

Strategy: compute the Jacobian (via grad) of each output time-step w.r.t. the
full input. Any non-zero gradient from a future input position means leakage.
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.alignment import CSBrainAlign


def test_causality():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, C, T, D = 1, 4, 10, 200
    model = CSBrainAlign(
        in_dim=D, out_dim=D, d_model=D, dim_feedforward=400,
        seq_len=T, n_layer=4, nhead=8, causal=True,
    ).to(device).eval()

    # Input that requires grad so we can compute Jacobian
    x = torch.randn(B, C, T, D, device=device, requires_grad=True)

    ch_coords = torch.randn(B, C, 3, device=device)

    batch = {
        'timeseries': x,
        'ch_coords': ch_coords,
        'valid_channel_mask': torch.ones(B, C, device=device, dtype=torch.bool),
        'valid_length_mask': torch.ones(B, T, device=device, dtype=torch.bool),
        'has_image': torch.ones(B, device=device, dtype=torch.bool),
        'image_encoder_inputs': {
            'pixel_values': torch.randn(B, 3, 224, 224, device=device),
        },
        'source': ['test'] * B,
    }

    out, _ = model(batch, mask=None)
    # out: (B, C, T, D) — after global tokens are stripped

    n_ch_out, n_t_out = out.shape[1], out.shape[2]

    violations = []
    for t_out in range(n_t_out):
        # Sum over batch, channel, feature dims to get a scalar for this time step
        scalar = out[:, :, t_out, :].sum()
        grads = torch.autograd.grad(scalar, x, retain_graph=True)[0]
        # grads shape: (B, C, T, D)
        # Check that grads for any input time t_in > t_out are zero
        for t_in in range(t_out + 1, T):
            grad_magnitude = grads[:, :, t_in, :].abs().max().item()
            if grad_magnitude > 1e-6:
                violations.append((t_out, t_in, grad_magnitude))

    if violations:
        print("CAUSALITY VIOLATED!")
        for t_out, t_in, mag in violations:
            print(f"  output t={t_out} depends on input t={t_in}  (max |grad| = {mag:.2e})")
    else:
        print("PASSED: model is strictly causal.")


if __name__ == '__main__':
    test_causality()
