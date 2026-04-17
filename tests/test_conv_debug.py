"""Debug TemEmbedEEGLayer conv dimensions."""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Simulate what TemEmbedEEGLayer does
# Input: (batch, chans, time, d_model) = (1, 4, 10, 200)
# After reshape: (batch*chans, d_model, time, 1) = (4, 200, 10, 1)
# Conv2d kernel: (kt, 1), operates on dims (2, 3) = (time, 1)

# So the conv's spatial dims are (time, 1), and kernel (kt, 1) convolves along time.
# With causal: we need to left-pad dim=2 (time).
# F.pad(x, (0, 0, kt-1, 0)) pads: last dim (0,0), second-to-last dim (kt-1, 0)
# For 4D tensor (N,C,H,W): F.pad(x, (w_left, w_right, h_left, h_right))
# So (0, 0, kt-1, 0) means: W pad (0,0), H pad (kt-1, 0) => left-pad H (time) by kt-1. Correct!

# Let's verify with a simple test
B, C, T, D = 1, 1, 5, 4
x = torch.randn(B*C, D, T, 1, requires_grad=True)
conv = torch.nn.Conv2d(D, D, kernel_size=(3, 1), padding=(0, 0))

# Causal pad
x_padded = F.pad(x, (0, 0, 2, 0))  # left-pad time by 2
print(f"x shape: {x.shape}, padded: {x_padded.shape}")
y = conv(x_padded)
print(f"y shape: {y.shape}")

# Check causality
for t_out in range(T):
    scalar = y[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x, retain_graph=True)[0]
    for t_in in range(t_out + 1, T):
        mag = grads[:, :, t_in, :].abs().max().item()
        if mag > 1e-6:
            print(f"  LEAK: out t={t_out} <- in t={t_in} (|grad|={mag:.2e})")

print("Simple conv test done (no output = causal)")

# Now test the actual TemEmbedEEGLayer
print("\n=== Actual TemEmbedEEGLayer ===")
from models.CSBrain_transformer import TemEmbedEEGLayer
tem = TemEmbedEEGLayer(dim_in=8, dim_out=8, kernel_sizes=[(1,),(3,),(5,)], causal=True)
tem.eval()
x2 = torch.randn(1, 2, 5, 8, requires_grad=True)
y2 = tem(x2)
print(f"in: {x2.shape}, out: {y2.shape}")
for t_out in range(5):
    scalar = y2[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x2, retain_graph=True)[0]
    for t_in in range(t_out + 1, 5):
        mag = grads[:, :, t_in, :].abs().max().item()
        if mag > 1e-6:
            print(f"  LEAK: out t={t_out} <- in t={t_in} (|grad|={mag:.2e})")
print("TemEmbedEEGLayer test done (no output = causal)")
