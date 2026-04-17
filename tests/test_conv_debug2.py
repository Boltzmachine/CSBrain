"""Minimal reproduction of TemEmbedEEGLayer causal conv."""
import torch
import torch.nn as nn
from torch.nn import functional as F

B, C, T, D = 1, 1, 5, 8

# Replicate what TemEmbedEEGLayer does manually
kernel_sizes = [(1,), (3,), (5,)]
dim_out = D
num_scales = 3
dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
print("dim_scales:", dim_scales)

convs = nn.ModuleList([
    nn.Conv2d(in_channels=D, out_channels=ds, kernel_size=(kt, 1), padding=(0, 0))
    for (kt,), ds in zip(kernel_sizes, dim_scales)
])
kt_sizes = [kt for (kt,) in kernel_sizes]

x = torch.randn(B, C, T, D, requires_grad=True)
x_r = x.view(B * C, D, T, 1)

fmaps = []
for conv, kt in zip(convs, kt_sizes):
    x_padded = F.pad(x_r, (0, 0, kt - 1, 0))
    print(f"  kt={kt}: x_r={x_r.shape} -> padded={x_padded.shape} -> conv out={conv(x_padded).shape}")
    fmaps.append(conv(x_padded))

y_r = torch.cat(fmaps, dim=1)
y = y_r.view(B, C, T, -1)
print(f"y shape: {y.shape}")

for t_out in range(T):
    scalar = y[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x, retain_graph=True)[0]
    for t_in in range(t_out + 1, T):
        mag = grads[:, :, t_in, :].abs().max().item()
        if mag > 1e-6:
            print(f"  LEAK: out t={t_out} <- in t={t_in} (|grad|={mag:.2e})")

print("Done (no LEAK output = causal)")
