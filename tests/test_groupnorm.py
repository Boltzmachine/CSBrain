"""Confirm GroupNorm causes the leak."""
import torch, torch.nn as nn

B, C, T, D = 1, 4, 10, 200

# Conv only (no norm)
conv = nn.Conv2d(1, D, kernel_size=(1, D), padding=(0, 0))
x = torch.randn(B, C, T, D, requires_grad=True)
mask_x = x.contiguous().view(B, 1, C*T, D)
out = conv(mask_x).permute(0, 2, 1, 3).view(B, C, T, D)

violations = 0
for t_out in range(T):
    scalar = out[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x, retain_graph=True)[0]
    for t_in in range(t_out+1, T):
        if grads[:, :, t_in, :].abs().max().item() > 1e-6:
            violations += 1
print(f"Conv only: {violations} violations")

# Conv + GroupNorm
seq = nn.Sequential(
    nn.Conv2d(1, D, kernel_size=(1, D), padding=(0, 0)),
    nn.GroupNorm(5, D),
)
x2 = torch.randn(B, C, T, D, requires_grad=True)
mask_x2 = x2.contiguous().view(B, 1, C*T, D)
out2 = seq(mask_x2).permute(0, 2, 1, 3).view(B, C, T, D)

violations = 0
for t_out in range(T):
    scalar = out2[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x2, retain_graph=True)[0]
    for t_in in range(t_out+1, T):
        if grads[:, :, t_in, :].abs().max().item() > 1e-6:
            violations += 1
print(f"Conv + GroupNorm: {violations} violations")
