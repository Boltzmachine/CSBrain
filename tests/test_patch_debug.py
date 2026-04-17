"""Debug PatchEmbedding causality."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.alignment import PatchEmbedding

B, C, T, D = 1, 4, 10, 200

# Test just the positional encoding conv part
pe = PatchEmbedding(in_dim=D, out_dim=D, d_model=D, seq_len=T, causal=True)
pe.eval()

# Step 1: Test proj_in isolation (no time mixing expected)
x = torch.randn(B, C, T, D, requires_grad=True)
mask_x = x.contiguous().view(B, 1, C * T, D)
patch_emb = pe.proj_in(mask_x)
patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(B, C, T, D)

print("=== proj_in only ===")
violations = 0
for t_out in range(T):
    scalar = patch_emb[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x, retain_graph=True)[0]
    for t_in in range(t_out + 1, T):
        mag = grads[:, :, t_in, :].abs().max().item()
        if mag > 1e-6:
            violations += 1
            if violations <= 3:
                print(f"  LEAK: out t={t_out} <- in t={t_in} (|grad|={mag:.2e})")
print(f"  {'[OK]' if violations == 0 else f'[FAIL] {violations} violations'}")

# Step 2: Test positional encoding conv isolation
print("\n=== pos_enc conv only (causal) ===")
x2 = torch.randn(B, C, T, D, requires_grad=True)
pe_input = x2.permute(0, 3, 1, 2)  # (B, D, C, T)
pe_input = F.pad(pe_input, (pe.pos_time_pad, 0))
pos_emb = pe.positional_encoding(pe_input).permute(0, 2, 3, 1)

violations = 0
for t_out in range(T):
    scalar = pos_emb[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x2, retain_graph=True)[0]
    for t_in in range(t_out + 1, T):
        mag = grads[:, :, t_in, :].abs().max().item()
        if mag > 1e-6:
            violations += 1
            if violations <= 3:
                print(f"  LEAK: out t={t_out} <- in t={t_in} (|grad|={mag:.2e})")
print(f"  {'[OK]' if violations == 0 else f'[FAIL] {violations} violations'}")

# Step 3: Test spectral part
print("\n=== spectral (per-patch FFT) ===")
x3 = torch.randn(B, C, T, D, requires_grad=True)
mask_x3 = x3.contiguous().view(B * C * T, D)
spectral = torch.fft.rfft(mask_x3, dim=-1, norm='forward')
spectral = torch.abs(spectral).contiguous().view(B, C, T, D // 2 + 1)
spectral_emb = pe.spectral_proj(spectral)

violations = 0
for t_out in range(T):
    scalar = spectral_emb[:, :, t_out, :].sum()
    grads = torch.autograd.grad(scalar, x3, retain_graph=True)[0]
    for t_in in range(t_out + 1, T):
        mag = grads[:, :, t_in, :].abs().max().item()
        if mag > 1e-6:
            violations += 1
            if violations <= 3:
                print(f"  LEAK: out t={t_out} <- in t={t_in} (|grad|={mag:.2e})")
print(f"  {'[OK]' if violations == 0 else f'[FAIL] {violations} violations'}")
