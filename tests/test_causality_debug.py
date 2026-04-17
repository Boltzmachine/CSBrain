"""Debug: find which component leaks future info."""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.CSBrain_transformer import TemEmbedEEGLayer
from models.CSBrain_transformerlayer import CSBrain_TransformerEncoderLayer
from models.alignment import PatchEmbedding

def check_causality(name, fn, x_shape, time_dim=2):
    """Check if fn is causal along time_dim."""
    x = torch.randn(*x_shape, requires_grad=True)
    y = fn(x)
    T_out = y.shape[time_dim]
    T_in = x.shape[time_dim]
    violations = 0
    for t_out in range(T_out):
        idx = [slice(None)] * y.dim()
        idx[time_dim] = t_out
        scalar = y[tuple(idx)].sum()
        grads = torch.autograd.grad(scalar, x, retain_graph=True)[0]
        for t_in in range(t_out + 1, T_in):
            idx_in = [slice(None)] * x.dim()
            idx_in[time_dim] = t_in
            mag = grads[tuple(idx_in)].abs().max().item()
            if mag > 1e-6:
                violations += 1
                if violations <= 3:
                    print(f"  LEAK: output t={t_out} <- input t={t_in} (|grad|={mag:.2e})")
    if violations == 0:
        print(f"  [OK] {name}: strictly causal")
    else:
        print(f"  [FAIL] {name}: {violations} violations")

# Test TemEmbedEEGLayer
print("=== TemEmbedEEGLayer (causal=True) ===")
tem = TemEmbedEEGLayer(dim_in=200, dim_out=200, kernel_sizes=[(1,),(3,),(5,)], causal=True)
tem.eval()
check_causality("TemEmbedEEGLayer", tem, (1, 4, 10, 200), time_dim=2)

print("\n=== TemEmbedEEGLayer (causal=False) ===")
tem_nc = TemEmbedEEGLayer(dim_in=200, dim_out=200, kernel_sizes=[(1,),(3,),(5,)], causal=False)
tem_nc.eval()
check_causality("TemEmbedEEGLayer", tem_nc, (1, 4, 10, 200), time_dim=2)

# Test PatchEmbedding
print("\n=== PatchEmbedding (causal=True) ===")
pe = PatchEmbedding(in_dim=200, out_dim=200, d_model=200, seq_len=10, causal=True)
pe.eval()
def pe_fn(x):
    return pe(x, mask=None)
check_causality("PatchEmbedding", pe_fn, (1, 4, 10, 200), time_dim=2)

print("\n=== PatchEmbedding (causal=False) ===")
pe_nc = PatchEmbedding(in_dim=200, out_dim=200, d_model=200, seq_len=10, causal=False)
pe_nc.eval()
def pe_nc_fn(x):
    return pe_nc(x, mask=None)
check_causality("PatchEmbedding", pe_nc_fn, (1, 4, 10, 200), time_dim=2)

# Test TransformerEncoderLayer
print("\n=== TransformerEncoderLayer (causal=True) ===")
enc = CSBrain_TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True, causal=True)
enc.eval()
def enc_fn(x):
    return enc(x, area_config=None)
check_causality("EncoderLayer", enc_fn, (1, 4, 10, 8), time_dim=2)

print("\n=== TransformerEncoderLayer (causal=False) ===")
enc_nc = CSBrain_TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True, causal=False)
enc_nc.eval()
def enc_nc_fn(x):
    return enc_nc(x, area_config=None)
check_causality("EncoderLayer", enc_nc_fn, (1, 4, 10, 8), time_dim=2)
