"""Checkpoint probe: does the frame-averaging LATERAL half carry left-vs-right
motor imagery, and the BILATERAL half carry the symmetric somatotopy axis?

Loads a WorldModel pretrain checkpoint exactly as the PhysioNet-MI finetune
does (utils.util.load_pretrain_checkpoint + apply_arch_params + model_for_physio),
runs the encoder in CANONICAL eval (flip=False) on seg0 windows, then splits the
window-level rep h = [bilateral(:d/2) ; lateral(d/2:)] and measures Fisher
separation along:
  * LR axis      : class 0 (left fist)  vs class 1 (right fist)   [anti-symmetric]
  * hand-vs-feet : class 2 (both fists) vs class 3 (both feet)    [symmetric somatotopy]
  * any-hand/feet: {0,1,2} vs {3}                                  [symmetric]
  * extent       : {0,1} (single) vs 2 (both fists)               [symmetric]
Fisher here = regularized (shrinkage) multivariate LDA separation
  J = (mu_A - mu_B)^T (Sw + eps I)^-1 (mu_A - mu_B)
on the chosen half. Higher = more linearly separable along that axis.

Usage: probe_lateral.py <ckpt_path> [noequi]
"""
import sys, types, argparse
import numpy as np
import torch

sys.path.insert(0, '/gpfs/radev/pi/ying_rex/wq44/CSBrain')
from utils.util import load_pretrain_checkpoint, apply_arch_params
from datasets import physio_dataset
from models import model_for_physio

CKPT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else CKPT.split('/')[-2]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Build a params namespace matching the erp-150 PhysioNet-MI finetune ----
p = argparse.Namespace(
    model='Align', downstream_dataset='PhysioNet-MI',
    datasets_dir='data/preprocessed/physionet_mi', num_of_classes=4,
    foundation_dir=CKPT, seed=42, use_pretrained_weights=True,
    use_initial_segment_only=True, segment_index=0, highpass_hz=0.0,
    batch_size=128, fs=200, frame_rep_mode='both', temporal_jitter=0,
    bilateral_head=False, frame_flip_aug=False, frame_flip_tta=False,
    lateralization_flip=False, flip_aug=False, flip_tta=False,
    symmetrize_aug=False, use_euclidean_alignment=False,
    dropout=0.1, linear_probe=False, multi_lr=False,
)
_, saved = load_pretrain_checkpoint(p.foundation_dir)
apply_arch_params(p, saved)
print(f"[{TAG}] arch: d_model={p.d_model} seq_len={p.seq_len} in_dim={p.in_dim} "
      f"n_layer={p.n_layer} frame_averaging={getattr(p,'frame_averaging',None)}")

torch.manual_seed(0)
ds = physio_dataset.LoadDataset(p)
loaders = ds.get_data_loader()
model = model_for_physio.Model(p).eval()
backbone = model.backbone
# The frozen DINOv2 image encoder is unused on the encoder_only path — drop it
# to save memory before moving to device.
if hasattr(backbone, 'pretrained_image_encoder'):
    del backbone.pretrained_image_encoder
backbone = backbone.to(DEV).eval()
fa = bool(getattr(backbone, 'frame_averaging', False))
print(f"[{TAG}] backbone.frame_averaging = {fa}")

seg_len = p.seq_len
in_dim = p.in_dim

@torch.no_grad()
def extract(split):
    G, P, Y = [], [], []
    for batch in loaders[split]:
        x = batch['x'].to(DEV)
        y = batch['y']
        x = x.reshape(x.size(0), x.size(1), -1, in_dim)        # (B,C,seq,in_dim)
        x = x[:, :, 0:seg_len, :].contiguous()                  # seg0 crop
        bdict = {
            'timeseries': x,
            'ch_coords': batch['ch_coords'].to(DEV),
            'ch_names': batch['ch_names'],
        }
        if fa:
            bdict['flip'] = torch.zeros(x.size(0), dtype=torch.bool, device=DEV)
        _, info = backbone(bdict, encoder_only=True)
        g = info['global_rep']                                  # (B, d)
        pt = info['patch_tokens']                               # (B, C, N, d)
        pmean = pt.mean(dim=(1, 2))                              # (B, d)
        G.append(g.float().cpu().numpy())
        P.append(pmean.float().cpu().numpy())
        Y.append(np.asarray(y))
    return np.concatenate(G), np.concatenate(P), np.concatenate(Y)

splits = {}
for s in ('train', 'test'):
    g, pm, y = extract(s)
    splits[s] = (g, pm, y)
    print(f"[{TAG}] {s}: n={len(y)} class_counts={np.bincount(y, minlength=4).tolist()} dim={g.shape[1]}")

def fisher(F, y, A, B, eps_frac=1e-2):
    """Regularized multivariate Fisher (LDA) separation between label-set A and B."""
    ia = np.isin(y, A); ib = np.isin(y, B)
    Xa, Xb = F[ia], F[ib]
    if len(Xa) < 5 or len(Xb) < 5:
        return float('nan')
    mu_a, mu_b = Xa.mean(0), Xb.mean(0)
    d = mu_a - mu_b
    # pooled within-class covariance
    Ca = np.cov(Xa, rowvar=False)
    Cb = np.cov(Xb, rowvar=False)
    na, nb = len(Xa), len(Xb)
    Sw = ((na - 1) * Ca + (nb - 1) * Cb) / (na + nb - 2)
    Sw = np.atleast_2d(Sw)
    eps = eps_frac * np.trace(Sw) / Sw.shape[0]
    Sw_reg = Sw + eps * np.eye(Sw.shape[0])
    try:
        J = d @ np.linalg.solve(Sw_reg, d)
    except np.linalg.LinAlgError:
        J = float('nan')
    return float(J)

AXES = {
    'LR (0v1, lateralized)':        ([0], [1]),
    'hand-v-feet (2v3, symmetric)': ([2], [3]),
    'anyhand-v-feet ({012}v3)':     ([0, 1, 2], [3]),
    'extent ({01}v2, symmetric)':   ([0, 1], [2]),
}

half = splits['train'][0].shape[1] // 2

def report(rep_name, gi):
    g_tr, y_tr = splits['train'][gi], splits['train'][2]
    g_te, y_te = splits['test'][gi], splits['test'][2]
    print(f"\n===== {TAG} :: {rep_name} rep  (d={g_tr.shape[1]}, half={half}) =====")
    print(f"{'axis':<32} {'FULL':>9} {'BILAT[:h]':>11} {'LAT[h:]':>11} {'LAT/BILAT':>10}")
    for split_name, (X, Y) in (('train', (g_tr, y_tr)), ('test', (g_te, y_te))):
        print(f"  -- {split_name} --")
        for name, (A, B) in AXES.items():
            jf = fisher(X, Y, A, B)
            jb = fisher(X[:, :half], Y, A, B)
            jl = fisher(X[:, half:], Y, A, B)
            ratio = jl / jb if (jb and jb == jb and jb > 1e-9) else float('nan')
            print(f"  {name:<30} {jf:9.3f} {jb:11.3f} {jl:11.3f} {ratio:10.2f}")

report('global_rep', 0)
report('patch_mean', 1)

# Also report the per-half gate energy (how much rep is in lateral half) via
# the raw feature variance fraction in each half (sanity on the split scale).
g_all = np.concatenate([splits['train'][0], splits['test'][0]])
vb = g_all[:, :half].var(0).mean()
vl = g_all[:, half:].var(0).mean()
print(f"\n[{TAG}] global_rep feature variance: bilat={vb:.4f} lat={vl:.4f} "
      f"lat_frac={vl/(vb+vl):.3f}")
