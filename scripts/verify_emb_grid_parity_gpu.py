"""GPU parity check for the time-keyed grid embedding cache.

Re-encodes frames straight from the grid FRAME cache on the GPU, using the SAME
batch_size the extractor used (so the GEMM shapes / reduction order match), and
compares against the cached embeddings.

Includes a CONTROL subject that was cached in an earlier, already-trusted run:
if the control deviates the same way as the newly-extracted subjects, the
deviation is float32 kernel noise from this comparison, not a data defect.

Reports max-abs-diff, max-REL-diff and worst cosine (the meaningful metrics for
high-dimensional float32 embeddings), not a naive absolute tolerance.

  python -m scripts.verify_emb_grid_parity_gpu --control P0001 --subjects P0025,P0030,P0035
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from datasets.egobrain_dataset import _get_normalize_params
from datasets.egobrain_extract_embeddings import encode_frame_embeddings

ROOT = 'data/EgoBrain'
ENC = 'facebook/dinov2-base'
FD = f'{ROOT}/cache_frames_grid_facebook_dinov2-base_g0.2_sz224'
ED = f'{ROOT}/cache_embeddings_grid_facebook_dinov2-base_g0.2_sz224'


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def rel(live, cached):
    d = np.abs(live - cached)
    scale = np.abs(cached) + 1e-3
    return float((d / scale).max())


def check(sub, model, mean, std, dev, bs, n_batches):
    fp, ep = f'{FD}/{sub}.h5', f'{ED}/{sub}.h5'
    out = []
    with h5py.File(fp, 'r') as fh, h5py.File(ep, 'r') as eh:
        n = fh['frames'].shape[0]
        n_full = max(1, n // bs)          # number of complete extractor batches
        # Sample the FIRST, MIDDLE and LAST complete batch. Batch 0 alone would
        # miss a buffer-indexing bug (wrong slice offset), which only manifests
        # part-way through the per-subject write.
        idxs = sorted(set(int(round(i * (n_full - 1) / max(1, n_batches - 1)))
                          for i in range(n_batches)))
        starts = sorted(set(i * bs for i in idxs if i * bs + bs <= n))
        for st in starts:
            px = np.asarray(fh['frames'][st:st + bs])
            x = torch.from_numpy(np.ascontiguousarray(px)).to(dev)
            x = x.float().div_(255.0).permute(0, 3, 1, 2)
            x = (x - mean) / std
            with torch.no_grad():
                emb = encode_frame_embeddings(model, x, n_register_tokens=0)
            c_cache = np.asarray(eh['cls'][st:st + bs])       # (bs,2,768)
            g_cache = np.asarray(eh['grid'][st:st + bs])      # (bs,2,256,768)
            for name, live, cached in (
                    ('cls',       emb['cls'].numpy(),       c_cache[:, 0]),
                    ('cls_flip',  emb['cls_flip'].numpy(),  c_cache[:, 1]),
                    ('grid',      emb['grid'].numpy(),      g_cache[:, 0]),
                    ('grid_flip', emb['grid_flip'].numpy(), g_cache[:, 1])):
                exact = float(np.mean(live == cached))
                out.append((sub, st, name, float(np.abs(live - cached).max()),
                            rel(live, cached), cos(live, cached), exact))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', default='P0001',
                   help='subject from the earlier, trusted extraction run')
    p.add_argument('--subjects', default='P0025,P0030,P0035')
    p.add_argument('--batch_size', type=int, default=256,
                   help='MUST match the extractor (--batch_size) for kernel parity')
    p.add_argument('--n_batches', type=int, default=2)
    a = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={dev} batch_size={a.batch_size}')
    if dev != 'cuda':
        print('WARNING: no GPU -> results are NOT comparable to a GPU-built cache')
    from transformers import AutoModel
    model = AutoModel.from_pretrained(ENC).eval().to(dev)
    for prm in model.parameters():
        prm.requires_grad_(False)
    mean, std = _get_normalize_params(ENC)
    mean, std = mean.to(dev), std.to(dev)

    subs = [a.control] + [s.strip() for s in a.subjects.split(',') if s.strip()]
    rows = []
    for s in subs:
        if not (os.path.exists(f'{FD}/{s}.h5') and os.path.exists(f'{ED}/{s}.h5')):
            print(f'{s}: MISSING'); continue
        rows += check(s, model, mean, std, dev, a.batch_size, a.n_batches)

    print(f'\n{"subj":7} {"start":>7} {"slice":10} {"maxabs":>10} {"maxrel":>9} '
          f'{"cosine":>12} {"exact%":>7}  {"role"}')
    for sub, st, name, ma, mr, cs, ex in rows:
        role = 'CONTROL(old)' if sub == a.control else 'new'
        print(f'{sub:7} {st:7d} {name:10} {ma:10.3e} {mr:9.2e} {cs:12.9f} '
              f'{100*ex:6.1f}%  {role}')

    ctrl = [r for r in rows if r[0] == a.control]
    new = [r for r in rows if r[0] != a.control]
    if ctrl and new:
        cm = max(r[3] for r in ctrl); nm = max(r[3] for r in new)
        cc = min(r[5] for r in ctrl); nc = min(r[5] for r in new)
        print(f'\ncontrol(old) worst: maxabs {cm:.3e}  cos {cc:.9f}')
        print(f'new subjects worst: maxabs {nm:.3e}  cos {nc:.9f}')
        ok = (nc >= min(cc, 0.9999)) and (nm <= max(cm * 10, 1e-2))
        print('\nVERDICT:', 'PASS - new subjects match the encoder as well as the '
              'trusted control (deviation is float32 kernel noise)' if ok else
              'FAIL - new subjects deviate MORE than the trusted control')


if __name__ == '__main__':
    main()
