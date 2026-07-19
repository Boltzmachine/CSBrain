"""Falsifier: how much of diag_flip_discrim_acc = 0.955 is explainable WITHOUT EEG?

The flip-alignment target is ``_image_lateral_descriptor`` = ``_colband_pool(center=True)``
with ``flip_n_col_bands = 2`` (models/alignment.py:1787-1826). With nb=2 the two
column bands are centred against their own mean, so

    desc = [ (b_L - b_R)/2 , (b_R - b_L)/2 ] = [ u , -u ],     u = (b_L - b_R)/2

i.e. the 1536-d "descriptor" is a redundant embedding of ONE 768-d vector u. And
the mirrored frame swaps the bands, so ``desc_mirror ~= -desc_orig``: the
"hard negative" is the target's ANTIPODE.

Consequence (see the algebra in the docstring of scripts/probe_bilateral_split.py):
with a linear head the presented flip sign s = +-1 CANCELS out of the
discrimination criterion, which collapses to a one-bit sign test

    <W_lat . e_i , u_i>  >  0

So a DEGENERATE model whose lateral half is a constant, EEG-blind vector
``e_i == c`` still scores

    H = max_w  P_i[ <w, u_i> > 0 ]

and H is a property of the VIDEO ALONE -- no EEG, no model, no training. If
H ~= 0.95, then diag_flip_discrim_acc = 0.955 is fully accounted for by
egocentric-video left/right anisotropy and carries ZERO evidence that the
bilateralization split learned anything.

This script computes H (and the antipodality of the target) straight from the
cached DINOv2 patch grids.

Usage: python scripts/probe_flip_target_geometry.py --n_subjects 12
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CACHE = os.path.join(
    REPO, 'data/EgoBrain/cache_embeddings_grid_facebook_dinov2-base_g0.2_sz224')


def colband_u(grid, n_bands=2):
    """grid (N, 256, 768) -> u (N, 768) = the centred column-band descriptor's
    single free vector. Mirrors alignment.py::_colband_pool(center=True) exactly:
    reshape to (N, s, s, d) [row-major, so dim1=rows, dim2=cols], split the COLUMN
    axis into n_bands, mean-pool each, subtract the cross-band mean."""
    N, P, d = grid.shape
    s = int(round(P ** 0.5))
    assert s * s == P
    g = grid.reshape(N, s, s, d)
    w = s // n_bands
    bands = np.stack([g[:, :, k * w:(k + 1) * w, :].mean(axis=(1, 2))
                      for k in range(n_bands)], axis=1)          # (N, nb, d)
    bands = bands - bands.mean(axis=1, keepdims=True)
    return bands[:, 0]                                            # u = (b_L - b_R)/2


def max_positive_halfspace(U, iters=400, lr=0.5, seed=0):
    """H = max_w P[<w, u_i> > 0], via gradient ascent on a low-temperature
    sigmoid surrogate, restarted from the mean direction and the top PCs."""
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    X = torch.tensor(Un, dtype=torch.float32)

    def frac(w):
        return float(((X @ w) > 0).float().mean())

    starts = [X.mean(0)]
    Xc = X - X.mean(0)
    _, _, V = torch.pca_lowrank(Xc, q=4)
    for k in range(4):
        starts.append(V[:, k])
        starts.append(-V[:, k])

    best_w, best = None, -1.0
    for w0 in starts:
        w = (w0 / (w0.norm() + 1e-12)).clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for t in range(iters):
            tau = 0.5 * (0.02 / 0.5) ** (t / max(iters - 1, 1))   # anneal 0.5 -> 0.02
            loss = -torch.sigmoid((X @ w) / tau).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w /= (w.norm() + 1e-12)
        f = frac(w.detach())
        if f > best:
            best, best_w = f, w.detach().clone()
    return best, best_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_subjects', type=int, default=12)
    ap.add_argument('--per_subject', type=int, default=4000)
    ap.add_argument('--out', default=os.path.join(REPO, 'outputs/split_probe/flip_target_geometry.json'))
    args = ap.parse_args()

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    rng = np.random.default_rng(0)

    # The grid cache is 1.7 TB and each frame is 2*256*768 float32 = 1.5 MB, so
    # random-index reads thrash GPFS. Read CONTIGUOUS blocks and reduce to u on
    # the fly -- we only ever need the column-band means.
    U_o, U_m = [], []
    subs = sorted(f for f in os.listdir(CACHE) if f.endswith('.h5'))[:args.n_subjects]
    BLK = 256
    for fn in subs:
        got = 0
        with h5py.File(os.path.join(CACHE, fn), 'r') as f:
            hi = f['has_image'][:]
            n_slots = len(hi)
            # stride the file so the sample spans the whole session, but read
            # each hit as a contiguous block
            starts = list(range(0, n_slots - BLK, max(BLK, (n_slots // 24) or BLK)))
            for s0 in starts:
                if got >= args.per_subject:
                    break
                blk = f['grid'][s0:s0 + BLK]              # (BLK, 2, 256, 768)
                m = hi[s0:s0 + BLK]
                if not m.any():
                    continue
                blk = blk[m].astype(np.float32)
                U_o.append(colband_u(blk[:, 0]))
                U_m.append(colband_u(blk[:, 1]))
                got += len(blk)
        print(f"  {fn}: {got} frames", flush=True)
    U_o = np.concatenate(U_o)
    U_m = np.concatenate(U_m)
    n = len(U_o)
    print(f"total frames: {n}", flush=True)

    # 1. Is the hard negative the antipode of the target? (DINOv2 is not
    #    flip-equivariant, so desc_mirror need NOT equal -desc_orig.)
    cos_om = ((U_o * U_m).sum(1)
              / (np.linalg.norm(U_o, axis=1) * np.linalg.norm(U_m, axis=1) + 1e-12))

    # 2. THE DECISION VECTOR. diag_flip_discrim_acc asks whether the presented
    #    prediction overlaps the PRESENTED descriptor more than the OPPOSITE one:
    #        <p, d_present> > <p, d_opposite>   <=>   <p, d_present - d_opposite> > 0
    #    The descriptors are cosine-normalised in the diagnostic (alignment.py:
    #    2251-2257), so the decision vector is
    #        delta_i = normalize(u_orig_i) - normalize(u_mirror_i)
    #    up to the row's flip sign -- which CANCELS, because a flipped row
    #    presents both the mirrored frame AND the sign-negated lateral half.
    def nrm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

    D = nrm(U_o) - nrm(U_m)

    # 3. THE NUMBER: the ceiling for an EEG-BLIND CONSTANT lateral vector. With
    #    e_i == c the flip sign cancels and the model scores exactly
    #    P_i[<W_lat c, delta_i> > 0], whose maximum over c is H. H is a property
    #    of the VIDEO ALONE: no EEG, no model, no training.
    H, w = max_positive_halfspace(D)
    mu = D.mean(0)
    mu /= np.linalg.norm(mu) + 1e-12
    H_mean_dir = float(((D @ mu) > 0).mean())
    kappa_delta = float((D.mean(0) ** 2).sum() / (D ** 2).sum(1).mean())
    kappa_img = float((U_o.mean(0) ** 2).sum() / (U_o ** 2).sum(1).mean())

    res = dict(
        n_frames=int(n), n_subjects=len(subs),
        cos_orig_mirror_mean=float(cos_om.mean()),
        cos_orig_mirror_p05=float(np.percentile(cos_om, 5)),
        cos_orig_mirror_p95=float(np.percentile(cos_om, 95)),
        frac_cos_below_minus_0p9=float((cos_om < -0.9).mean()),
        kappa_img_constant_fraction=kappa_img,
        kappa_delta_constant_fraction=kappa_delta,
        H_max_positive_halfspace=float(H),
        H_mean_direction=H_mean_dir,
        observed_diag_flip_discrim_acc=0.9554,
    )
    res['explained_by_video_alone'] = bool(
        res['H_max_positive_halfspace'] >= 0.94)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
