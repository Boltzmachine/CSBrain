"""Adjudication: proper nulls for the ONE surviving positive readout
(stage_swap: subject-held-out LDA on the lateral half), plus the missing
CONSTANT-GATE control (same trained backbone, gate replaced by its scalar mean).

Outputs to stdout only. No files under models/ or datasets/ are touched.
"""
import importlib.util
import sys
import time

import numpy as np
import torch

ROOT = '/gpfs/radev/pi/ying_rex/wq44/CSBrain'
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    'pbs', ROOT + '/scripts/probe_bilateral_split.py')
pbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbs)
torch.set_num_threads(4)

CK = ROOT + '/outputs/wm-new-gradneg/epoch10_loss2.3018484115600586.pth'
CACHE = ROOT + '/outputs/split_probe/enc_%s.npz'
print('DEV', pbs.DEV, flush=True)


def encode_all(tag, gate_const=None, random_init=False):
    import os
    f = CACHE % tag
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files}
    bb, p, info = pbs.build_backbone(CK, random_init=random_init)
    print(tag, 'built', info, flush=True)
    if gate_const is not None:
        fs = bb.frame_split
        g = float(gate_const)

        def const_forward(x, coord_pe, valid_channel_mask=None):
            gate = torch.full_like(x, g)
            if valid_channel_mask is not None:
                m = valid_channel_mask.to(x.dtype).view(
                    *valid_channel_mask.shape, 1, 1)
                gate = gate * m
            return gate * x, gate
        fs.forward = const_forward
        print(tag, 'gate forced constant', g, flush=True)
    out = {}
    t = time.time()
    for split, kw in (('tr', dict(mode='train')), ('te', dict(mode='test')),
                      ('tef', dict(mode='test', anat_flip=True))):
        ld = pbs.physio_loader(p, kw['mode'])
        X, y, s = pbs.encode_split(bb, ld, anat_flip=kw.get('anat_flip', False))
        out['X' + split], out['y' + split], out['s' + split] = X, y, s
        print(tag, split, X.shape, '%.0fs' % (time.time() - t), flush=True)
    np.savez(f, **out)
    return out


def lda_dir(X, y, A, B):
    ia, ib = np.isin(y, A), np.isin(y, B)
    Xa, Xb = X[ia], X[ib]
    d = Xa.mean(0) - Xb.mean(0)
    na, nb = len(Xa), len(Xb)
    Sw = ((na - 1) * np.cov(Xa, rowvar=False)
          + (nb - 1) * np.cov(Xb, rowvar=False)) / (na + nb - 2)
    Sw = np.atleast_2d(Sw)
    Sw += 1e-2 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, d)
    thr = 0.5 * (Xa @ w).mean() + 0.5 * (Xb @ w).mean()
    return w, thr


def acc_vec(X, y, sl, w, thr, A, B):
    m = np.isin(y, A + B)
    pred = np.where(X[m][:, sl] @ w > thr, A[0], B[0])
    truth = np.where(np.isin(y[m], A), A[0], B[0])
    return (pred == truth).astype(float), m


def report(tag, E, n_perm=200, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr, str_ = E['Xtr'], E['ytr'], E['str']
    Xte, yte, ste = E['Xte'], E['yte'], E['ste']
    Xtef, ytef = E['Xtef'], E['ytef']
    half = Xtr.shape[1] // 2
    print('\n===== %s  (train %d trials/%d subj, test %d/%d) lat_var_frac=%.2e'
          % (tag, len(ytr), len(set(str_)), len(yte), len(set(ste)),
             Xte[:, half:].var(0).sum() / Xte.var(0).sum()), flush=True)
    for name, sl in (('lat', slice(half, None)), ('bi', slice(0, half))):
        for axis, (A, B) in (('LR_0v1', ([0], [1])),
                             ('handfeet_2v3', ([2], [3]))):
            w, thr = lda_dir(Xtr[:, sl], ytr, A, B)
            c0, m = acc_vec(Xte, yte, sl, w, thr, A, B)
            c1, _ = acc_vec(Xtef, ytef, sl, w, thr, A, B)
            a0, a1 = c0.mean(), c1.mean()
            subj = ste[m]
            # subject-clustered bootstrap on the test subjects
            us = np.unique(subj)
            bs = []
            for _ in range(2000):
                pick = rng.choice(us, len(us), replace=True)
                bs.append(np.concatenate([c0[subj == u] for u in pick]).mean())
            bs = np.array(bs)
            # label-permutation null: shuffle TRAIN labels within subject,
            # refit LDA, evaluate on the untouched test set
            null = []
            for _ in range(n_perm):
                yp = ytr.copy()
                for u in np.unique(str_):
                    i = np.where(str_ == u)[0]
                    yp[i] = ytr[i][rng.permutation(len(i))]
                wp, tp = lda_dir(Xtr[:, sl], yp, A, B)
                null.append(acc_vec(Xte, yte, sl, wp, tp, A, B)[0].mean())
            null = np.array(null)
            print('  %-4s %-13s acc %.4f  [clustered 95%% CI %.3f-%.3f]  '
                  'perm-null %.4f+-%.4f  p=%.3f | anatflip %.4f  inversion %+.4f'
                  % (name, axis, a0, np.percentile(bs, 2.5),
                     np.percentile(bs, 97.5), null.mean(), null.std(),
                     (null >= a0).mean() if a0 > 0.5 else float('nan'),
                     a1, a0 - a1), flush=True)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'trained'
    if which == 'trained':
        report('gradneg ep10 (learned gate)', encode_all('grad10'))
    elif which == 'constgate':
        report('gradneg ep10 (gate FORCED CONSTANT 0.0725)',
               encode_all('grad10_cg', gate_const=0.0725))
    elif which == 'randinit':
        report('random-init', encode_all('randinit', random_init=True))
