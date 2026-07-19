"""The empirical judgement on diag_flip_discrim_acc = 0.955.

scripts/probe_flip_target_geometry.py showed that the flip-alignment DECISION
VECTOR delta_i = normalize(u_orig_i) - normalize(u_mirror_i) is so anisotropic
across egocentric video that a single fixed direction separates 99.98% of frames
(and even the plain mean direction gets 97.6%). That is an ALGEBRAIC argument
that an EEG-blind constant lateral half would score ~0.95+.

This script tests it EMPIRICALLY, with no algebra and no assumption about the
(nonlinear) frame_flip_align_proj: run the real checkpoint on real EgoBrain
frames, reproduce diag_flip_discrim_acc exactly as models/alignment.py:2250-2257
computes it, then SURGICALLY REPLACE the lateral half of global_rep with an
EEG-blind constant -- keeping only the architecturally-imposed flip sign -- and
recompute.

  real          the model as-is                             (expect ~0.955)
  const_mean    lateral half := s_i * mean_j(e_j)           EEG-blind
  const_opt     lateral half := s_i * c*, c* fitted to maximise discrim  EEG-blind
  const_rand    lateral half := s_i * (random fixed vector) EEG-blind
  zero          lateral half := 0                            (must collapse to 0.0:
                                                              strict '>' tie-break)

If const_mean / const_opt land at or above the real model, diag_flip_discrim_acc
carries ZERO evidence about the bilateralization split: it is a video prior.
"""
import argparse
import functools
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

from utils.util import load_pretrain_checkpoint, apply_arch_params  # noqa: E402
from models import get_model  # noqa: E402
from models.alignment import build_flip_perm_batch  # noqa: E402
from datasets.egobrain_dataset import EgoBrainDataset, collate_egobrain  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def discrim(pred_present, pred_opp, present_desc, opposite_desc):
    """models/alignment.py:2250-2257, verbatim."""
    pn = F.normalize(pred_present, dim=-1)
    po = F.normalize(pred_opp, dim=-1)
    dp = F.normalize(present_desc, dim=-1)
    do = F.normalize(opposite_desc, dim=-1)
    return 0.5 * (((pn * dp).sum(-1) > (pn * do).sum(-1)).float()
                  + ((po * do).sum(-1) > (po * dp).sum(-1)).float())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n_clips', type=int, default=3072)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    tag = args.tag or os.path.basename(os.path.dirname(args.ckpt))

    sd, saved = load_pretrain_checkpoint(args.ckpt)
    import argparse as _a
    p = _a.Namespace(model='Align', dropout=0.1)
    apply_arch_params(p, saved)
    enc = get_model(p, None, None)
    miss, unexp = enc.load_state_dict(
        {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')},
        strict=False)
    assert not miss, f"MISSING: {miss[:5]}"
    enc = enc.to(DEV).eval()
    half = enc.d_model // 2

    ds = EgoBrainDataset(
        data_dir=os.path.join(REPO, 'data/EgoBrain'),
        subjects=[f'P{i:04d}' for i in range(1, 13)],
        in_dim=p.in_dim, n_windows=1, window_s=1.0, stride_s=0.2, clip_s=4.0,
        erp_latency_s=-0.15, fs_out=200, max_channels=32,
        load_frames=True, frame_size=224,
        use_frame_grid=True, frame_grid_s=0.2, use_grid_embeddings=True,
        temporal_jitter=False, motion_resample=False)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(args.n_clips, len(ds)), replace=False)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx.tolist()), batch_size=args.batch_size,
        num_workers=0, collate_fn=functools.partial(collate_egobrain,
                                                    frame_objective=True))

    G_bi, E_lat, S, D_o, D_m = [], [], [], [], []
    g = torch.Generator(device='cpu').manual_seed(0)
    for batch in dl:
        hi = batch['has_image']
        if hi is None or not hi.any():
            continue
        x = (batch['timeseries'] / 100.0).to(DEV)
        cc = batch['ch_coords'].to(DEV)
        cn = batch['ch_names']
        vcm = batch['valid_channel_mask'].to(DEV)
        vlm = batch['valid_length_mask'].to(DEV)
        B = x.size(0)
        flip = (torch.rand(B, generator=g) < 0.5).to(DEV)

        # --- the model's own frame-averaging encoder, unrolled ---
        z = enc.patch_embedding(x, None)
        coord_pe, _ = enc._spherical_positional_encoding(cc)
        perm = build_flip_perm_batch(cn, vcm).to(DEV)
        Pz, _ = enc._build_Pz(z, coord_pe, perm, vcm)
        ez, vcm_g, vlm_g = enc._frame_add_context(z, cc, vcm, vlm)
        ep, _, _ = enc._frame_add_context(Pz, cc, vcm, vlm)
        a = enc._frame_run_layers(ez, enc._frame_backbone_layers, vcm_g, vlm_g)
        b = enc._frame_run_layers(ep, enc._frame_backbone_layers, vcm_g, vlm_g)
        Pa = enc._swap_channels(a, perm, channel_offset=1)
        Pb = enc._swap_channels(b, perm, channel_offset=1)
        inv = 0.5 * (a + b)
        fr = flip.view(B, 1, 1, 1)
        eq = torch.where(fr, 0.5 * (b - Pa), 0.5 * (a - Pb))
        bi_g = inv[:, 0, 0, :half]                      # (B, 20) bilateral half
        lat_g = eq[:, 0, 0, half:]                      # (B, 20) lateral half (PRESENTED)

        # --- the flip-align targets, exactly as the model builds them ---
        idxi = hi.nonzero(as_tuple=True)[0].to(DEV)
        d_o = enc._image_lateral_descriptor(
            grid=batch['frame_grid'].to(DEV).index_select(0, idxi))
        d_m = enc._image_lateral_descriptor(
            grid=batch['frame_grid_flip'].to(DEV).index_select(0, idxi))

        G_bi.append(bi_g.index_select(0, idxi).cpu())
        E_lat.append(lat_g.index_select(0, idxi).cpu())
        S.append(flip.index_select(0, idxi).cpu())
        D_o.append(d_o.cpu())
        D_m.append(d_m.cpu())

    G_bi = torch.cat(G_bi).to(DEV)
    E_lat = torch.cat(E_lat).to(DEV)
    S = torch.cat(S).to(DEV)
    D_o = torch.cat(D_o).to(DEV)
    D_m = torch.cat(D_m).to(DEV)
    n = G_bi.size(0)
    print(f"[{tag}] n={n} rows with images", flush=True)

    fr = S.view(-1, 1)
    present_desc = torch.where(fr, D_m, D_o)
    opposite_desc = torch.where(fr, D_o, D_m)
    proj = enc.frame_flip_align_proj

    def score(lat_presented):
        """lat_presented is the PRESENTED lateral half; the opposite rep is the
        same vector with the lateral half negated (alignment.py:2106-2112)."""
        rep_p = torch.cat([G_bi, lat_presented], dim=-1)
        rep_o = torch.cat([G_bi, -lat_presented], dim=-1)
        return float(discrim(proj(rep_p), proj(rep_o),
                             present_desc, opposite_desc).mean())

    res = {'tag': tag, 'ckpt': args.ckpt, 'n_rows': int(n)}
    res['real'] = score(E_lat)

    # The canonical (unflipped) lateral half, so a "constant" is defined in the
    # canonical frame and then re-signed by the row's presentation.
    sgn = torch.where(S, -1.0, 1.0).view(-1, 1)
    e_canon = E_lat * sgn                                # eq at flip=False
    c_mean = e_canon.mean(0, keepdim=True)
    res['const_mean'] = score(c_mean.expand(n, -1) * sgn)

    torch.manual_seed(0)
    c_rand = torch.randn(1, half, device=DEV)
    c_rand = c_rand / c_rand.norm() * c_mean.norm()
    res['const_rand'] = score(c_rand.expand(n, -1) * sgn)

    # c* : an EEG-blind constant fitted to MAXIMISE the discrimination.
    c = c_mean.clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=0.05)
    with torch.enable_grad():
        for _ in range(300):
            rep_p = torch.cat([G_bi, c.expand(n, -1) * sgn], dim=-1)
            rep_o = torch.cat([G_bi, -c.expand(n, -1) * sgn], dim=-1)
            pn = F.normalize(proj(rep_p), dim=-1)
            po = F.normalize(proj(rep_o), dim=-1)
            dp = F.normalize(present_desc, dim=-1)
            do = F.normalize(opposite_desc, dim=-1)
            m1 = (pn * dp).sum(-1) - (pn * do).sum(-1)
            m2 = (po * do).sum(-1) - (po * dp).sum(-1)
            loss = -(torch.sigmoid(m1 / 0.02) + torch.sigmoid(m2 / 0.02)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    res['const_opt'] = score(c.detach().expand(n, -1) * sgn)
    res['zero'] = score(torch.zeros_like(E_lat))

    # how EEG-dependent is the real lateral half at all?
    res['lat_kappa_constant_fraction'] = float(
        (e_canon.mean(0) ** 2).sum() / (e_canon ** 2).sum(1).mean())
    res['lat_var_frac_of_global_rep'] = float(
        E_lat.var(0).sum() / (E_lat.var(0).sum() + G_bi.var(0).sum()))

    res['VERDICT_diagnostic_is_eeg_blind'] = bool(
        res['const_opt'] >= res['real'] - 0.01)

    out = args.out or os.path.join(REPO, 'outputs/split_probe',
                                   f'flipdiscrim_ablation_{tag}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
