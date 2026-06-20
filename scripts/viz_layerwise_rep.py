"""Visualise whether each encoder layer's internal representation still
"looks like a time series" once the patch axis is flattened.

For the WorldModel checkpoint at ``outputs/recon_wm``:
  * the encoder is a CSBrainAlign with d_model=40, seq_len(N)=5 patches,
    in_dim=40 raw samples / patch -> a 1 s / 200-sample window.
  * we hook every transformer layer, grab the per-(channel, patch) token
    (B, C, N, d), and for one representative channel flatten (N, d)->N*d.
  * the raw input window flattened (N, in_dim)->N*in_dim is the TRUE
    time series, used as the reference row.

Run:
  conda run -n cbramod python scripts/viz_layerwise_rep.py \
      --ckpt outputs/recon_wm
Outputs PNGs under outputs/figs/.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from argparse import Namespace
from models import get_model
from utils.util import load_pretrain_checkpoint
from datasets.egobrain_dataset import EgoBrainDataset


def latest_epoch_ckpt(ckpt_arg):
    """Accept a dir (pick the highest-epoch .pth) or a file path."""
    if os.path.isdir(ckpt_arg):
        cands = glob.glob(os.path.join(ckpt_arg, "epoch*_loss*.pth"))
        if not cands:
            raise FileNotFoundError(f"no epoch*.pth in {ckpt_arg}")
        def ep(p):
            m = re.search(r"epoch(\d+)_", os.path.basename(p))
            return int(m.group(1)) if m else -1
        return max(cands, key=ep)
    return ckpt_arg


def build_batch(params, n_samples=8, subject="P0001"):
    """One EgoBrain window batch (no video frames)."""
    ds = EgoBrainDataset(
        data_dir=params.egobrain_root,
        subjects=[subject],
        in_dim=params.in_dim,
        n_windows=1,
        window_s=params.egobrain_window_s,
        stride_s=params.egobrain_stride_s,
        clip_s=params.egobrain_clip_s,
        erp_latency_s=params.egobrain_erp_latency_s,
        load_frames=False,                # <- no GoPro decode needed
        max_channels=params.egobrain_max_channels,
        ea_matrices=None,                 # raw signal, no EA whitening
    )
    idxs = list(range(0, min(n_samples, len(ds))))
    samples = [ds[i] for i in idxs]
    # All windows from one subject share the channel set -> equal C.
    ts = torch.stack([s["timeseries"][0] for s in samples])       # (B, C, N, d)
    ch_coords = torch.stack([s["ch_coords"] for s in samples])    # (B, C, 3)
    ch_names = samples[0]["ch_names"]
    return ts, ch_coords, ch_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/recon_wm")
    ap.add_argument("--subject", default="P0001")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--outdir", default="outputs/figs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = latest_epoch_ckpt(args.ckpt)
    print(f"[viz] checkpoint: {ckpt_path}")

    state_dict, saved = load_pretrain_checkpoint(ckpt_path, map_location="cpu")
    assert saved is not None, "checkpoint has no saved params"
    params = Namespace(**saved)

    # ---- model ----
    model = get_model(params, brain_regions=[], sorted_indices=[])
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoder.")]
    assert not enc_missing, f"encoder keys missing: {enc_missing[:8]}"
    print(f"[viz] loaded. (missing={len(missing)}, unexpected={len(unexpected)}; "
          f"encoder fully matched)")
    encoder = model.encoder.eval().to(device)

    # ---- input ----
    ts, ch_coords, ch_names = build_batch(params, args.n_samples, args.subject)
    ts = ts.to(device) / 100.0          # trainer feeds timeseries / 100
    ch_coords = ch_coords.to(device)
    B, C, N, d_in = ts.shape
    vcm = torch.ones(B, C, dtype=torch.bool, device=device)
    print(f"[viz] batch: B={B} C={C} N={N} in_dim={d_in}, channels={ch_names}")

    # ---- hook every transformer layer ----
    layers = encoder.encoder.layers
    captured = [None] * len(layers)

    def mk_hook(i):
        def hook(_m, _inp, out):
            captured[i] = out.detach().float().cpu()
        return hook

    handles = [layers[i].register_forward_hook(mk_hook(i))
               for i in range(len(layers))]

    with torch.no_grad():
        encoder({"timeseries": ts, "ch_coords": ch_coords,
                 "valid_channel_mask": vcm}, mask=None, encoder_only=True)
    for h in handles:
        h.remove()

    # layer output is (B, C+1, N+1, d) with a global channel/token prepended.
    reps = []
    for t in captured:
        if t.shape[1] == C + 1 and t.shape[2] == N + 1:
            t = t[:, 1:, 1:, :]          # drop global row/col -> (B, C, N, d)
        reps.append(t.numpy())
    d_model = reps[0].shape[-1]
    L = len(reps)
    print(f"[viz] captured {L} layers, token rep (C,N,d)=({C},{N},{d_model})")

    # ---- pick a representative sample + channel ----
    ts_np = ts.cpu().numpy()
    s = int(np.argmax(ts_np.reshape(B, -1).var(axis=1)))   # most active window
    pref = ["C3", "CZ", "Cz", "C4", "CP3", "FC3"]
    ch = next((ch_names.index(p) for p in pref if p in ch_names), None)
    if ch is None:
        ch = int(np.argmax(ts_np[s].reshape(C, -1).var(axis=1)))
    ch_label = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
    print(f"[viz] sample={s}, channel={ch_label} (idx {ch})")

    # Per-layer (and raw) flattened trace for that channel.
    raw_flat = ts_np[s, ch].reshape(-1)                    # (N*in_dim,) true TS
    layer_flat = [r[s, ch].reshape(-1) for r in reps]      # each (N*d_model,)

    def z(v):
        v = v - v.mean()
        sd = v.std()
        return v / sd if sd > 1e-8 else v

    # ================= Figure 1: flattened-patch traces =================
    rows = L + 1
    fig, axes = plt.subplots(rows, 1, figsize=(11, 1.55 * rows), sharex=False)
    # raw row
    ax = axes[0]
    ax.plot(z(raw_flat), color="k", lw=0.9)
    for b in range(1, N):
        ax.axvline(b * d_in, color="r", ls=":", lw=0.7, alpha=0.6)
    ax.set_ylabel("raw\ninput", rotation=0, ha="right", va="center", fontsize=9)
    ax.set_title(f"recon_wm — flatten patches -> 1D trace | subject {args.subject}, "
                 f"channel {ch_label}, sample {s}  (red dotted = patch boundary; "
                 f"each trace z-scored)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # layer rows
    for i, lf in enumerate(layer_flat):
        ax = axes[i + 1]
        ax.plot(z(lf), color="C0", lw=0.9)
        for b in range(1, N):
            ax.axvline(b * d_model, color="r", ls=":", lw=0.7, alpha=0.6)
        ax.set_ylabel(f"layer\n{i+1}", rotation=0, ha="right", va="center",
                      fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].set_xlabel("flattened index  =  patch 0 [d features] | patch 1 [d] | ... "
                        f"({N} patches x {d_model} dims)", fontsize=9)
    fig.tight_layout()
    f1 = os.path.join(args.outdir, "recon_wm_layerwise_flattened_trace.png")
    fig.savefig(f1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {f1}")

    # ================= Figure 2: per-layer heatmaps + continuity =========
    ncol = 5
    nrow = int(np.ceil((L + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.2 * nrow))
    axes = np.atleast_2d(axes).ravel()
    # raw: (N, in_dim) -> show patches as rows
    a = axes[0]
    a.imshow(ts_np[s, ch], aspect="auto", cmap="RdBu_r",
             interpolation="nearest")
    a.set_title("raw input\n(patch x time-sample)", fontsize=8)
    a.set_ylabel("patch"); a.set_xlabel("sample")
    for i, r in enumerate(reps):
        a = axes[i + 1]
        m = r[s, ch]                                       # (N, d_model)
        vmax = np.abs(m).max() + 1e-8
        a.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 interpolation="nearest")
        a.set_title(f"layer {i+1}\n(patch x feature)", fontsize=8)
        a.set_xticks([]); a.set_yticks(range(N))
    for j in range(L + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"recon_wm — per-layer token rep, channel {ch_label}: each row = one "
                 f"patch's {d_model}-d embedding (does it evolve smoothly across patches?)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    f2 = os.path.join(args.outdir, "recon_wm_layerwise_heatmap.png")
    fig.savefig(f2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {f2}")

    # ============ Figure 3: adjacent-patch continuity metric =============
    # Cosine similarity between consecutive patch embeddings, averaged over
    # patches and channels. High -> tokens drift slowly across time
    # (time-series-like continuity); low -> abstract / patch-localised.
    def adj_cos(arr):                      # arr (B,C,N,D)
        a = arr[:, :, :-1, :]
        b = arr[:, :, 1:, :]
        num = (a * b).sum(-1)
        den = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
        return float((num / den).mean())

    raw_cos = adj_cos(ts_np[:, :, :, :])   # raw uses in_dim as feature
    layer_cos = [adj_cos(r) for r in reps]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(range(L + 1))
    ax.plot(xs, [raw_cos] + layer_cos, "o-", color="C3")
    ax.set_xticks(xs)
    ax.set_xticklabels(["raw"] + [str(i + 1) for i in range(L)])
    ax.set_xlabel("layer"); ax.set_ylabel("mean adjacent-patch cosine sim")
    ax.set_title("Temporal continuity of internal rep across depth\n"
                 "(higher = consecutive patches stay similar = more time-series-like)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = os.path.join(args.outdir, "recon_wm_layerwise_continuity.png")
    fig.savefig(f3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {f3}")
    print(f"[viz] adjacent-patch cosine: raw={raw_cos:.3f} -> "
          f"L1={layer_cos[0]:.3f} ... L{L}={layer_cos[-1]:.3f}")


if __name__ == "__main__":
    main()
