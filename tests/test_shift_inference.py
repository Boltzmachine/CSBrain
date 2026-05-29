"""Inference probe: how does the prediction shift when the input
time-series is rolled by k samples along the time axis?

Loads a finetuned PhysioNet checkpoint, picks a few test samples (one per
class), runs the model on the same sample under a fine grid of integer
time-shifts, and saves a probability-vs-shift figure. Each shift is
np.roll along the flat time axis (64 ch, 800 samples = 4*200). With
--use_initial_segment_only the encoder only sees the first seq_len
patches (10 patches * 40 samples = 400 samples), so a small shift slides
the visible window inside the 800-sample buffer.
"""
import os
import sys
from argparse import Namespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import lmdb
import pickle

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from utils.util import load_pretrain_checkpoint, apply_arch_params  # noqa: E402
from datasets.cached_dataset import _to_spherical  # noqa: E402
from models import model_for_physio  # noqa: E402


FOUNDATION_DIR = 'outputs/Align-time-volume/epoch5_loss0.03542475029826164.pth'
CKPT_PATH = 'outputs/CSBrain/finetune_CSBrain_PhysioNet/epoch35_acc_0.50913_kappa_0.34553_f1_0.50915.pth'
DATA_DIR = 'data/preprocessed/physionet_mi'

CLASS_NAMES = ['L_fist', 'R_fist', 'both_fists', 'both_feet']
CLASS_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
FIGURE_OUT = 'outputs/shift_inference_probs.png'
SAMPLE_RATE_HZ = 200  # PhysioNet-MI is resampled to 200 Hz in preprocessing


def build_params(foundation_dir):
    p = Namespace(
        model='Align',
        downstream_dataset='PhysioNet-MI',
        datasets_dir=DATA_DIR,
        num_of_classes=4,
        foundation_dir=foundation_dir,
        use_pretrained_weights=False,  # we'll load the finetune ckpt directly
        use_initial_segment_only=True,
        segment_forward=False,
        dropout=0.3,
        linear_probe=False,
        use_finetune_weights=False,
        use_lora=False,
        image_mode='raw',
        image_size=0,
        stft_n_fft=64,
        stft_hop_length=16,
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8,
        need_mask=True, mask_ratio=0.5,
        TemEmbed_kernel_sizes="[(1,), (3,), (5,),]",
        use_CrossTemEmbed=False, use_SmallerToken=False,
        use_CSBrainTF=False, use_CSBrainTF_Tep_Spa=False,
        use_CSBrainTF_Tep_Bra=False, use_CSBrainTF_Tep_Bra_Tiny=False,
        use_CSBrainTF_Tep_Bra_Pal=False, use_IntraBraEmbed=False,
        project_to_source=False, num_sources=32,
        patch_embed_type='cnn', mamba_band_periods=None,
        n_mamba_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        vision_encoder='facebook/dinov2-base', image_pool_heads=4,
        use_volume_conduction=True, vc_tau_init=0.08,
        spectral_mode='instantaneous', stft_hop=16,
        use_llm_vq=False, num_language_tokens=0, max_llm_codebook_size=0,
        hemisphere_flip_aug=False,
    )
    _, saved = load_pretrain_checkpoint(foundation_dir)
    apply_arch_params(p, saved)
    return p


def make_batch(sample_dict, shifts, device):
    """Roll the 64x(4*200) flat time series by each k in `shifts` and stack.

    Returns a batch with shape (len(shifts), 64, 4, 200).
    """
    if np.isscalar(shifts):
        shifts = [int(shifts)]
    data = sample_dict['sample'] / 100.0          # (64, 4, 200)
    ch, S, P = data.shape
    flat = data.reshape(ch, S * P)

    stacked = np.empty((len(shifts), ch, S, P), dtype=np.float32)
    for i, shift in enumerate(shifts):
        rolled = np.roll(flat, int(shift), axis=-1) if shift else flat
        stacked[i] = rolled.reshape(ch, S, P)

    x = torch.from_numpy(stacked)
    ch_coords = _to_spherical(sample_dict['ch_coords']).unsqueeze(0).repeat(len(shifts), 1, 1)
    return {
        'x': x.to(device),
        'ch_coords': ch_coords.to(device),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = build_params(FOUNDATION_DIR)
    print(f"Device: {device}")
    print(f"Encoder seq_len={params.seq_len}, in_dim={params.in_dim} "
          f"-> visible window = {params.seq_len * params.in_dim} samples")

    model = model_for_physio.Model(params).to(device)
    model.eval()

    # Open the LMDB and pick one sample per class from the test split
    db = lmdb.open(DATA_DIR, readonly=True, lock=False)
    with db.begin(write=False) as txn:
        keys = pickle.loads(txn.get('__keys__'.encode()))['test']
    samples = {0: None, 1: None, 2: None, 3: None}
    with db.begin(write=False) as txn:
        for k in keys:
            s = pickle.loads(txn.get(k.encode()))
            if samples[s['label']] is None:
                samples[s['label']] = s
            if all(v is not None for v in samples.values()):
                break

    # Materialize LazyLinear in the classifier with a single dummy forward
    with torch.no_grad():
        dummy = make_batch(samples[0], shifts=[0], device=device)
        _ = model(dict(dummy))

    # Now load the finetune checkpoint over the freshly-built model
    sd = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing={len(missing)}  e.g. {missing[:3]}")
    if unexpected:
        print(f"[load] unexpected={len(unexpected)}  e.g. {unexpected[:3]}")
    model.eval()

    # Fine grid of single-sample shifts inside the visible-window range.
    # Visible window is seq_len*in_dim = 400 samples = 2 s @ 200 Hz; scan a
    # symmetric ±half range with step=1 so the curve is continuous.
    half = params.seq_len * params.in_dim  # = 400
    shifts = np.arange(-half // 2, half // 2 + 1, 1)  # ±200 samples = ±1 s
    chunk = 32  # CPU forward batch

    results = {}
    for cls, s in samples.items():
        if s is None:
            print(f"\n=== No test sample found for class {cls} ===")
            continue
        print(f"\n=== True class: {cls} ({CLASS_NAMES[cls]}) ===  "
              f"scanning {len(shifts)} shifts in chunks of {chunk}")
        probs_grid = np.zeros((len(shifts), len(CLASS_NAMES)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(shifts), chunk):
                end = min(start + chunk, len(shifts))
                batch = make_batch(s, shifts=shifts[start:end].tolist(), device=device)
                logits = model(dict(batch))
                probs_grid[start:end] = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs_grid.argmax(axis=1)
        n_correct = int((preds == cls).sum())
        print(f"argmax==true on {n_correct}/{len(shifts)} shifts "
              f"({100*n_correct/len(shifts):.1f}%)")
        results[cls] = (probs_grid, preds)

    # --- Plot ----------------------------------------------------------------
    n_cls = len(results)
    fig, axes = plt.subplots(n_cls, 1, figsize=(11, 2.5 * n_cls), sharex=True)
    if n_cls == 1:
        axes = [axes]
    shifts_ms = shifts / SAMPLE_RATE_HZ * 1000.0  # samples -> ms

    for ax, (cls, (probs_grid, preds)) in zip(axes, sorted(results.items())):
        for c in range(len(CLASS_NAMES)):
            lw = 2.2 if c == cls else 1.0
            ax.plot(shifts_ms, probs_grid[:, c], color=CLASS_COLORS[c],
                    label=CLASS_NAMES[c], linewidth=lw,
                    alpha=1.0 if c == cls else 0.7)
        # mark regions where argmax != true class
        wrong = preds != cls
        ax.fill_between(shifts_ms, 0, 1, where=wrong, color='gray', alpha=0.10,
                        step='mid', linewidth=0)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel('softmax prob')
        ax.set_title(f'True class: {cls} ({CLASS_NAMES[cls]})  '
                     f'— bold line = true class; gray bands = argmax wrong')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel(f'time shift (ms, np.roll along time axis; '
                        f'visible window = {params.seq_len*params.in_dim} '
                        f'samples = {params.seq_len*params.in_dim/SAMPLE_RATE_HZ*1000:.0f} ms)')
    axes[0].legend(loc='upper right', ncol=4, fontsize=9)
    fig.suptitle('Predicted class probabilities vs. time-shift\n'
                 f'ckpt = {os.path.basename(CKPT_PATH)}', y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE_OUT), exist_ok=True)
    fig.savefig(FIGURE_OUT, dpi=140, bbox_inches='tight')
    print(f"\nFigure saved to {FIGURE_OUT}")


if __name__ == '__main__':
    main()
