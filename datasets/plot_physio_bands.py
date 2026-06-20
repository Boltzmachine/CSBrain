"""Plot PhysioNet-MI samples decomposed into canonical EEG frequency bands.

Loads a few samples from the PhysioNet-MI LMDB, splits each per-channel signal
into the standard bands via rFFT band-passing (same idea as
``utils.util.bandpass_decompose``), and produces:

  1. ``physio_bands_traces.png`` -- time-domain band traces for the motor
     channels (C3/Cz/C4), one column per MI class.
  2. ``physio_bands_power.png``  -- mean band power per band/class, averaged
     over all channels of a handful of samples.

Run:
    conda run -n cbramod python -m datasets.plot_physio_bands \
        --data_dir data/preprocessed/physionet_mi --out_dir outputs/figs
"""
import argparse
import os
import pickle

import lmdb
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PhysioNet-MI: 200 Hz, 4 s epochs stored as (C, n_windows=4, T=200).
FS = 200.0

# Label map (see datasets/physio_dataset.py).
LABELS = {0: "left fist", 1: "right fist", 2: "both fists", 3: "both feet"}

# Canonical EEG bands (Hz). mu/alpha and beta carry the motor-imagery signal.
BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha/mu", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 100.0),
]
MOTOR_CHANNELS = ["C3", "Cz", "C4"]


def bandpass(sig, fs, lo, hi):
    """rFFT band-pass a (..., T) array, zeroing bins outside [lo, hi)."""
    spec = np.fft.rfft(sig, axis=-1)
    freqs = np.fft.rfftfreq(sig.shape[-1], d=1.0 / fs)
    keep = (freqs >= lo) & (freqs < hi)
    return np.fft.irfft(spec * keep, n=sig.shape[-1], axis=-1)


def load_samples(data_dir, mode="train"):
    db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True)
    with db.begin(write=False) as txn:
        keys = pickle.loads(txn.get(b"__keys__"))[mode]
        out = []
        for k in keys:
            p = pickle.loads(txn.get(k.encode()))
            # (C, n_windows, T) -> (C, n_windows*T) continuous signal.
            x = np.asarray(p["sample"], dtype=np.float64)
            C = x.shape[0]
            out.append(
                {
                    "key": k,
                    "x": x.reshape(C, -1),
                    "label": int(p["label"]),
                    "ch_names": list(p["ch_names"]),
                }
            )
    return out


def pick_one_per_class(samples):
    chosen = {}
    for s in samples:
        chosen.setdefault(s["label"], s)
        if len(chosen) == len(LABELS):
            break
    return [chosen[c] for c in sorted(chosen)]


def plot_traces(samples, out_path):
    """Rows = broadband + each band; cols = MI class. Motor channels overlaid."""
    rows = 1 + len(BANDS)
    cols = len(samples)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 1.7 * rows), sharex=True, squeeze=False
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(MOTOR_CHANNELS)))

    for j, s in enumerate(samples):
        ch_idx = {n.lower(): i for i, n in enumerate(s["ch_names"])}
        picks = [(c, ch_idx[c.lower()]) for c in MOTOR_CHANNELS if c.lower() in ch_idx]
        t = np.arange(s["x"].shape[-1]) / FS

        # Row 0: broadband.
        ax = axes[0][j]
        for (name, ci), col in zip(picks, colors):
            ax.plot(t, s["x"][ci], lw=0.7, color=col, label=name)
        ax.set_title(f"{LABELS[s['label']]}\n({s['key']})", fontsize=9)
        if j == 0:
            ax.set_ylabel("broadband", fontsize=8)
        if j == cols - 1:
            ax.legend(fontsize=7, loc="upper right")

        # Remaining rows: bands.
        for r, (bname, lo, hi) in enumerate(BANDS, start=1):
            ax = axes[r][j]
            for (name, ci), col in zip(picks, colors):
                ax.plot(t, bandpass(s["x"][ci], FS, lo, hi), lw=0.7, color=col)
            if j == 0:
                ax.set_ylabel(f"{bname}\n{lo:g}-{hi:g} Hz", fontsize=8)
            if r == rows - 1:
                ax.set_xlabel("time (s)", fontsize=8)

    fig.suptitle("PhysioNet-MI: motor channels (C3/Cz/C4) by frequency band",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path)


def plot_power(samples_by_class, out_path, n_per_class=20):
    """Mean log band power per class, averaged over channels and samples."""
    labels = sorted(samples_by_class)
    band_names = [b[0] for b in BANDS]
    power = np.zeros((len(labels), len(BANDS)))

    for li, c in enumerate(labels):
        subset = samples_by_class[c][:n_per_class]
        acc = np.zeros(len(BANDS))
        for s in subset:
            for bi, (_, lo, hi) in enumerate(BANDS):
                bp = bandpass(s["x"], FS, lo, hi)  # (C, T)
                acc[bi] += np.mean(bp ** 2)
        power[li] = acc / max(1, len(subset))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(BANDS))
    w = 0.8 / len(labels)
    for li, c in enumerate(labels):
        ax.bar(x + li * w, np.log10(power[li] + 1e-12), w, label=LABELS[c])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"{n}\n{lo:g}-{hi:g}Hz" for n, lo, hi in BANDS], fontsize=8)
    ax.set_ylabel("log10 mean band power")
    ax.set_title(f"PhysioNet-MI band power by class (mean of {n_per_class} samples/class)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/preprocessed/physionet_mi")
    ap.add_argument("--out_dir", default="outputs/figs")
    ap.add_argument("--mode", default="train")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.data_dir, args.mode)
    print(f"loaded {len(samples)} samples ({args.mode})")

    one_per_class = pick_one_per_class(samples)
    plot_traces(one_per_class, os.path.join(args.out_dir, "physio_bands_traces.png"))

    by_class = {}
    for s in samples:
        by_class.setdefault(s["label"], []).append(s)
    plot_power(by_class, os.path.join(args.out_dir, "physio_bands_power.png"))


if __name__ == "__main__":
    main()
