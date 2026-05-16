"""Plot EEG signal examples and instantaneous spectrum from TUEG and Alljoined-1.6M."""
import os
import sys
from glob import glob

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import stft

from datasets.tueg_dataset import preprocessing_recording as preprocess_tueg
from datasets.raw_eeg_dataset import preprocessing_recording as preprocess_alljoined

FS = 200  # all recordings resampled to 200 Hz
SECONDS = 10
N_CHANNELS = 8
OUT_DIR = "figs/eeg_examples"


def load_tueg():
    files = glob("./data/TUEG/v2.0.1/edf/**/*.edf", recursive=True)
    for f in files:
        raw = preprocess_tueg(f)
        if raw is not None:
            return raw, f
    raise RuntimeError("no usable TUEG file found")


def load_alljoined():
    files = sorted(glob("./data/Alljoined-1.6M/raw_eeg/**/*.edf", recursive=True))
    if not files:
        files = sorted(glob("./data/Alljoined-1.6M/raw_eeg/**/*.bdf", recursive=True))
    f = files[0]
    if f.endswith(".bdf"):
        raw = mne.io.read_raw_bdf(f, preload=True)
    else:
        raw = mne.io.read_raw_edf(f, preload=True)
    bad_prefixes = ("Timestamp", "OrTimestamp", "Counter", "Interpolated", "MOT.")
    drop = [c for c in raw.ch_names if c.startswith(bad_prefixes)]
    if drop:
        raw.drop_channels(drop)
    raw = preprocess_alljoined(raw)
    return raw, f


def plot_signals(ts, ch_names, fs, title, out_path):
    n = min(N_CHANNELS, ts.shape[0])
    n_samp = min(int(SECONDS * fs), ts.shape[1])
    t = np.arange(n_samp) / fs
    seg = ts[:n, :n_samp]

    spread = 4 * np.median(np.std(seg, axis=1)) + 1e-6
    offsets = np.arange(n) * spread

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n):
        ax.plot(t, seg[i] + offsets[i], lw=0.7, color=f"C{i % 10}")
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names[:n])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectrum(ts, ch_names, fs, title, out_path):
    n = min(N_CHANNELS, ts.shape[0])
    n_samp = min(int(SECONDS * fs), ts.shape[1])
    seg = ts[:n, :n_samp]

    nperseg = int(fs * 1.0)
    noverlap = int(nperseg * 0.9)

    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i in range(n):
        f, t, Z = stft(seg[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        P = 20 * np.log10(np.abs(Z) + 1e-8)
        mask = f <= 60
        im = axes[i].pcolormesh(t, f[mask], P[mask], shading="gouraud", cmap="magma")
        axes[i].set_ylabel(f"{ch_names[i]}\nHz", fontsize=8)
        fig.colorbar(im, ax=axes[i], pad=0.01, label="dB")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_one(name, loader):
    raw, file_path = loader()
    fs = int(raw.info["sfreq"])
    ts = (raw.get_data() * 1e6).astype(np.float32)
    ch_names = raw.ch_names
    print(f"[{name}] file={file_path} fs={fs} shape={ts.shape}")

    plot_signals(
        ts, ch_names, fs,
        title=f"{name} — {os.path.basename(file_path)}",
        out_path=os.path.join(OUT_DIR, f"{name}_signals.png"),
    )
    plot_spectrum(
        ts, ch_names, fs,
        title=f"{name} STFT spectrogram — {os.path.basename(file_path)}",
        out_path=os.path.join(OUT_DIR, f"{name}_spectrogram.png"),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mne.set_log_level("WARNING")
    run_one("TUEG", load_tueg)
    run_one("Alljoined-1.6M", load_alljoined)
    print(f"saved plots under {OUT_DIR}/")


if __name__ == "__main__":
    main()
