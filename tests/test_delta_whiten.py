"""Unit tests for the ME->MI rank-1 delta-whitening transform
(utils.util.apply_delta_whiten) and its wiring into EgoBrainDataset."""
import numpy as np
import pytest

from utils.util import apply_delta_whiten

FS = 200


def _rel_delta(x, fs=FS):
    """Montage-avg relative delta (0.5-4 Hz) power, % of 0.5-45 Hz."""
    from scipy.signal import welch
    f, P = welch(x, fs=fs, nperseg=min(200, x.shape[-1]), axis=-1)
    P = P.mean(0)
    band = lambda lo, hi: np.trapz(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])
    tot = band(0.5, 45)
    return float(band(0.5, 4) / tot)


def _make_signal(C=32, T=800, seed=0):
    """Delta-dominated synthetic EEG: strong 2 Hz + weak broadband (ME-like)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FS
    delta = 8.0 * np.sin(2 * np.pi * 2.0 * t)            # big delta
    mid = 1.0 * np.sin(2 * np.pi * 20.0 * t)             # small beta
    noise = 0.5 * rng.standard_normal((C, T))
    return (delta[None] + mid[None] + noise).astype(np.float32)


def test_g0_one_is_identity():
    x = _make_signal()
    out = apply_delta_whiten(x, FS, g0=1.0, cutoff_hz=8.0)
    assert out is x or np.allclose(out, x)


def test_g0_ge_one_and_bad_cutoff_are_noops():
    x = _make_signal()
    assert np.allclose(apply_delta_whiten(x, FS, 1.5, 8.0), x)
    assert np.allclose(apply_delta_whiten(x, FS, 0.6, 0.0), x)
    assert np.allclose(apply_delta_whiten(x, FS, None, 8.0), x)


def test_attenuates_relative_delta():
    x = _make_signal()
    out = apply_delta_whiten(x, FS, g0=0.6, cutoff_hz=8.0)
    assert _rel_delta(out) < _rel_delta(x) - 0.02, "delta should drop materially"


def test_stronger_gain_attenuates_more():
    x = _make_signal()
    d_mild = _rel_delta(apply_delta_whiten(x, FS, 0.8, 8.0))
    d_strong = _rel_delta(apply_delta_whiten(x, FS, 0.4, 8.0))
    assert d_strong < d_mild, "lower g0 must attenuate delta more"


def test_preserves_per_channel_rms_and_shape():
    x = _make_signal()
    out = apply_delta_whiten(x, FS, g0=0.65, cutoff_hz=8.0)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert np.isfinite(out).all()
    pre = np.sqrt((x ** 2).mean(-1))
    post = np.sqrt((out ** 2).mean(-1))
    assert np.allclose(pre, post, rtol=1e-4), "per-channel RMS must be preserved"


def test_leaves_high_band_untouched():
    """Power at/above the cutoff is unchanged (only the <cutoff ramp scales)."""
    x = _make_signal()
    out = apply_delta_whiten(x, FS, g0=0.5, cutoff_hz=8.0)
    f = np.fft.rfftfreq(x.shape[-1], d=1.0 / FS)
    Xf = np.abs(np.fft.rfft(x, axis=-1))
    Of = np.abs(np.fft.rfft(out, axis=-1))
    # RMS preservation rescales everything by a global per-channel factor; the
    # RATIO of high-band to a high-band reference bin must stay constant, i.e.
    # the >cutoff shape is preserved. Check bins well above cutoff (>=12 Hz).
    hi = f >= 12.0
    rx = Xf[:, hi] / (Xf[:, hi].sum(-1, keepdims=True) + 1e-9)
    ro = Of[:, hi] / (Of[:, hi].sum(-1, keepdims=True) + 1e-9)
    assert np.allclose(rx, ro, atol=1e-3), "high-band spectral shape must be intact"


def test_egobrain_dataset_accepts_args():
    """The dataset __init__ accepts the new kwargs without constructing data."""
    import inspect
    from datasets.egobrain_dataset import EgoBrainDataset
    sig = inspect.signature(EgoBrainDataset.__init__)
    assert 'delta_whiten_g0' in sig.parameters
    assert 'delta_whiten_cutoff_hz' in sig.parameters
    assert sig.parameters['delta_whiten_g0'].default == 1.0
