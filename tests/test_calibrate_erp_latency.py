"""Unit tests for scripts/calibrate_erp_latency.py pure functions.

The single most important test here is ``test_recovery_*``: a SYNTHETIC
END-TO-END recovery of a KNOWN injected EEG<->hand lag through the actual
estimator. If the estimator cannot recover a lag it planted itself, nothing
else it reports is trustworthy -- so this is parametrised over both signs.

  conda run -n cbramod python -m pytest tests/test_calibrate_erp_latency.py -x -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scripts.calibrate_erp_latency import (            # noqa: E402
    BANDS, FS, GRID_S, WINDOW_S,
    delta_to_erp_latency, erp_latency_to_delta, snap_erp_to_grid,
    power_cumsum, windowed_mean_power, centered_ma_highpass, centroid_peak,
    make_synthetic_subject, estimate_lag_from_eeg,
)


# --------------------------------------------------------------------------- #
# (a) Delta <-> erp_latency_s conversion, both code paths, hand-computed table
# --------------------------------------------------------------------------- #
def test_delta_to_erp_grid_path_hand_table():
    # GRID (live) path: erp = Delta + window_s/2 = Delta + 0.5.
    table = {
        -0.65: -0.15,   # the CURRENTLY SHIPPED grid config
        0.0: 0.5,       # Delta*=0 fallback (frame centred on window)
        0.5: 1.0,
        -1.0: -0.5,
        0.25: 0.75,
    }
    for delta, erp in table.items():
        assert delta_to_erp_latency(delta, WINDOW_S, 'grid') == pytest.approx(erp)


def test_delta_to_erp_clip_path_hand_table():
    # LEGACY clip path: erp = Delta (no window_s/2 term).
    for delta in (-0.65, -0.15, 0.0, 0.3, 1.2):
        assert delta_to_erp_latency(delta, WINDOW_S, 'clip') == pytest.approx(delta)


def test_paths_differ_by_half_window():
    for delta in (-1.0, -0.15, 0.0, 0.7):
        g = delta_to_erp_latency(delta, WINDOW_S, 'grid')
        c = delta_to_erp_latency(delta, WINDOW_S, 'clip')
        assert (g - c) == pytest.approx(WINDOW_S / 2.0)


def test_erp_to_delta_roundtrip():
    for erp in (-0.15, 0.0, 0.5, -0.6):
        for path in ('grid', 'clip'):
            d = erp_latency_to_delta(erp, WINDOW_S, path)
            assert delta_to_erp_latency(d, WINDOW_S, path) == pytest.approx(erp)
    # current knob semantics: erp=-0.15 on grid == physical Delta=-0.65.
    assert erp_latency_to_delta(-0.15, WINDOW_S, 'grid') == pytest.approx(-0.65)
    assert erp_latency_to_delta(-0.15, WINDOW_S, 'clip') == pytest.approx(-0.15)


def test_snap_erp_to_grid():
    assert snap_erp_to_grid(-0.15, FS) == pytest.approx(-0.15)   # already on grid
    assert snap_erp_to_grid(0.123, FS) == pytest.approx(0.125)   # 24.6 -> 25/200
    assert snap_erp_to_grid(0.5, FS) == pytest.approx(0.5)
    # snapped value is always an integer number of samples.
    for v in (-0.333, 0.017, 0.9999):
        assert (snap_erp_to_grid(v, FS) * FS) == pytest.approx(
            round(snap_erp_to_grid(v, FS) * FS))


def test_bad_path_raises():
    with pytest.raises(ValueError):
        delta_to_erp_latency(0.0, WINDOW_S, 'nope')
    with pytest.raises(ValueError):
        erp_latency_to_delta(0.0, WINDOW_S, 'nope')


# --------------------------------------------------------------------------- #
# (b) cumsum windowed-mean == direct np.mean, incl. near array edges
# --------------------------------------------------------------------------- #
def test_windowed_mean_equals_direct_mean():
    rng = np.random.default_rng(0)
    C, T = 3, 5000
    x = rng.standard_normal((C, T)) * 7.0 + 2.0
    cs = power_cumsum(x)                       # cumsum of x**2
    p = x ** 2
    w = 100
    # centres fully inside, in samples.
    centers = rng.integers(w, T - w, size=200).astype(float)
    mean, inbounds = windowed_mean_power(cs, centers, w)
    assert inbounds.all()
    for j, c in enumerate(centers):
        i0 = int(round(c - w / 2.0))
        i1 = i0 + w
        direct = p[:, i0:i1].mean(axis=1)
        assert np.allclose(mean[:, j], direct, rtol=1e-9, atol=1e-6)


def test_windowed_mean_fractional_center():
    # fractional (non-integer) sample centre must round to the same window np.mean uses.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 2000))
    cs = power_cumsum(x)
    p = x ** 2
    w = 51
    for c in (500.3, 500.7, 999.5, 1234.49):
        mean, ib = windowed_mean_power(cs, np.array([c]), w)
        i0 = int(round(c - w / 2.0))
        i1 = i0 + w
        assert ib[0]
        assert np.allclose(mean[:, 0], p[:, i0:i1].mean(axis=1), atol=1e-8)


def test_windowed_mean_edge_flags():
    x = np.random.default_rng(2).standard_normal((1, 1000))
    cs = power_cumsum(x)
    w = 100
    # left edge, right edge, and interior.
    centers = np.array([10.0, 5.0, 500.0, 995.0, 999.0])
    _, inbounds = windowed_mean_power(cs, centers, w)
    expected = np.array([
        (10 - 50) >= 0,           # False
        (5 - 50) >= 0,            # False
        (500 - 50 >= 0) and (500 - 50 + 100 <= 1000),   # True
        (995 - 50 + 100 <= 1000),  # False
        (999 - 50 + 100 <= 1000),  # False
    ])
    assert (inbounds == expected).all()


def test_power_cumsum_is_float64_and_shape():
    x = np.ones((4, 10), dtype=np.float32)
    cs = power_cumsum(x)
    assert cs.dtype == np.float64
    assert cs.shape == (4, 11)
    assert cs[0, 0] == 0.0
    assert cs[0, -1] == pytest.approx(10.0)


def test_centered_ma_highpass_removes_slow_preserves_fast():
    n = 2000
    t = np.arange(n)
    slow = 5.0 * np.sin(2 * np.pi * t / 800.0)         # very slow
    fast = np.sin(2 * np.pi * t / 8.0)                 # fast
    hp = centered_ma_highpass(slow + fast, M=50)
    # slow largely removed, fast largely kept (compare interior to avoid edges).
    core = slice(200, -200)
    assert np.std(hp[core]) < np.std((slow + fast)[core])
    assert np.corrcoef(hp[core], fast[core])[0, 1] > 0.9


def test_centered_ma_highpass_nan_aware():
    x = np.arange(100.0)
    x[40:45] = np.nan
    hp = centered_ma_highpass(x, M=10)
    assert np.isnan(hp[40:45]).all()          # nans pass through
    assert np.isfinite(hp[:40]).all()         # neighbours not poisoned
    assert np.isfinite(hp[45:]).all()


# --------------------------------------------------------------------------- #
# centroid peak abort guards
# --------------------------------------------------------------------------- #
def test_centroid_recovers_gaussian_center():
    lags = np.round(np.arange(-2, 2.001, 0.05), 4)
    g = np.exp(-0.5 * ((lags - 0.4) / 0.3) ** 2)
    pk = centroid_peak(lags, g)
    assert not pk['aborted']
    assert pk['centroid'] == pytest.approx(0.4, abs=0.05)


def test_centroid_aborts_on_edge_peak():
    lags = np.round(np.arange(-2, 2.001, 0.05), 4)
    g = np.exp(-0.5 * ((lags - 1.95) / 0.2) ** 2)   # peak at the sweep edge
    pk = centroid_peak(lags, g)
    assert pk['aborted']


# --------------------------------------------------------------------------- #
# (c) THE LINCHPIN: synthetic end-to-end recovery of a KNOWN injected lag.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('delta_true', [-0.6, -0.3, 0.0, 0.3, 0.6])
def test_recovery_injected_lag(delta_true):
    """Plant a lag Delta_true between EEG mu-band power and hand intensity, then
    assert the estimator recovers it to within one lag step. Parametrised over
    both signs so a sign flip cannot hide."""
    lag_step = 0.1
    lags = np.round(np.arange(-1.5, 1.5 + 1e-9, lag_step), 4)
    sub = make_synthetic_subject(delta_true, n_slots=3500, snr=4.0, slow_frac=0.5,
                                 seed=123, band='mu')
    est = estimate_lag_from_eeg(sub['eeg'], sub['y_slot'], sub['usable'], lags,
                                bands=('mu',), alpha=10.0, hp_M=50, K=5,
                                embargo_slots=25)
    # curve must have a real peak (not aborted) and localise the injected lag.
    assert np.isfinite(est['r']).any(), 'estimator produced an all-NaN curve'
    peak = est['peak']
    est_delta = peak['centroid']
    if not np.isfinite(est_delta):
        # fall back to the argmax if the centroid guard fired, but it must exist.
        est_delta = peak['argmax']
    assert np.isfinite(est_delta)
    err = abs(est_delta - delta_true)
    assert err <= lag_step + 1e-9, (
        f'injected Delta={delta_true}, recovered {est_delta:.3f} '
        f'(argmax={peak["argmax"]}, err={err:.3f} > one lag step {lag_step})')


LAGS_STD = np.round(np.arange(-1.5, 1.5 + 1e-9, 0.1), 4)


def _recover_delta(eeg, y_slot, usable, off_est, lags=LAGS_STD):
    """Run the estimator with a given estimator-side label_offset_s and return the
    recovered lag (centroid, falling back to argmax if the centroid guard fired)."""
    est = estimate_lag_from_eeg(eeg, y_slot, usable, lags,
                                bands=('mu',), alpha=10.0, hp_M=50, K=5,
                                embargo_slots=25, label_offset_s=off_est)
    assert np.isfinite(est['r']).any(), 'estimator produced an all-NaN curve'
    c = est['peak']['centroid']
    if not np.isfinite(c):
        c = est['peak']['argmax']
    assert np.isfinite(c)
    return float(c)


def _forward_diff_subject(delta_true, n_slots=3500, snr=4.0, seed=7, band='mu'):
    """Faithful model of the FORWARD/RAW hand-label cache (format_version 2).

    Unlike ``make_synthetic_subject`` (whose ``label_offset_s`` is a free knob that
    CANCELS when generator and estimator share it), this builds the label the way
    the extractor does: a continuous instantaneous speed ``v(t)`` sampled at ``fs``,
    then ``label[s] = mean(v over [s, s+1]*grid_s)`` -- the RAW forward pair, no
    smoothing, last slot NaN (no forward pair). The forward window's first moment
    sits at its MIDPOINT ``(s+0.5)*grid_s``, so the physically-correct label offset
    ``+0.5*grid_s`` is not injected by hand -- it EMERGES from the forward-difference
    construction. The EEG band power at time ``tau`` tracks ``v(tau + delta_true)``
    (EEG leads movement by ``delta_true``). A matched estimator must therefore need
    offset ``+0.5*grid_s`` to recover ``delta_true``; a smaller/zero offset must
    undershoot by exactly the missing fraction of a slot."""
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(seed)
    sps = int(round(GRID_S * FS))                 # samples per slot
    T = n_slots * sps
    t = np.arange(T) / FS
    v = uniform_filter1d(rng.standard_normal(T), size=sps)   # ~1-slot smooth
    v = v - v.min() + 0.05                                    # strictly positive
    label = np.full(n_slots, np.nan)
    for s in range(n_slots - 1):                  # last slot: no forward pair -> NaN
        label[s] = v[s * sps:(s + 1) * sps].mean()
    y_slot = np.log1p(label)
    usable = np.isfinite(y_slot)
    env = np.clip(np.interp(t + delta_true, t, v, left=v[0], right=v[-1]), 0, None)
    lo, hi = BANDS[band]
    fc = 0.5 * (lo + hi)
    carrier = np.sin(2 * np.pi * fc * t + rng.uniform(0, 2 * np.pi))
    sig = np.sqrt(env) * carrier
    noise = rng.standard_normal(T) / max(snr, 1e-3)
    eeg = (sig + noise)[None, :].astype(np.float32) * 10.0
    return {'eeg': eeg, 'y_slot': y_slot, 'usable': usable}


@pytest.mark.parametrize('delta_true', [-0.4, 0.0, 0.4])
def test_recovery_with_forward_offset(delta_true):
    """Forward/raw cache adds a +0.5-slot (+0.1 s) label-time offset. Inject the
    SAME offset on BOTH the generator and the estimator and assert recovery within
    one lag step -- proving the label_offset_s plumbing is self-consistent and does
    NOT reintroduce a bias (a mismatched sign would shift the recovered lag by the
    offset and fail this).

    NOTE: this is only a SELF-CONSISTENCY check; because the offset cancels
    (recovered = delta_true + off_est - off_gen) it says nothing about whether
    +0.1 is the correct VALUE/SIGN. That is covered by
    ``test_forward_difference_offset_value_and_sign`` (semantics) and
    ``test_dropped_estimator_offset_is_caught`` (mismatch)."""
    lag_step = 0.1
    off = 0.5 * GRID_S                          # +0.1 s, the forward-cache offset
    assert off == pytest.approx(0.1)
    sub = make_synthetic_subject(delta_true, n_slots=3500, snr=4.0, slow_frac=0.5,
                                 seed=321, band='mu', label_offset_s=off)
    est_delta = _recover_delta(sub['eeg'], sub['y_slot'], sub['usable'], off)
    err = abs(est_delta - delta_true)
    assert err <= lag_step + 1e-9, (
        f'injected Delta={delta_true} w/ offset {off}, recovered {est_delta:.3f} '
        f'(err={err:.3f} > one lag step {lag_step})')


@pytest.mark.parametrize('delta_true', [-0.4, 0.0, 0.4])
def test_dropped_estimator_offset_is_caught(delta_true):
    """FINDING-1 GUARD. The generator carries the true forward-cache offset
    (+0.5 slot) but the ESTIMATOR drops it (off_est=0). Recovery must then be
    biased by exactly one MISSING half-slot: recovered = delta_true - 0.5*grid_s.

    The old self-consistency test tolerated |err| <= lag_step (0.1) -- exactly the
    offset magnitude -- so a fully dropped offset landed on the threshold and one
    parametrized case leaked through. Here we assert the bias EQUALS -0.5*grid_s to
    a TIGHT tolerance (0.03 s, << the 0.1 s effect), so a dropped/half-applied
    offset fails by a ~0.07 s margin instead of hiding in centroid noise. We also
    assert the correctly-offset estimator lands on delta_true and, crucially, that
    ADDING the offset shifts the recovered lag to the RIGHT (pins the sign)."""
    off = 0.5 * GRID_S
    sub = make_synthetic_subject(delta_true, n_slots=3500, snr=4.0, slow_frac=0.5,
                                 seed=321, band='mu', label_offset_s=off)
    rec_with = _recover_delta(sub['eeg'], sub['y_slot'], sub['usable'], off)
    rec_without = _recover_delta(sub['eeg'], sub['y_slot'], sub['usable'], 0.0)
    # correctly-offset estimator is unbiased.
    assert abs(rec_with - delta_true) < 0.03, (
        f'matched offset should recover {delta_true}, got {rec_with:.3f}')
    # dropped offset undershoots by exactly one half-slot -- caught with big margin.
    assert abs(rec_without - (delta_true - off)) < 0.03, (
        f'dropped offset should bias to {delta_true - off:.3f}, got {rec_without:.3f}')
    # sign: adding the +0.5-slot offset moves the recovered lag MORE POSITIVE.
    assert (rec_with - rec_without) == pytest.approx(off, abs=0.03), (
        f'offset must enter with +1 gain: rec_with-rec_without={rec_with-rec_without:.3f} '
        f'!= off={off:.3f}')


@pytest.mark.parametrize('delta_true', [-0.4, 0.0, 0.4])
def test_forward_difference_offset_value_and_sign(delta_true):
    """FINDING-2 GUARD. Validate the +0.5-slot offset VALUE and SIGN against genuine
    forward-difference cache semantics -- not just plumbing self-consistency.

    ``_forward_diff_subject`` builds the label as the RAW forward-pair mean speed
    (the extractor's format_version-2 convention); the correct offset therefore
    EMERGES from the forward window's midpoint rather than being injected. We assert:
      * off = +0.5*grid_s (what detect_label_offset returns for this cache) recovers
        the injected lag unbiased;
      * off = 0.0 (the legacy value) UNDERSHOOTS by exactly one half-slot -- so a
        globally-wrong-but-consistent offset (finding 2's failure mode) is now
        detectable, and the correct value is pinned to 0.5*grid_s, not e.g. 0.15."""
    off = 0.5 * GRID_S
    assert off == pytest.approx(0.1)
    sub = _forward_diff_subject(delta_true, n_slots=3500, snr=4.0, seed=7)
    rec_correct = _recover_delta(sub['eeg'], sub['y_slot'], sub['usable'], off)
    rec_legacy = _recover_delta(sub['eeg'], sub['y_slot'], sub['usable'], 0.0)
    # the forward-cache offset recovers the true lag.
    assert abs(rec_correct - delta_true) < 0.03, (
        f'forward-diff cache needs off={off}: recovered {rec_correct:.3f} '
        f'for injected {delta_true}')
    # dropping it undershoots by exactly the half-slot (sign + magnitude pinned).
    assert abs(rec_legacy - (delta_true - off)) < 0.03, (
        f'legacy off=0 must undershoot by {off}: recovered {rec_legacy:.3f} '
        f'for injected {delta_true} (expected {delta_true - off:.3f})')
    assert rec_correct > rec_legacy, 'adding the offset must shift recovery right'


def test_recovery_sign_is_not_flipped():
    """A strong positive vs strong negative injection must land on opposite sides
    of zero -- the crudest guard against a sign convention bug."""
    lags = np.round(np.arange(-1.5, 1.5 + 1e-9, 0.1), 4)
    pos = make_synthetic_subject(0.6, 3500, snr=4.0, seed=7, band='mu')
    neg = make_synthetic_subject(-0.6, 3500, snr=4.0, seed=7, band='mu')
    ep = estimate_lag_from_eeg(pos['eeg'], pos['y_slot'], pos['usable'], lags,
                               bands=('mu',), alpha=10.0, hp_M=50, K=5,
                               embargo_slots=25)
    en = estimate_lag_from_eeg(neg['eeg'], neg['y_slot'], neg['usable'], lags,
                               bands=('mu',), alpha=10.0, hp_M=50, K=5,
                               embargo_slots=25)
    dp = ep['peak']['argmax']
    dn = en['peak']['argmax']
    assert dp > 0.2 and dn < -0.2, f'sign flip: pos->{dp}, neg->{dn}'
