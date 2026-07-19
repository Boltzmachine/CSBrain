"""Empirically calibrate ``--egobrain_erp_latency_s`` as a LAG-IDENTIFIABILITY
study, not a point estimate.

The pretraining knob ``erp_latency_s`` sets the temporal offset between an EEG
window and the egocentric video frame it is paired with. Its shipped value
(-0.15 on the live GRID path) was chosen arbitrarily and -- because the grid path
is START-anchored, not centre-anchored -- actually ships a PHYSICAL lag of
``Delta = erp - window_s/2 = -0.65 s`` (EEG lagging movement by 0.65 s, which is
physiologically backwards for a motor code). This script chooses the knob
empirically by finding the lag ``Delta := frame_time - eeg_window_centre`` at
which EEG best decodes the CONTINUOUS hand-movement intensity derived from the
video.

Sign convention (pinned everywhere):
  * Delta > 0  => EEG LEADS movement   (motor preparation / anticipatory ERD)
  * Delta < 0  => EEG LAGS movement    (visual re-afference / post-movement rebound)
  * Delta ~ 0  => movement-locked artifact (head motion, cable sway, EMG)

Knob conversion (audit-unanimous, report BOTH paths):
  erp_latency_s[GRID , the live path] = Delta* + window_s/2   (= Delta* + 0.5)
  erp_latency_s[LEGACY clip path]     = Delta*

The pilot measurements (see the task brief) strongly imply the honest verdict is
NOT IDENTIFIABLE: the raw target's decodability is a slow-drift confound that is
lag-INVARIANT, and the only lag-carrying (high-passed) component has effect size
r~0.025 against a null of r~0.017. This script's deliverable is therefore a
POWERED DECISION -- it either localises the lag with a null-cleared,
subject-bootstrapped CI, or it proves the lag is not identifiable on this cache
and refuses to move the knob. It can and will print NOT IDENTIFIABLE.

Estimators implemented (all pooled over subjects, subject is the replication unit):
  * (9a) high-passed continuous-regression lag curve r(Delta) with a genuinely-
         zero circular-shift null re-run through identical blocked CV + embargo.
  * (7a) a multi-lag TRF on band x channel-group envelopes (joint fit) as a
         better-powered cross-check; peak read from the smooth |w|(lag) kernel.
  * (6)  a 5-band x 5-group centroid+z matrix for the confound gate.
  * (9c) an identifiability / power analysis + parametric-bootstrap recovery sim.
  * (9d) a visual-motion POSITIVE CONTROL run through the identical pipeline.
  * (9b) a compact event-locked peri-onset ERD decoder (diagnostic).

Everything is pure numpy/scipy/sklearn/h5py/matplotlib and runs with no GPU:

  # full run (sbatch this; activate cbramod FIRST -- sh scripts don't self-activate)
  conda run -n cbramod python -m scripts.calibrate_erp_latency

  # fast smoke (~2 min): exercises the whole path end to end
  conda run -n cbramod python -m scripts.calibrate_erp_latency \
      --subjects P0001,P0002 --lag_step 0.1 --slot_stride 10

Per-subject filtered-power cumsum arrays cache under --cache_dir so re-runs with
a different lag grid are cheap. The heavy null / bootstrap loops parallelise over
subjects (default workers=8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

# Cap BLAS threads BEFORE numpy imports its backend -- otherwise each of the N
# parallel workers spawns a full BLAS thread pool and oversubscribes the box to a
# crawl. Must precede `import numpy`. 2 threads/worker x 8 workers matches the 16
# CPUs of the target job while keeping the per-subject ridge matmuls fast;
# override with EEG_CAL_BLAS_THREADS if the pool size differs.
_BLAS_THREADS = os.environ.get('EEG_CAL_BLAS_THREADS', '2')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, _BLAS_THREADS)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# --------------------------------------------------------------------------- #
# Constants -- montage / bands / geometry. Kept module-level so the unit tests
# and the pipeline share one definition.
# --------------------------------------------------------------------------- #
FS = 200                       # cache sample rate (Hz)
GRID_S = 0.2                   # hand-label / frame grid step (s)
WINDOW_S = 1.0                 # deployment EEG window (sh/pretrain_worldmodel.sh)

# Band -> (lo, hi) Hz. No band above ~45 Hz: pilot P2 shows the cache is
# low-passed ~45 Hz (55-95 Hz power ~1e4x down) so a high-gamma / EMG band is
# empty and would only add noise.
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'mu': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'lowgamma': (30.0, 45.0),
}
BAND_ORDER = ['delta', 'theta', 'mu', 'beta', 'lowgamma']

# Sharp-variant analysis windows (s): delta needs ~2 cycles so it gets a 1.0 s
# window; the faster bands get 0.5 s to halve the lag smear.
W_SHARP = {'delta': 1.0, 'theta': 0.5, 'mu': 0.5, 'beta': 0.5, 'lowgamma': 0.5}
W_DEPLOY = {b: 1.0 for b in BAND_ORDER}   # robustness variant matching deployment

# Channel groups (spec Sec.6). Only names present in a subject's montage are used.
GROUPS = {
    'central': ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6',
                'CP1', 'CP2', 'CP5', 'CP6'],
    'occipital': ['O1', 'O2', 'Oz', 'PO9', 'PO10'],
    'frontal': ['Fp1', 'Fp2', 'F7', 'F8', 'Fz', 'F3', 'F4'],
    'temporal': ['T7', 'T8', 'FT9', 'FT10'],
    'parietal': ['P3', 'P4', 'Pz', 'P7', 'P8'],
}
GROUP_ORDER = ['central', 'occipital', 'frontal', 'temporal', 'parietal']

EPS = 1e-8


# =========================================================================== #
# Section 1 -- Delta <-> erp_latency_s conversion (PURE, unit-tested)
# =========================================================================== #
def delta_to_erp_latency(delta_s: float, window_s: float = WINDOW_S,
                         path: str = 'grid') -> float:
    """Physical lag ``Delta`` (frame_time - eeg_window_centre) -> the flag value.

    GRID (live) path: frame sits at window START + erp, so the frame-to-centre
    offset is ``erp - window_s/2`` => ``erp = Delta + window_s/2``.
    LEGACY clip path: frame sits at window CENTRE + erp => ``erp = Delta``.
    """
    if path == 'grid':
        return float(delta_s + window_s / 2.0)
    if path == 'clip':
        return float(delta_s)
    raise ValueError(f"path must be 'grid' or 'clip', got {path!r}")


def erp_latency_to_delta(erp_s: float, window_s: float = WINDOW_S,
                         path: str = 'grid') -> float:
    """Inverse of :func:`delta_to_erp_latency`."""
    if path == 'grid':
        return float(erp_s - window_s / 2.0)
    if path == 'clip':
        return float(erp_s)
    raise ValueError(f"path must be 'grid' or 'clip', got {path!r}")


def snap_erp_to_grid(erp_s: float, fs: int = FS) -> float:
    """Snap ``erp`` so ``erp*fs`` is an integer (the frame-slot integer identity
    the grid path relies on). fs=200 => a 0.005 s grid."""
    return float(int(round(erp_s * fs)) / fs)


# =========================================================================== #
# Section 3 -- O(1)-per-lag cumsum windowed power (PURE, unit-tested)
# =========================================================================== #
def bandpass_filtfilt(x: np.ndarray, fs: int, lo: float, hi: float,
                      order: int = 4) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass over the LAST axis, reflect-
    padded. Zero-phase is mandatory: any feature-filter group delay would add
    directly to the estimated Delta*."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.9999)
    b, a = butter(order, [lo_n, hi_n], btype='band')
    padlen = 3 * max(len(a), len(b))
    n = x.shape[-1]
    padlen = min(padlen, n - 1) if n > 1 else 0
    return filtfilt(b, a, x, axis=-1, padtype='odd', padlen=padlen)


def power_cumsum(filtered: np.ndarray) -> np.ndarray:
    """(C, T) filtered signal -> (C, T+1) float64 cumulative sum of instantaneous
    power (filtered**2), with a leading 0 so a windowed sum over samples ``[i, j)``
    is ``P[:, j] - P[:, i]``. float64 is non-negotiable: T~1.5M samples of large
    uV^2 values overflow float32 precision and bias the windowed means."""
    p = np.asarray(filtered, dtype=np.float64) ** 2
    cs = np.empty((p.shape[0], p.shape[1] + 1), dtype=np.float64)
    cs[:, 0] = 0.0
    np.cumsum(p, axis=1, out=cs[:, 1:])
    return cs


def windowed_mean_power(cumsum: np.ndarray, center_samples: np.ndarray,
                        w_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """O(1) mean power over ``[c - w/2, c + w/2)`` for each centre in
    ``center_samples`` (float sample index), from a :func:`power_cumsum` array.

    Returns ``(mean (C, n) float64, inbounds (n,) bool)``. ``inbounds`` is False
    where the window support is not fully inside ``[0, T)``; those columns are
    computed with clipped indices (finite, but must be masked by the caller)."""
    T = cumsum.shape[1] - 1
    c = np.asarray(center_samples, dtype=np.float64)
    i0 = np.round(c - w_samples / 2.0).astype(np.int64)
    i1 = i0 + int(w_samples)
    inbounds = (i0 >= 0) & (i1 <= T)
    i0c = np.clip(i0, 0, T)
    i1c = np.clip(i1, 0, T)
    total = cumsum[:, i1c] - cumsum[:, i0c]        # (C, n)
    return total / float(w_samples), inbounds


# =========================================================================== #
# Section 5 -- slow-drift high-pass + blocked CV (PURE, unit-tested via recovery)
# =========================================================================== #
def centered_ma_highpass(x: np.ndarray, M: int) -> np.ndarray:
    """Subtract a CENTRED, symmetric ``M``-slot moving average along axis 0.

    Linear-phase / zero group delay so it cannot shift a lag peak (a causal EMA
    would -- forbidden). NaN-aware: the baseline at each slot averages only the
    finite entries in its window, so NaN targets do not poison neighbours; NaNs
    pass through in the output and are masked downstream. ``x`` may be (n,) or
    (n, F)."""
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[:, None]
    finite = np.isfinite(x).astype(np.float64)
    xf = np.where(np.isfinite(x), x, 0.0)
    half = M // 2
    # cumulative sums with a leading zero -> box-sum via difference.
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(xf, axis=0)], 0)
    cn = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(finite, axis=0)], 0)
    n = x.shape[0]
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    ssum = cs[hi] - cs[lo]
    scnt = cn[hi] - cn[lo]
    baseline = np.where(scnt > 0, ssum / np.maximum(scnt, 1.0), 0.0)
    out = x - baseline
    out[~np.isfinite(x)] = np.nan
    return out[:, 0] if single else out


def _ridge_solve(Xtr: np.ndarray, ytr: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge weights (no intercept; caller centres y)."""
    F = Xtr.shape[1]
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += alpha
    return np.linalg.solve(A, Xtr.T @ ytr)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def blocked_cv_r(X: np.ndarray, y: np.ndarray, slot_idx: np.ndarray,
                 alpha: float, K: int = 8, embargo_slots: int = 50) -> float:
    """Within-subject contiguous-block CV Pearson r (spec Sec.5.3-A).

    Splits the (time-sorted) samples into ``K`` contiguous blocks; for each held-
    out block, trains on the rest MINUS an ``embargo_slots`` guard band on either
    side (kills label AC + residual slow-drift leakage), standardises X on the
    TRAIN fold only, scores r within the held-out block after removing that
    block's own mean, then Fisher-z averages across blocks weighted by block N."""
    n = len(y)
    if n < K + 5:
        return np.nan
    edges = np.linspace(0, n, K + 1).astype(int)
    F = X.shape[1]
    zs, ws = [], []
    for k in range(K):
        te = np.arange(edges[k], edges[k + 1])
        if te.size < 5:
            continue
        lo = slot_idx[te[0]] - embargo_slots
        hi = slot_idx[te[-1]] + embargo_slots
        tr = np.where((slot_idx < lo) | (slot_idx > hi))[0]
        if tr.size < F + 2:
            continue
        mu = X[tr].mean(0)
        sd = X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        ytr = y[tr] - y[tr].mean()
        w = _ridge_solve(Xtr, ytr, alpha)
        pred = Xte @ w
        r = _pearson(pred, y[te])
        if np.isfinite(r):
            zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
            ws.append(te.size)
    if not zs:
        return np.nan
    zbar = np.average(zs, weights=ws)
    return float(np.tanh(zbar))


def fisher_mean(rs) -> float:
    """Fisher-z mean of a set of correlations (ignoring NaNs)."""
    rs = np.asarray([r for r in rs if np.isfinite(r)], dtype=np.float64)
    if rs.size == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))


# =========================================================================== #
# Section 7c -- blur-unbiased peak estimation (PURE)
# =========================================================================== #
def centroid_peak(lags: np.ndarray, g: np.ndarray) -> dict:
    """Above-half-max centroid of a lag curve with the spec's abort guards.

    A symmetric even blur kernel preserves the first moment, so the centroid is
    unbiased for Delta* whereas the argmax (mode) wanders. Returns a dict with
    ``centroid``, ``argmax``, ``parabola`` (vertex through the 5 points nearest
    the max), ``aborted`` (bool) and ``reason``."""
    lags = np.asarray(lags, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    out = {'centroid': np.nan, 'argmax': np.nan, 'parabola': np.nan,
           'aborted': True, 'reason': ''}
    if not np.isfinite(g).any():
        out['reason'] = 'all-nan curve'
        return out
    gg = np.where(np.isfinite(g), g, -np.inf)
    imax = int(np.argmax(gg))
    out['argmax'] = float(lags[imax])
    M = gg[imax]
    tail = np.isfinite(g) & (np.abs(lags) > 1.5)
    g0 = float(np.median(g[tail])) if tail.any() else 0.0
    if M <= g0:
        out['reason'] = 'peak below baseline'
        return out
    thr = g0 + (M - g0) / 2.0
    support = np.isfinite(g) & (g > thr)
    if not support.any():
        out['reason'] = 'empty support'
        return out
    wgt = (g[support] - g0)
    cen = float(np.sum(wgt * lags[support]) / np.sum(wgt))
    out['centroid'] = cen
    # parabolic vertex through 5 points nearest the max.
    lo = max(0, imax - 2)
    hi = min(len(lags), imax + 3)
    xs, ys = lags[lo:hi], g[lo:hi]
    if len(xs) >= 3 and np.isfinite(ys).all():
        try:
            a, b, _ = np.polyfit(xs, ys, 2)
            if a < 0:
                out['parabola'] = float(-b / (2 * a))
        except Exception:                                       # noqa: BLE001
            pass
    # abort guards.
    sup_lags = lags[support]
    if sup_lags.min() <= lags.min() + 1e-9 or sup_lags.max() >= lags.max() - 1e-9:
        out['reason'] = 'support touches sweep edge'
        return out
    if (sup_lags.max() - sup_lags.min()) > 1.5:
        out['reason'] = 'support spans > 1.5 s'
        return out
    # multimodal check: support must be a single contiguous run.
    idx = np.where(support)[0]
    if np.any(np.diff(idx) > 1):
        # allow a single 1-slot gap; more => multimodal.
        if np.sum(np.diff(idx) > 1) > 1 or np.max(np.diff(idx)) > 2:
            out['reason'] = 'multimodal support'
            return out
    out['aborted'] = False
    return out


# =========================================================================== #
# Per-subject feature extraction (parallel worker)
# =========================================================================== #
def _clip_path(cache_dir: str, sub: str, c: int) -> str:
    return os.path.join(cache_dir, sub, f'{c}.npy')


def load_continuous_eeg(data_dir: str, sub: str) -> tuple[np.ndarray, list, dict]:
    """Stitch a subject's per-clip EEG .npy files into one contiguous (C, T)
    float32 array (uV). The EEG is one continuous EDF per subject, so clips are
    contiguous with no internal discontinuity. Returns ``(eeg, ch_names, meta)``;
    ``ch_names`` is the FULL clip montage (channel selection happens later)."""
    cache_dir = os.path.join(data_dir, f'cache_eeg_{FS}hz')
    with open(os.path.join(cache_dir, sub, 'clips.json')) as f:
        meta = json.load(f)
    n_clips = int(meta['n_clips'])
    pieces = [np.load(_clip_path(cache_dir, sub, c)) for c in range(n_clips)]
    eeg = np.concatenate(pieces, axis=1).astype(np.float32)
    return eeg, list(meta['ch_names']), meta


def channel_group_index(ch_names) -> dict:
    """Map each group name -> list of channel indices present in ``ch_names``."""
    up = [c.upper() for c in ch_names]
    out = {}
    for g, names in GROUPS.items():
        idx = [i for i, c in enumerate(up) if c in [n.upper() for n in names]]
        out[g] = idx
    return out


def _cache_key(sub: str, tag: str) -> str:
    h = hashlib.md5(f'{sub}|{tag}|{sorted(BANDS.items())}|{FS}'.encode()).hexdigest()[:10]
    return f'{sub}_{tag}_{h}.npz'


def build_subject_cumsums(data_dir: str, sub: str, cache_dir: str,
                          overwrite: bool = False) -> dict:
    """Per-band float64 power cumsum + channel metadata for one subject, cached.

    Returns a dict with ``cumsums`` (band -> (C, T+1) float64), ``ch_names``
    (kept 10-20 subset), ``group_idx`` (group -> channel-index list into the kept
    subset), ``T`` (samples). Cached as an .npz under ``cache_dir`` keyed by a
    band/fs hash so re-runs with a different lag grid skip the filtfilt."""
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, _cache_key(sub, 'cumsum'))
    # Cache hit: everything needed (bands, kept ch_names, group_idx, T) is in the
    # npz -- do NOT re-load + re-filter the raw EEG (that alone is ~15 s/subject).
    if os.path.exists(cpath) and not overwrite:
        z = np.load(cpath, allow_pickle=True)
        cumsums = {b: z[f'cs_{b}'] for b in BAND_ORDER}
        return {'cumsums': cumsums, 'ch_names': list(z['ch_names']),
                'group_idx': json.loads(str(z['group_idx'])), 'T': int(z['T'])}
    from datasets.egobrain_dataset import _resolve_ch_coords
    eeg_all, ch_all, _ = load_continuous_eeg(data_dir, sub)
    kept_names, keep_mask, _ = _resolve_ch_coords(ch_all)
    eeg = eeg_all[keep_mask]
    T = eeg.shape[1]
    group_idx = channel_group_index(kept_names)
    cumsums = {}
    for b in BAND_ORDER:
        lo, hi = BANDS[b]
        filt = bandpass_filtfilt(eeg, FS, lo, hi)
        cumsums[b] = power_cumsum(filt)
    save = {f'cs_{b}': cumsums[b] for b in BAND_ORDER}
    save['ch_names'] = np.asarray(kept_names)
    save['group_idx'] = json.dumps(group_idx)
    save['T'] = T
    tmp = cpath + '.tmp.npz'
    np.savez(tmp, **save)
    os.replace(tmp, cpath)
    return {'cumsums': cumsums, 'ch_names': kept_names,
            'group_idx': group_idx, 'T': T}


def read_hand_target(hand_dir: str, sub: str, combine: str = 'max',
                     det_gate: float = 0.5) -> dict:
    """Continuous drasticness target on the 0.2 s grid.

    ``d[k] = log1p(max(left,right))`` (combine='max', dominant-hand movement) or
    ``log1p(L+R)`` (combine='sum'). A slot is usable iff it has video, the chosen
    intensity is finite, and the detection gate ``max(det_frac) >= det_gate``.
    Returns ``y`` (n_slots,) with NaN at unusable slots, and ``det_ok`` mask."""
    import h5py
    with h5py.File(os.path.join(hand_dir, f'{sub}.h5'), 'r') as h:
        li = np.asarray(h['left_intensity'][:], np.float64)
        ri = np.asarray(h['right_intensity'][:], np.float64)
        # Schema-tolerant detection gate. OLD cache: float *_det_frac in [0,1].
        # NEW (forward/raw) cache: per-slot bool left_det/right_det -> read as
        # det_frac in {0.0, 1.0}; det_gate default 0.5 then keeps detected slots.
        if 'left_det_frac' in h:
            ld = np.asarray(h['left_det_frac'][:], np.float64)
            rd = np.asarray(h['right_det_frac'][:], np.float64)
        else:
            ld = np.asarray(h['left_det'][:], np.float64)
            rd = np.asarray(h['right_det'][:], np.float64)
        hv = np.asarray(h['has_video'][:], bool)
    if combine == 'max':
        raw = np.fmax(li, ri)
    elif combine == 'sum':
        raw = np.where(np.isfinite(li), li, 0.0) + np.where(np.isfinite(ri), ri, 0.0)
        raw[~(np.isfinite(li) | np.isfinite(ri))] = np.nan
    else:
        raise ValueError(combine)
    det_ok = np.fmax(np.nan_to_num(ld, nan=0.0), np.nan_to_num(rd, nan=0.0)) >= det_gate
    y = np.log1p(np.clip(raw, 0.0, None))
    usable = hv & np.isfinite(y) & det_ok
    y = np.where(usable, y, np.nan)
    return {'y': y, 'usable': usable, 'n_slots': y.shape[0]}


# =========================================================================== #
# Core lag estimator (shared by pipeline, null, bootstrap, recovery TEST)
# =========================================================================== #
def gather_log_power(cumsums: dict, w_by_band: dict, centers_s: np.ndarray,
                     fs: int = FS) -> tuple[np.ndarray, np.ndarray, list]:
    """Log windowed-mean power features at real-valued centre times.

    Returns ``(X (n, F) float64, inbounds (n,) bool, colinfo)`` where ``colinfo``
    is a list of ``(band, ch)`` per column. ``inbounds`` is the AND over bands of
    each window's support being fully inside the recording."""
    centers_samp = np.asarray(centers_s, dtype=np.float64) * fs
    cols, colinfo, inbounds = [], [], None
    for b in BAND_ORDER:
        wsamp = int(round(w_by_band[b] * fs))
        mean, ib = windowed_mean_power(cumsums[b], centers_samp, wsamp)  # (C, n)
        cols.append(np.log(mean + EPS))
        inbounds = ib if inbounds is None else (inbounds & ib)
        C = cumsums[b].shape[0]
        colinfo += [(b, ci) for ci in range(C)]
    X = np.concatenate(cols, axis=0).T                     # (n, F)
    return X, inbounds, colinfo


def subject_lag_curve(cumsums: dict, y_slot: np.ndarray, usable: np.ndarray,
                      lags: np.ndarray, *, w_by_band: dict = None,
                      grid_s: float = GRID_S, fs: int = FS, alpha: float = 1e3,
                      hp_M: int = 50, K: int = 8, embargo_slots: int = 50,
                      slot_stride: int = 1, col_mask: np.ndarray = None,
                      highpass: bool = True, label_offset_s: float = 0.0) -> dict:
    """Per-subject high-passed continuous-regression lag curve r(Delta).

    This is the estimator core the pipeline, the null, the bootstrap AND the
    synthetic recovery test all call. Steps: for each lag gather log-power
    features at centre ``k*grid_s - Delta``; high-pass features + target with a
    centred MA (zero group delay); blocked CV ridge with embargo; Fisher-z pool
    over blocks -> r(Delta). Features/targets are built on the FULL contiguous
    slot grid (so the centred MA sees no gaps), then masked + subsampled.

    Returns ``{'r': (n_lags,), 'n': int, 'colinfo': list}``.
    """
    if w_by_band is None:
        w_by_band = W_SHARP
    n_slots = min(y_slot.shape[0], cumsums[BAND_ORDER[0]].shape[1] - 1)
    slots = np.arange(n_slots)
    t_L = slots * grid_s + label_offset_s
    y = np.asarray(y_slot[:n_slots], dtype=np.float64).copy()
    use = usable[:n_slots].copy()
    y[~use] = np.nan

    # In-bounds intersection across sweep extremes -> a single fair valid set.
    _, ib_min, colinfo = gather_log_power(cumsums, w_by_band, t_L - lags.max(), fs)
    _, ib_max, _ = gather_log_power(cumsums, w_by_band, t_L - lags.min(), fs)
    base_valid = use & ib_min & ib_max & np.isfinite(y)

    # High-pass the target once (contiguous grid).
    y_hp = centered_ma_highpass(y, hp_M) if highpass else y

    # Precompute per-lag high-passed feature matrices at the valid+subsampled set.
    valid_idx = np.where(base_valid)[0][::slot_stride]
    if valid_idx.size < K + 5:
        return {'r': np.full(len(lags), np.nan), 'n': int(valid_idx.size),
                'colinfo': colinfo}
    yv = y_hp[valid_idx]
    slot_v = slots[valid_idx]
    rs = np.full(len(lags), np.nan)
    for li, d in enumerate(lags):
        X, _, _ = gather_log_power(cumsums, w_by_band, t_L - d, fs)  # (n_slots, F)
        if col_mask is not None:
            X = X[:, col_mask]
        Xhp = centered_ma_highpass(X, hp_M) if highpass else X
        Xv = Xhp[valid_idx]
        m = np.isfinite(Xv).all(1) & np.isfinite(yv)
        if m.sum() < K + 5:
            continue
        rs[li] = blocked_cv_r(Xv[m], yv[m], slot_v[m], alpha, K, embargo_slots)
    return {'r': rs, 'n': int(valid_idx.size), 'colinfo': colinfo}


# --------------------------------------------------------------------------- #
# Cached fast path -- precompute per-lag HP'd feature matrices ONCE, then the
# null rolls / bootstraps / band-group columns just re-run blocked CV with a
# (possibly rolled or column-sliced) target. This is what makes >=200 null rolls
# affordable: gathering + filtering features does NOT depend on the target.
# --------------------------------------------------------------------------- #
def precompute_lag_cache(cumsums: dict, y_slot: np.ndarray, usable: np.ndarray,
                         lags: np.ndarray, *, w_by_band: dict, grid_s: float = GRID_S,
                         fs: int = FS, hp_M: int = 50, slot_stride: int = 1,
                         label_offset_s: float = 0.0) -> dict:
    """Per-lag high-passed feature matrices on a FIXED valid+subsampled slot set.

    The valid mask (in-bounds at both sweep extremes AND target-usable) is
    computed ONCE and reused for every lag / null roll (spec Sec.4 fairness). The
    target is high-passed on the full grid and returned; rolling THAT (not the
    mask) is the circular-shift null. Returns Xv per lag, slot indices, the fixed
    valid indices, the HP target, and colinfo."""
    n_slots = min(y_slot.shape[0], cumsums[BAND_ORDER[0]].shape[1] - 1)
    slots = np.arange(n_slots)
    t_L = slots * grid_s + label_offset_s
    y = np.asarray(y_slot[:n_slots], dtype=np.float64).copy()
    use = usable[:n_slots].copy()
    y[~use] = np.nan
    _, ib_min, colinfo = gather_log_power(cumsums, w_by_band, t_L - lags.max(), fs)
    _, ib_max, _ = gather_log_power(cumsums, w_by_band, t_L - lags.min(), fs)
    base_valid = use & ib_min & ib_max & np.isfinite(y)
    y_hp_full = centered_ma_highpass(y, hp_M)
    valid_idx = np.where(base_valid)[0][::slot_stride]
    slot_v = slots[valid_idx]
    Xv = []
    for d in lags:
        X, _, _ = gather_log_power(cumsums, w_by_band, t_L - d, fs)
        Xhp = centered_ma_highpass(X, hp_M)
        Xv.append(Xhp[valid_idx].astype(np.float32))
    return {'Xv': Xv, 'slot_v': slot_v, 'valid_idx': valid_idx,
            'y_hp_full': y_hp_full, 'colinfo': colinfo, 'n': int(valid_idx.size),
            'n_slots': n_slots}


class SubjectCVModel:
    """Precomputed blocked-CV ridge operators for ONE subject's fixed valid slot
    set, so the observed curve, every null roll and every band-group column can be
    scored by cheap matvecs instead of re-fitting ridge.

    Because the feature matrix ``X`` at each lag does not depend on the target,
    ridge prediction on the held-out block is LINEAR in the (train) target:
    ``pred_te = Xte_std @ (Ainv @ Xtr_std^T @ (ytr - mean))``. We cache
    ``W_op = Ainv @ Xtr_std^T`` (F x n_tr) and the standardised test rows per
    (lag, fold); a null roll is then two small matvecs. This is what makes the
    >=200-roll null affordable at full resolution. Semantics are identical to
    :func:`blocked_cv_r` (train-only standardisation, within-block Pearson r,
    Fisher-z average weighted by test size)."""

    def __init__(self, Xv: list, slot_v: np.ndarray, alpha: float,
                 K: int = 8, embargo_slots: int = 50):
        self.n = len(slot_v)
        self.nlags = len(Xv)
        edges = np.linspace(0, self.n, K + 1).astype(int)
        self.folds = []                      # (tr_idx, te_idx), shared across lags
        for k in range(K):
            te = np.arange(edges[k], edges[k + 1])
            if te.size < 5:
                continue
            lo = slot_v[te[0]] - embargo_slots
            hi = slot_v[te[-1]] + embargo_slots
            tr = np.where((slot_v < lo) | (slot_v > hi))[0]
            self.folds.append((tr, te))
        # Per lag, per fold: (W_op (F,n_tr), Xte_std (n_te,F), tr, te).
        self.ops = []
        for X in Xv:
            X = np.asarray(X, dtype=np.float64)
            F = X.shape[1]
            perlag = []
            for (tr, te) in self.folds:
                if tr.size < F + 2 or te.size < 5:
                    perlag.append(None)
                    continue
                mu = X[tr].mean(0)
                sd = X[tr].std(0) + 1e-8
                Xtr = (X[tr] - mu) / sd
                Xte = (X[te] - mu) / sd
                A = Xtr.T @ Xtr
                A[np.diag_indices_from(A)] += alpha
                W_op = np.linalg.solve(A, Xtr.T)            # (F, n_tr)
                perlag.append((W_op.astype(np.float32), Xte.astype(np.float32),
                               tr, te))
            self.ops.append(perlag)

    def curve(self, y_valid: np.ndarray) -> np.ndarray:
        """Lag curve r(Delta) for a target aligned to the fixed valid set."""
        y = np.asarray(y_valid, dtype=np.float64)
        rs = np.full(self.nlags, np.nan)
        for li, perlag in enumerate(self.ops):
            zs, ws = [], []
            for item in perlag:
                if item is None:
                    continue
                W_op, Xte, tr, te = item
                ytr = y[tr] - y[tr].mean()
                pred = Xte @ (W_op @ ytr)
                r = _pearson(pred, y[te])
                if np.isfinite(r):
                    zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
                    ws.append(te.size)
            if zs:
                rs[li] = float(np.tanh(np.average(zs, weights=ws)))
        return rs

    def curves_batch(self, Y: np.ndarray) -> np.ndarray:
        """Vectorised lag curves for a BATCH of targets ``Y`` (B, n) aligned to
        the valid set -- e.g. B circular-shift null rolls at once. Batches the
        per-fold matvecs into matmuls, turning the O(B*nlags*K) Python fold loop
        into O(nlags*K) matmuls. Returns ``(B, nlags)``."""
        Y = np.asarray(Y, dtype=np.float64)
        B = Y.shape[0]
        out = np.full((B, self.nlags), np.nan)
        for li, perlag in enumerate(self.ops):
            zsum = np.zeros(B)
            wsum = np.zeros(B)
            for item in perlag:
                if item is None:
                    continue
                W_op, Xte, tr, te = item
                Ytr = Y[:, tr]
                Ytr = Ytr - Ytr.mean(1, keepdims=True)          # (B, n_tr)
                pred = (Ytr @ W_op.T) @ Xte.T                    # (B, n_te)
                Yte = Y[:, te]
                pc = pred - pred.mean(1, keepdims=True)
                yc = Yte - Yte.mean(1, keepdims=True)
                num = (pc * yc).sum(1)
                den = np.sqrt((pc * pc).sum(1) * (yc * yc).sum(1))
                r = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
                good = np.isfinite(r)
                z = np.zeros(B)
                z[good] = np.arctanh(np.clip(r[good], -0.999, 0.999))
                zsum[good] += z[good] * te.size
                wsum[good] += te.size
            ok = wsum > 0
            out[ok, li] = np.tanh(zsum[ok] / wsum[ok])
        return out


def curve_from_cache(cache: dict, y_hp_full: np.ndarray, lags: np.ndarray,
                     alpha: float, K: int, embargo_slots: int,
                     col_mask: np.ndarray = None) -> np.ndarray:
    """Lag curve r(Delta) from a :func:`precompute_lag_cache`, given a (possibly
    rolled / re-derived) HP target on the full grid. Only blocked CV runs here --
    no re-gathering -- so nulls and bootstraps are cheap."""
    yv = np.asarray(y_hp_full)[cache['valid_idx']]
    slot_v = cache['slot_v']
    rs = np.full(len(lags), np.nan)
    for li in range(len(lags)):
        Xv = cache['Xv'][li]
        if col_mask is not None:
            Xv = Xv[:, col_mask]
        m = np.isfinite(Xv).all(1) & np.isfinite(yv)
        if m.sum() < K + 5:
            continue
        rs[li] = blocked_cv_r(Xv[m].astype(np.float64), yv[m], slot_v[m],
                              alpha, K, embargo_slots)
    return rs


# =========================================================================== #
# Section 7a -- multi-lag TRF on band x group envelopes (cross-check)
# =========================================================================== #
def band_group_envelopes(cumsums: dict, group_idx: dict, w_by_band: dict,
                          t_L: np.ndarray, fs: int = FS) -> np.ndarray:
    """(n_slots, 25) log-power ROI envelopes: mean over each group's channels,
    per band. Centre = slot time (Delta handled by the TRF lag taps)."""
    centers = np.asarray(t_L) * fs
    cols = []
    for b in BAND_ORDER:
        wsamp = int(round(w_by_band[b] * fs))
        mean, _ = windowed_mean_power(cumsums[b], centers, wsamp)   # (C, n)
        logp = np.log(mean + EPS)
        for g in GROUP_ORDER:
            idx = group_idx.get(g, [])
            cols.append(logp[idx].mean(0) if idx else np.zeros(logp.shape[1]))
    return np.stack(cols, axis=1)                                    # (n, 25)


def fit_trf(env: np.ndarray, y: np.ndarray, valid: np.ndarray, slot_idx,
            taps: np.ndarray, grid_s: float, alpha: float, smooth: float,
            K: int = 8, embargo_slots: int = 50, row_stride: int = 1) -> dict:
    """Joint multi-lag TRF: y[k] = sum_tap sum_feat w[tap,feat] env[k-tap] + b.

    Ridge with a 2nd-difference temporal-smoothness penalty across taps (mTRF
    prior that the kernel is band-limited). Lag readout g(tap)=||w[tap,:]||_2 is a
    smooth kernel whose centroid is far more stable than a noisy per-lag argmax.
    Returns ``{'taps','g','r'}``: g over taps + held-out r of the joint model.
    """
    n, F = env.shape
    tap_samps = np.round(taps / grid_s).astype(int)
    # Build lagged design: column block per tap.
    blocks = []
    for ts in tap_samps:
        sh = np.full_like(env, np.nan)
        if ts == 0:
            sh = env
        elif ts > 0:
            sh[ts:] = env[:-ts]
        else:
            sh[:ts] = env[-ts:]
        blocks.append(sh)
    D = np.concatenate(blocks, axis=1)                    # (n, F*ntap)
    ok = valid & np.isfinite(D).all(1) & np.isfinite(y)
    idx = np.where(ok)[0]
    if row_stride > 1:                 # decimate ROWS (keep full-res lag design)
        idx = idx[::row_stride]
    if idx.size < K + 10:
        return {'taps': taps, 'g': np.full(len(taps), np.nan), 'r': np.nan}
    Dz = D[idx]
    yz = y[idx]
    si = np.asarray(slot_idx)[idx]
    ntap = len(taps)
    # 2nd-difference smoothness across taps (per feature): penalty matrix.
    L = np.zeros((ntap, ntap))
    for i in range(1, ntap - 1):
        L[i, i - 1] = 1
        L[i, i] = -2
        L[i, i + 1] = 1
    # Reg = alpha*I + smooth*(kron(L^T L, I_F)).
    LtL = L.T @ L
    edges = np.linspace(0, idx.size, K + 1).astype(int)
    zs, ws = [], []
    for k in range(K):
        te = np.arange(edges[k], edges[k + 1])
        if te.size < 5:
            continue
        lo = si[te[0]] - embargo_slots
        hi = si[te[-1]] + embargo_slots
        tr = np.where((si < lo) | (si > hi))[0]
        if tr.size < Dz.shape[1] + 2:
            continue
        mu = Dz[tr].mean(0)
        sd = Dz[tr].std(0) + 1e-8
        Xtr = (Dz[tr] - mu) / sd
        Xte = (Dz[te] - mu) / sd
        ytr = yz[tr] - yz[tr].mean()
        A = Xtr.T @ Xtr
        A[np.diag_indices_from(A)] += alpha
        A += smooth * np.kron(LtL, np.eye(F))
        w = np.linalg.solve(A, Xtr.T @ ytr)
        r = _pearson(Xte @ w, yz[te])
        if np.isfinite(r):
            zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
            ws.append(te.size)
    # Full-fit weights for the g(tap) kernel (all valid data, standardised).
    mu = Dz.mean(0)
    sd = Dz.std(0) + 1e-8
    Xf = (Dz - mu) / sd
    A = Xf.T @ Xf
    A[np.diag_indices_from(A)] += alpha
    A += smooth * np.kron(LtL, np.eye(F))
    w = np.linalg.solve(A, Xf.T @ (yz - yz.mean()))
    W = w.reshape(ntap, F)
    g = np.linalg.norm(W, axis=1)
    r = float(np.tanh(np.average(zs, weights=ws))) if zs else np.nan
    return {'taps': taps, 'g': g, 'r': r}


# =========================================================================== #
# Section 9c -- parametric-bootstrap recovery simulation
# =========================================================================== #
def make_synthetic_subject(delta_true: float, n_slots: int, fs: int = FS,
                           grid_s: float = GRID_S, snr: float = 1.0,
                           slow_frac: float = 0.5, seed: int = 0,
                           band: str = 'mu', label_offset_s: float = 0.0) -> dict:
    """A fake subject whose EEG band power at time ``tau`` is a known lagged copy
    of a synthetic hand-intensity series: p_band(tau) ~ hand(tau + delta_true).

    Built by amplitude-modulating an in-band carrier by sqrt(shifted hand), so the
    WHOLE filtfilt->square->cumsum->windowed-mean->log->gather->ridge estimator is
    exercised end to end. Returns ``eeg (1, T)``, ``y_slot``, ``usable`` and the
    injected ``delta_true``. A lag sweep on this must recover ``delta_true``.
    """
    rng = np.random.default_rng(seed)
    T = n_slots * int(round(grid_s * fs))
    t = np.arange(T) / fs
    # Hand intensity on the grid: fast (white, smoothed) + slow drift.
    fast = rng.standard_normal(n_slots)
    from scipy.ndimage import uniform_filter1d
    fast = uniform_filter1d(fast, size=3)                 # mild temporal width
    slow = uniform_filter1d(rng.standard_normal(n_slots), size=60)
    hand = (1 - slow_frac) * (fast - fast.min()) + slow_frac * (slow - slow.min())
    hand = hand - hand.min() + 0.05
    y_slot = np.log1p(hand)
    # EEG envelope at continuous time = hand sampled at (t + delta_true). The
    # label at slot s is stamped at (s + label_offset_s/grid_s)*grid_s = s*grid_s
    # + label_offset_s (matching the estimator's t_L), so a matched estimator with
    # the SAME label_offset_s recovers delta_true exactly (no offset-induced bias).
    grid_t = np.arange(n_slots) * grid_s + label_offset_s
    env_c = np.interp(t + delta_true, grid_t, hand, left=hand[0], right=hand[-1])
    env_c = np.clip(env_c, 0, None)
    lo, hi = BANDS[band]
    fc = 0.5 * (lo + hi)
    carrier = np.sin(2 * np.pi * fc * t + rng.uniform(0, 2 * np.pi))
    sig = np.sqrt(env_c) * carrier
    noise = rng.standard_normal(T) / max(snr, 1e-3)
    eeg = (sig + noise)[None, :].astype(np.float32) * 10.0
    usable = np.ones(n_slots, dtype=bool)
    return {'eeg': eeg, 'y_slot': y_slot, 'usable': usable,
            'delta_true': delta_true}


def estimate_lag_from_eeg(eeg: np.ndarray, y_slot: np.ndarray, usable: np.ndarray,
                          lags: np.ndarray, *, bands=('mu',), alpha: float = 10.0,
                          hp_M: int = 50, K: int = 5, embargo_slots: int = 25,
                          w_by_band: dict = None, highpass: bool = True,
                          label_offset_s: float = 0.0) -> dict:
    """Run the estimator core on a raw (C, T) EEG array for a chosen band subset.

    Convenience wrapper used by the recovery sim / unit test: filters the given
    bands, builds cumsums, and returns the lag curve + centroid over ``lags``."""
    if w_by_band is None:
        w_by_band = {b: W_SHARP[b] for b in bands}
    cumsums = {}
    for b in bands:
        lo, hi = BANDS[b]
        cumsums[b] = power_cumsum(bandpass_filtfilt(eeg, FS, lo, hi))
    # Restrict BAND_ORDER view to the requested bands via a temporary override.
    global BAND_ORDER
    saved = BAND_ORDER
    try:
        BAND_ORDER = list(bands)
        cur = subject_lag_curve(cumsums, y_slot, usable, lags,
                                w_by_band=w_by_band, alpha=alpha, hp_M=hp_M, K=K,
                                embargo_slots=embargo_slots, highpass=highpass,
                                label_offset_s=label_offset_s)
    finally:
        BAND_ORDER = saved
    peak = centroid_peak(lags, cur['r'])
    return {'r': cur['r'], 'lags': lags, 'peak': peak, 'n': cur['n']}


def _recovery_one(args):
    """Module-level recovery worker (picklable for ProcessPoolExecutor)."""
    dt, a, sd, n_slots, lags = args
    sub = make_synthetic_subject(dt, n_slots, snr=a, seed=sd, band='mu')
    est = estimate_lag_from_eeg(sub['eeg'], sub['y_slot'], sub['usable'], lags,
                                bands=('mu',), alpha=10.0)
    c = est['peak']['centroid']
    pk = np.nanmax(est['r']) if np.isfinite(est['r']).any() else np.nan
    return (a, dt, c, pk)


def recovery_sim(lags: np.ndarray, deltas_true, amps, n_slots: int = 4000,
                 n_iter: int = 200, workers: int = 8, seed: int = 0) -> dict:
    """Parametric-bootstrap recovery distribution: inject known Delta at a range
    of effect sizes, run the ACTUAL estimator, report P(|Dhat-Dtrue|<=0.1),
    bias, SD per injected amplitude. This histogram is the ground truth for the
    Sec.8 CI, not a closed form."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = []
    rng = np.random.default_rng(seed)
    for a in amps:
        for dt in deltas_true:
            for _ in range(n_iter):
                tasks.append((dt, float(a), int(rng.integers(0, 2**31)), n_slots,
                              lags))
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_recovery_one, tasks))
    else:
        res = [_recovery_one(t) for t in tasks]
    out = {}
    for a in amps:
        rows = [(dt, c, pk) for (aa, dt, c, pk) in res if aa == a]
        errs = np.array([abs(c - dt) for (dt, c, pk) in rows if np.isfinite(c)])
        peaks = np.array([pk for (dt, c, pk) in rows if np.isfinite(pk)])
        p_ok = float(np.mean(errs <= 0.1)) if errs.size else 0.0
        out[a] = {'p_recover_0.1': p_ok,
                  'bias': float(np.nanmean([c - dt for (dt, c, pk) in rows])),
                  'sd': float(np.nanstd([c for (dt, c, pk) in rows])),
                  'mean_peak_r': float(np.nanmean(peaks)) if peaks.size else np.nan,
                  'n': len(rows)}
    return out


# =========================================================================== #
# Section 9d -- visual-motion positive-control target
# =========================================================================== #
def read_visual_target(emb_grid_dir: str, frames_grid_dir: str, sub: str,
                       step_slots: int = 5, n_slots: int = None) -> np.ndarray:
    """Per-slot visual motion (patch/l1) as a positive-control target: large EEG
    footprint (head/scene motion) => high SNR => tests the instrument. Uses the
    cached motion arrays via :func:`datasets.egobrain_motion.load_or_compute_motion`.
    NaN-padded/truncated to ``n_slots``. Returns NaN array on any failure."""
    try:
        from datasets.egobrain_motion import load_or_compute_motion
        m = load_or_compute_motion(emb_grid_dir, sub, step_slots, metric='l1',
                                   space='patch', frames_grid_dir=frames_grid_dir)
    except Exception:                                          # noqa: BLE001
        m = None
    if m is None:
        return None
    m = np.asarray(m, dtype=np.float64)
    if n_slots is not None:
        if m.shape[0] >= n_slots:
            m = m[:n_slots]
        else:
            m = np.concatenate([m, np.full(n_slots - m.shape[0], np.nan)])
    return np.log1p(np.clip(m, 0, None))


# =========================================================================== #
# Section 9b -- compact event-locked peri-onset ERD decoder (diagnostic)
# =========================================================================== #
def detect_onsets(y_hp: np.ndarray, usable: np.ndarray, grid_s: float = GRID_S,
                  refractory_s: float = 2.0, isolation_s: float = 2.5) -> np.ndarray:
    """EEG-blind movement onsets on the high-passed target: upward crossings of
    median+1.5*MAD with positive slope, refractory + isolation enforced. Returns
    onset slot indices."""
    y = np.where(usable, y_hp, np.nan)
    fin = np.isfinite(y)
    if fin.sum() < 100:
        return np.array([], dtype=int)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) + 1e-9
    thr = med + 1.5 * mad
    above = (y > thr) & fin
    dy = np.zeros_like(y)
    dy[1:] = y[1:] - y[:-1]
    cross = np.where(above & (~np.r_[False, above[:-1]]) & (dy > 0))[0]
    refr = int(round(refractory_s / grid_s))
    iso = int(round(isolation_s / grid_s))
    kept = []
    last = -10 ** 9
    for k in cross:
        if k - last < refr:
            continue
        kept.append(k)
        last = k
    kept = np.array(kept, dtype=int)
    if kept.size <= 1:
        return kept
    d = np.diff(kept)
    kmask = np.ones(kept.size, bool)
    kmask[:-1] &= d >= iso
    kmask[1:] &= d >= iso
    return kept[kmask]


def event_locked_auc(cumsums: dict, group_idx: dict, onsets: np.ndarray,
                     n_slots: int, taus: np.ndarray, grid_s: float = GRID_S,
                     fs: int = FS, label_offset_s: float = 0.0) -> dict:
    """Time-resolved onset-vs-baseline separability AUC(tau) of central mu/beta
    log-power (ERD signature). Baseline = per-onset pre-window [-2.5,-2.0] s.
    A Delta*<0 (anticipatory) AUC peak is the one signature an artifact cannot
    fake. Returns ``{'taus','auc'}`` (single-feature rank-AUC, subject-pooled by
    caller)."""
    cidx = group_idx.get('central', [])
    if not cidx or onsets.size < 20:
        return {'taus': taus, 'auc': np.full(len(taus), np.nan)}
    W = 0.5
    wsamp = int(round(W * fs))
    def feat(centers_s):
        vals = []
        for b in ('mu', 'beta'):
            mean, ib = windowed_mean_power(cumsums[b], np.asarray(centers_s) * fs,
                                           wsamp)
            vals.append(np.log(mean[cidx].mean(0) + EPS))
        return np.mean(vals, axis=0)
    ot = onsets * grid_s + label_offset_s
    base = feat(ot - 2.25)                                    # [-2.5,-2.0] centre
    auc = np.full(len(taus), np.nan)
    for ti, tau in enumerate(taus):
        ev = feat(ot + tau)
        a = ev[np.isfinite(ev)]
        b = base[np.isfinite(base)]
        if a.size < 10 or b.size < 10:
            continue
        # Mann-Whitney rank AUC.
        allv = np.concatenate([a, b])
        rank = np.argsort(np.argsort(allv)) + 1
        r1 = rank[:a.size].sum()
        u = r1 - a.size * (a.size + 1) / 2.0
        auc[ti] = u / (a.size * b.size)
    return {'taus': taus, 'auc': auc}


# =========================================================================== #
# Pooling / null / bootstrap over subjects
# =========================================================================== #
def _subject_worker(args):
    """Parallel worker: build one subject's cumsums + targets, return the raw
    lag curve (and diagnostics). Heavy filtfilt is cached to disk."""
    (data_dir, sub, cache_dir, hand_dir, lags, cfg) = args
    t0 = time.time()
    info = build_subject_cumsums(data_dir, sub, cache_dir)
    cumsums, group_idx = info['cumsums'], info['group_idx']
    ht = read_hand_target(hand_dir, sub, combine=cfg['combine'],
                          det_gate=cfg['det_gate'])
    y_slot, usable = ht['y'], ht['usable']
    w_by_band = W_SHARP if cfg['window_variant'] == 'sharp' else W_DEPLOY
    A, K, EMB = cfg['alpha'], cfg['K'], cfg['embargo']

    # One feature cache reused by main / null / bootstrap / band-group columns.
    loff = cfg['label_offset_s']
    cache = precompute_lag_cache(cumsums, y_slot, usable, lags,
                                 w_by_band=w_by_band, hp_M=cfg['hp_M'],
                                 slot_stride=cfg['slot_stride'], label_offset_s=loff)
    y_hp_full = cache['y_hp_full']
    yv = y_hp_full[cache['valid_idx']]            # finite target on the fixed set
    model = SubjectCVModel(cache['Xv'], cache['slot_v'], A, K, EMB)
    main_r = model.curve(yv)
    # Raw (un-high-passed) target curve -> must be lag-invariant (pilot P4b).
    raw = subject_lag_curve(cumsums, y_slot, usable, lags, w_by_band=w_by_band,
                            alpha=A, hp_M=cfg['hp_M'], K=K, embargo_slots=EMB,
                            slot_stride=cfg['slot_stride'], highpass=False,
                            label_offset_s=loff)
    main = {'r': main_r, 'n': cache['n'], 'colinfo': cache['colinfo']}
    n_slots = cache['n_slots']
    # Null: CIRCULAR-SHIFT the target within the fixed valid set (features + mask
    # unchanged) and re-score -- genuinely zero, structural inflation cancels.
    rng = np.random.default_rng(abs(hash(sub)) % (2**32))
    nv = yv.shape[0]
    lo_roll, hi_roll = max(1, nv // 10), max(2, 9 * nv // 10)
    shifts = rng.integers(lo_roll, max(hi_roll, lo_roll + 1), size=cfg['n_null'])
    Ynull = np.stack([np.roll(yv, int(s)) for s in shifts], axis=0)  # (R, n)
    null_curves = model.curves_batch(Ynull)                          # (R, nlags)
    # Band x group matrix (per-band single-group-column decoders).
    bg = np.full((len(BAND_ORDER), len(GROUP_ORDER)), np.nan)
    if cfg['do_matrix']:
        colinfo = cache['colinfo']
        for bi, b in enumerate(BAND_ORDER):
            for gi, g in enumerate(GROUP_ORDER):
                gcols = set(group_idx.get(g, []))
                cm = np.array([(cb == b and ci in gcols)
                               for (cb, ci) in colinfo])
                if cm.sum() == 0:
                    continue
                cr = curve_from_cache(cache, y_hp_full, lags, A, K, EMB, col_mask=cm)
                bg[bi, gi] = centroid_peak(lags, cr)['centroid']
    # TRF cross-check.
    n2 = n_slots
    t_L = np.arange(n2) * GRID_S + loff
    env = band_group_envelopes(cumsums, group_idx, w_by_band, t_L)
    y_hp2 = centered_ma_highpass(np.where(usable[:n2], y_slot[:n2], np.nan),
                                 cfg['hp_M'])
    valid2 = usable[:n2] & np.isfinite(y_hp2)
    trf = fit_trf(env, y_hp2, valid2, np.arange(n2), cfg['trf_taps'], GRID_S,
                  alpha=cfg['alpha'], smooth=cfg['trf_smooth'], K=cfg['K'],
                  embargo_slots=cfg['embargo'], row_stride=cfg['slot_stride'])
    # Event-locked diagnostic.
    onsets = detect_onsets(y_hp2, valid2)
    ev = event_locked_auc(cumsums, group_idx, onsets, n2, cfg['event_taus'],
                          label_offset_s=loff)
    # Visual-motion positive control (best-effort).
    vis_curve = None
    if cfg['do_visual']:
        vt = read_visual_target(cfg['emb_grid_dir'], cfg['frames_grid_dir'], sub,
                                step_slots=cfg['visual_step'], n_slots=n_slots)
        if vt is not None:
            vis_use = np.isfinite(vt)
            vc = subject_lag_curve(cumsums, vt, vis_use, lags, w_by_band=w_by_band,
                                   alpha=cfg['alpha'], hp_M=cfg['hp_M'],
                                   K=cfg['K'], embargo_slots=cfg['embargo'],
                                   slot_stride=cfg['slot_stride'],
                                   label_offset_s=loff)
            vis_curve = vc['r']
    return {
        'sub': sub, 'main': main['r'], 'raw': raw['r'], 'n': main['n'],
        'null': np.asarray(null_curves), 'bg': bg, 'trf_g': trf['g'],
        'trf_r': trf['r'], 'event_auc': ev['auc'], 'n_onsets': int(onsets.size),
        'vis': vis_curve, 'secs': time.time() - t0,
    }


# =========================================================================== #
# Decision rule (Section 8)
# =========================================================================== #
def decide(pooled: dict, lags: np.ndarray, cfg: dict) -> dict:
    """Pre-registered decision. Returns verdict + all supporting numbers.

    VERDICT in {IDENTIFIED, WEAK, NOT_IDENTIFIABLE}. Never emits an erp that
    fails the gate: only IDENTIFIED yields a moved knob; WEAK/NOT keep -0.15 or
    the agnostic fallback Delta*=0."""
    C = pooled['C']                       # pooled fast-target curve
    peak_r = float(np.nanmax(C)) if np.isfinite(C).any() else np.nan
    pk = centroid_peak(lags, C)
    # Null band: max-over-lag of each pooled null realisation.
    null_max = pooled['null_max']         # (n_null,)
    mu_n = float(np.nanmean(null_max)) if null_max.size else np.nan
    sd_n = float(np.nanstd(null_max)) if null_max.size else np.nan
    z = (peak_r - mu_n) / sd_n if (sd_n and np.isfinite(sd_n) and sd_n > 0) else np.nan
    # Subject-bootstrap CI on the centroid.
    boot = pooled['boot_centroids']
    boot = boot[np.isfinite(boot)]
    if boot.size >= 20:
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        ci_half = float((ci_hi - ci_lo) / 2.0)
    else:
        ci_lo = ci_hi = np.nan
        ci_half = np.inf
    # Cross-subject homogeneity of argmax (need a common Delta* for pooling).
    homogeneous = pooled['homogeneous']
    # TRF agreement.
    trf_peak = pooled['trf_peak']
    trf_agree = (np.isfinite(trf_peak) and np.isfinite(pk['centroid'])
                 and abs(trf_peak - pk['centroid']) <= max(ci_half, 0.2))
    # Neural gate + visual control + recovery.
    neural_ok = pooled['neural_ok']
    visual_ok = pooled['visual_localised']
    recovery_ok = pooled['recovery_ok']
    aborted = pk['aborted']

    identified = (np.isfinite(z) and z >= 3 and ci_half <= 0.20 and homogeneous
                  and trf_agree and neural_ok and visual_ok and recovery_ok
                  and not aborted)
    weak = (not identified) and (
        (np.isfinite(z) and 2 <= z < 3) or (0.20 < ci_half <= 0.40)) and not aborted
    verdict = ('IDENTIFIED' if identified else ('WEAK' if weak
               else 'NOT_IDENTIFIABLE'))
    delta_star = pk['centroid']
    res = {
        'verdict': verdict, 'peak_r': peak_r, 'mu_null': mu_n, 'sd_null': sd_n,
        'excess_z': z, 'centroid': delta_star, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'ci_half': ci_half, 'homogeneous': homogeneous, 'trf_peak': trf_peak,
        'trf_agree': trf_agree, 'neural_ok': neural_ok, 'visual_ok': visual_ok,
        'recovery_ok': recovery_ok, 'aborted': aborted, 'abort_reason': pk['reason'],
        'argmax': pk['argmax'], 'parabola': pk['parabola'],
    }
    # Knob recommendation -- only IDENTIFIED moves it.
    if verdict == 'IDENTIFIED':
        erp_grid = snap_erp_to_grid(delta_to_erp_latency(delta_star, path='grid'))
        erp_clip = snap_erp_to_grid(delta_to_erp_latency(delta_star, path='clip'))
        res['recommend'] = {'delta_star': delta_star, 'erp_grid': erp_grid,
                            'erp_clip': erp_clip, 'move': True}
    else:
        # fallback Delta*=0 (agnostic) only reported, not auto-applied for WEAK.
        res['recommend'] = {'delta_star': 0.0,
                            'erp_grid': snap_erp_to_grid(delta_to_erp_latency(0.0, path='grid')),
                            'erp_clip': snap_erp_to_grid(delta_to_erp_latency(0.0, path='clip')),
                            'move': False}
    return res


# =========================================================================== #
# Orchestration
# =========================================================================== #
def detect_label_offset(hand_dir: str, sub: str, grid_s: float = GRID_S) -> float:
    """Auto-detect the label-time offset (s) for a hand-label cache.

    The NEW forward/raw cache stamps slot ``s`` with the RAW hand speed over the
    single forward pair ``[s, s+1]*grid_s``; its natural timestamp is the interval
    MIDPOINT ``(s+0.5)*grid_s`` -- i.e. ``+0.5`` slot (=+0.1 s @ grid_s=0.2) LATER
    than the OLD symmetric-window convention centred exactly on ``s*grid_s``. This
    offset is ADDED to the label time so the recovered lag does not undershoot the
    true physical Delta by 0.1 s. Detected via the h5 attrs (speed_interval=='forward'
    / smoothing=='none' / format_version>=2) or the schema tell (no 'left_det_frac').
    Returns ``0.5*grid_s`` for the forward cache, else ``0.0``."""
    import h5py
    try:
        with h5py.File(os.path.join(hand_dir, f'{sub}.h5'), 'r') as h:
            forward = (str(h.attrs.get('speed_interval', '')) == 'forward'
                       or str(h.attrs.get('smoothing', '')) == 'none'
                       or int(h.attrs.get('format_version', 1)) >= 2
                       or 'left_det_frac' not in h)
    except Exception:                                          # noqa: BLE001
        forward = False
    return float(0.5 * grid_s) if forward else 0.0


def parse_subjects(spec: str, hand_dir: str) -> list:
    if spec.lower() == 'all':
        import re
        return sorted(re.match(r'^(P\d{4})\.h5$', f).group(1)
                      for f in os.listdir(hand_dir)
                      if re.match(r'^P\d{4}\.h5$', f))
    return [s.strip() for s in spec.split(',') if s.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', default='data/EgoBrain')
    ap.add_argument('--hand_dir',
                    default='data/EgoBrain/cache_hand_labels_grid_wilor_g0.2_raw_fs200')
    ap.add_argument('--emb_grid_dir',
                    default='data/EgoBrain/cache_embeddings_grid_facebook_dinov2-base_g0.2_sz224')
    ap.add_argument('--frames_grid_dir',
                    default='data/EgoBrain/cache_frames_grid_facebook_dinov2-base_g0.2_sz224')
    ap.add_argument('--subjects', default='all')
    ap.add_argument('--lag_min', type=float, default=-2.0)
    ap.add_argument('--lag_max', type=float, default=2.0)
    ap.add_argument('--lag_step', type=float, default=0.05)
    ap.add_argument('--slot_stride', type=int, default=1,
                    help='decimate the usable slots (smoke: 10)')
    ap.add_argument('--window_variant', choices=['sharp', 'deploy'], default='sharp')
    ap.add_argument('--combine', choices=['max', 'sum'], default='max')
    ap.add_argument('--det_gate', type=float, default=0.5)
    ap.add_argument('--label_offset_s', type=float, default=None,
                    help='label-time offset (s) added to slot times; '
                         'default None = auto-detect (+0.1 s for the forward/raw '
                         'cache, 0.0 for the legacy smoothed cache)')
    ap.add_argument('--alpha', type=float, default=1e3, help='frozen ridge alpha')
    ap.add_argument('--hp_slots', type=int, default=50, help='centred-MA slots (10 s)')
    ap.add_argument('--cv_folds', type=int, default=8)
    ap.add_argument('--embargo_slots', type=int, default=50)
    ap.add_argument('--n_null', type=int, default=200)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--trf_smooth', type=float, default=1.0)
    ap.add_argument('--visual_step', type=int, default=5)
    ap.add_argument('--no_visual', action='store_true')
    ap.add_argument('--no_matrix', action='store_true')
    ap.add_argument('--recovery_iters', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cache_dir',
                    default='/tmp/claude-20483/-gpfs-radev-pi-ying-rex-wq44-CSBrain/'
                            '184048cc-a117-4089-a3f7-1bb1a576982b/scratchpad/erp_cache')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--fig_dir', default='figs')
    args = ap.parse_args()

    lags = np.round(np.arange(args.lag_min, args.lag_max + 1e-9, args.lag_step), 4)
    trf_taps = np.round(np.arange(-1.5, 1.5 + 1e-9, max(args.lag_step, 0.05)), 4)
    event_taus = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.1), 4)
    subs = parse_subjects(args.subjects, args.hand_dir)

    # Label-time offset: forward/raw cache stamps slot s at midpoint (s+0.5)*grid_s
    # => +0.5 slot (+0.1 s) later than the old symmetric-window convention.
    if args.label_offset_s is not None:
        label_offset_s = float(args.label_offset_s)
        offset_src = 'override'
    else:
        label_offset_s = detect_label_offset(args.hand_dir, subs[0], GRID_S)
        offset_src = 'auto'

    print('=' * 78)
    print('ERP-LATENCY CALIBRATION  (lag-identifiability study)')
    print('=' * 78)
    print(f'subjects           : {len(subs)}  {subs if len(subs)<=6 else subs[:6]+["..."]}')
    print(f'Delta sweep        : [{args.lag_min}, {args.lag_max}] step {args.lag_step}'
          f'  ({len(lags)} lags)')
    print(f'window variant     : {args.window_variant}  (sharp W: {W_SHARP})')
    print(f'frozen ridge alpha : {args.alpha:g}   HP: centred-MA {args.hp_slots} slots'
          f' ({args.hp_slots*GRID_S:.0f} s)')
    print(f'CV                 : {args.cv_folds} blocks, embargo {args.embargo_slots}'
          f' slots ({args.embargo_slots*GRID_S:.0f} s)')
    print(f'null rolls         : {args.n_null}   subject boots: {args.n_boot}')
    print(f'label offset       : {label_offset_s:+.3f} s ({offset_src}; '
          f'+0.1 => forward/raw cache, 0.0 => legacy smoothed cache)')
    print('NOTE: egobrain_dataset.py:319 docstring was stale (claimed centre-'
          'anchoring; code is START-anchored) -- fixed in this change.')
    print('NOTE: label kernel is R=2, symmetric +/-0.4 s (0.8 s span); the brief/'
          'MEMORY "R=3 / +/-0.6 s" is wrong (CPython round(2.499..)=2).')
    print(f'current erp=-0.15 (grid) ships PHYSICAL Delta = '
          f'{erp_latency_to_delta(-0.15, path="grid"):+.2f} s (EEG lagging movement).')
    print('-' * 78)

    cfg = {
        'combine': args.combine, 'det_gate': args.det_gate, 'alpha': args.alpha,
        'hp_M': args.hp_slots, 'K': args.cv_folds, 'embargo': args.embargo_slots,
        'slot_stride': args.slot_stride, 'window_variant': args.window_variant,
        'n_null': args.n_null, 'do_matrix': not args.no_matrix,
        'do_visual': not args.no_visual, 'visual_step': args.visual_step,
        'emb_grid_dir': args.emb_grid_dir, 'frames_grid_dir': args.frames_grid_dir,
        'trf_taps': trf_taps, 'trf_smooth': args.trf_smooth,
        'event_taus': event_taus, 'label_offset_s': label_offset_s,
    }

    tasks = [(args.data_dir, s, args.cache_dir, args.hand_dir, lags, cfg)
             for s in subs]
    print(f'[stage A] per-subject feature build + curves ({args.workers} workers)...')
    t0 = time.time()
    results = []
    if args.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_subject_worker, tasks):
                print(f'   {r["sub"]}: n={r["n"]}  trf_r={r["trf_r"]:.3f}  '
                      f'onsets={r["n_onsets"]}  ({r["secs"]:.1f}s)')
                results.append(r)
    else:
        for tk in tasks:
            r = _subject_worker(tk)
            print(f'   {r["sub"]}: n={r["n"]}  trf_r={r["trf_r"]:.3f}  '
                  f'onsets={r["n_onsets"]}  ({r["secs"]:.1f}s)')
            results.append(r)
    print(f'[stage A] done in {time.time()-t0:.1f}s')

    # ------- pool subjects -------
    mainM = np.vstack([r['main'] for r in results])           # (S, L)
    rawM = np.vstack([r['raw'] for r in results])
    C = np.array([fisher_mean(mainM[:, i]) for i in range(len(lags))])
    C_raw = np.array([fisher_mean(rawM[:, i]) for i in range(len(lags))])
    # pooled null: average each roll's curve over subjects, take max-over-lag.
    n_null = min(min(r['null'].shape[0] for r in results), args.n_null) \
        if all(r['null'].size for r in results) else 0
    null_max = np.array([])
    if n_null > 0:
        stack = np.stack([r['null'][:n_null] for r in results], axis=0)  # (S, R, L)
        null_pooled = np.array([[fisher_mean(stack[:, roll, i]
                                             ) for i in range(len(lags))]
                                for roll in range(n_null)])               # (R, L)
        null_max = np.nanmax(null_pooled, axis=1)

    # subject bootstrap of the centroid.
    rng = np.random.default_rng(0)
    S = len(results)
    boot_centroids, boot_argmax = [], []
    for _ in range(args.n_boot):
        pick = rng.integers(0, S, S)
        Cb = np.array([fisher_mean(mainM[pick, i]) for i in range(len(lags))])
        pkb = centroid_peak(lags, Cb)
        boot_centroids.append(pkb['centroid'])
        boot_argmax.append(pkb['argmax'])
    boot_centroids = np.array(boot_centroids)
    boot_argmax = np.array(boot_argmax)

    # cross-subject argmax homogeneity: spread of per-subject argmaxes.
    per_argmax = np.array([lags[int(np.nanargmax(np.where(np.isfinite(r['main']),
                          r['main'], -np.inf)))] if np.isfinite(r['main']).any()
                          else np.nan for r in results])
    homogeneous = bool(np.nanstd(per_argmax) <= 0.6) if S > 1 else False

    # TRF pooled kernel peak.
    trf_g = np.nanmean(np.vstack([r['trf_g'] for r in results]), axis=0)
    trf_pk = centroid_peak(trf_taps, trf_g)
    trf_peak = trf_pk['centroid'] if not trf_pk['aborted'] else trf_pk['argmax']

    # band x group centroid matrix (median across subjects).
    bgM = np.nanmedian(np.stack([r['bg'] for r in results], 0), axis=0)

    # neural gate: a fast-target peak that is band-specific + central/occipital
    # localised. Approximate with: the pooled peak's excess over null AND the
    # central-mu/beta group columns not being NaN + finite matrix. Conservative:
    # require the central or occipital groups to carry a finite centroid AND the
    # peak not be uniformly present in parietal (control).
    neural_ok = False
    try:
        cen_mu = bgM[BAND_ORDER.index('mu'), GROUP_ORDER.index('central')]
        cen_beta = bgM[BAND_ORDER.index('beta'), GROUP_ORDER.index('central')]
        par = bgM[:, GROUP_ORDER.index('parietal')]
        neural_ok = bool((np.isfinite(cen_mu) or np.isfinite(cen_beta))
                         and np.nanstd(bgM) > 0)
    except Exception:                                          # noqa: BLE001
        neural_ok = False

    # visual-motion positive control.
    vis_curves = [r['vis'] for r in results if r['vis'] is not None]
    visual_localised = False
    vis_C = None
    if vis_curves:
        VM = np.vstack(vis_curves)
        vis_C = np.array([fisher_mean(VM[:, i]) for i in range(len(lags))])
        vpk = centroid_peak(lags, vis_C)
        visual_localised = (not vpk['aborted']) and (np.nanmax(vis_C) > 0.03)

    # recovery simulation (does the estimator recover a KNOWN injected lag?).
    print('[stage B] recovery simulation (injected-lag identifiability)...')
    amps = [0.5, 1.0, 2.0, 4.0]
    rec = recovery_sim(lags, deltas_true=[-0.4, 0.0, 0.4],
                       amps=amps, n_slots=3000, n_iter=args.recovery_iters,
                       workers=args.workers)
    # recovery_ok at the OBSERVED effect size: find the amp whose mean peak r
    # brackets the observed peak, require its P(recover)>=0.5.
    obs_peak = float(np.nanmax(C)) if np.isfinite(C).any() else 0.0
    recovery_ok = False
    for a in sorted(amps):
        if np.isfinite(rec[a]['mean_peak_r']) and rec[a]['mean_peak_r'] >= obs_peak:
            recovery_ok = rec[a]['p_recover_0.1'] >= 0.5
            break

    pooled = {
        'C': C, 'null_max': null_max, 'boot_centroids': boot_centroids,
        'homogeneous': homogeneous, 'trf_peak': trf_peak,
        'neural_ok': neural_ok, 'visual_localised': visual_localised,
        'recovery_ok': recovery_ok,
    }
    dec = decide(pooled, lags, cfg)

    # N_eff / SE / MDE.
    tau_int = 5.0                     # ~1.0 s integrated AC (pilot P9c)
    n_valid_tot = int(np.sum([r['n'] for r in results]))
    n_eff = n_valid_tot / tau_int
    sd_null = dec['sd_null'] if np.isfinite(dec['sd_null']) else np.nan
    se_pooled = sd_null / np.sqrt(max(S, 1)) if np.isfinite(sd_null) else np.nan
    s_res = 0.6
    mde = (se_pooled * s_res / 0.051) if np.isfinite(se_pooled) else np.nan

    # ---------------- print verdict banner ----------------
    print('=' * 78)
    _print_verdict(dec, rec, obs_peak, mde, n_eff, C_raw, lags, C)

    # ---------------- outputs ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    date = datetime.now().strftime('%Y-%m-%d')
    dump = {
        'config': vars(args), 'lags': lags.tolist(), 'C_fast': C.tolist(),
        'C_raw': C_raw.tolist(), 'null_max': null_max.tolist(),
        'boot_centroids': boot_centroids.tolist(),
        'boot_argmax': boot_argmax.tolist(), 'per_subject_argmax': per_argmax.tolist(),
        'bg_matrix': bgM.tolist(), 'bg_bands': BAND_ORDER, 'bg_groups': GROUP_ORDER,
        'trf_taps': trf_taps.tolist(), 'trf_g': trf_g.tolist(),
        'trf_peak': float(trf_peak) if np.isfinite(trf_peak) else None,
        'recovery': {str(k): v for k, v in rec.items()},
        'vis_C': vis_C.tolist() if vis_C is not None else None,
        'n_eff': n_eff, 'se_pooled': se_pooled, 'mde': mde,
        'decision': {k: (float(v) if isinstance(v, (np.floating, float, np.integer))
                         else v) for k, v in dec.items() if k != 'recommend'},
        'recommend': dec['recommend'],
    }
    jpath = os.path.join(args.out_dir, f'calibrate_erp_latency_{date}.json')
    with open(jpath, 'w') as f:
        json.dump(dump, f, indent=2, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else float(o)))
    print(f'\nwrote {jpath}')

    _append_eval_table(args.out_dir, date, dec, rec, obs_peak, mde, n_eff, bgM,
                       trf_peak, subs, args)
    _make_figures(args.fig_dir, date, lags, C, C_raw, null_max, trf_taps, trf_g,
                  bgM, rec, vis_C, boot_argmax, dec)
    print('done.')


def _print_verdict(dec, rec, obs_peak, mde, n_eff, C_raw, lags, C):
    v = dec['verdict']
    if v == 'IDENTIFIED':
        r = dec['recommend']
        print(f'IDENTIFIED  Delta*={dec["centroid"]:+.3f} s '
              f'(95% CI +/-{dec["ci_half"]:.3f});  '
              f'erp_latency_s[grid]=Delta*+0.5={r["erp_grid"]:+.3f},  '
              f'erp[clip]={r["erp_clip"]:+.3f}')
    elif v == 'WEAK':
        print(f'WEAK: Delta point est {dec["centroid"]:+.3f}, '
              f'CI +/-{dec["ci_half"]:.3f} -- insufficient to overturn; '
              f'keep current erp=-0.15')
    else:
        raw_flat = np.nanmax(C_raw) - np.nanmin(C_raw) if np.isfinite(C_raw).any() else np.nan
        print('NOT IDENTIFIABLE: EEG->hand-intensity decoding cannot localise '
              'erp_latency_s on this data.')
        print(f'  pooled peak r={obs_peak:.4f} vs max-null r={dec["mu_null"]:.4f} '
              f'(excess z={dec["excess_z"]:.2f})')
        print(f'  centroid={dec["centroid"] if np.isfinite(dec["centroid"]) else float("nan"):.3f} s, '
              f'subject-bootstrap 95% CI=+/-{dec["ci_half"]:.3f} s (>0.40 s)')
        print(f'  MDE for +/-0.1 s = r>={mde:.4f}  (observed lag-signal r={obs_peak:.4f})')
        print('  DO NOT change erp_latency_s on this evidence.')
        print('  Fallback: set Delta*=0 (EEG window centred on frame) => '
              'erp_latency_s[grid]=+0.5.')
        print('  Rationale: maximally agnostic AND corrects the current '
              'physiologically-backwards Delta=-0.65 s')
        print('  (the -0.15 value is a grid-path START-anchoring bug: intended '
              'clip-path -0.15 became grid -0.65).')
    # diagnostics always printed.
    print('-' * 78)
    if dec['aborted']:
        print(f'  [abort guard fired: {dec["abort_reason"]}]')
    print(f'  raw-target curve is lag-invariant (pilot P4b sanity): '
          f'range={np.nanmax(C_raw)-np.nanmin(C_raw):.4f}  '
          f'[label: slow coupling -- NOT usable for Delta]')
    print(f'  TRF peak={dec["trf_peak"] if np.isfinite(dec["trf_peak"]) else float("nan"):+.3f} s'
          f'  (agree with centroid: {dec["trf_agree"]})')
    print(f'  gates: neural={dec["neural_ok"]} visual={dec["visual_ok"]} '
          f'recovery={dec["recovery_ok"]} homogeneous={dec["homogeneous"]}')
    print('  recovery P(|Dhat-Dtrue|<=0.1) by injected amp:')
    for a in sorted(rec):
        print(f'     amp={a:>4}: P={rec[a]["p_recover_0.1"]:.2f} '
              f'mean_peak_r={rec[a]["mean_peak_r"]:.3f} '
              f'bias={rec[a]["bias"]:+.3f} sd={rec[a]["sd"]:.3f}')
    print('=' * 78)


def _append_eval_table(out_dir, date, dec, rec, obs_peak, mde, n_eff, bgM,
                       trf_peak, subs, args):
    path = os.path.join(out_dir, 'eval_tables.md')
    lines = []
    lines.append(f'\n## {date} -- ERP-latency calibration (lag-identifiability)\n')
    lines.append(f'**VERDICT: {dec["verdict"]}**  '
                 f'(subjects={len(subs)}, sweep {args.lag_min}..{args.lag_max}@'
                 f'{args.lag_step}s, window={args.window_variant})\n')
    lines.append(f'- pooled peak r={obs_peak:.4f}, max-null r={dec["mu_null"]:.4f}, '
                 f'excess z={dec["excess_z"]:.2f}\n')
    lines.append(f'- centroid Delta*={dec["centroid"]:.3f} s, '
                 f'subj-bootstrap 95% CI +/-{dec["ci_half"]:.3f} s\n')
    lines.append(f'- TRF peak={trf_peak:.3f} s (agree={dec["trf_agree"]}), '
                 f'homogeneous={dec["homogeneous"]}, MDE(+/-0.1)=r>={mde:.4f}\n')
    lines.append(f'- gates: neural={dec["neural_ok"]}, visual={dec["visual_ok"]}, '
                 f'recovery={dec["recovery_ok"]}\n')
    rec_ = dec['recommend']
    if dec['verdict'] == 'IDENTIFIED':
        lines.append(f'- **RECOMMEND** erp_latency_s[grid]={rec_["erp_grid"]:+.3f}, '
                     f'[clip]={rec_["erp_clip"]:+.3f}\n')
    else:
        lines.append('- **DO NOT move the knob.** Fallback Delta*=0 => '
                     f'erp[grid]={rec_["erp_grid"]:+.3f} corrects the current '
                     'backwards Delta=-0.65 s if a change is forced.\n')
    lines.append('\n| band\\group | ' + ' | '.join(GROUP_ORDER) + ' |\n')
    lines.append('|' + '---|' * (len(GROUP_ORDER) + 1) + '\n')
    for bi, b in enumerate(BAND_ORDER):
        cells = ' | '.join(f'{bgM[bi,gi]:+.2f}' if np.isfinite(bgM[bi, gi])
                           else 'nan' for gi in range(len(GROUP_ORDER)))
        lines.append(f'| {b} | {cells} |\n')
    lines.append('\nrecovery P(|Dhat-Dtrue|<=0.1): ' +
                 ', '.join(f'amp{a}={rec[a]["p_recover_0.1"]:.2f}'
                           f'(r~{rec[a]["mean_peak_r"]:.2f})' for a in sorted(rec))
                 + '\n')
    with open(path, 'a') as f:
        f.writelines(lines)
    print(f'appended results to {path}')


def _make_figures(fig_dir, date, lags, C, C_raw, null_max, trf_taps, trf_g,
                  bgM, rec, vis_C, boot_argmax, dec):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(18, 9))
        a = ax[0, 0]
        a.plot(lags, C, 'b-', label='fast (HP) target')
        a.plot(lags, C_raw, 'r--', alpha=0.6, label='raw (lag-invariant)')
        if null_max.size:
            a.axhspan(np.nanpercentile(null_max, 2.5),
                      np.nanpercentile(null_max, 97.5), color='gray', alpha=0.3,
                      label='null 95% band (max-over-lag)')
        a.axvline(0, color='k', lw=0.5)
        a.set_xlabel('Delta = frame_time - eeg_centre (s)')
        a.set_ylabel('CV Pearson r')
        a.set_title(f'lag curve  [{dec["verdict"]}]')
        a.legend(fontsize=8)
        a = ax[0, 1]
        a.plot(trf_taps, trf_g, 'g-')
        a.axvline(0, color='k', lw=0.5)
        a.set_title('TRF |w|(lag) kernel')
        a.set_xlabel('lag (s)')
        a = ax[0, 2]
        im = a.imshow(bgM, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        a.set_xticks(range(len(GROUP_ORDER)))
        a.set_xticklabels(GROUP_ORDER, rotation=45, fontsize=7)
        a.set_yticks(range(len(BAND_ORDER)))
        a.set_yticklabels(BAND_ORDER, fontsize=7)
        a.set_title('band x group centroid (s)')
        plt.colorbar(im, ax=a, fraction=0.046)
        a = ax[1, 0]
        amps = sorted(rec)
        a.plot(amps, [rec[x]['p_recover_0.1'] for x in amps], 'o-')
        a.axhline(0.5, color='r', ls='--')
        a.set_xlabel('injected amplitude (snr)')
        a.set_ylabel('P(|Dhat-Dtrue|<=0.1)')
        a.set_title('recovery power')
        a = ax[1, 1]
        if vis_C is not None:
            a.plot(lags, vis_C, 'm-')
            a.axvline(0, color='k', lw=0.5)
            a.set_title('visual-motion positive control')
            a.set_xlabel('Delta (s)')
        else:
            a.text(0.5, 0.5, 'no visual control', ha='center')
        a = ax[1, 2]
        bb = boot_argmax[np.isfinite(boot_argmax)]
        if bb.size:
            a.hist(bb, bins=30, color='steelblue')
        a.set_title('subject-bootstrap argmax')
        a.set_xlabel('Delta (s)')
        fig.tight_layout()
        out = os.path.join(fig_dir, f'calibrate_erp_latency_{date}.png')
        fig.savefig(out, dpi=100)
        print(f'wrote {out}')
    except Exception as e:                                     # noqa: BLE001
        print(f'[figure skipped] {e}')


if __name__ == '__main__':
    main()
