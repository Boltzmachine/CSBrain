#!/usr/bin/env python
"""Measurement sanity check for the EgoBrain EEG<->video pipeline.

Every "the EEG is ignored" result in this project (frame-pred diag_frame_eeg_gap
~0.16%, flip-align InfoNCE at chance, hand-side AUC ~0.5, hand-intensity r ~0,
and the ERP-latency study's visual-motion positive control at r ~0.000) shares a
single untested assumption: ``video_offset_s = 0.0`` -- the EEG<->video sync was
never measured. A minutes-scale offset would null every EEG<->video coupling and
be indistinguishable from "no signal in the EEG". This script runs the two
decisive controls, neither of which the calibration study passed:

  TEST A -- eyes-open vs eyes-closed occipital alpha (NO video, NO sync).
    Validates the EEG cache + epoching + spectral extraction end to end. Occipital
    8-13 Hz power must be higher eyes-closed at AUC > 0.9 (one of the most robust
    effects in EEG). If this FAILS, the EEG cache is broken and every EgoBrain
    null to date is meaningless -- nothing downstream is interpretable.

  TEST B -- EEG movement envelope vs visual motion, WIDE-lag cross-correlation.
    Egocentric visual motion is driven by head movement, which injects a broadband
    movement/EMG-artifact envelope into the EEG. Cross-correlate the two over lags
    of +/- minutes (the ERP study only swept +/-2 s). Three outcomes:
      * peak at lag ~0, significant   -> sync is fine; EEG<->video coupling is real
                                          (weak); the "EEG ignored" story is about
                                          conditional info given the anchor, not sync.
      * peak at a consistent large lag -> global sync offset (fixable; rescues the
                                          whole video-WM line).
      * no peak anywhere (but TEST A passes) -> coupling genuinely ~0 at this
                                          montage/preproc; the target is the ceiling.

CPU only. Reuses the stitcher / power helpers from calibrate_erp_latency.py so the
clip keying is identical to the rest of the project. Appends a verdict block to
outputs/eval_tables.md.

Run:
  conda run -n cbramod python -u scripts/verify_egobrain_sync.py
  conda run -n cbramod python -u scripts/verify_egobrain_sync.py --subjects P0001,P0002 --workers 2
"""
from __future__ import annotations

import os

# Cap BLAS threads before numpy loads its backend (per-worker parallelism handles
# the fan-out; nested BLAS threads would oversubscribe).
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import importlib.util
import json
import sys
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _load_cal():
    """Import calibrate_erp_latency by file path (its module-level imports are
    numpy/scipy only; datasets/h5py are lazy inside functions)."""
    path = os.path.join(REPO, 'scripts', 'calibrate_erp_latency.py')
    spec = importlib.util.spec_from_file_location('_cal_erp', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAL = _load_cal()
FS = CAL.FS                       # 200 Hz
GRID_S = CAL.GRID_S               # 0.2 s
GRID_SAMPLES = int(round(GRID_S * FS))   # 40 samples / slot


# --------------------------------------------------------------------------- #
# Mann-Whitney AUC (pure): P(pos > neg)
# --------------------------------------------------------------------------- #
def mann_whitney_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    u = r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


# --------------------------------------------------------------------------- #
# TEST A -- eyes open/closed occipital alpha
# --------------------------------------------------------------------------- #
def alpha_epoch_feats(eeg: np.ndarray, occ_idx, events, epoch_s: float = 2.0,
                      edge_trim_s: float = 1.0):
    """Per-2s-epoch log mean occipital 8-13 Hz power, split by eyes state.

    Returns (pairs, pooled_open, pooled_closed) where pairs is a list of
    (open_feats, closed_feats) for adjacently-recorded open/closed blocks
    (drift-free within a pair)."""
    if len(occ_idx) == 0:
        return [], np.array([]), np.array([])
    filt = CAL.bandpass_filtfilt(eeg[occ_idx], FS, 8.0, 13.0)     # (n_occ, T)
    cs = CAL.power_cumsum(filt)                                   # (n_occ, T+1)
    T = eeg.shape[1]
    w = int(round(epoch_s * FS))

    def feats_for(ev):
        s0 = int(round((ev['start_s'] + edge_trim_s) * FS))
        s1 = int(round((ev['end_s'] - edge_trim_s) * FS))
        if s1 - s0 < w:
            return np.array([])
        centers = np.arange(s0 + w // 2, s1 - w // 2, w, dtype=np.float64)
        if centers.size == 0:
            return np.array([])
        mp, ok = CAL.windowed_mean_power(cs, centers, w)          # (n_occ, n)
        mp = mp[:, ok]
        if mp.size == 0:
            return np.array([])
        return np.log(mp.mean(axis=0) + 1e-12)                    # (n_epochs,)

    opens = [e for e in events if 'eyesopen' in e['label']]
    closes = [e for e in events if 'eyesclose' in e['label']]
    opens.sort(key=lambda e: e['start_s'])
    closes.sort(key=lambda e: e['start_s'])

    pairs = []
    for o, c in zip(opens, closes):        # pair by recording order (adjacent blocks)
        fo, fc = feats_for(o), feats_for(c)
        if fo.size and fc.size:
            pairs.append((fo, fc))
    pooled_o = np.concatenate([feats_for(o) for o in opens]) if opens else np.array([])
    pooled_c = np.concatenate([feats_for(c) for c in closes]) if closes else np.array([])
    pooled_o = pooled_o[np.isfinite(pooled_o)] if pooled_o.size else pooled_o
    pooled_c = pooled_c[np.isfinite(pooled_c)] if pooled_c.size else pooled_c
    return pairs, pooled_o, pooled_c


# --------------------------------------------------------------------------- #
# TEST B -- EEG movement envelope vs visual motion, wide-lag cross-correlation
# --------------------------------------------------------------------------- #
def slot_power_envelope(eeg: np.ndarray, lo: float, hi: float, n_slots: int):
    """Mean band power per 0.2 s grid slot, averaged over all channels. Returns
    (n_slots,) with the same slot indexing as the motion / hand-label grids
    (slot s == EEG samples [s*40, (s+1)*40))."""
    filt = CAL.bandpass_filtfilt(eeg, FS, lo, hi)                 # (C, T)
    cs = CAL.power_cumsum(filt)                                   # (C, T+1)
    centers = (np.arange(n_slots) + 0.5) * GRID_SAMPLES
    mp, ok = CAL.windowed_mean_power(cs, centers, GRID_SAMPLES)   # (C, n_slots)
    env = np.log(mp.mean(axis=0) + 1e-12)                        # broadband-ish
    env[~ok] = np.nan
    return env


def xcorr_wide(a: np.ndarray, b: np.ndarray, max_lag: int, hp_slots: int):
    """NaN-aware normalized cross-correlation ccf[lag] ~ corr(a[t], b[t+lag]).

    Both series are slow-drift high-passed (centered MA, zero group delay) then
    globally z-scored; the per-lag numerator and overlap count are computed by
    FFT correlation. Positive lag == b LAGS a (b shifted later). Returns
    (lags (2*max_lag+1,), ccf, n_overlap)."""
    from scipy.signal import correlate
    a = CAL.centered_ma_highpass(np.asarray(a, float), hp_slots).ravel()
    b = CAL.centered_ma_highpass(np.asarray(b, float), hp_slots).ravel()
    L = min(a.size, b.size)
    a, b = a[:L], b[:L]
    ma = np.isfinite(a).astype(np.float64)
    mb = np.isfinite(b).astype(np.float64)

    def z(x, m):
        v = x[m > 0]
        mu, sd = v.mean(), v.std()
        sd = sd if sd > 1e-12 else 1.0
        out = np.where(m > 0, (x - mu) / sd, 0.0)
        return out
    az, bz = z(a, ma), z(b, mb)
    # full cross-correlation: corr(a, b)[k] = sum_t a[t] * b[t - (k-(L-1))]
    num = correlate(az, bz, mode='full', method='fft')
    cnt = correlate(ma, mb, mode='full', method='fft')
    center = L - 1
    lags = np.arange(-max_lag, max_lag + 1)
    idx = center + lags                    # lag L>0 -> a[t]*b[t+L]? verified below
    # scipy correlate(a,b)[center + k] = sum_t a[t+k]*b[t]  == corr(a[t], b[t-k]).
    # We want corr(a[t], b[t+lag]) = sum_t a[t]*b[t+lag] = value at k=-lag.
    idx = center - lags
    num_l = num[idx]
    cnt_l = np.clip(cnt[idx], 1.0, None)
    ccf = num_l / cnt_l
    # only trust lags with adequate overlap
    good = cnt[idx] >= 0.4 * L
    ccf = np.where(good, ccf, np.nan)
    return lags, ccf, cnt[idx]


def null_peak_dist(a, b, max_lag, hp_slots, n_null=200, min_shift_s=60.0, seed=0):
    """Null distribution of max|ccf| under circular shifts of b by >= min_shift_s.
    A real fixed lag survives; spurious structure does not."""
    rng = np.random.default_rng(seed)
    a = CAL.centered_ma_highpass(np.asarray(a, float), hp_slots).ravel()
    b = CAL.centered_ma_highpass(np.asarray(b, float), hp_slots).ravel()
    L = min(a.size, b.size)
    a, b = a[:L], b[:L]
    min_shift = int(round(min_shift_s / GRID_S))
    peaks = np.empty(n_null)
    for i in range(n_null):
        sh = rng.integers(min_shift, L - min_shift)
        _, ccf, _ = xcorr_wide(a, np.roll(b, sh), max_lag, hp_slots)
        peaks[i] = np.nanmax(np.abs(ccf))
    return peaks


# --------------------------------------------------------------------------- #
# Per-subject worker
# --------------------------------------------------------------------------- #
def process_subject(sub, cfg):
    out = {'sub': sub}
    try:
        eeg, ch_names, meta = CAL.load_continuous_eeg(cfg['data_dir'], sub)
    except Exception as e:                                        # noqa: BLE001
        out['error'] = f'load_eeg: {e}'
        return out
    events = meta.get('events', [])
    gidx = CAL.channel_group_index(ch_names)
    occ = gidx.get('occipital', [])

    # ---- TEST A: eyes open/closed occipital alpha ----
    try:
        pairs, po, pc = alpha_epoch_feats(eeg, occ, events)
        pair_aucs = [mann_whitney_auc(fc, fo) for (fo, fc) in pairs]  # closed>open
        out['alpha'] = {
            'n_occ': len(occ),
            'n_pairs': len(pairs),
            'pair_aucs': [round(float(x), 4) for x in pair_aucs],
            'pair_auc_mean': float(np.nanmean(pair_aucs)) if pair_aucs else float('nan'),
            'pooled_auc': mann_whitney_auc(pc, po),
            'closed_minus_open_logpow': (float(np.nanmedian(pc) - np.nanmedian(po))
                                         if po.size and pc.size else float('nan')),
            'n_open': int(po.size), 'n_closed': int(pc.size),
        }
    except Exception as e:                                        # noqa: BLE001
        out['alpha'] = {'error': str(e)}

    # ---- TEST B: EEG movement envelope vs visual motion ----
    try:
        from datasets.egobrain_motion import load_or_compute_motion
        m = load_or_compute_motion(cfg['emb_grid_dir'], sub, cfg['step_slots'],
                                   metric='l1', space='patch',
                                   frames_grid_dir=cfg['frames_grid_dir'])
        if m is None:
            out['sync'] = {'error': 'no motion cache'}
        else:
            m = np.log1p(np.clip(np.asarray(m, float), 0, None))
            n_slots = m.size
            max_lag = int(round(cfg['lag_s'] / GRID_S))
            hp = int(round(cfg['hp_s'] / GRID_S))
            res_bands = {}
            for tag, (lo, hi) in cfg['bands'].items():
                env = slot_power_envelope(eeg, lo, hi, n_slots)
                lags, ccf, _ = xcorr_wide(env, m, max_lag, hp)
                k = int(np.nanargmax(np.abs(ccf)))
                peak_lag = float(lags[k] * GRID_S)
                peak_r = float(ccf[k])
                r0 = float(ccf[np.abs(lags) == 0][0])
                if tag == cfg['null_band']:
                    nd = null_peak_dist(env, m, max_lag, hp,
                                        n_null=cfg['n_null'], seed=1)
                    p = float((np.sum(nd >= abs(peak_r)) + 1) / (nd.size + 1))
                    null95 = float(np.nanpercentile(nd, 95))
                else:
                    p, null95 = None, None
                res_bands[tag] = {
                    'peak_lag_s': round(peak_lag, 3),
                    'peak_abs_r': round(abs(peak_r), 5),
                    'peak_r': round(peak_r, 5),
                    'r_at_lag0': round(r0, 5),
                    'p_vs_null': (round(p, 4) if p is not None else None),
                    'null_p95_absr': (round(null95, 5) if null95 is not None else None),
                }
            out['sync'] = {'n_slots': int(n_slots), 'bands': res_bands}
    except Exception as e:                                        # noqa: BLE001
        import traceback
        out['sync'] = {'error': f'{e}', 'tb': traceback.format_exc()[-800:]}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/EgoBrain')
    ap.add_argument('--emb_grid_dir',
                    default='data/EgoBrain/cache_embeddings_grid_facebook_dinov2-base_g0.2_sz224')
    ap.add_argument('--frames_grid_dir',
                    default='data/EgoBrain/cache_frames_grid_facebook_dinov2-base_g0.2_sz224')
    ap.add_argument('--subjects', default='all',
                    help='"all" or comma list e.g. P0001,P0002')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--lag_s', type=float, default=300.0, help='+/- cross-corr lag window (s)')
    ap.add_argument('--hp_s', type=float, default=10.0, help='slow-drift high-pass span (s)')
    ap.add_argument('--step_slots', type=int, default=5, help='motion horizon in 0.2s slots')
    ap.add_argument('--n_null', type=int, default=200)
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--no_append', action='store_true')
    args = ap.parse_args()

    if args.subjects == 'all':
        subs = sorted(d for d in os.listdir(os.path.join(args.data_dir, f'cache_eeg_{FS}hz'))
                      if d.startswith('P') and
                      os.path.isdir(os.path.join(args.data_dir, f'cache_eeg_{FS}hz', d)))
    else:
        subs = [s.strip() for s in args.subjects.split(',') if s.strip()]

    cfg = {
        'data_dir': args.data_dir, 'emb_grid_dir': args.emb_grid_dir,
        'frames_grid_dir': args.frames_grid_dir, 'lag_s': args.lag_s,
        'hp_s': args.hp_s, 'step_slots': args.step_slots, 'n_null': args.n_null,
        'bands': {'broad': (2.0, 45.0), 'high': (20.0, 45.0)},
        'null_band': 'high',
    }
    print(f'[verify] {len(subs)} subjects, lag +/-{args.lag_s}s, workers={args.workers}',
          flush=True)

    results = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_subject, s, cfg): s for s in subs}
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                _print_one(r)
    else:
        for s in subs:
            r = process_subject(s, cfg)
            results.append(r)
            _print_one(r)

    results.sort(key=lambda r: r['sub'])
    summary = summarize(results, cfg)
    _print_summary(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    date = datetime.now().strftime('%Y-%m-%d')
    jpath = os.path.join(args.out_dir, f'verify_egobrain_sync_{date}.json')
    with open(jpath, 'w') as f:
        json.dump({'summary': summary, 'per_subject': results, 'cfg': cfg}, f, indent=2)
    print(f'[verify] wrote {jpath}', flush=True)
    if not args.no_append:
        _append_table(args.out_dir, date, summary)


def _print_one(r):
    a = r.get('alpha', {})
    s = r.get('sync', {})
    if 'error' in r:
        print(f"  {r['sub']}: ERROR {r['error']}", flush=True)
        return
    auc = a.get('pair_auc_mean', float('nan'))
    hb = s.get('bands', {}).get('high', {}) if 'bands' in s else {}
    print(f"  {r['sub']}: alphaAUC={auc:.3f} (n_pairs={a.get('n_pairs')}) | "
          f"sync[high] peak_lag={hb.get('peak_lag_s','?')}s |r|={hb.get('peak_abs_r','?')} "
          f"r0={hb.get('r_at_lag0','?')} p={hb.get('p_vs_null','?')}", flush=True)


def summarize(results, cfg):
    aucs = [r['alpha']['pair_auc_mean'] for r in results
            if 'alpha' in r and np.isfinite(r['alpha'].get('pair_auc_mean', np.nan))]
    aucs = np.array(aucs, float)
    nb = cfg['null_band']
    lags, r0s, absr, ps = [], [], [], []
    for r in results:
        hb = r.get('sync', {}).get('bands', {}).get(nb)
        if hb:
            lags.append(hb['peak_lag_s'])
            r0s.append(hb['r_at_lag0'])
            absr.append(hb['peak_abs_r'])
            if hb['p_vs_null'] is not None:
                ps.append(hb['p_vs_null'])
    lags = np.array(lags, float); r0s = np.array(r0s, float)
    absr = np.array(absr, float); ps = np.array(ps, float)
    n_sig = int(np.sum(ps < 0.05)) if ps.size else 0
    near0 = int(np.sum(np.abs(lags) <= 1.0)) if lags.size else 0
    return {
        'n_subjects': len(results),
        'alpha': {
            'n': int(aucs.size),
            'auc_median': float(np.median(aucs)) if aucs.size else float('nan'),
            'auc_min': float(aucs.min()) if aucs.size else float('nan'),
            'frac_auc_gt_0.9': float(np.mean(aucs > 0.9)) if aucs.size else float('nan'),
            'frac_auc_gt_0.8': float(np.mean(aucs > 0.8)) if aucs.size else float('nan'),
        },
        'sync_null_band': nb,
        'sync': {
            'n': int(lags.size),
            'peak_lag_median_s': float(np.median(lags)) if lags.size else float('nan'),
            'peak_lag_iqr_s': (float(np.percentile(lags, 75) - np.percentile(lags, 25))
                               if lags.size else float('nan')),
            'frac_peak_within_1s': float(near0 / lags.size) if lags.size else float('nan'),
            'r_at_lag0_median': float(np.median(r0s)) if r0s.size else float('nan'),
            'peak_absr_median': float(np.median(absr)) if absr.size else float('nan'),
            'n_sig_p05': n_sig,
            'frac_sig_p05': float(n_sig / ps.size) if ps.size else float('nan'),
        },
    }


def _verdicts(summary):
    a = summary['alpha']; s = summary['sync']
    va = ('PASS' if a['auc_median'] > 0.9 else
          'WEAK' if a['auc_median'] > 0.8 else 'FAIL')
    # sync interpretation
    if s['n'] == 0:
        vs = 'NO_MOTION'
    elif s['frac_sig_p05'] >= 0.5 and s['frac_peak_within_1s'] >= 0.5:
        vs = 'SYNC_OK_COUPLING_REAL'
    elif s['frac_sig_p05'] >= 0.5 and abs(s['peak_lag_median_s']) > 2.0:
        vs = 'SYNC_OFFSET'          # consistent large nonzero lag -> fixable
    else:
        vs = 'NO_COUPLING'          # nothing survives the null
    return va, vs


def _print_summary(summary):
    va, vs = _verdicts(summary)
    a = summary['alpha']; s = summary['sync']
    print('\n' + '=' * 70)
    print('TEST A  eyes-closed vs open occipital alpha (no video/sync):')
    print(f'  AUC median={a["auc_median"]:.3f}  min={a["auc_min"]:.3f}  '
          f'frac>0.9={a["frac_auc_gt_0.9"]:.2f}  ->  {va}')
    print(f'TEST B  EEG[{summary["sync_null_band"]}] envelope vs visual motion, '
          f'xcorr +/-lag:')
    print(f'  peak_lag median={s["peak_lag_median_s"]:.2f}s  IQR={s["peak_lag_iqr_s"]:.2f}s  '
          f'frac|lag|<=1s={s["frac_peak_within_1s"]:.2f}')
    print(f'  r@lag0 median={s["r_at_lag0_median"]:.4f}  peak|r| median={s["peak_absr_median"]:.4f}  '
          f'sig(p<.05)={s["n_sig_p05"]}/{s["n"]}  ->  {vs}')
    print('=' * 70, flush=True)


def _append_table(out_dir, date, summary):
    va, vs = _verdicts(summary)
    a = summary['alpha']; s = summary['sync']
    lines = [
        f'\n## {date} — EgoBrain EEG↔video sync + eyes-alpha positive control\n',
        'Two decisive measurement checks (CPU, `scripts/verify_egobrain_sync.py`), motivated by '
        '`video_offset_s=0.0` never having been verified and the ERP-latency study failing its own '
        'visual positive control. TEST A needs no video/sync; TEST B sweeps ±lag far past the ERP '
        "study's ±2 s.\n",
        '| check | metric | value | verdict |',
        '|---|---|---|---|',
        f'| A: eyes-closed>open occipital 8–13 Hz | AUC median (min) | '
        f'{a["auc_median"]:.3f} ({a["auc_min"]:.3f}), frac>0.9={a["frac_auc_gt_0.9"]:.2f} | **{va}** |',
        f'| B: EEG[{summary["sync_null_band"]}] env × visual motion | peak-lag median (IQR) | '
        f'{s["peak_lag_median_s"]:.2f}s ({s["peak_lag_iqr_s"]:.2f}s), frac|lag|≤1s={s["frac_peak_within_1s"]:.2f} | **{vs}** |',
        f'| B: significance | sig(p<.05) / peak\\|r\\| med / r@0 med | '
        f'{s["n_sig_p05"]}/{s["n"]} / {s["peak_absr_median"]:.4f} / {s["r_at_lag0_median"]:.4f} |  |',
        '',
        'Reading: TEST A **FAIL** ⇒ EEG cache/epoching broken, every EgoBrain null is moot. '
        'TEST B **SYNC_OFFSET** (consistent large nonzero peak lag) ⇒ the `video_offset_s=0.0` '
        'assumption is wrong and rescues the video-WM line; **SYNC_OK_COUPLING_REAL** ⇒ sync fine, '
        'weak-but-real coupling (the "EEG ignored" gap is conditional-info, not sync); '
        '**NO_COUPLING** with A passing ⇒ instrument is fine but EEG carries ~no head-motion signal '
        'at this montage/preproc → target is the ceiling.\n',
    ]
    path = os.path.join(out_dir, 'eval_tables.md')
    with open(path, 'a') as f:
        f.write('\n'.join(lines))
    print(f'[verify] appended verdict to {path}', flush=True)


if __name__ == '__main__':
    main()
