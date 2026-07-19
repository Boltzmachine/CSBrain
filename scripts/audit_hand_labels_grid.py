"""Audit the time-keyed grid hand-label cache (format 2: RAW, unsmoothed).

Checks, in order of importance:

1. COMPLETENESS — every video subject has an h5; per-subject ``n_slots`` matches
   the EEG ``clips.json`` AND the frame grid AND the embedding grid exactly
   (this is the alignment property: label slot k <-> frame slot k).
2. CONSISTENCY — ``has_video`` is identical to the frame grid's ``has_image``
   (the labels flag exactly the frames the model sees).
3. RAW-FORMAT INVARIANTS — format_version==2 / smoothing='none' /
   speed_interval='forward'; the last slot is NaN (no forward pair); a finite
   speed at slot s implies BOTH frames s and s+1 decoded.
4. NO SMOOTHING — the lag-1 autocorrelation of the stored speed must be near the
   intrinsic autocorrelation of hand motion, NOT the ~0.87 of the old 5-tap
   moving average. A window-averaged array would show r(lag1) >> r(lag4).
5. SANITY — coverage, detection rates, intensity percentiles.

Run:
    conda run -n cbramod python scripts/audit_hand_labels_grid.py
    conda run -n cbramod python scripts/audit_hand_labels_grid.py --hand_dir <dir>
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np


def n_slots_for(n_clips, clip_s, grid_s, margin_s):
    return int(np.floor((n_clips * clip_s + margin_s) / grid_s)) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--hand_dir', default=None)
    p.add_argument('--frame_dir', default=None)
    p.add_argument('--emb_dir', default=None)
    p.add_argument('--grid_s', type=float, default=0.2)
    p.add_argument('--margin_s', type=float, default=2.0)
    p.add_argument('--fs_out', type=int, default=200)
    a = p.parse_args()

    D = a.data_dir
    HG = a.hand_dir or os.path.join(
        D, f'cache_hand_labels_grid_wilor_g{a.grid_s}_raw_fs{a.fs_out}')
    FG = a.frame_dir or os.path.join(
        D, f'cache_frames_grid_facebook_dinov2-base_g{a.grid_s}_sz224')
    EG = a.emb_dir or os.path.join(
        D, f'cache_embeddings_grid_facebook_dinov2-base_g{a.grid_s}_sz224')

    all_subs = sorted(d for d in os.listdir(D)
                      if len(d) == 5 and d[0] == 'P' and d[1:].isdigit()
                      and os.path.isdir(os.path.join(D, d)))
    video_subs = []
    for s in all_subs:
        mp = os.path.join(D, f'cache_eeg_{a.fs_out}hz', s, 'clips.json')
        if os.path.exists(mp) and json.load(open(mp)).get('video') is not None:
            video_subs.append(s)
    have = sorted(os.path.basename(x)[:-3] for x in glob.glob(os.path.join(HG, '*.h5')))
    print(f'cache: {HG}')
    print(f'subjects with video: {len(video_subs)}   h5 present: {len(have)}')
    missing = sorted(set(video_subs) - set(have))
    if missing:
        print(f'  !! MISSING {len(missing)}: {",".join(missing)}')

    problems, rows = [], []
    tot = totv = totlf = totrf = 0
    li_all, ri_all, ac = [], [], []
    for s in have:
        with h5py.File(os.path.join(HG, f'{s}.h5'), 'r') as h:
            at = dict(h.attrs)
            # Check the format BEFORE touching v2-only datasets, so a legacy v1
            # (window-averaged) cache reports the version cleanly instead of
            # dying on a missing key.
            fv = int(at.get('format_version', -1))
            if fv != 2:
                problems.append(
                    f'{s}: format_version={fv} != 2 — this is the legacy '
                    f'WINDOW-AVERAGED (smoothed) layout; rebuild with '
                    f'datasets/egobrain_extract_hand_labels_grid.py')
                continue
            missing_ds = {'left_intensity', 'right_intensity', 'has_video',
                          'left_det', 'right_det'} - set(h.keys())
            if missing_ds:
                problems.append(f'{s}: missing datasets {sorted(missing_ds)}')
                continue
            li, ri = h['left_intensity'][...], h['right_intensity'][...]
            hv = h['has_video'][...].astype(bool)
            ld, rd = h['left_det'][...].astype(bool), h['right_det'][...].astype(bool)
        n = len(li)

        # (3) raw-format invariants
        smoothing = at.get('smoothing', b'')
        if isinstance(smoothing, bytes):
            smoothing = smoothing.decode()
        if smoothing != 'none':
            problems.append(f"{s}: smoothing attr = {smoothing!r} != 'none'")
        if not np.isnan(li[-1]) or not np.isnan(ri[-1]):
            problems.append(f'{s}: last slot must be NaN (no forward pair)')
        fin = np.isfinite(li)
        idx = np.where(fin[:-1])[0]
        if idx.size and not (hv[idx].all() and hv[idx + 1].all()):
            bad = int((~(hv[idx] & hv[idx + 1])).sum())
            problems.append(f'{s}: {bad} finite speeds whose pair lacks a decoded frame')

        # (1) slot-count alignment
        meta = json.load(open(os.path.join(D, f'cache_eeg_{a.fs_out}hz', s, 'clips.json')))
        exp = n_slots_for(int(meta['n_clips']), float(meta['clip_s']), a.grid_s, a.margin_s)
        fn = en = None
        if os.path.exists(os.path.join(FG, f'{s}.h5')):
            with h5py.File(os.path.join(FG, f'{s}.h5'), 'r') as h:
                fn = h['has_image'].shape[0]
                himg = h['has_image'][...].astype(bool)
            # (2) has_video must equal the frame grid's has_image
            nm = int((himg != hv).sum()) if fn == n else -1
            if nm != 0:
                problems.append(f'{s}: has_video != frame has_image ({nm} slots)')
        if os.path.exists(os.path.join(EG, f'{s}.h5')):
            with h5py.File(os.path.join(EG, f'{s}.h5'), 'r') as h:
                en = h['has_image'].shape[0]
        if not (n == exp and (fn is None or fn == n) and (en is None or en == n)):
            problems.append(f'{s}: n_slots={n} exp={exp} frame={fn} emb={en} MISALIGNED')

        # (4) no-smoothing probe: lag autocorrelation on a long finite run
        x = li[fin]
        if x.size > 5000:
            x = (x[:200000] - x[:200000].mean()) / (x[:200000].std() + 1e-9)
            ac.append([1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, 6)])

        tot += n; totv += int(hv.sum())
        totlf += int(fin.sum()); totrf += int(np.isfinite(ri).sum())
        li_all.append(li[fin]); ri_all.append(ri[np.isfinite(ri)])
        rows.append((s, n, exp, fn, en, hv.mean(), fin.mean(),
                     np.isfinite(ri).mean(), ld.mean(), rd.mean()))

    if not rows:
        print('\n=== PROBLEMS ===')
        for x in problems:
            print('  !', x)
        print('\nNo readable v2 subject files — nothing further to audit.')
        return 1

    print(f"\n{'sub':6} {'n':>6} {'exp':>6} {'frm':>6} {'emb':>6} {'vid%':>5} "
          f"{'Lfin%':>6} {'Rfin%':>6} {'Ldet%':>6} {'Rdet%':>6}")
    for s, n, exp, fn, en, v, lf, rf, ld, rd in rows:
        print(f'{s:6} {n:>6} {exp:>6} {str(fn):>6} {str(en):>6} {100*v:>4.0f}% '
              f'{100*lf:>5.1f}% {100*rf:>5.1f}% {100*ld:>5.1f}% {100*rd:>5.1f}%')

    li_all = np.concatenate(li_all); ri_all = np.concatenate(ri_all)
    print(f'\n=== aggregate ===\ntotal slots {tot:,}  has_video {totv:,} '
          f'({100*totv/max(tot,1):.2f}%)')
    print(f'finite L {totlf:,} ({100*totlf/max(tot,1):.1f}%)   '
          f'finite R {totrf:,} ({100*totrf/max(tot,1):.1f}%)')
    for nm, arr in (('LEFT ', li_all), ('RIGHT', ri_all)):
        q = np.percentile(arr, [5, 25, 50, 75, 90, 99])
        print(f'{nm} speed (hand-lengths/s): ' +
              ' '.join(f'p{p}={v:.3f}' for p, v in zip([5, 25, 50, 75, 90, 99], q)))

    if ac:
        m = np.mean(ac, axis=0)
        print('\n=== NO-SMOOTHING probe: lag autocorrelation of stored speed ===')
        for k, r in enumerate(m):
            print(f'  lag {k} ({0.2*k:.1f}s): r={r:.3f}')
        print(f'  smoothed 5-tap reference was r(lag1)=0.865, r(lag4)=0.462.')
        ratio = m[1] / max(m[4], 1e-9)
        if m[1] > 0.75 and ratio > 1.6:
            problems.append(f'SMOOTHING SUSPECTED: r(lag1)={m[1]:.3f} '
                            f'(ratio to lag4 = {ratio:.2f}) looks window-averaged')
        else:
            print(f'  -> r(lag1)={m[1]:.3f}: consistent with RAW per-pair speed '
                  f'(no moving average).')

    print('\n=== PROBLEMS ===')
    if problems:
        for x in problems:
            print('  !', x)
    else:
        print('  none — complete, aligned, raw/unsmoothed, invariants hold')
    return 1 if (problems or missing) else 0


if __name__ == '__main__':
    raise SystemExit(main())
