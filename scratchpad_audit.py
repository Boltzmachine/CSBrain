"""Audit the EgoBrain hand-label annotation cache for completeness + integrity."""
import os, glob, json
import numpy as np
import h5py

CACHE = 'data/EgoBrain/cache_hand_labels_wilor_w1.0s1.0_e0.5_nw2_k7_c4.0_fs200'
EXPECTED = [f'P{ i:04d}' for i in range(1, 25)]   # P0001..P0024 ship video
KEYS = ['label', 'left_intensity', 'right_intensity', 'left_active_frac',
        'right_active_frac', 'left_det_frac', 'right_det_frac', 'drasticness',
        'has_video']
NAMES = {-1: 'none', 0: 'left', 1: 'right', 2: 'both'}

files = sorted(glob.glob(f'{CACHE}/P*.h5'))
present = {os.path.basename(f)[:5] for f in files}
print(f'cache: {CACHE}')
print(f'files present: {len(files)} / {len(EXPECTED)} expected (P0001-P0024)')
missing = [s for s in EXPECTED if s not in present]
extra = [s for s in sorted(present) if s not in EXPECTED]
if missing: print(f'  MISSING: {missing}')
if extra:   print(f'  unexpected extra: {extra}')
print()

hdr = (f'{"subj":5} {"clips":>5} {"win":>3} {"vidfrac":>7} '
       f'{"none":>6} {"left":>5} {"right":>5} {"both":>5} '
       f'{"drastP50":>8} {"flaws"}')
print(hdr); print('-'*len(hdr))

agg = {k: 0 for k in NAMES}
agg_windows = 0
problems = []
for s in EXPECTED:
    f = f'{CACHE}/{s}.h5'
    if not os.path.exists(f):
        print(f'{s:5} {"--- not written ---"}'); continue
    flaws = []
    try:
        with h5py.File(f, 'r') as h:
            arrs = {k: h[k][...] for k in KEYS if k in h}
            attrs = dict(h.attrs)
            nclips = int(attrs.get('n_clips', -1))
        missing_keys = [k for k in KEYS if k not in arrs]
        if missing_keys: flaws.append(f'missing_keys={missing_keys}')
        lab = arrs['label']; nclip, nwin = lab.shape
        # shape consistency
        for k in KEYS:
            if k in arrs and arrs[k].shape != lab.shape:
                flaws.append(f'{k}.shape={arrs[k].shape}!={lab.shape}')
        if nclips != -1 and nclip != nclips:
            flaws.append(f'n_clips attr {nclips}!=array {nclip}')
        # label domain
        uniq = set(np.unique(lab).tolist())
        if 3 in uniq: flaws.append('LABEL 3 (both_feet) EMITTED!')
        bad = uniq - {-1, 0, 1, 2}
        if bad: flaws.append(f'bad_labels={bad}')
        counts = {v: int((lab == v).sum()) for v in NAMES}
        nwindows = lab.size
        hv = arrs['has_video']; vidfrac = float(hv.mean())
        # invariants: no-video windows must be label -1
        nv_nonneg = int(((~hv) & (lab != -1)).sum())
        if nv_nonneg: flaws.append(f'{nv_nonneg} no-video windows with label!=-1')
        # a labeled (0/1/2) window must have has_video True
        lab_no_vid = int(((lab >= 0) & (~hv)).sum())
        if lab_no_vid: flaws.append(f'{lab_no_vid} labeled windows w/o video')
        # all-none or all-novideo red flags
        if vidfrac == 0.0: flaws.append('has_video ALL FALSE')
        if counts[0]+counts[1]+counts[2] == 0: flaws.append('NO active labels at all')
        dr = arrs['drasticness']; fin = np.isfinite(dr)
        drp50 = float(np.nanmedian(dr[fin])) if fin.any() else float('nan')
        # drasticness should be finite wherever has_video and some detection
        for v in NAMES: agg[v] += counts[v]
        agg_windows += nwindows
        print(f'{s:5} {nclip:5d} {nwin:3d} {vidfrac:7.3f} '
              f'{counts[-1]:6d} {counts[0]:5d} {counts[1]:5d} {counts[2]:5d} '
              f'{drp50:8.3f} {";".join(flaws) if flaws else "ok"}')
        if flaws: problems.append((s, flaws))
    except Exception as e:
        print(f'{s:5} READ ERROR: {type(e).__name__}: {e}')
        problems.append((s, [f'read_error:{e}']))

print()
print('=== AGGREGATE over all subjects ===')
tot = sum(agg.values())
for v in NAMES:
    print(f'  {NAMES[v]:5}: {agg[v]:8d}  ({100*agg[v]/tot:.1f}%)' if tot else NAMES[v])
print(f'  total windows: {agg_windows}')
print()
if problems:
    print(f'=== {len(problems)} subject(s) WITH FLAGS ===')
    for s, fl in problems: print(f'  {s}: {fl}')
else:
    print('=== no integrity flags across all subjects ===')
