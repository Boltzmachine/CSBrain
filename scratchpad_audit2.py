"""Deeper audit: video-vs-EEG coverage + detection quality of 'none' windows."""
import os, glob, json
import numpy as np, h5py

CACHE = 'data/EgoBrain/cache_hand_labels_wilor_w1.0s1.0_e0.5_nw2_k7_c4.0_fs200'
EXP = [f'P{i:04d}' for i in range(1, 25)]

print(f'{"subj":5} {"eeg_min":>7} {"vid_min":>7} {"cover":>6} '
      f'{"anyDet%":>7} {"none_still%":>10} {"none_nodet%":>11}')
print('-'*64)
tot_w = tot_anydet = tot_none = tot_none_still = 0
flags = []
for s in EXP:
    f = f'{CACHE}/{s}.h5'
    if not os.path.exists(f):
        print(f'{s:5}  (no h5)'); continue
    meta = json.load(open(f'data/EgoBrain/cache_eeg_200hz/{s}/clips.json'))
    eeg_s = meta['n_clips'] * meta['clip_s']
    vid_s = sum((c.get('duration_s') or 0) for c in meta['video']['chapters'])
    with h5py.File(f, 'r') as h:
        lab = h['label'][...]
        ldf = h['left_det_frac'][...]; rdf = h['right_det_frac'][...]
        hv = h['has_video'][...]
    anydet = (ldf > 0) | (rdf > 0)             # at least one hand seen in window
    none_mask = lab == -1
    none_still = none_mask & anydet            # hands visible but below move thr
    none_nodet = none_mask & (~anydet)         # no hand detected at all
    nW = lab.size
    cover = vid_s / eeg_s
    a = float(anydet.mean())
    ns = int(none_still.sum()); nn = int(none_nodet.sum()); nnone = int(none_mask.sum())
    print(f'{s:5} {eeg_s/60:7.1f} {vid_s/60:7.1f} {cover:6.2f} '
          f'{100*a:7.1f} {100*ns/nnone if nnone else 0:10.1f} '
          f'{100*nn/nnone if nnone else 0:11.1f}')
    if cover < 1.0: flags.append(f'{s}: video<EEG (cover {cover:.2f}) but has_video={hv.mean():.3f}')
    if a < 0.5: flags.append(f'{s}: only {100*a:.0f}% windows have ANY hand detected')
    tot_w += nW; tot_anydet += int(anydet.sum())
    tot_none += nnone; tot_none_still += ns

print()
print('=== AGGREGATE ===')
print(f'  windows total: {tot_w}')
print(f'  with >=1 hand detected: {tot_anydet} ({100*tot_anydet/tot_w:.1f}%)')
print(f'  none windows: {tot_none}; of which hands-visible-but-still: '
      f'{tot_none_still} ({100*tot_none_still/tot_none:.1f}%), '
      f'no-hand-detected: {tot_none-tot_none_still} ({100*(tot_none-tot_none_still)/tot_none:.1f}%)')
print()
print('FLAGS:', flags if flags else 'none')
