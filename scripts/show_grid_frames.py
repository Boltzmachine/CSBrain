"""Dump a contact sheet of consecutive 0.2 s grid frames from the EgoBrain
frame-grid cache so we can eyeball how much the scene actually changes."""
import argparse, os
import h5py, numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--h5', default='data/EgoBrain/cache_frames_grid_facebook_dinov2-base_g0.2_sz224/P0001.h5')
p.add_argument('--starts', default='')          # comma sep slot indices; '' = auto
p.add_argument('--n', type=int, default=15)      # frames per row
p.add_argument('--rows', type=int, default=4)
p.add_argument('--thumb', type=int, default=140)
p.add_argument('--out', required=True)
p.add_argument('--pick', default='spread', choices=['spread', 'motion'])
a = p.parse_args()

with h5py.File(a.h5, 'r') as h:
    grid_s = float(h.attrs['grid_s'])
    n_slots = int(h.attrs['n_slots'])
    has = h['has_image'][:]
    valid = np.flatnonzero(has)
    print(f'{os.path.basename(a.h5)}: n_slots={n_slots} grid_s={grid_s} '
          f'valid={valid.size} ({valid.size*grid_s/60:.1f} min of video)')

    if a.starts:
        starts = [int(s) for s in a.starts.split(',')]
    elif a.pick == 'spread':
        lo, hi = valid[0], valid[-1] - a.n
        starts = [int(lo + (hi - lo) * f) for f in
                  np.linspace(0.1, 0.9, a.rows)]
    else:  # motion: scan a subsample for the biggest pixel deltas
        probe = valid[::25]
        probe = probe[probe + a.n < valid[-1]]
        sc = []
        for k in probe[:400]:
            f = h['frames'][k:k + a.n:3].astype(np.float32)
            sc.append(np.abs(np.diff(f, axis=0)).mean())
        order = np.argsort(sc)[::-1]
        starts = [int(probe[i]) for i in order[:a.rows]]

    thumb, pad, lab = a.thumb, 4, 22
    W = a.n * (thumb + pad) + pad
    H = a.rows * (thumb + pad + lab) + pad
    sheet = Image.new('RGB', (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    for r, s0 in enumerate(starts):
        fr = h['frames'][s0:s0 + a.n]
        y = pad + r * (thumb + pad + lab)
        prev = None
        for c in range(a.n):
            im = Image.fromarray(fr[c]).resize((thumb, thumb), Image.BILINEAR)
            x = pad + c * (thumb + pad)
            sheet.paste(im, (x, y))
            cur = fr[c].astype(np.float32)
            if prev is None:
                txt = f't={s0*grid_s:.1f}s'
            else:
                dd = np.abs(cur - prev).mean()
                txt = f'+{c*grid_s:.1f}s  d={dd:.1f}'
            d.text((x + 2, y + thumb + 4), txt, fill=(210, 210, 215))
            prev = cur
    sheet.save(a.out)
    print('wrote', a.out, sheet.size)
