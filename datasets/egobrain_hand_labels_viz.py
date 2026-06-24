"""Visual sanity-check for the EgoBrain hand-label pipeline.

Runs the SAME WiLoR pass + motion aggregation as datasets.egobrain_hand_labels
on a handful of windows, but instead of writing labels it renders an annotated
montage per window: each sampled frame with the detected hand keypoints +
skeleton (blue = left, red = right), bounding box, and handedness, plus a
caption with the derived discrete label and the continuous drasticness /
per-hand intensity. Open the PNGs to manually verify the labels look right.

    python datasets/egobrain_hand_labels_viz.py \\
        --data_dir data/EgoBrain --subject P0001 --n_examples 8 \\
        --out_dir outputs/hand_label_viz --device cuda

By default it spreads --n_examples clips evenly across the session (different
activities → varied hand use); pass --clips 50,300,800 to pick specific ones.
Geometry args must match the extraction config (defaults already do).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Works both as `datasets.egobrain_hand_labels_viz` (package, e.g. under
# cbramod) and as a direct script `python datasets/egobrain_hand_labels_viz.py`
# (the wilor env lacks the training-only deps that datasets/__init__ pulls in).
try:
    from datasets.egobrain_hand_labels import (
        LABEL_NAMES, window_frame_times, build_chapters, resolve_chapter,
        parse_wilor_frame, aggregate_window, estimate_ego_shift, HandEstimator,
        _decode_clip_windows,
    )
except ModuleNotFoundError:
    from egobrain_hand_labels import (
        LABEL_NAMES, window_frame_times, build_chapters, resolve_chapter,
        parse_wilor_frame, aggregate_window, estimate_ego_shift, HandEstimator,
        _decode_clip_windows,
    )

# Standard MANO/OpenPose 21-joint hand topology (wrist=0; 5 fingers ×4).
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
          (0, 5), (5, 6), (6, 7), (7, 8),
          (0, 9), (9, 10), (10, 11), (11, 12),
          (0, 13), (13, 14), (14, 15), (15, 16),
          (0, 17), (17, 18), (18, 19), (19, 20)]
_BLUE = (255, 80, 0)     # left  (BGR)
_RED = (0, 80, 255)      # right (BGR)
_VIS_H = 340             # per-frame display height


def _draw_hand(vis, hand, color, tag):
    import cv2
    if hand is None:
        return
    kp = hand.kp
    for a, b in _EDGES:
        pa, pb = tuple(np.round(kp[a]).astype(int)), tuple(np.round(kp[b]).astype(int))
        cv2.line(vis, pa, pb, color, 2, cv2.LINE_AA)
    for j in range(kp.shape[0]):
        cv2.circle(vis, tuple(np.round(kp[j]).astype(int)), 3, color, -1, cv2.LINE_AA)
    x0, y0, x1, y1 = np.round(hand.bbox).astype(int)
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
    cv2.putText(vis, tag, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def _montage(frames, lefts, rights, caption):
    """hstack the window's annotated frames under a caption header (BGR)."""
    import cv2
    tiles = []
    for f, l, r in zip(frames, lefts, rights):
        if f is None:
            tile = np.full((_VIS_H, int(_VIS_H * 4 / 3), 3), 40, np.uint8)
            cv2.putText(tile, 'no frame', (10, _VIS_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        else:
            vis = cv2.cvtColor(f, cv2.COLOR_RGB2BGR).copy()
            _draw_hand(vis, l, _BLUE, 'L')
            _draw_hand(vis, r, _RED, 'R')
            scale = _VIS_H / vis.shape[0]
            tile = cv2.resize(vis, (int(round(vis.shape[1] * scale)), _VIS_H))
        tiles.append(tile)
        tiles.append(np.full((_VIS_H, 3, 3), 255, np.uint8))   # separator
    strip = np.hstack(tiles[:-1])
    header = np.full((70, strip.shape[1], 3), 255, np.uint8)
    for k, line in enumerate(caption):
        cv2.putText(header, line, (10, 28 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([header, strip])


def _decode_window(vr, ch_fps, ch_n, t_local, max_frame_width):
    """Decode one window's frames + their integer frame indices (None = OOB)."""
    frames, fidx = _decode_clip_windows(vr, ch_fps, ch_n, {0: t_local},
                                        max_frame_width)
    return frames[0], fidx[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--subject', default='P0001')
    p.add_argument('--clips', default=None, help='comma-sep clip indices; '
                   'default spreads --n_examples across the session')
    p.add_argument('--n_examples', type=int, default=8)
    p.add_argument('--windows', default='0', help='comma-sep window indices')
    p.add_argument('--only_labels', default=None,
                   help='comma-sep label names (left_hand,right_hand,both_hands,'
                        'none) — only WRITE montages for these (all rows still '
                        'printed). Lets you scan many clips but save only the '
                        'ones you care about, e.g. --only_labels left_hand')
    p.add_argument('--out_dir', default='outputs/hand_label_viz')
    p.add_argument('--window_s', type=float, default=1.0)
    p.add_argument('--stride_s', type=float, default=1.0)
    p.add_argument('--clip_s', type=float, default=4.0)
    p.add_argument('--fs_out', type=int, default=200)
    p.add_argument('--erp_latency_s', type=float, default=0.5)
    p.add_argument('--frames_per_window', type=int, default=7)
    p.add_argument('--move_thresh', type=float, default=0.15)
    p.add_argument('--min_det_frac', type=float, default=0.4)
    p.add_argument('--handedness_source', default='hybrid',
                   choices=['model', 'position', 'hybrid'])
    p.add_argument('--no_ego_compensate', action='store_true')
    p.add_argument('--max_frame_width', type=int, default=1280)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    args = p.parse_args()

    # Initialise CUDA BEFORE importing decord: importing decord first corrupts
    # torch's lazy CUDA init and segfaults in model.to(cuda) (decord×torch).
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
    import cv2
    import decord
    cv2.setNumThreads(0)        # avoid cv2/OpenMP×torch threading conflicts

    meta_path = os.path.join(args.data_dir, f'cache_eeg_{args.fs_out}hz',
                             args.subject, 'clips.json')
    with open(meta_path) as f:
        meta = json.load(f)
    n_clips = int(meta['n_clips'])
    chapters = build_chapters(meta['video'], args.data_dir)
    offset = float(meta['video'].get('video_offset_s', 0.0))

    if args.clips:
        clips = [int(x) for x in args.clips.split(',')]
    else:
        clips = sorted(set(np.linspace(0, n_clips - 1, args.n_examples).astype(int).tolist()))
    windows = [int(x) for x in args.windows.split(',')]
    K = args.frames_per_window
    out_dir = os.path.join(args.out_dir, args.subject)
    os.makedirs(out_dir, exist_ok=True)

    est = HandEstimator('wilor', args.device, args.dtype)
    win_samp = int(round(args.window_s * args.fs_out))
    stride_samp = int(round(args.stride_s * args.fs_out))

    only_labels = ({s.strip() for s in args.only_labels.split(',') if s.strip()}
                   if args.only_labels else None)

    def _fmt(v):
        return 'nan' if v != v else f'{v:.2f}'

    print(f'{"clip":>5} {"win":>3}  {"label":>10}  {"L_int":>6} {"R_int":>6} '
          f'{"drastic":>7}  {"Ldet":>4} {"Rdet":>4}', flush=True)

    # Bucket every (clip, window) by its owning chapter, then process ONE
    # chapter (one open VideoReader) at a time — caching all 4K readers
    # exhausts resources and crashes mid-run. Mirrors the extractor's loop.
    work: dict[str, list] = {}
    for c in clips:
        for i in windows:
            ts = window_frame_times(
                c, i, clip_s=args.clip_s, window_s=args.window_s,
                stride_s=args.stride_s, fs_out=args.fs_out,
                erp_latency_s=args.erp_latency_s, frames_per_window=K)
            center_t = (c * args.clip_s
                        + (i * stride_samp + win_samp / 2) / args.fs_out
                        + args.erp_latency_s)
            res = resolve_chapter(chapters, offset, center_t)
            if res is None:
                print(f'{c:>5} {i:>3}  (no video at this time)', flush=True)
                continue
            ch, _ = res
            t_local = [(r[1] if (r is not None and r[0]['path'] == ch['path'])
                        else -1.0)
                       for r in (resolve_chapter(chapters, offset, t) for t in ts)]
            work.setdefault(ch['path'], []).append((c, i, t_local))

    import gc
    ch_start = {c['path']: c['start_s'] for c in chapters}
    n_written = 0
    for ch_path in sorted(work, key=lambda p: ch_start.get(p, 0.0)):
        vr = decord.VideoReader(ch_path)
        ch_fps = float(vr.get_avg_fps()); ch_n = int(len(vr))
        for c, i, t_local in work[ch_path]:
            try:
                frames, fidx = _decode_window(vr, ch_fps, ch_n, t_local,
                                              args.max_frame_width)
                lefts, rights, grays = [], [], []
                for f in frames:
                    if f is None:
                        lefts.append(None); rights.append(None)
                        grays.append(None); continue
                    l, r = parse_wilor_frame(est.predict_frame(f),
                                             args.handedness_source, img_w=f.shape[1])
                    lefts.append(l); rights.append(r)
                    grays.append(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY))
                ego = [None]
                for k in range(1, len(frames)):
                    if (args.no_ego_compensate or grays[k - 1] is None
                            or grays[k] is None):
                        ego.append(None); continue
                    boxes = [h.bbox for h in (lefts[k - 1], lefts[k],
                                              rights[k - 1], rights[k])
                             if h is not None]
                    ego.append(estimate_ego_shift(grays[k - 1], grays[k], boxes))
                dts = [None]
                for k in range(1, len(frames)):
                    if (fidx[k] is None or fidx[k - 1] is None
                            or fidx[k] == fidx[k - 1] or ch_fps <= 0):
                        dts.append(None)
                    else:
                        dts.append((fidx[k] - fidx[k - 1]) / ch_fps)
                m = aggregate_window(lefts, rights, ego, dts=dts,
                                     move_thresh=args.move_thresh,
                                     min_det_frac=args.min_det_frac)
                name = LABEL_NAMES[m['label']]
                print(f'{c:>5} {i:>3}  {name:>10}  {_fmt(m["left_intensity"]):>6} '
                      f'{_fmt(m["right_intensity"]):>6} {m["drasticness"]:>7.2f}  '
                      f'{m["left_det_frac"]:>4.2f} {m["right_det_frac"]:>4.2f}',
                      flush=True)
                if only_labels and name not in only_labels:
                    continue
                cap = [
                    f'{args.subject} clip {c} win {i}   ->  LABEL = {name}',
                    f'L_int={_fmt(m["left_intensity"])} R_int={_fmt(m["right_intensity"])} '
                    f'drastic={m["drasticness"]:.2f}  (move_thr={args.move_thresh})',
                ]
                png = os.path.join(out_dir, f'clip{c:04d}_win{i}_{name}.png')
                cv2.imwrite(png, _montage(frames, lefts, rights, cap))
                n_written += 1
            except Exception as e:
                print(f'{c:>5} {i:>3}  ERROR {type(e).__name__}: {e}', flush=True)
        del vr; gc.collect()
    print(f'\nwrote {n_written} montage(s) to {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
