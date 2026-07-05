"""Offline extractor: per-slot HAND-MOVEMENT intensity for EgoBrain on a
CONTINUOUS, time-keyed grid (one value every ``grid_s`` seconds of EEG-clock
time), one per-subject HDF5 file.

This is the knob-agnostic successor to :mod:`datasets.egobrain_hand_labels`.
The clip-keyed extractor stored intensities at the fixed ``(clip, window)``
positions implied by ``window_s / stride_s / erp_latency_s / n_windows /
clip_s`` — which are exactly the knobs :class:`EgoBrainDataset` bakes into the
LEGACY clip path. Under ``--egobrain_use_frame_grid`` the dataset instead
samples EEG windows at arbitrary 0.2 s-snapped offsets across the whole
recording and looks up each window's frame by ABSOLUTE EEG-clock slot; the
clip-keyed labels can't be indexed that way, so the aux objective silently
trains on nothing. This extractor keys the labels the SAME way the grid frames
are keyed (slot ``k`` <-> EEG-clock time ``k*grid_s``), so
``EgoBrainDataset._read_grid_hand_labels`` can read them at each window's frame
slot — exactly as ``_read_grid_embeddings`` reads the grid DINOv2 embeddings.

Design (two passes per subject):

  1. WiLoR pass (expensive, GPU). Decode the video frame at each grid slot
     (slot ``k`` = EEG time ``k*grid_s`` -> video time ``k*grid_s -
     video_offset``, same chapter routing as
     :mod:`datasets.egobrain_extract_frames_grid`), run WiLoR, and store the
     per-slot left/right :class:`Hand` track, the ego-motion shift between
     consecutive slots, and the real per-pair dt. Streaming / batched so peak
     memory is one decode batch, not the whole recording.

  2. Aggregation pass (cheap, numpy). For each slot ``s`` the stored intensity
     is the ego-compensated, scale-normalised hand speed over a FIXED
     ``hand_ref_s``-second window of slots CENTRED on ``s`` — computed by the
     same tested :func:`datasets.egobrain_hand_labels._hand_speed` over that
     slot window. Because a slice of the per-slot ``track/ego/dts`` arrays is a
     valid ``_hand_speed`` input, this reuses the clip-path motion logic
     verbatim; only the KEYING (per absolute slot, not per ``(clip, window)``)
     differs. ``hand_ref_s`` is the ONLY window knob baked into the cache — it
     is independent of the training window/stride/erp/n_windows, matching the
     grid philosophy.

Output layout (a NEW cache dir; never touches raw data, the EEG cache, the
frame cache, or the clip-keyed hand cache — and refuses to overwrite a
per-subject file built with a different config unless ``--overwrite``):

    <data_dir>/cache_hand_labels_grid_<backend>_g<grid_s>_r<hand_ref_s>_fs<fs>/
        grid_label_mapping.json     provenance + config
        <subject>.h5
            left_intensity   (n_slots,) float32 hand-lengths/s (NaN = unmeasurable)
            right_intensity  (n_slots,) float32
            left_det_frac    (n_slots,) float32 frac slots in the ref window with left hand seen
            right_det_frac   (n_slots,) float32
            has_video        (n_slots,) bool    this slot's frame decoded in-bounds
            attrs: full config + backend + grid_s + hand_ref_s + n_slots

The slot grid (``n_slots``, slot<->time mapping, chapter routing) is IDENTICAL
to :mod:`datasets.egobrain_extract_frames_grid` so the label at slot ``k`` lines
up with the frame / embedding at slot ``k``. The dir name deliberately omits
``vision_encoder`` / ``frame_size``: hand intensities come from the RAW GoPro
video (WiLoR does its own preprocessing), not the EEG model's vision encoder.

Runs in the dedicated ``wilor`` env (see sh/install_wilor.sh), NOT cbramod.
Invoke as a DIRECT script so datasets/__init__.py (training-only deps) isn't
imported — this module needs none of it:
    python datasets/egobrain_extract_hand_labels_grid.py \\
        --data_dir data/EgoBrain --subjects all \\
        --grid_s 0.2 --hand_ref_s 1.0 --fs_out 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

# Reuse the clip-keyed extractor's pure + backend helpers WITHOUT triggering
# datasets/__init__.py (which imports training-only deps absent in the wilor
# env). When run as a direct script the sibling module is importable as a
# top-level name (its dir is sys.path[0]); when imported as a package member
# (cbramod unit tests) the qualified path works and __init__ is harmless.
try:                                                        # package import (tests)
    from datasets.egobrain_hand_labels import (
        Hand, parse_wilor_frame, _hand_speed, estimate_ego_shift,
        HandEstimator, build_chapters, resolve_chapter,
    )
except ImportError:                                         # direct script (wilor env)
    from egobrain_hand_labels import (                      # noqa: F401
        Hand, parse_wilor_frame, _hand_speed, estimate_ego_shift,
        HandEstimator, build_chapters, resolve_chapter,
    )


_FORMAT_VERSION = 1


# ===========================================================================
# Pure grid geometry / aggregation — no video/model deps, unit-tested.
# ===========================================================================
def n_slots_for(n_clips: int, clip_s: float, grid_s: float,
                margin_s: float) -> int:
    """Number of grid slots covering the continuous recording, IDENTICAL to
    :mod:`datasets.egobrain_extract_frames_grid` (``floor((n_clips*clip_s +
    margin)/grid_s) + 1``). Slot ``k`` <-> EEG-clock time ``k*grid_s`` so the
    label slot indexes line up 1:1 with the frame / embedding grid."""
    total_duration_s = n_clips * clip_s
    return int(np.floor((total_duration_s + margin_s) / grid_s)) + 1


def ref_window_radius(hand_ref_s: float, grid_s: float) -> int:
    """Half-width ``R`` (in slots) of the intensity reference window: the
    per-slot intensity averages the ego-compensated hand speed over the
    ``2R+1`` slots centred on the slot (``2R`` consecutive-slot pairs, i.e.
    ``2R*grid_s`` s of motion ~= ``hand_ref_s``). ``R>=1`` so at least one
    pair exists."""
    return max(1, int(round(hand_ref_s / (2.0 * grid_s))))


def aggregate_grid(left_track: list, right_track: list,
                   ego: list, dts: list, has_video: np.ndarray, *,
                   grid_s: float, hand_ref_s: float) -> dict:
    """Per-slot windowed hand intensities from the per-slot tracks.

    ``left_track[k]`` / ``right_track[k]`` are the :class:`Hand` (or ``None``)
    detected at slot ``k``; ``ego[k]`` is the background shift for the pair
    ``(k-1, k)`` (or ``None``); ``dts[k]`` is that pair's real elapsed time (or
    ``None`` when the slots aren't truly adjacent / a frame is missing). All
    four are indexed by ABSOLUTE slot, so a slice ``[a:b]`` is a valid
    ``_hand_speed`` input for the window ``[a, b)`` — this is why the clip-path
    motion logic is reused verbatim.

    Intensity at slot ``s`` = mean ego-compensated, scale-normalised speed over
    the ``[s-R, s+R]`` slot window (``R = ref_window_radius``); ``NaN`` when the
    window has no measurable pair (never seen / only non-adjacent frames),
    distinct from ``0.0`` = seen and still — so the reader can mask it out.
    Returns arrays keyed by slot: ``left_intensity`` / ``right_intensity``
    (float32, NaN-where-unmeasurable), ``left_det_frac`` / ``right_det_frac``
    (float32, fraction of the window's slots with that hand detected), and the
    passed-through ``has_video``.
    """
    n = len(left_track)
    R = ref_window_radius(hand_ref_s, grid_s)
    li = np.full(n, np.nan, np.float32)
    ri = np.full(n, np.nan, np.float32)
    ldf = np.zeros(n, np.float32)
    rdf = np.zeros(n, np.float32)
    for s in range(n):
        a = max(0, s - R)
        b = min(n, s + R + 1)
        lt, rt = left_track[a:b], right_track[a:b]
        eg, dt = ego[a:b], dts[a:b]
        l_mean, _, l_pairs = _hand_speed(lt, eg, dt)
        r_mean, _, r_pairs = _hand_speed(rt, eg, dt)
        li[s] = l_mean if l_pairs >= 1 else np.nan
        ri[s] = r_mean if r_pairs >= 1 else np.nan
        span = b - a
        ldf[s] = sum(h is not None for h in lt) / span if span else 0.0
        rdf[s] = sum(h is not None for h in rt) / span if span else 0.0
    return {'left_intensity': li, 'right_intensity': ri,
            'left_det_frac': ldf, 'right_det_frac': rdf,
            'has_video': np.asarray(has_video, dtype=bool)}


def default_grid_out_dir(data_dir: str, backend: str, grid_s: float,
                         hand_ref_s: float, fs_out: int) -> str:
    """Cache dir; encodes only the knobs that change the stored values.
    Deliberately omits vision_encoder/frame_size (labels come from the raw
    video) and window/stride/erp/n_windows/clip_s (the point of the grid)."""
    return os.path.join(
        data_dir,
        f'cache_hand_labels_grid_{backend}_g{grid_s}_r{hand_ref_s}_fs{fs_out}')


# All config keys that determine the stored arrays — used to refuse silently
# reusing a sidecar built with a different config (mirrors the clip-keyed
# _config_mismatch). move_thresh/min_det_frac are NOT here: the grid cache
# stores only the CONTINUOUS intensity (no discrete label), which is
# threshold-independent.
_CONFIG_KEYS = ('backend', 'grid_s', 'hand_ref_s', 'fs_out', 'clip_s',
                'margin_s', 'handedness_source', 'ego_compensate',
                'max_frame_width')


def _config_mismatch(out_path: str, cfg: dict) -> Optional[str]:
    import h5py
    try:
        with h5py.File(out_path, 'r') as h:
            attrs = dict(h.attrs)
    except OSError as e:
        return f'unreadable ({e})'
    if int(attrs.get('format_version', -1)) != _FORMAT_VERSION:
        return f'format_version {attrs.get("format_version")} != {_FORMAT_VERSION}'
    for k in _CONFIG_KEYS:
        if k not in attrs:
            return f'{k} missing'
        a = attrs[k]
        if isinstance(a, bytes):
            a = a.decode()
        b = cfg[k]
        if isinstance(b, float):
            if abs(float(a) - b) >= 1e-9:
                return f'{k}: {a} != {b}'
        elif a != b:
            return f'{k}: {a!r} != {b!r}'
    return None


# ===========================================================================
# Per-subject processing.
# ===========================================================================
def _decode_slot_batch(ch_path: str, ch_fps: float, ch_n: int,
                       slot_times: list, max_frame_width: int, threads: int):
    """Decode one batch of ``(slot, t_in_chapter)`` frames from ``ch_path``.

    Returns ``(frames, fidx)`` parallel to ``slot_times``: ``frames[j]`` is the
    decoded RGB array (downscaled to ``max_frame_width``) or ``None`` (out of
    bounds), and ``fidx[j]`` is the integer video-frame index or ``None`` — the
    caller uses ``fidx`` for the real per-pair dt and to drop duplicate frames.
    Recreates the reader per batch so decord's frame buffer can't grow
    unboundedly over the 0.2 s grid (same OOM guard as
    egobrain_extract_frames_grid)."""
    import decord
    decord.bridge.set_bridge('native')
    fis = []
    for _slot, t in slot_times:
        fi = int(round(t * ch_fps)) if t is not None and t >= 0 else None
        fis.append(fi if (fi is not None and 0 <= fi < ch_n) else None)
    uniq = sorted({fi for fi in fis if fi is not None})
    frames_out = [None] * len(slot_times)
    if uniq:
        vr = decord.VideoReader(ch_path, num_threads=threads)
        batch = vr.get_batch(uniq).asnumpy()                # (U, H, W, 3) RGB
        del vr
        if max_frame_width and batch.shape[2] > max_frame_width:
            import cv2
            scale = max_frame_width / batch.shape[2]
            nh = int(round(batch.shape[1] * scale))
            batch = np.stack([cv2.resize(f, (max_frame_width, nh)) for f in batch])
        row = {fi: r for r, fi in enumerate(uniq)}
        for j, fi in enumerate(fis):
            if fi is not None:
                frames_out[j] = batch[row[fi]]
    return frames_out, fis


def process_subject(sub: str, cfg: dict, estimator: Optional[HandEstimator]
                    ) -> dict:
    data_dir = cfg['data_dir']
    eeg_meta_path = os.path.join(
        data_dir, f'cache_eeg_{cfg["fs_out"]}hz', sub, 'clips.json')
    if not os.path.exists(eeg_meta_path):
        return {'subject': sub, 'status': f'missing EEG cache {eeg_meta_path}'}
    with open(eeg_meta_path) as f:
        meta = json.load(f)
    if int(meta['fs_out']) != cfg['fs_out']:
        return {'subject': sub, 'status': f"fs_out mismatch {meta['fs_out']}"}
    clip_s = float(meta['clip_s'])
    if abs(clip_s - cfg['clip_s']) > 1e-6:
        return {'subject': sub,
                'status': f"clip_s mismatch cache={clip_s} cfg={cfg['clip_s']}"}
    n_clips = int(meta['n_clips'])
    video_info = meta.get('video')
    if video_info is None:
        return {'subject': sub, 'status': 'no_video'}         # P0025-P0040
    chapters = build_chapters(video_info, data_dir)
    if not chapters:
        return {'subject': sub, 'status': 'no usable video chapters'}
    for ch in chapters:
        if not os.path.exists(ch['path']):
            return {'subject': sub, 'status': f"chapter missing {ch['path']}"}
    video_offset_s = float(video_info.get('video_offset_s', 0.0))

    out_path = os.path.join(cfg['out_dir'], f'{sub}.h5')
    if os.path.exists(out_path) and not cfg['overwrite']:
        mism = _config_mismatch(out_path, cfg)
        if mism is None:
            return {'subject': sub, 'status': 'skip', 'n_clips': n_clips}
        return {'subject': sub,
                'status': f'EXISTS with different config ({mism}); pass '
                          f'--overwrite to rebuild or use a fresh --out_dir'}

    grid_s = float(cfg['grid_s'])
    n_slots = n_slots_for(n_clips, clip_s, grid_s, cfg['margin_s'])

    # Bucket every slot into its chapter (slot k -> EEG time k*grid_s -> video
    # time k*grid_s - video_offset). IDENTICAL routing to the frame grid.
    ch_buckets: dict[str, list] = {ch['path']: [] for ch in chapters}
    for k in range(n_slots):
        res = resolve_chapter(chapters, video_offset_s, k * grid_s)
        if res is None:
            continue
        ch, t_in_ch = res
        ch_buckets[ch['path']].append((k, t_in_ch))

    # Per-slot state (absolute-slot indexed).
    left_track: list = [None] * n_slots
    right_track: list = [None] * n_slots
    ego: list = [None] * n_slots            # ego[k]  = shift for pair (k-1, k)
    dts: list = [None] * n_slots            # dts[k]  = real dt for pair (k-1, k)
    has_video = np.zeros(n_slots, dtype=bool)

    import cv2
    import gc
    decode_batch = int(cfg['decode_batch'])
    threads = int(cfg['decord_threads'])
    total_wanted = sum(len(w) for w in ch_buckets.values())
    n_ch = sum(1 for w in ch_buckets.values() if w)
    done = 0
    ci = 0
    print(f'[{sub}] WiLoR over {total_wanted}/{n_slots} grid frames, '
          f'{n_ch} chapter(s)', flush=True)
    import decord
    for ch in chapters:
        wanted = ch_buckets[ch['path']]                       # ascending by slot
        if not wanted:
            continue
        ci += 1
        vr_meta = decord.VideoReader(ch['path'], num_threads=threads)
        ch_fps = float(vr_meta.get_avg_fps())
        ch_n = int(len(vr_meta))
        assert ch_fps > 0, f"bad fps {ch['path']}"
        del vr_meta
        gc.collect()
        # Rolling previous VALID slot (gray + hand bboxes + frame index), reset
        # per chapter so no pair straddles a chapter boundary or an OOB gap.
        prev_gray = prev_fidx = prev_slot = None
        for start in range(0, len(wanted), decode_batch):
            chunk = wanted[start:start + decode_batch]
            frames, fis = _decode_slot_batch(
                ch['path'], ch_fps, ch_n, chunk, cfg['max_frame_width'], threads)
            for j, (slot, _t) in enumerate(chunk):
                frame, fi = frames[j], fis[j]
                if frame is None or estimator is None:
                    # OOB / undecodable: leave hand None, break the chain so the
                    # next valid slot doesn't pair across the gap.
                    prev_gray = prev_fidx = prev_slot = None
                    continue
                has_video[slot] = True
                outs = estimator.predict_frame(frame)
                l, r = parse_wilor_frame(outs, cfg['handedness_source'],
                                         img_w=frame.shape[1])
                left_track[slot] = l
                right_track[slot] = r
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if prev_slot == slot - 1:
                    # Truly-adjacent slots: record the pair (ego shift + dt).
                    if cfg['ego_compensate'] and prev_gray is not None:
                        boxes = [h.bbox for h in (l, r,
                                 left_track[slot - 1], right_track[slot - 1])
                                 if h is not None]
                        ego[slot] = estimate_ego_shift(prev_gray, gray, boxes)
                    if (prev_fidx is not None and fi is not None
                            and fi != prev_fidx and ch_fps > 0):
                        dts[slot] = (fi - prev_fidx) / ch_fps
                prev_gray, prev_fidx, prev_slot = gray, fi, slot
            done += len(chunk)
            if (start // decode_batch) % 5 == 0 or done == total_wanted:
                pct = 100.0 * done / max(1, total_wanted)
                print(f'[{sub}] ch {ci}/{n_ch}  {done}/{total_wanted} '
                      f'frames ({pct:.0f}%)', flush=True)
        gc.collect()

    arrays = aggregate_grid(left_track, right_track, ego, dts, has_video,
                            grid_s=grid_s, hand_ref_s=cfg['hand_ref_s'])
    _write_subject(out_path, cfg, arrays, sub, n_clips, n_slots, clip_s,
                   video_offset_s)
    return {'subject': sub, 'status': 'ok', 'n_clips': n_clips,
            'n_slots': n_slots}


def _write_subject(out_path, cfg, arrays, sub, n_clips, n_slots, clip_s,
                   video_offset_s):
    """Atomic, no-clobber HDF5 write (PID-unique tmp + os.replace)."""
    import h5py
    os.makedirs(cfg['out_dir'], exist_ok=True)
    tmp = out_path + f'.{os.getpid()}.tmp'
    with h5py.File(tmp, 'w') as h:
        for k, v in arrays.items():
            h.create_dataset(k, data=v, compression='gzip', compression_opts=4)
        h.attrs['subject'] = sub
        h.attrs['n_clips'] = n_clips
        h.attrs['n_slots'] = n_slots
        h.attrs['clip_s'] = clip_s
        h.attrs['video_offset_s'] = video_offset_s
        for k in ('backend', 'grid_s', 'hand_ref_s', 'fs_out', 'margin_s',
                  'handedness_source', 'ego_compensate', 'max_frame_width'):
            h.attrs[k] = cfg[k]
        h.attrs['format_version'] = _FORMAT_VERSION
    os.replace(tmp, out_path)


def load_hand_labels_grid(out_dir: str, sub: str) -> dict:
    """Read a subject's grid sidecar back as numpy arrays + attrs."""
    import h5py
    with h5py.File(os.path.join(out_dir, f'{sub}.h5'), 'r') as h:
        d = {k: h[k][...] for k in h.keys()}
        d['attrs'] = dict(h.attrs)
    return d


# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--subjects', required=True, help='comma-sep ids or "all"')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--backend', default='wilor', choices=['wilor'])
    # Grid geometry — grid_s MUST match the training --egobrain_frame_grid_s and
    # the frame/embedding grid so the label slot indexes line up.
    p.add_argument('--grid_s', type=float, default=0.2,
                   help='time-grid spacing (s); MUST match the frame grid '
                        '(--egobrain_frame_grid_s) so slot k aligns 1:1')
    p.add_argument('--hand_ref_s', type=float, default=1.0,
                   help='reference window (s) the per-slot intensity averages '
                        'hand speed over, centred on the slot; the only window '
                        'knob baked into the cache (training window/stride/erp/'
                        'n_windows do NOT affect it)')
    p.add_argument('--margin_s', type=float, default=2.0,
                   help='extra seconds of slots past the EEG end (must match '
                        'the frame grid so n_slots agrees)')
    p.add_argument('--clip_s', type=float, default=4.0,
                   help='EEG cache clip length (only to read n_clips/clip_s + '
                        'size the slot grid; validated against clips.json)')
    p.add_argument('--fs_out', type=int, default=200)
    # Labelling (same WiLoR knobs as the clip-keyed extractor).
    p.add_argument('--handedness_source', default='hybrid',
                   choices=['model', 'position', 'hybrid'])
    p.add_argument('--no_ego_compensate', action='store_true',
                   help='disable background-flow ego-motion compensation')
    p.add_argument('--max_frame_width', type=int, default=1280,
                   help='downscale wider frames for the detector (0 = native)')
    p.add_argument('--decode_batch', type=int, default=256,
                   help='slots per decord get_batch (bounds decode memory)')
    p.add_argument('--decord_threads', type=int, default=0,
                   help='threads per decord VideoReader (0=auto=all node cores; '
                        'set to the per-job CPU allocation when sharding)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry_run', action='store_true',
                   help='validate config + slot grid without loading WiLoR')
    args = p.parse_args()

    import cv2
    cv2.setNumThreads(0)
    if not args.dry_run:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()

    out_dir = args.out_dir or default_grid_out_dir(
        args.data_dir, args.backend, args.grid_s, args.hand_ref_s, args.fs_out)
    cfg = dict(
        data_dir=args.data_dir, out_dir=out_dir, backend=args.backend,
        grid_s=args.grid_s, hand_ref_s=args.hand_ref_s, margin_s=args.margin_s,
        clip_s=args.clip_s, fs_out=args.fs_out,
        handedness_source=args.handedness_source,
        ego_compensate=not args.no_ego_compensate,
        max_frame_width=args.max_frame_width, decode_batch=args.decode_batch,
        decord_threads=args.decord_threads, overwrite=args.overwrite)

    if args.subjects.lower() == 'all':
        import re
        subjects = sorted(d for d in os.listdir(args.data_dir)
                          if re.match(r'^P\d{4}$', d)
                          and os.path.isdir(os.path.join(args.data_dir, d)))
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'grid_label_mapping.json'), 'w') as f:
        json.dump({'note': 'time-keyed (grid) per-slot continuous hand-movement '
                           'intensity; slot k <-> EEG time k*grid_s, aligned 1:1 '
                           'with the frame/embedding grid. Read by '
                           'EgoBrainDataset._read_grid_hand_labels under '
                           '--egobrain_use_frame_grid.',
                   'config': {k: cfg[k] for k in cfg if k != 'data_dir'}}, f,
                  indent=2)
    print(f'[hand-labels-grid] {len(subjects)} subject(s) → {out_dir}')

    estimator = None
    if not args.dry_run:
        estimator = HandEstimator(args.backend, args.device, args.dtype)

    results = []
    for sub in subjects:
        if args.dry_run:
            results.append({'subject': sub, 'status': 'dry_run'}); continue
        try:
            results.append(process_subject(sub, cfg, estimator))
        except Exception as e:                                # keep going
            results.append({'subject': sub, 'status': f'ERROR: {e}'})
            import traceback; traceback.print_exc()
    for r in results:
        if r['status'] not in {'ok', 'skip', 'no_video', 'dry_run'}:
            print(f"[err] {r['subject']}: {r['status']}", file=sys.stderr)
    ok = sum(r['status'] == 'ok' for r in results)
    print(f"[hand-labels-grid] done: ok={ok} "
          f"skip={sum(r['status']=='skip' for r in results)} "
          f"no_video={sum(r['status']=='no_video' for r in results)} "
          f"err={sum(r['status'] not in {'ok','skip','no_video','dry_run'} for r in results)}")
    failed = [r['subject'] for r in results
              if r['status'] not in {'ok', 'skip', 'no_video', 'dry_run'}]
    if failed:
        print(f'[FATAL] {len(failed)} subject(s) failed: '
              f'{",".join(sorted(failed))}. The grid hand-label cache is '
              f'INCOMPLETE; fix the errors above and re-run before training '
              f'with --aux_hand_pred + --egobrain_hand_grid_dir.',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
