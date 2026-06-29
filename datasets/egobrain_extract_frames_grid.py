"""Offline extractor: pre-decode EgoBrain video frames onto a CONTINUOUS,
time-keyed grid (one frame every ``grid_s`` seconds of EEG-clock time), one
per-subject HDF5 file.

This is the knob-agnostic successor to :mod:`datasets.egobrain_extract_frames`.
The old extractor cached frames at the fixed ``(clip, window)`` positions
implied by ``window_s / stride_s / erp_latency_s / n_windows / clip_s`` — so it
(a) only covered the first ``n_windows*stride_s`` seconds of every 4 s clip and
(b) re-baked on every change of those knobs. Here we instead key frames by
ABSOLUTE EEG-clock time on a uniform ``grid_s`` grid spanning the whole
recording, so:

* The cache is INDEPENDENT of ``window_s``, ``stride_s``, ``erp_latency_s``,
  ``n_windows`` and ``clip_s``. Those are applied at LOOKUP time in
  ``EgoBrainDataset`` (slot = round((window_centre + erp) / grid_s)).
* The dataset can sample EEG windows at any offset across the continuous
  recording (no 4 s clip boundary) and still find an aligned frame.

Frame content is byte-identical to the old extractor's: same HF resize +
center-crop (``_resize_shortest_then_center_crop``), same chapter routing,
same ``round(t_video * fps)`` index. Only the time grid the frames sit on
differs.

Layout written:
    <cache_dir>/<subject>.h5
        frames     (n_slots, H, W, 3) uint8 — slot k = EEG-clock time k*grid_s
        has_image  (n_slots,)         bool  — frame within video bounds
        attrs:
            vision_encoder, frame_size, grid_s, fs_out, video_offset_s,
            total_duration_s, n_slots, n_chapters, format_version

The dir name encodes only ``vision_encoder`` (-> resize/crop + frame_size),
``frame_size`` and ``grid_s`` — the knobs that actually change the bytes:
    cache_frames_grid_<enc>_g<grid_s>_sz<frame_size>/

Run (see sh/extract_frames_grid.sh):
    conda run -n cbramod python -m datasets.egobrain_extract_frames_grid \\
        --data_dir data/EgoBrain --subjects all \\
        --vision_encoder facebook/dinov2-base \\
        --grid_s 0.2 --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

from datasets.cinebrain_dataset import _frame_size_for
from datasets.egobrain_extract_frames import (
    _processor_resize_crop_config,
    _decode_frames_for_clip,
)


_FORMAT_VERSION = 1

_WORKER_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    _WORKER_CFG.update(cfg)


def _resolve_chapters(data_dir: str, video_info: dict) -> tuple[list[dict], float]:
    """Return (resolved_chapters, video_offset_s).

    ``resolved_chapters`` is the ordered chapter list with absolute paths,
    cumulative ``start_s`` and (possibly None) ``duration_s`` — identical to
    the bucketing :mod:`datasets.egobrain_extract_frames` does.
    """
    if 'chapters' in video_info:
        chapters = list(video_info['chapters'])
    elif 'path' in video_info:                                  # legacy fallback
        chapters = [{'path': video_info['path']}]
    else:
        raise ValueError('video meta has neither "chapters" nor "path"; '
                         'rerun egobrain_preprocess to refresh clips.json')
    resolved: list[dict] = []
    cum = 0.0
    for ch in chapters:
        ch_path = os.path.join(data_dir, ch['path'])
        if not os.path.exists(ch_path):
            raise FileNotFoundError(f'chapter missing: {ch_path}')
        entry = {'path': ch_path, 'start_s': cum,
                 'duration_s': ch.get('duration_s')}
        resolved.append(entry)
        cum = float('inf') if entry['duration_s'] is None else cum + float(
            entry['duration_s'])
    video_offset_s = float(video_info.get('video_offset_s', 0.0))
    return resolved, video_offset_s


def _route_chapter(resolved_chapters: list[dict], t_video: float
                   ) -> Optional[tuple[dict, float]]:
    """Map a VIDEO-clock timestamp to (chapter, t_in_chapter)."""
    if t_video < 0:
        return None
    for ch in resolved_chapters:
        dur = ch.get('duration_s')
        if dur is None or t_video < ch['start_s'] + dur:
            return ch, t_video - ch['start_s']
    return None


def _extract_subject(sub: str) -> dict:
    cfg = _WORKER_CFG
    cache_root = cfg['cache_dir']
    out_path = os.path.join(cache_root, f'{sub}.h5')

    eeg_meta_path = os.path.join(
        cfg['data_dir'], f'cache_eeg_{cfg["fs_out"]}hz', sub, 'clips.json')
    if not os.path.exists(eeg_meta_path):
        return {'subject': sub,
                'status': f'missing EEG cache {eeg_meta_path}; run '
                          f'egobrain_preprocess first'}
    with open(eeg_meta_path) as f:
        eeg_meta = json.load(f)
    n_clips = int(eeg_meta['n_clips'])
    clip_s = float(eeg_meta['clip_s'])
    # The continuous EEG timeline the dataset samples over is exactly the
    # clips concatenated back together: [0, n_clips * clip_s). Cover that span
    # plus a small margin so a window near the very end whose frame lands at
    # (centre + erp) just past the end still resolves to a slot (out-of-video
    # slots are simply has_image=False).
    total_duration_s = n_clips * clip_s
    grid_s = float(cfg['grid_s'])
    margin_s = float(cfg['margin_s'])
    n_slots = int(np.floor((total_duration_s + margin_s) / grid_s)) + 1

    video_info = eeg_meta.get('video')
    if video_info is None:
        # P0025-P0040 ship no video; mirror egobrain_extract_frames and skip.
        return {'subject': sub, 'status': 'no_video'}
    try:
        resolved_chapters, video_offset_s = _resolve_chapters(
            cfg['data_dir'], video_info)
    except (ValueError, FileNotFoundError) as e:
        return {'subject': sub, 'status': str(e)}

    target_size = int(cfg['frame_size'])
    shortest_edge, crop_size, resample = _processor_resize_crop_config(
        cfg['vision_encoder'])
    if int(cfg['frame_size']) != crop_size:
        crop_size = int(cfg['frame_size'])
    target_size = crop_size

    if os.path.exists(out_path) and not cfg['overwrite']:
        with h5py.File(out_path, 'r') as h:
            attrs = dict(h.attrs)

        def _match(key, want):
            a = attrs.get(key)
            if isinstance(a, np.bytes_):
                a = a.decode()
            if isinstance(a, bytes):
                a = a.decode()
            return a == want
        if (_match('vision_encoder', cfg['vision_encoder'])
                and _match('frame_size', target_size)
                and attrs.get('grid_s') is not None
                and abs(float(attrs['grid_s']) - grid_s) < 1e-9
                and int(attrs.get('n_slots', -1)) == n_slots
                and int(attrs.get('format_version', -1)) == _FORMAT_VERSION):
            return {'subject': sub, 'status': 'skip', 'n_slots': n_slots}

    # Bucket every slot into its chapter so each VideoReader is opened once.
    ch_buckets: dict[str, list[tuple[int, float]]] = {
        ch['path']: [] for ch in resolved_chapters}
    for k in range(n_slots):
        eeg_t = k * grid_s
        resolved = _route_chapter(resolved_chapters, eeg_t - video_offset_s)
        if resolved is None:
            continue
        ch, t_in_ch = resolved
        ch_buckets[ch['path']].append((k, t_in_ch))

    os.makedirs(cache_root, exist_ok=True)
    tmp_path = out_path + '.tmp'

    frames_buffer = np.zeros((n_slots, target_size, target_size, 3),
                             dtype=np.uint8)
    has_buffer = np.zeros((n_slots,), dtype=bool)

    import gc
    import decord
    from datasets.cinebrain_dataset import _VIDEO_READER_CACHE
    _VIDEO_READER_CACHE.clear()
    gc.collect()

    decode_batch = int(cfg['decode_batch'])
    threads = int(cfg['decord_threads'])
    # Within-subject progress: the outer per-subject tqdm is useless when
    # sharding one subject per job (0/1), and a 2 h recording decodes for tens
    # of minutes, so log decode progress over the grid frames themselves.
    total_wanted = sum(len(w) for w in ch_buckets.values())
    n_ch = sum(1 for w in ch_buckets.values() if w)
    done_slots = 0
    ci = 0
    print(f'[{sub}] decoding {total_wanted}/{n_slots} grid frames over '
          f'{n_ch} chapter(s)', flush=True)
    for ch_path, wanted in ch_buckets.items():
        if not wanted:
            continue
        ci += 1
        decord.bridge.set_bridge('native')
        # decord's internal frame buffer grows unboundedly across repeated
        # get_batch calls on ONE reader (tens of GB over a full 0.2 s grid ->
        # OOM). The 0.2 s grid issues ~150+ batches/subject (vs ~11 for the
        # legacy nw=2 cache, which is why only this one OOMs). Bound it by
        # recreating the reader PER decode batch — peak stays one batch of 4K
        # frames (~6 GB at decode_batch=256) + the per-subject output buffer.
        # Cap threads too (0=auto grabs all node cores -> oversubscription when
        # several readers run; pin to the job's CPU allocation).
        vr_meta = decord.VideoReader(ch_path, num_threads=threads)
        ch_fps = float(vr_meta.get_avg_fps())
        ch_n_frames = int(len(vr_meta))
        assert ch_fps > 0, f'bad fps for {ch_path}'
        del vr_meta
        gc.collect()
        for bi, start in enumerate(range(0, len(wanted), decode_batch)):
            chunk = wanted[start:start + decode_batch]
            slots = [w[0] for w in chunk]
            t_videos = [w[1] for w in chunk]
            vr = decord.VideoReader(ch_path, num_threads=threads)
            ch_frames, ch_ok = _decode_frames_for_clip(
                vr, ch_fps, t_videos, ch_n_frames,
                crop_size=target_size, shortest_edge=shortest_edge,
                resample=resample)
            del vr
            gc.collect()
            for j, slot in enumerate(slots):
                frames_buffer[slot] = ch_frames[j]
                has_buffer[slot] = ch_ok[j]
            done_slots += len(chunk)
            # Log every few batches (and at the end) so the SLURM log shows
            # live within-subject decode progress.
            if bi % 5 == 0 or done_slots == total_wanted:
                pct = 100.0 * done_slots / max(1, total_wanted)
                print(f'[{sub}] ch {ci}/{n_ch}  {done_slots}/{total_wanted} '
                      f'frames ({pct:.0f}%)', flush=True)

    with h5py.File(tmp_path, 'w') as h:
        h.create_dataset(
            'frames', data=frames_buffer, dtype='uint8',
            chunks=(1, target_size, target_size, 3),
            compression='gzip', compression_opts=4)
        h.create_dataset('has_image', data=has_buffer, dtype='bool')
        h.attrs['subject'] = sub
        h.attrs['vision_encoder'] = cfg['vision_encoder']
        h.attrs['frame_size'] = target_size
        h.attrs['grid_s'] = grid_s
        h.attrs['fs_out'] = int(cfg['fs_out'])
        h.attrs['clip_s'] = clip_s
        h.attrs['video_offset_s'] = video_offset_s
        h.attrs['total_duration_s'] = total_duration_s
        h.attrs['n_slots'] = n_slots
        h.attrs['n_chapters'] = len(resolved_chapters)
        h.attrs['format_version'] = _FORMAT_VERSION
    os.replace(tmp_path, out_path)
    return {'subject': sub, 'status': 'ok', 'n_slots': n_slots}


def _grid_cache_dir(data_dir: str, vision_encoder: str, grid_s: float,
                    frame_size: int) -> str:
    """Default cache dir; encodes only the byte-changing knobs (encoder ->
    resize/crop + frame_size, and grid_s). Deliberately omits window/stride/
    erp/n_windows/clip_s — the whole point of the time-keyed grid."""
    enc_slug = vision_encoder.replace('/', '_')
    return os.path.join(
        data_dir, f'cache_frames_grid_{enc_slug}_g{grid_s}_sz{frame_size}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--cache_dir', default=None,
                   help='defaults to '
                        '<data_dir>/cache_frames_grid_<encoder>_g<grid_s>_sz<sz>')
    p.add_argument('--subjects', required=True,
                   help='comma-separated subject ids, or "all"')
    p.add_argument('--vision_encoder', default='facebook/dinov2-base',
                   help='HF model id; sets resize/crop + frame size '
                        '(DINOv2 = 224, V-JEPA 2 = 256)')
    p.add_argument('--frame_size', type=int, default=None,
                   help='override frame size; default derived from encoder')
    p.add_argument('--grid_s', type=float, default=0.2,
                   help='time grid spacing in seconds (default 0.2 = the EEG '
                        'patch size; the lookup snaps window centres to this)')
    p.add_argument('--margin_s', type=float, default=2.0,
                   help='extra seconds of slots past the EEG end so end-of-'
                        'recording windows whose frame lands at centre+erp '
                        'still resolve (out-of-video slots are has_image=False)')
    p.add_argument('--fs_out', type=int, default=200,
                   help='matches the EEG preprocess --fs_out (only used to '
                        'locate the EEG cache + read clip_s/n_clips)')
    p.add_argument('--decode_batch', type=int, default=256,
                   help='slots per get_batch call (bounds decord memory)')
    p.add_argument('--decord_threads', type=int, default=0,
                   help='threads per decord VideoReader (0=auto=all node cores, '
                        'which oversubscribes when several readers run at once). '
                        'Set to the per-job CPU allocation when sharding subjects '
                        'across SLURM jobs.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    frame_size = (args.frame_size if args.frame_size is not None
                  else _frame_size_for(args.vision_encoder))
    cache_dir = args.cache_dir or _grid_cache_dir(
        args.data_dir, args.vision_encoder, args.grid_s, frame_size)

    if args.subjects.lower() == 'all':
        subjects = sorted(
            d for d in os.listdir(args.data_dir)
            if re.match(r'^P\d{4}$', d)
            and os.path.isdir(os.path.join(args.data_dir, d)))
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    print(f'extracting grid frames (grid_s={args.grid_s}) for '
          f'{len(subjects)} subject(s) → {cache_dir}')

    cfg = dict(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        vision_encoder=args.vision_encoder,
        frame_size=frame_size,
        grid_s=args.grid_s,
        margin_s=args.margin_s,
        fs_out=args.fs_out,
        decode_batch=args.decode_batch,
        decord_threads=args.decord_threads,
        overwrite=args.overwrite,
    )

    if args.num_workers <= 1:
        _init_worker(cfg)
        results = [_extract_subject(s) for s in tqdm(subjects)]
    else:
        with mp.get_context('spawn').Pool(
                args.num_workers, initializer=_init_worker,
                initargs=(cfg,), maxtasksperchild=1) as pool:
            results = list(tqdm(
                pool.imap_unordered(_extract_subject, subjects),
                total=len(subjects)))

    ok = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skip')
    no_video = sum(1 for r in results if r['status'] == 'no_video')
    err = len(results) - ok - skip - no_video
    failed = [r['subject'] for r in results
              if r['status'] not in {'ok', 'skip', 'no_video'}]
    for r in results:
        if r['status'] not in {'ok', 'skip', 'no_video'}:
            print(f'[err] {r["subject"]}: {r["status"]}', file=sys.stderr)
    print(f'done: ok={ok} skip={skip} no_video={no_video} err={err}')
    # Exit non-zero so a partial build in an sbatch job is not mistaken for
    # success — otherwise the missing subject HDF5s only surface much later as
    # silently-absent frames under --egobrain_use_frame_grid.
    if failed:
        print(f'[FATAL] {len(failed)} subject(s) failed: '
              f'{",".join(sorted(failed))}. The grid frame cache is '
              f'INCOMPLETE; fix the errors above and re-run before training '
              f'with --egobrain_use_frame_grid.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
