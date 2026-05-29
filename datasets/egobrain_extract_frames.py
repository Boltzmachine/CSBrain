"""Offline extractor: pre-decode EgoBrain video frames into per-subject HDF5.

Live decoding of EgoBrain's 4-5 GB GoPro MP4s is ~265 ms/frame even with
the LRU reader cache, dominating training step time. This script does
the decode once per (subject × clip × window) and writes the resulting
uint8 frames into a single HDF5 file per subject — 40 files total,
trivial inode pressure on HPC, ~7 GB after gzip compression.

Layout written:
    <cache_dir>/<subject>.h5
        frames        (n_clips, n_windows, H, W, 3) uint8 — chunked per clip
        has_image     (n_clips, n_windows)         bool  — frame within video bounds
        attrs:
            vision_encoder, frame_size, window_s, stride_s,
            erp_latency_s, clip_s, n_windows, fps_video, video_path

At train time, ``EgoBrainDataset`` checks for the cache dir; if present
it reads from HDF5 instead of decoding. The cache invalidates whenever
``window_s``, ``stride_s``, ``erp_latency_s``, ``clip_s``, ``n_windows``,
or ``vision_encoder`` change — re-run this script with the new values.

Run:
    conda run -n cbramod python -m datasets.egobrain_extract_frames \\
        --data_dir data/EgoBrain --subjects all \\
        --vision_encoder facebook/dinov2-base \\
        --window_s 1.0 --stride_s 1.0 --erp_latency_s 0.5 \\
        --n_windows 2 --num_workers 4
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

from datasets.cinebrain_dataset import (
    _frame_size_for, _get_video_reader, _get_vision_processor,
)


def _processor_resize_crop_config(vision_encoder: str) -> tuple[int, int, int]:
    """Look up HF processor resize+crop config for ``vision_encoder``.

    Returns (shortest_edge, crop_size, resample). DINOv2 reports
    {shortest_edge: 256} and crop 224x224 with BICUBIC. V-JEPA 2 reports
    similar (256/256) with BILINEAR. Falls back to ``crop_size = crop_size``
    if the processor doesn't expose ``size`` separately.
    """
    proc = _get_vision_processor(vision_encoder)
    size = getattr(proc, 'size', None) or {}
    crop = getattr(proc, 'crop_size', None) or {}
    if isinstance(size, dict) and 'shortest_edge' in size:
        shortest_edge = int(size['shortest_edge'])
    elif isinstance(size, dict) and 'height' in size and 'width' in size:
        shortest_edge = int(min(size['height'], size['width']))
    else:
        shortest_edge = _frame_size_for(vision_encoder)
    if isinstance(crop, dict) and 'height' in crop and 'width' in crop:
        crop_size = int(min(crop['height'], crop['width']))
    else:
        crop_size = _frame_size_for(vision_encoder)
    resample = int(getattr(proc, 'resample', 3))      # default BICUBIC=3
    return shortest_edge, crop_size, resample


_WORKER_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    _WORKER_CFG.update(cfg)


def _resize_shortest_then_center_crop(
        img: np.ndarray, shortest_edge: int, crop_size: int,
        resample: int) -> np.ndarray:
    """Mirror HF AutoImageProcessor's resize+crop: scale so the shortest
    side equals ``shortest_edge``, then center-crop a ``crop_size`` square.

    Matches DINOv2's BitImageProcessor (shortest_edge=256, crop=224x224).
    Going through this path means the cached frame is byte-equivalent to
    HF's resize+crop output, so the dataset's fast path can skip HF and
    only run rescale+normalize.
    """
    from PIL import Image
    H, W = img.shape[:2]
    if H < W:
        new_h = shortest_edge
        new_w = int(round(W * shortest_edge / H))
    else:
        new_w = shortest_edge
        new_h = int(round(H * shortest_edge / W))
    pil = Image.fromarray(img).resize((new_w, new_h), resample)
    # Center crop.
    left = (new_w - crop_size) // 2
    top = (new_h - crop_size) // 2
    pil = pil.crop((left, top, left + crop_size, top + crop_size))
    return np.asarray(pil, dtype=np.uint8)


def _decode_frames_for_clip(vr, fps: float, t_videos: list,
                            n_frames_total: int, crop_size: int,
                            shortest_edge: int, resample: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Return (uint8 frames (W, crop_size, crop_size, 3), has_image (W,)).

    Out-of-range timestamps produce zero frames with ``has_image=False``.
    """
    W = len(t_videos)
    out = np.zeros((W, crop_size, crop_size, 3), dtype=np.uint8)
    ok = np.zeros((W,), dtype=bool)
    wanted_idx = []
    wanted_slot = []
    for w, t in enumerate(t_videos):
        if t < 0:
            continue
        idx = int(round(t * fps))
        if idx < 0 or idx >= n_frames_total:
            continue
        wanted_idx.append(idx)
        wanted_slot.append(w)
    if not wanted_idx:
        return out, ok
    # ``get_batch`` requires monotonic non-decreasing indices on most
    # decord builds; sort + unscatter to preserve caller order.
    order = sorted(range(len(wanted_idx)), key=lambda i: wanted_idx[i])
    sorted_idx = [wanted_idx[i] for i in order]
    frames = vr.get_batch(sorted_idx).asnumpy()        # (K, H, W, 3) uint8
    for k, src in enumerate(order):
        slot = wanted_slot[src]
        out[slot] = _resize_shortest_then_center_crop(
            frames[k], shortest_edge, crop_size, resample)
        ok[slot] = True
    return out, ok


def _extract_subject(sub: str) -> dict:
    cfg = _WORKER_CFG
    cache_root = cfg['cache_dir']
    out_path = os.path.join(cache_root, f'{sub}.h5')

    # Read the EEG-cache metadata so we know n_clips, clip_s, video path,
    # and offset for this subject. The EEG preprocess must have been run
    # first (the launch script's prereq step).
    eeg_meta_path = os.path.join(
        cfg['data_dir'], f'cache_eeg_{cfg["fs_out"]}hz', sub, 'clips.json')
    if not os.path.exists(eeg_meta_path):
        return {'subject': sub,
                'status': f'missing EEG cache {eeg_meta_path}; run egobrain_preprocess first'}
    with open(eeg_meta_path) as f:
        eeg_meta = json.load(f)
    n_clips = int(eeg_meta['n_clips'])
    clip_s = float(eeg_meta['clip_s'])
    video_info = eeg_meta.get('video')
    if video_info is None:
        # Expected for P0025-P0040: EgoBrain only ships video for the
        # first 24 subjects. Not an error; the dataset's __getitem__
        # leaves has_image=False for these so the alignment loss skips
        # those rows while masked recon + latent prediction still train.
        return {'subject': sub, 'status': 'no_video'}
    if 'chapters' in video_info:
        chapters = list(video_info['chapters'])
    elif 'path' in video_info:                              # legacy fallback
        chapters = [{'path': video_info['path']}]
    else:
        return {'subject': sub,
                'status': 'video meta has neither "chapters" nor "path"; '
                          'rerun egobrain_preprocess to refresh clips.json'}
    # Resolve relative paths + accumulate per-chapter start_s. A chapter
    # with no recorded duration is treated as covering [start_s, +inf),
    # matching the dataset's legacy-fallback behaviour.
    resolved_chapters: list[dict] = []
    cum = 0.0
    for ch in chapters:
        ch_path = os.path.join(cfg['data_dir'], ch['path'])
        if not os.path.exists(ch_path):
            return {'subject': sub, 'status': f'chapter missing: {ch_path}'}
        entry = {'path': ch_path, 'start_s': cum,
                 'duration_s': ch.get('duration_s')}
        resolved_chapters.append(entry)
        if entry['duration_s'] is None:
            cum = float('inf')
        else:
            cum += float(entry['duration_s'])
    video_offset_s = float(video_info.get('video_offset_s', 0.0))

    if os.path.exists(out_path) and not cfg['overwrite']:
        # Validate that the existing cache matches the requested config.
        with h5py.File(out_path, 'r') as h:
            attrs = dict(h.attrs)
        cur_cfg = (attrs.get('vision_encoder'), attrs.get('frame_size'),
                   attrs.get('window_s'), attrs.get('stride_s'),
                   attrs.get('erp_latency_s'), attrs.get('clip_s'),
                   attrs.get('n_windows'))
        want_cfg = (cfg['vision_encoder'], cfg['frame_size'],
                    cfg['window_s'], cfg['stride_s'],
                    cfg['erp_latency_s'], clip_s, cfg['n_windows'])
        if all(a == b or (a == np.bytes_(b) if isinstance(a, np.bytes_) else False)
               for a, b in zip(cur_cfg, want_cfg)):
            return {'subject': sub, 'status': 'skip',
                    'n_clips': n_clips}

    target_size = int(cfg['frame_size'])
    n_windows = int(cfg['n_windows'])
    window_s = float(cfg['window_s'])
    stride_s = float(cfg['stride_s'])
    erp_latency_s = float(cfg['erp_latency_s'])
    fs_out = int(cfg['fs_out'])
    window_samples = int(round(window_s * fs_out))
    stride_samples = int(round(stride_s * fs_out))

    shortest_edge, crop_size, resample = _processor_resize_crop_config(
        cfg['vision_encoder'])
    # The user can override the cached crop via --frame_size; if so,
    # the dataset's fast path also needs to know not to send these
    # through HF's resize (else HF re-resizes to its own crop size).
    if int(cfg['frame_size']) != crop_size:
        crop_size = int(cfg['frame_size'])
    target_size = crop_size

    def _resolve_chapter(t_video: float) -> Optional[tuple[dict, float]]:
        """Route ``t_video`` to its chapter (handles infinite-duration tail)."""
        if t_video < 0:
            return None
        for ch in resolved_chapters:
            dur = ch.get('duration_s')
            if dur is None or t_video < ch['start_s'] + dur:
                return ch, t_video - ch['start_s']
        return None

    # Write to a sibling .tmp file then atomic-rename so a crashed run
    # never leaves a half-populated .h5 that future runs would treat
    # as a valid skip.
    tmp_path = out_path + '.tmp'
    os.makedirs(cache_root, exist_ok=True)
    with h5py.File(tmp_path, 'w') as h:
        frames_ds = h.create_dataset(
            'frames',
            shape=(n_clips, n_windows, target_size, target_size, 3),
            dtype='uint8',
            chunks=(1, n_windows, target_size, target_size, 3),
            compression='gzip', compression_opts=4,
        )
        has_ds = h.create_dataset(
            'has_image', shape=(n_clips, n_windows), dtype='bool')
        h.attrs['subject'] = sub
        h.attrs['vision_encoder'] = cfg['vision_encoder']
        h.attrs['frame_size'] = target_size
        h.attrs['window_s'] = window_s
        h.attrs['stride_s'] = stride_s
        h.attrs['erp_latency_s'] = erp_latency_s
        h.attrs['clip_s'] = clip_s
        h.attrs['n_windows'] = n_windows
        h.attrs['fs_out'] = fs_out
        h.attrs['video_offset_s'] = video_offset_s
        h.attrs['n_chapters'] = len(resolved_chapters)
        h.attrs['total_video_duration_s'] = sum(
            (c.get('duration_s') or 0.0) for c in resolved_chapters)

        # Pre-bucket every (clip, window) into its chapter. This lets us
        # open each chapter's VideoReader at most once, decode every
        # frame the chapter needs, then close it before opening the next.
        # Critical for memory: decord's per-reader internal buffers can
        # grow into tens of GB under random access; bounding the working
        # set to ONE reader at a time keeps each worker well under any
        # reasonable cgroup limit.
        ch_buckets: dict[str, list[tuple[int, int, float]]] = {
            ch['path']: [] for ch in resolved_chapters
        }
        for c in range(n_clips):
            t0 = c * clip_s
            for i in range(n_windows):
                win_center = (
                    i * stride_samples + window_samples / 2) / fs_out
                t_eeg = t0 + win_center + erp_latency_s
                resolved = _resolve_chapter(t_eeg - video_offset_s)
                if resolved is None:
                    continue
                ch, t_in_ch = resolved
                ch_buckets[ch['path']].append((c, i, t_in_ch))

        # Decode chapter by chapter. We never hold more than one open
        # VideoReader at a time; explicit ``del`` + ``gc`` drops decord's
        # internal frame cache as soon as we move on.
        import gc
        import decord
        # In case the global LRU happens to still hold a reader from a
        # prior subject in this worker, clear it before we open ours.
        from datasets.cinebrain_dataset import _VIDEO_READER_CACHE
        _VIDEO_READER_CACHE.clear()
        gc.collect()

        h5_frames_buffer = np.zeros(
            (n_clips, n_windows, target_size, target_size, 3), dtype=np.uint8)
        h5_has_buffer = np.zeros((n_clips, n_windows), dtype=bool)
        for ch_path, wanted in ch_buckets.items():
            if not wanted:
                continue
            decord.bridge.set_bridge('native')
            vr = decord.VideoReader(ch_path)
            ch_fps = float(vr.get_avg_fps())
            ch_n_frames = int(len(vr))
            assert ch_fps > 0, f'bad fps for {ch_path}'

            # Process this chapter's frames in batches that share the
            # same clip — that keeps each get_batch call small enough
            # for decord to decode contiguously without ballooning.
            # Within each batch we sort indices and unscatter, matching
            # the live-decode helper.
            by_clip: dict[int, list[tuple[int, float]]] = {}
            for c, i, t in wanted:
                by_clip.setdefault(c, []).append((i, t))
            for c, ws in by_clip.items():
                slots = [w[0] for w in ws]
                t_videos = [w[1] for w in ws]
                ch_frames, ch_ok = _decode_frames_for_clip(
                    vr, ch_fps, t_videos, ch_n_frames,
                    crop_size=target_size,
                    shortest_edge=shortest_edge,
                    resample=resample)
                for k, slot in enumerate(slots):
                    h5_frames_buffer[c, slot] = ch_frames[k]
                    h5_has_buffer[c, slot] = ch_ok[k]
            # Drop the reader before opening the next chapter.
            del vr
            gc.collect()

        # Single bulk write — cheaper than n_clips individual chunk
        # writes because h5py can pipeline compression.
        frames_ds[...] = h5_frames_buffer
        has_ds[...] = h5_has_buffer

    os.replace(tmp_path, out_path)
    return {'subject': sub, 'status': 'ok', 'n_clips': n_clips}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--cache_dir', default=None,
                   help='defaults to <data_dir>/cache_frames_<encoder>_w<ws>s<ss>_e<erp>')
    p.add_argument('--subjects', required=True,
                   help='comma-separated subject ids, or "all"')
    p.add_argument('--vision_encoder', default='facebook/dinov2-base',
                   help='HF model id. Sets target frame size '
                        '(DINOv2 = 224, V-JEPA 2 = 256).')
    p.add_argument('--frame_size', type=int, default=None,
                   help='override frame size; default derived from vision_encoder')
    p.add_argument('--window_s', type=float, default=1.0)
    p.add_argument('--stride_s', type=float, default=1.0)
    p.add_argument('--erp_latency_s', type=float, default=0.5,
                   help='matches the training-time --egobrain_erp_latency_s')
    p.add_argument('--n_windows', type=int, default=2,
                   help='must be >= 1 + max_horizon used at training time')
    p.add_argument('--fs_out', type=int, default=200,
                   help='matches the EEG preprocess --fs_out')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    frame_size = (args.frame_size if args.frame_size is not None
                  else _frame_size_for(args.vision_encoder))

    if args.cache_dir is None:
        # Encode the cache-invalidating settings into the dir name so
        # different configs coexist on disk.
        enc_slug = args.vision_encoder.replace('/', '_')
        cache_dir = os.path.join(
            args.data_dir,
            f'cache_frames_{enc_slug}'
            f'_w{args.window_s}s{args.stride_s}'
            f'_e{args.erp_latency_s}_nw{args.n_windows}'
            f'_sz{frame_size}')
    else:
        cache_dir = args.cache_dir

    if args.subjects.lower() == 'all':
        subjects = sorted(
            d for d in os.listdir(args.data_dir)
            if re.match(r'^P\d{4}$', d)
            and os.path.isdir(os.path.join(args.data_dir, d)))
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    print(f'extracting frames for {len(subjects)} subject(s) → {cache_dir}')

    cfg = dict(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        vision_encoder=args.vision_encoder,
        frame_size=frame_size,
        window_s=args.window_s,
        stride_s=args.stride_s,
        erp_latency_s=args.erp_latency_s,
        n_windows=args.n_windows,
        fs_out=args.fs_out,
        overwrite=args.overwrite,
    )

    if args.num_workers <= 1:
        _init_worker(cfg)
        results = [_extract_subject(s) for s in tqdm(subjects)]
    else:
        # ``maxtasksperchild=1`` means each subject gets a fresh worker
        # process; when it exits the OS reclaims everything, so decord's
        # leak-prone internal frame buffers can't accumulate across
        # subjects. Combined with the chapter-sequential loop above,
        # peak memory per worker stays under ~3 GB.
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
    for r in results:
        if r['status'] not in {'ok', 'skip', 'no_video'}:
            print(f'[err] {r["subject"]}: {r["status"]}', file=sys.stderr)
    no_video_subs = sorted(r['subject'] for r in results
                           if r['status'] == 'no_video')
    if no_video_subs:
        print(f'[info] {len(no_video_subs)} subject(s) ship without video '
              f'(EgoBrain only releases MP4 for P0001-P0024); these will '
              f'still train EEG-only losses: {",".join(no_video_subs)}')
    print(f'done: ok={ok} skip={skip} no_video={no_video} err={err}')


if __name__ == '__main__':
    main()
