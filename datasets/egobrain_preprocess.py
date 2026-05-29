"""Offline preprocessor for the EgoBrain EEG dataset.

EgoBrain ships **one continuous EDF per subject** (32 channels, 256 Hz,
EmoTiv FLEX 2 Gel, 10-20 placement). The paper's recommended chain is:
band-pass 0.1-75 Hz → 50 Hz notch → resample to 200 Hz. This script
reproduces that chain, slices the cleaned recording into fixed-length
non-overlapping clips, and dumps one ``.npy`` per clip:

    <cache_dir>/<subject>/<clip_idx>.npy           # (C, fs_out*clip_s) float32 µV

Companion per-subject metadata file:

    <cache_dir>/<subject>/clips.json
        {
          "fs_out": 200,
          "clip_s": 4.0,
          "n_clips": 5400,
          "ch_names": ["AF3", ...],         # length C, EDF order
          "video": {
            "path": "data/EgoBrain/P0001/GoPro_HERO12_Ego/GX010104_anon.MP4",
            "video_offset_s": 0.0,           # video_t = eeg_t - offset
          } | null,
          "events": [{"start_s": ..., "end_s": ..., "label": "..."}, ...]
        }

A single subject's recording can be ~6 hours, so the cache lives on disk
and is loaded lazily by ``EgoBrainDataset``. The expensive MNE
filter/notch/resample is paid once, the training run reads .npy.

Run (one-time, idempotent):
    conda run -n cbramod python -m datasets.egobrain_preprocess \
        --data_dir data/EgoBrain --subjects P0001 --num_workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import sys
from glob import glob
from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm

# Import the same preprocessing primitive used by CineBrain so the filter/
# notch/resample chain is byte-identical across both datasets.
from datasets.cinebrain_dataset import preprocess_segment


_WORKER_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    _WORKER_CFG.update(cfg)


def _find_edf(subject_dir: str) -> Optional[str]:
    """Return the (non-marker) main EDF for the subject, or None."""
    eeg_dir = os.path.join(subject_dir, 'EmoTiv_FLEX_2_Gel', 'Phase_2')
    if not os.path.isdir(eeg_dir):
        return None
    cands = [f for f in os.listdir(eeg_dir)
             if f.endswith('.edf') and not f.endswith('.md.edf')]
    if not cands:
        return None
    cands.sort()
    return os.path.join(eeg_dir, cands[0])


def _find_interval_markers(subject_dir: str) -> Optional[str]:
    eeg_dir = os.path.join(subject_dir, 'EmoTiv_FLEX_2_Gel', 'Phase_2')
    if not os.path.isdir(eeg_dir):
        return None
    for f in sorted(os.listdir(eeg_dir)):
        if f.endswith('_intervalMarker.csv'):
            return os.path.join(eeg_dir, f)
    return None


_GOPRO_CHAPTER_RE = re.compile(r'^GX(\d{2})(\d{4})(?:_anon)?\.MP4$',
                               re.IGNORECASE)


def _parse_chapter(name: str) -> Optional[tuple[int, int]]:
    """Return (chapter, group_id) for a GoPro filename, or None.

    GoPro names a chaptered recording as ``GX{chap:02d}{grp:04d}.MP4``;
    the camera caps each chapter at ~11 min (~4 GB) before rolling over.
    Files that share a group_id belong to a single logical recording and
    the 2-digit chapter number gives their order in time.
    """
    m = _GOPRO_CHAPTER_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _find_video_chapters(subject_dir: str) -> list[dict]:
    """Return every GoPro chapter for the subject as a sorted dict list.

    Each entry: ``{"path": abs_path, "chapter": int, "group": int}``.
    Sorted by (group, chapter) so chapters from a single recording appear
    in time order. Anonymised files (``_anon`` suffix) are preferred when
    both anon and raw versions exist for the same (group, chapter).

    Returns ``[]`` when the GoPro dir doesn't exist or holds no MP4s;
    callers treat this as "no video for this subject", which is the
    expected case for P0025-P0040 in the EgoBrain release.
    """
    video_dir = os.path.join(subject_dir, 'GoPro_HERO12_Ego')
    if not os.path.isdir(video_dir):
        return []
    seen: dict[tuple[int, int], str] = {}
    for f in os.listdir(video_dir):
        parsed = _parse_chapter(f)
        if parsed is None:
            continue
        # Prefer the _anon variant when both forms exist for the same key.
        if parsed in seen and '_anon' not in f.lower():
            continue
        seen[parsed] = f
    chapters = []
    for (chap, grp), name in sorted(seen.items()):
        chapters.append({
            'path': os.path.join(video_dir, name),
            'chapter': chap,
            'group': grp,
        })
    return chapters


def _probe_chapter_durations(chapters: list[dict]) -> list[dict]:
    """Read fps + n_frames for each chapter (one decord open per file).

    Returns the same list with each entry augmented by
    ``n_frames``, ``fps``, ``duration_s``. Used by the preprocess so the
    dataset and the frame extractor can compute cumulative chapter
    offsets without reopening every file.
    """
    import decord
    decord.bridge.set_bridge('native')
    out = []
    for ch in chapters:
        vr = decord.VideoReader(ch['path'])
        fps = float(vr.get_avg_fps())
        n_frames = int(len(vr))
        out.append({**ch,
                    'n_frames': n_frames,
                    'fps': fps,
                    'duration_s': n_frames / fps if fps > 0 else 0.0})
    return out


def _parse_interval_markers(path: str) -> list[dict]:
    """Parse EmoTiv ``*_intervalMarker.csv`` → list of {start_s, end_s, label}.

    EgoBrain's marker CSV columns are ``latency, duration, type,
    marker_value, key, timestamp, marker_id`` (latency/duration both in
    seconds from EDF start). Older EmoTiv exports use different column
    names (start_time / end_time / etc.); we look up columns by name
    case-insensitively and treat ``duration`` as a delta. Unknown layouts
    → empty list, which the dataset handles gracefully.
    """
    events: list[dict] = []
    try:
        with open(path, newline='') as f:
            rdr = csv.DictReader(f)
            cols = {k.lower(): k for k in (rdr.fieldnames or [])}

            def _find(*opts):
                for o in opts:
                    if o in cols:
                        return cols[o]
                return None

            start_col = _find('latency', 'startdate', 'start time', 'start',
                              'start_time', 'starttime')
            dur_col = _find('duration')
            end_col = _find('enddate', 'end time', 'end', 'end_time', 'endtime')
            # EgoBrain uses ``type`` for the human-readable event name (e.g.
            # 'phase_Play_Toy'); fall back to marker_value / label for
            # older layouts.
            lbl_col = _find('type', 'marker_value', 'label', 'value',
                            'name', 'event')
            if start_col is None:
                return events
            for row in rdr:
                try:
                    s = float(row[start_col])
                except (TypeError, ValueError):
                    continue
                if dur_col is not None:
                    try:
                        e = s + float(row[dur_col])
                    except (TypeError, ValueError):
                        e = s
                elif end_col is not None:
                    try:
                        e = float(row[end_col])
                    except (TypeError, ValueError):
                        e = s
                else:
                    e = s
                lbl = row.get(lbl_col, '') if lbl_col else ''
                events.append({'start_s': float(s), 'end_s': float(e),
                               'label': str(lbl)})
    except OSError:
        pass
    return events


def _eeg_channel_whitelist() -> set[str]:
    """Upper-cased 10-20 standard montage names (used as an EEG whitelist).

    EgoBrain ships an EmoTiv FLEX 2 EDF with ~114 channels, most of which
    are derived signals (contact-quality CQ_*, electrode-quality EQ_*,
    BATTERY*, FwClockTime, marker fields, ...) that confound a blanket
    name-pattern blacklist. Whitelisting against the 10-20 standard is
    robust to vendor additions.
    """
    import mne
    mon = mne.channels.make_standard_montage('standard_1020')
    return {n.upper() for n in mon.ch_names}


def _is_eeg_channel(name: str, eeg_names: set[str]) -> bool:
    key = name.upper().strip()
    for prefix in ('EEG ', 'EEG.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.endswith('-REF'):
        key = key[:-4]
    return key in eeg_names


def _load_and_preprocess_edf(edf_path: str, fs_out: int,
                             l_freq: float, h_freq: float,
                             notch_hz: float
                             ) -> tuple[np.ndarray, list[str], float]:
    """Load EDF, run MNE filter/notch/resample, return (data, ch_names, fs_in).

    Data layout: (C, T_resampled) float32, µV-scaled and clipped to ±300 so
    the value range matches the rest of the project's preprocessing.
    """
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    # Keep only channels whose name matches the 10-20 standard montage.
    # The EmoTiv FLEX 2 EDF interleaves EEG with CQ_*/EQ_*/marker channels,
    # so a positive whitelist is more robust than blacklisting by pattern.
    eeg_names = _eeg_channel_whitelist()
    keep = [ch for ch in raw.ch_names if _is_eeg_channel(ch, eeg_names)]
    if not keep:
        raise RuntimeError(
            f"no 10-20 EEG channels found in {edf_path}; "
            f"first 10 ch_names: {raw.ch_names[:10]}")
    raw.pick(keep)
    # MNE's filter functions operate in-place on raw data; we replicate
    # the CineBrain chain by calling ``preprocess_segment`` so future
    # changes there propagate automatically.
    fs_in = float(raw.info['sfreq'])
    data = raw.get_data()  # (C, T) in volts
    data = preprocess_segment(
        data, fs_in=int(round(fs_in)), fs_out=fs_out,
        l_freq=l_freq, h_freq=h_freq, notch_hz=notch_hz)
    # MNE returns volts; scale to µV like the CineBrain cache does.
    data = data * 1e6 if np.abs(data).max() < 1.0 else data
    np.clip(data, -300, 300, out=data)
    return data.astype(np.float32), list(raw.ch_names), fs_in


def _segment_into_clips(data: np.ndarray, fs_out: int, clip_s: float
                        ) -> np.ndarray:
    """(C, T) → (n_clips, C, fs_out*clip_s); drop trailing partial clip."""
    clip_samples = int(round(clip_s * fs_out))
    C, T = data.shape
    n_clips = T // clip_samples
    data = data[:, :n_clips * clip_samples]
    return data.reshape(C, n_clips, clip_samples).transpose(1, 0, 2)


def _preprocess_subject(sub: str) -> dict:
    cfg = _WORKER_CFG
    sub_dir = os.path.join(cfg['data_dir'], sub)
    edf_path = _find_edf(sub_dir)
    if edf_path is None:
        return {'subject': sub, 'status': f'no EDF under {sub_dir}'}

    out_dir = os.path.join(cfg['cache_dir'], sub)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, 'clips.json')

    if (os.path.exists(meta_path) and not cfg['overwrite']):
        with open(meta_path) as f:
            meta = json.load(f)
        n_clips = meta.get('n_clips', 0)
        # Re-verify .npy files exist to detect partial caches.
        if all(os.path.exists(os.path.join(out_dir, f'{i}.npy'))
               for i in range(n_clips)):
            # Existing video meta is either None (no video for this subject),
            # the new {"chapters": [...]} schema, or the legacy {"path": ...}
            # single-chapter shape. Refresh only the legacy shape so we
            # cover the full session without re-running the expensive EEG
            # filter/notch/resample.
            video_meta = meta.get('video')
            needs_video_refresh = (
                isinstance(video_meta, dict) and
                'chapters' not in video_meta)
            if not needs_video_refresh:
                return {'subject': sub, 'status': 'skip',
                        'n_clips': n_clips}
            chapters = _find_video_chapters(sub_dir)
            if chapters:
                try:
                    chapters_full = _probe_chapter_durations(chapters)
                except Exception as e:                       # noqa: BLE001
                    print(f'[warn] {sub}: chapter probe failed ({e})',
                          file=sys.stderr)
                    chapters_full = chapters
                meta['video'] = {
                    'chapters': [
                        {'path': os.path.relpath(c['path'], cfg['data_dir']),
                         'chapter': c['chapter'],
                         'group': c['group'],
                         'n_frames': c.get('n_frames'),
                         'fps': c.get('fps'),
                         'duration_s': c.get('duration_s')}
                        for c in chapters_full
                    ],
                    'video_offset_s': video_meta.get(
                        'video_offset_s', cfg['video_offset_s']),
                    'total_duration_s': sum(
                        c.get('duration_s', 0.0) for c in chapters_full),
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                return {'subject': sub, 'status': 'video_refresh',
                        'n_clips': n_clips,
                        'n_chapters': len(chapters_full)}
            return {'subject': sub, 'status': 'skip',
                    'n_clips': n_clips}

    try:
        data, ch_names, fs_in = _load_and_preprocess_edf(
            edf_path, cfg['fs_out'], cfg['l_freq'], cfg['h_freq'],
            cfg['notch_hz'])
    except Exception as e:                                   # noqa: BLE001
        return {'subject': sub, 'status': f'edf-load: {e}'}

    clips = _segment_into_clips(data, cfg['fs_out'], cfg['clip_s'])
    n_clips = clips.shape[0]
    for i in range(n_clips):
        out = os.path.join(out_dir, f'{i}.npy')
        tmp = out + '.tmp'
        with open(tmp, 'wb') as fh:
            np.save(fh, clips[i], allow_pickle=False)
        os.replace(tmp, out)

    chapters = _find_video_chapters(sub_dir)
    markers_path = _find_interval_markers(sub_dir)
    events = _parse_interval_markers(markers_path) if markers_path else []

    if chapters:
        try:
            chapters_full = _probe_chapter_durations(chapters)
        except Exception as e:                               # noqa: BLE001
            # Probing failed (corrupt MP4, decord issue) — record paths
            # without durations so the dataset can fall back to live
            # decode without crashing.
            print(f'[warn] {sub}: chapter probe failed ({e}); '
                  f'storing chapter paths without durations',
                  file=sys.stderr)
            chapters_full = chapters
        video_meta = {
            'chapters': [
                {'path': os.path.relpath(c['path'], cfg['data_dir']),
                 'chapter': c['chapter'],
                 'group': c['group'],
                 'n_frames': c.get('n_frames'),
                 'fps': c.get('fps'),
                 'duration_s': c.get('duration_s')}
                for c in chapters_full
            ],
            'video_offset_s': cfg['video_offset_s'],
            'total_duration_s': sum(
                c.get('duration_s', 0.0) for c in chapters_full),
        }
    else:
        video_meta = None

    meta = {
        'fs_in': fs_in,
        'fs_out': cfg['fs_out'],
        'clip_s': cfg['clip_s'],
        'n_clips': int(n_clips),
        'n_channels': int(clips.shape[1]),
        'ch_names': ch_names,
        'edf_path': os.path.relpath(edf_path, cfg['data_dir']),
        'video': video_meta,
        'events': events,
        'preprocess': {
            'l_freq': cfg['l_freq'], 'h_freq': cfg['h_freq'],
            'notch_hz': cfg['notch_hz'],
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return {'subject': sub, 'status': 'ok', 'n_clips': int(n_clips)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--cache_dir', default=None,
                   help='defaults to <data_dir>/cache_eeg_<fs_out>hz')
    p.add_argument('--subjects', required=True,
                   help='comma-separated subject ids, or "all" '
                        'for every subdir of data_dir matching P\\d{4}')
    p.add_argument('--fs_out', type=int, default=200)
    p.add_argument('--clip_s', type=float, default=4.0,
                   help='clip length in seconds (must equal '
                        'cinebrain_n_windows * cinebrain_stride_s + '
                        '(cinebrain_window_s - cinebrain_stride_s))')
    p.add_argument('--l_freq', type=float, default=0.3,
                   help='lower bandpass cutoff (Hz). Default matches the '
                        'CineBrain / TUEG project conventions; the EgoBrain '
                        'paper uses 0.1 — both are reasonable.')
    p.add_argument('--h_freq', type=float, default=75.0)
    p.add_argument('--notch_hz', type=float, default=50.0,
                   help='powerline frequency. EgoBrain was recorded in Japan '
                        '(50 Hz); set 60 in North America.')
    p.add_argument('--video_offset_s', type=float, default=0.0,
                   help='video_t = eeg_t - offset; the paper reports <1 s '
                        'jitter between EEG and video so 0 is a sane default')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    cache_dir = (args.cache_dir
                 or os.path.join(args.data_dir, f'cache_eeg_{args.fs_out}hz'))

    if args.subjects.lower() == 'all':
        subjects = sorted(
            d for d in os.listdir(args.data_dir)
            if re.match(r'^P\d{4}$', d)
            and os.path.isdir(os.path.join(args.data_dir, d)))
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    print(f'preprocessing {len(subjects)} subject(s) → {cache_dir}')

    cfg = dict(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        fs_out=args.fs_out,
        clip_s=args.clip_s,
        l_freq=args.l_freq, h_freq=args.h_freq, notch_hz=args.notch_hz,
        video_offset_s=args.video_offset_s,
        overwrite=args.overwrite,
    )

    if args.num_workers <= 1:
        _init_worker(cfg)
        results = [_preprocess_subject(s) for s in tqdm(subjects)]
    else:
        with mp.get_context('spawn').Pool(
                args.num_workers, initializer=_init_worker,
                initargs=(cfg,), maxtasksperchild=4) as pool:
            results = list(tqdm(
                pool.imap_unordered(_preprocess_subject, subjects),
                total=len(subjects)))

    ok = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skip')
    refreshed = sum(1 for r in results if r['status'] == 'video_refresh')
    err = len(results) - ok - skip - refreshed
    for r in results:
        if r['status'] not in {'ok', 'skip', 'video_refresh'}:
            print(f'[err] {r["subject"]}: {r["status"]}', file=sys.stderr)
    print(f'done: ok={ok} skip={skip} video_refresh={refreshed} err={err}')


if __name__ == '__main__':
    main()
