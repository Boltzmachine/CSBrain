"""Derive per-window HAND-MOVEMENT labels for EgoBrain from the egocentric video.

EgoBrain ships only coarse activity-phase markers (``phase_Play_Toy`` …); it
has *no* laterality annotation. This module derives, for every EEG window the
world-model uses, an explicit label of WHICH HAND is moving by running a SOTA
in-the-wild hand reconstructor (WiLoR) on the first-person GoPro frames that
span the window, tracking each hand, and thresholding its (ego-motion
compensated, scale-normalised) motion.

Two products per ``(subject, clip, window)``:

  * a DISCRETE label, integer-compatible with PhysioNet-MI (the finetune task)
    so a head trained here transfers cleanly:
        0 = left  hand moving   (PhysioNet "left fist")
        1 = right hand moving   (PhysioNet "right fist")
        2 = both  hands moving  (PhysioNet "both fists")
       -1 = neither / undetermined   (no hand movement detected, or no/edge
            frames). PhysioNet-MI's 4-class scheme has no "rest" index, so we
            use a sentinel rather than collide with class 3 (= "both feet",
            which has no hand analogue and is intentionally never emitted).

  * a CONTINUOUS "drasticness" metric: each hand's ego-compensated,
    scale-normalised joint speed over the window, in hand-lengths/second.
    Stored per hand (``left_intensity`` / ``right_intensity``) plus a combined
    scalar (``drasticness`` = max of the two). The discrete label is just a
    threshold on these, so labels can be re-derived offline (see
    ``recompute_labels``) without re-running WiLoR.

Why WiLoR: it is the current SOTA for monocular in-the-wild 3D hand
localisation+reconstruction (FreiHAND/HO3D), end-to-end (built-in detector +
handedness), and ships a turnkey inference package (``WiLoR-mini``). The
``HandEstimator`` interface is deliberately backend-agnostic so HaWoR
(egocentric *world-space* hand motion — does ego-motion compensation natively
via SLAM) can be dropped in later for the motion component.

Output layout (a NEW cache dir; never touches raw data, the EEG cache, or the
frame cache — and refuses to overwrite an existing per-subject file unless
``--overwrite``):

    <data_dir>/cache_hand_labels_<backend>_w<ws>s<ss>_e<erp>_nw<nw>_k<K>/
        label_mapping.json          provenance + config + thresholds
        <subject>.h5
            label            (n_clips, n_windows) int8   {-1,0,1,2}
            left_intensity   (n_clips, n_windows) float32 hand-lengths/s (NaN if no frame)
            right_intensity  (n_clips, n_windows) float32
            drasticness      (n_clips, n_windows) float32 max(left,right)
            left_active_frac (n_clips, n_windows) float32 frac frames left hand seen+moving
            right_active_frac(n_clips, n_windows) float32
            left_det_frac    (n_clips, n_windows) float32 frac frames left hand detected
            right_det_frac   (n_clips, n_windows) float32
            has_video        (n_clips, n_windows) bool    >=2 in-bounds frames decoded
            attrs: full config + thresholds + backend + label-mapping json

The window→video timing exactly mirrors ``datasets/egobrain_extract_frames.py``
(``t_eeg = c*clip_s + (i*stride + window/2)/fs_out + erp_latency`` then
``t_video = t_eeg - video_offset``), so the keys align 1:1 with the DINOv2
frame cache and with ``EgoBrainDataset``'s ``(sub, clip)`` items.

Runs in the dedicated `wilor` env (see sh/install_wilor.sh), NOT cbramod.
Invoke as a DIRECT script so datasets/__init__.py (training-only deps) isn't
imported — this module needs none of it:
    python datasets/egobrain_hand_labels.py \\
        --data_dir data/EgoBrain --subjects all \\
        --window_s 1.0 --stride_s 1.0 --erp_latency_s 0.5 \\
        --n_windows 2 --clip_s 4.0 --fs_out 200 --frames_per_window 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Label constants. 0/1/2 are byte-for-byte the PhysioNet-MI hand classes
# (datasets/plot_physio_bands.py: {0: left fist, 1: right fist, 2: both fists,
# 3: both feet}); see finetune_main.py --flip_label_map default "1,0,2,3".
# ---------------------------------------------------------------------------
LABEL_LEFT = 0
LABEL_RIGHT = 1
LABEL_BOTH = 2
LABEL_NONE = -1          # neither / undetermined (PhysioNet has no rest index)
LABEL_BOTH_FEET = 3      # PhysioNet "both feet" — no hand analogue, never emitted

LABEL_NAMES = {
    LABEL_LEFT: 'left_hand',
    LABEL_RIGHT: 'right_hand',
    LABEL_BOTH: 'both_hands',
    LABEL_NONE: 'none',
}


# ===========================================================================
# Pure timing — no video/model deps, unit-tested. Mirrors
# datasets/egobrain_extract_frames.py so labels key-align with the frame cache.
# ===========================================================================
def window_frame_times(c: int, i: int, *, clip_s: float, window_s: float,
                       stride_s: float, fs_out: int, erp_latency_s: float,
                       frames_per_window: int) -> list[float]:
    """EEG-clock timestamps (seconds from EDF start) of the ``frames_per_window``
    frames sampled across window ``i`` of clip ``c``.

    The window spans EEG samples ``[i*stride, i*stride + window]`` inside the
    clip; in seconds-from-start that is ``[t0 + i*stride_s, t0 + i*stride_s +
    window_s]`` with ``t0 = c*clip_s``. We sample ``K`` evenly-spaced points
    across that span and add ``erp_latency_s`` — identical to the frame
    extractor's single-frame formula at the window CENTRE
    (``win_center = (i*stride_samples + window_samples/2)/fs_out``), so the
    middle sample here coincides with the cached frame.
    """
    window_samples = int(round(window_s * fs_out))
    stride_samples = int(round(stride_s * fs_out))
    t0 = c * clip_s
    win_start = (i * stride_samples) / fs_out
    win_len = window_samples / fs_out
    K = max(1, int(frames_per_window))
    if K == 1:
        fracs = [0.5]
    else:
        fracs = [k / (K - 1) for k in range(K)]
    return [t0 + win_start + f * win_len + erp_latency_s for f in fracs]


def resolve_chapter(chapters: list[dict], video_offset_s: float, eeg_t: float
                    ) -> Optional[tuple[dict, float]]:
    """Route an EEG timestamp to ``(chapter, t_in_chapter)`` or ``None``.

    Standalone twin of ``EgoBrainDataset._resolve_chapter`` /
    ``egobrain_extract_frames._resolve_chapter``: subtract the EEG→video
    offset, then walk the time-sorted chapter list (a chapter with unknown
    duration covers ``[start_s, +inf)``).
    """
    t_video = eeg_t - video_offset_s
    if t_video < 0:
        return None
    for ch in chapters:
        dur = ch.get('duration_s')
        if dur is None or t_video < ch['start_s'] + dur:
            return ch, t_video - ch['start_s']
    return None


def build_chapters(video_info: dict, data_dir: str) -> list[dict]:
    """Resolve clips.json ``video`` → ordered chapters with abs paths +
    cumulative ``start_s`` (mirrors the dataset/extractor)."""
    if 'chapters' in video_info:
        raw = list(video_info['chapters'])
    elif 'path' in video_info:                                  # legacy
        raw = [{'path': video_info['path']}]
    else:
        return []
    out: list[dict] = []
    cum = 0.0
    for ch in raw:
        entry = {'path': os.path.join(data_dir, ch['path']), 'start_s': cum,
                 'duration_s': ch.get('duration_s')}
        out.append(entry)
        cum = float('inf') if entry['duration_s'] is None \
            else cum + float(entry['duration_s'])
    return out


# ===========================================================================
# Per-frame hand parse: a frame's WiLoR output → at most one left + one right
# hand, each summarised as (keypoints[21,2] px, wrist px, scale px, score).
# ===========================================================================
class Hand:
    __slots__ = ('kp', 'wrist', 'scale', 'bbox', 'score')

    def __init__(self, kp: np.ndarray, score: float,
                 bbox: Optional[np.ndarray] = None):
        # kp: (21, 2) image-pixel joint coordinates (OpenPose/MANO order;
        # joint 0 = wrist, joint 9 = middle-finger MCP).
        self.kp = kp.astype(np.float64)
        self.wrist = self.kp[0].copy()
        palm = float(np.linalg.norm(self.kp[0] - self.kp[9]))
        # Fall back to keypoint bbox diagonal if the palm is degenerate.
        if not np.isfinite(palm) or palm < 1e-3:
            mn, mx = self.kp.min(0), self.kp.max(0)
            palm = float(np.linalg.norm(mx - mn)) or 1.0
        self.scale = palm
        if bbox is None:
            mn, mx = self.kp.min(0), self.kp.max(0)
            bbox = np.array([mn[0], mn[1], mx[0], mx[1]], dtype=np.float64)
        self.bbox = bbox
        self.score = float(score)


def parse_wilor_frame(outputs, handedness_source: str = 'hybrid',
                      img_w: Optional[float] = None
                      ) -> tuple[Optional[Hand], Optional[Hand]]:
    """Reduce one frame's WiLoR ``predict`` output to (left_hand, right_hand).

    ``outputs`` is the list of per-hand dicts WiLoR returns. Each has
    ``is_right`` (1.0 right / 0.0 left) and
    ``wilor_preds['pred_keypoints_2d']`` of shape (1, 21, 2) in image pixels.
    If several hands share a handedness we keep the highest-scoring one; in
    ``hybrid`` mode, when exactly two same-handed hands are detected they are
    split by image-x (left half → left hand), which guards against the
    handedness flips common in egocentric (own-hand) views. ``img_w`` (frame
    width) is only needed to place a lone hand in pure ``position`` mode; if
    omitted there, it falls back to the model's flag.
    """
    cands = []
    for out in outputs:
        wp = out.get('wilor_preds', out)
        kp2d = np.asarray(wp['pred_keypoints_2d'])
        kp = kp2d[0] if kp2d.ndim == 3 else kp2d            # (21, 2)
        score = float(out.get('score', out.get('hand_conf', 1.0)))
        bbox = out.get('hand_bbox')
        bbox = np.asarray(bbox, dtype=np.float64) if bbox is not None else None
        is_right = bool(round(float(out.get('is_right', 0.0))))
        cands.append({'hand': Hand(kp, score, bbox), 'is_right': is_right})
    if not cands:
        return None, None

    def cx(c):
        return float(c['hand'].wrist[0])

    if handedness_source == 'position':
        # Ignore the model's flag and assign by image-x.
        cands.sort(key=cx)
        if len(cands) == 1:                                 # lone hand:
            h = Hand_of(cands[0])
            if img_w is not None:                           # which half of frame?
                return (h, None) if cx(cands[0]) < img_w / 2 else (None, h)
            # No width given → best-effort fall back to the model's flag.
            return (None, h) if cands[0]['is_right'] else (h, None)
        return Hand_of(cands[0]), Hand_of(cands[-1])

    lefts = [c for c in cands if not c['is_right']]
    rights = [c for c in cands if c['is_right']]
    # Hybrid repair: model says both hands are the same side, but two hands
    # are present → trust geometry and split by x.
    if handedness_source == 'hybrid' and len(cands) == 2 and \
            (len(lefts) == 2 or len(rights) == 2):
        cands.sort(key=cx)
        return Hand_of(cands[0]), Hand_of(cands[1])
    left = Hand_of(max(lefts, key=lambda c: c['hand'].score)) if lefts else None
    right = Hand_of(max(rights, key=lambda c: c['hand'].score)) if rights else None
    return left, right


def Hand_of(c) -> Hand:
    return c['hand']


# ===========================================================================
# Motion aggregation over a window's frame track → metrics + discrete label.
# Pure (numpy only) → unit-tested with synthetic tracks.
# ===========================================================================
def _hand_speed(track: list[Optional[Hand]], ego: list[Optional[np.ndarray]],
                dts: list[Optional[float]]) -> tuple[float, float, int]:
    """Mean & peak ego-compensated, scale-normalised joint speed of one hand.

    ``track[k]`` is the hand in frame k (or None if undetected); ``ego[k]``
    is the global background shift (px) from frame k-1→k (or None to skip
    compensation); ``dts[k]`` is the REAL elapsed time (s) of pair k-1→k, or
    ``None``/non-positive to skip the pair (a frame that wasn't decoded, or
    two slots that rounded to the SAME video frame — a duplicate frame carries
    no motion and must not inject a spurious zero-speed pair). Speed for a
    valid pair = mean over the 21 joints of ‖Δjoint − ego_shift‖, divided by
    the median hand scale and the pair's real dt → hand-lengths/second. A
    uniform ego subtraction cancels on the articulation component, so this
    captures translation AND finger motion. Returns
    ``(mean_speed, peak_speed, n_pairs)``; ``(0,0,0)`` if no valid pair.
    """
    scales = [h.scale for h in track if h is not None]
    if not scales:
        return 0.0, 0.0, 0
    scale = float(np.median(scales)) or 1.0
    speeds = []
    for k in range(1, len(track)):
        a, b = track[k - 1], track[k]
        if a is None or b is None:
            continue
        dt = dts[k] if (dts is not None and k < len(dts)) else None
        if dt is None or not (dt > 0):          # missing or duplicate frame
            continue
        shift = ego[k] if (k < len(ego) and ego[k] is not None) \
            else np.zeros(2)
        disp = (b.kp - a.kp) - shift[None, :]               # (21, 2)
        speeds.append(float(np.mean(np.linalg.norm(disp, axis=1))) / scale / dt)
    if not speeds:
        return 0.0, 0.0, 0
    return float(np.mean(speeds)), float(np.max(speeds)), len(speeds)


def aggregate_window(left_track: list[Optional[Hand]],
                     right_track: list[Optional[Hand]],
                     ego: list[Optional[np.ndarray]], *,
                     dts: list[Optional[float]],
                     move_thresh: float, min_det_frac: float,
                     min_pairs: int = 1) -> dict:
    """Per-window metrics + PhysioNet-aligned discrete label.

    ``dts`` is the per-pair real elapsed time (see ``_hand_speed``). A hand is
    "active" if it is detected in >= ``min_det_frac`` of frames AND its mean
    ego-compensated speed exceeds ``move_thresh`` (hand-lengths/s).
    label: both active → 2, left only → 0, right only → 1, neither → -1.
    """
    n = max(len(left_track), len(right_track))
    l_det = sum(h is not None for h in left_track)
    r_det = sum(h is not None for h in right_track)
    l_mean, l_peak, l_pairs = _hand_speed(left_track, ego, dts)
    r_mean, r_peak, r_pairs = _hand_speed(right_track, ego, dts)
    l_det_frac = l_det / n if n else 0.0
    r_det_frac = r_det / n if n else 0.0

    l_active = (l_det_frac >= min_det_frac and l_pairs >= min_pairs
                and l_mean >= move_thresh)
    r_active = (r_det_frac >= min_det_frac and r_pairs >= min_pairs
                and r_mean >= move_thresh)
    if l_active and r_active:
        label = LABEL_BOTH
    elif l_active:
        label = LABEL_LEFT
    elif r_active:
        label = LABEL_RIGHT
    else:
        label = LABEL_NONE

    # Intensity is NaN when the speed was UNMEASURABLE (no valid consecutive
    # pair — never seen, or only non-adjacent/duplicate frames), distinct from
    # "seen and still" = 0.0, so downstream can mask the unmeasured.
    l_int = l_mean if l_pairs >= 1 else np.nan
    r_int = r_mean if r_pairs >= 1 else np.nan
    drastic = np.nanmax([l_int if np.isfinite(l_int) else 0.0,
                         r_int if np.isfinite(r_int) else 0.0])
    return {
        'label': label,
        'left_intensity': l_int, 'right_intensity': r_int,
        'left_peak': l_peak, 'right_peak': r_peak,
        'left_active_frac': (l_mean >= move_thresh) * l_det_frac,
        'right_active_frac': (r_mean >= move_thresh) * r_det_frac,
        'left_det_frac': l_det_frac, 'right_det_frac': r_det_frac,
        'drasticness': float(drastic),
    }


def recompute_labels(left_intensity: np.ndarray, right_intensity: np.ndarray,
                     left_det_frac: np.ndarray, right_det_frac: np.ndarray, *,
                     move_thresh: float, min_det_frac: float) -> np.ndarray:
    """Re-derive discrete labels from stored continuous metrics at NEW
    thresholds — no WiLoR rerun. Operates elementwise on the cached arrays."""
    l_active = (left_det_frac >= min_det_frac) & \
        (np.nan_to_num(left_intensity, nan=-1.0) >= move_thresh)
    r_active = (right_det_frac >= min_det_frac) & \
        (np.nan_to_num(right_intensity, nan=-1.0) >= move_thresh)
    out = np.full(left_intensity.shape, LABEL_NONE, dtype=np.int8)
    out[l_active & r_active] = LABEL_BOTH
    out[l_active & ~r_active] = LABEL_LEFT
    out[~l_active & r_active] = LABEL_RIGHT
    return out


# ===========================================================================
# Ego-motion (cv2, lazy). Head-mounted GoPro → global flow mimics hand motion;
# estimate the dominant background shift on NON-hand pixels and subtract it.
# ===========================================================================
def estimate_ego_shift(prev_gray, cur_gray, exclude_boxes: list[np.ndarray]):
    """Median sparse-LK background translation (px) prev→cur, masking hands.

    Returns a length-2 np.array, or None if it can't be estimated (caller
    then skips compensation for that pair). Translation-only is intentional:
    seated desk tasks are dominated by head pan/tilt, and a robust median
    shift is far less likely to absorb genuine hand motion than a full
    affine/homography would.
    """
    import cv2
    H, W = cur_gray.shape[:2]
    mask = np.full((H, W), 255, np.uint8)
    pad = 0.15
    for b in exclude_boxes:
        if b is None:
            continue
        x0, y0, x1, y1 = b
        w, h = (x1 - x0), (y1 - y0)
        x0 = int(max(0, x0 - pad * w)); y0 = int(max(0, y0 - pad * h))
        x1 = int(min(W, x1 + pad * w)); y1 = int(min(H, y1 + pad * h))
        mask[y0:y1, x0:x1] = 0
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                  minDistance=8, mask=mask)
    if pts is None or len(pts) < 8:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
    if nxt is None:
        return None
    st = st.reshape(-1).astype(bool)
    if st.sum() < 8:
        return None
    flow = (nxt.reshape(-1, 2) - pts.reshape(-1, 2))[st]
    return np.median(flow, axis=0)


# ===========================================================================
# Hand estimator backend (WiLoR via WiLoR-mini), lazily imported.
# ===========================================================================
class HandEstimator:
    """Thin backend wrapper. ``predict_frame(rgb)`` → list of per-hand dicts
    (the shape ``parse_wilor_frame`` consumes)."""

    def __init__(self, backend: str = 'wilor', device: str = 'cuda',
                 dtype: str = 'float16'):
        self.backend = backend
        if backend != 'wilor':
            raise ValueError(f'unknown hand backend {backend!r} '
                             f'(only "wilor" is wired; HaWoR is a planned drop-in)')
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline \
            import WiLorHandPose3dEstimationPipeline
        dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        td = {'float16': torch.float16, 'float32': torch.float32}[dtype]
        if dev.type == 'cpu':
            td = torch.float32                              # fp16 is GPU-only
        self.pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=td)

    def predict_frame(self, rgb: np.ndarray) -> list:
        return self.pipe.predict(rgb)


# ===========================================================================
# Subject processing.
# ===========================================================================
def _decode_clip_windows(vr, ch_fps: float, ch_n_frames: int,
                         windows_times: dict[int, list[float]],
                         max_frame_width: int):
    """Decode the frames for every (window → [t_video,…]) of one clip from an
    open VideoReader. Returns ``(frames, fidx)`` where each is
    ``{win_idx: [.. per slot]}``: ``frames`` holds the decoded RGB array or
    ``None`` (out of bounds), and ``fidx`` holds the integer video-frame index
    or ``None`` — the caller uses ``fidx`` to compute real per-pair dt and to
    drop duplicate-frame pairs."""
    import decord
    decord.bridge.set_bridge('native')
    # Collect, dedupe and sort all wanted frame indices for a contiguous read.
    want: dict[int, list[tuple[int, int]]] = {}   # frame_idx -> [(win, slot)]
    per_win_idx: dict[int, list[Optional[int]]] = {}
    for win, ts in windows_times.items():
        per_win_idx[win] = []
        for slot, t in enumerate(ts):
            if t < 0:
                per_win_idx[win].append(None); continue
            fi = int(round(t * ch_fps))
            if fi < 0 or fi >= ch_n_frames:
                per_win_idx[win].append(None); continue
            per_win_idx[win].append(fi)
            want.setdefault(fi, []).append((win, slot))
    if not want:
        empty = {win: [None] * len(ts) for win, ts in windows_times.items()}
        return empty, {win: list(idxs) for win, idxs in per_win_idx.items()}
    uniq = sorted(want)
    frames = vr.get_batch(uniq).asnumpy()                 # (U, H, W, 3) RGB
    if max_frame_width and frames.shape[2] > max_frame_width:
        import cv2
        scale = max_frame_width / frames.shape[2]
        nh = int(round(frames.shape[1] * scale))
        frames = np.stack([cv2.resize(f, (max_frame_width, nh)) for f in frames])
    fi_to_row = {fi: r for r, fi in enumerate(uniq)}
    out = {win: [None] * len(ts) for win, ts in windows_times.items()}
    for win, idxs in per_win_idx.items():
        for slot, fi in enumerate(idxs):
            if fi is not None:
                out[win][slot] = frames[fi_to_row[fi]]
    return out, per_win_idx


def process_subject(sub: str, cfg: dict, estimator: Optional[HandEstimator]
                    ) -> dict:
    data_dir = cfg['data_dir']
    eeg_meta_path = os.path.join(
        data_dir, f'cache_eeg_{cfg["fs_out"]}hz', sub, 'clips.json')
    if not os.path.exists(eeg_meta_path):
        return {'subject': sub, 'status': f'missing EEG cache {eeg_meta_path}'}
    with open(eeg_meta_path) as f:
        meta = json.load(f)
    if abs(float(meta['clip_s']) - cfg['clip_s']) > 1e-6:
        return {'subject': sub,
                'status': f"clip_s mismatch cache={meta['clip_s']} cfg={cfg['clip_s']}"}
    if int(meta['fs_out']) != cfg['fs_out']:
        return {'subject': sub, 'status': f"fs_out mismatch {meta['fs_out']}"}
    n_clips = int(meta['n_clips'])
    video_info = meta.get('video')
    if video_info is None:
        return {'subject': sub, 'status': 'no_video'}     # P0025-P0040
    chapters = build_chapters(video_info, data_dir)
    if not chapters:
        return {'subject': sub, 'status': 'no usable video chapters'}
    for ch in chapters:
        if not os.path.exists(ch['path']):
            return {'subject': sub, 'status': f"chapter missing {ch['path']}"}
    video_offset_s = float(video_info.get('video_offset_s', 0.0))

    out_path = os.path.join(cfg['out_dir'], f'{sub}.h5')
    if os.path.exists(out_path) and not cfg['overwrite']:
        # Only skip if the existing sidecar was built with THIS exact config;
        # otherwise it would silently return stale, mis-aligned labels. Mirrors
        # egobrain_extract_frames' attr-validation skip.
        mism = _config_mismatch(out_path, cfg)
        if mism is None:
            return {'subject': sub, 'status': 'skip', 'n_clips': n_clips}
        return {'subject': sub,
                'status': f'EXISTS with different config ({mism}); pass '
                          f'--overwrite to rebuild or use a fresh --out_dir'}

    n_windows = cfg['n_windows']
    K = cfg['frames_per_window']

    shp = (n_clips, n_windows)
    label = np.full(shp, LABEL_NONE, dtype=np.int8)
    li = np.full(shp, np.nan, np.float32); ri = np.full(shp, np.nan, np.float32)
    laf = np.zeros(shp, np.float32); raf = np.zeros(shp, np.float32)
    ldf = np.zeros(shp, np.float32); rdf = np.zeros(shp, np.float32)
    drastic = np.full(shp, np.nan, np.float32)
    has_video = np.zeros(shp, bool)

    # Bucket every (clip,window) into its chapter so each VideoReader opens
    # once (matches egobrain_extract_frames' memory-bounded chapter loop).
    ch_buckets: dict[str, dict[int, dict[int, list[float]]]] = {
        ch['path']: {} for ch in chapters}
    for c in range(n_clips):
        for i in range(n_windows):
            ts = window_frame_times(
                c, i, clip_s=cfg['clip_s'], window_s=cfg['window_s'],
                stride_s=cfg['stride_s'], fs_out=cfg['fs_out'],
                erp_latency_s=cfg['erp_latency_s'], frames_per_window=K)
            # Route by the window-CENTRE time, computed with the extractor's
            # exact formula (NOT ts[middle], which only equals the centre for
            # odd K). A window never straddles two chapters in practice; the
            # centre picks the owning chapter.
            win_samp = int(round(cfg['window_s'] * cfg['fs_out']))
            stride_samp = int(round(cfg['stride_s'] * cfg['fs_out']))
            center_t = (c * cfg['clip_s']
                        + (i * stride_samp + win_samp / 2) / cfg['fs_out']
                        + cfg['erp_latency_s'])
            res = resolve_chapter(chapters, video_offset_s, center_t)
            if res is None:
                continue
            ch, _ = res
            t_in = [resolve_chapter(chapters, video_offset_s, t) for t in ts]
            # Map each sample to in-chapter time of the SAME chapter (skip
            # samples that fell outside it → None handled as OOB downstream).
            t_local = [(r[1] if (r is not None and r[0]['path'] == ch['path'])
                        else -1.0) for r in t_in]
            ch_buckets[ch['path']].setdefault(c, {})[i] = t_local

    import decord, gc
    from tqdm import tqdm
    for ch in chapters:
        clips = ch_buckets[ch['path']]
        if not clips:
            continue
        vr = decord.VideoReader(ch['path'])
        ch_fps = float(vr.get_avg_fps()); ch_n = int(len(vr))
        assert ch_fps > 0, f"bad fps {ch['path']}"
        for c, win_times in tqdm(clips.items(), desc=f'{sub}:{os.path.basename(ch["path"])}',
                                 leave=False):
            decoded, fidx = _decode_clip_windows(vr, ch_fps, ch_n, win_times,
                                                 cfg['max_frame_width'])
            for i, frames in decoded.items():
                _fill_window(frames, fidx[i], ch_fps, estimator, cfg, label,
                             li, ri, laf, raf, ldf, rdf, drastic, has_video,
                             c, i)
        del vr; gc.collect()

    _write_subject(out_path, cfg, dict(
        label=label, left_intensity=li, right_intensity=ri,
        left_active_frac=laf, right_active_frac=raf,
        left_det_frac=ldf, right_det_frac=rdf, drasticness=drastic,
        has_video=has_video), sub, n_clips)
    return {'subject': sub, 'status': 'ok', 'n_clips': n_clips}


def _fill_window(frames, fidx, ch_fps, estimator, cfg, label, li, ri, laf,
                 raf, ldf, rdf, drastic, has_video, c, i):
    """Run the model on one window's frames and write its metrics in place.

    ``fidx[k]`` is the decoded video-frame index of slot k (or None); it gives
    the REAL per-pair dt and lets us drop duplicate-frame pairs.
    """
    valid = [f for f in frames if f is not None]
    if len(valid) < 2 or estimator is None:
        return
    has_video[c, i] = True
    left_track: list[Optional[Hand]] = []
    right_track: list[Optional[Hand]] = []
    grays = []
    import cv2
    for f in frames:
        if f is None:
            left_track.append(None); right_track.append(None)
            grays.append(None); continue
        outs = estimator.predict_frame(f)
        l, r = parse_wilor_frame(outs, cfg['handedness_source'],
                                 img_w=f.shape[1])
        left_track.append(l); right_track.append(r)
        grays.append(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY))
    # Ego shift per consecutive pair (None where a frame is missing or it
    # can't be estimated / compensation disabled).
    ego: list[Optional[np.ndarray]] = [None]
    for k in range(1, len(frames)):
        if (not cfg['ego_compensate'] or grays[k - 1] is None
                or grays[k] is None):
            ego.append(None); continue
        boxes = [h.bbox for h in (left_track[k - 1], left_track[k],
                                  right_track[k - 1], right_track[k])
                 if h is not None]
        ego.append(estimate_ego_shift(grays[k - 1], grays[k], boxes))
    # Real per-pair dt from decoded frame indices; None (skip) when a slot is
    # missing or two slots landed on the SAME video frame (duplicate).
    dts: list[Optional[float]] = [None]
    for k in range(1, len(frames)):
        if (fidx[k] is None or fidx[k - 1] is None
                or fidx[k] == fidx[k - 1] or ch_fps <= 0):
            dts.append(None)
        else:
            dts.append((fidx[k] - fidx[k - 1]) / ch_fps)
    m = aggregate_window(left_track, right_track, ego, dts=dts,
                         move_thresh=cfg['move_thresh'],
                         min_det_frac=cfg['min_det_frac'])
    label[c, i] = m['label']
    li[c, i] = m['left_intensity']; ri[c, i] = m['right_intensity']
    laf[c, i] = m['left_active_frac']; raf[c, i] = m['right_active_frac']
    ldf[c, i] = m['left_det_frac']; rdf[c, i] = m['right_det_frac']
    drastic[c, i] = m['drasticness']


def _write_subject(out_path, cfg, arrays, sub, n_clips):
    """Atomic, no-clobber HDF5 write (tmp + os.replace)."""
    import h5py
    os.makedirs(cfg['out_dir'], exist_ok=True)
    tmp = out_path + f'.{os.getpid()}.tmp'      # PID-unique: no concurrent-write clash
    with h5py.File(tmp, 'w') as h:
        for k, v in arrays.items():
            h.create_dataset(k, data=v, compression='gzip', compression_opts=4)
        h.attrs['subject'] = sub
        h.attrs['n_clips'] = n_clips
        for k in ('backend', 'window_s', 'stride_s', 'erp_latency_s', 'clip_s',
                  'n_windows', 'fs_out', 'frames_per_window', 'move_thresh',
                  'min_det_frac', 'handedness_source', 'ego_compensate',
                  'max_frame_width'):
            h.attrs[k] = cfg[k]
        h.attrs['label_mapping'] = json.dumps(LABEL_NAMES)
        h.attrs['code_version'] = 1
    os.replace(tmp, out_path)


def load_hand_labels(out_dir: str, sub: str) -> dict:
    """Read a subject's sidecar back as a dict of numpy arrays + attrs."""
    import h5py
    with h5py.File(os.path.join(out_dir, f'{sub}.h5'), 'r') as h:
        d = {k: h[k][...] for k in h.keys()}
        d['attrs'] = dict(h.attrs)
    return d


# All config keys that determine the stored arrays — used to refuse silently
# reusing a sidecar built with a different config. move_thresh/min_det_frac are
# included (they set the DISCRETE label); to re-threshold without a full WiLoR
# rerun use recompute_labels on the stored continuous arrays instead.
_CONFIG_KEYS = ('backend', 'window_s', 'stride_s', 'erp_latency_s', 'clip_s',
                'n_windows', 'fs_out', 'frames_per_window', 'move_thresh',
                'min_det_frac', 'handedness_source', 'ego_compensate',
                'max_frame_width')


def _config_mismatch(out_path: str, cfg: dict) -> Optional[str]:
    """First config key whose stored attr differs from ``cfg`` (so a stale
    sidecar is never silently reused), or ``None`` if it matches exactly."""
    import h5py
    try:
        with h5py.File(out_path, 'r') as h:
            attrs = dict(h.attrs)
    except OSError as e:
        return f'unreadable ({e})'
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
def default_out_dir(data_dir, backend, window_s, stride_s, erp_latency_s,
                    n_windows, frames_per_window, clip_s, fs_out):
    # clip_s & fs_out are in the name because both feed window_frame_times and
    # so change which video frame each label maps to.
    return os.path.join(
        data_dir,
        f'cache_hand_labels_{backend}_w{window_s}s{stride_s}'
        f'_e{erp_latency_s}_nw{n_windows}_k{frames_per_window}'
        f'_c{clip_s}_fs{fs_out}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='data/EgoBrain')
    p.add_argument('--subjects', required=True, help='comma-sep ids or "all"')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--backend', default='wilor', choices=['wilor'])
    # Window geometry — MUST match the training/frame-cache config to key-align.
    p.add_argument('--window_s', type=float, default=1.0)
    p.add_argument('--stride_s', type=float, default=1.0)
    p.add_argument('--clip_s', type=float, default=4.0)
    p.add_argument('--fs_out', type=int, default=200)
    p.add_argument('--erp_latency_s', type=float, default=0.5)
    p.add_argument('--n_windows', type=int, default=2)
    p.add_argument('--frames_per_window', type=int, default=7,
                   help='K frames sampled across each window for motion (>=2)')
    # Labelling.
    p.add_argument('--move_thresh', type=float, default=0.15,
                   help='hand-lengths/s above which a hand counts as moving '
                        '(CALIBRATE on a hand-checked sample; continuous '
                        'metrics let you re-threshold without rerunning)')
    p.add_argument('--min_det_frac', type=float, default=0.4,
                   help='min fraction of frames a hand must be detected to vote')
    p.add_argument('--handedness_source', default='hybrid',
                   choices=['model', 'position', 'hybrid'])
    p.add_argument('--no_ego_compensate', action='store_true',
                   help='disable background-flow ego-motion compensation')
    p.add_argument('--max_frame_width', type=int, default=1280,
                   help='downscale wider frames for the detector (0 = native)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    p.add_argument('--overwrite', action='store_true',
                   help='re-derive subjects whose sidecar already exists '
                        '(only ever touches files in THIS cache dir)')
    p.add_argument('--dry_run', action='store_true',
                   help='validate config + timing without loading WiLoR')
    args = p.parse_args()

    if args.frames_per_window < 2:
        p.error('--frames_per_window must be >= 2 to measure motion')

    # cv2's internal threading conflicts with torch/OpenMP on multi-core GPU
    # nodes; single-thread cv2 (decode is ffmpeg, WiLoR is GPU, so this costs
    # ~nothing here). Set once, process-wide.
    import cv2
    cv2.setNumThreads(0)
    # Initialise CUDA up front, BEFORE any decord import (importing decord
    # before torch's lazy CUDA init segfaults model.to(cuda); HandEstimator is
    # already built before process_subject opens decord, but make it explicit).
    if not args.dry_run:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()

    out_dir = args.out_dir or default_out_dir(
        args.data_dir, args.backend, args.window_s, args.stride_s,
        args.erp_latency_s, args.n_windows, args.frames_per_window,
        args.clip_s, args.fs_out)
    cfg = dict(
        data_dir=args.data_dir, out_dir=out_dir, backend=args.backend,
        window_s=args.window_s, stride_s=args.stride_s, clip_s=args.clip_s,
        fs_out=args.fs_out, erp_latency_s=args.erp_latency_s,
        n_windows=args.n_windows, frames_per_window=args.frames_per_window,
        move_thresh=args.move_thresh, min_det_frac=args.min_det_frac,
        handedness_source=args.handedness_source,
        ego_compensate=not args.no_ego_compensate,
        max_frame_width=args.max_frame_width, overwrite=args.overwrite)

    if args.subjects.lower() == 'all':
        import re
        subjects = sorted(d for d in os.listdir(args.data_dir)
                          if re.match(r'^P\d{4}$', d)
                          and os.path.isdir(os.path.join(args.data_dir, d)))
    else:
        subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'label_mapping.json'), 'w') as f:
        json.dump({'labels': LABEL_NAMES,
                   'note': 'integers 0/1/2 match PhysioNet-MI (left/right/both '
                           'fist); 3 (both feet) intentionally excluded; -1 = '
                           'none/undetermined',
                   'config': {k: cfg[k] for k in cfg if k != 'data_dir'}}, f,
                  indent=2)
    print(f'[hand-labels] {len(subjects)} subject(s) → {out_dir}')

    estimator = None
    if not args.dry_run:
        estimator = HandEstimator(args.backend, args.device, args.dtype)

    results = []
    for sub in subjects:
        if args.dry_run:
            results.append({'subject': sub, 'status': 'dry_run'}); continue
        try:
            results.append(process_subject(sub, cfg, estimator))
        except Exception as e:                              # keep going
            results.append({'subject': sub, 'status': f'ERROR: {e}'})
            import traceback; traceback.print_exc()
    for r in results:
        if r['status'] not in {'ok', 'skip', 'no_video', 'dry_run'}:
            print(f"[err] {r['subject']}: {r['status']}", file=sys.stderr)
    ok = sum(r['status'] == 'ok' for r in results)
    print(f"[hand-labels] done: ok={ok} "
          f"skip={sum(r['status']=='skip' for r in results)} "
          f"no_video={sum(r['status']=='no_video' for r in results)} "
          f"err={sum(r['status'] not in {'ok','skip','no_video','dry_run'} for r in results)}")


if __name__ == '__main__':
    main()
