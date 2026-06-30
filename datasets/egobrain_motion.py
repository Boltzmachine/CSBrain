"""Per-anchor visual MOTION scores for the EgoBrain frame grid.

EgoBrain's egocentric video is mostly static (the wearer sits still for long
stretches), so a uniform anchor draw — :meth:`EgoBrainDataset._sample_base_slot`
samples ``k`` uniformly over the recording — spends most of the training budget
on near-frozen scenes where the world-model's next-frame prediction is trivial
(consecutive 1 s frames are ~identical; see project_consecutive_frame_triviality).
This module measures, per subject, how much the frame moves across the prediction
step so the sampler can bias toward dynamic moments.

For a step of ``step_slots`` grid slots the per-anchor score is the distance
between the frame representation at slot ``k`` and slot ``k+step_slots`` — exactly
the two frames an n_windows=2 / stride=step world-model sample sees (window-0
frame slot = k, window-1 frame slot = k + stride/grid). Two representation
**spaces** and two **metrics** are supported:

  * ``space='patch'`` — the per-patch DINOv2 grid tokens (n_slots, P, d), the
    world-model's actual frame-prediction target. Portable to V-JEPA (which has
    no CLS token), so this is the default. Read streaming from the (~15 GB/subj
    orient-0) ``grid`` slice of the embedding cache.
  * ``space='pixel'`` — the raw resized/cropped uint8 frames (n_slots, H, W, 3)
    from the frame grid cache. A low-level optical signal (captures head ego-
    motion + lighting as well as hand motion).
  * ``space='cls'`` — the global DINOv2 CLS vector (kept for reference; not used
    for V-JEPA-portable reweighting).

  * ``metric='l1'``  : mean |Δ| (the loss's metric; for pixels this is the
    classic mean-abs frame difference in [0,255]).
  * ``metric='cos'`` : 1 - cosine similarity. For ``patch`` it is the per-patch
    cosine averaged over patches (matches the model's per-token normalised
    cosine diagnostic); for ``pixel`` the cosine of the flattened frame.

Scores for ``k`` whose own slot or its ``k+step`` partner lacks a frame
(``has_image`` False) are NaN; the tail ``k >= n_slots - step`` is NaN too.
Scores cache to ``<emb_grid_dir>/_motioncache/<sub>_<space>_<metric>_d<step>.npy``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def motion_cache_dir(emb_grid_dir: str) -> str:
    return os.path.join(emb_grid_dir, '_motioncache')


def motion_cache_path(emb_grid_dir: str, sub: str, space: str, metric: str,
                      step_slots: int) -> str:
    return os.path.join(motion_cache_dir(emb_grid_dir),
                        f'{sub}_{space}_{metric}_d{step_slots}.npy')


def _pair_dist(a: np.ndarray, b: np.ndarray, metric: str,
               per_token: bool) -> np.ndarray:
    """Distance between matched arrays ``a``, ``b`` shaped (B, ...).

    ``per_token`` (patch space): the feature dim is the LAST axis and there is a
    token axis in between — cosine is taken per token then averaged over tokens;
    L1 is the mean abs diff over (tokens, features). Otherwise (cls / pixel) the
    whole non-batch shape is one flat vector.
    """
    a = a.astype(np.float32); b = b.astype(np.float32)
    if metric == 'l1':
        return np.abs(b - a).reshape(a.shape[0], -1).mean(axis=1)
    if metric != 'cos':
        raise ValueError(f"unknown metric {metric!r}; use 'l1' or 'cos'")
    if per_token:                                   # (B, P, d) -> mean_P (1-cos)
        dot = (a * b).sum(axis=-1)                  # (B, P)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cosp = dot / np.clip(na * nb, 1e-8, None)
        return (1.0 - cosp).mean(axis=1)
    af = a.reshape(a.shape[0], -1); bf = b.reshape(b.shape[0], -1)   # (B, D)
    dot = (af * bf).sum(axis=1)
    denom = np.clip(np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1),
                    1e-8, None)
    return 1.0 - dot / denom


def _streaming_motion(read_block, n: int, step_slots: int, metric: str,
                      per_token: bool, has: np.ndarray,
                      block: int = 3000) -> np.ndarray:
    """Per-anchor motion (n,) float32 over a large slot-indexed source read in
    blocks via ``read_block(s, e)`` -> array (e-s, ...). Each block reads an
    extra ``step_slots`` tail so ``motion[k]`` for ``k in [s, e)`` has its
    ``k+step`` partner in-block. NaN where either frame is missing / on the tail.
    """
    out = np.full(n, np.nan, dtype=np.float32)
    if n <= step_slots:
        return out
    end_valid = n - step_slots                       # last k with a partner
    for s in range(0, end_valid, block):
        e = min(s + block, end_valid)
        buf = read_block(s, e + step_slots)          # (e+step-s, ...)
        a = buf[:e - s]
        b = buf[step_slots:step_slots + (e - s)]
        d = _pair_dist(a, b, metric, per_token).astype(np.float32)
        valid = has[s:e] & has[s + step_slots:e + step_slots]
        out[s:e] = np.where(valid, d, np.nan)
    return out


def compute_subject_motion(path: str, step_slots: int, metric: str = 'cos',
                           space: str = 'patch',
                           block: int = 3000) -> np.ndarray:
    """Per-anchor motion (n_slots,) float32; NaN where undefined.

    ``path`` is the per-subject HDF5: the embedding cache for ``space in
    {'patch','cls'}`` (``grid`` / ``cls`` + ``has_image``), the frame grid cache
    for ``space=='pixel'`` (``frames`` + ``has_image``)."""
    import h5py
    h = h5py.File(path, 'r')
    try:
        has = np.asarray(h['has_image'][:]).astype(bool)
        if space == 'cls':
            cls = np.asarray(h['cls'][:, 0, :], dtype=np.float32)
            return _streaming_motion(lambda s, e: cls[s:e], cls.shape[0],
                                     step_slots, metric, False, has, block=10**9)
        if space == 'patch':
            ds = h['grid']                            # (n, 2, P, d)
            n = ds.shape[0]
            return _streaming_motion(
                lambda s, e: np.asarray(ds[s:e, 0], dtype=np.float32),
                n, step_slots, metric, True, has, block=block)
        if space == 'pixel':
            ds = h['frames']                          # (n, H, W, 3) uint8
            n = ds.shape[0]
            return _streaming_motion(
                lambda s, e: np.asarray(ds[s:e], dtype=np.float32),
                n, step_slots, metric, False, has, block=block)
        raise ValueError(f"unknown space {space!r}; use 'patch'/'pixel'/'cls'")
    finally:
        h.close()


def compute_subject_motion_both(path: str, step_slots: int, space: str,
                                block: int = 3000) -> dict:
    """Both metrics ('l1','cos') for a space in ONE read pass (the read/gzip-
    decompress is the cost; computing a second metric on the in-RAM block is
    ~free). Returns {'l1': (n,), 'cos': (n,)}."""
    import h5py
    h = h5py.File(path, 'r')
    try:
        has = np.asarray(h['has_image'][:]).astype(bool)
        if space == 'patch':
            ds = h['grid']; n = ds.shape[0]; per_tok = True
            reader = lambda s, e: np.asarray(ds[s:e, 0], dtype=np.float32)
        elif space == 'pixel':
            ds = h['frames']; n = ds.shape[0]; per_tok = False
            reader = lambda s, e: np.asarray(ds[s:e], dtype=np.float32)
        elif space == 'cls':
            cls = np.asarray(h['cls'][:, 0, :], dtype=np.float32)
            n = cls.shape[0]; per_tok = False; block = 10**9
            reader = lambda s, e: cls[s:e]
        else:
            raise ValueError(f"unknown space {space!r}")
        out = {'l1': np.full(n, np.nan, np.float32),
               'cos': np.full(n, np.nan, np.float32)}
        end_valid = n - step_slots
        for s in range(0, max(end_valid, 0), block):
            e = min(s + block, end_valid)
            buf = reader(s, e + step_slots)
            a = buf[:e - s]; b = buf[step_slots:step_slots + (e - s)]
            valid = has[s:e] & has[s + step_slots:e + step_slots]
            for metric in ('l1', 'cos'):
                d = _pair_dist(a, b, metric, per_tok).astype(np.float32)
                out[metric][s:e] = np.where(valid, d, np.nan)
        return out
    finally:
        h.close()


def load_or_compute_motion(emb_grid_dir: str, sub: str, step_slots: int,
                           metric: str = 'cos', space: str = 'patch',
                           frames_grid_dir: Optional[str] = None,
                           overwrite: bool = False) -> Optional[np.ndarray]:
    """Cached :func:`compute_subject_motion`. The score cache always lives under
    ``emb_grid_dir/_motioncache`` (the canonical motion home) regardless of
    space. ``frames_grid_dir`` is required for ``space=='pixel'``. Returns None
    if the subject's source HDF5 is absent (e.g. no-video subjects)."""
    if space == 'pixel':
        if frames_grid_dir is None:
            raise ValueError("space='pixel' needs frames_grid_dir")
        src = os.path.join(frames_grid_dir, f'{sub}.h5')
    else:
        src = os.path.join(emb_grid_dir, f'{sub}.h5')
    if not os.path.exists(src):
        return None
    cpath = motion_cache_path(emb_grid_dir, sub, space, metric, step_slots)
    if os.path.exists(cpath) and not overwrite:
        return np.load(cpath)
    mot = compute_subject_motion(src, step_slots, metric, space)
    os.makedirs(motion_cache_dir(emb_grid_dir), exist_ok=True)
    tmp = cpath + '.tmp.npy'
    np.save(tmp, mot)
    os.replace(tmp, cpath)
    return mot


# ---------------------------------------------------------------------------
# Anchor reweighting: turn the per-step motion array into a per-anchor sampling
# weight, then a CDF the dataset draws from. Pure (NumPy) so the dataset wiring
# stays thin and the logic is unit-testable in isolation.
# ---------------------------------------------------------------------------


def anchor_motion(motion_step: np.ndarray, n_windows: int,
                  step_slots: int) -> np.ndarray:
    """Span-aware per-anchor motion from the per-step array.

    A world-model sample anchored at slot ``k`` shows frames at slots ``k, k+step,
    ..., k+(n_windows-1)*step``; its visual "difficulty" is the motion ACROSS
    those frames. ``motion_step[k] = dist(cls[k], cls[k+step])`` already gives the
    consecutive-window distance, so the anchor score is the mean of the ``W-1``
    consecutive distances starting at ``k``. For ``n_windows==2`` this is exactly
    ``motion_step``. NaN where ANY of the ``W-1`` terms is undefined (so the
    anchor never samples a window whose frames are partly missing)."""
    motion_step = np.asarray(motion_step, dtype=np.float64)
    n = motion_step.shape[0]
    if n_windows <= 2:
        return motion_step.astype(np.float32)
    acc = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    for i in range(n_windows - 1):
        shift = i * step_slots
        seg = np.full(n, np.nan, dtype=np.float64)
        seg[:n - shift] = motion_step[shift:] if shift else motion_step
        fin = np.isfinite(seg)
        acc[fin] += seg[fin]
        cnt[fin] += 1.0
    full = cnt == (n_windows - 1)
    return np.where(full, acc / np.maximum(cnt, 1.0), np.nan).astype(np.float32)


def build_anchor_weights(am: np.ndarray, alpha: float,
                         cap_pct: Optional[float] = 99.0) -> np.ndarray:
    """Per-anchor sampling weight ``w[k] = clip(am[k], 0, cap)^alpha``; ``0``
    where ``am`` is undefined (NaN) so those anchors are never drawn. ``cap`` is
    the ``cap_pct`` percentile over the finite scores (winsorises the extreme
    tail — a handful of huge jumps don't swamp the budget). ``alpha=0`` makes
    every defined anchor equal-weight (recovers uniform-over-valid)."""
    w = np.asarray(am, dtype=np.float64)
    fin = np.isfinite(w)
    if not fin.any():
        return np.zeros_like(w)
    if cap_pct is not None and cap_pct < 100.0:
        cap = float(np.percentile(w[fin], cap_pct))
        w = np.minimum(w, cap)
    return np.where(fin, np.power(np.clip(w, 0.0, None), float(alpha)), 0.0)


def build_anchor_cdf(weights: np.ndarray, k_min: int, k_max: int,
                     floor_mix: float) -> Optional[np.ndarray]:
    """CDF over the inclusive anchor range ``[k_min, k_max]`` for weighted draws.

    Mixes ``floor_mix`` of a uniform distribution into the (normalised) motion
    weights so static regions keep a guaranteed floor of coverage and the draw
    can never starve part of the recording. Returns ``None`` (caller falls back
    to a uniform draw) when the range is empty or carries zero motion weight."""
    if k_max < k_min:
        return None
    seg = np.asarray(weights[k_min:k_max + 1], dtype=np.float64)
    L = seg.shape[0]
    tot = seg.sum()
    if L <= 0 or tot <= 0:
        return None
    fm = float(np.clip(floor_mix, 0.0, 1.0))
    p = (1.0 - fm) * (seg / tot) + fm * (1.0 / L)
    cdf = np.cumsum(p)
    cdf[-1] = 1.0                       # guard fp drift so searchsorted is total
    return cdf
