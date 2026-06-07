"""Euclidean Alignment (EA) preprocessing — He & Wu 2020 / Wu 2025.

Per-subject whitening that maps each subject's mean trial covariance to the
identity, reducing the cross-subject second-order distribution shift that
hurts BCI generalisation. The technique is the third entry in Table II of
arXiv:2601.17883 (EEG foundation models survey) — Eq. 9-10:

    R̄_s = (1/n_s) Σ_i  X_i X_i^T        # over subject s's n_s trials
    G_ea(X) = R̄_s^{-1/2}  X              # applied to every trial of subject s

This module provides:

  * ``compute_ea_matrix`` — accumulate R̄ from a trial iterator, return a
    rescaled R̄^{-1/2} (see "scale matching" below).
  * ``apply_ea`` — vectorised matmul along the channel dim.
  * ``save_ea_matrices`` / ``load_ea_matrices`` — torch.save sidecar I/O.

Scale matching
--------------
The original EA formula leaves whitened signals at unit RMS per channel.
The rest of the project assumes raw-µV cached signals divided by 100 at
training time (CSBrain trainer convention). To make EA a drop-in
replacement without changing every trainer + dataset's /100, we multiply
the whitening matrix by ``s_raw`` — the mean per-trial RMS of the raw
signal across that subject's training trials — so post-EA signals land on
the same magnitude as the raw signal. This is signal-magnitude-preserving:
EA still removes per-subject covariance structure, but the absolute RMS
matches what the rest of the pipeline expects.

When ``rescale=False``, raw whitening (unit RMS) is returned for users who
want to skip the /100 step themselves.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np


def _matrix_sqrt_inv(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return R^{-1/2} via symmetric eigendecomposition.

    ``R`` must be symmetric PSD (a covariance matrix). ``eps`` is added to
    each eigenvalue before the inverse-sqrt so rank-deficient or
    near-singular covariances stay numerically stable.
    """
    R = np.asarray(R, dtype=np.float64)
    R = 0.5 * (R + R.T)                                 # symmetrise
    w, V = np.linalg.eigh(R)
    w = np.clip(w, a_min=eps, a_max=None)
    return (V * (w ** -0.5)) @ V.T                      # = V diag(w^-1/2) V^T


def compute_ea_matrix(trial_iter: Iterable[np.ndarray],
                      n_channels: int,
                      eps: float = 1e-6,
                      rescale: bool = True
                      ) -> tuple[np.ndarray, int, float]:
    """Compute one subject's EA whitening matrix from its trial stream.

    Parameters
    ----------
    trial_iter
        Yields ``(C, T)`` float arrays for every trial belonging to the
        subject. Anything in the trial — windowing / patching / normalisation
        choices — should match what the model sees at training time.
    n_channels
        Expected C. Trials with a different channel count are skipped with
        a stderr warning (heterogeneous-channel subjects can't be aligned
        in one C×C matrix).
    eps
        Eigenvalue floor for numerical stability of the inverse square root.
    rescale
        If True (default), multiply the matrix by the trials' mean per-trial
        RMS so the post-EA signal magnitude matches the raw input. See module
        docstring.

    Returns
    -------
    (R_inv_sqrt, n_trials, s_raw)
        ``R_inv_sqrt`` is ``(C, C)`` float64. ``n_trials`` is the count of
        trials actually used. ``s_raw`` is the mean per-trial RMS (returned
        even when ``rescale=False`` so callers can audit it).
    """
    import sys
    R = np.zeros((n_channels, n_channels), dtype=np.float64)
    sq_sum = 0.0                # for RMS = sqrt(E[x^2])
    el_count = 0                # total elements that went into sq_sum
    n_trials = 0
    n_skip = 0
    for X in trial_iter:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'trial must be (C, T); got shape {X.shape}')
        if X.shape[0] != n_channels:
            n_skip += 1
            continue
        R += X @ X.T
        sq_sum += float((X * X).sum())
        el_count += X.size
        n_trials += 1
    if n_trials == 0:
        raise RuntimeError('no trials with the expected channel count')
    if n_skip:
        print(f'[ea] skipped {n_skip} trial(s) with non-{n_channels}-channel '
              f'shapes', file=sys.stderr)
    R /= n_trials                                        # mean covariance
    R_inv_sqrt = _matrix_sqrt_inv(R, eps=eps)
    s_raw = float(np.sqrt(sq_sum / el_count))            # per-element RMS
    if rescale:
        # The paper's R̄ accumulates X X^T (no /T), so after whitening
        # E[(R̄^{-1/2} X)_{i,t}^2] = 1/T per element — i.e. post-EA per-
        # element RMS is 1/sqrt(T). To match the pre-EA per-element RMS
        # (so the downstream /100 trainer step still produces a unit-ish
        # signal), multiply by s_raw * sqrt(T_avg).
        T_avg = el_count / (n_trials * n_channels)
        R_inv_sqrt = R_inv_sqrt * (s_raw * float(np.sqrt(T_avg)))
    return R_inv_sqrt.astype(np.float32), n_trials, s_raw


def finalize_ea_matrix(R_sum: np.ndarray,
                       n_trials: int,
                       sq_sum: float,
                       el_count: int,
                       n_channels: int,
                       eps: float = 1e-6,
                       rescale: bool = True
                       ) -> tuple[np.ndarray, float]:
    """Turn streaming accumulators into a (rescaled) whitening matrix.

    Used by the per-dataset scan loops in ``datasets/compute_ea.py`` that
    can't materialise all trials at once but accumulate R_sum = Σ X X^T,
    sq_sum = Σ Σ X^2, n_trials, and el_count incrementally.

    Returns ``(R_inv_sqrt_float32, s_raw)`` — matching the
    ``compute_ea_matrix`` API minus the trial count.
    """
    if n_trials == 0:
        raise RuntimeError('no trials accumulated')
    R = R_sum / n_trials
    R_inv_sqrt = _matrix_sqrt_inv(R, eps=eps)
    s_raw = float(np.sqrt(sq_sum / el_count))
    if rescale:
        T_avg = el_count / (n_trials * n_channels)
        R_inv_sqrt = R_inv_sqrt * (s_raw * float(np.sqrt(T_avg)))
    return R_inv_sqrt.astype(np.float32), s_raw


def apply_ea(x: np.ndarray, R_inv_sqrt: np.ndarray) -> np.ndarray:
    """Apply a precomputed whitening matrix along the channel axis.

    ``x`` shape: ``(C, T)`` or ``(..., C, T)`` (last two axes channel×time).
    Returns an array of the same shape and dtype as ``x``.
    """
    if x.shape[-2] != R_inv_sqrt.shape[1]:
        raise ValueError(
            f'channel-count mismatch: x has {x.shape[-2]} channels but '
            f'EA matrix expects {R_inv_sqrt.shape[1]}')
    # einsum keeps batch dims intact while doing a (C,C)@(C,T) matmul.
    R = R_inv_sqrt.astype(x.dtype, copy=False)
    return np.einsum('ij,...jt->...it', R, x, optimize=True)


# ---------------------------------------------------------------------------
# Sidecar I/O — torch.save dict of subject_id -> (C, C) float32 matrix.
# Stored alongside each dataset's existing cache so a single ``--dataset_dir``
# argument keeps both finds them, but in a separate file so the existing
# cache is untouched (the user can delete the sidecar to roll back).
# ---------------------------------------------------------------------------


def save_ea_matrices(matrices: dict[str, np.ndarray], path: str,
                     meta: Optional[dict] = None) -> None:
    """Persist per-subject EA matrices + metadata via torch.save.

    File layout: ``{"matrices": {sub_id: ndarray}, "meta": {...}}``.
    """
    import torch
    payload = {
        'matrices': {k: np.asarray(v, dtype=np.float32) for k, v in matrices.items()},
        'meta': dict(meta or {}),
    }
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ea_matrices(path: str) -> dict[str, np.ndarray]:
    """Read the per-subject matrices from a sidecar file.

    Returns just the {subject_id: matrix} dict; metadata is logged to stderr
    so callers can sanity-check the matrices were built for the right
    sampling rate / channel set without forcing every consumer to plumb the
    meta around.
    """
    import sys
    import torch
    payload = torch.load(path, weights_only=False)
    matrices = payload['matrices']
    meta = payload.get('meta', {})
    if meta:
        items = ', '.join(f'{k}={v}' for k, v in meta.items())
        print(f'[ea] loaded {len(matrices)} matrices from {path} ({items})',
              file=sys.stderr)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in matrices.items()}
