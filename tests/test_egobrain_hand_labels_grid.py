"""Pure-logic tests for datasets/egobrain_extract_hand_labels_grid (no GPU/
WiLoR/cv2/decord). Covers the grid geometry (slot count matches the frame
grid), the reference-window aggregation over synthetic per-slot Hand tracks,
the knob-agnostic cache slug, and the config-mismatch guard.

Run:
    conda run -n cbramod python -m pytest tests/test_egobrain_hand_labels_grid.py -q
"""
import os

import numpy as np
import pytest

from datasets.egobrain_hand_labels import Hand
from datasets.egobrain_extract_hand_labels_grid import (
    n_slots_for, ref_window_radius, aggregate_grid, default_grid_out_dir,
    _config_mismatch, _FORMAT_VERSION,
)

# 21-joint template: wrist at origin, middle-MCP one unit up -> palm scale = 1.
_TMPL = np.zeros((21, 2), np.float64)
_TMPL[9] = (0.0, 1.0)
for _j in range(1, 21):
    if _j != 9:
        _TMPL[_j] = (0.1 * _j, 0.05 * _j)


def make_hand(center, scale=1.0):
    kp = _TMPL * scale + np.asarray(center, float)[None, :]
    return Hand(kp, 1.0)


def moving_track(n, start=(0.0, 0.0), step=(0.2, 0.0), scale=1.0):
    return [make_hand(np.asarray(start, float) + k * np.asarray(step, float),
                      scale) for k in range(n)]


def _dts(n, dt=0.2):
    """dts[0] unused (no pair before slot 0); dts[k>=1] = dt."""
    return [None] + [dt] * (n - 1)


# --------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------

def test_n_slots_matches_frame_grid_formula():
    # Must equal datasets.egobrain_extract_frames_grid's slot count exactly, or
    # label slot k won't align with frame slot k.
    from datasets.egobrain_extract_frames_grid import _extract_subject  # noqa
    n_clips, clip_s, grid_s, margin = 3, 4.0, 0.2, 2.0
    want = int(np.floor((n_clips * clip_s + margin) / grid_s)) + 1
    assert n_slots_for(n_clips, clip_s, grid_s, margin) == want
    # 3*4 + 2 = 14 s / 0.2 = 70 -> +1 = 71
    assert n_slots_for(3, 4.0, 0.2, 2.0) == 71


def test_ref_window_radius():
    assert ref_window_radius(1.0, 0.2) == 2      # round(1.0/0.4)=2 -> 2R=0.8s span
    assert ref_window_radius(2.0, 0.2) == 5
    assert ref_window_radius(0.1, 0.2) == 1      # clamped to >=1 (need a pair)


def test_slug_is_knob_agnostic():
    d = default_grid_out_dir('data/EgoBrain', 'wilor', 0.2, 1.0, 200)
    assert d.endswith('cache_hand_labels_grid_wilor_g0.2_r1.0_fs200')
    base = os.path.basename(d)
    # No vision encoder, no window/stride/erp/n_windows/clip in the slug.
    for tok in ('dinov2', '_w1.0', 's1.0', '_e0', '_nw', '_c4', '_sz'):
        assert tok not in base


# --------------------------------------------------------------------------
# Aggregation over synthetic per-slot tracks
# --------------------------------------------------------------------------

def test_moving_left_still_right_intensities():
    n = 9
    left = moving_track(n, step=(0.2, 0.0), scale=1.0)   # 0.2 px/slot, scale 1
    right = moving_track(n, step=(0.0, 0.0), scale=1.0)  # detected but still
    ego = [None] * n
    dts = _dts(n, dt=0.2)
    has_video = np.ones(n, bool)
    out = aggregate_grid(left, right, ego, dts, has_video,
                         grid_s=0.2, hand_ref_s=1.0)
    # Interior slot: speed = |0.2px| / scale 1 / dt 0.2s = 1.0 hand-lengths/s.
    assert np.isfinite(out['left_intensity'][4])
    assert abs(out['left_intensity'][4] - 1.0) < 1e-5
    # Still right hand: seen (pairs exist) -> 0.0, NOT NaN.
    assert out['right_intensity'][4] == 0.0
    # Both hands detected in every window slot.
    assert out['left_det_frac'][4] == 1.0
    assert out['right_det_frac'][4] == 1.0
    assert out['has_video'].all()


def test_undetected_hand_is_nan():
    n = 7
    left = moving_track(n, step=(0.3, 0.0))
    right = [None] * n                                   # never detected
    out = aggregate_grid(left, right, [None] * n, _dts(n), np.ones(n, bool),
                         grid_s=0.2, hand_ref_s=1.0)
    assert np.isnan(out['right_intensity'][3])           # unmeasurable -> NaN
    assert out['right_det_frac'][3] == 0.0
    assert np.isfinite(out['left_intensity'][3])


def test_no_valid_pairs_is_nan():
    # Hands detected but no valid dt anywhere -> no measurable pair -> NaN.
    n = 5
    left = moving_track(n)
    out = aggregate_grid(left, [None] * n, [None] * n, [None] * n,
                         np.ones(n, bool), grid_s=0.2, hand_ref_s=1.0)
    assert np.isnan(out['left_intensity'][2])


def test_scale_invariance():
    # Doubling the hand scale AND the pixel step leaves the normalised speed
    # unchanged (speed is in hand-lengths/s).
    n = 7
    a = aggregate_grid(moving_track(n, step=(0.2, 0.0), scale=1.0),
                       [None] * n, [None] * n, _dts(n), np.ones(n, bool),
                       grid_s=0.2, hand_ref_s=1.0)
    b = aggregate_grid(moving_track(n, step=(0.4, 0.0), scale=2.0),
                       [None] * n, [None] * n, _dts(n), np.ones(n, bool),
                       grid_s=0.2, hand_ref_s=1.0)
    assert abs(a['left_intensity'][3] - b['left_intensity'][3]) < 1e-5


def test_ego_compensation_cancels_uniform_shift():
    # A pure background pan that moves the hand's pixels but not its articulation
    # is subtracted out -> ~0 speed after ego compensation.
    n = 7
    shift = np.array([0.5, 0.0])
    left = [make_hand(k * shift) for k in range(n)]       # hand rides the pan
    ego = [None] + [shift.copy() for _ in range(n - 1)]   # same pan per pair
    out = aggregate_grid(left, [None] * n, ego, _dts(n), np.ones(n, bool),
                         grid_s=0.2, hand_ref_s=1.0)
    assert abs(out['left_intensity'][3]) < 1e-6


def test_config_mismatch(tmp_path):
    import h5py
    cfg = dict(backend='wilor', grid_s=0.2, hand_ref_s=1.0, fs_out=200,
               clip_s=4.0, margin_s=2.0, handedness_source='hybrid',
               ego_compensate=True, max_frame_width=1280)
    p = str(tmp_path / 'P0001.h5')
    with h5py.File(p, 'w') as h:
        h.create_dataset('left_intensity', data=np.zeros(3, np.float32))
        for k, v in cfg.items():
            h.attrs[k] = v
        h.attrs['format_version'] = _FORMAT_VERSION
    assert _config_mismatch(p, cfg) is None                     # exact match
    bad = dict(cfg); bad['grid_s'] = 0.1
    assert 'grid_s' in _config_mismatch(p, bad)                 # spacing differs
    bad2 = dict(cfg); bad2['handedness_source'] = 'model'
    assert 'handedness_source' in _config_mismatch(p, bad2)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
