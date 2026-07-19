"""Pure-logic tests for datasets/egobrain_extract_hand_labels_grid (no GPU/
WiLoR/cv2/decord). Covers the grid geometry (slot count matches the frame
grid), the RAW (unsmoothed) single-pair speed aggregation over synthetic
per-slot Hand tracks, the knob-agnostic cache slug, and the config-mismatch
guard.

Run:
    conda run -n cbramod python -m pytest tests/test_egobrain_hand_labels_grid.py -q
"""
import os

import numpy as np
import pytest

from datasets.egobrain_hand_labels import Hand
from datasets.egobrain_extract_hand_labels_grid import (
    n_slots_for, aggregate_grid_raw, default_grid_out_dir,
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
    """dts[k] = real elapsed time of the pair (k-1, k); dts[0] unused."""
    return [None] + [dt] * (n - 1)


# --------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------

def test_n_slots_matches_frame_grid_formula():
    # Must equal datasets.egobrain_extract_frames_grid's slot count exactly, or
    # label slot k won't align with frame slot k.
    n_clips, clip_s, grid_s, margin = 3, 4.0, 0.2, 2.0
    want = int(np.floor((n_clips * clip_s + margin) / grid_s)) + 1
    assert n_slots_for(n_clips, clip_s, grid_s, margin) == want
    assert n_slots_for(3, 4.0, 0.2, 2.0) == 71          # 14s/0.2 = 70 -> +1


def test_slug_is_knob_agnostic_and_marks_raw():
    d = default_grid_out_dir('data/EgoBrain', 'wilor', 0.2, 200)
    assert d.endswith('cache_hand_labels_grid_wilor_g0.2_raw_fs200')
    base = os.path.basename(d)
    assert '_raw_' in base                              # distinct from legacy _r1.0_
    # No vision encoder, no window/stride/erp/n_windows/clip/hand_ref in the slug.
    for tok in ('dinov2', '_w1.0', 's1.0', '_e0', '_nw', '_c4', '_sz', '_r1.0'):
        assert tok not in base


# --------------------------------------------------------------------------
# RAW single-pair (forward) speed — the whole point: NO temporal averaging
# --------------------------------------------------------------------------

def test_forward_pair_semantics_no_averaging():
    """intensity[s] must be the speed of the pair (s, s+1) ALONE — not an
    average over neighbours. Build a track whose per-pair speed VARIES, and
    check each slot recovers exactly its own forward pair's speed."""
    # Hand jumps by 0.1,0.2,0.3,0.4 px on pairs (0,1),(1,2),(2,3),(3,4).
    steps = [0.1, 0.2, 0.3, 0.4]
    pos = np.cumsum([0.0] + steps)
    left = [make_hand((p, 0.0), scale=1.0) for p in pos]     # 5 slots
    n = len(left)
    out = aggregate_grid_raw(left, [None] * n, [None] * n, _dts(n, dt=0.2),
                             np.ones(n, bool))
    li = out['left_intensity']
    # slot s -> pair (s, s+1) -> speed = steps[s] / scale 1 / dt 0.2
    for s, st in enumerate(steps):
        assert abs(li[s] - st / 0.2) < 1e-4, f"slot {s}: {li[s]} != {st/0.2}"
    # A moving average would have blurred these distinct values together.
    assert not np.allclose(li[:4], li[:4].mean())
    # last slot has no forward pair
    assert np.isnan(li[n - 1])


def test_moving_left_still_right():
    n = 6
    left = moving_track(n, step=(0.2, 0.0), scale=1.0)   # 0.2px/slot, scale 1
    right = moving_track(n, step=(0.0, 0.0), scale=1.0)  # detected but still
    out = aggregate_grid_raw(left, right, [None] * n, _dts(n, 0.2),
                             np.ones(n, bool))
    assert abs(out['left_intensity'][2] - 1.0) < 1e-5    # 0.2px/1/0.2s
    assert out['right_intensity'][2] == 0.0              # seen & still -> 0, not NaN
    assert out['left_det'][2] and out['right_det'][2]
    assert out['has_video'].all()


def test_undetected_hand_is_nan_and_det_is_per_slot():
    n = 5
    left = moving_track(n, step=(0.3, 0.0))
    right = [None] * n
    right[2] = make_hand((0.0, 0.0))                     # detected at slot 2 only
    out = aggregate_grid_raw(left, right, [None] * n, _dts(n), np.ones(n, bool))
    assert np.isfinite(out['left_intensity'][1])
    # right: pair (2,3) needs BOTH frames -> slot 3 is None -> NaN everywhere
    assert np.isnan(out['right_intensity'][2])
    # det is PER-SLOT (not a windowed fraction)
    assert out['right_det'].tolist() == [False, False, True, False, False]


def test_missing_dt_is_nan():
    # Hands detected but the pair has no valid dt (duplicate/missing frame).
    n = 4
    left = moving_track(n)
    dts = [None] * n                                     # no valid pair anywhere
    out = aggregate_grid_raw(left, [None] * n, [None] * n, dts, np.ones(n, bool))
    assert np.isnan(out['left_intensity']).all()


def test_scale_invariance():
    # Doubling hand scale AND pixel step leaves the normalised speed unchanged.
    n = 5
    a = aggregate_grid_raw(moving_track(n, step=(0.2, 0.0), scale=1.0),
                           [None] * n, [None] * n, _dts(n), np.ones(n, bool))
    b = aggregate_grid_raw(moving_track(n, step=(0.4, 0.0), scale=2.0),
                           [None] * n, [None] * n, _dts(n), np.ones(n, bool))
    assert abs(a['left_intensity'][2] - b['left_intensity'][2]) < 1e-5


def test_ego_compensation_cancels_uniform_shift():
    # A pure background pan that carries the hand's pixels but not its
    # articulation is subtracted out -> ~0 speed.
    n = 5
    shift = np.array([0.5, 0.0])
    left = [make_hand(k * shift) for k in range(n)]
    ego = [None] + [shift.copy() for _ in range(n - 1)]   # ego[k] = pair (k-1,k)
    out = aggregate_grid_raw(left, [None] * n, ego, _dts(n), np.ones(n, bool))
    assert abs(out['left_intensity'][2]) < 1e-6


def test_derive_backward_and_window_offline():
    """The stored raw array must make any other convention derivable with numpy
    alone — no WiLoR rerun. Backward at s == stored[s-1]; a centred W-average is
    a plain nanmean."""
    steps = [0.1, 0.2, 0.3, 0.4]
    pos = np.cumsum([0.0] + steps)
    left = [make_hand((p, 0.0)) for p in pos]
    n = len(left)
    li = aggregate_grid_raw(left, [None] * n, [None] * n, _dts(n, 0.2),
                            np.ones(n, bool))['left_intensity']
    backward_at_3 = li[2]                                # pair (2,3)
    assert abs(backward_at_3 - 0.3 / 0.2) < 1e-4
    centred = np.nanmean(li[0:4])                        # 4-pair average
    assert abs(centred - np.mean([s / 0.2 for s in steps])) < 1e-4


def test_config_mismatch(tmp_path):
    import h5py
    cfg = dict(backend='wilor', grid_s=0.2, fs_out=200, clip_s=4.0,
               margin_s=2.0, handedness_source='hybrid', ego_compensate=True,
               max_frame_width=1280)
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


def test_config_mismatch_rejects_legacy_smoothed_v1(tmp_path):
    import h5py
    cfg = dict(backend='wilor', grid_s=0.2, fs_out=200, clip_s=4.0,
               margin_s=2.0, handedness_source='hybrid', ego_compensate=True,
               max_frame_width=1280)
    p = str(tmp_path / 'P0001.h5')
    with h5py.File(p, 'w') as h:
        for k, v in cfg.items():
            h.attrs[k] = v
        h.attrs['format_version'] = 1                    # legacy smoothed layout
    assert 'format_version' in _config_mismatch(p, cfg)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
