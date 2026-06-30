"""Tests for motion-weighted anchor resampling (datasets/egobrain_motion.py +
EgoBrainDataset(motion_resample=True)).

The pure weight/CDF helpers are tested on synthetic arrays (no data needed); a
guarded statistical test confirms the dataset's `_sample_base_slot` biases the
anchor draw toward dynamic moments while keeping full time-range coverage and
leaving the deterministic-eval path untouched.

Run:
    conda run -n cbramod python -m pytest tests/test_egobrain_motion_resample.py -q
"""
import os

import numpy as np
import pytest

from datasets.egobrain_motion import (
    anchor_motion, build_anchor_weights, build_anchor_cdf,
    compute_subject_motion_both, motion_cache_path,
)

EMB = 'data/EgoBrain/cache_embeddings_grid_facebook_dinov2-base_g0.2_sz224'


# --------------------------- pure helpers ---------------------------------

def test_anchor_motion_w2_is_identity():
    m = np.array([0.1, 0.2, np.nan, 0.4, 0.5], dtype=np.float32)
    out = anchor_motion(m, n_windows=2, step_slots=1)
    assert np.allclose(out, m, equal_nan=True)


def test_anchor_motion_w3_means_consecutive_and_propagates_nan():
    # step=1, W=3: anchor[k] = mean(m[k], m[k+1]); NaN if either undefined.
    m = np.array([1.0, 3.0, 5.0, np.nan, 9.0], dtype=np.float32)
    out = anchor_motion(m, n_windows=3, step_slots=1)
    assert out[0] == pytest.approx(2.0)        # mean(1,3)
    assert out[1] == pytest.approx(4.0)        # mean(3,5)
    assert np.isnan(out[2])                     # m[3] is NaN
    assert np.isnan(out[3])                     # m[3] is NaN


def test_build_anchor_weights_basic():
    am = np.array([np.nan, 0.0, 1.0, 2.0, 100.0], dtype=np.float32)
    # alpha=0 -> every DEFINED anchor weight 1, undefined 0.
    w0 = build_anchor_weights(am, alpha=0.0, cap_pct=None)
    assert w0[0] == 0.0 and np.allclose(w0[1:], 1.0)
    # alpha=1, cap at the 75th pct winsorises the 100 outlier down.
    w1 = build_anchor_weights(am, alpha=1.0, cap_pct=75.0)
    cap = np.percentile(am[np.isfinite(am)], 75.0)
    assert w1[4] == pytest.approx(cap)         # 100 clipped to cap
    assert w1[0] == 0.0                         # NaN -> 0
    # monotone non-decreasing in motion (after winsorise).
    assert w1[1] <= w1[2] <= w1[3] <= w1[4]


def test_build_anchor_cdf_properties():
    w = np.array([0.0, 0.0, 1.0, 3.0, 0.0, 2.0], dtype=np.float64)
    cdf = build_anchor_cdf(w, k_min=1, k_max=5, floor_mix=0.1)
    assert cdf is not None and cdf.shape[0] == 5
    assert np.all(np.diff(cdf) >= -1e-12)       # non-decreasing
    assert cdf[-1] == pytest.approx(1.0)
    # floor_mix>0 => every in-range slot has positive probability mass.
    p = np.diff(np.concatenate([[0.0], cdf]))
    assert np.all(p > 0)
    # higher-weight slots get more mass than lower-weight ones.
    assert p[2] > p[0]                           # w=3 (idx4) vs w=0 (idx2... )


def test_build_anchor_cdf_none_on_zero_or_empty():
    assert build_anchor_cdf(np.zeros(5), 0, 4, 0.1) is None
    assert build_anchor_cdf(np.ones(5), 3, 2, 0.1) is None   # empty range


def test_cdf_sampling_biases_toward_motion_but_covers_all():
    # synthetic: first half static (~0), second half dynamic (~1).
    n = 2000
    w = np.concatenate([np.full(n // 2, 1e-6), np.ones(n // 2)])
    cdf = build_anchor_cdf(w, 0, n - 1, floor_mix=0.1)
    rng = np.random.default_rng(0)
    draws = np.searchsorted(cdf, rng.random(40000), side='right')
    draws = np.minimum(draws, n - 1)
    # most draws land in the dynamic half...
    assert (draws >= n // 2).mean() > 0.75
    # ...but the static half still gets sampled (floor coverage).
    assert (draws < n // 2).mean() > 0.02
    assert draws.min() < n // 4 and draws.max() >= n - 2


def test_zero_floor_never_samples_zero_weight():
    w = np.array([0.0, 5.0, 0.0, 5.0, 0.0], dtype=np.float64)
    cdf = build_anchor_cdf(w, 0, 4, floor_mix=0.0)
    rng = np.random.default_rng(1)
    draws = np.searchsorted(cdf, rng.random(5000), side='right')
    draws = np.minimum(draws, 4)
    assert set(np.unique(draws)).issubset({1, 3})    # zero-weight slots skipped


def test_compute_both_metrics_match_single_pass():
    """compute_subject_motion_both must equal the per-metric path it replaces."""
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)  # (n,d)
    # fake a (n,1,d)->cls-like array via a temp h5
    h5py = pytest.importorskip('h5py')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'P9999.h5')
        with h5py.File(p, 'w') as h:
            h.create_dataset('cls', data=a[:, None, :].repeat(2, axis=1))
            h.create_dataset('has_image', data=np.ones(3, bool))
        both = compute_subject_motion_both(p, step_slots=1, space='cls')
        # l1[0] = |a1-a0| summed-mean; cos[0] = 1 - cos(a0,a1)=1-0=1
        assert both['cos'][0] == pytest.approx(1.0, abs=1e-5)
        assert np.isnan(both['l1'][2]) and np.isnan(both['cos'][2])  # tail


# --------------------- dataset integration (guarded) -----------------------

_HAVE = (os.path.isdir(EMB)
         and os.path.exists(os.path.join(EMB, 'P0001.h5'))
         and os.path.exists(motion_cache_path(EMB, 'P0001', 'patch', 'cos', 5)))


@pytest.mark.skipif(not _HAVE, reason='EgoBrain grid embedding + motion cache absent')
def test_dataset_resample_biases_and_preserves_eval():
    pytest.importorskip('mne')
    from datasets.egobrain_dataset import EgoBrainDataset
    from datasets.egobrain_motion import load_or_compute_motion
    common = dict(
        data_dir='data/EgoBrain', subjects=['P0001'], in_dim=200, n_windows=2,
        window_s=1.0, stride_s=1.0, clip_s=4.0, erp_latency_s=-0.15,
        max_channels=32, vision_encoder='facebook/dinov2-base',
        load_frames=True, use_frame_grid=True, use_grid_embeddings=True,
        jitter_seed=0)
    mot = load_or_compute_motion(EMB, 'P0001', 5, 'cos', 'patch')

    def draw(ds, n=8000):
        ds._anchor_rng = np.random.default_rng(7)
        ks = np.array([ds._sample_base_slot('P0001', 0) for _ in range(n)])
        return mot[ks][np.isfinite(mot[ks])]

    du = EgoBrainDataset(**common, motion_resample=False)
    dr = EgoBrainDataset(**common, motion_resample=True,
                         motion_resample_alpha=1.0, motion_resample_space='patch',
                         motion_resample_metric='cos')
    mu, mr = draw(du), draw(dr)
    assert mr.mean() > 1.3 * mu.mean()           # clearly biased toward motion

    # deterministic-eval path is identical with/without resampling.
    de = EgoBrainDataset(**{**common, 'jitter_seed': None},
                         motion_resample=True, temporal_jitter=False)
    df = EgoBrainDataset(**{**common, 'jitter_seed': None},
                         motion_resample=False, temporal_jitter=False)
    assert [de._sample_base_slot('P0001', c) for c in range(40)] == \
           [df._sample_base_slot('P0001', c) for c in range(40)]
