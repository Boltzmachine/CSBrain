"""Unit tests for :mod:`datasets.euclidean_alignment`.

Covers:

1. ``_matrix_sqrt_inv`` is a left-inverse of the matrix square root:
   ``R^{-1/2} R (R^{-1/2})^T == I`` for SPD R.

2. ``compute_ea_matrix`` produces a matrix that, when applied to the
   original trials, makes their mean covariance close to (scaled) I —
   the defining property of EA whitening (Eq. 9-10 of arXiv:2601.17883).

3. With ``rescale=True``, EA preserves per-trial RMS so the rest of the
   project's /100 trainer convention still gives ~unit-RMS model input.

4. ``apply_ea`` handles both ``(C, T)`` and batched ``(B, C, T)`` shapes
   and is a true linear transform (additivity check).

5. The opt-out path (no ea_matrices passed) leaves the dataset output
   bit-identical to the pre-EA behaviour — guaranteeing the flag is a
   true no-op when off.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class TestMatrixSqrtInv(unittest.TestCase):
    def test_inverse_sqrt_property(self):
        from datasets.euclidean_alignment import _matrix_sqrt_inv
        rng = np.random.default_rng(42)
        for C in (4, 8, 32, 64):
            A = rng.standard_normal((C, C))
            R = A @ A.T + 0.1 * np.eye(C)               # SPD
            R_inv_sqrt = _matrix_sqrt_inv(R)
            # R^{-1/2} R R^{-1/2} == I
            should_be_I = R_inv_sqrt @ R @ R_inv_sqrt.T
            np.testing.assert_allclose(
                should_be_I, np.eye(C), atol=1e-6,
                err_msg=f'C={C}: not identity')

    def test_handles_singular_covariance(self):
        from datasets.euclidean_alignment import _matrix_sqrt_inv
        # Rank-deficient covariance: one zero eigenvalue.
        R = np.diag([1.0, 0.5, 0.0])
        R_inv_sqrt = _matrix_sqrt_inv(R, eps=1e-4)
        # Should not be inf/NaN; the zero eigenvalue is floored at eps.
        self.assertTrue(np.isfinite(R_inv_sqrt).all())


class TestComputeEaMatrix(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # Two synthetic subjects with different channel covariances. Each
        # subject has 50 trials of (C=8, T=100). Per-subject EA should
        # whiten one subject's mean covariance to ~I.
        self.rng = rng

    def _make_subject_trials(self, A, n_trials=50, T=100):
        """Return ``n_trials`` of shape (C, T) where each col x_t ~ N(0, A A^T)."""
        C = A.shape[0]
        return [A @ self.rng.standard_normal((C, T)) for _ in range(n_trials)]

    def test_whitened_mean_cov_is_identity_when_rescale_off(self):
        from datasets.euclidean_alignment import (
            compute_ea_matrix, apply_ea,
        )
        # Strong cross-channel correlation:
        A = np.array([[2.0, 1.0, 0.0, 0.0],
                      [1.0, 2.0, 0.5, 0.0],
                      [0.0, 0.5, 1.5, 0.3],
                      [0.0, 0.0, 0.3, 1.0]])
        trials = self._make_subject_trials(A, n_trials=200, T=200)
        R_inv_sqrt, n, s_raw = compute_ea_matrix(
            iter(trials), n_channels=4, rescale=False)
        self.assertEqual(n, 200)
        # Apply whitening and compute the mean second-moment matrix of the
        # result. Per Eq. 9 the paper uses an un-normalised X X^T (no /T),
        # so after whitening the mean of Y Y^T over trials should be I.
        whitened = [apply_ea(x, R_inv_sqrt) for x in trials]
        mean_cov = np.mean([w @ w.T for w in whitened], axis=0)
        diag = np.diag(mean_cov)
        np.testing.assert_allclose(diag, np.ones(4), rtol=0.05,
                                    err_msg='diagonal not unity')
        offdiag = mean_cov - np.diag(diag)
        self.assertLess(np.abs(offdiag).max(), 0.05,
                        f'off-diagonal too large: {np.abs(offdiag).max():.3f}')

    def test_rescale_preserves_signal_rms(self):
        """rescale=True should leave post-EA RMS ≈ pre-EA RMS so the
        downstream /100 trainer convention still applies.
        """
        from datasets.euclidean_alignment import (
            compute_ea_matrix, apply_ea,
        )
        A = np.array([[3.0, 0.5], [0.5, 2.0]])
        trials = self._make_subject_trials(A, n_trials=200, T=200)
        rms_raw = float(np.sqrt(np.mean([np.mean(t**2) for t in trials])))
        R_inv_sqrt, n, s_raw = compute_ea_matrix(
            iter(trials), n_channels=2, rescale=True)
        whitened = [apply_ea(x, R_inv_sqrt) for x in trials]
        rms_aligned = float(np.sqrt(
            np.mean([np.mean(w**2) for w in whitened])))
        # The rescale matches s_raw to within sampling noise.
        self.assertAlmostEqual(rms_raw, rms_aligned, delta=0.1 * rms_raw)

    def test_per_subject_independence(self):
        """Subject A's EA matrix should NOT whiten subject B's trials.

        Concretely: subject B's mean covariance after applying SUBJECT A's
        matrix is far from identity, whereas after applying its OWN
        matrix it's close to identity.
        """
        from datasets.euclidean_alignment import (
            compute_ea_matrix, apply_ea,
        )
        A_a = np.array([[3.0, 0.0], [0.0, 1.0]])
        A_b = np.array([[1.0, 0.0], [0.0, 3.0]])
        trials_a = self._make_subject_trials(A_a, n_trials=400, T=200)
        trials_b = self._make_subject_trials(A_b, n_trials=400, T=200)
        R_a, _, _ = compute_ea_matrix(iter(trials_a), 2, rescale=False)
        R_b, _, _ = compute_ea_matrix(iter(trials_b), 2, rescale=False)
        # Same subject -> mean (Y Y^T) close to I (per paper Eq 9; no /T).
        cov_b_own = np.mean([apply_ea(x, R_b) @ apply_ea(x, R_b).T
                             for x in trials_b], axis=0)
        self.assertLess(np.abs(cov_b_own - np.eye(2)).max(), 0.05)
        # Cross-subject -> NOT identity. B has var ratio (1, 9); A has (9, 1).
        # Applying R_a to B's trials whitens ch 0 by sqrt(9)/sqrt(1)=3 too much,
        # and ch 1 by sqrt(1)/sqrt(9)=1/3 too little — final diag ratio ≈ 1:81.
        cov_b_wrong = np.mean(
            [apply_ea(x, R_a) @ apply_ea(x, R_a).T
             for x in trials_b], axis=0)
        ratio = np.diag(cov_b_wrong).max() / np.diag(cov_b_wrong).min()
        self.assertGreater(ratio, 50,
                           f'cross-subject EA still left near-identity: '
                           f'ratio={ratio:.1f}, cov={cov_b_wrong}')


class TestApplyEa(unittest.TestCase):
    def test_batched_shape(self):
        from datasets.euclidean_alignment import apply_ea
        rng = np.random.default_rng(0)
        R = rng.standard_normal((4, 4)).astype(np.float32)
        x = rng.standard_normal((2, 5, 4, 100)).astype(np.float32)  # (B,B2,C,T)
        out = apply_ea(x, R)
        self.assertEqual(out.shape, x.shape)
        # Spot check the first sample matches a manual matmul.
        manual = np.einsum('ij,jt->it', R, x[0, 0])
        np.testing.assert_allclose(out[0, 0], manual, rtol=1e-5)

    def test_linearity(self):
        from datasets.euclidean_alignment import apply_ea
        rng = np.random.default_rng(0)
        R = rng.standard_normal((4, 4)).astype(np.float32)
        a = rng.standard_normal((4, 50)).astype(np.float32)
        b = rng.standard_normal((4, 50)).astype(np.float32)
        np.testing.assert_allclose(
            apply_ea(a + b, R), apply_ea(a, R) + apply_ea(b, R),
            atol=1e-5)


class TestSidecarRoundTrip(unittest.TestCase):
    def test_save_load_round_trip(self):
        from datasets.euclidean_alignment import (
            save_ea_matrices, load_ea_matrices,
        )
        rng = np.random.default_rng(0)
        matrices = {
            'sub-01': rng.standard_normal((8, 8)).astype(np.float32),
            'sub-02': rng.standard_normal((8, 8)).astype(np.float32),
        }
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'ea.pt')
            save_ea_matrices(matrices, p, meta={'dataset': 'test'})
            loaded = load_ea_matrices(p)
            self.assertEqual(set(loaded.keys()), set(matrices.keys()))
            for k in matrices:
                np.testing.assert_allclose(loaded[k], matrices[k])


class TestEgoBrainOptIntegration(unittest.TestCase):
    """When ``ea_matrices=None`` the dataset must behave bit-identically
    to its pre-EA self. Use the same synthetic-cache helper as the
    existing EgoBrain tests.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='ea_egobrain_test_')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_cache(self, root, sub='P0001', n_clips=4):
        import json
        sub_dir = os.path.join(root, sub)
        os.makedirs(sub_dir, exist_ok=True)
        ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4',
                    'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
                    'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                    'Pz', 'P4', 'P8', 'PO7', 'PO3', 'PO4', 'PO8', 'Oz']
        rng = np.random.default_rng(0)
        for c in range(n_clips):
            arr = rng.standard_normal((len(ch_names), 800)).astype(np.float32)
            np.save(os.path.join(sub_dir, f'{c}.npy'), arr, allow_pickle=False)
        meta = {
            'fs_in': 256, 'fs_out': 200, 'clip_s': 4.0,
            'n_clips': n_clips, 'n_channels': len(ch_names),
            'ch_names': ch_names,
            'edf_path': f'{sub}/fake.edf', 'video': None, 'events': [],
            'preprocess': {'l_freq': 0.3, 'h_freq': 75.0, 'notch_hz': 50.0},
        }
        with open(os.path.join(sub_dir, 'clips.json'), 'w') as f:
            json.dump(meta, f)

    def test_ea_off_is_identical_to_no_ea(self):
        from datasets.egobrain_dataset import EgoBrainDataset
        cache = os.path.join(self.tmpdir, 'cache')
        self._make_fake_cache(cache, 'P0001', n_clips=3)
        ds = EgoBrainDataset(
            data_dir=self.tmpdir, subjects=['P0001'], cache_dir=cache,
            in_dim=40, n_windows=2, window_s=2.0, stride_s=1.0,
            clip_s=4.0, load_frames=False, max_channels=32,
            ea_matrices=None,
        )
        s = ds[0]
        # Without EA the loaded clip should equal the raw .npy after the
        # 32-channel whitelist.
        from datasets.egobrain_dataset import _resolve_ch_coords
        raw = np.load(os.path.join(cache, 'P0001', '0.npy'))
        import json
        with open(os.path.join(cache, 'P0001', 'clips.json')) as f:
            meta = json.load(f)
        _, keep_mask, _ = _resolve_ch_coords(meta['ch_names'])
        # Dataset config: window_s=2.0 @ fs_out=200 → 400 samples; in_dim=40
        # → 10 patches per window. Window 0 starts at sample 0.
        expected = raw[keep_mask][:, :400]
        actual_window0 = s['timeseries'][0].numpy()
        np.testing.assert_allclose(
            actual_window0,
            expected.reshape(32, 10, 40), rtol=1e-6,
            err_msg='ea_matrices=None should be byte-identical to no-EA path')

    def test_ea_on_changes_output(self):
        from datasets.egobrain_dataset import EgoBrainDataset
        cache = os.path.join(self.tmpdir, 'cache')
        self._make_fake_cache(cache, 'P0001', n_clips=3)
        rng = np.random.default_rng(7)
        # Non-trivial whitening matrix.
        R = rng.standard_normal((32, 32)).astype(np.float32)
        ds_off = EgoBrainDataset(
            data_dir=self.tmpdir, subjects=['P0001'], cache_dir=cache,
            in_dim=40, n_windows=2, window_s=2.0, stride_s=1.0,
            clip_s=4.0, load_frames=False, max_channels=32,
            ea_matrices=None,
        )
        ds_on = EgoBrainDataset(
            data_dir=self.tmpdir, subjects=['P0001'], cache_dir=cache,
            in_dim=40, n_windows=2, window_s=2.0, stride_s=1.0,
            clip_s=4.0, load_frames=False, max_channels=32,
            ea_matrices={'P0001': R},
        )
        s_off = ds_off[0]['timeseries']
        s_on = ds_on[0]['timeseries']
        # EA must change the output (clearly distinct).
        self.assertFalse(torch.allclose(s_off, s_on),
                         'EA matrix had no effect on dataset output')
        # The lookup for a missing subject should pass through unchanged.
        ds_miss = EgoBrainDataset(
            data_dir=self.tmpdir, subjects=['P0001'], cache_dir=cache,
            in_dim=40, n_windows=2, window_s=2.0, stride_s=1.0,
            clip_s=4.0, load_frames=False, max_channels=32,
            ea_matrices={'P9999': R},
        )
        s_miss = ds_miss[0]['timeseries']
        self.assertTrue(torch.allclose(s_miss, s_off),
                        'subject not in EA dict should pass through unchanged')


if __name__ == '__main__':
    unittest.main()
