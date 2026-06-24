"""Pure-logic tests for datasets/egobrain_hand_labels (no GPU/WiLoR/cv2/decord).

Covers: window→video timing matches egobrain_extract_frames, chapter routing,
the motion aggregation + discrete label rules, scale invariance, ego-motion
compensation, the continuous re-threshold path, and handedness assignment.
"""
import unittest

import numpy as np

from datasets.egobrain_hand_labels import (
    LABEL_LEFT, LABEL_RIGHT, LABEL_BOTH, LABEL_NONE,
    window_frame_times, resolve_chapter, build_chapters,
    Hand, parse_wilor_frame, aggregate_window, recompute_labels,
)

# 21-joint template: wrist at origin, middle-MCP one unit "up" → palm scale = 1.
_TMPL = np.zeros((21, 2), np.float64)
_TMPL[9] = (0.0, 1.0)
for j in range(1, 21):                       # spread the rest so kp isn't degenerate
    if j != 9:
        _TMPL[j] = (0.1 * j, 0.05 * j)


def make_hand(center, scale, score=1.0):
    kp = _TMPL * scale + np.asarray(center, float)[None, :]
    return Hand(kp, score)


def moving_track(n, start, step, scale):
    """n Hands translating by `step` px each frame at fixed `scale`."""
    return [make_hand(np.asarray(start, float) + k * np.asarray(step, float),
                      scale) for k in range(n)]


def uniform_dts(n, dt=1 / 6):
    """Per-pair dt list: dts[0] is unused, dts[k>=1] = dt."""
    return [None] + [dt] * (n - 1)


class TestTiming(unittest.TestCase):
    def test_center_matches_extractor_formula(self):
        # The middle frame must equal egobrain_extract_frames' single-frame
        # center time: t0 + (i*stride_samples + window_samples/2)/fs_out + erp.
        clip_s, window_s, stride_s, fs_out, erp = 4.0, 1.0, 1.0, 200, 0.5
        for c in (0, 5, 17):
            for i in (0, 1):
                ts = window_frame_times(
                    c, i, clip_s=clip_s, window_s=window_s, stride_s=stride_s,
                    fs_out=fs_out, erp_latency_s=erp, frames_per_window=7)
                win_samp = round(window_s * fs_out)
                stride_samp = round(stride_s * fs_out)
                expect = c * clip_s + (i * stride_samp + win_samp / 2) / fs_out + erp
                self.assertAlmostEqual(ts[len(ts) // 2], expect, places=9)

    def test_span_and_monotonic(self):
        ts = window_frame_times(2, 1, clip_s=4.0, window_s=1.0, stride_s=1.0,
                                fs_out=200, erp_latency_s=0.5,
                                frames_per_window=7)
        self.assertEqual(len(ts), 7)
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))
        # window i=1 spans [t0+1, t0+2] s in-clip, +erp; t0 = 2*4 = 8.
        self.assertAlmostEqual(ts[0], 8.0 + 1.0 + 0.5, places=9)
        self.assertAlmostEqual(ts[-1], 8.0 + 2.0 + 0.5, places=9)


class TestChapters(unittest.TestCase):
    def test_build_and_resolve(self):
        info = {'video_offset_s': 2.0,
                'chapters': [{'path': 'a.mp4', 'duration_s': 10.0},
                             {'path': 'b.mp4', 'duration_s': 10.0}]}
        chs = build_chapters(info, '/root')
        self.assertEqual(chs[0]['start_s'], 0.0)
        self.assertEqual(chs[1]['start_s'], 10.0)
        self.assertTrue(chs[0]['path'].endswith('/root/a.mp4'))
        # eeg_t=3 → video t=1 → chapter a, local 1.
        ch, local = resolve_chapter(chs, 2.0, 3.0)
        self.assertTrue(ch['path'].endswith('a.mp4'))
        self.assertAlmostEqual(local, 1.0)
        # eeg_t=14 → video 12 → chapter b, local 2.
        ch, local = resolve_chapter(chs, 2.0, 14.0)
        self.assertTrue(ch['path'].endswith('b.mp4'))
        self.assertAlmostEqual(local, 2.0)
        # Before video start and past the end → None.
        self.assertIsNone(resolve_chapter(chs, 2.0, 1.0))
        self.assertIsNone(resolve_chapter(chs, 2.0, 100.0))

    def test_unknown_duration_tail_is_open(self):
        chs = build_chapters({'chapters': [{'path': 'a.mp4'}]}, '/r')
        ch, local = resolve_chapter(chs, 0.0, 999.0)
        self.assertAlmostEqual(local, 999.0)


class TestAggregate(unittest.TestCase):
    KW = dict(dts=uniform_dts(7), move_thresh=0.15, min_det_frac=0.4)

    def _ego_none(self, n):
        return [None] * n

    def test_left_only(self):
        n = 7
        left = moving_track(n, (0, 0), (2.0, 0), scale=10.0)   # 0.2 hl @ dt → speed>thr
        right = moving_track(n, (50, 0), (0.0, 0), scale=10.0)  # still
        m = aggregate_window(left, right, self._ego_none(n), **self.KW)
        self.assertEqual(m['label'], LABEL_LEFT)
        self.assertGreater(m['left_intensity'], self.KW['move_thresh'])
        self.assertAlmostEqual(m['right_intensity'], 0.0, places=6)

    def test_right_only(self):
        n = 7
        left = moving_track(n, (0, 0), (0.0, 0), scale=10.0)
        right = moving_track(n, (50, 0), (3.0, 0), scale=10.0)
        m = aggregate_window(left, right, self._ego_none(n), **self.KW)
        self.assertEqual(m['label'], LABEL_RIGHT)

    def test_both(self):
        n = 7
        left = moving_track(n, (0, 0), (2.0, 0), scale=10.0)
        right = moving_track(n, (50, 0), (2.0, 0), scale=10.0)
        m = aggregate_window(left, right, self._ego_none(n), **self.KW)
        self.assertEqual(m['label'], LABEL_BOTH)
        self.assertAlmostEqual(m['drasticness'],
                               max(m['left_intensity'], m['right_intensity']),
                               places=6)

    def test_none_when_both_still(self):
        n = 7
        left = moving_track(n, (0, 0), (0.0, 0), scale=10.0)
        right = moving_track(n, (50, 0), (0.0, 0), scale=10.0)
        m = aggregate_window(left, right, self._ego_none(n), **self.KW)
        self.assertEqual(m['label'], LABEL_NONE)

    def test_undetected_hand_is_nan_not_zero(self):
        n = 7
        left = moving_track(n, (0, 0), (2.0, 0), scale=10.0)
        right = [None] * n                                     # never seen
        m = aggregate_window(left, right, self._ego_none(n), **self.KW)
        self.assertEqual(m['label'], LABEL_LEFT)
        self.assertTrue(np.isnan(m['right_intensity']))
        self.assertEqual(m['right_det_frac'], 0.0)

    def test_det_frac_gate(self):
        # Moving fast but only detected in 2/7 frames (<0.4) → not active.
        n = 7
        left = [make_hand((k * 5.0, 0), 10.0) if k < 2 else None
                for k in range(n)]
        right = [None] * n
        m = aggregate_window(left, right, self._ego_none(n), **self.KW)
        self.assertEqual(m['label'], LABEL_NONE)

    def test_scale_invariance(self):
        # 2x bigger hand moving 2x the pixels → identical intensity.
        n = 7
        a = moving_track(n, (0, 0), (2.0, 0), scale=10.0)
        b = moving_track(n, (0, 0), (4.0, 0), scale=20.0)
        ma = aggregate_window(a, [None] * n, self._ego_none(n), **self.KW)
        mb = aggregate_window(b, [None] * n, self._ego_none(n), **self.KW)
        self.assertAlmostEqual(ma['left_intensity'], mb['left_intensity'],
                               places=6)

    def test_ego_compensation_cancels_global_shift(self):
        # Hand "moves" only because the whole frame pans by (3,0)/frame.
        n = 7
        left = moving_track(n, (0, 0), (3.0, 0), scale=10.0)
        ego = [None] + [np.array([3.0, 0.0])] * (n - 1)
        m_comp = aggregate_window(left, [None] * n, ego, **self.KW)
        m_raw = aggregate_window(left, [None] * n, self._ego_none(n), **self.KW)
        self.assertAlmostEqual(m_comp['left_intensity'], 0.0, places=6)
        self.assertGreater(m_raw['left_intensity'], 0.0)
        self.assertEqual(m_comp['label'], LABEL_NONE)

    def test_no_consecutive_pair_is_nan_not_zero(self):
        # Detected in 3 of 7 frames but NEVER in adjacent frames → speed is
        # UNMEASURABLE → intensity NaN (not 0.0), label NONE.
        n = 7
        left = [make_hand((0, 0), 10.0) if k in (0, 3, 6) else None
                for k in range(n)]
        m = aggregate_window(left, [None] * n, self._ego_none(n),
                             dts=uniform_dts(n), move_thresh=0.15,
                             min_det_frac=0.3)
        self.assertTrue(np.isnan(m['left_intensity']))
        self.assertEqual(m['label'], LABEL_NONE)

    def test_duplicate_frame_pair_excluded(self):
        # Slots 2 and 3 are the SAME decoded frame (duplicate). Marking that
        # pair's dt None must drop its spurious zero-speed contribution, so the
        # intensity is HIGHER than naively counting the zero pair.
        n = 7
        xs = [0, 2, 4, 4, 6, 8, 10]                 # slot2==slot3 (duplicate)
        left = [make_hand((x, 0), 10.0) for x in xs]
        dts_dup = [None, 1/6, 1/6, None, 1/6, 1/6, 1/6]   # pair 2->3 skipped
        m_skip = aggregate_window(left, [None] * n, self._ego_none(n),
                                  dts=dts_dup, move_thresh=0.15,
                                  min_det_frac=0.4)
        m_naive = aggregate_window(left, [None] * n, self._ego_none(n),
                                   dts=uniform_dts(n), move_thresh=0.15,
                                   min_det_frac=0.4)
        self.assertGreater(m_skip['left_intensity'], m_naive['left_intensity'])
        self.assertAlmostEqual(m_skip['left_intensity'], 0.2 / (1 / 6),
                               places=6)   # all kept pairs move 2px @ scale10


class TestRecompute(unittest.TestCase):
    def test_matches_aggregate(self):
        li = np.array([[0.30, 0.30], [np.nan, 0.40]], np.float32)
        ri = np.array([[0.02, 0.50], [0.05, np.nan]], np.float32)
        ldf = np.array([[1.0, 1.0], [0.0, 1.0]], np.float32)
        rdf = np.array([[1.0, 1.0], [1.0, 0.0]], np.float32)
        lab = recompute_labels(li, ri, ldf, rdf, move_thresh=0.15,
                               min_det_frac=0.4)
        # [0,0] left moving, right still → LEFT
        self.assertEqual(lab[0, 0], LABEL_LEFT)
        # [0,1] both above thr → BOTH
        self.assertEqual(lab[0, 1], LABEL_BOTH)
        # [1,0] left never detected (det 0), right below thr → NONE
        self.assertEqual(lab[1, 0], LABEL_NONE)
        # [1,1] left moving, right undetected → LEFT
        self.assertEqual(lab[1, 1], LABEL_LEFT)

    def test_rethreshold_changes_labels(self):
        li = np.array([[0.20]], np.float32)
        ri = np.array([[0.01]], np.float32)
        df = np.array([[1.0]], np.float32)
        self.assertEqual(
            recompute_labels(li, ri, df, df, move_thresh=0.15,
                             min_det_frac=0.4)[0, 0], LABEL_LEFT)
        self.assertEqual(
            recompute_labels(li, ri, df, df, move_thresh=0.30,
                             min_det_frac=0.4)[0, 0], LABEL_NONE)


class TestHandedness(unittest.TestCase):
    def _out(self, kp_x, is_right, score=1.0):
        kp = _TMPL.copy() + np.array([kp_x, 0.0])
        return {'is_right': float(is_right), 'score': score,
                'wilor_preds': {'pred_keypoints_2d': kp[None]}}

    def test_model_mode(self):
        outs = [self._out(10, 0), self._out(200, 1)]
        l, r = parse_wilor_frame(outs, 'model')
        self.assertAlmostEqual(l.wrist[0], 10)
        self.assertAlmostEqual(r.wrist[0], 200)

    def test_hybrid_splits_two_same_side(self):
        # Model wrongly calls both hands right → hybrid splits by x.
        outs = [self._out(10, 1), self._out(200, 1)]
        l, r = parse_wilor_frame(outs, 'hybrid')
        self.assertIsNotNone(l); self.assertIsNotNone(r)
        self.assertLess(l.wrist[0], r.wrist[0])

    def test_position_mode_single_hand(self):
        outs = [self._out(10, 1)]                 # model says right, but it's
        l, r = parse_wilor_frame(outs, 'position', img_w=640)  # in the left half
        self.assertIsNotNone(l); self.assertIsNone(r)

    def test_keep_highest_score_same_side(self):
        outs = [self._out(10, 1, score=0.3), self._out(20, 1, score=0.9)]
        l, r = parse_wilor_frame(outs, 'model')
        self.assertIsNone(l)
        self.assertAlmostEqual(r.wrist[0], 20)    # higher score wins


if __name__ == '__main__':
    unittest.main()
