"""Tests for multi-modal score fusion and session analytics."""

from __future__ import annotations

import unittest

from biometric_engine.analytics import AuthEvent, SessionAnalytics
from biometric_engine.ml_fusion import FusionConfig, FusionResult, ScoreFusion


class TestFusionConfig(unittest.TestCase):
    def test_weights_normalize(self) -> None:
        cfg = FusionConfig(recognition_weight=3.0, liveness_weight=1.0)
        self.assertAlmostEqual(cfg.recognition_weight, 0.75)
        self.assertAlmostEqual(cfg.liveness_weight, 0.25)

    def test_invalid_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            FusionConfig(fusion_method="unknown")

    def test_zero_weights_raises(self) -> None:
        with self.assertRaises(ValueError):
            FusionConfig(recognition_weight=0.0, liveness_weight=0.0)


class TestScoreFusion(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = ScoreFusion()

    def test_weighted_sum_accepts_high_scores(self) -> None:
        result = self.fusion.fuse(0.90, 0.85)
        self.assertTrue(result.accepted)
        self.assertGreater(result.fused_score, result.threshold)

    def test_weighted_sum_rejects_low_scores(self) -> None:
        result = self.fusion.fuse(0.20, 0.25)
        self.assertFalse(result.accepted)

    def test_fused_score_within_range(self) -> None:
        result = self.fusion.fuse(0.60, 0.70)
        self.assertGreaterEqual(result.fused_score, 0.0)
        self.assertLessEqual(result.fused_score, 1.0)

    def test_product_method(self) -> None:
        cfg = FusionConfig(fusion_method="weighted_product", fused_threshold=0.50)
        fusion = ScoreFusion(cfg)
        result = fusion.fuse(0.90, 0.85)
        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "weighted_product")

    def test_min_method(self) -> None:
        cfg = FusionConfig(fusion_method="min", fused_threshold=0.50)
        fusion = ScoreFusion(cfg)
        result = fusion.fuse(0.90, 0.30)  # Min is 0.30 → reject at 0.50
        self.assertFalse(result.accepted)
        self.assertAlmostEqual(result.fused_score, 0.30, places=2)

    def test_threshold_override(self) -> None:
        result = self.fusion.fuse(0.60, 0.60, threshold=0.90)
        self.assertFalse(result.accepted)
        result2 = self.fusion.fuse(0.60, 0.60, threshold=0.10)
        self.assertTrue(result2.accepted)

    def test_explanation_is_populated(self) -> None:
        result = self.fusion.fuse(0.75, 0.65)
        self.assertIn("Recognition", result.explanation)
        self.assertIn("Liveness", result.explanation)
        self.assertIn("Fused", result.explanation)

    def test_roc_curve_requires_history(self) -> None:
        self.assertEqual(self.fusion.roc_curve(), [])

    def test_roc_curve_with_history(self) -> None:
        for _ in range(10):
            self.fusion.record_outcome(0.85, is_genuine=True)
            self.fusion.record_outcome(0.30, is_genuine=False)
        curve = self.fusion.roc_curve()
        self.assertGreater(len(curve), 0)
        # Verify structure
        point = curve[0]
        for key in ("threshold", "far", "tar", "frr"):
            self.assertIn(key, point)

    def test_eer_is_nan_without_history(self) -> None:
        import math
        self.assertTrue(math.isnan(self.fusion.eer))

    def test_eer_reasonable_with_history(self) -> None:
        for _ in range(20):
            self.fusion.record_outcome(0.85, is_genuine=True)
            self.fusion.record_outcome(0.30, is_genuine=False)
        eer = self.fusion.eer
        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 0.5)

    def test_stats_dict_has_all_keys(self) -> None:
        stats = self.fusion.stats
        for key in ("genuine_samples", "impostor_samples", "eer", "adaptive_threshold",
                    "recognition_weight", "liveness_weight", "fusion_method"):
            self.assertIn(key, stats)


class TestSessionAnalytics(unittest.TestCase):
    def _make_event(self, accepted: bool, is_genuine: bool) -> AuthEvent:
        return AuthEvent(
            timestamp="2026-01-01T00:00:00Z",
            subject_id="alice" if accepted else None,
            claimed_id="alice",
            accepted=accepted,
            recognition_score=0.80 if accepted else 0.20,
            liveness_score=0.75 if accepted else 0.25,
            fused_score=0.78 if accepted else 0.22,
            attack_type_hint="genuine" if is_genuine else "static",
            latency_ms=120.0,
            challenge="blink",
            is_genuine=is_genuine,
        )

    def setUp(self) -> None:
        self.analytics = SessionAnalytics(session_id="test-session")

    def test_empty_report(self) -> None:
        report = self.analytics.report()
        self.assertEqual(report.events_total, 0)
        self.assertEqual(report.accepts, 0)

    def test_record_and_report(self) -> None:
        self.analytics.record(self._make_event(True, is_genuine=True))
        self.analytics.record(self._make_event(False, is_genuine=False))
        report = self.analytics.report()
        self.assertEqual(report.events_total, 2)
        self.assertEqual(report.accepts, 1)
        self.assertEqual(report.denials, 1)

    def test_far_and_frr(self) -> None:
        # 2 genuine (1 pass, 1 fail) + 2 impostor (1 accept — FA, 1 deny)
        self.analytics.record(self._make_event(True, is_genuine=True))
        self.analytics.record(self._make_event(False, is_genuine=True))    # FRR
        self.analytics.record(self._make_event(True, is_genuine=False))    # FAR
        self.analytics.record(self._make_event(False, is_genuine=False))
        report = self.analytics.report()
        self.assertAlmostEqual(report.far, 0.5, places=2)   # 1/2 impostors accepted
        self.assertAlmostEqual(report.frr, 0.5, places=2)   # 1/2 genuines rejected
        self.assertAlmostEqual(report.tar, 0.5, places=2)   # 1 - frr

    def test_time_series(self) -> None:
        self.analytics.record(self._make_event(True, is_genuine=True))
        ts = self.analytics.time_series("fused_score")
        self.assertEqual(len(ts), 1)
        self.assertIn("timestamp", ts[0])
        self.assertIn("value", ts[0])

    def test_score_distribution(self) -> None:
        for _ in range(5):
            self.analytics.record(self._make_event(True, is_genuine=True))
            self.analytics.record(self._make_event(False, is_genuine=False))
        dist = self.analytics.score_distribution(bins=5)
        self.assertIn("bin_edges", dist)
        self.assertIn("genuine", dist)
        self.assertIn("impostor", dist)
        self.assertEqual(len(dist["genuine"]), 5)

    def test_len(self) -> None:
        self.assertEqual(len(self.analytics), 0)
        self.analytics.record(self._make_event(True, is_genuine=True))
        self.assertEqual(len(self.analytics), 1)

    def test_enrollment_tracking(self) -> None:
        self.analytics.record_enrollment("alice")
        self.analytics.record_enrollment("bob")
        report = self.analytics.report()
        self.assertEqual(report.enrolled_subjects, 2)

    def test_to_dict(self) -> None:
        self.analytics.record(self._make_event(True, is_genuine=True))
        d = self.analytics.to_dict()
        self.assertIn("events_total", d)
        self.assertEqual(d["events_total"], 1)


if __name__ == "__main__":
    unittest.main()
