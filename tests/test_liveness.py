from __future__ import annotations

import unittest

from datalake_offline_biometrics.image_ops import synthetic_face
from datalake_offline_biometrics.liveness import LivenessDetector, LivenessResult


class TestLivenessDetector(unittest.TestCase):
    def setUp(self) -> None:
        # Use a lower threshold appropriate for synthetic CLAHE-normalized frames
        # (real camera frames have stronger motion, texture, and blink signals)
        self.detector = LivenessDetector(threshold=0.40)

    def test_insufficient_frames_raises(self) -> None:
        frames = [synthetic_face(), synthetic_face()]
        with self.assertRaises(ValueError):
            self.detector.assess(frames)

    def test_static_frames_rejected(self) -> None:
        # A static image repeated across all frames (photo presentation attack)
        single_frame = synthetic_face(noise=0)
        frames = [single_frame for _ in range(8)]
        result = self.detector.assess(frames)

        self.assertFalse(result.passed)
        self.assertIn("insufficient frame-to-frame motion", result.reasons)
        self.assertLess(result.score, self.detector.threshold)

    def test_dynamic_motion_returns_higher_score_than_static(self) -> None:
        """Dynamic frames should score higher than static repeated frames."""
        static_frame = synthetic_face(noise=0)
        static_frames = [static_frame for _ in range(5)]
        static_result = self.detector.assess(static_frames)

        dynamic_frames = [
            synthetic_face(shift_x=0, blink=False, noise=4),
            synthetic_face(shift_x=2, blink=False, noise=5),
            synthetic_face(shift_x=3, blink=False, noise=4),
            synthetic_face(shift_x=2, blink=False, noise=5),
            synthetic_face(shift_x=0, blink=False, noise=4),
        ]
        dynamic_result = self.detector.assess(dynamic_frames)

        self.assertGreater(dynamic_result.score, static_result.score)

    def test_challenge_response_unmet(self) -> None:
        # No blink occurs, but challenge asks for blink
        frames = [
            synthetic_face(shift_x=0, blink=False, noise=1),
            synthetic_face(shift_x=1, blink=False, noise=1),
            synthetic_face(shift_x=0, blink=False, noise=1),
        ]
        result = self.detector.assess(frames, challenge=["blink"])
        self.assertIn("challenge not satisfied: blink", result.reasons)

    def test_result_has_required_metrics(self) -> None:
        frames = [
            synthetic_face(shift_x=k, noise=2) for k in range(5)
        ]
        result = self.detector.assess(frames)
        for key in ("motion", "texture", "entropy", "ms_lbp_entropy", "symmetry", "optical_flow_arc"):
            self.assertIn(key, result.metrics, f"missing metric: {key}")

    def test_attack_type_hint_present(self) -> None:
        single_frame = synthetic_face(noise=0)
        frames = [single_frame for _ in range(5)]
        result = self.detector.assess(frames)
        self.assertIn(result.attack_type_hint, ("static", "replay", "printed", "genuine", "unknown"))

    def test_static_attack_classified(self) -> None:
        single_frame = synthetic_face(noise=0)
        frames = [single_frame for _ in range(5)]
        result = self.detector.assess(frames)
        self.assertEqual(result.attack_type_hint, "static")

    def test_nod_challenge_supported(self) -> None:
        frames = [
            synthetic_face(shift_y=k, noise=2) for k in range(5)
        ]
        result = self.detector.assess(frames, challenge=["nod"])
        self.assertIn("challenge_nod", result.metrics)

    def test_liveness_result_fields(self) -> None:
        frames = [synthetic_face(noise=2) for _ in range(5)]
        result = self.detector.assess(frames)
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.threshold, float)
        self.assertIsInstance(result.attack_type_hint, str)
        self.assertIsInstance(result.attack_confidence, float)
        self.assertIsInstance(result.reasons, tuple)
        self.assertIsInstance(result.metrics, dict)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_optical_flow_arc_greater_with_motion(self) -> None:
        static_frames = [synthetic_face(noise=0) for _ in range(5)]
        static_result = self.detector.assess(static_frames)

        motion_frames = [synthetic_face(shift_x=i * 3, noise=2) for i in range(5)]
        motion_result = self.detector.assess(motion_frames)

        static_arc = static_result.metrics.get("optical_flow_arc", 0.0)
        motion_arc = motion_result.metrics.get("optical_flow_arc", 0.0)
        self.assertGreater(motion_arc, static_arc)


if __name__ == "__main__":
    unittest.main()
