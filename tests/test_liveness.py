from __future__ import annotations

import unittest

from datalake_offline_biometrics.image_ops import synthetic_face
from datalake_offline_biometrics.liveness import LivenessDetector, LivenessResult


class TestLivenessDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = LivenessDetector(threshold=0.60)

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

    def test_dynamic_motion_with_blink_accepted(self) -> None:
        # Natural motion with a blink in the middle
        frames = [
            synthetic_face(shift_x=0, blink=False, noise=2),
            synthetic_face(shift_x=1, blink=False, noise=3),
            synthetic_face(shift_x=2, blink=True, noise=2),
            synthetic_face(shift_x=1, blink=False, noise=3),
            synthetic_face(shift_x=0, blink=False, noise=2),
        ]
        result = self.detector.assess(frames, challenge=["blink"])

        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, self.detector.threshold)
        self.assertIn("motion", result.metrics)
        self.assertIn("texture", result.metrics)
        self.assertIn("challenge_blink", result.metrics)

    def test_challenge_response_unmet(self) -> None:
        # No blink occurs, but challenge asks for blink
        frames = [
            synthetic_face(shift_x=0, blink=False, noise=1),
            synthetic_face(shift_x=1, blink=False, noise=1),
            synthetic_face(shift_x=0, blink=False, noise=1),
        ]
        result = self.detector.assess(frames, challenge=["blink"])
        self.assertIn("challenge not satisfied: blink", result.reasons)


if __name__ == "__main__":
    unittest.main()
