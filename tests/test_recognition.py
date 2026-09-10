from __future__ import annotations

import unittest

from datalake_offline_biometrics.image_ops import synthetic_face
from datalake_offline_biometrics.recognition import (
    FaceTemplate,
    LBPHFaceRecognizer,
    UNIFORM_LBP_LOOKUP,
)


class TestRecognition(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer = LBPHFaceRecognizer()

    def test_uniform_lbp_lookup(self) -> None:
        self.assertEqual(len(UNIFORM_LBP_LOOKUP), 256)
        # Unique bin IDs should be 0..58 (59 bins)
        self.assertEqual(len(set(UNIFORM_LBP_LOOKUP)), 59)

    def test_face_template_serialization(self) -> None:
        template = FaceTemplate(
            subject_id="test-operator",
            vector=(0.1, 0.2, 0.7),
            extractor="lbph-test",
            metadata={"site": "toll-plaza-01"},
        )
        data = template.to_dict()
        restored = FaceTemplate.from_dict(data)

        self.assertEqual(restored.subject_id, "test-operator")
        self.assertEqual(restored.vector, (0.1, 0.2, 0.7))
        self.assertEqual(restored.extractor, "lbph-test")
        self.assertEqual(restored.metadata["site"], "toll-plaza-01")

    def test_create_template(self) -> None:
        face = synthetic_face(size=96)
        template = self.recognizer.create_template("op-1", face)
        self.assertIsInstance(template, FaceTemplate)
        self.assertEqual(template.subject_id, "op-1")
        self.assertGreater(len(template.vector), 0)

    def test_distance_and_similarity(self) -> None:
        sample_vec = self.recognizer.extract(synthetic_face(size=96))
        dim = len(sample_vec)
        vec_a = tuple(0.2 for _ in range(dim))
        vec_b = tuple(0.2 for _ in range(dim))
        dist = self.recognizer.distance(vec_a, vec_b)
        self.assertAlmostEqual(dist, 0.0)
        self.assertAlmostEqual(self.recognizer.similarity(dist), 1.0)

    def test_enroll_and_identify(self) -> None:
        face_alice = synthetic_face(eye_gap=30, mouth_curve=0, noise=2)
        face_bob = synthetic_face(eye_gap=20, mouth_curve=8, noise=2)

        tmpl_alice = self.recognizer.create_template("alice", face_alice)
        tmpl_bob = self.recognizer.create_template("bob", face_bob)

        # Probe with another sample of alice
        probe_alice = synthetic_face(eye_gap=30, mouth_curve=1, noise=3)
        result = self.recognizer.identify(probe_alice, [tmpl_alice, tmpl_bob])

        self.assertEqual(result.subject_id, "alice")
        self.assertGreater(result.similarity, 0.75)


if __name__ == "__main__":
    unittest.main()
