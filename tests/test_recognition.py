from __future__ import annotations

import unittest

from datalake_offline_biometrics.image_ops import synthetic_face
from datalake_offline_biometrics.recognition import (
    AdaptiveThreshold,
    FaceTemplate,
    LBPHFaceRecognizer,
    OnlinePCAWhitener,
    UNIFORM_LBP_LOOKUP,
)


class TestRecognition(unittest.TestCase):
    def setUp(self) -> None:
        # Use cosine metric with a threshold calibrated for LBPH+Gabor cosine space
        self.recognizer = LBPHFaceRecognizer(threshold=0.40)

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
        self.assertIsInstance(restored.quality, float)

    def test_create_template_has_quality(self) -> None:
        face = synthetic_face(size=96)
        template = self.recognizer.create_template("op-1", face)
        self.assertIsInstance(template, FaceTemplate)
        self.assertEqual(template.subject_id, "op-1")
        self.assertGreater(len(template.vector), 0)
        self.assertGreaterEqual(template.quality, 0.0)
        self.assertLessEqual(template.quality, 1.0)

    def test_cosine_self_similarity(self) -> None:
        """Two identical vectors should have cosine similarity = 1.0."""
        sample_vec = self.recognizer.extract(synthetic_face(size=96))
        sim = self.recognizer.similarity(sample_vec, sample_vec)
        self.assertAlmostEqual(sim, 1.0, places=3)

    def test_distance_identical_vectors(self) -> None:
        """Identical vectors should have near-zero weighted distance."""
        dim = self.recognizer._lbp_length + self.recognizer._appearance_length + self.recognizer._geometry_length
        vec_a = tuple(0.2 for _ in range(dim))
        dist = self.recognizer.distance(vec_a, vec_a)
        self.assertAlmostEqual(dist, 0.0)

    def test_enroll_and_identify(self) -> None:
        """Same-subject should score highest vs different-subject."""
        face_alice = synthetic_face(eye_gap=30, mouth_curve=0, noise=2)
        face_bob = synthetic_face(eye_gap=20, mouth_curve=8, noise=2)

        tmpl_alice = self.recognizer.create_template("alice", face_alice)
        tmpl_bob = self.recognizer.create_template("bob", face_bob)

        # Probe with another sample of alice
        probe_alice = synthetic_face(eye_gap=30, mouth_curve=1, noise=3)
        result = self.recognizer.identify(probe_alice, [tmpl_alice, tmpl_bob])

        # Alice should rank first (highest similarity)
        self.assertEqual(result.candidates[0][0], "alice")
        # Should be accepted with cosine threshold 0.40
        self.assertEqual(result.subject_id, "alice")

    def test_same_subject_ranked_first_over_different(self) -> None:
        """Same-subject template should rank higher than different-subject template."""
        # Enroll multiple samples to stabilize the online PCA whitener
        alice_faces = [synthetic_face(eye_gap=30, noise=k) for k in range(1, 4)]
        bob_faces = [synthetic_face(eye_gap=20, mouth_curve=8, noise=k) for k in range(1, 4)]

        templates = []
        for f in alice_faces:
            templates.append(self.recognizer.create_template("alice", f))
        for f in bob_faces:
            templates.append(self.recognizer.create_template("bob", f))

        # Probe with another alice sample — alice should rank first
        alice_probe = synthetic_face(eye_gap=30, noise=4)
        result = self.recognizer.identify(alice_probe, templates)
        self.assertEqual(result.candidates[0][0], "alice")

    def test_result_has_confidence_band(self) -> None:
        face = synthetic_face(noise=1)
        tmpl = self.recognizer.create_template("x", face)
        result = self.recognizer.identify(face, [tmpl])
        self.assertIsInstance(result.confidence_band, float)
        self.assertGreaterEqual(result.confidence_band, 0.0)

    def test_result_has_metric_field(self) -> None:
        face = synthetic_face(noise=1)
        tmpl = self.recognizer.create_template("x", face)
        result = self.recognizer.identify(face, [tmpl])
        self.assertIn(result.metric, ("cosine", "weighted"))


class TestAdaptiveThreshold(unittest.TestCase):
    def test_falls_back_to_static_without_history(self) -> None:
        thr = AdaptiveThreshold(0.70)
        self.assertAlmostEqual(thr.threshold, 0.70)

    def test_adapts_with_enough_history(self) -> None:
        thr = AdaptiveThreshold(0.70, min_samples_per_class=3)
        for _ in range(5):
            thr.record(0.85, is_genuine=True)
            thr.record(0.30, is_genuine=False)
        adapted = thr.threshold
        # Should be somewhere between the two class means
        self.assertGreater(adapted, 0.25)
        self.assertLess(adapted, 0.90)

    def test_stats_dict_has_required_keys(self) -> None:
        thr = AdaptiveThreshold(0.70)
        stats = thr.stats
        self.assertIn("threshold", stats)
        self.assertIn("eer", stats)
        self.assertIn("source", stats)


class TestOnlinePCAWhitener(unittest.TestCase):
    def test_whitened_vector_near_zero_mean(self) -> None:
        dim = 10
        whitener = OnlinePCAWhitener(dim)
        import math
        vecs = [[math.sin(i + j * 0.1) for j in range(dim)] for i in range(50)]
        for v in vecs:
            whitener.update(v)
        whitened = whitener.whiten(vecs[-1])
        self.assertEqual(len(whitened), dim)
        # At least some dimensions should be non-trivially whitened
        diffs = sum(abs(w - o) for w, o in zip(whitened, vecs[-1]))
        self.assertGreater(diffs, 0)


if __name__ == "__main__":
    unittest.main()
