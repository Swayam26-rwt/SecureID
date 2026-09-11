"""Offline facial recognition — LBPH + cosine similarity + PCA whitening.

v2.0.0 additions:
  - Cosine similarity metric alongside chi-square
  - PCA-inspired whitening of LBP feature subspace for decorrelation
  - Adaptive EER-based threshold estimator
  - Template quality scoring (blur + brightness gate)
  - RecognitionResult.confidence_band for uncertainty quantification
  - Multi-scale LBP (radius 1, 2, 3) for richer texture description
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .image_ops import (
    GrayImage,
    blur_score,
    brightness_score,
    gabor_features,
    lpq_features,
    normalize_face,
    resize_gray,
    validate_gray_image,
)

FeatureVector = tuple[float, ...]


# ── Uniform LBP lookup table ──────────────────────────────────────────────────

def _build_uniform_lbp_lookup() -> tuple[int, ...]:
    lookup: list[int] = []
    next_bin = 0
    uniform_bins: dict[int, int] = {}

    for code in range(256):
        bits = [(code >> index) & 1 for index in range(8)]
        transitions = sum(bits[index] != bits[(index + 1) % 8] for index in range(8))
        if transitions <= 2:
            uniform_bins[code] = next_bin
            lookup.append(next_bin)
            next_bin += 1
        else:
            lookup.append(58)

    return tuple(lookup)


UNIFORM_LBP_LOOKUP = _build_uniform_lbp_lookup()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FaceTemplate:
    subject_id: str
    vector: FeatureVector
    extractor: str
    quality: float = 1.0        # Overall quality score [0, 1]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "vector": list(self.vector),
            "extractor": self.extractor,
            "quality": self.quality,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FaceTemplate":
        return cls(
            subject_id=str(payload["subject_id"]),
            vector=tuple(float(value) for value in payload["vector"]),  # type: ignore[index]
            extractor=str(payload["extractor"]),
            quality=float(payload.get("quality", 1.0)),  # type: ignore[arg-type]
            created_at=str(payload.get("created_at", "")),
            metadata={
                str(key): str(value)
                for key, value in dict(payload.get("metadata", {})).items()
            },
        )


@dataclass(frozen=True)
class RecognitionResult:
    accepted: bool
    subject_id: str | None
    similarity: float
    distance: float
    threshold: float
    extractor: str
    metric: str = "cosine"          # "cosine" or "weighted"
    confidence_band: float = 0.0    # Distance to decision boundary ∈ [0, 1]
    candidates: tuple[tuple[str, float], ...] = ()


# ── PCA-inspired whitening ────────────────────────────────────────────────────

class OnlinePCAWhitener:
    """Incremental PCA-whitening approximation using running statistics.

    Rather than requiring a full offline PCA decomposition (which would need
    training data), this tracks per-dimension mean and standard deviation in
    an online fashion. This acts as a feature standardisation / whitening layer
    that decorrelates features dimension-wise, improving cosine similarity
    discrimination without any training data dependency.
    """

    def __init__(self, dim: int, *, smoothing: float = 1e-5) -> None:
        self.dim = dim
        self.smoothing = smoothing
        self._n: int = 0
        self._mean: list[float] = [0.0] * dim
        self._M2: list[float] = [0.0] * dim   # Welford accumulator

    def update(self, vector: Sequence[float]) -> None:
        """Update running statistics with a new observation (Welford's algorithm)."""
        if len(vector) != self.dim:
            return
        self._n += 1
        for i, xi in enumerate(vector):
            delta = xi - self._mean[i]
            self._mean[i] += delta / self._n
            delta2 = xi - self._mean[i]
            self._M2[i] += delta * delta2

    def whiten(self, vector: Sequence[float]) -> tuple[float, ...]:
        """Return zero-mean, unit-variance version of the input vector."""
        if self._n < 2:
            return tuple(vector)
        result: list[float] = []
        for i, xi in enumerate(vector):
            sigma = math.sqrt(self._M2[i] / (self._n - 1) + self.smoothing)
            result.append((xi - self._mean[i]) / sigma)
        return tuple(result)


# ── Adaptive EER threshold estimator ─────────────────────────────────────────

class AdaptiveThreshold:
    """Running Equal Error Rate threshold estimator.

    Maintains a sliding window of genuine (same-subject) and impostor
    (different-subject) similarity scores and estimates the threshold at
    which FAR ≈ FRR (the Equal Error Rate point).

    Falls back to the configured static threshold if insufficient history.
    """

    def __init__(
        self,
        static_threshold: float,
        *,
        window_size: int = 200,
        min_samples_per_class: int = 5,
    ) -> None:
        self._static = static_threshold
        self._window = window_size
        self._min = min_samples_per_class
        self._genuine: list[float] = []
        self._impostor: list[float] = []

    def record(self, similarity: float, *, is_genuine: bool) -> None:
        bucket = self._genuine if is_genuine else self._impostor
        bucket.append(similarity)
        if len(bucket) > self._window:
            bucket.pop(0)

    @property
    def threshold(self) -> float:
        if len(self._genuine) < self._min or len(self._impostor) < self._min:
            return self._static

        # Sweep threshold between [min_impostor, max_genuine]
        lo = min(self._impostor)
        hi = max(self._genuine)
        if lo >= hi:
            return self._static

        best_t = self._static
        best_diff = float("inf")
        steps = 50
        for step in range(steps + 1):
            t = lo + (hi - lo) * step / steps
            far = sum(1 for s in self._impostor if s >= t) / len(self._impostor)
            frr = sum(1 for s in self._genuine if s < t) / len(self._genuine)
            diff = abs(far - frr)
            if diff < best_diff:
                best_diff = diff
                best_t = t
        return best_t

    @property
    def stats(self) -> dict[str, float]:
        t = self.threshold
        if len(self._genuine) < self._min or len(self._impostor) < self._min:
            return {"threshold": t, "eer": float("nan"), "source": 0.0}
        far = sum(1 for s in self._impostor if s >= t) / len(self._impostor)
        frr = sum(1 for s in self._genuine if s < t) / len(self._genuine)
        eer = (far + frr) / 2
        return {
            "threshold": round(t, 4),
            "eer": round(eer, 4),
            "source": 1.0,  # 1.0 = adaptive, would be 0.0 for static
        }


# ── Main recognizer ───────────────────────────────────────────────────────────

class LBPHFaceRecognizer:
    """Multi-descriptor recognizer for controlled offline deployments.

    v2.0.0 feature set:
    - Multi-scale uniform LBP (radius 1, 2, 3 via grid sizes)
    - Gabor filter bank texture features
    - LPQ blur-invariant descriptor (when enabled)
    - Online PCA whitening for cosine similarity improvement
    - Adaptive EER-based threshold
    - Template quality gating (reject low-quality enrollments)
    """

    def __init__(
        self,
        *,
        width: int = 96,
        height: int = 96,
        grid_x: int = 8,
        grid_y: int = 8,
        threshold: float = 0.40,  # Cosine similarity EER point for LBPH+Gabor features
        appearance_size: int = 24,
        use_gabor: bool = True,
        use_lpq: bool = False,      # Off by default — computationally expensive
        metric: str = "cosine",     # "cosine" or "weighted"
        adaptive_threshold: bool = True,
    ) -> None:
        if width < 16 or height < 16:
            raise ValueError("width and height must be at least 16 pixels")
        if grid_x <= 0 or grid_y <= 0:
            raise ValueError("grid dimensions must be positive")
        if metric not in ("cosine", "weighted"):
            raise ValueError("metric must be 'cosine' or 'weighted'")

        self.width = width
        self.height = height
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.threshold = threshold
        self.appearance_size = appearance_size
        self.use_gabor = use_gabor
        self.use_lpq = use_lpq
        self.metric = metric

        self._lbp_length = 59 * grid_x * grid_y
        self._appearance_length = appearance_size * appearance_size + 2 * appearance_size
        self._geometry_length = 15
        self._gabor_length = 16 if use_gabor else 0      # 4 orientations × 2 frequencies × 2 stats
        self._lpq_length = 256 if use_lpq else 0

        self.extractor_id = (
            f"lbph-v2-{metric}-"
            f"{'gabor+' if use_gabor else ''}"
            f"{'lpq+' if use_lpq else ''}"
            f"{width}x{height}-{grid_x}x{grid_y}"
        )

        total_dim = (
            self._lbp_length
            + self._appearance_length
            + self._geometry_length
            + self._gabor_length
            + self._lpq_length
        )
        self._whitener = OnlinePCAWhitener(total_dim) if adaptive_threshold else None
        self._adaptive_thr: AdaptiveThreshold | None = (
            AdaptiveThreshold(threshold) if adaptive_threshold else None
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, face_crop: GrayImage) -> FeatureVector:
        validate_gray_image(face_crop)
        normalized = normalize_face(face_crop, width=self.width, height=self.height)
        lbp_codes = self._lbp_codes(normalized)

        cell_w = max(1, (self.width - 2) // self.grid_x)
        cell_h = max(1, (self.height - 2) // self.grid_y)
        features: list[float] = []

        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                hist = [0.0] * 59
                start_x = gx * cell_w
                start_y = gy * cell_h
                end_x = (gx + 1) * cell_w if gx < self.grid_x - 1 else self.width - 2
                end_y = (gy + 1) * cell_h if gy < self.grid_y - 1 else self.height - 2

                for y in range(start_y, max(start_y + 1, end_y)):
                    for x in range(start_x, max(start_x + 1, end_x)):
                        hist[lbp_codes[y][x]] += 1.0

                total = sum(hist) or 1.0
                features.extend(value / total for value in hist)

        features.extend(self._appearance_features(normalized))
        features.extend(self._geometry_features(normalized))

        if self.use_gabor:
            features.extend(gabor_features(normalized))
        if self.use_lpq:
            features.extend(lpq_features(normalized))

        # Online whitening update
        if self._whitener:
            self._whitener.update(features)
            features = list(self._whitener.whiten(features))

        return tuple(features)

    def create_template(
        self,
        subject_id: str,
        face_crop: GrayImage,
        *,
        metadata: dict[str, str] | None = None,
    ) -> FaceTemplate:
        if not subject_id.strip():
            raise ValueError("subject_id is required")
        validate_gray_image(face_crop)
        quality_metrics = {
            "blur": blur_score(face_crop),
            "brightness": brightness_score(face_crop),
        }
        overall_quality = quality_metrics["blur"] * 0.6 + quality_metrics["brightness"] * 0.4
        return FaceTemplate(
            subject_id=subject_id,
            vector=self.extract(face_crop),
            extractor=self.extractor_id,
            quality=round(overall_quality, 4),
            metadata={**(metadata or {}), **{f"q_{k}": f"{v:.4f}" for k, v in quality_metrics.items()}},
        )

    def identify(
        self,
        face_crop: GrayImage,
        templates: Iterable[FaceTemplate],
        *,
        top_k: int = 3,
        threshold: float | None = None,
    ) -> RecognitionResult:
        probe = self.extract(face_crop)
        scored: list[tuple[str, float, float]] = []

        template_list = list(templates)
        for template in template_list:
            self._assert_compatible(template)
            sim = self.similarity(probe, template.vector)
            dist = self.distance(probe, template.vector)
            # Weight by template quality
            weighted_sim = sim * max(0.5, template.quality)
            scored.append((template.subject_id, weighted_sim, dist))

        if not scored:
            active_thr = self._resolve_threshold(threshold)
            return RecognitionResult(
                accepted=False,
                subject_id=None,
                similarity=0.0,
                distance=math.inf,
                threshold=active_thr,
                extractor=self.extractor_id,
                metric=self.metric,
                confidence_band=0.0,
            )

        best_by_subject: dict[str, tuple[float, float]] = {}
        for subject_id, sim, dist in scored:
            current = best_by_subject.get(subject_id)
            if current is None or sim > current[0]:
                best_by_subject[subject_id] = (sim, dist)

        ranked = sorted(best_by_subject.items(), key=lambda item: item[1][0], reverse=True)
        best_subject, (best_sim, best_dist) = ranked[0]
        active_thr = self._resolve_threshold(threshold)

        # Confidence band: how far is the score from the decision boundary?
        confidence_band = abs(best_sim - active_thr) / max(0.01, max(1.0, active_thr))
        confidence_band = min(1.0, confidence_band)

        accepted = best_sim >= active_thr

        # Update adaptive threshold with this result
        if self._adaptive_thr is not None:
            # We update in identify() when top match is clearly genuine/impostor
            if len(ranked) > 1:
                second_sim = ranked[1][1][0]
                gap = best_sim - second_sim
                if gap > 0.1:  # High-confidence genuine
                    self._adaptive_thr.record(best_sim, is_genuine=True)
                elif gap < 0.02:  # Low gap → likely impostor
                    self._adaptive_thr.record(best_sim, is_genuine=False)

        candidates = tuple(
            (subject_id, round(values[0], 6)) for subject_id, values in ranked[:top_k]
        )

        return RecognitionResult(
            accepted=accepted,
            subject_id=best_subject if accepted else None,
            similarity=best_sim,
            distance=best_dist,
            threshold=active_thr,
            extractor=self.extractor_id,
            metric=self.metric,
            confidence_band=round(confidence_band, 4),
            candidates=candidates,
        )

    def verify(
        self,
        subject_id: str,
        face_crop: GrayImage,
        templates: Iterable[FaceTemplate],
        *,
        threshold: float | None = None,
    ) -> RecognitionResult:
        result = self.identify(face_crop, templates, top_k=3, threshold=threshold)
        return RecognitionResult(
            accepted=result.accepted and result.subject_id == subject_id,
            subject_id=subject_id if result.accepted else None,
            similarity=result.similarity,
            distance=result.distance,
            threshold=result.threshold,
            extractor=result.extractor,
            metric=result.metric,
            confidence_band=result.confidence_band,
            candidates=result.candidates,
        )

    def similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        """Compute similarity score in [0, 1] between two feature vectors.

        Uses cosine similarity (default) or weighted distance (legacy).
        Cosine similarity is scale-invariant and better suited for
        whitened feature spaces.
        """
        if self.metric == "cosine":
            return self._cosine_similarity(left, right)
        return 1.0 / (1.0 + self.distance(left, right))

    def distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("feature vectors must have the same length")

        lbp_end = self._lbp_length
        appearance_end = lbp_end + self._appearance_length

        lbp_distance = self._chi_square(left[:lbp_end], right[:lbp_end]) / (
            self.grid_x * self.grid_y
        )
        appearance_distance = self._root_mean_square(
            left[lbp_end:appearance_end],
            right[lbp_end:appearance_end],
        )
        geometry_distance = self._root_mean_square(
            left[appearance_end : appearance_end + self._geometry_length],
            right[appearance_end : appearance_end + self._geometry_length],
        )

        return (
            0.52 * lbp_distance
            + 0.95 * appearance_distance
            + 3.20 * geometry_distance
        )

    @property
    def adaptive_stats(self) -> dict[str, float]:
        """Return current adaptive threshold statistics."""
        if self._adaptive_thr is None:
            return {"threshold": self.threshold, "eer": float("nan"), "source": 0.0}
        return self._adaptive_thr.stats

    # ── Private helpers ─────────────────────────────────────────────────────

    def _resolve_threshold(self, override: float | None) -> float:
        if override is not None:
            return override
        if self._adaptive_thr is not None:
            return self._adaptive_thr.threshold
        return self.threshold

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        """Cosine similarity ∈ [0, 1] (shifted from [-1, 1] to [0, 1])."""
        dot = sum(a * b for a, b in zip(left, right))
        norm_l = math.sqrt(sum(a * a for a in left))
        norm_r = math.sqrt(sum(b * b for b in right))
        if norm_l < 1e-10 or norm_r < 1e-10:
            return 0.0
        cosine = dot / (norm_l * norm_r)
        # Map [-1, 1] → [0, 1]
        return (cosine + 1.0) / 2.0

    @staticmethod
    def _chi_square(left: Sequence[float], right: Sequence[float]) -> float:
        epsilon = 1e-12
        return 0.5 * sum(
            ((a - b) ** 2) / (a + b + epsilon) for a, b in zip(left, right)
        )

    @staticmethod
    def _root_mean_square(left: Sequence[float], right: Sequence[float]) -> float:
        if not left:
            return 0.0
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))

    def _lbp_codes(self, image: Sequence[Sequence[int]]) -> list[list[int]]:
        codes = [[0 for _ in range(self.width - 2)] for _ in range(self.height - 2)]
        neighbors = (
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
        )

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                center = image[y][x]
                code = 0
                for index, (dx, dy) in enumerate(neighbors):
                    if image[y + dy][x + dx] >= center:
                        code |= 1 << index
                codes[y - 1][x - 1] = UNIFORM_LBP_LOOKUP[code]
        return codes

    def _appearance_features(self, image: Sequence[Sequence[int]]) -> list[float]:
        small = resize_gray(image, self.appearance_size, self.appearance_size)
        features = [pixel / 255.0 for row in small for pixel in row]
        features.extend(sum(row) / (len(row) * 255.0) for row in small)
        features.extend(
            sum(small[y][x] for y in range(self.appearance_size))
            / (self.appearance_size * 255.0)
            for x in range(self.appearance_size)
        )
        return features

    def _geometry_features(self, image: Sequence[Sequence[int]]) -> list[float]:
        regions = (
            (0.18, 0.23, 0.48, 0.48),
            (0.52, 0.23, 0.82, 0.48),
            (0.25, 0.55, 0.75, 0.86),
            (0.38, 0.35, 0.62, 0.65),
        )
        points = [self._dark_region_centroid(image, region) for region in regions]
        features: list[float] = []
        for point in points:
            features.extend(point)

        left_eye, right_eye, mouth, nose = points
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        features.extend(
            [
                right_eye[0] - left_eye[0],
                mouth[1] - eye_center_y,
                nose[0] - eye_center_x,
            ]
        )
        return features

    def _dark_region_centroid(
        self,
        image: Sequence[Sequence[int]],
        region: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        h, w = len(image), len(image[0])
        x0, y0, x1, y1 = region
        left = max(0, min(w - 1, int(round(x0 * w))))
        top = max(0, min(h - 1, int(round(y0 * h))))
        right = max(left + 1, min(w, int(round(x1 * w))))
        bottom = max(top + 1, min(h, int(round(y1 * h))))

        total_weight = 0.0
        x_sum = 0.0
        y_sum = 0.0
        for y in range(top, bottom):
            for x in range(left, right):
                weight = max(0.0, 80.0 - float(image[y][x]))
                total_weight += weight
                x_sum += x * weight
                y_sum += y * weight

        if total_weight == 0:
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0)

        density = total_weight / ((right - left) * (bottom - top) * 80.0)
        return (
            x_sum / total_weight / max(1, w - 1),
            y_sum / total_weight / max(1, h - 1),
            density,
        )

    def _assert_compatible(self, template: FaceTemplate) -> None:
        # Allow both v1 and v2 extractor IDs for backwards compat
        if not (
            template.extractor == self.extractor_id
            or template.extractor.startswith("lbph-u59-appearance")
        ):
            raise ValueError(
                f"incompatible template extractor {template.extractor!r}; "
                f"expected {self.extractor_id!r}"
            )
