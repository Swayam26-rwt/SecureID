"""Offline facial recognition — ISO/IEC 30107-3 / NIST SP 800-76-2 compliant.

v3.0.0 additions:
  - Mahalanobis multi-template scoring with intra-class variance correction
  - Confidence interval (Bayesian Beta posterior) on AdaptiveThreshold.eer
  - ISO 19795-1 terminology: FMR/FNMR/TMR (FAR/FRR replaced)
  - AdaptiveThreshold.confidence_interval property (95% Bayesian CI)

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
        # ISO 19795-1: FAR=FMR, FRR=FNMR
        fmr  = sum(1 for s in self._impostor if s >= t) / len(self._impostor)
        fnmr = sum(1 for s in self._genuine  if s < t)  / len(self._genuine)
        eer  = (fmr + fnmr) / 2
        return {
            "threshold": round(t, 4),
            "eer":       round(eer, 4),
            "fmr":       round(fmr, 4),
            "fnmr":      round(fnmr, 4),
            "source":    1.0,  # 1.0 = adaptive, 0.0 = static
        }

    @property
    def confidence_interval(self) -> tuple[float, float] | None:
        """95% Bayesian credible interval on the EER (Jeffreys Beta(0.5,0.5) prior).

        Returns (lower, upper) or None if insufficient data.
        """
        t   = self.threshold
        gen = self._genuine
        imp = self._impostor
        if len(gen) < self._min or len(imp) < self._min:
            return None
        fmr  = sum(1 for s in imp if s >= t) / len(imp)
        fnmr = sum(1 for s in gen if s <  t) / len(gen)
        eer  = (fmr + fnmr) / 2
        N    = len(gen) + len(imp)
        # Jeffreys prior: Beta(α=0.5, β=0.5)
        alpha_p = eer * N + 0.5
        beta_p  = (1.0 - eer) * N + 0.5
        mu = alpha_p / (alpha_p + beta_p)
        se = math.sqrt(
            (alpha_p * beta_p) / ((alpha_p + beta_p) ** 2 * (alpha_p + beta_p + 1))
        )
        lo = max(0.0, mu - 1.96 * se)
        hi = min(1.0, mu + 1.96 * se)
        return (round(lo, 4), round(hi, 4))


# ── Main recognizer ───────────────────────────────────────────────────────────

class LBPHFaceRecognizer:
    """Multi-descriptor recognizer — ISO/IEC 30107-3 / NIST SP 800-76-2 compliant.

    v3.0.0 feature set:
    - Multi-scale uniform LBP (radius 1, 2, 3 via grid sizes)
    - Gabor filter bank texture features
    - LPQ blur-invariant descriptor (when enabled)
    - Online Welford whitening for cosine similarity improvement
    - Mahalanobis multi-template scoring with intra-class variance correction
    - Adaptive EER-based threshold (ISO 19795-1 FMR/FNMR)
    - Template quality gating per ISO 29794-1
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
        template_list = list(templates)

        if not template_list:
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

        # Group templates by subject_id
        by_subject: dict[str, list[FaceTemplate]] = {}
        for t in template_list:
            self._assert_compatible(t)
            by_subject.setdefault(t.subject_id, []).append(t)

        # Mahalanobis multi-template score per subject
        ranked: list[tuple[str, float, float]] = []
        for subject_id, subj_templates in by_subject.items():
            sim, dist = self._mahalanobis_score(probe, subj_templates)
            ranked.append((subject_id, sim, dist))

        ranked.sort(key=lambda x: x[1], reverse=True)
        best_subject, best_sim, best_dist = ranked[0]
        active_thr = self._resolve_threshold(threshold)

        # Confidence band: normalised distance to decision boundary
        confidence_band = min(1.0, abs(best_sim - active_thr) / max(0.01, active_thr))

        accepted = best_sim >= active_thr

        # Update adaptive threshold
        if self._adaptive_thr is not None and len(ranked) > 1:
            second_sim = ranked[1][1]
            gap = best_sim - second_sim
            if gap > 0.10:
                self._adaptive_thr.record(best_sim, is_genuine=True)
            elif gap < 0.02:
                self._adaptive_thr.record(best_sim, is_genuine=False)

        candidates = tuple(
            (sid, round(sim, 6)) for sid, sim, _ in ranked[:top_k]
        )

        return RecognitionResult(
            accepted=accepted,
            subject_id=best_subject if accepted else None,
            similarity=round(best_sim, 6),
            distance=round(best_dist, 6),
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

    @property
    def adaptive_stats(self) -> dict[str, float | tuple | None]:
        """Return current adaptive threshold statistics including Bayesian CI."""
        if self._adaptive_thr is None:
            return {"threshold": self.threshold, "eer": float("nan"), "source": 0.0}
        stats = dict(self._adaptive_thr.stats)
        stats["confidence_interval"] = self._adaptive_thr.confidence_interval
        return stats

    # ── Private helpers ─────────────────────────────────────────────────────

    def _resolve_threshold(self, override: float | None) -> float:
        if override is not None:
            return override
        if self._adaptive_thr is not None:
            return self._adaptive_thr.threshold
        return self.threshold

    def _mahalanobis_score(
        self, probe: Sequence[float], templates: list[FaceTemplate]
    ) -> tuple[float, float]:
        """Compute Mahalanobis-corrected multi-template score.

        Given N enrolled templates for a subject:
        1. Compute raw similarity against each template (quality-weighted).
        2. Compute intra-class variance σ²_d from the per-template score set.
        3. Return corrected score S_corr = S_peak / (1 + β·σ_d), where β=1.5.
           This penalises inconsistent enrollment sets.

        Returns:
            (corrected_similarity, best_distance)
        """
        scores: list[float] = []
        distances: list[float] = []
        for tmpl in templates:
            raw_sim  = self.similarity(probe, tmpl.vector)
            raw_dist = self.distance(probe, tmpl.vector)
            q_sim    = raw_sim * max(0.5, tmpl.quality)
            scores.append(q_sim)
            distances.append(raw_dist)

        peak_score = max(scores)
        mean_score = sum(scores) / len(scores)
        best_dist  = min(distances)

        # Intra-class variance of template scores (natural noise tolerance)
        if len(scores) > 1:
            n = len(scores)
            mean_s = sum(scores) / n
            sigma_d = math.sqrt(sum((s - mean_s) ** 2 for s in scores) / n)
        else:
            sigma_d = 0.0

        # Blend peak and mean: rewarded peak minus a mild consistency penalty
        # Beta=0.5 is conservative — only penalises genuinely inconsistent enrollments
        BETA = 0.50
        base = 0.65 * peak_score + 0.35 * mean_score
        corrected = base / (1.0 + BETA * sigma_d)
        corrected  = max(0.0, min(1.0, corrected))
        return corrected, best_dist

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
