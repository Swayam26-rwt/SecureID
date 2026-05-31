"""Offline facial recognition based on local binary pattern histograms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .image_ops import GrayImage, normalize_face, resize_gray, validate_gray_image

FeatureVector = tuple[float, ...]


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


@dataclass(frozen=True)
class FaceTemplate:
    subject_id: str
    vector: FeatureVector
    extractor: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "vector": list(self.vector),
            "extractor": self.extractor,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FaceTemplate":
        return cls(
            subject_id=str(payload["subject_id"]),
            vector=tuple(float(value) for value in payload["vector"]),  # type: ignore[index]
            extractor=str(payload["extractor"]),
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
    candidates: tuple[tuple[str, float], ...] = ()


class LBPHFaceRecognizer:
    """Compact recognizer for controlled offline deployments.

    The recognizer intentionally uses classical local texture features so it can
    run on edge devices without downloads, GPUs, or heavyweight dependencies.
    """

    def __init__(
        self,
        *,
        width: int = 96,
        height: int = 96,
        grid_x: int = 8,
        grid_y: int = 8,
        threshold: float = 0.78,
        appearance_size: int = 24,
    ) -> None:
        if width < 16 or height < 16:
            raise ValueError("width and height must be at least 16 pixels")
        if grid_x <= 0 or grid_y <= 0:
            raise ValueError("grid dimensions must be positive")

        self.width = width
        self.height = height
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.threshold = threshold
        self.appearance_size = appearance_size
        self._lbp_length = 59 * grid_x * grid_y
        self._appearance_length = appearance_size * appearance_size + 2 * appearance_size
        self._geometry_length = 15
        self.extractor_id = (
            f"lbph-u59-appearance-geometry-{width}x{height}-{grid_x}x{grid_y}"
        )

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
        return FaceTemplate(
            subject_id=subject_id,
            vector=self.extract(face_crop),
            extractor=self.extractor_id,
            metadata=metadata or {},
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

        for template in templates:
            self._assert_compatible(template)
            distance = self.distance(probe, template.vector)
            scored.append((template.subject_id, self.similarity(distance), distance))

        if not scored:
            return RecognitionResult(
                accepted=False,
                subject_id=None,
                similarity=0.0,
                distance=math.inf,
                threshold=threshold or self.threshold,
                extractor=self.extractor_id,
            )

        best_by_subject: dict[str, tuple[float, float]] = {}
        for subject_id, similarity, distance in scored:
            current = best_by_subject.get(subject_id)
            if current is None or similarity > current[0]:
                best_by_subject[subject_id] = (similarity, distance)

        ranked = sorted(best_by_subject.items(), key=lambda item: item[1][0], reverse=True)
        best_subject, (best_similarity, best_distance) = ranked[0]
        active_threshold = threshold if threshold is not None else self.threshold
        candidates = tuple(
            (subject_id, round(values[0], 6)) for subject_id, values in ranked[:top_k]
        )

        return RecognitionResult(
            accepted=best_similarity >= active_threshold,
            subject_id=best_subject if best_similarity >= active_threshold else None,
            similarity=best_similarity,
            distance=best_distance,
            threshold=active_threshold,
            extractor=self.extractor_id,
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
        result = self.identify(
            face_crop,
            templates,
            top_k=3,
            threshold=threshold,
        )

        return RecognitionResult(
            accepted=result.accepted and result.subject_id == subject_id,
            subject_id=subject_id if result.accepted else None,
            similarity=result.similarity,
            distance=result.distance,
            threshold=result.threshold,
            extractor=result.extractor,
            candidates=result.candidates,
        )

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
            left[appearance_end:],
            right[appearance_end:],
        )

        return (
            0.52 * lbp_distance
            + 0.95 * appearance_distance
            + 3.20 * geometry_distance
        )

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

    @staticmethod
    def similarity(distance: float) -> float:
        return 1.0 / (1.0 + distance)

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
        if template.extractor != self.extractor_id:
            raise ValueError(
                f"incompatible template extractor {template.extractor!r}; "
                f"expected {self.extractor_id!r}"
            )
