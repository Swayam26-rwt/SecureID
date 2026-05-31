"""Offline liveness checks from short bursts of normalized face frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .image_ops import (
    GrayImage,
    crop_fraction,
    entropy,
    laplacian_variance,
    mean,
    mean_absolute_difference,
    normalize_face,
    variance,
    weighted_centroid,
)


@dataclass(frozen=True)
class LivenessResult:
    passed: bool
    score: float
    threshold: float
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


class LivenessDetector:
    """Passive and challenge-response liveness detector.

    The detector works without landmarks or a neural anti-spoofing model. For
    best security, call it with a short randomized challenge such as ["blink"]
    or ["turn_left", "blink"] and 8-20 face crops captured over 1-3 seconds.
    """

    def __init__(
        self,
        *,
        width: int = 96,
        height: int = 96,
        threshold: float = 0.62,
    ) -> None:
        self.width = width
        self.height = height
        self.threshold = threshold

    def assess(
        self,
        frames: Sequence[GrayImage],
        *,
        challenge: Sequence[str] | None = None,
        threshold: float | None = None,
    ) -> LivenessResult:
        if len(frames) < 3:
            raise ValueError("liveness assessment requires at least 3 frames")

        normalized = [
            normalize_face(frame, width=self.width, height=self.height)
            for frame in frames
        ]
        active_threshold = threshold if threshold is not None else self.threshold
        challenge = tuple(challenge or ())

        metrics = self._metrics(normalized)
        challenge_scores = self._challenge_scores(normalized, challenge)
        metrics.update(challenge_scores)

        texture_score = self._range_score(metrics["texture"], low=35.0, high=360.0)
        entropy_score = self._range_score(metrics["entropy"], low=0.45, high=0.92)
        motion_score = self._window_score(
            metrics["motion"],
            too_low=0.004,
            ideal_low=0.018,
            ideal_high=0.16,
            too_high=0.38,
        )
        stability_score = self._window_score(
            metrics["exposure_variance"],
            too_low=0.00001,
            ideal_low=0.0003,
            ideal_high=0.035,
            too_high=0.12,
        )

        weighted_scores = [
            (texture_score, 0.24),
            (entropy_score, 0.16),
            (motion_score, 0.34),
            (stability_score, 0.12),
        ]

        for name in challenge:
            weighted_scores.append((metrics.get(f"challenge_{name}", 0.0), 0.30))

        score_total = sum(score * weight for score, weight in weighted_scores)
        weight_total = sum(weight for _, weight in weighted_scores) or 1.0
        score = max(0.0, min(1.0, score_total / weight_total))

        reasons = self._reasons(metrics, score, active_threshold, challenge)
        return LivenessResult(
            passed=score >= active_threshold and not reasons,
            score=score,
            threshold=active_threshold,
            metrics={key: round(value, 6) for key, value in metrics.items()},
            reasons=tuple(reasons),
        )

    def _metrics(self, frames: Sequence[Sequence[Sequence[int]]]) -> dict[str, float]:
        textures = [laplacian_variance(frame) for frame in frames]
        entropies = [entropy(frame) for frame in frames]
        frame_diffs = [
            mean_absolute_difference(frames[index - 1], frames[index])
            for index in range(1, len(frames))
        ]
        brightness = [
            mean(float(pixel) / 255.0 for row in frame for pixel in row)
            for frame in frames
        ]
        centroids = [weighted_centroid(frame) for frame in frames]
        x_shift = max(x for x, _ in centroids) - min(x for x, _ in centroids)
        y_shift = max(y for _, y in centroids) - min(y for _, y in centroids)

        return {
            "texture": mean(textures),
            "entropy": mean(entropies),
            "motion": mean(frame_diffs),
            "exposure_variance": variance(brightness),
            "x_shift": x_shift,
            "y_shift": y_shift,
            "blink": self._blink_score(frames),
        }

    def _challenge_scores(
        self,
        frames: Sequence[Sequence[Sequence[int]]],
        challenge: Sequence[str],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        centroids = [weighted_centroid(frame) for frame in frames]
        start_x, start_y = centroids[0]
        end_x, end_y = centroids[-1]

        for item in challenge:
            if item == "blink":
                metrics["challenge_blink"] = self._blink_score(frames)
            elif item == "turn_left":
                metrics["challenge_turn_left"] = self._direction_score(end_x - start_x)
            elif item == "turn_right":
                metrics["challenge_turn_right"] = self._direction_score(start_x - end_x)
            elif item == "nod":
                metrics["challenge_nod"] = self._direction_score(abs(end_y - start_y))
            else:
                metrics[f"challenge_{item}"] = 0.0
        return metrics

    def _blink_score(self, frames: Sequence[Sequence[Sequence[int]]]) -> float:
        openness = [self._eye_openness(frame) for frame in frames]
        if len(openness) < 3:
            return 0.0

        peak = max(openness)
        valley = min(openness)
        amplitude = peak - valley
        valley_index = openness.index(valley)
        has_recovery = (
            valley_index > 0
            and valley_index < len(openness) - 1
            and max(openness[:valley_index]) - valley > 0.012
            and max(openness[valley_index + 1 :]) - valley > 0.012
        )
        if not has_recovery:
            return 0.0
        return max(0.0, min(1.0, amplitude / 0.035))

    def _eye_openness(self, frame: Sequence[Sequence[int]]) -> float:
        left_eye = crop_fraction(frame, x0=0.22, y0=0.27, x1=0.45, y1=0.45)
        right_eye = crop_fraction(frame, x0=0.55, y0=0.27, x1=0.78, y1=0.45)
        eye_pixels = [float(pixel) / 255.0 for crop in (left_eye, right_eye) for row in crop for pixel in row]
        dark_ratio = sum(pixel < 0.32 for pixel in eye_pixels) / len(eye_pixels)
        return dark_ratio + 0.35 * variance(eye_pixels)

    @staticmethod
    def _range_score(value: float, *, low: float, high: float) -> float:
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)

    @staticmethod
    def _window_score(
        value: float,
        *,
        too_low: float,
        ideal_low: float,
        ideal_high: float,
        too_high: float,
    ) -> float:
        if value <= too_low or value >= too_high:
            return 0.0
        if ideal_low <= value <= ideal_high:
            return 1.0
        if value < ideal_low:
            return (value - too_low) / (ideal_low - too_low)
        return (too_high - value) / (too_high - ideal_high)

    @staticmethod
    def _direction_score(delta: float) -> float:
        if delta <= 0.015:
            return 0.0
        if delta >= 0.075:
            return 1.0
        return (delta - 0.015) / 0.06

    @staticmethod
    def _reasons(
        metrics: dict[str, float],
        score: float,
        threshold: float,
        challenge: Sequence[str],
    ) -> list[str]:
        reasons: list[str] = []
        if metrics["motion"] < 0.004:
            reasons.append("insufficient frame-to-frame motion")
        if metrics["motion"] > 0.38:
            reasons.append("excessive motion or unstable face crop")
        if metrics["texture"] < 35.0:
            reasons.append("insufficient facial texture detail")
        for item in challenge:
            if metrics.get(f"challenge_{item}", 0.0) < 0.55:
                reasons.append(f"challenge not satisfied: {item}")
        if score < threshold:
            reasons.append("liveness score below threshold")
        return reasons
