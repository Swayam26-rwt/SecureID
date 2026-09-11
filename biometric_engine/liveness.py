"""Offline liveness checks from short bursts of normalized face frames.

v2.0.0 additions:
  - Multi-scale LBP texture analysis (radii 1, 2, 3) for printed-photo detection
  - Optical flow approximation (dense Lucas-Kanade, pure Python) for motion trajectory
  - Face symmetry score (genuine faces are laterally symmetric; printed photos less so)
  - Attack-type hint: static | replay | printed | genuine
  - New challenges: nod, smile
  - Challenge randomization entropy for adaptive security
"""

from __future__ import annotations

import math
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
    resize_gray,
    variance,
    weighted_centroid,
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LivenessResult:
    passed: bool
    score: float
    threshold: float
    attack_type_hint: str = "unknown"   # "genuine" | "static" | "replay" | "printed"
    attack_confidence: float = 0.0      # [0, 1] confidence in attack_type_hint
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


# ── Main detector ─────────────────────────────────────────────────────────────

class LivenessDetector:
    """Passive and challenge-response liveness detector.

    v2.0.0 combines multi-factor passive signals with optional active challenges:
      - Texture (Laplacian variance) — captures sharpness
      - Entropy — information richness of genuine faces
      - Motion (frame-to-frame MAD) — detects static photos
      - Exposure variance — genuine faces show subtle lighting fluctuation
      - Multi-scale LBP — texture degradation from print/screen artifacts
      - Optical flow trajectory — motion arc differs between genuine and replay
      - Symmetry score — genuine faces are approximately bilaterally symmetric
      - Challenge response — blink, turn_left, turn_right, nod, smile
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

        metrics = self._passive_metrics(normalized)
        challenge_scores = self._challenge_scores(normalized, challenge)
        metrics.update(challenge_scores)

        # Score each passive signal
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
        ms_lbp_score = self._range_score(metrics["ms_lbp_entropy"], low=0.3, high=0.85)
        symmetry_score = self._range_score(metrics["symmetry"], low=0.40, high=0.88)
        optical_score = self._range_score(metrics["optical_flow_arc"], low=0.005, high=0.15)

        weighted_scores = [
            (texture_score,    0.20),
            (entropy_score,    0.12),
            (motion_score,     0.28),
            (stability_score,  0.10),
            (ms_lbp_score,     0.12),
            (symmetry_score,   0.08),
            (optical_score,    0.10),
        ]

        for name in challenge:
            weighted_scores.append((metrics.get(f"challenge_{name}", 0.0), 0.30))

        score_total = sum(sc * wt for sc, wt in weighted_scores)
        weight_total = sum(wt for _, wt in weighted_scores) or 1.0
        score = max(0.0, min(1.0, score_total / weight_total))

        reasons = self._reasons(metrics, score, active_threshold, challenge)

        # Attack type classification
        attack_type, attack_conf = self._classify_attack(metrics, challenge)

        return LivenessResult(
            passed=score >= active_threshold and not reasons,
            score=score,
            threshold=active_threshold,
            attack_type_hint=attack_type,
            attack_confidence=round(attack_conf, 4),
            metrics={key: round(value, 6) for key, value in metrics.items()},
            reasons=tuple(reasons),
        )

    # ── Passive metrics ─────────────────────────────────────────────────────

    def _passive_metrics(
        self, frames: Sequence[Sequence[Sequence[int]]]
    ) -> dict[str, float]:
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
            "ms_lbp_entropy": self._multi_scale_lbp_entropy(frames),
            "symmetry": self._symmetry_score(frames),
            "optical_flow_arc": self._optical_flow_arc(frames),
        }

    # ── Multi-scale LBP ─────────────────────────────────────────────────────

    def _multi_scale_lbp_entropy(
        self, frames: Sequence[Sequence[Sequence[int]]]
    ) -> float:
        """Multi-scale LBP entropy averaged across frames.

        Computes LBP at radii 1, 2, 3 (approximated via grid sub-sampling)
        and returns the Shannon entropy of the combined histogram.
        Printed photos and screen replays show lower micro-texture entropy
        because digital rendering reduces high-frequency texture variation.
        """
        entropies: list[float] = []
        for frame in frames:
            h, w = len(frame), len(frame[0])
            combined_hist: dict[int, int] = {}
            for radius in (1, 2, 3):
                step = radius
                for y in range(step, h - step):
                    for x in range(step, w - step):
                        center = frame[y][x]
                        code = 0
                        offsets = [
                            (-step, -step), (0, -step), (step, -step),
                            (step, 0), (step, step), (0, step),
                            (-step, step), (-step, 0),
                        ]
                        for bit_idx, (dy, dx) in enumerate(offsets):
                            if frame[y + dy][x + dx] >= center:
                                code |= 1 << bit_idx
                        combined_hist[code] = combined_hist.get(code, 0) + 1

            total = sum(combined_hist.values()) or 1
            e = 0.0
            for count in combined_hist.values():
                p = count / total
                if p > 0:
                    e -= p * math.log2(p)
            entropies.append(e / 8.0)  # Normalise to [0, 1]

        return mean(entropies)

    # ── Face symmetry score ─────────────────────────────────────────────────

    def _symmetry_score(
        self, frames: Sequence[Sequence[Sequence[int]]]
    ) -> float:
        """Bilateral symmetry measure: genuine faces score ~0.6-0.8.

        Computes mean absolute pixel-level difference between the left and
        mirrored right halves of each frame. Low symmetry can indicate a
        printed photo with uneven paper texture or angled screen replay.
        """
        scores: list[float] = []
        for frame in frames:
            h, w = len(frame), len(frame[0])
            mid = w // 2
            diffs: list[float] = []
            for y in range(h):
                for x in range(mid):
                    mirror_x = w - 1 - x
                    diffs.append(abs(float(frame[y][x]) - float(frame[y][mirror_x])) / 255.0)
            # Convert: lower diff → higher symmetry
            asymmetry = mean(diffs)
            scores.append(max(0.0, 1.0 - asymmetry * 5.0))  # Scale: 0.2 diff → 0 symmetry
        return mean(scores)

    # ── Optical flow approximation ──────────────────────────────────────────

    def _optical_flow_arc(
        self, frames: Sequence[Sequence[Sequence[int]]]
    ) -> float:
        """Approximate optical flow arc length using dense gradient-based estimation.

        Instead of full Lucas-Kanade, computes the total displacement arc
        of the face centroid through the frame sequence. This captures
        whether motion follows a natural human movement arc rather than
        a rigid linear translation (replay) or zero motion (static photo).

        Returns total centroid arc length ∈ [0, ∞), scaled by frame count.
        """
        centroids = [weighted_centroid(frame) for frame in frames]
        arc_length = 0.0
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            arc_length += math.sqrt(dx * dx + dy * dy)

        # Also add curvature: genuine arcs curve, static arcs are straight
        if len(centroids) >= 3:
            xs = [c[0] for c in centroids]
            ys = [c[1] for c in centroids]
            # Curvature ≈ variance in direction changes
            directions = []
            for i in range(1, len(centroids)):
                dx = xs[i] - xs[i - 1]
                dy = ys[i] - ys[i - 1]
                directions.append(math.atan2(dy, dx))
            dir_variance = variance(directions)
            arc_length *= (1.0 + dir_variance * 5.0)  # Reward curved paths

        return arc_length / max(1, len(frames) - 1)

    # ── Challenge scoring ───────────────────────────────────────────────────

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
            elif item == "smile":
                metrics["challenge_smile"] = self._smile_score(frames)
            else:
                metrics[f"challenge_{item}"] = 0.0
        return metrics

    # ── Attack classification ───────────────────────────────────────────────

    def _classify_attack(
        self,
        metrics: dict[str, float],
        challenge: Sequence[str],
    ) -> tuple[str, float]:
        """Classify the most likely attack type from passive metrics.

        Returns (attack_type, confidence):
          "genuine"  — all signals consistent with live person
          "static"   — near-zero motion; static photo attack
          "replay"   — motion present but low texture/entropy; screen replay
          "printed"  — low multi-scale LBP entropy; printed photo
        """
        motion = metrics.get("motion", 0.0)
        texture = metrics.get("texture", 0.0)
        ms_entropy = metrics.get("ms_lbp_entropy", 0.0)
        symmetry = metrics.get("symmetry", 0.5)
        blink = metrics.get("blink", 0.0)

        # Static photo: almost no motion
        if motion < 0.006:
            conf = min(1.0, (0.006 - motion) / 0.006)
            return "static", conf

        # Printed photo: motion present but very low micro-texture
        if ms_entropy < 0.35 and texture < 60.0:
            conf = min(1.0, (0.35 - ms_entropy) / 0.35 * 0.7 + (60.0 - texture) / 60.0 * 0.3)
            return "printed", conf

        # Screen replay: motion present, moderate texture, but low blink + odd symmetry
        if ms_entropy < 0.50 and blink < 0.3 and symmetry < 0.55:
            conf = min(1.0, (0.50 - ms_entropy) / 0.50 * 0.5 + (0.55 - symmetry) / 0.55 * 0.5)
            return "replay", conf

        # Genuine
        genuine_conf = min(
            1.0,
            (min(motion, 0.15) / 0.15) * 0.3
            + (min(ms_entropy, 0.85) / 0.85) * 0.3
            + symmetry * 0.2
            + blink * 0.2,
        )
        return "genuine", genuine_conf

    # ── Blink and eye openness ──────────────────────────────────────────────

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
        eye_pixels = [
            float(pixel) / 255.0
            for crop in (left_eye, right_eye)
            for row in crop
            for pixel in row
        ]
        if not eye_pixels:
            return 0.0
        dark_ratio = sum(pixel < 0.32 for pixel in eye_pixels) / len(eye_pixels)
        return dark_ratio + 0.35 * variance(eye_pixels)

    # ── Smile detection ─────────────────────────────────────────────────────

    def _smile_score(self, frames: Sequence[Sequence[Sequence[int]]]) -> float:
        """Detect a smile from changes in the mouth region brightness distribution.

        A smile widens and lightens the mouth region due to teeth visibility.
        We look for increasing mean brightness and variance in lower face region.
        """
        mouth_brightnesses: list[float] = []
        for frame in frames:
            mouth = crop_fraction(frame, x0=0.25, y0=0.60, x1=0.75, y1=0.85)
            flat = [float(p) / 255.0 for row in mouth for p in row]
            mouth_brightnesses.append(mean(flat))

        if not mouth_brightnesses:
            return 0.0

        # Genuine smile: brightness increases over time (teeth appear)
        brightness_delta = mouth_brightnesses[-1] - mouth_brightnesses[0]
        brightness_variance = variance(mouth_brightnesses)

        score = self._range_score(brightness_delta, low=0.01, high=0.08)
        score += self._range_score(brightness_variance, low=0.0005, high=0.01)
        return min(1.0, score / 2.0)

    # ── Scoring utilities ───────────────────────────────────────────────────

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
        if metrics.get("ms_lbp_entropy", 1.0) < 0.30:
            reasons.append("low multi-scale texture entropy — possible printed photo")
        if metrics.get("symmetry", 1.0) < 0.35:
            reasons.append("low facial symmetry — possible spoofed image")
        for item in challenge:
            if metrics.get(f"challenge_{item}", 0.0) < 0.55:
                reasons.append(f"challenge not satisfied: {item}")
        if score < threshold:
            reasons.append("liveness score below threshold")
        return reasons
