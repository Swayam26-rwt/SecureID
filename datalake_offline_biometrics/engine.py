"""Integration facade for offline biometric operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .image_ops import GrayImage
from .liveness import LivenessDetector, LivenessResult
from .recognition import FaceTemplate, LBPHFaceRecognizer, RecognitionResult


@dataclass(frozen=True)
class AuthenticationResult:
    accepted: bool
    recognition: RecognitionResult
    liveness: LivenessResult
    reasons: tuple[str, ...]


class OfflineBiometricEngine:
    """Single entry point for Datalake edge authentication flows."""

    def __init__(
        self,
        *,
        recognizer: LBPHFaceRecognizer | None = None,
        liveness_detector: LivenessDetector | None = None,
        templates: Iterable[FaceTemplate] | None = None,
    ) -> None:
        self.recognizer = recognizer or LBPHFaceRecognizer()
        self.liveness_detector = liveness_detector or LivenessDetector()
        self._templates: list[FaceTemplate] = list(templates or ())

    @property
    def templates(self) -> tuple[FaceTemplate, ...]:
        return tuple(self._templates)

    def enroll(
        self,
        subject_id: str,
        face_crops: Sequence[GrayImage],
        *,
        metadata: dict[str, str] | None = None,
    ) -> tuple[FaceTemplate, ...]:
        if not face_crops:
            raise ValueError("at least one face crop is required for enrollment")
        created = tuple(
            self.recognizer.create_template(subject_id, crop, metadata=metadata)
            for crop in face_crops
        )
        self._templates.extend(created)
        return created

    def identify(self, face_crop: GrayImage, *, top_k: int = 3) -> RecognitionResult:
        return self.recognizer.identify(face_crop, self._templates, top_k=top_k)

    def verify(
        self,
        subject_id: str,
        face_crop: GrayImage,
        *,
        threshold: float | None = None,
    ) -> RecognitionResult:
        return self.recognizer.verify(
            subject_id,
            face_crop,
            self._templates,
            threshold=threshold,
        )

    def assess_liveness(
        self,
        frames: Sequence[GrayImage],
        *,
        challenge: Sequence[str] | None = None,
        threshold: float | None = None,
    ) -> LivenessResult:
        return self.liveness_detector.assess(
            frames,
            challenge=challenge,
            threshold=threshold,
        )

    def authenticate(
        self,
        subject_id: str,
        face_crop: GrayImage,
        liveness_frames: Sequence[GrayImage],
        *,
        challenge: Sequence[str] | None = None,
        recognition_threshold: float | None = None,
        liveness_threshold: float | None = None,
    ) -> AuthenticationResult:
        recognition = self.verify(
            subject_id,
            face_crop,
            threshold=recognition_threshold,
        )
        liveness = self.assess_liveness(
            liveness_frames,
            challenge=challenge,
            threshold=liveness_threshold,
        )
        reasons: list[str] = []
        if not recognition.accepted:
            reasons.append("face recognition failed")
        if not liveness.passed:
            reasons.extend(liveness.reasons)

        return AuthenticationResult(
            accepted=recognition.accepted and liveness.passed,
            recognition=recognition,
            liveness=liveness,
            reasons=tuple(reasons),
        )

    def replace_templates(self, templates: Iterable[FaceTemplate]) -> None:
        self._templates = list(templates)
