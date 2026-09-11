"""Integration facade for offline biometric operations.

v2.0.0 additions:
  - Score-level fusion via ml_fusion.ScoreFusion
  - batch_enroll() for multi-pose enrollment workflow
  - explain() method returning human-readable XAI decision trace
  - Cryptographically-chained audit trail with hash chaining
  - SessionAnalytics integration for FAR/FRR tracking
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .analytics import AuthEvent, SessionAnalytics
from .image_ops import GrayImage, image_quality_score
from .liveness import LivenessDetector, LivenessResult
from .ml_fusion import FusionConfig, FusionResult, ScoreFusion
from .recognition import FaceTemplate, LBPHFaceRecognizer, RecognitionResult


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuthenticationResult:
    accepted: bool
    recognition: RecognitionResult
    liveness: LivenessResult
    fusion: FusionResult
    reasons: tuple[str, ...]
    explanation: tuple[str, ...]    # XAI human-readable trace
    audit_hash: str = ""            # SHA-256 of this decision for chaining


@dataclass
class AuditEntry:
    """Tamper-evident audit log entry with hash chaining."""

    index: int
    timestamp: str
    subject_id: str | None
    claimed_id: str
    accepted: bool
    fused_score: float
    attack_hint: str
    entry_hash: str    # SHA-256(prev_hash + payload)


# ── Engine ────────────────────────────────────────────────────────────────────

class OfflineBiometricEngine:
    """Single entry point for Datalake 3.0 edge authentication flows.

    v2.0.0 features:
    - Multi-modal score fusion (recognition + liveness → unified decision)
    - Batch enrollment with quality gating
    - Cryptographically chained audit trail
    - XAI explain() method for human-readable decision traces
    - SessionAnalytics integration for FAR/FRR/TAR tracking
    """

    def __init__(
        self,
        *,
        recognizer: LBPHFaceRecognizer | None = None,
        liveness_detector: LivenessDetector | None = None,
        fusion_config: FusionConfig | None = None,
        templates: Iterable[FaceTemplate] | None = None,
        min_enrollment_quality: float = 0.30,
    ) -> None:
        self.recognizer = recognizer or LBPHFaceRecognizer()
        self.liveness_detector = liveness_detector or LivenessDetector()
        self.fusion = ScoreFusion(fusion_config)
        self._templates: list[FaceTemplate] = list(templates or ())
        self._min_quality = min_enrollment_quality
        self._audit_log: list[AuditEntry] = []
        self._prev_hash: str = "0" * 64   # Genesis hash
        self._analytics = SessionAnalytics()

    @property
    def templates(self) -> tuple[FaceTemplate, ...]:
        return tuple(self._templates)

    @property
    def audit_log(self) -> tuple[AuditEntry, ...]:
        return tuple(self._audit_log)

    @property
    def analytics(self) -> SessionAnalytics:
        return self._analytics

    # ── Enrollment ─────────────────────────────────────────────────────────

    def enroll(
        self,
        subject_id: str,
        face_crops: Sequence[GrayImage],
        *,
        metadata: dict[str, str] | None = None,
        quality_gate: bool = True,
    ) -> tuple[FaceTemplate, ...]:
        """Enroll a subject with one or more face crops.

        Args:
            subject_id: Unique subject identifier.
            face_crops: List of face crop matrices.
            metadata: Optional extra metadata for templates.
            quality_gate: If True, reject crops below min_enrollment_quality.

        Returns:
            Tuple of created FaceTemplate objects.
        """
        if not face_crops:
            raise ValueError("at least one face crop is required for enrollment")

        created: list[FaceTemplate] = []
        rejected = 0
        for crop in face_crops:
            if quality_gate:
                quality = image_quality_score(crop)
                if quality["overall"] < self._min_quality:
                    rejected += 1
                    continue
            template = self.recognizer.create_template(subject_id, crop, metadata=metadata)
            self._templates.append(template)
            created.append(template)

        if not created:
            raise ValueError(
                f"All {len(face_crops)} crops rejected by quality gate "
                f"(min quality {self._min_quality:.2f}). "
                f"Consider lowering min_enrollment_quality or improving image conditions."
            )
        self._analytics.record_enrollment(subject_id)
        return tuple(created)

    def batch_enroll(
        self,
        enrollments: Sequence[tuple[str, Sequence[GrayImage]]],
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, int]:
        """Enroll multiple subjects in one call.

        Args:
            enrollments: List of (subject_id, face_crops) tuples.
            metadata: Shared metadata for all templates.

        Returns:
            Dict mapping subject_id → number of templates created.
        """
        result: dict[str, int] = {}
        for subject_id, crops in enrollments:
            try:
                created = self.enroll(subject_id, crops, metadata=metadata)
                result[subject_id] = len(created)
            except ValueError:
                result[subject_id] = 0
        return result

    # ── Identification & verification ───────────────────────────────────────

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

    # ── Liveness ────────────────────────────────────────────────────────────

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

    # ── Authentication ──────────────────────────────────────────────────────

    def authenticate(
        self,
        subject_id: str,
        face_crop: GrayImage,
        liveness_frames: Sequence[GrayImage],
        *,
        challenge: Sequence[str] | None = None,
        recognition_threshold: float | None = None,
        liveness_threshold: float | None = None,
        record_for_adaptation: bool = True,
    ) -> AuthenticationResult:
        """Authenticate a subject with multi-modal fusion.

        Runs face recognition + liveness in sequence, fuses scores,
        logs the decision to the cryptographic audit chain, and
        optionally records the outcome for adaptive threshold learning.
        """
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
        fusion_result = self.fusion.fuse(
            recognition.similarity,
            liveness.score,
        )

        accepted = fusion_result.accepted

        reasons: list[str] = []
        if not recognition.accepted:
            reasons.append("face recognition failed")
        if not liveness.passed:
            reasons.extend(liveness.reasons)

        explanation = self.explain(recognition, liveness, fusion_result)

        audit_hash = self._chain_audit(
            AuditEntry(
                index=len(self._audit_log),
                timestamp=datetime.now(timezone.utc).isoformat(),
                subject_id=subject_id,
                claimed_id=subject_id,
                accepted=accepted,
                fused_score=fusion_result.fused_score,
                attack_hint=liveness.attack_type_hint,
                entry_hash="",  # Will be filled below
            )
        )

        # Record for adaptive learning
        if record_for_adaptation:
            self.fusion.record_outcome(fusion_result.fused_score, is_genuine=accepted)

        # Record analytics event
        self._analytics.record(
            AuthEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                subject_id=subject_id if accepted else None,
                claimed_id=subject_id,
                accepted=accepted,
                recognition_score=recognition.similarity,
                liveness_score=liveness.score,
                fused_score=fusion_result.fused_score,
                attack_type_hint=liveness.attack_type_hint,
                latency_ms=0.0,   # Caller may set this if timing is available
                challenge=",".join(challenge or []),
                is_genuine=recognition.accepted,
            )
        )

        return AuthenticationResult(
            accepted=accepted,
            recognition=recognition,
            liveness=liveness,
            fusion=fusion_result,
            reasons=tuple(reasons),
            explanation=explanation,
            audit_hash=audit_hash,
        )

    # ── XAI explanation ─────────────────────────────────────────────────────

    @staticmethod
    def explain(
        recognition: RecognitionResult,
        liveness: LivenessResult,
        fusion: FusionResult,
    ) -> tuple[str, ...]:
        """Generate human-readable explanation of the authentication decision.

        Returns a tuple of explanation strings ordered from most to least decisive.
        Suitable for display in UI audit panels and operator reports.
        """
        lines: list[str] = [
            f"Recognition: similarity {recognition.similarity:.3f} "
            f"(threshold {recognition.threshold:.2f}, metric={recognition.metric}) "
            f"→ {'PASS' if recognition.accepted else 'FAIL'}",

            f"Liveness: score {liveness.score:.3f} "
            f"(threshold {liveness.threshold:.2f}) "
            f"→ {'PASS' if liveness.passed else 'FAIL'}",

            f"Attack hint: {liveness.attack_type_hint} "
            f"(confidence {liveness.attack_confidence:.0%})",

            f"Fusion ({fusion.method}): fused score {fusion.fused_score:.3f} "
            f"vs threshold {fusion.threshold:.3f} "
            f"→ {'ACCEPT' if fusion.accepted else 'DENY'}",
        ]

        if recognition.confidence_band > 0:
            lines.append(
                f"Decision margin: {recognition.confidence_band:.3f} from boundary "
                f"({'high' if recognition.confidence_band > 0.3 else 'low'} confidence)"
            )

        for metric_key in ("texture", "motion", "ms_lbp_entropy", "symmetry", "blink"):
            val = liveness.metrics.get(metric_key)
            if val is not None:
                lines.append(f"  {metric_key}: {val:.4f}")

        if liveness.reasons:
            lines.append("Failure reasons: " + "; ".join(liveness.reasons))

        return tuple(lines)

    # ── Template management ─────────────────────────────────────────────────

    def replace_templates(self, templates: Iterable[FaceTemplate]) -> None:
        self._templates = list(templates)

    # ── Audit chain ─────────────────────────────────────────────────────────

    def _chain_audit(self, entry: AuditEntry) -> str:
        """Append audit entry with cryptographic hash chaining.

        Each entry's hash includes the previous entry's hash, creating
        a tamper-evident chain similar to a blockchain structure.
        """
        payload = json.dumps(
            {
                "index": entry.index,
                "timestamp": entry.timestamp,
                "subject_id": entry.subject_id,
                "accepted": entry.accepted,
                "fused_score": entry.fused_score,
                "attack_hint": entry.attack_hint,
                "prev_hash": self._prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        entry_hash = hashlib.sha256(payload).hexdigest()
        chained_entry = AuditEntry(
            index=entry.index,
            timestamp=entry.timestamp,
            subject_id=entry.subject_id,
            claimed_id=entry.claimed_id,
            accepted=entry.accepted,
            fused_score=entry.fused_score,
            attack_hint=entry.attack_hint,
            entry_hash=entry_hash,
        )
        self._audit_log.append(chained_entry)
        self._prev_hash = entry_hash
        return entry_hash
