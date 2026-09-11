"""Operational session analytics for biometric deployments.

Tracks per-session FAR, FRR, TAR, match latency, and rejection statistics.
Generates JSON reports suitable for dashboard consumption and audit export.

All computation uses only Python standard library.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuthEvent:
    """A single authentication attempt record."""

    timestamp: str
    subject_id: str | None
    claimed_id: str
    accepted: bool
    recognition_score: float
    liveness_score: float
    fused_score: float
    attack_type_hint: str
    latency_ms: float
    challenge: str
    is_genuine: bool   # Ground truth: claimed_id == subject_id (simulator-known)


@dataclass
class SessionReport:
    """Summary statistics for a completed or ongoing session."""

    session_id: str
    started_at: str
    events_total: int = 0
    accepts: int = 0
    denials: int = 0

    # Error rates
    genuine_attempts: int = 0
    impostor_attempts: int = 0
    false_accepts: int = 0       # Impostors accepted (FAR numerator)
    false_rejects: int = 0       # Genuines denied (FRR numerator)

    # Scores
    mean_recognition_score: float = 0.0
    mean_liveness_score: float = 0.0
    mean_fused_score: float = 0.0
    mean_latency_ms: float = 0.0

    # Derived rates
    far: float = 0.0    # False Accept Rate = FA / impostor_attempts
    frr: float = 0.0    # False Reject Rate = FR / genuine_attempts
    tar: float = 0.0    # True Accept Rate = 1 - FRR

    # Attack breakdown
    static_attacks: int = 0
    replay_attacks: int = 0
    printed_attacks: int = 0
    genuine_attempts_flagged: int = 0

    enrolled_subjects: int = 0
    session_ended_at: str = ""


class SessionAnalytics:
    """Operational analytics collector for edge biometric deployments.

    Records authentication events in memory, computes rolling statistics,
    and exports JSON session reports. Designed to work both in real-time
    (streaming updates) and batch mode (post-session analysis).

    Usage::

        analytics = SessionAnalytics()
        analytics.record(event)
        report = analytics.report()
        analytics.export_json("session_report.json")
    """

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id or f"session-{int(time.time())}"
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._events: list[AuthEvent] = []
        self._enrolled_subjects: set[str] = set()

    @property
    def session_id(self) -> str:
        return self._session_id

    def record(self, event: AuthEvent) -> None:
        """Add a new authentication event to the session."""
        self._events.append(event)

    def record_enrollment(self, subject_id: str) -> None:
        """Track enrollment of a subject."""
        self._enrolled_subjects.add(subject_id)

    def report(self) -> SessionReport:
        """Compute a current SessionReport from recorded events."""
        report = SessionReport(
            session_id=self._session_id,
            started_at=self._started_at,
            enrolled_subjects=len(self._enrolled_subjects),
        )

        if not self._events:
            return report

        report.events_total = len(self._events)
        report.accepts = sum(1 for e in self._events if e.accepted)
        report.denials = report.events_total - report.accepts

        report.genuine_attempts = sum(1 for e in self._events if e.is_genuine)
        report.impostor_attempts = sum(1 for e in self._events if not e.is_genuine)
        report.false_accepts = sum(1 for e in self._events if not e.is_genuine and e.accepted)
        report.false_rejects = sum(1 for e in self._events if e.is_genuine and not e.accepted)

        rec_scores = [e.recognition_score for e in self._events]
        live_scores = [e.liveness_score for e in self._events]
        fused_scores = [e.fused_score for e in self._events]
        latencies = [e.latency_ms for e in self._events]

        report.mean_recognition_score = round(sum(rec_scores) / len(rec_scores), 4)
        report.mean_liveness_score = round(sum(live_scores) / len(live_scores), 4)
        report.mean_fused_score = round(sum(fused_scores) / len(fused_scores), 4)
        report.mean_latency_ms = round(sum(latencies) / len(latencies), 2)

        report.far = round(report.false_accepts / max(1, report.impostor_attempts), 4)
        report.frr = round(report.false_rejects / max(1, report.genuine_attempts), 4)
        report.tar = round(1.0 - report.frr, 4)

        report.static_attacks = sum(1 for e in self._events if e.attack_type_hint == "static")
        report.replay_attacks = sum(1 for e in self._events if e.attack_type_hint == "replay")
        report.printed_attacks = sum(1 for e in self._events if e.attack_type_hint == "printed")
        report.genuine_attempts_flagged = sum(1 for e in self._events if e.attack_type_hint == "genuine" and not e.accepted)

        return report

    def time_series(self, metric: str = "fused_score") -> list[dict[str, Any]]:
        """Return time-series data for a given metric field.

        Args:
            metric: Field name from AuthEvent (default 'fused_score').

        Returns:
            List of {timestamp, value} dicts, sorted chronologically.
        """
        result = []
        for event in self._events:
            value = getattr(event, metric, None)
            if value is not None:
                result.append({"timestamp": event.timestamp, "value": value})
        return result

    def score_distribution(
        self, score_field: str = "fused_score", bins: int = 10
    ) -> dict[str, list[float]]:
        """Compute score histogram for genuine and impostor attempts.

        Returns:
            dict with 'bin_edges', 'genuine', 'impostor' lists.
        """
        g_scores = [getattr(e, score_field) for e in self._events if e.is_genuine]
        i_scores = [getattr(e, score_field) for e in self._events if not e.is_genuine]
        all_scores = g_scores + i_scores
        if not all_scores:
            return {"bin_edges": [], "genuine": [], "impostor": []}

        lo, hi = min(all_scores), max(all_scores)
        if lo >= hi:
            hi = lo + 0.001
        bin_edges = [lo + (hi - lo) * k / bins for k in range(bins + 1)]

        def count_bins(scores: list[float]) -> list[float]:
            counts = [0.0] * bins
            for s in scores:
                idx = min(bins - 1, int((s - lo) / (hi - lo) * bins))
                counts[idx] += 1
            total = sum(counts) or 1
            return [c / total for c in counts]

        return {
            "bin_edges": [round(e, 4) for e in bin_edges],
            "genuine": count_bins(g_scores),
            "impostor": count_bins(i_scores),
        }

    def export_json(self, path: str) -> None:
        """Export full session report + events to a JSON file."""
        report = self.report()
        payload = {
            "report": asdict(report),
            "events": [
                {k: v for k, v in asdict(event).items()}
                for event in self._events
            ],
            "distribution": self.score_distribution(),
        }
        import pathlib
        pathlib.Path(path).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the current report."""
        return asdict(self.report())

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        r = self.report()
        return (
            f"SessionAnalytics(id={self._session_id!r}, "
            f"events={r.events_total}, "
            f"FAR={r.far:.4f}, FRR={r.frr:.4f}, TAR={r.tar:.4f})"
        )
