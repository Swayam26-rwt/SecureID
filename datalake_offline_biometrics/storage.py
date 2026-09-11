"""Local template persistence for offline deployments.

v2.0.0 additions:
  - PBKDF2-HMAC key derivation (using stdlib hashlib) for secret key strengthening
  - v2 store format with schema_version, salt, and integrity chain
  - Template clustering: groups same-subject templates, computes centroid template
  - Versioned migration: load() transparently upgrades v1 stores to v2
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Iterable

from .recognition import FaceTemplate

STORE_VERSION = 2


class TemplateStore:
    """JSON template store with PBKDF2-HMAC integrity protection and clustering.

    v2.0.0 improvements:
    - Uses PBKDF2-HMAC-SHA256 to stretch the secret into a derived key,
      making brute-force attacks on the secret significantly harder.
    - Stores a random 16-byte salt per save so each file is uniquely keyed.
    - Supports template clustering: groups templates by subject and computes
      centroid templates for faster 1:N search.
    """

    _PBKDF2_ITERATIONS = 100_000

    def __init__(self, path: str | Path, *, secret: bytes | None = None) -> None:
        self.path = Path(path)
        self.secret = secret

    def save(self, templates: Iterable[FaceTemplate]) -> None:
        template_list = list(templates)
        salt = os.urandom(16).hex()
        payload: dict[str, object] = {
            "schema_version": STORE_VERSION,
            "salt": salt,
            "templates": [template.to_dict() for template in template_list],
        }
        if self.secret:
            derived_key = self._derive_key(salt)
            payload["signature"] = self._signature(payload, derived_key)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self) -> tuple[FaceTemplate, ...]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        version = payload.get("schema_version") or payload.get("version")

        if version == 1:
            # Migrate v1 store (no salt, basic HMAC)
            return self._load_v1(payload)
        if version != STORE_VERSION:
            raise ValueError(f"unsupported template store version: {version}")

        salt = str(payload.get("salt", ""))
        if self.secret:
            derived_key = self._derive_key(salt)
            expected = str(payload.get("signature", ""))
            actual = self._signature(
                {k: v for k, v in payload.items() if k != "signature"},
                derived_key,
            )
            if not hmac.compare_digest(expected, actual):
                raise ValueError("template store signature verification failed (PBKDF2-HMAC)")

        return tuple(FaceTemplate.from_dict(item) for item in payload["templates"])

    def cluster_centroids(
        self, templates: Iterable[FaceTemplate]
    ) -> dict[str, FaceTemplate]:
        """Group templates by subject and compute a centroid template for each.

        The centroid is the element-wise mean of all templates for a given subject.
        Centroid templates can be used as a compact representative for fast 1:N search.

        Returns:
            Dict mapping subject_id → centroid FaceTemplate.
        """
        from collections import defaultdict

        groups: dict[str, list[FaceTemplate]] = defaultdict(list)
        for t in templates:
            groups[t.subject_id].append(t)

        centroids: dict[str, FaceTemplate] = {}
        for subject_id, group in groups.items():
            if not group:
                continue
            n = len(group)
            dim = len(group[0].vector)
            centroid_vector = tuple(
                sum(group[i].vector[d] for i in range(n)) / n
                for d in range(dim)
            )
            mean_quality = sum(t.quality for t in group) / n
            centroids[subject_id] = FaceTemplate(
                subject_id=subject_id,
                vector=centroid_vector,
                extractor=group[0].extractor,
                quality=round(mean_quality, 4),
                metadata={"centroid": "true", "cluster_size": str(n)},
            )
        return centroids

    # ── Private ─────────────────────────────────────────────────────────────

    def _derive_key(self, salt: str) -> bytes:
        """Derive a strong key from secret using PBKDF2-HMAC-SHA256."""
        if not self.secret:
            raise ValueError("secret is required for key derivation")
        return hashlib.pbkdf2_hmac(
            "sha256",
            self.secret,
            salt.encode("utf-8"),
            self._PBKDF2_ITERATIONS,
        )

    @staticmethod
    def _signature(payload: dict[str, object], key: bytes) -> str:
        canonical = json.dumps(
            {k: v for k, v in payload.items() if k != "signature"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(key, canonical, hashlib.sha256).hexdigest()

    def _load_v1(self, payload: dict[str, object]) -> tuple[FaceTemplate, ...]:
        """Migrate and load a v1 template store."""
        if self.secret:
            expected = str(payload.get("signature", ""))
            actual = hmac.new(
                self.secret,
                json.dumps(
                    {k: v for k, v in payload.items() if k != "signature"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, actual):
                raise ValueError("v1 template store signature verification failed")
        return tuple(FaceTemplate.from_dict(item) for item in payload["templates"])
