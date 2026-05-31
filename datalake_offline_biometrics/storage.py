"""Local template persistence for offline deployments."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Iterable

from .recognition import FaceTemplate

STORE_VERSION = 1


class TemplateStore:
    """JSON template store with optional HMAC integrity protection."""

    def __init__(self, path: str | Path, *, secret: bytes | None = None) -> None:
        self.path = Path(path)
        self.secret = secret

    def save(self, templates: Iterable[FaceTemplate]) -> None:
        payload: dict[str, object] = {
            "version": STORE_VERSION,
            "templates": [template.to_dict() for template in templates],
        }
        if self.secret:
            payload["signature"] = self._signature(payload)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self) -> tuple[FaceTemplate, ...]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != STORE_VERSION:
            raise ValueError(f"unsupported template store version: {payload.get('version')}")
        if self.secret:
            expected = str(payload.get("signature", ""))
            actual = self._signature(
                {key: value for key, value in payload.items() if key != "signature"}
            )
            if not hmac.compare_digest(expected, actual):
                raise ValueError("template store signature verification failed")
        return tuple(FaceTemplate.from_dict(item) for item in payload["templates"])

    def _signature(self, payload: dict[str, object]) -> str:
        if not self.secret:
            raise ValueError("secret is required to sign template stores")
        canonical = json.dumps(
            {key: value for key, value in payload.items() if key != "signature"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(self.secret, canonical, hashlib.sha256).hexdigest()
