# Datalake 3.0 Offline Biometrics

Lightweight, zero-network facial recognition and liveness detection primitives for Datalake 3.0 edge deployments.

This package is intentionally dependency-free. It accepts already-cropped grayscale face images from the host app, extracts compact LBPH-style templates, evaluates recognition locally, and checks liveness from short frame bursts using motion, blink, texture, and optional challenge-response signals.

## Capabilities

- Entirely offline: no cloud API, telemetry, model downloads, or network calls.
- Lightweight: pure Python standard library implementation.
- Privacy-aware: stores face templates, not raw face frames.
- Integration-ready: a single `OfflineBiometricEngine` supports enrollment, identification, verification, liveness, and combined authentication.
- Inspectable: all scores include metrics and reasons for edge-device tuning.

## Quick Start

```python
from datalake_offline_biometrics import OfflineBiometricEngine

engine = OfflineBiometricEngine()

engine.enroll("operator-42", face_crops=[face_crop_1, face_crop_2])

auth = engine.authenticate(
    subject_id="operator-42",
    face_crop=probe_face_crop,
    liveness_frames=burst_of_face_crops,
    challenge=["blink"],
)

if auth.accepted:
    print("Access granted", auth.recognition.similarity, auth.liveness.score)
else:
    print("Access denied", auth.reasons)
```

Each image is a two-dimensional list or tuple of grayscale pixel values in the range `0..255`. If Datalake already uses OpenCV, convert with:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_crop = gray[y:y+h, x:x+w].tolist()
```

See [docs/offline_biometrics_integration.md](docs/offline_biometrics_integration.md) for integration, thresholds, storage, and operational notes.

## Live Demo

The repository includes a static GitHub Pages demo in `index.html` with supporting files under `assets/`. It runs entirely in the browser and demonstrates enrollment templates, identity verification, liveness scoring, spoof rejection, and mismatch rejection without any backend service.
