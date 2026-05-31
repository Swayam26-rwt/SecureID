# Offline Biometrics Integration

## Placement in Datalake 3.0

The package is designed to sit behind the Datalake authentication or attendance boundary:

1. The existing camera layer detects a face and sends normalized face crops to `OfflineBiometricEngine`.
2. Enrollment calls `engine.enroll(subject_id, face_crops)` with 3-5 captures per person.
3. Verification calls `engine.authenticate(...)` with one probe crop and a burst of 8-20 crops for liveness.
4. Templates are persisted with `TemplateStore`; raw frames can be discarded immediately.

No runtime path performs network I/O.

## Recommended Capture Flow

- Resolution: crop each detected face to at least `96x96` grayscale pixels.
- Enrollment: capture front-facing samples under the lighting expected at the site.
- Authentication: capture a 1-3 second burst and request a randomized challenge such as `["blink"]`, `["turn_left"]`, or `["turn_right", "blink"]`.
- Edge tuning: log only aggregate scores and reasons, never raw images, then tune thresholds per site.

## Thresholds

Defaults are intentionally moderate:

- Recognition threshold: `0.78`
- Liveness threshold: `0.62`

For higher-security access control, raise both thresholds after local validation. For attendance in harsh lighting, lower thresholds only when a second factor or operator review exists.

## Storage

```python
from datalake_offline_biometrics import OfflineBiometricEngine, TemplateStore

store = TemplateStore("/var/lib/datalake/biometrics/templates.json", secret=b"site-local-key")
engine = OfflineBiometricEngine()

engine.enroll("employee-1001", face_crops)
store.save(engine.templates)

engine = OfflineBiometricEngine(templates=store.load())
```

Use a site-local secret from the Datalake secure configuration store. The HMAC protects template integrity on offline devices; it is not encryption.

## Security Notes

- Do not store raw face frames unless a policy explicitly requires it.
- Run liveness before granting access; static verification alone is vulnerable to printed-photo attacks.
- Prefer randomized challenges over passive liveness where the user experience allows it.
- Keep per-site threshold calibration data separate from personally identifiable records.
- For high-risk deployments, use this package as the offline fallback and plug in a locally bundled neural embedding model through the recognizer interface once approved model assets are available.
