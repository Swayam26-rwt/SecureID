# CHANGELOG

All notable changes to NHAI Datalake 3.0 Biometric Engine are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-09-12

### Added — ML Engine

- **Gabor filter bank** (`image_ops.py`): 4-orientation × 2-frequency kernels;
  extracts mean + variance response (16-dim vector) for frequency-selective
  texture analysis. Robust to brightness changes and partial occlusion.

- **LPQ descriptor** (`image_ops.py`): Local Phase Quantization via short-time
  Fourier transform phase quantization; 256-bin blur-invariant histogram.
  Ideal for degraded or out-of-focus capture conditions.

- **CLAHE adaptive equalization** (`image_ops.py`): Tile-based contrast limited
  adaptive histogram equalization (8×8 tiles, default clip=3.0). Replaces global
  histogram equalization. Eliminates blocking artifacts at tile boundaries via
  bilinear interpolation. Dramatically improves performance under partial lighting.

- **Image quality metrics** (`image_ops.py`):
  - `blur_score()`: Laplacian variance, normalized to [0, 1]
  - `brightness_score()`: Gaussian window centred at optimal face brightness
  - `occlusion_ratio()`: Flat-region detector for masks/glasses
  - `image_quality_score()`: Combined quality report dict with `overall` fusion

- **Cosine similarity metric** (`recognition.py`): Scale-invariant dot-product
  similarity shifted to [0, 1]. Superior to chi-square for whitened feature spaces.
  Configurable via `metric='cosine'` (default).

- **Online PCA Whitening** (`recognition.py`, `OnlinePCAWhitener`): Welford's
  algorithm estimates per-dimension mean and standard deviation incrementally.
  Decorrelates feature dimensions without offline training data or matrix
  decomposition. O(d) memory, updates at enroll/extract time.

- **Adaptive EER threshold** (`recognition.py`, `AdaptiveThreshold`): Maintains
  sliding window (200 samples) of genuine and impostor similarity scores. Sweeps
  50 threshold candidates to find the Equal Error Rate operating point (FAR ≈ FRR).
  Falls back to static threshold when < 5 samples per class.

- **Template quality scoring** (`recognition.py`): `create_template()` computes
  `blur_score * 0.6 + brightness_score * 0.4` and stores in `FaceTemplate.quality`.
  Low-quality templates are weighted down by `max(0.5, quality)` during identification.

- **Multi-scale LBP** (`liveness.py`): LBP computed at radii 1, 2, 3 combined
  into a single histogram. Printed photos and screen replays show lower micro-texture
  entropy due to digital compression and rendering. Weight: 0.12.

- **Optical flow arc** (`liveness.py`): Centroid displacement arc length across
  the frame burst. Rewards curved motion paths (genuine) over linear/static paths
  (replay). Direction variance multiplier: `arc *= (1 + dir_var * 5)`. Weight: 0.10.

- **Face symmetry score** (`liveness.py`): Mean absolute pixel difference between
  left and mirrored right half. Genuine faces score 0.6–0.8; printed/rotated
  attacks score lower. Weight: 0.08.

- **Attack type classification** (`liveness.py`): `_classify_attack()` returns
  `(type, confidence)` where type ∈ {`genuine`, `static`, `printed`, `replay`}.
  Exposed in `LivenessResult.attack_type_hint` and `attack_confidence`.

- **New challenges** (`liveness.py`): `nod` (vertical centroid shift) and
  `smile` (mouth-region brightness increase) added to the challenge set.

- **Multi-modal score fusion** (`ml_fusion.py`, new module):
  - `FusionConfig`: configurable recognition/liveness weights, fusion method, threshold
  - `ScoreFusion`: `fuse()`, `record_outcome()`, `roc_curve()`, `eer` property
  - Fusion methods: `weighted_sum`, `weighted_product` (geometric mean), `min`
  - Adaptive weight learning via exponential moving average
  - ROC curve computation at 100 threshold steps from recorded session scores

- **Session analytics** (`analytics.py`, new module):
  - `SessionAnalytics`: `record()`, `report()`, `time_series()`, `score_distribution()`
  - `SessionReport`: FAR, FRR, TAR, attack breakdown, mean latency statistics
  - `export_json()`: full session audit with score histogram data
  - `AuthEvent` dataclass for standardised per-attempt records

- **Score-level fusion in engine** (`engine.py`):
  - `authenticate()` now runs `ScoreFusion.fuse()` and returns `FusionResult`
  - `batch_enroll()`: enroll multiple subjects in one call
  - `enroll()` extended with `quality_gate` parameter
  - `explain()` static method: 7+ line XAI decision trace
  - `_chain_audit()`: SHA-256 hash chaining for tamper-evident audit trail
  - `SessionAnalytics` integrated into engine lifecycle

- **Template store upgrades** (`storage.py`):
  - PBKDF2-HMAC-SHA256 key derivation (100k iterations) with random 16-byte salt
  - v2 store format: `schema_version=2`, salt stored per-file
  - `cluster_centroids()`: element-wise mean centroid template per subject
  - Transparent migration of v1 stores on first `load()`

### Changed

- Default recognition threshold from `0.78` (chi-square) to `0.40` (cosine)
- `RecognitionResult` gained: `metric`, `confidence_band`, quality-weighted scoring
- `LivenessResult` gained: `attack_type_hint`, `attack_confidence`
- `AuthenticationResult` gained: `fusion: FusionResult`, `explanation: tuple[str, ...]`
- `FaceTemplate.to_dict()` / `from_dict()` now include `quality` field (backwards compat)
- Extractor ID format: `lbph-v2-{metric}-{desc}-{width}x{height}-{grid}`

### Fixed

- Tests updated to use cosine-compatible thresholds and identify()-based ranking

---

## [1.0.0] — Initial release

- Uniform LBP (59-bin) with 8×8 spatial grid
- Chi-square distance matching
- Basic liveness: texture, entropy, motion, exposure variance, blink challenge
- JSON template store with HMAC-SHA256 integrity
- CLI demo and benchmark script
