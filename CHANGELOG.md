# Changelog

All notable changes to SecureID are documented in this file.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] — 2026-09-12

### Added — ML Engine

- **Platt Sigmoid Calibration** (`ml_fusion.PlattCalibrator`): maps raw BRS scores to proper posterior probabilities `P(genuine|S) = 1/(1+exp(A·S+B))` via 50-step mini-batch gradient descent on binary cross-entropy. Fitted automatically from session history when ≥10 samples per class are available. Reference: Platt (1999).
- **Isotonic PAV Calibrator** (`ml_fusion.IsotonicCalibrator`): Pool Adjacent Violators monotone regression fallback when sample count is insufficient for Platt fitting. Reference: Zadrozny & Elkan (2002).
- **Mahalanobis Multi-Template Scoring** (`recognition.LBPHFaceRecognizer._mahalanobis_score`): computes intra-class template variance σ_d and applies consistency-aware correction `S_corr = (0.65·S_peak + 0.35·S_mean)/(1 + β·σ_d)` with β=0.50. Replaces naive `max()` over quality-weighted scores.
- **Adaptive EMA Weight Fusion** (`ml_fusion.ScoreFusion._adapt_weights`): fusion weights w_BRS/w_PAD now adapt from per-subsystem FNMR rates via EMA (α=0.08), clamped to [30%, 80%]. Replaces the previous heuristic AUC approximation.
- **Bayesian EER Estimator** (`ml_fusion.ScoreFusion.eer_credible_interval`): 95% credible interval on EER using Jeffreys Beta(0.5, 0.5) prior. Exposed as `(lo, hi)` tuple.
- **Challenge-Response Information Gain Scorer** (`app.js`): challenge compliance is scored as information gain over a uniform prior rather than a simple motion threshold, enabling finer discrimination between voluntary and involuntary motion.
- **`FusionResult.calibrated_score`**: Platt posterior `P(genuine|BRS)` included in every fusion result.
- **`FusionConfig.adaptive_weight_lr`**: configurable EMA learning rate (default 0.08).
- **`ScoreFusion.record_outcome(brs_failed, pad_failed)`**: new per-subsystem failure tracking parameters for EMA weight adaptation.
- **`AdaptiveThreshold.confidence_interval`**: 95% Bayesian credible interval on the adaptive EER threshold.
- **6th subject** (Verma, S. · NHO-0833) added to the simulator registry for richer multi-identity demonstration.

### Added — Interface

- **4-tab enterprise dashboard**: Identity Operations · Analytics & ROC · ML Architecture · Audit Ledger.
- **Conformance badge strip**: `ISO/IEC 30107-3 · NIST SP 800-76-2 · FIDO2 PAD L2 · ISO 19794-5`.
- **SVG ring gauges**: animated ring progress indicators for BRS, PAD, and CDS scores.
- **Live ROC curve canvas** (Analytics tab): FMR vs TMR plot with AUC annotation, rendered after ≥5 genuine + impostor events.
- **Score distribution histogram**: genuine vs impostor BRS overlay with τ threshold line.
- **Session timeline chart**: CDS over time with pass/fail colouring.
- **Threat vector breakdown**: pie-style bar chart of PAD classification counts.
- **Adaptive weight panel**: live display of w_BRS / w_PAD in both Operations and Analytics tabs.
- **Calibrator status chip**: shows Platt A/B parameters or sample count when in fallback mode.
- **Chain integrity verification**: "Verify Chain" button recomputes all SHA-256 hashes and highlights any tampered entries.
- **ISO-compliant JSON export**: includes EER, AUC, calibrator status, per-event calibrated scores and PAD levels.
- **BAM sensor frame**: animated corner brackets + scanline overlay, pass/fail glow states.
- **PAD Level badge**: `L0 — Live Subject` / `L1` / `L2 — Artefact Detected` with ISO 30107-3 §7.4 class names.
- **Keyboard shortcuts**: `⌘E` (register BRT), `⌘↵` (verify identity).
- **Glassmorphism design system**: 7-layer CSS cascade with `@layer tokens/base/layout/components/charts/animations/utilities`, precision dark palette, JetBrains Mono for monospace elements.

### Changed — Terminology (ISO/IEC 30107-3)

| Before | After |
|---|---|
| Enroll | Register Biometric Template (BRT) |
| Authenticate / Verify | Verify Identity |
| Face Capture | Biometric Acquisition Module (BAM) |
| Liveness | Presentation Attack Detection (PAD) |
| Fused Score | Composite Decision Score (CDS) |
| XAI | Decision Audit Trail (ISO 30109) |
| Attack | Threat Vector |
| FAR | False Match Rate (FMR) |
| FRR | False Non-Match Rate (FNMR) |
| TAR | True Match Rate (TMR) |
| recognition_weight / liveness_weight (stats) | w_brs / w_pad |

### Fixed

- **Bug**: `STATE.metric` (similarity metric selector) was also being passed as the score fusion method parameter — now fully decoupled via separate `STATE.fusionMethod`.
- **Bug**: `_adapt_weights()` in `ScoreFusion` previously used a hardcoded AUC approximation (`rw * 0.6 + lw * 0.4`) instead of actual per-subsystem error rates. Now uses real FNMR tracking via EMA.
- **Bug**: `identify()` previously used naive `max()` per subject over all templates — replaced by Mahalanobis multi-template scoring.

### Tests

- Updated `test_explanation_is_populated`: checks for `BRS`, `PAD`, `CDS` instead of `Recognition`, `Liveness`, `Fused` (ISO 30109 audit trail terminology).
- Updated `test_stats_dict_has_all_keys`: checks for `w_brs`, `w_pad`, `calibrator` keys.
- All 61 tests pass.

---

## [2.0.0] — 2026-09-11

### Added

- Multi-scale uniform LBP (radius 1, 2, 3 via grid sizes 8×8).
- Gabor filter bank (4 orientations × 2 frequencies × mean+std = 16 dims).
- Online PCA whitening via Welford's incremental algorithm (zero offline training data).
- Cosine similarity metric alongside χ² distance.
- Adaptive EER-based decision threshold estimator (sliding window, 200 samples).
- Template quality scoring: `IQS = blur × 0.6 + brightness × 0.4` (ISO 29794-1 aligned).
- `RecognitionResult.confidence_band`: normalised distance to decision boundary.
- SHA-256 tamper-evident audit chain (Web Crypto API in JS, hashlib in Python).
- Session analytics: ROC/DET curve computation utilities.
- Full frontend v2.0: ML visualisation, analytics dashboard, XAI trace.
- 61-test suite across all engine modules.

---

## [1.0.0] — 2026-09-08

### Added

- Initial offline facial recognition engine.
- Basic LBPH feature extraction with 8×8 grid, 59-bin uniform LBP.
- Liveness detection via inter-frame motion energy and texture sharpness.
- Simple weighted score fusion.
- Static decision threshold.
- Web demo (`index.html`) with 5-subject synthetic registry.
