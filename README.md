# SecureID v3.0 &nbsp;·&nbsp; ISO/IEC 30107-3 Biometric Decision Platform

<p align="center">
  <img src="https://img.shields.io/badge/Standard-ISO%2FIEC%2030107--3-00e5a0?style=flat-square&logoColor=white" />
  <img src="https://img.shields.io/badge/Standard-NIST%20SP%20800--76--2-38b6ff?style=flat-square" />
  <img src="https://img.shields.io/badge/Standard-FIDO2%20PAD%20L2-b47fff?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-f5a623?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Dependencies-Zero-00e5a0?style=flat-square" />
  <img src="https://img.shields.io/badge/Tests-61%20passing-00e5a0?style=flat-square" />
  <img src="https://img.shields.io/badge/version-3.0.0-38b6ff?style=flat-square" />
</p>

<p align="center">
  <strong>Air-gapped · Zero external dependencies · Standard library only</strong><br/>
  A complete offline biometric identity verification engine implementing ISO/IEC 30107-3 Presentation Attack Detection,
  Platt sigmoid score calibration, Mahalanobis multi-template matching, Bayesian EER estimation,
  adaptive EMA weight fusion, and a SHA-256 tamper-evident audit chain.
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [ML Pipeline](#ml-pipeline)
4. [Quick Start](#quick-start)
5. [Web Dashboard](#web-dashboard)
6. [Python API](#python-api)
7. [Test Suite](#test-suite)
8. [Standards Compliance](#standards-compliance)
9. [Design Constraints](#design-constraints)
10. [Contributing](#contributing)

---

## Overview

SecureID is a **research-grade biometric decision platform** built entirely on the Python standard library and vanilla JavaScript. It ships two parallel implementations:

| Layer | Technology | Purpose |
|---|---|---|
| **Python Engine** | `biometric_engine/` | Production-grade biometric pipeline, testable, importable as a package |
| **Web Simulator** | `index.html` + `assets/` | ISO 30107-3 operations dashboard, runs from `file://` or any static server |

The web simulator is a **near-complete manual port** of the Python engine's mathematics, enabling in-browser demonstration with no build step, no Node.js, and no network dependency.

---

## Architecture

```
SecureID v3.0
│
├── biometric_engine/          # Core Python package
│   ├── engine.py              # OfflineBiometricEngine facade (ISO 30107-3 API)
│   ├── recognition.py         # LBPHFaceRecognizer + Mahalanobis multi-template scoring
│   ├── ml_fusion.py           # ScoreFusion + PlattCalibrator + IsotonicCalibrator
│   ├── liveness.py            # LivenessDetector — passive PAD + challenge-response
│   ├── image_ops.py           # Low-level image primitives (pure Python)
│   └── analytics.py           # SessionAnalytics + audit chain
│
├── assets/
│   ├── app.js                 # Full ML pipeline port (JS) — runs in-browser
│   └── styles.css             # Enterprise design system (CSS Layers, glassmorphism)
│
├── index.html                 # 4-tab ISO 30107-3 operations dashboard
└── tests/                     # 61 unit tests (pytest)
```

### Data Flow

```
Sensor Frame (96×96 grey)
       │
       ▼
┌─────────────────────────────────────────────┐
│  BAM — Biometric Acquisition Module         │
│  Multi-scale LBP (8×8 grid, 59 bins)        │
│  + Gabor bank (4θ × 2f) + Appearance        │
│  + Facial geometry centroids                 │
│  → 4407-dimensional raw feature vector      │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Welford Online Whitening                   │
│  Incremental μ/σ² per dimension (O(d) mem) │
│  → Whitened feature vector z̃               │
└─────────────┬───────────────────────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌──────────┐    ┌──────────────────────────────┐
│ BRS      │    │ PAD Engine (ISO 30107-3 §6)  │
│ Cosine / │    │ · MS-LBP texture entropy     │
│ χ² sim   │    │ · Optical-flow arc curvature │
│ → S_raw  │    │ · Bilateral symmetry residual│
└────┬─────┘    │ · Laplacian sharpness        │
     │          │ · Challenge info-gain (IG)   │
     ▼          └──────────────┬───────────────┘
┌──────────┐                   │
│ Platt    │                   │ P_pad ∈ [0,1]
│ Sigmoid  │                   │
│ Calib.   │                   │
│ P(gen|S) │                   │
└────┬─────┘                   │
     │ P_rec                   │
     └──────────┬──────────────┘
                ▼
┌─────────────────────────────────────────────┐
│  Adaptive EMA Score Fusion                  │
│  CDS = w_BRS·P_rec + w_PAD·P_pad            │
│  Weights adapt from per-subsystem FNMR      │
│  (EMA α=0.08, clamped [30%,80%])            │
└─────────────┬───────────────────────────────┘
              │
              ▼
       CDS ≥ τ_EER ?
       ┌──────┴──────┐
      YES            NO
       │              │
   VERIFIED       REJECTED
       │
       ▼
┌─────────────────────────────────────────────┐
│  SHA-256 Tamper-Evident Audit Chain         │
│  H_i = SHA256(H_{i-1} ∥ Payload_i)         │
│  Web Crypto API / hashlib (FIPS 180-4)      │
└─────────────────────────────────────────────┘
```

---

## ML Pipeline

### 1. Feature Extraction

| Descriptor | Dims | Standard |
|---|---|---|
| Uniform LBP (8×8 spatial grid, 59 bins) | 3776 | ISO 19794-5 |
| Gabor bank (4 orientations × 2 frequencies × mean+std) | 16 | — |
| Appearance sub-sampling (24×24 + row means) | 600 | — |
| Facial geometry (dark-region centroids) | 15 | — |
| **Total** | **4407** | |

### 2. Online Welford Whitening

Incremental per-dimension standardisation using Welford's algorithm. Requires no training data, O(d) memory. Equivalent to dimension-wise PCA whitening without offline covariance decomposition.

### 3. Biometric Reference Score (BRS)

**Cosine similarity** (default): scale-invariant, suited for whitened feature spaces.  
**χ² distance** (alternative): histogram-aware, better for raw LBP distributions.

### 4. Platt Sigmoid Calibration *(v3.0)*

Maps raw BRS ∈ [0,1] to a proper posterior probability:

```
P(genuine | S) = 1 / (1 + exp(A·S + B))
```

A and B are fitted via 50-step mini-batch gradient descent on binary cross-entropy from session history. Falls back to isotonic (PAV) linear rescaling when fewer than 10 samples per class are available.

**Reference**: Platt (1999), *Probabilistic Outputs for Support Vector Machines*

### 5. Mahalanobis Multi-Template Scoring *(v3.0)*

Given N enrolled Biometric Reference Templates (BRTs) per subject:

```
S_corr = (0.65·S_peak + 0.35·S_mean) / (1 + β·σ_d)
```

where σ_d is the intra-class standard deviation of template-level scores and β=0.50. This penalises subjects with inconsistent enrollment captures while rewarding stable multi-pose BRT sets.

**Reference**: ISO 19795-1 §7.2 multi-sample fusion

### 6. Presentation Attack Detection (PAD)

ISO/IEC 30107-3 §6 passive PAD fusing five channels:

| Signal | Weight | Description |
|---|---|---|
| Inter-frame motion energy | 28% | Δ pixel mean across frame burst |
| MS-LBP texture entropy | 12% | Multi-scale (R∈{1,2,3}) LBP entropy |
| Laplacian sharpness | 20% | Variance of 3×3 Laplacian filter |
| Bilateral symmetry residual | 8% | Left–right pixel difference |
| Optical-flow arc curvature | 10% | Centroid trajectory non-linearity |
| Challenge-response IG | 30% | Information gain over uniform prior |

**PAD classes**: `L0 — Live Subject` · `L1 — Low Confidence` · `L2 — Artefact Detected`  
**Artefact types**: Static Artefact (Photo) · Video Replay · Printed (Halftone)

### 7. Adaptive EMA Weight Fusion *(v3.0)*

```
CDS = w_BRS · P_rec + w_PAD · P_pad
```

Weights adapt from per-subsystem FNMR rates:
- When BRS drives failures → decrease w_BRS, increase w_PAD
- When PAD drives failures → decrease w_PAD, increase w_BRS

Update rule: `w_BRS(t+1) = (1−α)·w_BRS(t) + α·target` where α=0.08.  
Weights clamped to [0.30, 0.80] to prevent degenerate single-modal fusion.

Also supports **Geometric Mean** (log-linear) and **Min-Gate** (strict-AND) modes.

### 8. Bayesian EER Estimator *(v3.0)*

Maintains a 200-sample sliding window of genuine/impostor scores. Sweeps 100 threshold candidates to locate the Equal Error Rate operating point (FMR ≈ FNMR). 

95% Bayesian credible interval using Jeffreys prior Beta(0.5, 0.5):

```
EER_CI₉₅ = μ ± 1.96 · √(αβ / ((α+β)²(α+β+1)))
```

**Reference**: ISO 19795-1 §8 — Performance Testing

### 9. SHA-256 Tamper-Evident Audit Chain

Every verification event is cryptographically chained:

```
H_i = SHA-256(H_{i-1} ‖ subject_id ‖ accepted ‖ CDS ‖ P(genuine) ‖ PAD_class ‖ timestamp)
```

Web implementation uses the **Web Crypto API** (`crypto.subtle.digest`).  
Python implementation uses `hashlib.sha256` (FIPS 180-4).  
Chain integrity can be verified on-demand.

---

## Quick Start

### Run the Web Dashboard

```bash
git clone https://github.com/Swayam26-rwt/SecureID.git
cd SecureID
python3 -m http.server 8080
# open http://localhost:8080
```

Or simply open `index.html` directly in Chrome — no build step required.

### Install the Python Package

```bash
pip install -e ".[dev]"   # editable install with test deps
```

### Basic Python API

```python
from biometric_engine import OfflineBiometricEngine
from biometric_engine.image_ops import synthetic_face

engine = OfflineBiometricEngine()

# Register 5 biometric reference templates (multi-pose)
for i in range(5):
    face = synthetic_face(eye_gap=30, noise=i)
    engine.enroll("alice", face)

# Verify with a live frame burst (passive PAD)
probe    = synthetic_face(eye_gap=30, noise=6)
liveness = [synthetic_face(eye_gap=30, noise=j, shift_x=j) for j in range(8)]
result   = engine.authenticate("alice", probe, liveness)

print(result.accepted)          # True / False
print(result.fused_score)       # CDS ∈ [0, 1]
print(result.calibrated_score)  # P(genuine | BRS) via Platt
print(result.pad_class)         # "Live Subject" | "Static Artefact" | …
print(result.explanation)       # ISO 30109 Decision Audit Trail entry
```

---

## Web Dashboard

The `index.html` dashboard is a full 4-tab enterprise operations interface:

| Tab | Content |
|---|---|
| **Identity Operations** | BAM sensor frame, SVG ring gauges (BRS/PAD/CDS), PAD badge, Decision Audit Trail |
| **Analytics & ROC** | Live ROC curve + AUC, FMR/FNMR/TMR KPIs, score distribution, threat breakdown |
| **ML Architecture** | 9 algorithm cards with standard references, live calibrator/weight status |
| **Audit Ledger** | SHA-256 hash chain, chain integrity verification, ISO-compliant JSON export |

**Keyboard shortcuts**: `⌘E` / `Ctrl+E` — Register BRT · `⌘↵` / `Ctrl+Enter` — Verify Identity

---

## Python API

### `OfflineBiometricEngine`

```python
engine.enroll(subject_id, face_crop)           # Register BRT (ISO 19794-5)
engine.authenticate(subject_id, probe, frames) # BRS + PAD + Platt + CDS
engine.identify(probe, frames)                 # 1:N identification
engine.export_audit_log()                      # JSON with EER, AUC, events
```

### `ScoreFusion` (ISO 30107-3 §8)

```python
from biometric_engine.ml_fusion import ScoreFusion, FusionConfig, PlattCalibrator

fusion = ScoreFusion(FusionConfig(
    fusion_method="weighted_sum",
    adaptive=True,
    adaptive_weight_lr=0.08,
))
result = fusion.fuse(recognition_score=0.82, liveness_score=0.74)
# result.accepted, .fused_score, .calibrated_score, .explanation

print(fusion.eer)                     # Equal Error Rate
print(fusion.eer_credible_interval)   # (lo, hi) Bayesian 95% CI
print(fusion.calibrator_status)       # "Platt (A=-3.82, B=1.94)"
print(fusion.stats)                   # {w_brs, w_pad, eer, eer_ci_lower, …}
```

### `LBPHFaceRecognizer`

```python
from biometric_engine.recognition import LBPHFaceRecognizer

rec = LBPHFaceRecognizer(metric="cosine", adaptive_threshold=True)
tmpl  = rec.create_template("alice", face_crop)
result = rec.identify(probe, [tmpl1, tmpl2, tmpl3])
# Uses Mahalanobis multi-template scoring internally

print(rec.adaptive_stats)             # {threshold, eer, fmr, fnmr, confidence_interval}
```

---

## Test Suite

```bash
python -m pytest tests/ -v
```

```
61 passed in ~14s
```

| Module | Tests |
|---|---|
| `test_image_ops.py` | 10 |
| `test_liveness.py` | 10 |
| `test_ml_fusion.py` | 23 |
| `test_recognition.py` | 12 |
| `test_offline_biometrics.py` | 5 |
| `test_session_analytics.py` | 1 |

---

## Standards Compliance

| Standard | Scope | Implementation |
|---|---|---|
| **ISO/IEC 30107-3** | Presentation Attack Detection | 5-channel passive PAD + challenge-response |
| **ISO/IEC 19795-1** | Biometric Performance Testing | FMR/FNMR/TMR metrics, ROC/AUC, EER |
| **ISO/IEC 19794-5** | Face Image Data Format | 96×96 normalised greyscale, multi-pose enrollment |
| **ISO 30109** | Decision Audit Trail (XAI) | Structured audit trail per verification event |
| **ISO 29794-1** | Biometric Sample Quality | IQS gating: blur × 0.6 + brightness × 0.4 |
| **NIST SP 800-76-2** | Biometric Specifications for PIV | Feature extraction, whitening, threshold standards |
| **FIPS 180-4** | SHA-2 Hash Standard | SHA-256 tamper-evident audit chain |
| **NIST IR 8427** | Score Calibration | Platt sigmoid + isotonic PAV calibration |

---

## Design Constraints

These constraints are **non-negotiable** and enforced in CI:

1. **Zero External Dependencies** — core Python engine runs on `math`, `typing`, `hashlib`, `collections` only
2. **Self-Contained Frontend** — `index.html` works via `file://` and static HTTP servers, no bundler needed
3. **Air-Gapped Operation** — no network calls, no telemetry, no cloud dependency
4. **Strict Typing** — all Python functions have complete type hints (`from __future__ import annotations`)
5. **Conventional Commits** — `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

```bash
git clone https://github.com/Swayam26-rwt/SecureID.git
cd SecureID
pip install -e ".[dev]"
python -m pytest tests/ -v          # verify baseline
python -m mypy biometric_engine/    # strict type checking
python -m ruff check biometric_engine/
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

---

<p align="center">
  <sub>
    SecureID v3.0 · ISO/IEC 30107-3 · NIST SP 800-76-2 · FIDO2 PAD L2 ·
    Built with Python standard library only · Zero external dependencies
  </sub>
</p>
