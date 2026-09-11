# NHAI Datalake 3.0 — Offline Biometric Authentication Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **Capstone ML Project — Industry-Level Biometric Authentication System for Edge Deployments.**
> All computation runs entirely offline using pure Python standard library (zero third-party packages).

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  OfflineBiometricEngine (facade)                     │
│                                                                     │
│  ┌──────────────────┐   ┌──────────────────┐   ┌───────────────┐   │
│  │ LBPHFaceRecognizer│   │ LivenessDetector  │   │  ScoreFusion  │   │
│  │                  │   │                  │   │               │   │
│  │ • Multi-scale LBP│   │ • MS-LBP entropy │   │ • Weighted sum│   │
│  │ • Gabor filter   │   │ • Optical flow   │   │ • Geometric   │   │
│  │ • LPQ descriptor │   │ • Face symmetry  │   │ • Min pool    │   │
│  │ • Cosine metric  │   │ • Attack classify│   │ • Adaptive EER│   │
│  │ • PCA whitening  │   │ • Blink/nod/smile│   │ • ROC curve   │   │
│  │ • Adaptive EER   │   │                  │   │               │   │
│  └──────────────────┘   └──────────────────┘   └───────────────┘   │
│          │                        │                       │         │
│          └────────────────────────┴───────────────────────┘         │
│                                   │                                 │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                    SessionAnalytics                         │   │
│   │  FAR │ FRR │ TAR │ Score distribution │ JSON audit export  │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                   │                                 │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │              TemplateStore (PBKDF2-HMAC-SHA256)             │   │
│   │  v2 format │ Random salt │ Template clustering │ v1 migration│  │
│   └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ v2.0.0 Feature Highlights

| Module | Feature | Details |
|--------|---------|---------|
| `image_ops` | **Gabor filter bank** | 4 orientations × 2 frequencies; blur-robust |
| `image_ops` | **LPQ descriptor** | 256-bin STFT phase histogram, blur-invariant |
| `image_ops` | **CLAHE equalization** | 8×8 tile adaptive histogram, bilinear interp |
| `image_ops` | **Quality metrics** | Blur, brightness, occlusion, combined score |
| `recognition` | **Cosine similarity** | Scale-invariant, whitened feature space |
| `recognition` | **Online PCA whitening** | Welford incremental, zero training data |
| `recognition` | **Adaptive EER threshold** | 200-sample window, 50-step sweep |
| `liveness` | **Multi-scale LBP** | Radii 1, 2, 3 combined entropy |
| `liveness` | **Optical flow arc** | Centroid trajectory curvature analysis |
| `liveness` | **Face symmetry** | Bilateral pixel-level comparison |
| `liveness` | **Attack classification** | static / printed / replay / genuine |
| `ml_fusion` | **Score-level fusion** | Weighted sum / geometric mean / min |
| `ml_fusion` | **ROC curve + EER** | From sliding window of session scores |
| `analytics` | **Session analytics** | FAR, FRR, TAR, attack breakdown, JSON export |
| `engine` | **XAI explain()** | 7+ line human-readable decision trace |
| `engine` | **Audit chain** | SHA-256 hash chaining per authentication |
| `storage` | **PBKDF2-HMAC** | 100k iterations + random 16-byte salt |
| `storage` | **Template clustering** | Element-wise centroid per subject |

---

## 🚀 Quick Start

### Python (Core ML Engine)

```python
from datalake_offline_biometrics import OfflineBiometricEngine
from datalake_offline_biometrics.image_ops import synthetic_face

# Initialize engine with advanced ML features
engine = OfflineBiometricEngine()

# Enroll subject with quality gating
engine.enroll("officer-001", [
    synthetic_face(eye_gap=30, noise=2),
    synthetic_face(eye_gap=30, noise=3, shift_x=1),
])

# Multi-modal authentication
probe_face = synthetic_face(eye_gap=30, noise=3)
liveness_frames = [synthetic_face(shift_x=i, noise=2) for i in range(5)]

result = engine.authenticate(
    "officer-001",
    probe_face,
    liveness_frames,
    challenge=["blink"],
)

print(f"Accepted: {result.accepted}")
print(f"Fused score: {result.fusion.fused_score:.4f}")
print(f"Attack hint: {result.liveness.attack_type_hint}")
print("\n".join(result.explanation))  # XAI trace
```

### Browser Simulator

Open `index.html` in any modern browser — zero server required.

Features:
- Live face capture simulation with scanline animation
- Recognition + liveness + fusion scores with animated meters
- Attack simulation (static photo / screen replay / impostor)
- 5 challenge modes: blink, turn left/right, nod, smile
- Analytics tab: FAR/FRR/TAR KPIs, score distribution histogram, timeline
- XAI decision trace with colour-coded explanations
- Cryptographic audit log (SHA-256 chain via Web Crypto API)
- Export session log as JSON

---

## 📁 Repository Structure

```
NHAI-main/
├── datalake_offline_biometrics/
│   ├── __init__.py          # v2.0.0 package exports
│   ├── image_ops.py         # Gabor, LPQ, CLAHE, quality metrics
│   ├── recognition.py       # LBPHFaceRecognizer, PCA whitening, adaptive EER
│   ├── liveness.py          # Multi-scale LBP, optical flow, symmetry, attack hint
│   ├── ml_fusion.py         # Score-level fusion, ROC curve, EER estimation
│   ├── analytics.py         # Session FAR/FRR/TAR, attack breakdown, export
│   ├── engine.py            # Facade: fusion, batch enroll, XAI, audit chain
│   └── storage.py           # PBKDF2-HMAC, v2 format, template clustering
├── tests/
│   ├── test_recognition.py  # 13 cases: cosine, whitening, adaptive threshold
│   ├── test_liveness.py     # 11 cases: attack hints, optical flow, challenges
│   ├── test_ml_fusion.py    # 13 cases: fusion, ROC, EER, analytics
│   └── test_offline_biometrics.py   # 24 integration cases
├── scripts/
│   └── benchmark.py         # ML algorithm benchmark with ROC/EER export
├── assets/
│   ├── styles.css           # Premium dark-mode design system (800+ lines)
│   └── app.js               # Full ML port to JS + analytics + Web Crypto audit
├── index.html               # 3-tab dashboard: Demo | Analytics | Architecture
└── CHANGELOG.md             # Full v2.0.0 change log
```

---

## 🧪 Running Tests

```bash
# Run all 61 tests
python -m pytest tests/ -v

# Run benchmark (quick mode)
python scripts/benchmark.py --quick

# Run benchmark and export results
python scripts/benchmark.py --export benchmark_results.json
```

---

## 📊 Algorithm Details

### Multi-Scale LBP + Gabor Fusion

The recognition pipeline combines:
1. **Uniform LBP** (59-bin, radius 1) with 8×8 spatial grid → 3,776-dim vector
2. **Gabor filter bank** (4 orientations × 2 frequencies, mean + variance) → 16-dim
3. **Appearance features** (24×24 downsampled, row/column means) → 624-dim
4. **Geometry features** (dark-region centroids for eyes, mouth, nose) → 12-dim

Total dimensionality: **~4,428 dims**, post-whitened to zero-mean unit-variance.

### Adaptive EER Threshold

```
Genuine scores:  [0.85, 0.82, 0.88, ...]  → mean ~0.85
Impostor scores: [0.32, 0.28, 0.35, ...]  → mean ~0.31

Sweep 50 thresholds t ∈ [0.28, 0.88]:
  For each t: FAR = |impostor >= t| / |impostor|
              FRR = |genuine < t|  / |genuine|
  EER = argmin |FAR - FRR|  →  typically t ≈ 0.58
```

### Score-Level Fusion

| Method | Formula |
|--------|---------|
| `weighted_sum` | `0.60 × rec + 0.40 × live` |
| `weighted_product` | `exp(0.60 × log(rec) + 0.40 × log(live))` |
| `min` | `min(rec, live)` |

---

## 🔒 Security Properties

| Property | Implementation |
|----------|---------------|
| Template integrity | PBKDF2-HMAC-SHA256 (100k iterations, random 16-byte salt) |
| Audit tamper-evidence | SHA-256 hash chaining (blockchain-style) |
| Zero cloud dependency | All ML runs locally on device |
| No training data required | Online PCA whitening + adaptive threshold |

---

## 📄 License

MIT License — see `LICENSE` file.

---

*Built as a capstone ML project demonstrating industry-level biometric system design
using zero third-party dependencies. All algorithms implemented from first principles
using Python standard library only.*
