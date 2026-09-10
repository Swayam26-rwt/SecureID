# NHAI Datalake 3.0 Offline Biometrics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#capabilities)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#running-tests)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-orange.svg)](https://swayam26-rwt.github.io/NHAI/)

Lightweight, zero-network facial recognition and liveness detection engine engineered for **National Highways Authority of India (NHAI) Datalake 3.0** edge terminals, toll plazas, and remote operational facilities.

This package is intentionally **100% dependency-free**. It accepts cropped grayscale face matrices from camera hardware, extracts compact local binary pattern histogram (LBPH) templates, evaluates identity similarity locally, and assesses multi-factor liveness from short burst sequences using motion variance, blink/gesture challenge response, and spatial texture entropy.

---

## Capabilities

- **Strictly Offline**: Air-gapped design with zero cloud telemetry, no model downloads, and no external API reliance.
- **Pure Standard Library**: Zero third-party dependencies required for runtime execution (`math`, `typing`, `collections`, `hmac`).
- **Privacy by Design**: Persists compact mathematical templates (~1.2 KB), immediately discarding raw camera frames from memory.
- **Unified Engine**: Single `OfflineBiometricEngine` handles enrollment, verification, 1:N identification, multi-factor liveness, and HMAC signature verification.
- **Self-Contained Web Demo**: Full client-side browser simulator (`index.html`) executable directly via `file://` or static web hosting.

---

## Live Interactive Demo

The repository contains a complete browser-based demonstration in `index.html` with modern UI aesthetics, dual dark/light themes, real-time synthetic vector rendering, and instant scenario testing:

- **Verification Pass** (Matched identity + valid liveness)
- **Spoof Rejection** (Static image presentation attack caught)
- **Mismatch Rejection** (Unregistered operator rejected)
- **Live Audit Log** (Rolling tamper-evident operational history)

Try it directly on [GitHub Pages](https://swayam26-rwt.github.io/NHAI/) or open `index.html` in any modern browser.

---

## Quick Start

### Installation

```bash
git clone https://github.com/Swayam26-rwt/NHAI.git
cd NHAI
pip install -e .
```

### Python API

```python
from datalake_offline_biometrics import OfflineBiometricEngine

# Initialize the engine
engine = OfflineBiometricEngine()

# Enroll an authorized operator with sample face crops
engine.enroll("operator-42", face_crops=[face_crop_1, face_crop_2])

# Authenticate with probe crop and liveness burst
auth = engine.authenticate(
    subject_id="operator-42",
    face_crop=probe_face_crop,
    liveness_frames=burst_of_face_crops,
    challenge=["blink"],
)

if auth.accepted:
    print(f"Access granted! Match similarity: {auth.recognition.similarity:.2f}")
else:
    print(f"Access denied. Reasons: {', '.join(auth.reasons)}")
```

---

## Repository Structure

```
NHAI/
├── assets/
│   ├── app.js                           # Modernized browser biometrics simulator
│   └── styles.css                       # High-performance design system with dark mode
├── datalake_offline_biometrics/
│   ├── __init__.py                      # Package exports & public API
│   ├── engine.py                        # Central OfflineBiometricEngine coordinator
│   ├── image_ops.py                     # Bilinear resize, histogram equalization, clamping
│   ├── liveness.py                      # Multi-frame motion, texture entropy, and challenges
│   ├── recognition.py                   # LBPH feature extraction & chi-square matching
│   └── storage.py                       # HMAC-SHA256 authenticated template persistence
├── docs/
│   └── offline_biometrics_integration.md # Architecture specs & edge deployment guide
├── tests/
│   └── test_offline_biometrics.py       # Comprehensive unit test suite
├── index.html                           # Accessible, responsive biometrics interface
├── pyproject.toml                       # Python package configuration (PEP 621)
├── CONTRIBUTING.md                      # Developer guidelines
├── SECURITY.md                          # Vulnerability reporting & edge privacy
└── LICENSE                              # MIT License
```

---

## Running Tests

Execute the built-in test suite:

```bash
python3 -m unittest discover -s tests
```

---

## Documentation

For full deployment guides, edge hardware benchmarks, and threshold tuning details, refer to [docs/offline_biometrics_integration.md](docs/offline_biometrics_integration.md).
