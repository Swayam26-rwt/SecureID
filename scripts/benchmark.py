#!/usr/bin/env python3
"""Edge performance benchmark script for NHAI Datalake 3.0 Biometrics."""

from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datalake_offline_biometrics import OfflineBiometricEngine, synthetic_face


def run(iterations: int = 50) -> None:
    print("=" * 60)
    print("NHAI Datalake 3.0 Biometrics — Edge Hardware Benchmark")
    print("=" * 60)

    engine = OfflineBiometricEngine()
    face_1 = synthetic_face(eye_gap=30, mouth_curve=0, noise=2)
    face_2 = synthetic_face(eye_gap=30, mouth_curve=1, shift_x=1, noise=2)
    burst = [synthetic_face(shift_x=i % 2, blink=(i == 4), noise=2) for i in range(10)]

    # Enrollment
    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.enroll("operator-bench", [face_1, face_2])
    t_enroll = (time.perf_counter() - t0) / iterations * 1000.0

    # Verification
    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.authenticate(
            subject_id="operator-bench",
            face_crop=face_1,
            liveness_frames=burst,
            challenge=["blink"],
        )
    t_auth = (time.perf_counter() - t0) / iterations * 1000.0

    print(f"Iterations:             {iterations}")
    print(f"Enrollment Latency:     {t_enroll:.2f} ms")
    print(f"Authentication Latency: {t_auth:.2f} ms")
    print(f"Verification Rate:      {1000.0 / t_auth:.1f} ops/sec")
    print("=" * 60)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    run(n)
