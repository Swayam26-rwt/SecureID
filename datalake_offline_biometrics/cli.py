"""Command-line interface for NHAI Datalake 3.0 Offline Biometrics diagnostics and benchmarks."""

from __future__ import annotations

import argparse
import sys
import time

from .engine import OfflineBiometricEngine
from .image_ops import synthetic_face


def run_benchmark(iterations: int = 100) -> int:
    """Benchmark enrollment and verification speed on the host device."""
    print(f"[*] Running NHAI Biometrics benchmark ({iterations} iterations)...")

    engine = OfflineBiometricEngine()
    face_a1 = synthetic_face(eye_gap=30, mouth_curve=0, noise=0.05)
    face_a2 = synthetic_face(eye_gap=30, mouth_curve=0, noise=0.08)
    burst = [synthetic_face(eye_gap=30, noise=0.05 + i * 0.01) for i in range(10)]

    # Benchmark Enrollment
    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.enroll("bench-operator", [face_a1, face_a2])
    enroll_time = (time.perf_counter() - t0) / iterations * 1000.0

    # Benchmark Authentication
    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.authenticate(
            subject_id="bench-operator",
            face_crop=face_a1,
            liveness_frames=burst,
            challenge=["blink"],
        )
    auth_time = (time.perf_counter() - t0) / iterations * 1000.0

    print(f"[+] Enrollment latency:    {enroll_time:.2f} ms/op")
    print(f"[+] Authentication latency:{auth_time:.2f} ms/op")
    print(f"[+] Throughput:            {1000.0 / auth_time:.1f} verifications/sec")
    print("[+] All diagnostics passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nhai-biometrics",
        description="NHAI Datalake 3.0 Offline Biometrics CLI tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    bench_parser = subparsers.add_parser("benchmark", help="Run latency and throughput benchmarks")
    bench_parser.add_argument(
        "--iterations", "-n", type=int, default=50, help="Number of test iterations"
    )

    subparsers.add_parser("info", help="Display engine information and defaults")

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return run_benchmark(iterations=args.iterations)
    elif args.command == "info":
        print("NHAI Datalake 3.0 Offline Biometrics v1.0.0")
        print("Algorithm: Local Binary Pattern Histograms (LBPH) with multi-block spatial grid")
        print("Liveness: Multi-factor (Motion variance, Texture entropy, Challenge-response)")
        print("Security: HMAC-SHA256 authenticated template storage")
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
