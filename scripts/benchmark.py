"""Hardware latency, throughput, and ML algorithm comparison benchmark.

v2.0.0: Extended with Gabor vs LPQ throughput comparison, cosine vs weighted
metric latency, ROC curve JSON export, and per-algorithm feature dimension report.

Usage:
    python scripts/benchmark.py [--quick] [--export results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from datalake_offline_biometrics.image_ops import (
    blur_score,
    gabor_features,
    image_quality_score,
    lpq_features,
    synthetic_face,
)
from datalake_offline_biometrics.liveness import LivenessDetector
from datalake_offline_biometrics.ml_fusion import FusionConfig, ScoreFusion
from datalake_offline_biometrics.recognition import LBPHFaceRecognizer


def timeit(fn, *, n: int = 5) -> tuple[float, float]:
    """Return (mean_ms, min_ms) over n runs."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), min(times)


def bench_feature_extraction(n: int) -> dict[str, object]:
    """Benchmark LBP+Gabor vs LBP-only vs LPQ extraction."""
    face = synthetic_face(size=96)

    recognizer_gabor = LBPHFaceRecognizer(use_gabor=True, use_lpq=False)
    recognizer_base = LBPHFaceRecognizer(use_gabor=False, use_lpq=False)

    gabor_mean, gabor_min = timeit(lambda: recognizer_gabor.extract(face), n=n)
    base_mean, base_min = timeit(lambda: recognizer_base.extract(face), n=n)

    lpq_mean, lpq_min = timeit(lambda: lpq_features(face), n=max(2, n // 3))

    vec_gabor = recognizer_gabor.extract(face)
    vec_base = recognizer_base.extract(face)

    return {
        "lbph_only": {
            "mean_ms": round(base_mean, 2),
            "min_ms": round(base_min, 2),
            "feature_dim": len(vec_base),
        },
        "lbph_gabor": {
            "mean_ms": round(gabor_mean, 2),
            "min_ms": round(gabor_min, 2),
            "feature_dim": len(vec_gabor),
            "overhead_pct": round((gabor_mean - base_mean) / max(1, base_mean) * 100, 1),
        },
        "lpq_standalone": {
            "mean_ms": round(lpq_mean, 2),
            "min_ms": round(lpq_min, 2),
            "feature_dim": 256,
        },
    }


def bench_matching_metrics(n: int) -> dict[str, object]:
    """Compare cosine vs weighted distance similarity latency."""
    face_a = synthetic_face(eye_gap=30, noise=1)
    face_b = synthetic_face(eye_gap=20, mouth_curve=8, noise=1)

    rec_cosine = LBPHFaceRecognizer(metric="cosine")
    rec_weighted = LBPHFaceRecognizer(metric="weighted")

    vec_a = rec_cosine.extract(face_a)
    vec_b = rec_cosine.extract(face_b)

    cos_mean, cos_min = timeit(lambda: rec_cosine.similarity(vec_a, vec_b), n=n)
    wt_mean, wt_min = timeit(lambda: rec_weighted.distance(vec_a, vec_b), n=n)

    return {
        "cosine_similarity": {"mean_ms": round(cos_mean, 3), "min_ms": round(cos_min, 3)},
        "weighted_distance": {"mean_ms": round(wt_mean, 3), "min_ms": round(wt_min, 3)},
    }


def bench_liveness(n: int) -> dict[str, object]:
    """Benchmark full liveness assessment."""
    detector = LivenessDetector()
    frames = [synthetic_face(shift_x=k, noise=2) for k in range(5)]
    mean_ms, min_ms = timeit(lambda: detector.assess(frames, challenge=["blink"]), n=n)
    return {
        "frames": 5,
        "mean_ms": round(mean_ms, 2),
        "min_ms": round(min_ms, 2),
    }


def bench_quality_metrics(n: int) -> dict[str, object]:
    """Benchmark image quality scoring pipeline."""
    face = synthetic_face(size=96)
    mean_ms, min_ms = timeit(lambda: image_quality_score(face), n=n)
    scores = image_quality_score(face)
    return {
        "mean_ms": round(mean_ms, 2),
        "min_ms": round(min_ms, 2),
        "sample_scores": scores,
    }


def roc_export(fusion: ScoreFusion, steps: int = 100) -> list[dict[str, float]]:
    """Generate a sample ROC curve from synthetic genuine/impostor scores."""
    import random
    random.seed(42)
    for _ in range(50):
        fusion.record_outcome(0.72 + random.gauss(0, 0.08), is_genuine=True)
        fusion.record_outcome(0.38 + random.gauss(0, 0.10), is_genuine=False)
    return fusion.roc_curve(steps=steps)


def run(*, quick: bool = False) -> dict[str, object]:
    n = 3 if quick else 8

    print("─" * 60)
    print("  NHAI Datalake 3.0 — ML Algorithm Benchmark Suite v2.0.0")
    print("─" * 60)

    print("\n[1/5] Feature extraction latency …")
    fe = bench_feature_extraction(n)
    for name, data in fe.items():
        print(f"  {name:<22}  {data['mean_ms']:>8.2f} ms  dim={data.get('feature_dim', '—')}")

    print("\n[2/5] Matching metric latency …")
    mm = bench_matching_metrics(n * 5)
    for name, data in mm.items():
        print(f"  {name:<22}  {data['mean_ms']:>8.3f} ms")

    print("\n[3/5] Full liveness assessment …")
    lv = bench_liveness(n)
    print(f"  {lv['frames']}-frame burst  {lv['mean_ms']:>8.2f} ms avg")

    print("\n[4/5] Image quality scoring …")
    qm = bench_quality_metrics(n)
    print(f"  quality_score():  {qm['mean_ms']:>8.2f} ms  sample={qm['sample_scores']}")

    print("\n[5/5] ROC curve (synthetic scores) …")
    fusion = ScoreFusion()
    roc = roc_export(fusion)
    eer = fusion.eer
    print(f"  EER ≈ {eer:.4f}  (from {len(roc)} ROC points)")
    if roc:
        eer_point = min(roc, key=lambda p: abs(p["far"] - p["frr"]))
        print(f"  EER point: threshold={eer_point['threshold']}, FAR={eer_point['far']}, FRR={eer_point['frr']}")

    print("\n" + "─" * 60)

    return {
        "feature_extraction": fe,
        "matching_metrics": mm,
        "liveness": lv,
        "quality_metrics": qm,
        "roc_curve": roc,
        "eer": eer,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHAI biometrics ML benchmark")
    parser.add_argument("--quick", action="store_true", help="Fewer iterations for CI")
    parser.add_argument("--export", metavar="FILE", help="Export results to JSON file")
    args = parser.parse_args()

    results = run(quick=args.quick)

    if args.export:
        Path(args.export).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults exported → {args.export}")
