"""
Performance benchmark — Haar vs MediaPipe.

Metrics per backend:
  - Avg / min / max ms per frame
  - FPS
  - Detection rate (% frames with at least one face)
  - Landmark jitter (avg pixel displacement between consecutive detected frames)

Usage:
    python test/benchmark.py
    python test/benchmark.py --frames 200   # limit frame count
"""

import os
import sys
import time
import argparse
import statistics

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "video", "WIN_20260216_18_00_23_Pro.mp4")
LANDMARK_INDICES = [10, 234, 454, 61, 291, 164]

# ── frame loader ──────────────────────────────────────────────────────────────

def load_frames(path: str, max_frames: int | None = None) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ── jitter calculation ────────────────────────────────────────────────────────

def landmark_jitter(results: list, frames: list) -> float:
    """
    Average pixel displacement of each landmark between consecutive
    frames where a face was detected.
    """
    h, w = frames[0].shape[:2]
    prev_pts = None
    displacements = []

    for result in results:
        if not result.face_landmarks:
            prev_pts = None
            continue
        face = result.face_landmarks[0]
        pts = np.array(
            [[face[i].x * w, face[i].y * h] for i in LANDMARK_INDICES],
            dtype=np.float32,
        )
        if prev_pts is not None:
            dists = np.linalg.norm(pts - prev_pts, axis=1)
            displacements.append(float(np.mean(dists)))
        prev_pts = pts

    return statistics.mean(displacements) if displacements else 0.0


# ── single backend benchmark ──────────────────────────────────────────────────

def run_benchmark(backend_name: str, fm, frames: list) -> dict:
    times_ms = []
    results = []

    print(f"  running {backend_name}...", end="", flush=True)
    for frame in frames:
        t0 = time.perf_counter()
        result = fm.process(frame)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
        results.append(result)
    print(" done")

    detected = sum(1 for r in results if r.face_landmarks)

    return {
        "backend":        backend_name,
        "frames":         len(frames),
        "avg_ms":         statistics.mean(times_ms),
        "min_ms":         min(times_ms),
        "max_ms":         max(times_ms),
        "stdev_ms":       statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "fps":            1000 / statistics.mean(times_ms),
        "detected":       detected,
        "detection_rate": detected / len(frames) * 100,
        "jitter_px":      landmark_jitter(results, frames),
    }


# ── pretty table ──────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    col_w = 22
    metrics = [
        ("Frames tested",    lambda r: str(r["frames"])),
        ("Avg ms / frame",   lambda r: f"{r['avg_ms']:.2f} ms"),
        ("Min ms / frame",   lambda r: f"{r['min_ms']:.2f} ms"),
        ("Max ms / frame",   lambda r: f"{r['max_ms']:.2f} ms"),
        ("Std dev ms",       lambda r: f"{r['stdev_ms']:.2f} ms"),
        ("FPS",              lambda r: f"{r['fps']:.1f}"),
        ("Detected frames",  lambda r: f"{r['detected']} / {r['frames']}"),
        ("Detection rate",   lambda r: f"{r['detection_rate']:.1f}%"),
        ("Landmark jitter",  lambda r: f"{r['jitter_px']:.2f} px"),
    ]

    backends = [r["backend"] for r in results]
    header = f"{'Metric':<28}" + "".join(f"{b:>{col_w}}" for b in backends)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  CHURCOL — Face Detection Backend Benchmark")
    print("=" * len(header))
    print(header)
    print(sep)
    for label, fn in metrics:
        row = f"{label:<28}" + "".join(f"{fn(r):>{col_w}}" for r in results)
        print(row)
    print(sep)

    # winner callouts
    print()
    fastest = min(results, key=lambda r: r["avg_ms"])
    most_detected = max(results, key=lambda r: r["detection_rate"])
    smoothest = min(results, key=lambda r: r["jitter_px"])

    print(f"  ⚡ Fastest          : {fastest['backend']}  ({fastest['avg_ms']:.2f} ms avg)")
    print(f"  🎯 Most detections  : {most_detected['backend']}  ({most_detected['detection_rate']:.1f}%)")
    print(f"  📐 Least jitter     : {smoothest['backend']}  ({smoothest['jitter_px']:.2f} px)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face detection backend benchmark")
    parser.add_argument("--frames", type=int, default=None,
                        help="Max frames to test (default: all ~857)")
    args = parser.parse_args()

    print(f"\nLoading frames from video...", end="", flush=True)
    frames = load_frames(VIDEO_PATH, args.frames)
    print(f" {len(frames)} frames loaded")

    benchmark_results = []

    # ── Haar ─────────────────────────────────────────────────────────────────
    from core.backends.haar import FaceMesh as HaarFM
    haar_fm = HaarFM(max_faces=2)
    benchmark_results.append(run_benchmark("Haar", haar_fm, frames))
    haar_fm.close()

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    try:
        from core.backends.mediapipe_backend import FaceMesh as MPFM
        mp_fm = MPFM(max_faces=2)
        benchmark_results.append(run_benchmark("MediaPipe", mp_fm, frames))
        mp_fm.close()
    except ImportError:
        print("  MediaPipe not installed — skipping")

    print_table(benchmark_results)


if __name__ == "__main__":
    main()
