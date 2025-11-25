#!/usr/bin/env python3
"""
Benchmark CLNF optimizations - measure actual FPS improvement.

Tests the full CLNF pipeline with Numba JIT acceleration.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import time
from pathlib import Path


def benchmark_full_pipeline(num_frames=10):
    """Benchmark the full CLNF pipeline on video frames."""
    print("\n" + "="*60)
    print("CLNF Full Pipeline Benchmark")
    print("="*60)

    from pyclnf.clnf import CLNF

    # Load video
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    # Initialize CLNF
    print("\nInitializing CLNF...")
    t0 = time.perf_counter()
    clnf = CLNF('pyclnf/pyclnf/models', regularization=40)
    init_time = time.perf_counter() - t0
    print(f"  Initialization time: {init_time*1000:.1f}ms")

    # Read frames
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * 10)  # Sample every 10th frame
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    print(f"\nLoaded {len(frames)} frames for testing")

    # Warmup run (JIT compilation)
    print("\nWarmup run (JIT compilation)...")
    if frames:
        _ = clnf.detect_and_fit(frames[0])

    # Benchmark runs
    print(f"\nBenchmarking {len(frames)} frames...")
    times = []
    successful = 0

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        result = clnf.detect_and_fit(frame)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if result is not None and result[0] is not None:
            successful += 1

        # Print progress
        fps = 1.0 / elapsed if elapsed > 0 else 0
        print(f"  Frame {i+1}: {elapsed*1000:.1f}ms ({fps:.1f} FPS)")

    # Statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_fps = 1.0 / avg_time

    print("\n" + "-"*60)
    print("Results:")
    print("-"*60)
    print(f"  Successful detections: {successful}/{len(frames)}")
    print(f"  Average time: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
    print(f"  Min time: {min_time*1000:.1f}ms")
    print(f"  Max time: {max_time*1000:.1f}ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Best FPS: {1.0/min_time:.1f}")

    return avg_fps


def benchmark_components():
    """Benchmark individual CLNF components."""
    print("\n" + "="*60)
    print("Component-Level Benchmarks")
    print("="*60)

    from pyclnf.core.pdm import PDM
    from pyclnf.core.patch_expert import CCNFPatchExpert
    from pyclnf.core.optimizer import NURLMSOptimizer

    # Load models
    pdm = PDM('pyclnf/pyclnf/models/exported_pdm')
    patch_expert = CCNFPatchExpert('pyclnf/pyclnf/models/exported_ccnf_0.25/view_00/patch_30')
    optimizer = NURLMSOptimizer()

    # Generate test data
    np.random.seed(42)
    params = pdm.init_params((100, 100, 200, 250))
    test_patch = np.random.randint(0, 256, (patch_expert.height, patch_expert.width), dtype=np.uint8)
    response_map = np.random.rand(11, 11).astype(np.float64)

    n_iter = 1000

    # Benchmark Jacobian
    print("\n1. Jacobian Computation")
    t0 = time.perf_counter()
    for _ in range(n_iter):
        J = pdm.compute_jacobian(params)
    jac_time = (time.perf_counter() - t0) / n_iter * 1000
    print(f"   Time per call: {jac_time:.3f}ms")
    print(f"   Calls per second: {1000/jac_time:.0f}")

    # Benchmark Patch Response
    print("\n2. Patch Expert Response")
    t0 = time.perf_counter()
    for _ in range(n_iter):
        resp = patch_expert.compute_response(test_patch)
    patch_time = (time.perf_counter() - t0) / n_iter * 1000
    print(f"   Time per call: {patch_time:.3f}ms")
    print(f"   Calls per second: {1000/patch_time:.0f}")

    # Benchmark KDE Mean-Shift
    print("\n3. KDE Mean-Shift")
    a = -0.5 / (1.75 * 1.75)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ms_x, ms_y = optimizer._kde_mean_shift(response_map, 5.5, 5.5, a)
    kde_time = (time.perf_counter() - t0) / n_iter * 1000
    print(f"   Time per call: {kde_time:.3f}ms")
    print(f"   Calls per second: {1000/kde_time:.0f}")

    # Estimate optimization iteration time
    # Per iteration: 68 landmarks × (patch response + KDE) + Jacobian
    n_landmarks = 68
    window_size = 11
    positions_per_landmark = window_size * window_size  # 121

    patch_time_per_iter = n_landmarks * positions_per_landmark * patch_time
    kde_time_per_iter = n_landmarks * kde_time
    jac_time_per_iter = jac_time

    total_iter_time = patch_time_per_iter + kde_time_per_iter + jac_time_per_iter

    print("\n4. Estimated Optimization Iteration Time")
    print(f"   Patch responses ({n_landmarks}×{positions_per_landmark}): {patch_time_per_iter:.1f}ms")
    print(f"   KDE mean-shift ({n_landmarks}×): {kde_time_per_iter:.1f}ms")
    print(f"   Jacobian: {jac_time_per_iter:.3f}ms")
    print(f"   Total per iteration: {total_iter_time:.1f}ms")

    # With 3 window sizes × ~3 iterations each = ~9 iterations
    n_iterations = 9
    total_optimization_time = total_iter_time * n_iterations
    print(f"\n   Estimated total optimization ({n_iterations} iters): {total_optimization_time:.1f}ms")


def main():
    print("#"*60)
    print("# CLNF Optimization Benchmark Suite")
    print("#"*60)

    # Component benchmarks
    benchmark_components()

    # Full pipeline benchmark
    avg_fps = benchmark_full_pipeline(num_frames=5)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if avg_fps:
        print(f"\nAchieved FPS: {avg_fps:.1f}")

        # Compare to baseline (estimated ~5 FPS without optimization)
        baseline_fps = 5.0
        speedup = avg_fps / baseline_fps
        print(f"Estimated speedup vs baseline: {speedup:.1f}x")

        if avg_fps >= 20:
            print("\n✓ Target achieved: Real-time capable (≥20 FPS)")
        elif avg_fps >= 10:
            print("\n◐ Good progress: Near real-time (10-20 FPS)")
        else:
            print("\n○ Further optimization needed (<10 FPS)")


if __name__ == '__main__':
    main()
