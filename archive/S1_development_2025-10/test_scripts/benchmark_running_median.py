#!/usr/bin/env python3
"""
Benchmark Running Median: Python vs Cython

Tests performance of histogram-based running median tracker.
Expected speedup: 10-20x with Cython

This is a CRITICAL bottleneck - runs on every frame for 4702 features.
"""

import numpy as np
import time
from histogram_median_tracker import DualHistogramMedianTracker

try:
    from cython_histogram_median import DualHistogramMedianTrackerCython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("⚠ Cython histogram median not available")

print("=" * 80)
print("Running Median Performance Benchmark")
print("=" * 80)

# Configuration (matching AU extraction pipeline)
HOG_DIM = 4464
GEOM_DIM = 238
NUM_FRAMES = 1000  # Simulated video frames

print(f"\nTest configuration:")
print(f"  HOG features: {HOG_DIM}")
print(f"  Geometric features: {GEOM_DIM}")
print(f"  Total features: {HOG_DIM + GEOM_DIM} = 4702")
print(f"  Frames: {NUM_FRAMES}")
print(f"  Updates: {NUM_FRAMES // 2} (every 2nd frame)")

# Generate synthetic data
np.random.seed(42)
hog_data = np.random.randn(NUM_FRAMES, HOG_DIM).astype(np.float32)
geom_data = np.random.randn(NUM_FRAMES, GEOM_DIM).astype(np.float32)

print("\n" + "=" * 80)
print("PYTHON IMPLEMENTATION")
print("=" * 80)

# Benchmark Python implementation
tracker_python = DualHistogramMedianTracker(
    hog_dim=HOG_DIM,
    geom_dim=GEOM_DIM,
    hog_bins=200,
    geom_bins=200
)

start_time = time.time()

for i in range(NUM_FRAMES):
    update_hist = (i % 2 == 1)  # Update every 2nd frame
    tracker_python.update(hog_data[i], geom_data[i], update_histogram=update_hist)

    # Get median periodically (like AU extraction does)
    if i % 10 == 0:
        hog_median = tracker_python.get_hog_median()
        geom_median = tracker_python.get_geom_median()

python_time = time.time() - start_time

print(f"\nTime: {python_time:.4f} seconds")
print(f"Time per frame: {python_time / NUM_FRAMES * 1000:.2f} ms")
print(f"Frames per second: {NUM_FRAMES / python_time:.1f} FPS")

if CYTHON_AVAILABLE:
    print("\n" + "=" * 80)
    print("CYTHON IMPLEMENTATION")
    print("=" * 80)

    # Benchmark Cython implementation
    tracker_cython = DualHistogramMedianTrackerCython(
        hog_dim=HOG_DIM,
        geom_dim=GEOM_DIM,
        hog_bins=200,
        geom_bins=200
    )

    start_time = time.time()

    for i in range(NUM_FRAMES):
        update_hist = (i % 2 == 1)  # Update every 2nd frame
        tracker_cython.update(hog_data[i], geom_data[i], update_histogram=update_hist)

        # Get median periodically
        if i % 10 == 0:
            hog_median = tracker_cython.get_hog_median()
            geom_median = tracker_cython.get_geom_median()

    cython_time = time.time() - start_time

    print(f"\nTime: {cython_time:.4f} seconds")
    print(f"Time per frame: {cython_time / NUM_FRAMES * 1000:.2f} ms")
    print(f"Frames per second: {NUM_FRAMES / cython_time:.1f} FPS")

    # Calculate speedup
    speedup = python_time / cython_time

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\nPython:  {python_time:.4f}s ({python_time / NUM_FRAMES * 1000:.2f} ms/frame)")
    print(f"Cython:  {cython_time:.4f}s ({cython_time / NUM_FRAMES * 1000:.2f} ms/frame)")
    print(f"\nSpeedup: {speedup:.2f}x faster with Cython 🚀")

    # Time savings
    time_saved = python_time - cython_time
    percent_saved = (time_saved / python_time) * 100

    print(f"\nTime saved: {time_saved:.2f}s ({percent_saved:.1f}%)")

    # Real-world impact
    video_length = 60  # seconds
    fps = 30
    total_frames = video_length * fps

    python_process_time = (python_time / NUM_FRAMES) * total_frames
    cython_process_time = (cython_time / NUM_FRAMES) * total_frames

    print(f"\nReal-world impact (60-second video @ 30fps = {total_frames} frames):")
    print(f"  Python: {python_process_time:.1f} seconds")
    print(f"  Cython: {cython_process_time:.1f} seconds")
    print(f"  Savings: {python_process_time - cython_process_time:.1f} seconds per video")

    # Assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if speedup >= 15:
        print(f"\n✅ EXCELLENT: {speedup:.1f}x speedup exceeds 15x target!")
    elif speedup >= 10:
        print(f"\n✅ GOOD: {speedup:.1f}x speedup meets 10x target")
    elif speedup >= 5:
        print(f"\n⚠ MODERATE: {speedup:.1f}x speedup (target was 10x)")
    else:
        print(f"\n❌ LOW: {speedup:.1f}x speedup (expected 10-20x)")

else:
    print("\n" + "=" * 80)
    print("⚠ Cython not available - cannot benchmark")
    print("=" * 80)

print("\n" + "=" * 80)
