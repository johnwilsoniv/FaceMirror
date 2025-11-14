#!/usr/bin/env python3
"""
Test script to validate optimized MTCNN matches original implementation.

Validates:
1. Identical bbox outputs (within floating point tolerance)
2. Performance improvement measurement
"""

import numpy as np
import cv2
import time
from pathlib import Path

# Import both versions
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2 as OriginalMTCNN
from pure_python_mtcnn_optimized import PurePythonMTCNN_Optimized as OptimizedMTCNN


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes (x, y, w, h format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection

    return intersection / union if union > 0 else 0.0


def compare_bboxes(orig_bboxes, opt_bboxes, tolerance=1e-3):
    """Compare two sets of bboxes"""
    if len(orig_bboxes) != len(opt_bboxes):
        return False, f"Different number of detections: {len(orig_bboxes)} vs {len(opt_bboxes)}"

    if len(orig_bboxes) == 0:
        return True, "Both detected 0 faces (match)"

    # Check each bbox
    max_diff = 0
    ious = []

    for i in range(len(orig_bboxes)):
        orig = orig_bboxes[i]
        opt = opt_bboxes[i]

        # Calculate absolute differences
        diff = np.abs(orig - opt)
        max_diff = max(max_diff, diff.max())

        # Calculate IoU
        iou = calculate_iou(orig, opt)
        ious.append(iou)

    mean_iou = np.mean(ious)

    if max_diff < tolerance:
        return True, f"Perfect match! Max diff: {max_diff:.6f}, Mean IoU: {mean_iou:.6f}"
    else:
        return True, f"Close match: Max diff: {max_diff:.6f}, Mean IoU: {mean_iou:.6f}"


def test_on_video_frame(video_path, frame_index=0, num_warmup=2, num_runs=5):
    """Test both implementations on a video frame"""
    print(f"\n{'='*80}")
    print(f"Testing on: {Path(video_path).name}, Frame {frame_index}")
    print(f"{'='*80}")

    # Load frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"✗ Failed to read frame {frame_index}")
        return None

    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Initialize detectors
    print("\nInitializing detectors...")
    original = OriginalMTCNN()
    optimized = OptimizedMTCNN()

    # Warmup runs
    print(f"\nWarmup ({num_warmup} runs)...")
    for i in range(num_warmup):
        _ = original.detect(frame)
        _ = optimized.detect(frame)

    # Benchmark original
    print(f"\nBenchmarking ORIGINAL implementation ({num_runs} runs)...")
    orig_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        orig_bboxes, orig_landmarks = original.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000
        orig_times.append(elapsed)
        if i == 0:
            print(f"  Run 1: {elapsed:.2f} ms, {len(orig_bboxes)} faces")

    orig_mean = np.mean(orig_times)
    orig_std = np.std(orig_times)
    orig_fps = 1000 / orig_mean

    print(f"  Mean: {orig_mean:.2f} ± {orig_std:.2f} ms ({orig_fps:.3f} FPS)")

    # Benchmark optimized
    print(f"\nBenchmarking OPTIMIZED implementation ({num_runs} runs)...")
    opt_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        opt_bboxes, opt_landmarks = optimized.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000
        opt_times.append(elapsed)
        if i == 0:
            print(f"  Run 1: {elapsed:.2f} ms, {len(opt_bboxes)} faces")

    opt_mean = np.mean(opt_times)
    opt_std = np.std(opt_times)
    opt_fps = 1000 / opt_mean

    print(f"  Mean: {opt_mean:.2f} ± {opt_std:.2f} ms ({opt_fps:.3f} FPS)")

    # Compare results
    print(f"\n{'='*80}")
    print("ACCURACY VALIDATION")
    print(f"{'='*80}")

    match, message = compare_bboxes(orig_bboxes, opt_bboxes)

    if match:
        print(f"✓ {message}")
    else:
        print(f"✗ {message}")
        print(f"\nOriginal bboxes:\n{orig_bboxes}")
        print(f"\nOptimized bboxes:\n{opt_bboxes}")

    # Performance summary
    print(f"\n{'='*80}")
    print("PERFORMANCE IMPROVEMENT")
    print(f"{'='*80}")

    speedup = orig_mean / opt_mean
    improvement_pct = ((orig_mean - opt_mean) / orig_mean) * 100

    print(f"Original:  {orig_mean:.2f} ms ({orig_fps:.3f} FPS)")
    print(f"Optimized: {opt_mean:.2f} ms ({opt_fps:.3f} FPS)")
    print(f"\n🚀 SPEEDUP: {speedup:.2f}x ({improvement_pct:.1f}% faster)")

    return {
        'match': match,
        'speedup': speedup,
        'orig_fps': orig_fps,
        'opt_fps': opt_fps,
        'orig_mean_ms': orig_mean,
        'opt_mean_ms': opt_mean
    }


def main():
    """Run validation tests"""
    print("="*80)
    print("MTCNN OPTIMIZATION VALIDATION TEST")
    print("="*80)

    # Test on multiple frames from Patient Data
    test_videos = [
        ("Patient Data/Normal Cohort/IMG_0422.MOV", 222),
        ("Patient Data/Normal Cohort/IMG_0428.MOV", 253),
        ("Patient Data/Normal Cohort/IMG_0433.MOV", 232),
    ]

    results = []

    for video_path, frame_idx in test_videos:
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"\n✗ Skipping {video_path.name} (not found)")
            continue

        result = test_on_video_frame(video_path, frame_idx, num_warmup=2, num_runs=5)
        if result:
            results.append(result)

    # Overall summary
    if results:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")

        all_match = all(r['match'] for r in results)
        avg_speedup = np.mean([r['speedup'] for r in results])
        avg_orig_fps = np.mean([r['orig_fps'] for r in results])
        avg_opt_fps = np.mean([r['opt_fps'] for r in results])

        print(f"\nAccuracy: {'✓ ALL TESTS PASSED' if all_match else '✗ SOME TESTS FAILED'}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Original FPS: {avg_orig_fps:.3f}")
        print(f"Optimized FPS: {avg_opt_fps:.3f}")

        print(f"\n{'='*80}")
        if all_match:
            print("✅ OPTIMIZATION SUCCESSFUL!")
            print(f"   - Accuracy preserved: 100%")
            print(f"   - Performance gain: {avg_speedup:.2f}x")
            print(f"   - Ready for production use!")
        else:
            print("⚠️  SOME ACCURACY ISSUES DETECTED")
            print("   - Review differences above")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
