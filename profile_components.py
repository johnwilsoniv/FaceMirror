#!/usr/bin/env python3
"""
Simple component-level profiling for the Python AU pipeline.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys
from collections import defaultdict
import json

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def profile_pipeline_simple(video_path: str, num_frames: int = 20):
    """Profile each component of the pipeline separately."""

    print("="*80)
    print("COMPONENT-LEVEL PROFILING")
    print("="*80)

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau import FullPythonAUPipeline

    # Component timings
    timings = {
        'detection': [],
        'landmarks': [],
        'au_prediction': [],
        'total': []
    }

    # Initialize
    print("\nInitializing components...")
    detector = MTCNN()
    clnf = CLNF(model_dir="pyclnf/models")

    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    print(f"\nProcessing {num_frames} frames...")

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        total_start = time.perf_counter()

        # 1. Face Detection
        det_start = time.perf_counter()
        detection = detector.detect(frame)
        det_time = (time.perf_counter() - det_start) * 1000
        timings['detection'].append(det_time)

        if detection is not None and isinstance(detection, tuple) and len(detection) == 2:
            bboxes, confidences = detection
            if len(bboxes) > 0:
                bbox = bboxes[0]
                x, y, w, h = [int(v) for v in bbox]

                # 2. Landmark Detection
                lm_start = time.perf_counter()
                landmarks, info = clnf.fit(frame, (x, y, w, h))
                lm_time = (time.perf_counter() - lm_start) * 1000
                timings['landmarks'].append(lm_time)

                # 3. AU Prediction (simplified - just HOG extraction)
                au_start = time.perf_counter()

                # Simulate AU prediction time (HOG + geometry + prediction)
                # In reality, we'd call the full AU pipeline here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                face_roi = gray[y:y+h, x:x+w]

                # Resize to standard size for HOG
                if face_roi.size > 0:
                    face_resized = cv2.resize(face_roi, (112, 112))

                    # HOG feature extraction (simplified)
                    from skimage.feature import hog
                    hog_features = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=False)

                au_time = (time.perf_counter() - au_start) * 1000
                timings['au_prediction'].append(au_time)

        total_time = (time.perf_counter() - total_start) * 1000
        timings['total'].append(total_time)

        if frame_count % 5 == 0:
            print(f"  Frame {frame_count}: {total_time:.1f}ms")

    cap.release()

    # Calculate statistics
    print("\n" + "="*60)
    print("TIMING STATISTICS (ms)")
    print("="*60)

    results = {}
    for component, times in timings.items():
        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)

            print(f"\n{component.upper()}:")
            print(f"  Mean:  {mean_time:.1f}ms")
            print(f"  Std:   {std_time:.1f}ms")
            print(f"  Min:   {min_time:.1f}ms")
            print(f"  Max:   {max_time:.1f}ms")

            results[component] = {
                'mean': mean_time,
                'std': std_time,
                'min': min_time,
                'max': max_time
            }

    # Calculate percentages
    if results and 'total' in results:
        total_mean = results['total']['mean']
        print("\n" + "="*60)
        print("COMPONENT BREAKDOWN")
        print("="*60)

        breakdown = []
        for comp in ['detection', 'landmarks', 'au_prediction']:
            if comp in results:
                pct = (results[comp]['mean'] / total_mean) * 100
                breakdown.append((results[comp]['mean'], comp, pct))

        breakdown.sort(reverse=True)
        for time_ms, name, pct in breakdown:
            print(f"  {name:15s}: {time_ms:6.1f}ms ({pct:5.1f}%)")

    # Performance summary
    if results and 'total' in results:
        avg_fps = 1000.0 / results['total']['mean']
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Average frame time: {results['total']['mean']:.1f}ms")

        # Compare to targets
        print(f"\n  vs OpenFace C++ (99ms): {results['total']['mean']/99:.1f}x slower")
        print(f"  vs Real-time (33ms): {results['total']['mean']/33:.1f}x slower")

    # Save results
    with open('component_timings.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to component_timings.json")

    return results


def identify_bottlenecks(results):
    """Identify and prioritize optimization targets."""

    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    if not results:
        return

    # Find biggest bottleneck
    components = ['detection', 'landmarks', 'au_prediction']
    bottlenecks = []

    for comp in components:
        if comp in results:
            bottlenecks.append((results[comp]['mean'], comp))

    bottlenecks.sort(reverse=True)

    if bottlenecks:
        primary = bottlenecks[0]
        print(f"\n🔥 PRIMARY BOTTLENECK: {primary[1]} ({primary[0]:.1f}ms)")

        if primary[1] == 'landmarks':
            print("\nOptimization strategies for CLNF:")
            print("  1. Apply Numba JIT to hot loops")
            print("  2. Reduce iterations (currently 5-10)")
            print("  3. Cache patch expert computations")
            print("  4. Use Cython for core optimizer")
            print("  Expected improvement: 2-3x speedup")

        elif primary[1] == 'detection':
            print("\nOptimization strategies for MTCNN:")
            print("  1. Convert to ONNX/CoreML")
            print("  2. Skip detection on consecutive frames")
            print("  3. Use lighter model (YOLOv8-face)")
            print("  4. Batch process pyramid levels")
            print("  Expected improvement: 3-5x speedup")

        elif primary[1] == 'au_prediction':
            print("\nOptimization strategies for AU:")
            print("  1. Optimize HOG extraction")
            print("  2. Batch predict all AUs")
            print("  3. Use vectorized operations")
            print("  4. Cache features for similar faces")
            print("  Expected improvement: 2x speedup")

        if len(bottlenecks) > 1:
            print(f"\n📊 SECONDARY BOTTLENECK: {bottlenecks[1][1]} ({bottlenecks[1][0]:.1f}ms)")


def main():
    """Run component profiling."""

    video_path = "Patient Data/Normal Cohort/Shorty.mov"

    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Profile components
    results = profile_pipeline_simple(video_path, num_frames=20)

    # Identify optimization targets
    identify_bottlenecks(results)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()