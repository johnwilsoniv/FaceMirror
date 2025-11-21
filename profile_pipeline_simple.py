#!/usr/bin/env python3
"""
Simple but effective profiling of the Python AU pipeline.
Profiles the same way the comparison script runs it.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys
import json
from collections import defaultdict
import cProfile
import pstats
import io

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


def profile_pipeline():
    """Profile the pipeline as it's actually used."""

    print("="*80)
    print("PYTHON AU PIPELINE PROFILER")
    print("="*80)

    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Initialize components
    print("\nInitializing pipeline components...")
    init_start = time.perf_counter()

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau import FullPythonAUPipeline

    detector = MTCNN()
    clnf = CLNF(model_dir="pyclnf/models")
    au_pipeline = FullPythonAUPipeline(
        pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
        au_models_dir="pyfaceau/weights/AU_predictors",
        triangulation_file="pyfaceau/weights/tris_68_full.txt",
        patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
        verbose=False
    )

    init_time = time.perf_counter() - init_start
    print(f"Initialization completed in {init_time:.2f}s")

    # Component timing storage
    timings = defaultdict(list)

    # Process frames
    cap = cv2.VideoCapture(video_path)
    num_frames = 30
    frame_count = 0

    print(f"\nProfiling {num_frames} frames...")
    print("-" * 60)

    # Start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_start = time.perf_counter()

        # 1. Face Detection (MTCNN)
        det_start = time.perf_counter()
        detection = detector.detect(frame)
        det_time = (time.perf_counter() - det_start) * 1000
        timings['mtcnn_detection'].append(det_time)

        if detection is not None and isinstance(detection, tuple) and len(detection) == 2:
            bboxes, confidences = detection
            if len(bboxes) > 0:
                bbox = bboxes[0]
                x, y, w, h = [int(v) for v in bbox]

                # 2. Landmark Fitting (CLNF)
                lm_start = time.perf_counter()
                landmarks, info = clnf.fit(frame, (x, y, w, h))
                lm_time = (time.perf_counter() - lm_start) * 1000
                timings['clnf_fitting'].append(lm_time)

                # Record CLNF details
                timings['clnf_iterations'].append(info.get('iterations', 0))
                timings['clnf_converged'].append(1 if info.get('converged', False) else 0)

                if landmarks is not None and len(landmarks) == 68:
                    # 3. AU Prediction (full pipeline)
                    au_start = time.perf_counter()
                    try:
                        # This is how compare_au_accuracy.py does it
                        au_result = au_pipeline._process_frame(
                            frame,
                            frame_idx=frame_count,
                            timestamp=frame_count/30.0
                        )
                        aus = au_result.get('aus', {}) if au_result else {}
                    except:
                        aus = {}
                    au_time = (time.perf_counter() - au_start) * 1000
                    timings['au_prediction'].append(au_time)

        # Total frame time
        frame_time = (time.perf_counter() - frame_start) * 1000
        timings['frame_total'].append(frame_time)

        # Progress update
        if frame_count % 5 == 0:
            print(f"Frame {frame_count:3d}: {frame_time:6.1f}ms "
                  f"(Det: {det_time:5.1f}ms, "
                  f"LM: {timings['clnf_fitting'][-1] if timings['clnf_fitting'] else 0:5.1f}ms, "
                  f"AU: {timings['au_prediction'][-1] if timings['au_prediction'] else 0:5.1f}ms)")

    profiler.disable()
    cap.release()

    # Analysis
    print("\n" + "="*60)
    print("TIMING ANALYSIS")
    print("="*60)

    # Component statistics
    components = [
        ('mtcnn_detection', 'MTCNN Face Detection'),
        ('clnf_fitting', 'CLNF Landmark Fitting'),
        ('au_prediction', 'AU Prediction'),
        ('frame_total', 'Total Frame Time')
    ]

    results = {}
    for key, name in components:
        if key in timings and timings[key]:
            times = timings[key]
            mean = np.mean(times)
            std = np.std(times)
            min_val = np.min(times)
            max_val = np.max(times)

            print(f"\n{name}:")
            print(f"  Mean: {mean:7.1f}ms")
            print(f"  Std:  {std:7.1f}ms")
            print(f"  Min:  {min_val:7.1f}ms")
            print(f"  Max:  {max_val:7.1f}ms")

            results[key] = {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val
            }

    # CLNF convergence stats
    if timings['clnf_iterations']:
        print(f"\nCLNF Convergence:")
        print(f"  Avg iterations: {np.mean(timings['clnf_iterations']):.1f}")
        print(f"  Convergence rate: {np.mean(timings['clnf_converged'])*100:.1f}%")

    # Breakdown percentages
    if 'frame_total' in results:
        total_mean = results['frame_total']['mean']
        print("\n" + "="*60)
        print("COMPONENT BREAKDOWN")
        print("="*60)

        breakdown = []
        for key in ['mtcnn_detection', 'clnf_fitting', 'au_prediction']:
            if key in results:
                pct = (results[key]['mean'] / total_mean) * 100
                breakdown.append((results[key]['mean'], key, pct))

        breakdown.sort(reverse=True)
        for time_ms, key, pct in breakdown:
            name = key.replace('_', ' ').title()
            print(f"  {name:20s}: {time_ms:6.1f}ms ({pct:5.1f}%)")

    # Performance summary
    if 'frame_total' in results:
        fps = 1000.0 / results['frame_total']['mean']
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average frame time: {results['frame_total']['mean']:.1f}ms")
        print(f"  vs OpenFace C++ (10.1 FPS): {fps/10.1:.2f}x")
        print(f"  vs Real-time (30 FPS): {fps/30.0:.2f}x")

    # Function-level profiling
    print("\n" + "="*60)
    print("TOP 15 TIME-CONSUMING FUNCTIONS")
    print("="*60)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())

    # Bottleneck identification
    print("\n" + "="*60)
    print("PRIMARY BOTTLENECK")
    print("="*60)

    if breakdown:
        primary = breakdown[0]
        print(f"\n🔥 {primary[1].replace('_', ' ').title()}: {primary[0]:.1f}ms ({primary[2]:.1f}%)")

        if 'clnf' in primary[1]:
            print("\nOptimization strategies:")
            print("  1. Apply Numba JIT to response map computation")
            print("  2. Reduce iterations (currently {:.1f})".format(
                np.mean(timings['clnf_iterations']) if timings['clnf_iterations'] else 5))
            print("  3. Cache patch experts between frames")
            print("  4. Implement Cython optimizer")
            print("  Expected speedup: 2-3x")

        elif 'mtcnn' in primary[1]:
            print("\nOptimization strategies:")
            print("  1. Convert to ONNX/CoreML for acceleration")
            print("  2. Implement face tracking")
            print("  3. Use lighter detector (YOLOv8-face)")
            print("  Expected speedup: 3-5x")

        elif 'au' in primary[1]:
            print("\nOptimization strategies:")
            print("  1. Optimize HOG extraction")
            print("  2. Batch AU predictions")
            print("  3. Cache features")
            print("  Expected speedup: 1.5-2x")

    # Save results
    with open('profiling_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to profiling_results.json")

    return results


def main():
    """Run profiling."""
    results = profile_pipeline()

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Implement optimizations for the primary bottleneck")
    print("2. Re-run profiling to measure improvement")
    print("3. Move to next bottleneck")


if __name__ == "__main__":
    main()