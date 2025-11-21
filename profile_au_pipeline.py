#!/usr/bin/env python3
"""
Comprehensive profiling script for Python AU pipeline.
Measures component-level performance and identifies bottlenecks.
"""

import cv2
import numpy as np
import time
import cProfile
import pstats
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import json
from contextlib import contextmanager
from collections import defaultdict

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


class ComponentProfiler:
    """Profile individual components of the AU pipeline."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.component_stack = []

    @contextmanager
    def measure(self, component: str):
        """Context manager to measure component timing."""
        start = time.perf_counter()
        self.component_stack.append(component)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[component].append(elapsed * 1000)  # Convert to ms
            self.component_stack.pop()

    def get_summary(self) -> Dict:
        """Get timing summary statistics."""
        summary = {}
        for component, times in self.timings.items():
            if times:
                summary[component] = {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'total_ms': np.sum(times),
                    'count': len(times),
                    'percentage': 0  # Will be calculated later
                }

        # Calculate percentages
        total_time = sum(s['mean_ms'] for s in summary.values()
                        if not any(sub in s for sub in ['total', 'frame']))
        for component in summary:
            if 'total' not in component and 'frame' not in component:
                summary[component]['percentage'] = (
                    summary[component]['mean_ms'] / total_time * 100
                    if total_time > 0 else 0
                )

        return summary


def profile_mtcnn_detection(detector, frame, profiler: ComponentProfiler):
    """Profile MTCNN face detection with detailed breakdown."""

    with profiler.measure('mtcnn_total'):
        # Preprocessing
        with profiler.measure('mtcnn_preprocess'):
            # MTCNN internally converts and scales
            pass

        # Detection (includes all 3 networks)
        with profiler.measure('mtcnn_detect'):
            detection = detector.detect(frame)

        # Postprocessing
        with profiler.measure('mtcnn_postprocess'):
            if detection is not None and isinstance(detection, tuple) and len(detection) == 2:
                bboxes, confidences = detection
                if len(bboxes) > 0:
                    bbox = bboxes[0]
                    x, y, w, h = [int(v) for v in bbox]
                    return (x, y, w, h), confidences[0]

    return None, None


def profile_clnf_fitting(clnf, frame, bbox, profiler: ComponentProfiler):
    """Profile CLNF landmark fitting with detailed breakdown."""

    with profiler.measure('clnf_total'):
        # Initialization from bbox
        with profiler.measure('clnf_init'):
            # CLNF.fit handles initialization internally
            pass

        # Iterative fitting
        with profiler.measure('clnf_fit'):
            landmarks, info = clnf.fit(frame, bbox)

        # Extract convergence info
        iterations = info.get('iterations', 0)
        converged = info.get('converged', False)

        return landmarks, {'iterations': iterations, 'converged': converged}

    return None, {}


def profile_au_prediction(au_pipeline, frame, profiler: ComponentProfiler):
    """Profile AU prediction with detailed breakdown."""

    with profiler.measure('au_total'):
        # HOG feature extraction
        with profiler.measure('au_hog_extraction'):
            # This happens inside process_frame
            pass

        # Geometry features
        with profiler.measure('au_geometry'):
            # This happens inside process_frame
            pass

        # AU prediction
        with profiler.measure('au_prediction'):
            result = au_pipeline._process_frame(frame)

        aus = result.get('aus', {}) if result else {}
        return aus

    return {}


def profile_full_pipeline(video_path: str, num_frames: int = 30):
    """Profile the complete AU pipeline on a video."""

    print("="*80)
    print("PYTHON AU PIPELINE PROFILER")
    print("="*80)
    print(f"\nTest video: {video_path}")
    print(f"Profiling {num_frames} frames...")

    # Initialize components
    print("\nInitializing pipeline components...")

    from pymtcnn import MTCNN
    from pyclnf import CLNF
    from pyfaceau import FullPythonAUPipeline

    # Time initialization
    init_start = time.perf_counter()

    detector = MTCNN()
    clnf = CLNF(model_dir="pyclnf/models")
    au_pipeline = FullPythonAUPipeline(
        pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
        au_models_dir="pyfaceau/weights/AU_predictors",
        triangulation_file="pyfaceau/weights/tris_68_full.txt",
        patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
        verbose=False
    )

    init_time = (time.perf_counter() - init_start) * 1000
    print(f"Initialization time: {init_time:.1f}ms")

    # Open video
    cap = cv2.VideoCapture(video_path)
    profiler = ComponentProfiler()

    # Profile frames
    frame_count = 0
    successful_frames = 0

    print(f"\nProfiling frames...")

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        with profiler.measure('frame_total'):
            # 1. Face Detection (MTCNN)
            bbox, confidence = profile_mtcnn_detection(detector, frame, profiler)

            if bbox is not None:
                # 2. Landmark Detection (CLNF)
                landmarks, clnf_info = profile_clnf_fitting(clnf, frame, bbox, profiler)

                if landmarks is not None and len(landmarks) == 68:
                    # 3. AU Prediction
                    aus = profile_au_prediction(au_pipeline, frame, profiler)

                    if aus:
                        successful_frames += 1

        # Progress indicator
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count}/{num_frames} frames...")

    cap.release()

    # Generate report
    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)

    summary = profiler.get_summary()

    # Component breakdown
    print("\nComponent Timing Breakdown:")
    print("-" * 50)

    components = [
        ('mtcnn_detect', 'MTCNN Detection'),
        ('clnf_fit', 'CLNF Landmark Fitting'),
        ('au_prediction', 'AU Prediction'),
    ]

    for comp_key, comp_name in components:
        if comp_key in summary:
            stats = summary[comp_key]
            print(f"\n{comp_name}:")
            print(f"  Mean: {stats['mean_ms']:.1f}ms ({stats['percentage']:.1f}%)")
            print(f"  Std:  {stats['std_ms']:.1f}ms")
            print(f"  Min:  {stats['min_ms']:.1f}ms")
            print(f"  Max:  {stats['max_ms']:.1f}ms")

    # Overall performance
    if 'frame_total' in summary:
        frame_stats = summary['frame_total']
        fps = 1000.0 / frame_stats['mean_ms'] if frame_stats['mean_ms'] > 0 else 0

        print("\n" + "="*50)
        print("Overall Performance:")
        print(f"  Average frame time: {frame_stats['mean_ms']:.1f}ms")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Successful frames: {successful_frames}/{frame_count}")

    # Identify bottlenecks
    print("\n" + "="*50)
    print("Performance Bottlenecks (sorted by impact):")

    bottlenecks = []
    for comp_key, comp_name in components:
        if comp_key in summary:
            bottlenecks.append((
                summary[comp_key]['mean_ms'],
                comp_name,
                summary[comp_key]['percentage']
            ))

    bottlenecks.sort(reverse=True)

    for i, (time_ms, name, pct) in enumerate(bottlenecks, 1):
        print(f"  {i}. {name}: {time_ms:.1f}ms ({pct:.1f}%)")

    # Save detailed results
    results_file = "profiling_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed results saved to {results_file}")

    # Optimization recommendations
    print("\n" + "="*50)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*50)

    # Determine primary bottleneck
    if bottlenecks:
        primary_bottleneck = bottlenecks[0][1]
        primary_time = bottlenecks[0][0]

        if "CLNF" in primary_bottleneck:
            print(f"\n🔥 Primary bottleneck: CLNF Landmark Fitting ({primary_time:.1f}ms)")
            print("Recommendations:")
            print("  1. Apply Numba JIT compilation to optimization loops")
            print("  2. Reduce iterations with better initialization")
            print("  3. Cache patch expert responses between similar frames")
            print("  4. Implement Cython version of core optimizer")

        elif "MTCNN" in primary_bottleneck:
            print(f"\n🔥 Primary bottleneck: MTCNN Detection ({primary_time:.1f}ms)")
            print("Recommendations:")
            print("  1. Convert to ONNX/CoreML for hardware acceleration")
            print("  2. Implement face tracking to skip detection")
            print("  3. Use smaller/faster detection model (YOLOv8-face)")
            print("  4. Batch process multiple scales in P-Net")

        elif "AU" in primary_bottleneck:
            print(f"\n🔥 Primary bottleneck: AU Prediction ({primary_time:.1f}ms)")
            print("Recommendations:")
            print("  1. Optimize HOG feature extraction")
            print("  2. Vectorize geometry feature computation")
            print("  3. Use batch prediction for multiple AUs")
            print("  4. Cache features for similar expressions")

    return summary


def profile_with_cprofile(video_path: str):
    """Use cProfile for detailed function-level profiling."""

    print("\nRunning cProfile analysis...")

    profiler = cProfile.Profile()
    profiler.enable()

    # Run the pipeline
    profile_full_pipeline(video_path, num_frames=10)

    profiler.disable()

    # Save and print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')

    print("\n" + "="*60)
    print("TOP 20 TIME-CONSUMING FUNCTIONS")
    print("="*60)
    stats.print_stats(20)

    # Save to file
    stats.dump_stats('pipeline_profile.stats')
    print("\nDetailed cProfile stats saved to pipeline_profile.stats")
    print("View with: python -m pstats pipeline_profile.stats")


def main():
    """Run comprehensive profiling."""

    video_path = "Patient Data/Normal Cohort/Shorty.mov"

    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Component-level profiling
    summary = profile_full_pipeline(video_path, num_frames=30)

    # Function-level profiling with cProfile
    print("\n" + "="*80)
    profile_with_cprofile(video_path)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review profiling_results.json for component timings")
    print("2. Analyze pipeline_profile.stats for function-level bottlenecks")
    print("3. Implement optimizations based on recommendations")
    print("4. Re-run profiling to measure improvements")


if __name__ == "__main__":
    main()