#!/usr/bin/env python3
"""
Detailed profiling script for Python AU pipeline with component-level timing.
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
import io

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


class ComponentTimer:
    """High-precision component timer with hierarchical tracking."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.stack = []
        self.current_parent = None

    @contextmanager
    def measure(self, name: str):
        """Time a component."""
        start = time.perf_counter()
        parent = self.current_parent
        self.current_parent = name
        self.stack.append((name, start, parent))

        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.timings[name].append(elapsed)

            # Restore parent
            self.stack.pop()
            if self.stack:
                self.current_parent = self.stack[-1][0]
            else:
                self.current_parent = None


def profile_complete_pipeline(video_path: str, num_frames: int = 30):
    """Profile the complete AU pipeline with detailed breakdowns."""

    print("="*80)
    print("DETAILED AU PIPELINE PROFILER")
    print("="*80)
    print(f"\nVideo: {video_path}")
    print(f"Frames to profile: {num_frames}")

    timer = ComponentTimer()

    # Initialize components
    print("\n[Initialization]")

    with timer.measure('initialization'):
        from pymtcnn import MTCNN
        from pyclnf import CLNF
        from pyfaceau.prediction.au_predictor import AUPredictor
        from pyfaceau.alignment.face_aligner import FaceAligner
        from pyfaceau.features.hog_extractor import HOGFeatureExtractor
        from pyfaceau.features.geometry_features import GeometryFeatureExtractor
        from pyfaceau.alignment.pose_estimator import PoseEstimator

        with timer.measure('mtcnn_init'):
            detector = MTCNN()
            print("  ✓ MTCNN initialized")

        with timer.measure('clnf_init'):
            clnf = CLNF(model_dir="pyclnf/models")
            print("  ✓ CLNF initialized")

        with timer.measure('au_components_init'):
            # Initialize AU prediction components
            pdm_file = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
            au_models_dir = "pyfaceau/weights/AU_predictors"
            triangulation_file = "pyfaceau/weights/tris_68_full.txt"

            pose_estimator = PoseEstimator(pdm_file)
            face_aligner = FaceAligner()
            hog_extractor = HOGFeatureExtractor()
            geometry_extractor = GeometryFeatureExtractor(triangulation_file)
            au_predictor = AUPredictor(au_models_dir)
            print("  ✓ AU components initialized")

    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    successful_frames = 0

    print(f"\n[Processing {num_frames} frames]")

    # Enable cProfile for detailed function analysis
    profiler = cProfile.Profile()
    profiler.enable()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        with timer.measure('frame_total'):
            # 1. Face Detection
            with timer.measure('face_detection'):
                with timer.measure('mtcnn_detect'):
                    detection = detector.detect(frame)

                if detection is not None and isinstance(detection, tuple) and len(detection) == 2:
                    bboxes, confidences = detection
                    if len(bboxes) > 0:
                        bbox = bboxes[0]
                        x, y, w, h = [int(v) for v in bbox]
                        face_detected = True
                    else:
                        face_detected = False
                else:
                    face_detected = False

            if face_detected:
                # 2. Landmark Detection
                with timer.measure('landmark_detection'):
                    with timer.measure('clnf_init_bbox'):
                        # CLNF initialization from bbox
                        pass

                    with timer.measure('clnf_fit'):
                        landmarks, info = clnf.fit(frame, (x, y, w, h))

                    with timer.measure('clnf_postprocess'):
                        converged = info.get('converged', False)
                        iterations = info.get('iterations', 0)

                if landmarks is not None and len(landmarks) == 68:
                    # 3. Pose Estimation
                    with timer.measure('pose_estimation'):
                        pose = pose_estimator.estimate_pose(landmarks)

                    # 4. Face Alignment
                    with timer.measure('face_alignment'):
                        aligned_face = face_aligner.align_face(frame, landmarks, pose)

                    # 5. Feature Extraction
                    with timer.measure('feature_extraction'):
                        with timer.measure('hog_extraction'):
                            hog_features = hog_extractor.extract(aligned_face)

                        with timer.measure('geometry_extraction'):
                            geometry_features = geometry_extractor.extract(landmarks, pose)

                    # 6. AU Prediction
                    with timer.measure('au_prediction'):
                        with timer.measure('au_prepare_features'):
                            # Combine features
                            features = np.concatenate([hog_features, geometry_features])

                        with timer.measure('au_svm_predict'):
                            aus = au_predictor.predict(features)

                    successful_frames += 1

        # Progress update
        if frame_count % 5 == 0:
            avg_time = np.mean(timer.timings['frame_total']) if timer.timings['frame_total'] else 0
            print(f"  Frame {frame_count:3d}: {avg_time:.1f}ms avg")

    profiler.disable()
    cap.release()

    # Generate timing report
    print("\n" + "="*60)
    print("COMPONENT TIMING ANALYSIS")
    print("="*60)

    # Calculate statistics for each component
    stats = {}
    for component, times in timer.timings.items():
        if times:
            stats[component] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }

    # Major components
    major_components = [
        ('face_detection', 'Face Detection (MTCNN)'),
        ('landmark_detection', 'Landmark Detection (CLNF)'),
        ('pose_estimation', 'Pose Estimation'),
        ('face_alignment', 'Face Alignment'),
        ('feature_extraction', 'Feature Extraction'),
        ('au_prediction', 'AU Prediction'),
    ]

    print("\n[Major Components]")
    for key, name in major_components:
        if key in stats:
            s = stats[key]
            pct = (s['mean'] / stats['frame_total']['mean'] * 100) if 'frame_total' in stats else 0
            print(f"\n{name}:")
            print(f"  Mean: {s['mean']:7.1f}ms ({pct:5.1f}%)")
            print(f"  Std:  {s['std']:7.1f}ms")
            print(f"  Min:  {s['min']:7.1f}ms")
            print(f"  Max:  {s['max']:7.1f}ms")

    # Sub-components
    print("\n[Sub-Components]")
    sub_components = [
        ('mtcnn_detect', 'MTCNN Network Forward'),
        ('clnf_fit', 'CLNF Optimization'),
        ('hog_extraction', 'HOG Features'),
        ('geometry_extraction', 'Geometry Features'),
        ('au_svm_predict', 'SVM Prediction'),
    ]

    for key, name in sub_components:
        if key in stats:
            s = stats[key]
            print(f"  {name:25s}: {s['mean']:6.1f}ms (±{s['std']:.1f})")

    # Overall performance
    if 'frame_total' in stats:
        frame_stats = stats['frame_total']
        fps = 1000.0 / frame_stats['mean']

        print("\n" + "="*60)
        print("OVERALL PERFORMANCE")
        print("="*60)
        print(f"  Frames processed: {frame_count}")
        print(f"  Successful frames: {successful_frames}")
        print(f"  Average frame time: {frame_stats['mean']:.1f}ms")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  vs OpenFace C++ (10.1 FPS): {fps/10.1:.1f}x speed")
        print(f"  vs Real-time (30 FPS): {fps/30.0:.1f}x speed")

    # cProfile function analysis
    print("\n" + "="*60)
    print("TOP TIME-CONSUMING FUNCTIONS")
    print("="*60)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(15)

    print(s.getvalue())

    # Identify bottlenecks
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)

    bottlenecks = []
    for key, name in major_components:
        if key in stats:
            bottlenecks.append((stats[key]['mean'], name, key))

    bottlenecks.sort(reverse=True)

    print("\nComponents by time (sorted):")
    for i, (time_ms, name, key) in enumerate(bottlenecks[:3], 1):
        pct = (time_ms / stats['frame_total']['mean'] * 100) if 'frame_total' in stats else 0
        print(f"  {i}. {name}: {time_ms:.1f}ms ({pct:.1f}%)")

    # Save results
    with open('detailed_profiling_results.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n✅ Detailed results saved to detailed_profiling_results.json")

    # Optimization recommendations
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    if bottlenecks:
        primary = bottlenecks[0]
        print(f"\n🔥 Primary Bottleneck: {primary[1]} ({primary[0]:.1f}ms)")

        if 'landmark' in primary[2].lower() or 'clnf' in primary[2].lower():
            print("\nRecommended optimizations for CLNF:")
            print("  1. Numba JIT compilation for response map computation")
            print("  2. Reduce iterations from 5-10 to 2-3")
            print("  3. Cache patch experts between frames")
            print("  4. Cython implementation of optimizer")
            print("  Expected improvement: 2-3x speedup")

        elif 'detection' in primary[2].lower() or 'mtcnn' in primary[2].lower():
            print("\nRecommended optimizations for MTCNN:")
            print("  1. ONNX/CoreML hardware acceleration")
            print("  2. Face tracking to skip detection")
            print("  3. Batch pyramid processing")
            print("  4. Replace with YOLOv8-face")
            print("  Expected improvement: 3-5x speedup")

    return stats


def main():
    """Run detailed profiling."""

    video_path = "Patient Data/Normal Cohort/Shorty.mov"

    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Run detailed profiling
    stats = profile_complete_pipeline(video_path, num_frames=20)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review detailed_profiling_results.json")
    print("2. Implement recommended optimizations")
    print("3. Re-run to measure improvements")


if __name__ == "__main__":
    main()