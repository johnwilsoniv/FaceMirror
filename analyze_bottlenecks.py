#!/usr/bin/env python3
"""
Analyze which components are the bottlenecks in the pipeline.
"""

import time
import sys
from pathlib import Path
import numpy as np
import cv2
import warnings

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))

warnings.filterwarnings('ignore')


def profile_components():
    """Profile each component separately."""

    print("=" * 60)
    print("COMPONENT-LEVEL PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Load test video
    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)

    test_frames = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        test_frames.append(frame)
    cap.release()

    print(f"\nTesting with {len(test_frames)} frames")

    # Initialize components
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        from pymtcnn import MTCNN
        from pyclnf import CLNF
        from pyfaceau import FullPythonAUPipeline

        detector = MTCNN()
        clnf = CLNF(
            model_dir="pyclnf/models",
            max_iterations=5,
            convergence_threshold=0.5,
            debug_mode=False
        )
        au_pipeline = FullPythonAUPipeline(
            pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
            au_models_dir="pyfaceau/weights/AU_predictors",
            triangulation_file="pyfaceau/weights/tris_68_full.txt",
            patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
            verbose=False
        )
    finally:
        sys.stdout = old_stdout

    # Profile each component
    detection_times = []
    landmark_times = []
    au_times = []
    total_times = []

    print("\nProcessing frames...")
    print("-" * 40)

    for i, frame in enumerate(test_frames):
        total_start = time.perf_counter()

        # 1. Detection
        detect_start = time.perf_counter()
        detection = detector.detect(frame)
        detect_time = (time.perf_counter() - detect_start) * 1000
        detection_times.append(detect_time)

        if detection and isinstance(detection, tuple) and len(detection) == 2:
            bboxes, _ = detection
            if len(bboxes) > 0:
                bbox = bboxes[0]
                x, y, w, h = [int(v) for v in bbox]
                bbox = (x, y, w, h)

                # 2. Landmarks
                landmark_start = time.perf_counter()
                landmarks, _ = clnf.fit(frame, bbox)
                landmark_time = (time.perf_counter() - landmark_start) * 1000
                landmark_times.append(landmark_time)

                # 3. AU Prediction
                au_start = time.perf_counter()
                result = au_pipeline._process_frame(frame, i, i/30.0)
                au_time = (time.perf_counter() - au_start) * 1000
                au_times.append(au_time)

        total_time = (time.perf_counter() - total_start) * 1000
        total_times.append(total_time)

        if i == 0 or i == len(test_frames) - 1:
            print(f"Frame {i}: Detection={detect_time:.0f}ms, Landmarks={landmark_times[-1]:.0f}ms, AU={au_times[-1]:.0f}ms")

    # Calculate statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE BREAKDOWN")
    print("=" * 60)

    avg_detection = np.mean(detection_times)
    avg_landmark = np.mean(landmark_times)
    avg_au = np.mean(au_times)
    avg_total = np.mean(total_times)

    print(f"\n{'Component':<20} {'Avg Time':<12} {'% of Total':<12} {'FPS if only this':<20}")
    print("-" * 60)

    components = [
        ("Detection (MTCNN)", avg_detection, avg_detection/avg_total*100, 1000/avg_detection),
        ("Landmarks (CLNF)", avg_landmark, avg_landmark/avg_total*100, 1000/avg_landmark),
        ("AU Prediction", avg_au, avg_au/avg_total*100, 1000/avg_au),
        ("TOTAL", avg_total, 100, 1000/avg_total)
    ]

    for name, time_ms, percent, fps in components:
        print(f"{name:<20} {time_ms:<12.1f}ms {percent:<12.1f}% {fps:<20.1f} FPS")

    # Identify bottlenecks
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)

    bottlenecks = sorted(components[:-1], key=lambda x: x[1], reverse=True)

    print("\nTop bottlenecks:")
    for i, (name, time_ms, percent, _) in enumerate(bottlenecks, 1):
        print(f"{i}. {name}: {time_ms:.1f}ms ({percent:.1f}%)")
        if i == 1:
            print(f"   → Primary bottleneck - focus optimization here!")

    print("\nOptimization potential:")
    print("-" * 40)

    # Calculate potential speedup
    if bottlenecks[0][1] > 100:  # If bottleneck > 100ms
        potential_reduction = bottlenecks[0][1] * 0.7  # Assume 70% reduction possible
        new_total = avg_total - potential_reduction
        potential_fps = 1000 / new_total
        print(f"If {bottlenecks[0][0]} reduced by 70%:")
        print(f"  New total time: {new_total:.1f}ms")
        print(f"  Potential FPS: {potential_fps:.1f}")

    # Detailed breakdown
    print("\n" + "=" * 60)
    print("DETAILED TIMING STATISTICS")
    print("=" * 60)

    for name, times in [("Detection", detection_times),
                        ("Landmarks", landmark_times),
                        ("AU Prediction", au_times)]:
        if times:
            print(f"\n{name}:")
            print(f"  Min: {np.min(times):.1f}ms")
            print(f"  Max: {np.max(times):.1f}ms")
            print(f"  Mean: {np.mean(times):.1f}ms")
            print(f"  Std: {np.std(times):.1f}ms")

    return {
        'detection': avg_detection,
        'landmarks': avg_landmark,
        'au_prediction': avg_au,
        'total': avg_total
    }


if __name__ == "__main__":
    results = profile_components()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    print("\nBased on the analysis:")

    # Find biggest bottleneck
    bottleneck = max(results.items(), key=lambda x: x[1] if x[0] != 'total' else 0)

    if bottleneck[0] == 'landmarks':
        print("1. CLNF landmarks are the primary bottleneck")
        print("   → Convert CLNF patch experts to ONNX")
        print("   → Use GPU for patch convolutions")
        print("   → Consider lighter landmark model")
    elif bottleneck[0] == 'au_prediction':
        print("1. AU prediction is the primary bottleneck")
        print("   → Convert SVM models to neural networks")
        print("   → Use GPU for batch inference")
        print("   → Implement model quantization")
    elif bottleneck[0] == 'detection':
        print("1. Face detection is the primary bottleneck")
        print("   → MTCNN already uses CoreML")
        print("   → Consider temporal tracking")
        print("   → Skip frames when face is stable")

    print("\n2. General optimizations:")
    print("   → Enable GPU acceleration for all components")
    print("   → Use batch processing")
    print("   → Implement model quantization")
    print("   → Convert to end-to-end neural network")