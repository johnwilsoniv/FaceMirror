#!/usr/bin/env python3
"""
Benchmark script for Python AU pipeline optimizations.

Measures performance of each component and overall pipeline speed.
Compares original vs optimized implementations.
"""

import time
import numpy as np
import cv2
from pathlib import Path
import sys
import warnings

# Add pyclnf to path
sys.path.insert(0, str(Path(__file__).parent))


def benchmark_component(func, *args, n_iterations=100, warmup=10, **kwargs):
    """
    Benchmark a function with proper warmup.

    Args:
        func: Function to benchmark
        args: Positional arguments
        n_iterations: Number of iterations to measure
        warmup: Number of warmup iterations
        kwargs: Keyword arguments

    Returns:
        Dict with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Measure
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
        'median': np.median(times) * 1000,
        'total': np.sum(times) * 1000
    }


def benchmark_cen_response():
    """Benchmark CEN patch expert response generation."""
    print("\n" + "=" * 60)
    print("BENCHMARKING CEN RESPONSE GENERATION")
    print("=" * 60)

    # Create test data
    test_patch = np.random.randn(51, 51).astype(np.float32)

    # Import both implementations
    from pyclnf.core.cen_patch_expert import CENPatchExpert, im2col_bias, contrast_norm

    try:
        from pyclnf.core.cen_patch_expert_optimized import (
            OptimizedCENPatchExpert,
            im2col_bias_optimized,
            contrast_norm_optimized,
            enable_optimizations
        )
        optimized_available = True

        # Enable optimizations
        enable_optimizations()
    except ImportError:
        optimized_available = False
        print("⚠️ Optimized version not available")

    # Benchmark im2col_bias
    print("\n1. IM2COL_BIAS (patch extraction):")
    print("-" * 40)

    orig_times = benchmark_component(im2col_bias, test_patch, 11, 11, n_iterations=1000)
    print(f"   Original:  {orig_times['mean']:.3f}ms ± {orig_times['std']:.3f}ms")

    if optimized_available:
        opt_times = benchmark_component(im2col_bias_optimized, test_patch, 11, 11, n_iterations=1000)
        print(f"   Optimized: {opt_times['mean']:.3f}ms ± {opt_times['std']:.3f}ms")
        speedup = orig_times['mean'] / opt_times['mean']
        print(f"   🚀 Speedup: {speedup:.1f}x")

    # Benchmark contrast_norm
    print("\n2. CONTRAST_NORM (normalization):")
    print("-" * 40)

    test_matrix = np.random.randn(1000, 122).astype(np.float32)
    orig_times = benchmark_component(contrast_norm, test_matrix, n_iterations=100)
    print(f"   Original:  {orig_times['mean']:.3f}ms ± {orig_times['std']:.3f}ms")

    if optimized_available:
        opt_times = benchmark_component(contrast_norm_optimized, test_matrix, n_iterations=100)
        print(f"   Optimized: {opt_times['mean']:.3f}ms ± {opt_times['std']:.3f}ms")
        speedup = orig_times['mean'] / opt_times['mean']
        print(f"   🚀 Speedup: {speedup:.1f}x")


def benchmark_full_pipeline():
    """Benchmark full AU prediction pipeline."""
    print("\n" + "=" * 60)
    print("BENCHMARKING FULL PIPELINE")
    print("=" * 60)

    # Check if test video exists
    test_videos = [
        Path("Patient Data/IMG_0441.MOV"),
        Path("test_videos/patient1_test.mp4"),
        Path("calibration_frames/patient1_frame1.jpg")
    ]

    test_file = None
    for path in test_videos:
        if path.exists():
            test_file = path
            break

    if not test_file:
        print("⚠️ No test video/image found. Creating synthetic test...")
        # Create synthetic test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        print(f"Using test file: {test_file}")
        if test_file.suffix in ['.jpg', '.png']:
            test_image = cv2.imread(str(test_file))
        else:
            cap = cv2.VideoCapture(str(test_file))
            ret, test_image = cap.read()
            cap.release()

    # Import pipeline components
    try:
        from pymtcnn import MTCNN
        from pyclnf import CLNF
        from pyfhog import extract_fhog
        from pyfaceau.features import extract_geometric_features
        from pyfaceau.prediction import AUPredictor

        print("\n✅ All pipeline components available")
    except ImportError as e:
        print(f"\n❌ Missing components: {e}")
        return

    # Initialize components
    print("\nInitializing pipeline components...")

    # Face detection
    print("  - MTCNN face detector...")
    detector = MTCNN()

    # Landmark detection
    print("  - CLNF landmark detector...")
    clnf = CLNF(
        model_path="pyclnf/models",
        use_cen=True,  # Use CEN patch experts
        verbose=False
    )

    # AU prediction
    print("  - AU predictor...")
    try:
        au_predictor = AUPredictor(
            models_path="pyfaceau/models/au_predictors",
            num_au_outputs=17
        )
    except:
        print("    (AU predictor not available, skipping)")
        au_predictor = None

    # Benchmark each component
    print("\n" + "-" * 60)
    print("COMPONENT BENCHMARKS:")
    print("-" * 60)

    # 1. Face Detection
    print("\n1. FACE DETECTION (MTCNN):")
    face_times = benchmark_component(
        detector.detect_faces,
        test_image,
        n_iterations=10,
        warmup=2
    )
    print(f"   Time: {face_times['mean']:.1f}ms ± {face_times['std']:.1f}ms")

    # Get face for next steps
    faces = detector.detect_faces(test_image)
    if faces:
        bbox = faces[0]['box']
        x, y, w, h = bbox
        face_roi = test_image[y:y+h, x:x+w]
    else:
        print("   No face detected, using full image")
        face_roi = test_image

    # 2. Landmark Detection
    print("\n2. LANDMARK DETECTION (CLNF):")

    # Initialize with face
    clnf.initialize_from_bbox(test_image, [x, y, w, h] if faces else None)

    landmark_times = benchmark_component(
        clnf.fit_image,
        test_image,
        n_iterations=10,
        warmup=2
    )
    print(f"   Time: {landmark_times['mean']:.1f}ms ± {landmark_times['std']:.1f}ms")

    # Get landmarks
    success = clnf.fit_image(test_image)
    if success:
        landmarks = clnf.get_landmarks()
        print(f"   Detected {len(landmarks)} landmarks")
    else:
        print("   Landmark detection failed")
        landmarks = None

    # 3. HOG Features
    print("\n3. HOG FEATURE EXTRACTION:")

    if landmarks is not None:
        # Align face for HOG
        from pyfaceau.alignment import align_face
        aligned_face = align_face(test_image, landmarks)

        hog_times = benchmark_component(
            extract_fhog,
            aligned_face,
            cell_size=8,
            n_iterations=20,
            warmup=5
        )
        print(f"   Time: {hog_times['mean']:.1f}ms ± {hog_times['std']:.1f}ms")

        hog_features = extract_fhog(aligned_face, cell_size=8)
        print(f"   HOG dimensions: {hog_features.shape}")
    else:
        print("   Skipped (no landmarks)")
        hog_features = None

    # 4. AU Prediction
    if au_predictor and landmarks is not None and hog_features is not None:
        print("\n4. AU PREDICTION:")

        # Extract geometric features
        geom_features = extract_geometric_features(landmarks)

        au_times = benchmark_component(
            au_predictor.predict,
            hog_features,
            geom_features,
            n_iterations=100,
            warmup=10
        )
        print(f"   Time: {au_times['mean']:.1f}ms ± {au_times['std']:.1f}ms")

        aus = au_predictor.predict(hog_features, geom_features)
        print(f"   Predicted {len(aus)} AUs")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY:")
    print("=" * 60)

    total_time = (
        face_times['mean'] +
        landmark_times['mean'] +
        (hog_times['mean'] if 'hog_times' in locals() else 0) +
        (au_times['mean'] if 'au_times' in locals() else 0)
    )

    print(f"\nTotal pipeline time: {total_time:.1f}ms")
    print(f"FPS capability: {1000/total_time:.1f} fps")

    print("\nBottlenecks:")
    components = [
        ("Face Detection", face_times['mean']),
        ("Landmark Detection", landmark_times['mean']),
        ("HOG Extraction", hog_times['mean'] if 'hog_times' in locals() else 0),
        ("AU Prediction", au_times['mean'] if 'au_times' in locals() else 0)
    ]

    components.sort(key=lambda x: x[1], reverse=True)
    for name, time_ms in components:
        if time_ms > 0:
            percentage = (time_ms / total_time) * 100
            print(f"  {name:20s}: {time_ms:6.1f}ms ({percentage:4.1f}%)")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("PYTHON AU PIPELINE OPTIMIZATION BENCHMARKS")
    print("=" * 60)

    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import numba
        print("✅ Numba available - optimizations enabled")
    except ImportError:
        print("⚠️ Numba not installed - install with: pip install numba")

    # Run benchmarks
    benchmark_cen_response()
    benchmark_full_pipeline()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()