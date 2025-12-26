#!/usr/bin/env python3
"""
Validate S1 FaceMirror GPU Pipeline

Tests:
1. FPS performance (target: 10-15 fps)
2. Landmark accuracy vs reference
3. AU extraction accuracy vs C++ OpenFace reference
"""

import sys
import time
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy import stats

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test paths
TEST_VIDEO = Path(__file__).parent.parent / "S Data/Normal Cohort/IMG_0942.MOV"
CPP_REFERENCE = Path(__file__).parent.parent / "archive/test_artifacts_root/cpp_reference_output/IMG_0942.csv"


def test_fps_performance(max_frames=300):
    """Test FPS performance of the landmark detector."""
    print("\n" + "="*60)
    print("FPS PERFORMANCE TEST")
    print("="*60)

    from pyfaceau_detector import PyFaceAU68LandmarkDetector

    # Initialize detector
    print("\nInitializing GPU-accelerated detector...")
    start_init = time.time()
    detector = PyFaceAU68LandmarkDetector(debug_mode=True)
    init_time = time.time() - start_init
    print(f"Initialization time: {init_time:.2f}s")

    # Open video
    if not TEST_VIDEO.exists():
        print(f"ERROR: Test video not found: {TEST_VIDEO}")
        return None

    cap = cv2.VideoCapture(str(TEST_VIDEO))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {TEST_VIDEO.name}")
    print(f"  Total frames: {total_frames}")
    print(f"  Video FPS: {video_fps:.1f}")
    print(f"  Testing first {min(max_frames, total_frames)} frames")

    # Process frames
    frame_times = []
    detection_times = []
    landmark_times = []
    successful_frames = 0

    print("\nProcessing...")

    for i in range(min(max_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Time the full detection
        start = time.time()
        landmarks, info = detector.get_face_mesh(frame, detection_interval=30)
        elapsed = time.time() - start

        frame_times.append(elapsed)

        if landmarks is not None:
            successful_frames += 1

        # Progress
        if (i + 1) % 50 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-50:]) if frame_times else 0
            print(f"  Frame {i+1}/{min(max_frames, total_frames)}: {avg_fps:.1f} fps")

    cap.release()
    detector.cleanup_memory()

    # Calculate statistics
    frame_times = np.array(frame_times)
    avg_time = np.mean(frame_times)
    std_time = np.std(frame_times)
    avg_fps = 1.0 / avg_time

    # Exclude first 10 frames (warmup) for steady-state FPS
    if len(frame_times) > 10:
        steady_state_fps = 1.0 / np.mean(frame_times[10:])
    else:
        steady_state_fps = avg_fps

    print("\n" + "-"*40)
    print("RESULTS:")
    print(f"  Frames processed: {len(frame_times)}")
    print(f"  Successful detections: {successful_frames} ({100*successful_frames/len(frame_times):.1f}%)")
    print(f"  Average time/frame: {avg_time*1000:.1f} ms")
    print(f"  Std dev: {std_time*1000:.1f} ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Steady-state FPS (after warmup): {steady_state_fps:.1f}")

    # Pass/fail
    target_fps = 10.0
    if steady_state_fps >= target_fps:
        print(f"\n  ✓ PASS: FPS >= {target_fps}")
    else:
        print(f"\n  ✗ FAIL: FPS < {target_fps}")

    return {
        'avg_fps': avg_fps,
        'steady_state_fps': steady_state_fps,
        'avg_time_ms': avg_time * 1000,
        'successful_frames': successful_frames,
        'total_frames': len(frame_times),
        'passed': steady_state_fps >= target_fps
    }


def test_au_accuracy(max_frames=None):
    """Test AU extraction accuracy against C++ reference."""
    print("\n" + "="*60)
    print("AU ACCURACY TEST")
    print("="*60)

    # Check for reference file
    if not CPP_REFERENCE.exists():
        print(f"ERROR: Reference file not found: {CPP_REFERENCE}")
        return None

    # Load reference data
    print(f"\nLoading C++ reference: {CPP_REFERENCE.name}")
    ref_df = pd.read_csv(CPP_REFERENCE)

    # Get AU columns
    au_cols = [c for c in ref_df.columns if c.startswith('AU') and '_r' in c]
    print(f"  AU columns found: {len(au_cols)}")

    # Run Python pipeline
    print("\nRunning Python AU pipeline...")

    try:
        from pyfaceau import FullPythonAUPipeline

        weights_dir = Path(__file__).parent / "weights"

        pipeline = FullPythonAUPipeline(
            pdm_file=str(weights_dir / 'In-the-wild_aligned_PDM_68.txt'),
            au_models_dir=str(weights_dir / 'AU_predictors'),
            triangulation_file=str(weights_dir / 'tris_68_full.txt'),
            patch_expert_file=str(weights_dir / 'svr_patches_0.25_general.txt'),
            mtcnn_backend='coreml',
            track_faces=True,
            use_nnclnf='pyclnf'
        )

        # Process video
        start = time.time()
        py_df = pipeline.process_video(str(TEST_VIDEO), max_frames=max_frames)
        elapsed = time.time() - start

        print(f"  Processing time: {elapsed:.1f}s")
        print(f"  Frames processed: {len(py_df)}")

    except Exception as e:
        print(f"ERROR: Failed to run Python pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Compare AU values
    print("\nComparing AU values...")

    # Align frame counts
    min_frames = min(len(ref_df), len(py_df))
    ref_df = ref_df.head(min_frames)
    py_df = py_df.head(min_frames)

    results = {}
    passed = 0
    failed = 0

    for au_col in au_cols:
        if au_col not in py_df.columns:
            continue

        ref_values = ref_df[au_col].values
        py_values = py_df[au_col].values

        # Calculate correlation
        if np.std(ref_values) > 0.01 and np.std(py_values) > 0.01:
            corr, _ = stats.pearsonr(ref_values, py_values)
        else:
            corr = 1.0 if np.allclose(ref_values, py_values, atol=0.1) else 0.0

        # Calculate RMSE
        rmse = np.sqrt(np.mean((ref_values - py_values) ** 2))

        # Pass if correlation >= 0.95
        au_passed = corr >= 0.95
        if au_passed:
            passed += 1
        else:
            failed += 1

        results[au_col] = {
            'correlation': corr,
            'rmse': rmse,
            'passed': au_passed
        }

    # Print results
    print("\n" + "-"*40)
    print("AU CORRELATION RESULTS:")
    print("-"*40)
    print(f"{'AU':<12} {'Correlation':>12} {'RMSE':>8} {'Status':>8}")
    print("-"*40)

    for au_col, r in sorted(results.items()):
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"{au_col:<12} {r['correlation']:>12.4f} {r['rmse']:>8.4f} {status:>8}")

    print("-"*40)
    print(f"Total: {passed}/{passed+failed} AUs passed (>= 0.95 correlation)")

    overall_passed = passed >= 15  # Target: 15/17 AUs pass
    if overall_passed:
        print(f"\n  ✓ PASS: {passed}/17 AUs >= 0.95 correlation")
    else:
        print(f"\n  ✗ FAIL: Only {passed}/17 AUs >= 0.95 correlation (need 15)")

    return {
        'au_results': results,
        'passed_count': passed,
        'failed_count': failed,
        'overall_passed': overall_passed
    }


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("S1 FACEMIRROR GPU PIPELINE VALIDATION")
    print("="*60)

    # Test 1: FPS Performance
    fps_results = test_fps_performance(max_frames=300)

    # Test 2: AU Accuracy
    au_results = test_au_accuracy(max_frames=300)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    if fps_results:
        fps_status = "✓ PASS" if fps_results['passed'] else "✗ FAIL"
        print(f"\nFPS Performance: {fps_status}")
        print(f"  Steady-state FPS: {fps_results['steady_state_fps']:.1f}")
    else:
        print("\nFPS Performance: ✗ SKIPPED (error)")

    if au_results:
        au_status = "✓ PASS" if au_results['overall_passed'] else "✗ FAIL"
        print(f"\nAU Accuracy: {au_status}")
        print(f"  AUs passing: {au_results['passed_count']}/17")
    else:
        print("\nAU Accuracy: ✗ SKIPPED (error)")

    # Overall
    print("\n" + "-"*40)
    all_passed = (
        (fps_results and fps_results['passed']) and
        (au_results and au_results['overall_passed'])
    )

    if all_passed:
        print("OVERALL: ✓ ALL TESTS PASSED")
    else:
        print("OVERALL: ✗ SOME TESTS FAILED")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
