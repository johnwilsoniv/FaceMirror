#!/usr/bin/env python3
"""
Test MTCNN fallback pipeline against C++ OpenFace 2.2.

This script compares the Python PyFaceAU implementation (with MTCNN fallback)
against the C++ OpenFace 2.2 implementation for a file that's failing RetinaFace.
"""

import sys
import cv2
import numpy as np
import subprocess
import pandas as pd
from pathlib import Path
import shutil
import tempfile

# Add S1 Face Mirror and pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector


def run_python_pipeline(video_path, output_dir, debug=True):
    """
    Run Python PyFaceAU pipeline (with MTCNN fallback).

    Returns:
        Dict with results and statistics
    """
    print("\n" + "="*80)
    print("RUNNING PYTHON PYFACEAU PIPELINE")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize detector with debug mode
    detector = PyFaceAU68LandmarkDetector(
        debug_mode=debug,
        skip_redetection=False,
        skip_face_detection=False,
        use_clnf_refinement=True
    )

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # Results tracking
    results = {
        'frame_numbers': [],
        'detection_success': [],
        'used_fallback': [],
        'validation_passed': [],
        'validation_reason': [],
        'landmarks': []
    }

    frame_idx = 0
    frames_processed = 0
    frames_detected = 0
    frames_with_fallback = 0
    frames_validation_failed = 0

    # Process every 10th frame for speed
    frame_skip = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames
        if frame_idx % frame_skip != 0:
            continue

        frames_processed += 1

        # Get landmarks with validation info
        result = detector.get_face_mesh(frame)

        if result is not None and len(result) == 2:
            landmarks, validation_info = result
        else:
            landmarks = result
            validation_info = None

        # Track results
        success = landmarks is not None
        used_fallback = validation_info.get('used_fallback', False) if validation_info else False
        validation_passed = validation_info.get('validation_passed', False) if validation_info else False
        validation_reason = validation_info.get('reason', 'Unknown') if validation_info else 'Unknown'

        results['frame_numbers'].append(frame_idx)
        results['detection_success'].append(success)
        results['used_fallback'].append(used_fallback)
        results['validation_passed'].append(validation_passed)
        results['validation_reason'].append(validation_reason)
        results['landmarks'].append(landmarks if success else None)

        if success:
            frames_detected += 1
        if used_fallback:
            frames_with_fallback += 1
        if not validation_passed and validation_info is not None:
            frames_validation_failed += 1

        # Print progress
        if frames_processed % 10 == 0:
            print(f"Processed {frames_processed} frames ({frames_detected} detected, "
                  f"{frames_with_fallback} with MTCNN fallback)")

    cap.release()

    # Save results to CSV
    csv_path = output_dir / 'python_results.csv'
    df = pd.DataFrame({
        'frame': results['frame_numbers'],
        'detected': results['detection_success'],
        'used_mtcnn_fallback': results['used_fallback'],
        'validation_passed': results['validation_passed'],
        'validation_reason': results['validation_reason']
    })
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "="*80)
    print("PYTHON PIPELINE SUMMARY")
    print("="*80)
    print(f"Frames processed: {frames_processed}")
    print(f"Frames detected: {frames_detected} ({frames_detected/frames_processed*100:.1f}%)")
    print(f"Frames using MTCNN fallback: {frames_with_fallback} ({frames_with_fallback/frames_processed*100:.1f}%)")
    print(f"Frames with validation failures: {frames_validation_failed} ({frames_validation_failed/frames_processed*100:.1f}%)")
    print(f"Results saved to: {csv_path}")

    return {
        'frames_processed': frames_processed,
        'frames_detected': frames_detected,
        'frames_with_fallback': frames_with_fallback,
        'frames_validation_failed': frames_validation_failed,
        'results': results,
        'csv_path': csv_path
    }


def run_cpp_openface(video_path, output_dir, openface_bin):
    """
    Run C++ OpenFace 2.2.

    Returns:
        Dict with results
    """
    print("\n" + "="*80)
    print("RUNNING C++ OPENFACE 2.2")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run OpenFace FeatureExtraction
    cmd = [
        str(openface_bin),
        '-f', str(video_path),
        '-out_dir', str(output_dir),
        '-verbose'
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\n--- OpenFace STDOUT ---")
    print(result.stdout)

    if result.stderr:
        print("\n--- OpenFace STDERR ---")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"OpenFace failed with return code {result.returncode}")

    # Find output CSV
    video_name = Path(video_path).stem
    csv_path = output_dir / f"{video_name}.csv"

    if not csv_path.exists():
        raise RuntimeError(f"OpenFace output CSV not found: {csv_path}")

    # Load and analyze results
    df = pd.read_csv(csv_path)

    # Check for 'success' column (indicates detection success per frame)
    if 'success' in df.columns:
        frames_detected = df['success'].sum()
        frames_processed = len(df)
    else:
        # Assume all frames were detected if no 'success' column
        frames_detected = len(df)
        frames_processed = len(df)

    print("\n" + "="*80)
    print("C++ OPENFACE 2.2 SUMMARY")
    print("="*80)
    print(f"Frames processed: {frames_processed}")
    print(f"Frames detected: {frames_detected} ({frames_detected/frames_processed*100:.1f}%)")
    print(f"Output CSV: {csv_path}")

    return {
        'frames_processed': frames_processed,
        'frames_detected': frames_detected,
        'csv_path': csv_path,
        'df': df
    }


def compare_results(python_results, cpp_results):
    """Compare Python and C++ results."""
    print("\n" + "="*80)
    print("COMPARISON: PYTHON vs C++ OPENFACE 2.2")
    print("="*80)

    # Detection rates
    py_rate = python_results['frames_detected'] / python_results['frames_processed'] * 100
    cpp_rate = cpp_results['frames_detected'] / cpp_results['frames_processed'] * 100

    print(f"\nDetection Success Rate:")
    print(f"  Python (PyFaceAU + MTCNN fallback): {py_rate:.1f}%")
    print(f"  C++ OpenFace 2.2:                   {cpp_rate:.1f}%")
    print(f"  Difference:                         {py_rate - cpp_rate:+.1f}%")

    # MTCNN usage
    mtcnn_rate = python_results['frames_with_fallback'] / python_results['frames_processed'] * 100
    print(f"\nMTCNN Fallback Usage:")
    print(f"  Frames using MTCNN: {python_results['frames_with_fallback']} ({mtcnn_rate:.1f}%)")

    # Validation failures
    val_fail_rate = python_results['frames_validation_failed'] / python_results['frames_processed'] * 100
    print(f"\nValidation Failures:")
    print(f"  Frames with validation failures: {python_results['frames_validation_failed']} ({val_fail_rate:.1f}%)")

    print("\n" + "="*80)


def main():
    """Main test function."""
    # Test configuration
    video_path = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0435.MOV')
    openface_bin = Path('~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction').expanduser()

    # Create temp output directory
    temp_dir = Path(tempfile.mkdtemp(prefix='mtcnn_vs_cpp_'))
    python_output = temp_dir / 'python_output'
    cpp_output = temp_dir / 'cpp_output'

    print("="*80)
    print("MTCNN FALLBACK vs C++ OPENFACE 2.2 TEST")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"OpenFace binary: {openface_bin}")
    print(f"Output directory: {temp_dir}")

    if not video_path.exists():
        print(f"\nERROR: Video not found: {video_path}")
        return 1

    if not openface_bin.exists():
        print(f"\nERROR: OpenFace binary not found: {openface_bin}")
        return 1

    # Run Python pipeline
    try:
        python_results = run_python_pipeline(video_path, python_output, debug=True)
    except Exception as e:
        print(f"\nERROR running Python pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run C++ OpenFace
    try:
        cpp_results = run_cpp_openface(video_path, cpp_output, openface_bin)
    except Exception as e:
        print(f"\nERROR running C++ OpenFace: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare results
    compare_results(python_results, cpp_results)

    print(f"\nResults saved to: {temp_dir}")
    print("  - python_output/python_results.csv")
    print("  - cpp_output/IMG_0435.csv")

    return 0


if __name__ == '__main__':
    sys.exit(main())