#!/usr/bin/env python3
"""
Comprehensive test of Python PyFaceAU vs C++ OpenFace 2.2 across multiple videos.
Identifies problem files and generates side-by-side validation images.
"""

import sys
import cv2
import numpy as np
import subprocess
import pandas as pd
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Tuple

# Add S1 Face Mirror and pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector


def run_python_pipeline(video_path, output_dir, debug=False):
    """
    Run Python PyFaceAU pipeline (with MTCNN fallback).

    Returns:
        Dict with results and statistics
    """
    print(f"\n{'='*80}")
    print(f"PYTHON PIPELINE: {Path(video_path).name}")
    print(f"{'='*80}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize detector
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

    print(f"  Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # Results tracking
    results = {
        'frame_numbers': [],
        'detection_success': [],
        'used_fallback': [],
        'validation_passed': [],
        'validation_reason': [],
        'confidence': [],
        'landmarks': [],
        'frames': []  # Store frames for visualization
    }

    frame_idx = 0
    frames_processed = 0
    frames_detected = 0
    frames_with_fallback = 0
    frames_validation_failed = 0

    # Process every 30th frame for speed (about 2 fps sampling)
    frame_skip = 30

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
        confidence = validation_info.get('confidence', 0.0) if validation_info else 0.0

        results['frame_numbers'].append(frame_idx)
        results['detection_success'].append(success)
        results['used_fallback'].append(used_fallback)
        results['validation_passed'].append(validation_passed)
        results['validation_reason'].append(validation_reason)
        results['confidence'].append(confidence)
        results['landmarks'].append(landmarks if success else None)
        results['frames'].append(frame.copy())

        if success:
            frames_detected += 1
        if used_fallback:
            frames_with_fallback += 1
        if not validation_passed and validation_info is not None:
            frames_validation_failed += 1

    cap.release()

    # Save results to CSV
    csv_path = output_dir / 'python_results.csv'
    df = pd.DataFrame({
        'frame': results['frame_numbers'],
        'detected': results['detection_success'],
        'used_mtcnn_fallback': results['used_fallback'],
        'validation_passed': results['validation_passed'],
        'validation_reason': results['validation_reason'],
        'confidence': results['confidence']
    })
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"  Frames processed: {frames_processed}")
    print(f"  Frames detected: {frames_detected} ({frames_detected/frames_processed*100:.1f}%)")
    print(f"  MTCNN fallback: {frames_with_fallback} ({frames_with_fallback/frames_processed*100:.1f}%)")
    print(f"  Validation failures: {frames_validation_failed} ({frames_validation_failed/frames_processed*100:.1f}%)")

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
    print(f"\n{'='*80}")
    print(f"C++ OPENFACE: {Path(video_path).name}")
    print(f"{'='*80}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run OpenFace FeatureExtraction
    cmd = [
        str(openface_bin),
        '-f', str(video_path),
        '-out_dir', str(output_dir)
    ]

    print(f"  Running OpenFace...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  ERROR: OpenFace failed with return code {result.returncode}")
        print(f"  STDERR: {result.stderr}")
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
        detection_rate = frames_detected / frames_processed * 100
    else:
        # Assume all frames were detected if no 'success' column
        frames_detected = len(df)
        frames_processed = len(df)
        detection_rate = 100.0

    print(f"  Frames processed: {frames_processed}")
    print(f"  Frames detected: {frames_detected} ({detection_rate:.1f}%)")

    return {
        'frames_processed': frames_processed,
        'frames_detected': frames_detected,
        'csv_path': csv_path,
        'df': df
    }


def generate_validation_image(python_results, video_path, output_path):
    """
    Generate side-by-side validation image showing:
    - Original frame
    - Python detection with landmarks
    - Status indicators
    """
    results = python_results['results']

    # Find interesting frames to visualize
    # Priority: frames with fallback > validation failures > normal detections
    frame_indices = []

    # Get frames with MTCNN fallback
    fallback_indices = [i for i, used in enumerate(results['used_fallback']) if used]
    if fallback_indices:
        frame_indices.extend(fallback_indices[:3])  # Up to 3 fallback frames

    # Get frames with validation failures
    val_fail_indices = [i for i, passed in enumerate(results['validation_passed']) if not passed]
    if val_fail_indices and len(frame_indices) < 3:
        frame_indices.extend(val_fail_indices[:3 - len(frame_indices)])

    # Get some normal frames if we don't have enough
    if len(frame_indices) < 3:
        normal_indices = [i for i in range(len(results['frames'])) if i not in frame_indices]
        frame_indices.extend(normal_indices[::len(normal_indices)//3][:3 - len(frame_indices)])

    if not frame_indices:
        # Just use first 3 frames
        frame_indices = list(range(min(3, len(results['frames']))))

    # Create composite image
    rows = []

    for idx in frame_indices:
        frame = results['frames'][idx]
        landmarks = results['landmarks'][idx]
        used_fallback = results['used_fallback'][idx]
        validation_passed = results['validation_passed'][idx]
        validation_reason = results['validation_reason'][idx]
        frame_num = results['frame_numbers'][idx]

        # Draw landmarks on frame copy
        vis_frame = frame.copy()

        if landmarks is not None:
            # Draw landmarks
            for pt in landmarks:
                cv2.circle(vis_frame, tuple(pt.astype(int)), 2, (0, 255, 0), -1)

            # Draw facial outline
            jaw_indices = list(range(0, 17))
            for i in range(len(jaw_indices) - 1):
                pt1 = tuple(landmarks[jaw_indices[i]].astype(int))
                pt2 = tuple(landmarks[jaw_indices[i+1]].astype(int))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 1)

        # Add status text
        status_color = (0, 255, 0) if validation_passed else (0, 0, 255)
        status_text = "✓ Valid" if validation_passed else "✗ Failed"

        cv2.putText(vis_frame, f"Frame {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if used_fallback:
            cv2.putText(vis_frame, "MTCNN Fallback", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add to rows
        rows.append(vis_frame)

    # Stack vertically
    if rows:
        composite = np.vstack(rows)
        cv2.imwrite(str(output_path), composite)
        print(f"  Saved validation image: {output_path}")


def main():
    """Main test function."""
    # Test configuration
    videos = [
        '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0435.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0437.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0438.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Paralysis Cohort/IMG_2737.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Paralysis Cohort/IMG_5694.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Paralysis Cohort/IMG_9330.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Paralysis Cohort/IMG_8401.MOV'
    ]

    openface_bin = Path('~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction').expanduser()

    # Create output directory
    output_base = Path('/tmp/video_comparison_test')
    output_base.mkdir(exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE VIDEO COMPARISON TEST")
    print("="*80)
    print(f"Testing {len(videos)} videos")
    print(f"Output directory: {output_base}")

    # Results summary
    summary = []

    for video_path in videos:
        video_path = Path(video_path)

        if not video_path.exists():
            print(f"\n⚠️  Skipping {video_path.name}: File not found")
            continue

        video_name = video_path.stem

        # Create output directories
        python_output = output_base / video_name / 'python_output'
        cpp_output = output_base / video_name / 'cpp_output'

        print(f"\n{'='*80}")
        print(f"PROCESSING: {video_name}")
        print(f"{'='*80}")

        # Run Python pipeline
        try:
            python_results = run_python_pipeline(video_path, python_output, debug=False)
        except Exception as e:
            print(f"\n❌ ERROR in Python pipeline: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Run C++ OpenFace
        try:
            cpp_results = run_cpp_openface(video_path, cpp_output, openface_bin)
        except Exception as e:
            print(f"\n❌ ERROR in C++ OpenFace: {e}")
            import traceback
            traceback.print_exc()
            cpp_results = None

        # Generate validation images
        try:
            vis_output = output_base / video_name / f'{video_name}_validation.jpg'
            generate_validation_image(python_results, video_path, vis_output)
        except Exception as e:
            print(f"\n⚠️  Could not generate validation image: {e}")

        # Add to summary
        summary.append({
            'video': video_name,
            'cohort': 'Normal' if 'Normal Cohort' in str(video_path) else 'Paralysis',
            'py_frames': python_results['frames_processed'],
            'py_detected': python_results['frames_detected'],
            'py_detection_rate': python_results['frames_detected'] / python_results['frames_processed'] * 100,
            'py_mtcnn_usage': python_results['frames_with_fallback'] / python_results['frames_processed'] * 100,
            'py_validation_failures': python_results['frames_validation_failed'],
            'cpp_frames': cpp_results['frames_processed'] if cpp_results else 0,
            'cpp_detected': cpp_results['frames_detected'] if cpp_results else 0,
            'cpp_detection_rate': cpp_results['frames_detected'] / cpp_results['frames_processed'] * 100 if cpp_results else 0
        })

    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY")
    print("="*80)

    df_summary = pd.DataFrame(summary)

    print("\nDetection Performance:")
    print(df_summary.to_string(index=False))

    # Identify problem files
    print("\n" + "="*80)
    print("PROBLEM FILE IDENTIFICATION")
    print("="*80)

    problem_files = df_summary[df_summary['py_mtcnn_usage'] > 0]

    if len(problem_files) > 0:
        print(f"\n🔴 Found {len(problem_files)} files requiring MTCNN fallback:")
        for _, row in problem_files.iterrows():
            print(f"  • {row['video']}: {row['py_mtcnn_usage']:.1f}% of frames used MTCNN")
    else:
        print("\n✅ No files required MTCNN fallback - RetinaFace worked on all videos!")

    validation_failures = df_summary[df_summary['py_validation_failures'] > 0]

    if len(validation_failures) > 0:
        print(f"\n⚠️  Found {len(validation_failures)} files with validation failures:")
        for _, row in validation_failures.iterrows():
            print(f"  • {row['video']}: {row['py_validation_failures']} frames failed validation")

    # Save summary
    summary_path = output_base / 'summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Individual results in: {output_base}/*/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
