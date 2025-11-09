#!/usr/bin/env python3
"""
Test Python PyFaceAU vs C++ OpenFace on new video set.
"""

import sys
import cv2
import numpy as np
import subprocess
import pandas as pd
from pathlib import Path

# Add S1 Face Mirror and pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "S1 Face Mirror"))
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

from pyfaceau_detector import PyFaceAU68LandmarkDetector


def parse_openface_landmarks(row):
    """Parse 68 landmarks from OpenFace CSV row."""
    landmarks = []
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    for i in range(68):
        x = row.get(x_cols[i], None)
        y = row.get(y_cols[i], None)

        if x is not None and y is not None and not pd.isna(x) and not pd.isna(y):
            landmarks.append([float(x), float(y)])
        else:
            return None

    return np.array(landmarks, dtype=np.float32)


def draw_landmarks_on_frame(frame, landmarks, color=(0, 255, 0), label=None):
    """Draw landmarks on frame with optional label."""
    vis_frame = frame.copy()

    if landmarks is not None:
        # Draw landmarks
        for pt in landmarks:
            cv2.circle(vis_frame, tuple(pt.astype(int)), 3, color, -1)

        # Draw facial outline (jaw)
        jaw_indices = list(range(0, 17))
        for i in range(len(jaw_indices) - 1):
            pt1 = tuple(landmarks[jaw_indices[i]].astype(int))
            pt2 = tuple(landmarks[jaw_indices[i+1]].astype(int))
            cv2.line(vis_frame, pt1, pt2, color, 2)

        # Draw eyes
        left_eye = list(range(36, 42))
        right_eye = list(range(42, 48))

        for indices in [left_eye, right_eye]:
            for i in range(len(indices)):
                pt1 = tuple(landmarks[indices[i]].astype(int))
                pt2 = tuple(landmarks[indices[(i+1) % len(indices)]].astype(int))
                cv2.line(vis_frame, pt1, pt2, color, 2)

        # Draw nose
        nose_bridge = list(range(27, 31))
        for i in range(len(nose_bridge) - 1):
            pt1 = tuple(landmarks[nose_bridge[i]].astype(int))
            pt2 = tuple(landmarks[nose_bridge[i+1]].astype(int))
            cv2.line(vis_frame, pt1, pt2, color, 2)

        # Draw mouth
        outer_mouth = list(range(48, 60))
        for i in range(len(outer_mouth)):
            pt1 = tuple(landmarks[outer_mouth[i]].astype(int))
            pt2 = tuple(landmarks[outer_mouth[(i+1) % len(outer_mouth)]].astype(int))
            cv2.line(vis_frame, pt1, pt2, color, 2)

    if label:
        cv2.putText(vis_frame, label, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(vis_frame, label, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    return vis_frame


def run_python_pipeline(video_path, output_dir, debug=False):
    """Run Python PyFaceAU pipeline."""
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
        'frames': []
    }

    frame_idx = 0
    frames_processed = 0
    frames_detected = 0
    frames_with_fallback = 0
    frames_validation_failed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
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
        results['landmarks'].append(landmarks.copy() if success else None)
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
    """Run C++ OpenFace 2.2."""
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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  OpenFace timed out after 5 minutes")
        return None

    if result.returncode != 0:
        print(f"  ⚠️  OpenFace failed with return code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:200]}")
        return None

    # Find output CSV
    video_name = Path(video_path).stem
    csv_path = output_dir / f"{video_name}.csv"

    if not csv_path.exists():
        print(f"  ⚠️  OpenFace output CSV not found: {csv_path}")
        return None

    # Load and analyze results
    df = pd.read_csv(csv_path)

    # Check for 'success' column
    if 'success' in df.columns:
        frames_detected = df['success'].sum()
        frames_processed = len(df)
        detection_rate = frames_detected / frames_processed * 100
    else:
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


def generate_side_by_side_comparison(python_results, cpp_results, video_path, output_path):
    """Generate side-by-side comparison: Python (left) vs C++ OpenFace (right)."""
    py_results = python_results['results']

    if cpp_results is None:
        # Generate Python-only visualization
        print(f"  Creating Python-only visualization (OpenFace unavailable)")
        generate_python_only_comparison(python_results, video_path, output_path)
        return

    cpp_df = cpp_results['df']

    # Select 3 interesting frames
    frame_indices_to_show = []

    # Priority 1: Frames with MTCNN fallback
    fallback_indices = [i for i, used in enumerate(py_results['used_fallback']) if used]
    if fallback_indices:
        if len(fallback_indices) >= 3:
            frame_indices_to_show = [
                fallback_indices[0],
                fallback_indices[len(fallback_indices)//2],
                fallback_indices[-1]
            ]
        else:
            frame_indices_to_show.extend(fallback_indices)

    # Priority 2: Frames with validation failures
    if len(frame_indices_to_show) < 3:
        val_fail_indices = [i for i, passed in enumerate(py_results['validation_passed'])
                           if not passed and i not in frame_indices_to_show]
        if val_fail_indices:
            needed = 3 - len(frame_indices_to_show)
            step = max(1, len(val_fail_indices) // needed)
            frame_indices_to_show.extend(val_fail_indices[::step][:needed])

    # Priority 3: Just take evenly distributed frames
    if len(frame_indices_to_show) < 3:
        total_frames = len(py_results['frames'])
        step = max(1, total_frames // 3)
        for i in range(0, total_frames, step):
            if i not in frame_indices_to_show:
                frame_indices_to_show.append(i)
            if len(frame_indices_to_show) >= 3:
                break

    # Ensure exactly 3 frames
    frame_indices_to_show = frame_indices_to_show[:3]

    if not frame_indices_to_show:
        print("  ⚠️  No frames to visualize")
        return

    # Create comparison images
    comparison_rows = []

    for py_idx in frame_indices_to_show:
        py_frame_num = py_results['frame_numbers'][py_idx]
        py_landmarks = py_results['landmarks'][py_idx]
        py_used_fallback = py_results['used_fallback'][py_idx]
        py_validation_passed = py_results['validation_passed'][py_idx]
        py_validation_reason = py_results['validation_reason'][py_idx]

        # Get the original frame
        frame = py_results['frames'][py_idx]

        # Get C++ OpenFace landmarks for this frame
        cpp_row = cpp_df[cpp_df['frame'] == py_frame_num]

        if len(cpp_row) > 0:
            cpp_landmarks = parse_openface_landmarks(cpp_row.iloc[0])
            cpp_success = cpp_row.iloc[0].get('success', 1) == 1
        else:
            cpp_landmarks = None
            cpp_success = False

        # Create Python visualization
        py_label = "Python PyFaceAU"
        if py_used_fallback:
            py_label += " (MTCNN)"

        py_color = (0, 255, 0) if py_validation_passed else (0, 165, 255)  # Orange if failed
        py_vis = draw_landmarks_on_frame(frame, py_landmarks, color=py_color, label=py_label)

        # Add status text
        status_y = 80
        if not py_validation_passed:
            cv2.putText(py_vis, f"Val FAIL: {py_validation_reason[:40]}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Create C++ visualization
        cpp_label = "C++ OpenFace 2.2"
        cpp_color = (255, 0, 0)  # Blue
        cpp_vis = draw_landmarks_on_frame(frame, cpp_landmarks, color=cpp_color, label=cpp_label)

        # Add frame number at bottom
        cv2.putText(py_vis, f"Frame {py_frame_num}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(cpp_vis, f"Frame {py_frame_num}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Concatenate horizontally
        comparison = np.hstack([py_vis, cpp_vis])
        comparison_rows.append(comparison)

    # Stack all comparisons vertically
    if comparison_rows:
        final_image = np.vstack(comparison_rows)
        cv2.imwrite(str(output_path), final_image)
        print(f"  ✅ Saved comparison: {output_path}")


def generate_python_only_comparison(python_results, video_path, output_path):
    """Generate Python-only visualization when OpenFace is unavailable."""
    py_results = python_results['results']

    # Select 3 frames
    frame_indices = []
    fallback_indices = [i for i, used in enumerate(py_results['used_fallback']) if used]
    if fallback_indices:
        frame_indices.extend(fallback_indices[:3])

    if len(frame_indices) < 3:
        val_fail = [i for i, p in enumerate(py_results['validation_passed']) if not p]
        if val_fail:
            frame_indices.extend(val_fail[:3-len(frame_indices)])

    if len(frame_indices) < 3:
        step = max(1, len(py_results['frames']) // 3)
        for i in range(0, len(py_results['frames']), step):
            if i not in frame_indices:
                frame_indices.append(i)
            if len(frame_indices) >= 3:
                break

    frame_indices = frame_indices[:3]

    rows = []
    for idx in frame_indices:
        frame = py_results['frames'][idx]
        landmarks = py_results['landmarks'][idx]
        used_fallback = py_results['used_fallback'][idx]
        validation_passed = py_results['validation_passed'][idx]
        frame_num = py_results['frame_numbers'][idx]

        label = "Python PyFaceAU"
        if used_fallback:
            label += " (MTCNN)"

        color = (0, 255, 0) if validation_passed else (0, 165, 255)
        vis = draw_landmarks_on_frame(frame, landmarks, color=color, label=label)

        cv2.putText(vis, f"Frame {frame_num}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        rows.append(vis)

    if rows:
        final_image = np.vstack(rows)
        cv2.imwrite(str(output_path), final_image)
        print(f"  ✅ Saved Python-only visualization: {output_path}")


def main():
    """Main test function."""
    videos = [
        '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0441_source.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0452_source.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0504_source.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0861_source.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_source.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_8401_source.MOV',
        '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_9330_source.MOV'
    ]

    openface_bin = Path('~/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction').expanduser()
    output_base = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/test_output')

    print("="*80)
    print("TESTING NEW VIDEO SET")
    print("="*80)
    print(f"Processing {len(videos)} videos")

    summary = []

    for video_path in videos:
        video_path = Path(video_path)

        if not video_path.exists():
            print(f"\n⚠️  Skipping {video_path.name}: File not found")
            continue

        video_name = video_path.stem.replace('_source', '')

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
        cpp_results = None
        try:
            cpp_results = run_cpp_openface(video_path, cpp_output, openface_bin)
        except Exception as e:
            print(f"\n⚠️  OpenFace error: {e}")

        # Generate comparison
        try:
            comparison_path = output_base / f'{video_name}_comparison.jpg'
            generate_side_by_side_comparison(python_results, cpp_results, video_path, comparison_path)
        except Exception as e:
            print(f"\n⚠️  Could not generate comparison: {e}")
            import traceback
            traceback.print_exc()

        # Add to summary
        summary.append({
            'video': video_name,
            'py_frames': python_results['frames_processed'],
            'py_detected': python_results['frames_detected'],
            'py_detection_rate': python_results['frames_detected'] / python_results['frames_processed'] * 100,
            'py_mtcnn_usage': python_results['frames_with_fallback'] / python_results['frames_processed'] * 100,
            'py_validation_failures': python_results['frames_validation_failed'],
            'cpp_frames': cpp_results['frames_processed'] if cpp_results else 0,
            'cpp_detected': cpp_results['frames_detected'] if cpp_results else 0,
            'cpp_detection_rate': cpp_results['frames_detected'] / cpp_results['frames_processed'] * 100 if cpp_results else 0
        })

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    df_summary = pd.DataFrame(summary)
    print("\n" + df_summary.to_string(index=False))

    # Save summary
    summary_path = output_base / 'new_video_set_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\n📊 Summary saved: {summary_path}")
    print(f"📁 Comparisons saved: {output_base}/*_comparison.jpg")

    return 0


if __name__ == '__main__':
    sys.exit(main())
