#!/usr/bin/env python3
"""
Comprehensive landmark accuracy verification.
Compares Python pipeline with C++ for overall model and eye regions.
Processes entire video and reports aggregate statistics.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')
sys.path.insert(0, 'pyfaceau')

import numpy as np
import cv2
import subprocess
import os
import pandas as pd
import time

def get_cpp_landmarks_video(video_path: str) -> pd.DataFrame:
    """Run C++ FeatureExtraction on video and return all landmarks."""
    out_dir = '/tmp/openface_verify'
    os.makedirs(out_dir, exist_ok=True)

    # Run C++ on whole video
    cmd = [
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', video_path,
        '-out_dir', out_dir,
        '-2Dfp'
    ]

    print("Running C++ FeatureExtraction on video...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse CSV output
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(out_dir, f'{base_name}.csv')

    if not os.path.exists(csv_path):
        print(f"Error: C++ output not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"  C++ processed {len(df)} frames")
    return df

def extract_cpp_landmarks(df: pd.DataFrame, frame_idx: int) -> np.ndarray:
    """Extract landmarks for a specific frame from C++ DataFrame."""
    if frame_idx >= len(df):
        return None

    landmarks = np.zeros((68, 2))
    row = df.iloc[frame_idx]

    for i in range(68):
        for x_col, y_col in [(f'x_{i}', f'y_{i}'), (f' x_{i}', f' y_{i}')]:
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = row[x_col]
                landmarks[i, 1] = row[y_col]
                break

    return landmarks

def compute_errors(cpp_lm: np.ndarray, py_lm: np.ndarray):
    """Compute per-landmark errors."""
    errors = np.sqrt(np.sum((cpp_lm - py_lm)**2, axis=1))
    return errors

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify landmark accuracy against C++")
    parser.add_argument("video", nargs="?",
                       default="/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov",
                       help="Path to video file")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    print("=" * 70)
    print("FULL VIDEO LANDMARK ACCURACY VERIFICATION")
    print("=" * 70)

    video_path = args.video

    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return

    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"Size: {width}x{height}, Frames: {total_frames}, FPS: {fps:.1f}")

    # Run C++ on whole video
    cpp_df = get_cpp_landmarks_video(video_path)
    if cpp_df is None:
        return

    # Initialize Python CLNF
    print("\nInitializing Python pipeline...")
    from pyclnf.clnf import CLNF

    clnf = CLNF(
        'pyclnf/models',
        regularization=20,
        max_iterations=10,
        convergence_threshold=0.01,
        use_eye_refinement=True,
        min_iterations=5
    )

    # Process frames
    max_frames = args.max_frames or total_frames
    skip = args.skip

    all_errors = []
    all_eye_errors = []
    failed_frames = 0
    processed_frames = 0

    print(f"\nProcessing frames (skip={skip})...")
    start_time = time.time()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while frame_idx < min(max_frames, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        # Get C++ landmarks for this frame
        cpp_landmarks = extract_cpp_landmarks(cpp_df, frame_idx)
        if cpp_landmarks is None:
            frame_idx += 1
            continue

        # Get Python landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = clnf.detect_and_fit(frame)

        if result is None or result[0] is None:
            failed_frames += 1
            frame_idx += 1
            continue

        py_landmarks = result[0]

        # Compute errors
        errors = compute_errors(cpp_landmarks, py_landmarks)
        all_errors.append(errors)
        all_eye_errors.append(errors[36:48])

        processed_frames += 1

        # Progress
        if processed_frames % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = processed_frames / elapsed
            print(f"  Frame {frame_idx}/{total_frames} ({fps_actual:.1f} fps)")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    if processed_frames == 0:
        print("Error: No frames processed successfully")
        return

    # Aggregate statistics
    all_errors = np.array(all_errors)  # (num_frames, 68)
    all_eye_errors = np.array(all_eye_errors)  # (num_frames, 12)

    mean_errors = np.mean(all_errors, axis=0)  # Per-landmark mean across frames
    overall_mean = np.mean(all_errors)
    overall_median = np.median(all_errors)
    overall_max = np.max(all_errors)
    overall_std = np.std(all_errors)

    eye_mean = np.mean(all_eye_errors)

    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Frames processed: {processed_frames}")
    print(f"Failed frames: {failed_frames}")
    print(f"Time: {elapsed:.1f}s ({processed_frames/elapsed:.1f} fps)")

    # Overall statistics
    print(f"\n{'=' * 70}")
    print("OVERALL LANDMARK ACCURACY (68 points)")
    print(f"{'=' * 70}")
    print(f"Mean error:   {overall_mean:.2f}px")
    print(f"Median error: {overall_median:.2f}px")
    print(f"Max error:    {overall_max:.2f}px")
    print(f"Std dev:      {overall_std:.2f}px")

    # Region breakdown
    regions = {
        'Jaw (0-16)': list(range(17)),
        'Right eyebrow (17-21)': list(range(17, 22)),
        'Left eyebrow (22-26)': list(range(22, 27)),
        'Nose (27-35)': list(range(27, 36)),
        'Right eye (36-41)': list(range(36, 42)),
        'Left eye (42-47)': list(range(42, 48)),
        'Outer lip (48-59)': list(range(48, 60)),
        'Inner lip (60-67)': list(range(60, 68)),
    }

    print(f"\n{'=' * 70}")
    print("PER-REGION ACCURACY (mean across all frames)")
    print(f"{'=' * 70}")

    for region_name, indices in regions.items():
        region_errors = mean_errors[indices]
        region_all = all_errors[:, indices]
        print(f"\n{region_name}:")
        print(f"  Mean: {np.mean(region_errors):.2f}px, Max: {np.max(region_all):.2f}px")

    # Accuracy thresholds
    print(f"\n{'=' * 70}")
    print("ACCURACY ASSESSMENT")
    print(f"{'=' * 70}")

    if overall_mean < 2.0:
        overall_status = "EXCELLENT"
    elif overall_mean < 5.0:
        overall_status = "GOOD"
    elif overall_mean < 10.0:
        overall_status = "MODERATE"
    else:
        overall_status = "POOR"

    if eye_mean < 2.0:
        eye_status = "EXCELLENT"
    elif eye_mean < 3.0:
        eye_status = "GOOD"
    elif eye_mean < 5.0:
        eye_status = "MODERATE"
    else:
        eye_status = "POOR"

    print(f"\nOverall accuracy: {overall_status} ({overall_mean:.2f}px mean error)")
    print(f"Eye accuracy:     {eye_status} ({eye_mean:.2f}px mean error)")

    # Worst landmarks
    print(f"\n{'=' * 70}")
    print("WORST PERFORMING LANDMARKS (highest mean error)")
    print(f"{'=' * 70}")

    worst_indices = np.argsort(mean_errors)[-5:][::-1]
    for idx in worst_indices:
        print(f"  Landmark {idx}: {mean_errors[idx]:.2f}px mean error")

    # Frame-by-frame analysis
    frame_means = np.mean(all_errors, axis=1)
    worst_frame_idx = np.argmax(frame_means)
    best_frame_idx = np.argmin(frame_means)

    print(f"\n{'=' * 70}")
    print("FRAME ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Best frame:  {best_frame_idx * skip} (mean error: {frame_means[best_frame_idx]:.2f}px)")
    print(f"Worst frame: {worst_frame_idx * skip} (mean error: {frame_means[worst_frame_idx]:.2f}px)")
    print(f"Std across frames: {np.std(frame_means):.2f}px")

if __name__ == '__main__':
    main()
