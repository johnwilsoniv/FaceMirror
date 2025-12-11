#!/usr/bin/env python3
"""
Quick 100-frame validation test for Big Red 200.
Tests one video with detailed accuracy reporting.
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import cv2

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyclnf'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pymtcnn'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-index', type=int, default=0)
    parser.add_argument('--n-frames', type=int, default=100)
    args = parser.parse_args()

    # Video list (same as main scripts)
    VIDEOS = [
        ("S Data/Normal Cohort", "IMG_0422.MOV"),
        ("S Data/Normal Cohort", "IMG_0428.MOV"),
        ("S Data/Normal Cohort", "IMG_0433.MOV"),
        ("S Data/Normal Cohort", "IMG_0434.MOV"),
        ("S Data/Normal Cohort", "IMG_0435.MOV"),
        ("S Data/Normal Cohort", "IMG_0438.MOV"),
        ("S Data/Normal Cohort", "IMG_0452.MOV"),
        ("S Data/Normal Cohort", "IMG_0453.MOV"),
        ("S Data/Normal Cohort", "IMG_0579.MOV"),
        ("S Data/Normal Cohort", "IMG_0942.MOV"),
        ("S Data/Paralysis Cohort", "IMG_0592.MOV"),
        ("S Data/Paralysis Cohort", "IMG_0861.MOV"),
        ("S Data/Paralysis Cohort", "IMG_1366.MOV"),
    ]

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cohort, filename = VIDEOS[args.video_index]
    video_path = os.path.join(project_dir, cohort, filename)
    video_name = os.path.splitext(filename)[0]

    # C++ reference
    cpp_dir = os.path.join(project_dir, "cpp_reference", f"video_{args.video_index}")
    cpp_csv = os.path.join(cpp_dir, f"{video_name}.csv")

    print(f"=" * 60)
    print(f"100-Frame Validation Test")
    print(f"Video: {filename} (index {args.video_index})")
    print(f"Video path: {video_path}")
    print(f"C++ reference: {cpp_csv}")
    print(f"=" * 60)

    # Check files exist
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        return 1
    if not os.path.exists(cpp_csv):
        print(f"ERROR: C++ reference not found: {cpp_csv}")
        return 1

    # Load C++ reference
    print("\n[1/4] Loading C++ reference...")
    cpp_df = pd.read_csv(cpp_csv)
    cpp_df.columns = cpp_df.columns.str.strip()  # Fix space-prefixed columns

    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    if not all(c in cpp_df.columns for c in x_cols + y_cols):
        print(f"ERROR: Missing landmark columns in C++ CSV")
        print(f"Available columns: {list(cpp_df.columns[:10])}...")
        return 1

    cpp_landmarks = np.stack([
        cpp_df[x_cols].values,
        cpp_df[y_cols].values
    ], axis=-1).astype(np.float32)

    print(f"  Loaded {len(cpp_df)} frames from C++ reference")
    print(f"  Landmark shape: {cpp_landmarks.shape}")

    # Load CLNF
    print("\n[2/4] Loading pyCLNF...")
    from pyclnf import CLNF
    clnf = CLNF(
        convergence_profile='video',  # MUST use 'video' for proper tracking mode
        detector='pymtcnn',  # Will use ONNX on Linux
        use_validator=False,
        use_eye_refinement=True,
    )
    print(f"  CLNF loaded with {clnf.pdm.n_points} landmarks")

    # Open video
    print("\n[3/4] Processing video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(args.n_frames, total_frames, len(cpp_df))
    print(f"  Total frames: {total_frames}, testing: {n_frames}")

    # Process frames
    errors = []
    jaw_errors = []
    eye_errors = []

    for frame_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  [Frame {frame_idx}] Failed to read")
            continue

        # Detect landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx == 0:
            # First frame: detect + fit
            landmarks, info = clnf.detect_and_fit(frame)  # Pass BGR for MTCNN
        else:
            # Subsequent: track from previous
            x_min, y_min = prev_landmarks.min(axis=0)
            x_max, y_max = prev_landmarks.max(axis=0)
            w, h = x_max - x_min, y_max - y_min
            margin = 0.1
            bbox = (x_min - w*margin, y_min - h*margin, w*(1+2*margin), h*(1+2*margin))
            landmarks, info = clnf.fit(gray, bbox)

        if landmarks is None:
            print(f"  [Frame {frame_idx}] Detection failed")
            continue

        prev_landmarks = landmarks.copy()

        # Compare to C++
        cpp_lm = cpp_landmarks[frame_idx]
        diff = landmarks - cpp_lm
        per_lm_error = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)

        mean_err = per_lm_error.mean()
        jaw_err = per_lm_error[:17].mean()
        eye_err = per_lm_error[36:48].mean()

        errors.append(mean_err)
        jaw_errors.append(jaw_err)
        eye_errors.append(eye_err)

        # Print every 10 frames
        if frame_idx % 10 == 0:
            print(f"  [Frame {frame_idx:3d}] Mean: {mean_err:.3f}px, Jaw: {jaw_err:.3f}px, Eyes: {eye_err:.3f}px, Converged: {info.get('converged', '?')}, Iters: {info.get('iterations', '?')}")

    cap.release()

    # Summary
    print("\n" + "=" * 60)
    print("[4/4] RESULTS")
    print("=" * 60)
    print(f"Frames processed: {len(errors)}/{n_frames}")
    print(f"Overall mean error: {np.mean(errors):.3f} px")
    print(f"Jaw mean error:     {np.mean(jaw_errors):.3f} px")
    print(f"Eye mean error:     {np.mean(eye_errors):.3f} px")
    print(f"Max error:          {np.max(errors):.3f} px")
    print(f"Std error:          {np.std(errors):.3f} px")

    # Pass/fail
    if np.mean(errors) < 2.0:
        print("\n✓ VALIDATION PASSED (mean error < 2.0px)")
        return 0
    else:
        print("\n✗ VALIDATION FAILED (mean error >= 2.0px)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
