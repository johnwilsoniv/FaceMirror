#!/usr/bin/env python3
"""
Compare Python CLNF accuracy against OpenFace C++.
Uses pre-generated OpenFace CSV for comparison.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import pandas as pd
import time
from pathlib import Path

from pyclnf.clnf import CLNF

def load_openface_landmarks(csv_path: str) -> dict:
    """Load OpenFace landmarks from CSV."""
    df = pd.read_csv(csv_path)

    landmarks_by_frame = {}
    for idx, row in df.iterrows():
        frame_num = int(row['frame'])
        landmarks = np.zeros((68, 2))

        for i in range(68):
            # Try both column name formats
            for x_col, y_col in [(f'x_{i}', f'y_{i}'), (f' x_{i}', f' y_{i}')]:
                if x_col in df.columns and y_col in df.columns:
                    landmarks[i, 0] = row[x_col]
                    landmarks[i, 1] = row[y_col]
                    break

        landmarks_by_frame[frame_num] = landmarks

    return landmarks_by_frame

def main():
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0942.MOV'
    csv_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/IMG_0942.csv'
    num_frames = 75

    print("="*60)
    print("CLNF Accuracy Comparison: Python vs C++ OpenFace")
    print("="*60)

    # Load OpenFace landmarks
    print("\nLoading OpenFace C++ landmarks...")
    cpp_landmarks = load_openface_landmarks(csv_path)
    print(f"  Loaded {len(cpp_landmarks)} frames from OpenFace")

    # Load video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Python CLNF with optimal settings from PYCLNF_ACCURACY_REPORT.md
    print("\nInitializing Python CLNF...")
    clnf = CLNF(
        'pyclnf/pyclnf/models',
        regularization=20,           # Optimal for Python
        max_iterations=30,           # 10 per window × 3 windows
        convergence_threshold=0.01,
        window_sizes=[11, 9, 7]      # WS=5 disabled (hurts accuracy)
    )

    # Process frames
    print(f"\nComparing {num_frames} frames...")

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    all_errors = []
    per_landmark_errors = [[] for _ in range(68)]

    for i, frame_idx in enumerate(frame_indices):
        # Get C++ landmarks for this frame
        if frame_idx not in cpp_landmarks:
            continue

        cpp_lm = cpp_landmarks[frame_idx]

        # Get Python landmarks
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        result = clnf.detect_and_fit(frame)
        if result is None or result[0] is None:
            continue

        py_lm = result[0]

        # Compute errors
        errors = np.sqrt(np.sum((cpp_lm - py_lm)**2, axis=1))
        all_errors.append(errors)

        for j in range(68):
            per_landmark_errors[j].append(errors[j])

        # Progress
        if (i + 1) % 15 == 0:
            mean_err = np.mean(errors)
            print(f"  Frame {i+1}/{num_frames}: mean error = {mean_err:.2f}px")

    cap.release()

    if not all_errors:
        print("No frames processed!")
        return

    # Statistics
    all_errors = np.array(all_errors)

    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)

    overall_mean = np.mean(all_errors)
    overall_std = np.std(all_errors)
    overall_max = np.max(all_errors)

    print(f"\nOverall Statistics ({len(all_errors)} frames):")
    print(f"  Mean error: {overall_mean:.2f}px")
    print(f"  Std dev:    {overall_std:.2f}px")
    print(f"  Max error:  {overall_max:.2f}px")

    # Per-region analysis
    regions = {
        'Jaw (0-16)': range(0, 17),
        'Right eyebrow (17-21)': range(17, 22),
        'Left eyebrow (22-26)': range(22, 27),
        'Nose (27-35)': range(27, 36),
        'Right eye (36-41)': range(36, 42),
        'Left eye (42-47)': range(42, 48),
        'Outer mouth (48-59)': range(48, 60),
        'Inner mouth (60-67)': range(60, 68)
    }

    print("\nPer-Region Mean Error:")
    for region_name, indices in regions.items():
        region_errors = [per_landmark_errors[i] for i in indices]
        region_mean = np.mean([np.mean(e) for e in region_errors if e])
        print(f"  {region_name}: {region_mean:.2f}px")

    # Worst landmarks
    landmark_means = [np.mean(e) if e else 0 for e in per_landmark_errors]
    worst_5 = np.argsort(landmark_means)[-5:][::-1]

    print("\nWorst 5 Landmarks:")
    for idx in worst_5:
        print(f"  Landmark {idx}: {landmark_means[idx]:.2f}px mean error")

    # Accuracy assessment
    print("\n" + "-"*60)
    if overall_mean < 3.0:
        print("EXCELLENT: Mean error < 3px")
    elif overall_mean < 5.0:
        print("GOOD: Mean error < 5px")
    elif overall_mean < 10.0:
        print("ACCEPTABLE: Mean error < 10px")
    else:
        print("POOR: Mean error >= 10px - needs investigation")
    print("-"*60)

if __name__ == '__main__':
    main()
