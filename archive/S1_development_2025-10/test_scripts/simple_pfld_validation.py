#!/usr/bin/env python3
"""
Simple PFLD validation using bounding boxes from CSV

Tests PFLD landmark detector against C++ baseline using bboxes from CSV
instead of running RetinaFace detection (which is slow).
"""

import cv2
import pandas as pd
import numpy as np
import sys

from pfld_landmark_detector import PFLDLandmarkDetector

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PFLD_MODEL = "weights/pfld_68_landmarks.onnx"

def load_csv_landmarks(csv_path):
    """Load 68 landmarks from CSV baseline."""
    df = pd.read_csv(csv_path)

    # Extract landmark columns
    landmark_cols_x = [f'x_{i}' for i in range(68)]
    landmark_cols_y = [f'y_{i}' for i in range(68)]

    landmarks_x = df[landmark_cols_x].values
    landmarks_y = df[landmark_cols_y].values
    landmarks = np.stack([landmarks_x, landmarks_y], axis=2)

    return landmarks, df['success'].values

def calculate_landmark_rmse(pred_landmarks, gt_landmarks):
    """Calculate RMSE between predicted and ground truth landmarks."""
    if pred_landmarks is None or gt_landmarks is None:
        return None

    diff = pred_landmarks - gt_landmarks
    squared_errors = np.sum(diff ** 2, axis=1)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return rmse

def estimate_bbox_from_landmarks(landmarks):
    """Estimate bounding box from landmark points."""
    x_min = landmarks[:, 0].min()
    y_min = landmarks[:, 1].min()
    x_max = landmarks[:, 0].max()
    y_max = landmarks[:, 1].max()

    # Add 20% padding
    width = x_max - x_min
    height = y_max - y_min
    pad_w = width * 0.2
    pad_h = height * 0.2

    bbox = [x_min - pad_w, y_min - pad_h, x_max + pad_w, y_max + pad_h]
    return bbox

def main():
    print("=" * 80)
    print("SIMPLE PFLD VALIDATION (using CSV bounding boxes)")
    print("=" * 80)
    print()

    # Load CSV baseline
    print(f"Loading baseline: {CSV_PATH}")
    csv_landmarks, csv_success = load_csv_landmarks(CSV_PATH)
    total_frames = len(csv_landmarks)
    print(f"  Total frames: {total_frames}")
    print()

    # Load PFLD
    print(f"Loading PFLD: {PFLD_MODEL}")
    detector = PFLDLandmarkDetector(PFLD_MODEL)
    print("  Model loaded")
    print()

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return
    print("  Video opened")
    print()

    # Test on first 50 frames
    TEST_FRAMES = 50
    print(f"Testing on first {TEST_FRAMES} frames...")
    print()

    results = []
    frame_idx = 0

    while frame_idx < TEST_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # Get ground truth landmarks and estimate bbox
        gt_landmarks = csv_landmarks[frame_idx]
        bbox = estimate_bbox_from_landmarks(gt_landmarks)

        # Detect landmarks with PFLD
        pfld_landmarks = detector.detect_landmarks(frame, bbox)

        if pfld_landmarks is not None:
            rmse = calculate_landmark_rmse(pfld_landmarks, gt_landmarks)
            results.append({
                'frame': frame_idx + 1,
                'detected': 1,
                'rmse': rmse
            })
        else:
            results.append({
                'frame': frame_idx + 1,
                'detected': 0,
                'rmse': None
            })

        frame_idx += 1

        if (frame_idx % 10) == 0:
            print(f"  Processed {frame_idx}/{TEST_FRAMES} frames")

    cap.release()
    print(f"  Completed: {frame_idx} frames")
    print()

    # Calculate statistics
    results_df = pd.DataFrame(results)
    detected_results = results_df[results_df['detected'] == 1]

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    pfld_detections = results_df['detected'].sum()
    print(f"Detection rate: {pfld_detections}/{frame_idx} ({100*pfld_detections/frame_idx:.1f}%)")
    print()

    if len(detected_results) > 0:
        mean_rmse = detected_results['rmse'].mean()
        median_rmse = detected_results['rmse'].median()
        max_rmse = detected_results['rmse'].max()
        min_rmse = detected_results['rmse'].min()

        print(f"Landmark Accuracy:")
        print(f"  Mean RMSE:   {mean_rmse:.2f} pixels")
        print(f"  Median RMSE: {median_rmse:.2f} pixels")
        print(f"  Min RMSE:    {min_rmse:.2f} pixels")
        print(f"  Max RMSE:    {max_rmse:.2f} pixels")
        print()

        # Verdict
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print()

        if mean_rmse < 3.0:
            print(f"✅ EXCELLENT: Mean RMSE = {mean_rmse:.2f} pixels (< 3 pixel target)")
            print("   PFLD landmarks are highly accurate")
            print("   Component 3: VALIDATED ✅")
        elif mean_rmse < 5.0:
            print(f"✓ GOOD: Mean RMSE = {mean_rmse:.2f} pixels (< 5 pixels)")
            print("   PFLD landmarks are acceptable")
            print("   Component 3: USABLE ✓")
        else:
            print(f"⚠️  HIGH ERROR: Mean RMSE = {mean_rmse:.2f} pixels")
            print("   PFLD needs tuning")
            print("   Component 3: NEEDS REVIEW ⚠️")
    else:
        print("❌ FAILED: No detections")

    print()

    # Save results
    output_file = "simple_pfld_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
