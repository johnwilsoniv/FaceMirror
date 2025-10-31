#!/usr/bin/env python3
"""Test FAN2 landmark accuracy against CSV baseline"""

import cv2
import pandas as pd
import numpy as np

from fan2_landmark_detector import FAN2LandmarkDetector

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
FAN2_MODEL = "weights/fan2_68_landmark.onnx"

def load_csv_landmarks(csv_path):
    """Load 68 landmarks from CSV baseline."""
    df = pd.read_csv(csv_path)
    landmark_cols_x = [f'x_{i}' for i in range(68)]
    landmark_cols_y = [f'y_{i}' for i in range(68)]
    landmarks_x = df[landmark_cols_x].values
    landmarks_y = df[landmark_cols_y].values
    landmarks = np.stack([landmarks_x, landmarks_y], axis=2)
    return landmarks, df['success'].values

def estimate_bbox_from_landmarks(landmarks):
    """Estimate bounding box from landmark points with padding."""
    x_min = landmarks[:, 0].min()
    y_min = landmarks[:, 1].min()
    x_max = landmarks[:, 0].max()
    y_max = landmarks[:, 1].max()

    # Add 20% padding (same as PFLD test)
    width = x_max - x_min
    height = y_max - y_min
    pad_w = width * 0.2
    pad_h = height * 0.2

    bbox = [x_min - pad_w, y_min - pad_h, x_max + pad_w, y_max + pad_h]
    return bbox

def calculate_rmse(pred, gt):
    """Calculate RMSE between predicted and ground truth landmarks."""
    if pred is None or gt is None:
        return None
    diff = pred - gt
    mse = np.mean(np.sum(diff ** 2, axis=1))
    return np.sqrt(mse)

def main():
    print("=" * 80)
    print("FAN2 ACCURACY TEST vs C++ OpenFace Baseline")
    print("=" * 80)
    print()

    # Load CSV baseline
    print(f"Loading baseline: {CSV_PATH}")
    csv_landmarks, csv_success = load_csv_landmarks(CSV_PATH)
    print(f"  Total frames: {len(csv_landmarks)}")
    print()

    # Load FAN2
    print(f"Loading FAN2: {FAN2_MODEL}")
    detector = FAN2LandmarkDetector(FAN2_MODEL)
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

        # Get ground truth and bbox
        gt_landmarks = csv_landmarks[frame_idx]
        bbox = estimate_bbox_from_landmarks(gt_landmarks)

        # Detect with FAN2
        fan2_landmarks, confidences = detector.detect_landmarks(frame, bbox)

        if fan2_landmarks is not None:
            rmse = calculate_rmse(fan2_landmarks, gt_landmarks)
            mean_conf = confidences.mean()
            results.append({
                'frame': frame_idx + 1,
                'detected': 1,
                'rmse': rmse,
                'mean_confidence': mean_conf
            })
        else:
            results.append({
                'frame': frame_idx + 1,
                'detected': 0,
                'rmse': None,
                'mean_confidence': 0.0
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

    detection_rate = results_df['detected'].sum()
    print(f"Detection rate: {detection_rate}/{frame_idx} ({100*detection_rate/frame_idx:.1f}%)")
    print()

    if len(detected_results) > 0:
        mean_rmse = detected_results['rmse'].mean()
        median_rmse = detected_results['rmse'].median()
        max_rmse = detected_results['rmse'].max()
        min_rmse = detected_results['rmse'].min()
        mean_conf = detected_results['mean_confidence'].mean()

        print(f"Landmark Accuracy:")
        print(f"  Mean RMSE:   {mean_rmse:.2f} pixels")
        print(f"  Median RMSE: {median_rmse:.2f} pixels")
        print(f"  Min RMSE:    {min_rmse:.2f} pixels")
        print(f"  Max RMSE:    {max_rmse:.2f} pixels")
        print()
        print(f"Mean confidence: {mean_conf:.3f}")
        print()

        # Comparison with PFLD
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print()
        print(f"PFLD (previous):  Mean RMSE = 13.26 pixels")
        print(f"FAN2 (current):   Mean RMSE = {mean_rmse:.2f} pixels")
        if mean_rmse < 13.26:
            improvement = ((13.26 - mean_rmse) / 13.26) * 100
            print(f"Improvement:      {improvement:.1f}% better")
        print()

        # Verdict
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print()

        if mean_rmse < 3.0:
            print(f"✅ EXCELLENT: Mean RMSE = {mean_rmse:.2f} pixels (< 3 pixel target)")
            print("   FAN2 landmarks are highly accurate")
            print("   Component 3: VALIDATED ✅")
        elif mean_rmse < 5.0:
            print(f"✓ GOOD: Mean RMSE = {mean_rmse:.2f} pixels (< 5 pixels)")
            print("   FAN2 landmarks are acceptable")
            print("   Component 3: USABLE ✓")
        elif mean_rmse < 10.0:
            print(f"△ FAIR: Mean RMSE = {mean_rmse:.2f} pixels (< 10 pixels)")
            print("   FAN2 better than PFLD but not ideal")
            print("   Component 3: NEEDS REVIEW △")
        else:
            print(f"⚠️  HIGH ERROR: Mean RMSE = {mean_rmse:.2f} pixels")
            print("   FAN2 needs tuning or different model")
            print("   Component 3: NEEDS IMPROVEMENT ⚠️")
    else:
        print("❌ FAILED: No detections")

    print()

    # Save results
    output_file = "fan2_accuracy_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
