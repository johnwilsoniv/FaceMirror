#!/usr/bin/env python3
"""Compare Python eye refinement vs C++ ground truth."""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.clnf import CLNF

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
CPP_LANDMARKS_CSV = "/tmp/Shorty_landmarks.csv"
MODEL_DIR = "pyclnf/models"

def load_cpp_landmarks(csv_path, frame_idx=0):
    """Load C++ landmarks from CSV file."""
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Parse header
    header = lines[0].strip().split(',')

    # Find landmark columns
    x_cols = [i for i, h in enumerate(header) if h.strip().startswith('x_')]
    y_cols = [i for i, h in enumerate(header) if h.strip().startswith('y_')]

    # Parse frame data
    data = lines[1 + frame_idx].strip().split(',')

    landmarks = np.zeros((68, 2))
    for i, (xi, yi) in enumerate(zip(x_cols, y_cols)):
        landmarks[i, 0] = float(data[xi])
        landmarks[i, 1] = float(data[yi])

    return landmarks

def main():
    print("=" * 60)
    print("Python Eye Refinement vs C++ Ground Truth")
    print("=" * 60)

    # Load C++ ground truth
    print("\nLoading C++ ground truth...")
    cpp_landmarks = load_cpp_landmarks(CPP_LANDMARKS_CSV)
    print(f"Loaded {len(cpp_landmarks)} landmarks from C++")

    # Extract frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Frame shape: {gray.shape}")

    # Left eye landmarks (36-41)
    LEFT_EYE = list(range(36, 42))

    # Test without eye refinement
    print("\n" + "-" * 60)
    print("Testing WITHOUT eye refinement:")
    print("-" * 60)

    clnf_no_eye = CLNF(MODEL_DIR, use_eye_refinement=False)
    try:
        landmarks1, info1 = clnf_no_eye.detect_and_fit(gray)
    except ValueError as e:
        print(f"Detection failed without eye refinement: {e}")
        return

    # Compute errors for left eye
    errors_no_eye = []
    for i in LEFT_EYE:
        error = np.linalg.norm(landmarks1[i] - cpp_landmarks[i])
        errors_no_eye.append(error)
        print(f"  Landmark {i}: Python={landmarks1[i]}, C++={cpp_landmarks[i]}, error={error:.3f}px")

    mean_no_eye = np.mean(errors_no_eye)
    print(f"\nMean left eye error (no refinement): {mean_no_eye:.3f} px")

    # Test with eye refinement
    print("\n" + "-" * 60)
    print("Testing WITH eye refinement:")
    print("-" * 60)

    clnf_eye = CLNF(MODEL_DIR, use_eye_refinement=True)
    try:
        landmarks2, info2 = clnf_eye.detect_and_fit(gray)
    except ValueError as e:
        print(f"Detection failed with eye refinement: {e}")
        return

    # Compute errors for left eye
    errors_eye = []
    for i in LEFT_EYE:
        error = np.linalg.norm(landmarks2[i] - cpp_landmarks[i])
        errors_eye.append(error)
        print(f"  Landmark {i}: Python={landmarks2[i]}, C++={cpp_landmarks[i]}, error={error:.3f}px")

    mean_eye = np.mean(errors_eye)
    print(f"\nMean left eye error (with refinement): {mean_eye:.3f} px")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean left eye error WITHOUT refinement: {mean_no_eye:.3f} px")
    print(f"Mean left eye error WITH refinement:    {mean_eye:.3f} px")
    print(f"Difference: {mean_no_eye - mean_eye:.3f} px")

    if mean_eye < mean_no_eye:
        improvement = ((mean_no_eye - mean_eye) / mean_no_eye) * 100
        print(f"\n✓ Eye refinement IMPROVES accuracy by {improvement:.1f}%")
        print("  Recommendation: ENABLE eye refinement")
    else:
        degradation = ((mean_eye - mean_no_eye) / mean_no_eye) * 100
        print(f"\n✗ Eye refinement DEGRADES accuracy by {degradation:.1f}%")
        print("  Recommendation: KEEP eye refinement disabled")

if __name__ == "__main__":
    main()
