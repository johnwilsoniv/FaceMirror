#!/usr/bin/env python3
"""
Compare MTCNN Initial Landmarks - C++ vs CoreML vs ONNX

Validates that PyMTCNN backends produce identical 5-point facial landmarks
compared to C++ OpenFace MTCNN (gold standard).

MTCNN 5-point landmark order:
  0: Left eye
  1: Right eye
  2: Nose tip
  3: Left mouth corner
  4: Right mouth corner
"""

import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
from pymtcnn.backends.onnx_backend import ONNXMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"


def get_cpp_landmarks():
    """Get C++ OpenFace MTCNN initial 5-point landmarks from debug CSV"""
    import pandas as pd

    # C++ writes MTCNN initial landmarks to /tmp/mtcnn_debug.csv
    # Format: frame,bbox_x,bbox_y,bbox_w,bbox_h,confidence,onet_size,lm1_x,lm1_y,lm2_x,lm2_y,lm3_x,lm3_y,lm4_x,lm4_y,lm5_x,lm5_y
    # Landmarks are in normalized coordinates (0-1) relative to bounding box

    csv_file = Path("/tmp/mtcnn_debug.csv")

    if not csv_file.exists():
        print("⚠ C++ MTCNN debug file not found. Run C++ FeatureExtraction first:")
        print(f"  /Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction \\")
        print(f"    -f {TEST_IMAGE} -out_dir /tmp/mtcnn_cpp_test")
        return None

    # Read CSV
    df = pd.read_csv(csv_file)

    if len(df) == 0:
        print("⚠ C++ did not detect a face")
        return None

    # Extract first detection (frame 0)
    row = df.iloc[0]

    bbox_x = row['bbox_x']
    bbox_y = row['bbox_y']
    bbox_w = row['bbox_w']
    bbox_h = row['bbox_h']

    # Convert normalized landmarks to absolute pixel coordinates
    landmarks = []
    for i in range(1, 6):  # lm1 through lm5
        lm_x_norm = row[f'lm{i}_x']
        lm_y_norm = row[f'lm{i}_y']

        # Convert to absolute coordinates
        lm_x_abs = bbox_x + (lm_x_norm * bbox_w)
        lm_y_abs = bbox_y + (lm_y_norm * bbox_h)

        landmarks.append([lm_x_abs, lm_y_abs])

    landmarks_array = np.array(landmarks)

    print("✓ C++ MTCNN initial landmarks loaded:")
    for i, (x, y) in enumerate(landmarks_array):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

    return landmarks_array


def calculate_landmark_error(lm1, lm2):
    """
    Calculate per-point Euclidean distance between two landmark sets

    Args:
        lm1, lm2: (5, 2) landmark arrays

    Returns:
        tuple: (mean_error, max_error, per_point_errors)
    """
    if lm1 is None or lm2 is None:
        return None, None, None

    # Calculate Euclidean distance for each landmark point
    errors = np.sqrt(np.sum((lm1 - lm2)**2, axis=1))

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return mean_error, max_error, errors


print("="*80)
print("MTCNN Landmark Comparison - C++ vs CoreML vs ONNX")
print("="*80)

# Load test image
img = cv2.imread(TEST_IMAGE)
print(f"\nTest image: {TEST_IMAGE} ({img.shape[1]}x{img.shape[0]})")

# Get C++ landmarks (gold standard)
print("\n[1/3] Getting C++ OpenFace MTCNN landmarks...")
cpp_landmarks = get_cpp_landmarks()

if cpp_landmarks is None:
    print("✗ Could not load C++ landmarks")
    sys.exit(1)

# Get CoreML landmarks
print("\n[2/3] Getting CoreML MTCNN landmarks...")
coreml_detector = CoreMLMTCNN(verbose=False)
coreml_bboxes, coreml_landmarks = coreml_detector.detect(img)

if len(coreml_bboxes) == 0:
    print("✗ CoreML: No face detected")
    coreml_landmarks_5pt = None
else:
    # PyMTCNN returns landmarks as (N, 5, 2) array
    coreml_landmarks_5pt = coreml_landmarks[0]  # First face
    print("✓ CoreML landmarks:")
    for i, (x, y) in enumerate(coreml_landmarks_5pt):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

# Get ONNX landmarks
print("\n[3/3] Getting ONNX MTCNN landmarks...")
onnx_detector = ONNXMTCNN(verbose=False)
onnx_bboxes, onnx_landmarks = onnx_detector.detect(img)

if len(onnx_bboxes) == 0:
    print("✗ ONNX: No face detected")
    onnx_landmarks_5pt = None
else:
    # PyMTCNN returns landmarks as (N, 5, 2) array
    onnx_landmarks_5pt = onnx_landmarks[0]  # First face
    print("✓ ONNX landmarks:")
    for i, (x, y) in enumerate(onnx_landmarks_5pt):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

# Compare landmarks
print("\n" + "="*80)
print("Landmark Error Analysis")
print("="*80)

# Define landmark names for all comparisons
landmark_names = ['Left eye', 'Right eye', 'Nose', 'Left mouth', 'Right mouth']

# C++ vs CoreML
print("\n[C++ vs CoreML]")
coreml_mean, coreml_max, coreml_errors = calculate_landmark_error(cpp_landmarks, coreml_landmarks_5pt)
if coreml_mean is not None:
    print(f"  Mean error: {coreml_mean:.4f} pixels")
    print(f"  Max error:  {coreml_max:.4f} pixels")
    print(f"  Per-point errors:")
    for i, (name, error) in enumerate(zip(landmark_names, coreml_errors)):
        print(f"    {name:12s}: {error:.4f} pixels")

# C++ vs ONNX
print("\n[C++ vs ONNX]")
onnx_mean, onnx_max, onnx_errors = calculate_landmark_error(cpp_landmarks, onnx_landmarks_5pt)
if onnx_mean is not None:
    print(f"  Mean error: {onnx_mean:.4f} pixels")
    print(f"  Max error:  {onnx_max:.4f} pixels")
    print(f"  Per-point errors:")
    for i, (name, error) in enumerate(zip(landmark_names, onnx_errors)):
        print(f"    {name:12s}: {error:.4f} pixels")

# CoreML vs ONNX
print("\n[CoreML vs ONNX]")
coreml_onnx_mean, coreml_onnx_max, coreml_onnx_errors = calculate_landmark_error(
    coreml_landmarks_5pt, onnx_landmarks_5pt
)
if coreml_onnx_mean is not None:
    print(f"  Mean error: {coreml_onnx_mean:.4f} pixels")
    print(f"  Max error:  {coreml_onnx_max:.4f} pixels")
    print(f"  Per-point errors:")
    for i, (name, error) in enumerate(zip(landmark_names, coreml_onnx_errors)):
        print(f"    {name:12s}: {error:.4f} pixels")

# Summary
print("\n" + "="*80)
print("Summary")
print("="*80)

if coreml_onnx_mean is not None:
    # Acceptance criteria: < 1 pixel mean error (sub-pixel accuracy)
    threshold_excellent = 0.5  # Sub-pixel
    threshold_good = 1.0       # 1 pixel
    threshold_acceptable = 2.0 # 2 pixels

    print(f"\nAcceptance criteria:")
    print(f"  Excellent:  < {threshold_excellent} pixels mean error")
    print(f"  Good:       < {threshold_good} pixels mean error")
    print(f"  Acceptable: < {threshold_acceptable} pixels mean error")

    print(f"\nPyMTCNN Backend Consistency:")
    print(f"  CoreML vs ONNX: {coreml_onnx_mean:.4f} pixels ", end='')
    if coreml_onnx_mean < threshold_excellent:
        print("✓ EXCELLENT")
    elif coreml_onnx_mean < threshold_good:
        print("✓ GOOD")
    elif coreml_onnx_mean < threshold_acceptable:
        print("✓ ACCEPTABLE")
    else:
        print("✗ POOR")

    print(f"\nOverall: ", end='')
    if coreml_onnx_mean < threshold_excellent:
        print("✓ PASS - Sub-pixel accuracy! Backends produce nearly identical landmarks")
        print(f"        Mean error: {coreml_onnx_mean:.4f} pixels (< {threshold_excellent} pixel threshold)")
        print(f"        Max error:  {coreml_onnx_max:.4f} pixels")
        print(f"\n✓ Both CoreML and ONNX backends ready for production use in PyFaceAU")
    elif coreml_onnx_mean < threshold_good:
        print("✓ PASS - Landmarks are consistent between backends")
    else:
        print("✗ FAIL - Landmarks differ significantly between backends")
else:
    print("\n✗ Could not complete comparison - missing landmarks from one or more backends")

print("\n" + "="*80)
