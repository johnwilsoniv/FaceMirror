#!/usr/bin/env python3
"""Compare MTCNN landmarks - convert PyMTCNN to normalized format"""

import numpy as np
import cv2
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
from pymtcnn.backends.onnx_backend import ONNXMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("MTCNN Landmark Comparison - Normalized Coordinates")
print("="*80)

# Load image
img = cv2.imread(TEST_IMAGE)
print(f"\nTest image: {TEST_IMAGE} ({img.shape[1]}x{img.shape[0]})")

# Get C++ landmarks (normalized)
df = pd.read_csv("/tmp/mtcnn_debug.csv")
row = df.iloc[0]

cpp_bbox_x = row['bbox_x']
cpp_bbox_y = row['bbox_y']
cpp_bbox_w = row['bbox_w']
cpp_bbox_h = row['bbox_h']

cpp_landmarks_norm = np.zeros((5, 2))
for i in range(1, 6):
    cpp_landmarks_norm[i-1, 0] = row[f'lm{i}_x']
    cpp_landmarks_norm[i-1, 1] = row[f'lm{i}_y']

print("\n[1/3] C++ MTCNN (normalized coordinates):")
print(f"  BBox: x={cpp_bbox_x:.2f}, y={cpp_bbox_y:.2f}, w={cpp_bbox_w:.2f}, h={cpp_bbox_h:.2f}")
print(f"  Landmarks (normalized):")
for i, (x, y) in enumerate(cpp_landmarks_norm):
    print(f"    Point {i}: ({x:.6f}, {y:.6f})")

# Get CoreML landmarks (absolute) and convert to normalized
coreml_detector = CoreMLMTCNN(verbose=False)
coreml_bboxes, coreml_landmarks = coreml_detector.detect(img)

if len(coreml_bboxes) == 0:
    print("✗ CoreML: No face detected")
    sys.exit(1)

coreml_bbox = coreml_bboxes[0]  # [x, y, w, h]
coreml_landmarks_abs = coreml_landmarks[0]  # (5, 2) absolute coordinates

# Convert to normalized
coreml_landmarks_norm = np.zeros((5, 2))
for i in range(5):
    coreml_landmarks_norm[i, 0] = (coreml_landmarks_abs[i, 0] - coreml_bbox[0]) / coreml_bbox[2]
    coreml_landmarks_norm[i, 1] = (coreml_landmarks_abs[i, 1] - coreml_bbox[1]) / coreml_bbox[3]

print("\n[2/3] CoreML MTCNN (converted to normalized):")
print(f"  BBox: x={coreml_bbox[0]:.2f}, y={coreml_bbox[1]:.2f}, w={coreml_bbox[2]:.2f}, h={coreml_bbox[3]:.2f}")
print(f"  Landmarks (normalized):")
for i, (x, y) in enumerate(coreml_landmarks_norm):
    print(f"    Point {i}: ({x:.6f}, {y:.6f})")

# Get ONNX landmarks (absolute) and convert to normalized
onnx_detector = ONNXMTCNN(verbose=False)
onnx_bboxes, onnx_landmarks = onnx_detector.detect(img)

if len(onnx_bboxes) == 0:
    print("✗ ONNX: No face detected")
    sys.exit(1)

onnx_bbox = onnx_bboxes[0]  # [x, y, w, h]
onnx_landmarks_abs = onnx_landmarks[0]  # (5, 2) absolute coordinates

# Convert to normalized
onnx_landmarks_norm = np.zeros((5, 2))
for i in range(5):
    onnx_landmarks_norm[i, 0] = (onnx_landmarks_abs[i, 0] - onnx_bbox[0]) / onnx_bbox[2]
    onnx_landmarks_norm[i, 1] = (onnx_landmarks_abs[i, 1] - onnx_bbox[1]) / onnx_bbox[3]

print("\n[3/3] ONNX MTCNN (converted to normalized):")
print(f"  BBox: x={onnx_bbox[0]:.2f}, y={onnx_bbox[1]:.2f}, w={onnx_bbox[2]:.2f}, h={onnx_bbox[3]:.2f}")
print(f"  Landmarks (normalized):")
for i, (x, y) in enumerate(onnx_landmarks_norm):
    print(f"    Point {i}: ({x:.6f}, {y:.6f})")

# Compare landmarks
print("\n" + "="*80)
print("Landmark Error Analysis (Normalized Coordinates)")
print("="*80)

landmark_names = ['Left eye', 'Right eye', 'Nose', 'Left mouth', 'Right mouth']

# C++ vs CoreML
cpp_coreml_errors = np.sqrt(np.sum((cpp_landmarks_norm - coreml_landmarks_norm)**2, axis=1))
print("\n[C++ vs CoreML]")
print(f"  Mean error: {np.mean(cpp_coreml_errors):.6f} (normalized units)")
print(f"  Max error:  {np.max(cpp_coreml_errors):.6f} (normalized units)")
print(f"  Per-point errors:")
for name, error in zip(landmark_names, cpp_coreml_errors):
    print(f"    {name:12s}: {error:.6f}")

# C++ vs ONNX
cpp_onnx_errors = np.sqrt(np.sum((cpp_landmarks_norm - onnx_landmarks_norm)**2, axis=1))
print("\n[C++ vs ONNX]")
print(f"  Mean error: {np.mean(cpp_onnx_errors):.6f} (normalized units)")
print(f"  Max error:  {np.max(cpp_onnx_errors):.6f} (normalized units)")
print(f"  Per-point errors:")
for name, error in zip(landmark_names, cpp_onnx_errors):
    print(f"    {name:12s}: {error:.6f}")

# CoreML vs ONNX
coreml_onnx_errors = np.sqrt(np.sum((coreml_landmarks_norm - onnx_landmarks_norm)**2, axis=1))
print("\n[CoreML vs ONNX]")
print(f"  Mean error: {np.mean(coreml_onnx_errors):.6f} (normalized units)")
print(f"  Max error:  {np.max(coreml_onnx_errors):.6f} (normalized units)")
print(f"  Per-point errors:")
for name, error in zip(landmark_names, coreml_onnx_errors):
    print(f"    {name:12s}: {error:.6f}")

# Convert normalized errors to pixel errors (using bbox width/height)
print("\n" + "="*80)
print("Pixel Error Analysis")
print("="*80)

# Use average bbox size for pixel conversion
avg_bbox_w = (cpp_bbox_w + coreml_bbox[2] + onnx_bbox[2]) / 3
avg_bbox_h = (cpp_bbox_h + coreml_bbox[3] + onnx_bbox[3]) / 3

print(f"\nAverage bbox size: w={avg_bbox_w:.1f}, h={avg_bbox_h:.1f}")

cpp_coreml_pixels = cpp_coreml_errors * np.sqrt(avg_bbox_w**2 + avg_bbox_h**2) / np.sqrt(2)
cpp_onnx_pixels = cpp_onnx_errors * np.sqrt(avg_bbox_w**2 + avg_bbox_h**2) / np.sqrt(2)

print(f"\n[C++ vs CoreML] (approximate pixel error):")
print(f"  Mean: {np.mean(cpp_coreml_pixels):.2f} pixels")
print(f"  Max:  {np.max(cpp_coreml_pixels):.2f} pixels")

print(f"\n[C++ vs ONNX] (approximate pixel error):")
print(f"  Mean: {np.mean(cpp_onnx_pixels):.2f} pixels")
print(f"  Max:  {np.max(cpp_onnx_pixels):.2f} pixels")

print("\n" + "="*80)
