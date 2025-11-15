#!/usr/bin/env python3
"""Check exact bbox coordinates and landmark normalization"""

import numpy as np
import cv2
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

# Load image
img = cv2.imread(TEST_IMAGE)
print(f"Image: {TEST_IMAGE} ({img.shape[1]}x{img.shape[0]})")

# Get C++ data
df = pd.read_csv("/tmp/mtcnn_debug.csv")
row = df.iloc[0]

cpp_bbox_x = row['bbox_x']
cpp_bbox_y = row['bbox_y']
cpp_bbox_w = row['bbox_w']
cpp_bbox_h = row['bbox_h']

print("\n" + "="*80)
print("C++ MTCNN (from debug CSV)")
print("="*80)
print(f"BBox: x={cpp_bbox_x:.4f}, y={cpp_bbox_y:.4f}, w={cpp_bbox_w:.4f}, h={cpp_bbox_h:.4f}")
print(f"BBox corners: ({cpp_bbox_x:.4f}, {cpp_bbox_y:.4f}) -> ({cpp_bbox_x+cpp_bbox_w:.4f}, {cpp_bbox_y+cpp_bbox_h:.4f})")

# Get PyMTCNN data
detector = CoreMLMTCNN(verbose=False)
bboxes, landmarks = detector.detect(img)

if len(bboxes) == 0:
    print("No face detected")
    sys.exit(1)

bbox = bboxes[0]  # [x, y, w, h]
print("\n" + "="*80)
print("PyMTCNN CoreML")
print("="*80)
print(f"BBox: x={bbox[0]:.4f}, y={bbox[1]:.4f}, w={bbox[2]:.4f}, h={bbox[3]:.4f}")
print(f"BBox corners: ({bbox[0]:.4f}, {bbox[1]:.4f}) -> ({bbox[0]+bbox[2]:.4f}, {bbox[1]+bbox[3]:.4f})")

# Compute IoU
x1_int = max(cpp_bbox_x, bbox[0])
y1_int = max(cpp_bbox_y, bbox[1])
x2_int = min(cpp_bbox_x + cpp_bbox_w, bbox[0] + bbox[2])
y2_int = min(cpp_bbox_y + cpp_bbox_h, bbox[1] + bbox[3])

if x2_int > x1_int and y2_int > y1_int:
    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area_cpp = cpp_bbox_w * cpp_bbox_h
    area_pymtcnn = bbox[2] * bbox[3]
    union = area_cpp + area_pymtcnn - intersection
    iou = intersection / union
    print(f"\nBBox IoU: {iou:.6f}")

# Compare landmarks in absolute coordinates
print("\n" + "="*80)
print("Landmarks Comparison (Absolute Pixel Coordinates)")
print("="*80)

# C++ landmarks (convert from normalized)
cpp_landmarks = np.zeros((5, 2))
for i in range(1, 6):
    lm_x_norm = row[f'lm{i}_x']
    lm_y_norm = row[f'lm{i}_y']
    cpp_landmarks[i-1, 0] = cpp_bbox_x + lm_x_norm * cpp_bbox_w
    cpp_landmarks[i-1, 1] = cpp_bbox_y + lm_y_norm * cpp_bbox_h

# PyMTCNN landmarks (already in absolute)
pymtcnn_landmarks = landmarks[0]

print("\nC++ MTCNN landmarks:")
for i, (x, y) in enumerate(cpp_landmarks):
    print(f"  Point {i}: ({x:.4f}, {y:.4f})")

print("\nPyMTCNN CoreML landmarks:")
for i, (x, y) in enumerate(pymtcnn_landmarks):
    print(f"  Point {i}: ({x:.4f}, {y:.4f})")

print("\nDifference (PyMTCNN - C++):")
diffs = pymtcnn_landmarks - cpp_landmarks
for i, (dx, dy) in enumerate(diffs):
    dist = np.sqrt(dx**2 + dy**2)
    print(f"  Point {i}: dx={dx:+.4f}, dy={dy:+.4f}, dist={dist:.4f} px")

print(f"\nMean error: {np.mean(np.sqrt(np.sum(diffs**2, axis=1))):.4f} pixels")

# Now check: what if C++ landmarks are stored relative to a DIFFERENT bbox?
# Let me check if C++ stores landmarks before or after final calibration
print("\n" + "="*80)
print("Hypothesis: C++ landmarks stored relative to PRE-calibration bbox?")
print("="*80)

# What was the bbox BEFORE calibration in C++?
# We need to reverse the calibration:
# new_x1 = x1 + w * -0.0075
# new_y1 = y1 + h * 0.2459
# new_w = w * 1.0323
# new_h = h * 0.7751

# Reverse:
# w = new_w / 1.0323
# h = new_h / 0.7751
# x1 = new_x1 - w * -0.0075 = new_x1 + w * 0.0075
# y1 = new_y1 - h * 0.2459

cpp_precal_w = cpp_bbox_w / 1.0323
cpp_precal_h = cpp_bbox_h / 0.7751
cpp_precal_x = cpp_bbox_x + cpp_precal_w * 0.0075
cpp_precal_y = cpp_bbox_y - cpp_precal_h * 0.2459

print(f"C++ pre-calibration bbox (estimated):")
print(f"  x={cpp_precal_x:.4f}, y={cpp_precal_y:.4f}, w={cpp_precal_w:.4f}, h={cpp_precal_h:.4f}")

# Convert C++ landmarks using pre-cal bbox
cpp_landmarks_precal = np.zeros((5, 2))
for i in range(1, 6):
    lm_x_norm = row[f'lm{i}_x']
    lm_y_norm = row[f'lm{i}_y']
    cpp_landmarks_precal[i-1, 0] = cpp_precal_x + lm_x_norm * cpp_precal_w
    cpp_landmarks_precal[i-1, 1] = cpp_precal_y + lm_y_norm * cpp_precal_h

print("\nC++ landmarks (if normalized to PRE-calibration bbox):")
for i, (x, y) in enumerate(cpp_landmarks_precal):
    print(f"  Point {i}: ({x:.4f}, {y:.4f})")

print("\nDifference vs PyMTCNN (if C++ used pre-cal bbox):")
diffs_precal = pymtcnn_landmarks - cpp_landmarks_precal
for i, (dx, dy) in enumerate(diffs_precal):
    dist = np.sqrt(dx**2 + dy**2)
    print(f"  Point {i}: dx={dx:+.4f}, dy={dy:+.4f}, dist={dist:.4f} px")

print(f"\nMean error: {np.mean(np.sqrt(np.sum(diffs_precal**2, axis=1))):.4f} pixels")
