#!/usr/bin/env python3
"""Debug PFLD coordinate system by visualizing predictions vs ground truth"""

import cv2
import pandas as pd
import numpy as np

from pfld_landmark_detector import PFLDLandmarkDetector

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PFLD_MODEL = "weights/pfld_68_landmarks.onnx"

# Load CSV
df = pd.read_csv(CSV_PATH)
landmark_cols_x = [f'x_{i}' for i in range(68)]
landmark_cols_y = [f'y_{i}' for i in range(68)]
landmarks_x = df[landmark_cols_x].values
landmarks_y = df[landmark_cols_y].values
csv_landmarks = np.stack([landmarks_x, landmarks_y], axis=2)

# Load PFLD
detector = PFLDLandmarkDetector(PFLD_MODEL)

# Open video and read first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read frame")
    exit(1)

# Get ground truth landmarks for frame 0
gt_landmarks = csv_landmarks[0]

# Estimate bbox from GT landmarks
x_min = gt_landmarks[:, 0].min()
y_min = gt_landmarks[:, 1].min()
x_max = gt_landmarks[:, 0].max()
y_max = gt_landmarks[:, 1].max()
width = x_max - x_min
height = y_max - y_min
pad_w = width * 0.2
pad_h = height * 0.2
bbox = [x_min - pad_w, y_min - pad_h, x_max + pad_w, y_max + pad_h]

print(f"Frame shape: {frame.shape}")
print(f"Ground truth bbox: {bbox}")
print(f"GT landmarks range X: {x_min:.1f} to {x_max:.1f}")
print(f"GT landmarks range Y: {y_min:.1f} to {y_max:.1f}")
print()

# Detect with PFLD
pfld_landmarks = detector.detect_landmarks(frame, bbox)

print(f"PFLD landmarks shape: {pfld_landmarks.shape}")
print(f"PFLD landmarks range X: {pfld_landmarks[:, 0].min():.1f} to {pfld_landmarks[:, 0].max():.1f}")
print(f"PFLD landmarks range Y: {pfld_landmarks[:, 1].min():.1f} to {pfld_landmarks[:, 1].max():.1f}")
print()

# Calculate per-landmark errors
errors = np.linalg.norm(pfld_landmarks - gt_landmarks, axis=1)
print(f"Per-landmark errors:")
print(f"  Mean: {errors.mean():.2f} pixels")
print(f"  Median: {np.median(errors):.2f} pixels")
print(f"  Max: {errors.max():.2f} pixels (point {errors.argmax()})")
print(f"  Min: {errors.min():.2f} pixels (point {errors.argmin()})")
print()

# Check first 5 landmarks
print("First 5 landmarks comparison:")
for i in range(5):
    gt = gt_landmarks[i]
    pred = pfld_landmarks[i]
    error = errors[i]
    print(f"  Point {i}: GT ({gt[0]:.1f}, {gt[1]:.1f}) vs PFLD ({pred[0]:.1f}, {pred[1]:.1f}) error={error:.2f}px")
print()

# Visualize
frame_vis = frame.copy()

# Draw ground truth in GREEN
for i, (x, y) in enumerate(gt_landmarks):
    cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 255, 0), -1)

# Draw PFLD predictions in RED
for i, (x, y) in enumerate(pfld_landmarks):
    cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 0, 255), -1)

# Draw bbox in BLUE
x1, y1, x2, y2 = map(int, bbox)
cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Save visualization
output_path = "pfld_debug_frame0.jpg"
cv2.imwrite(output_path, frame_vis)
print(f"Visualization saved to: {output_path}")
print(f"  GREEN = Ground truth (CSV)")
print(f"  RED = PFLD predictions")
print(f"  BLUE = Bounding box")
