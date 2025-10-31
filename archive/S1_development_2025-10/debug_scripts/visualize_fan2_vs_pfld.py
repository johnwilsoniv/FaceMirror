#!/usr/bin/env python3
"""Visualize FAN2 vs PFLD vs Ground Truth"""

import cv2
import pandas as pd
import numpy as np

from fan2_landmark_detector import FAN2LandmarkDetector
from pfld_landmark_detector import PFLDLandmarkDetector

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)
gt_x = df[[f'x_{i}' for i in range(68)]].values[0]
gt_y = df[[f'y_{i}' for i in range(68)]].values[0]
gt_landmarks = np.stack([gt_x, gt_y], axis=1)

# Load models
fan2 = FAN2LandmarkDetector('weights/fan2_68_landmark.onnx')
pfld = PFLDLandmarkDetector('weights/pfld_68_landmarks.onnx')

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

# Get bbox
x_min, y_min = gt_landmarks[:, 0].min(), gt_landmarks[:, 1].min()
x_max, y_max = gt_landmarks[:, 0].max(), gt_landmarks[:, 1].max()
pad_w, pad_h = (x_max-x_min)*0.2, (y_max-y_min)*0.2
bbox = [x_min - pad_w, y_min - pad_h, x_max + pad_w, y_max + pad_h]

# Detect with both models
fan2_landmarks, fan2_conf = fan2.detect_landmarks(frame, bbox)
pfld_landmarks = pfld.detect_landmarks(frame, bbox)

# Calculate errors
fan2_rmse = np.sqrt(np.mean(np.sum((fan2_landmarks - gt_landmarks)**2, axis=1)))
pfld_rmse = np.sqrt(np.mean(np.sum((pfld_landmarks - gt_landmarks)**2, axis=1)))

# Create three visualizations
frame_gt = frame.copy()
frame_fan2 = frame.copy()
frame_pfld = frame.copy()

# Ground truth (GREEN)
for x, y in gt_landmarks:
    cv2.circle(frame_gt, (int(x), int(y)), 3, (0, 255, 0), -1)

# FAN2 (BLUE) with GT overlay (GREEN thin)
for x, y in fan2_landmarks:
    cv2.circle(frame_fan2, (int(x), int(y)), 3, (255, 0, 0), -1)
for x, y in gt_landmarks:
    cv2.circle(frame_fan2, (int(x), int(y)), 1, (0, 255, 0), -1)

# PFLD (RED) with GT overlay (GREEN thin)
for x, y in pfld_landmarks:
    cv2.circle(frame_pfld, (int(x), int(y)), 3, (0, 0, 255), -1)
for x, y in gt_landmarks:
    cv2.circle(frame_pfld, (int(x), int(y)), 1, (0, 255, 0), -1)

# Add labels
cv2.putText(frame_gt, "Ground Truth (CSV)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
cv2.putText(frame_fan2, f"FAN2: RMSE={fan2_rmse:.2f}px", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.putText(frame_fan2, "BLUE=FAN2, GREEN=GT", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
cv2.putText(frame_pfld, f"PFLD: RMSE={pfld_rmse:.2f}px", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.putText(frame_pfld, "RED=PFLD, GREEN=GT", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# Save
cv2.imwrite("comparison_ground_truth.jpg", frame_gt)
cv2.imwrite("comparison_fan2.jpg", frame_fan2)
cv2.imwrite("comparison_pfld.jpg", frame_pfld)

print("Saved comparison images:")
print("  comparison_ground_truth.jpg - Ground truth landmarks")
print("  comparison_fan2.jpg - FAN2 (BLUE) vs GT (GREEN)")
print("  comparison_pfld.jpg - PFLD (RED) vs GT (GREEN)")
print()
print(f"FAN2 RMSE: {fan2_rmse:.2f} pixels")
print(f"PFLD RMSE: {pfld_rmse:.2f} pixels")
print(f"Improvement: {((pfld_rmse - fan2_rmse) / pfld_rmse * 100):.1f}%")
