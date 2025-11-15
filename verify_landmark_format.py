#!/usr/bin/env python3
"""Verify C++ landmark format"""

import pandas as pd
import cv2

# Read C++ debug CSV
df = pd.read_csv("/tmp/mtcnn_debug.csv")
row = df.iloc[0]

bbox_x = row['bbox_x']
bbox_y = row['bbox_y']
bbox_w = row['bbox_w']
bbox_h = row['bbox_h']

print("C++ BBox:")
print(f"  x={bbox_x:.2f}, y={bbox_y:.2f}, w={bbox_w:.2f}, h={bbox_h:.2f}")
print(f"  x1={bbox_x:.2f}, y1={bbox_y:.2f}")
print(f"  x2={bbox_x+bbox_w:.2f}, y2={bbox_y+bbox_h:.2f}")

print("\nC++ Landmarks (from CSV):")
for i in range(1, 6):
    lm_x = row[f'lm{i}_x']
    lm_y = row[f'lm{i}_y']
    print(f"  lm{i}: x={lm_x:.6f}, y={lm_y:.6f}")

print("\nChecking if landmarks are normalized (0-1 range):")
for i in range(1, 6):
    lm_x = row[f'lm{i}_x']
    lm_y = row[f'lm{i}_y']
    is_norm = (0 <= lm_x <= 1) and (0 <= lm_y <= 1)
    print(f"  lm{i}: {is_norm}")

# Load image to check if landmarks could be absolute
img = cv2.imread("calibration_frames/patient1_frame1.jpg")
print(f"\nImage size: {img.shape[1]}x{img.shape[0]}")

print("\nIf landmarks are absolute coordinates:")
for i in range(1, 6):
    lm_x = row[f'lm{i}_x']
    lm_y = row[f'lm{i}_y']
    in_image = (0 <= lm_x <= img.shape[1]) and (0 <= lm_y <= img.shape[0])
    print(f"  lm{i}: ({lm_x:.2f}, {lm_y:.2f}) - in image bounds: {in_image}")
