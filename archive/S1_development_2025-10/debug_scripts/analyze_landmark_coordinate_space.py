#!/usr/bin/env python3
"""
Analyze if landmarks are in image space or already transformed space
"""

import numpy as np
import pandas as pd
import cv2

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Load video to get image dimensions
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

img_height, img_width = frame.shape[:2]

print("=" * 70)
print("Landmark Coordinate Space Analysis")
print("=" * 70)

print(f"\nImage dimensions: {img_width} × {img_height}")

# Check landmarks from multiple frames
for frame_num in [1, 617, 740]:
    row = df[df['frame'] == frame_num].iloc[0]

    # Extract landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)

    print(f"\nFrame {frame_num}:")
    print(f"  X range: {x.min():.1f} to {x.max():.1f}")
    print(f"  Y range: {y.min():.1f} to {y.max():.1f}")
    print(f"  Centroid: ({x.mean():.1f}, {y.mean():.1f})")

    # Check if landmarks are within image bounds
    in_bounds = (x.min() >= 0 and x.max() < img_width and
                 y.min() >= 0 and y.max() < img_height)

    if in_bounds:
        print(f"  ✓ All landmarks within image bounds")
    else:
        print(f"  ⚠ Some landmarks outside image bounds!")

    # Compare with pose parameters
    pose_tx = row['p_tx']
    pose_ty = row['p_ty']
    print(f"  Pose tx, ty: ({pose_tx:.1f}, {pose_ty:.1f})")

    # Check if pose params match centroid
    if abs(pose_tx - x.mean()) < 10 and abs(pose_ty - y.mean()) < 10:
        print(f"  ✓ Pose params ≈ landmark centroid")
    else:
        diff_x = pose_tx - x.mean()
        diff_y = pose_ty - y.mean()
        print(f"  ⚠ Pose params differ from centroid by ({diff_x:.1f}, {diff_y:.1f})")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("  If landmarks are in [0, img_width] × [0, img_height],")
print("  they are in IMAGE SPACE (pixel coordinates)")
print("  If landmarks are centered around (0, 0),")
print("  they might be in NORMALIZED/MODEL SPACE")
print("=" * 70)
