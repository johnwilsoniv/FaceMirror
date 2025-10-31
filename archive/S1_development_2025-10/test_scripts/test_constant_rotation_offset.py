#!/usr/bin/env python3
"""
Test if alignment WITHOUT eyes has a CONSTANT rotational offset

If the offset is constant, we can simply apply a correction factor!
"""

import numpy as np
import pandas as pd
import cv2
from openface22_face_aligner import OpenFace22FaceAligner

# Load data
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Test frames including expression-sensitive ones
test_frames = [1, 124, 247, 370, 493, 617, 740, 863, 986, 1110]

# Temporarily modify rigid indices to exclude eyes
NON_EYE_RIGID = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35]
original_rigid = aligner.RIGID_INDICES
aligner.RIGID_INDICES = NON_EYE_RIGID

print("=" * 70)
print("Testing Constant Rotation Offset (WITHOUT eye landmarks)")
print("=" * 70)

angles = []
for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Extract landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Compute rotation with non-eye rigid points
    source_rigid = aligner._extract_rigid_points(landmarks_68)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)

    # Extract rotation angle
    angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi
    angles.append(angle)

    print(f"Frame {frame_num:4d}: {angle:7.2f}°")

# Restore original rigid indices
aligner.RIGID_INDICES = original_rigid

# Statistics
angles = np.array(angles)
mean_angle = np.mean(angles)
std_angle = np.std(angles)
min_angle = np.min(angles)
max_angle = np.max(angles)
range_angle = max_angle - min_angle

print("\n" + "=" * 70)
print("Statistics:")
print("=" * 70)
print(f"  Mean:   {mean_angle:7.2f}°")
print(f"  Std:    {std_angle:7.2f}°")
print(f"  Min:    {min_angle:7.2f}°")
print(f"  Max:    {max_angle:7.2f}°")
print(f"  Range:  {range_angle:7.2f}°")

print("\n" + "=" * 70)
print("Conclusion:")
print("=" * 70)

if std_angle < 1.0:
    correction = -mean_angle
    print(f"  ✓ EXCELLENT! Rotation is nearly constant (std={std_angle:.2f}°)")
    print(f"  ✓ We can apply a simple correction: rotate by {correction:.2f}°")
    print(f"\n  SOLUTION: Add {correction:.2f}° rotation after Kabsch!")
    print(f"  This will give us:")
    print(f"    - Expression invariance (no eyes)")
    print(f"    - Correct rotation (with correction)")
    print(f"    - Pure Python (no C++ dependency)")
elif std_angle < 2.0:
    correction = -mean_angle
    print(f"  ~ GOOD! Rotation is mostly constant (std={std_angle:.2f}°)")
    print(f"  ~ Could apply correction: rotate by {correction:.2f}°")
    print(f"  ~ Residual variation: {std_angle:.2f}° (acceptable?)")
else:
    print(f"  ✗ Rotation varies too much (std={std_angle:.2f}°)")
    print(f"  ✗ Simple correction won't work")

print("=" * 70)
