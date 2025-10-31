#!/usr/bin/env python3
"""
Measure rotation angles across all test frames to verify consistency
"""

import numpy as np
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner

# Load aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Test frames
test_frames = [1, 124, 247, 370, 493, 617, 740, 863, 986, 1110]

print("=" * 70)
print("Rotation Angle Consistency Analysis")
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

    pose_tx = row['p_tx']
    pose_ty = row['p_ty']

    # Extract rigid points
    source_rigid = aligner._extract_rigid_points(landmarks_68)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)

    # Compute scale-rotation matrix
    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)

    # Extract rotation angle
    angle = np.arctan2(scale_rot[1, 0], scale_rot[0, 0]) * 180 / np.pi

    # Extract scale
    scale = np.sqrt(np.linalg.det(scale_rot))

    angles.append(angle)

    print(f"Frame {frame_num:4d}: rotation = {angle:7.2f}°, scale = {scale:.4f}")

print("\n" + "=" * 70)
print("Rotation Statistics:")
print(f"  Mean:   {np.mean(angles):7.2f}°")
print(f"  Median: {np.median(angles):7.2f}°")
print(f"  Std:    {np.std(angles):7.2f}°")
print(f"  Min:    {np.min(angles):7.2f}°")
print(f"  Max:    {np.max(angles):7.2f}°")
print(f"  Range:  {np.max(angles) - np.min(angles):7.2f}°")
print("=" * 70)

if np.std(angles) < 1.0:
    print("✓ EXCELLENT: Rotation is very consistent (std < 1°)")
elif np.std(angles) < 3.0:
    print("✓ GOOD: Rotation is fairly consistent (std < 3°)")
elif np.std(angles) < 5.0:
    print("⚠ FAIR: Some rotation variation (std < 5°)")
else:
    print("✗ POOR: Significant rotation variation (std >= 5°)")
