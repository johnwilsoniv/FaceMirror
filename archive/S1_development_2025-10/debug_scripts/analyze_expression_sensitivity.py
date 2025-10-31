#!/usr/bin/env python3
"""
Deep dive into why expression (eye closure) affects Python rotation but not C++

This is a CRITICAL clue - if C++ truly uses the same 24 rigid points including
8 eye landmarks, eye closure SHOULD affect rotation. But it doesn't.

This suggests:
1. C++ might not actually use those eye landmarks for rotation
2. C++ might apply temporal smoothing
3. CSV landmarks might differ from what C++ actually uses
4. There might be hidden weighting
"""

import numpy as np
import pandas as pd
import cv2
from openface22_face_aligner import OpenFace22FaceAligner

# Load data
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Compare frames around eye closure
frames_to_analyze = {
    493: "eyes open",
    617: "eyes closed",
    863: "eyes open",
}

print("=" * 70)
print("Expression Sensitivity Analysis: The Smoking Gun")
print("=" * 70)

# Extract and compare eye landmarks
EYE_LANDMARKS = [36, 39, 40, 41, 42, 45, 46, 47]  # Eye landmarks in rigid points
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]
NON_EYE_RIGID = [i for i in RIGID_INDICES if i not in EYE_LANDMARKS]

print(f"\nRigid points breakdown:")
print(f"  Total rigid points: {len(RIGID_INDICES)}")
print(f"  Eye landmarks: {len(EYE_LANDMARKS)} (33% of rigid points)")
print(f"  Non-eye rigid: {len(NON_EYE_RIGID)}")

results = []

for frame_num, description in frames_to_analyze.items():
    row = df[df['frame'] == frame_num].iloc[0]

    # Extract landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Test 1: Current method (all 24 rigid points)
    source_rigid_all = aligner._extract_rigid_points(landmarks_68)
    dest_rigid_all = aligner._extract_rigid_points(aligner.reference_shape)
    scale_rot_all = aligner._align_shapes_with_scale(source_rigid_all, dest_rigid_all)
    angle_all = np.arctan2(scale_rot_all[1,0], scale_rot_all[0,0]) * 180 / np.pi

    # Test 2: WITHOUT eye landmarks (only 16 non-eye rigid points)
    source_rigid_no_eyes = landmarks_68[NON_EYE_RIGID]
    dest_rigid_no_eyes = aligner.reference_shape[NON_EYE_RIGID]
    scale_rot_no_eyes = aligner._align_shapes_with_scale(source_rigid_no_eyes, dest_rigid_no_eyes)
    angle_no_eyes = np.arctan2(scale_rot_no_eyes[1,0], scale_rot_no_eyes[0,0]) * 180 / np.pi

    # Extract eye landmark positions
    eye_landmarks = landmarks_68[EYE_LANDMARKS]
    eye_height = eye_landmarks[:, 1].max() - eye_landmarks[:, 1].min()

    results.append({
        'frame': frame_num,
        'description': description,
        'angle_all': angle_all,
        'angle_no_eyes': angle_no_eyes,
        'eye_height': eye_height
    })

    print(f"\nFrame {frame_num} ({description}):")
    print(f"  With eye landmarks (24 points):    {angle_all:7.2f}°")
    print(f"  WITHOUT eye landmarks (16 points): {angle_no_eyes:7.2f}°")
    print(f"  Difference:                        {angle_all - angle_no_eyes:7.2f}°")
    print(f"  Eye vertical span:                 {eye_height:.1f} pixels")

print("\n" + "=" * 70)
print("Analysis:")
print("=" * 70)

# Compare rotation stability
angle_all_std = np.std([r['angle_all'] for r in results])
angle_no_eyes_std = np.std([r['angle_no_eyes'] for r in results])

print(f"\nRotation stability:")
print(f"  With eye landmarks:    std = {angle_all_std:.2f}°")
print(f"  WITHOUT eye landmarks: std = {angle_no_eyes_std:.2f}°")

if angle_no_eyes_std < angle_all_std:
    improvement = ((angle_all_std - angle_no_eyes_std) / angle_all_std) * 100
    print(f"  ✓ REMOVING eye landmarks improves stability by {improvement:.1f}%!")
    print(f"\n  HYPOTHESIS: C++ might NOT actually use eye landmarks in rotation computation,")
    print(f"              despite extract_rigid_points() including them!")
else:
    print(f"  ✗ Removing eye landmarks doesn't improve stability")

# Check eye height variation
eye_heights = [r['eye_height'] for r in results]
eyes_open_height = (eye_heights[0] + eye_heights[2]) / 2
eyes_closed_height = eye_heights[1]
height_change = ((eyes_closed_height - eyes_open_height) / eyes_open_height) * 100

print(f"\nEye height variation:")
print(f"  Eyes open average:  {eyes_open_height:.1f} pixels")
print(f"  Eyes closed:        {eyes_closed_height:.1f} pixels")
print(f"  Change:             {height_change:+.1f}%")

print("\n" + "=" * 70)
print("CONCLUSION:")
if angle_no_eyes_std < angle_all_std:
    print("  Excluding eye landmarks from rotation computation SIGNIFICANTLY")
    print("  improves stability across expressions. This suggests C++ may:")
    print("  1. Not actually use eye landmarks despite extract_rigid_points()")
    print("  2. Weight eye landmarks much lower than other rigid points")
    print("  3. Have additional filtering that effectively excludes them")
else:
    print("  Eye landmarks don't significantly affect stability.")
    print("  The issue must be elsewhere (smoothing, CSV vs actual landmarks, etc.)")
print("=" * 70)
