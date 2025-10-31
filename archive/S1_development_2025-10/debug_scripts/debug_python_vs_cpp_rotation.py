#!/usr/bin/env python3
"""
Debug: Why does Python Kabsch produce different rotation than C++?

We know:
1. CSV landmarks are PDM-reconstructed (RMSE < 0.1 px confirmed)
2. C++ AlignFace uses these landmarks → produces ~0° rotation
3. Python AlignFace uses these landmarks → produces -8° to +2° rotation

This script compares our Python Kabsch implementation directly
"""

import numpy as np
import pandas as pd
from pdm_parser import PDMParser
from openface22_face_aligner import OpenFace22FaceAligner

# Load PDM and aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

print("=" * 80)
print("Debugging Python vs C++ Rotation Difference")
print("=" * 80)

test_frames = [1, 493, 617, 863]

print(f"\n{'Frame':>6} {'Python Rot':>12} {'C++ (Expected)':>15} {'Difference':>12} {'Expression':>15}")
print("-" * 80)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get CSV landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    csv_x = row[x_cols].values.astype(np.float32)
    csv_y = row[y_cols].values.astype(np.float32)
    csv_landmarks = np.stack([csv_x, csv_y], axis=1)  # (68, 2)

    # Extract rigid points
    source_rigid = aligner._extract_rigid_points(csv_landmarks)
    dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)

    # Compute Kabsch alignment (what our Python does)
    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
    python_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi

    # C++ produces ~0° (from visual inspection of aligned faces)
    cpp_expected = 0.0

    difference = python_angle - cpp_expected

    expression = "eyes closed" if frame_num == 617 else ""

    print(f"{frame_num:>6} {python_angle:>12.2f} {cpp_expected:>15.2f} {difference:>12.2f} {expression:>15}")

print("\n" + "=" * 80)
print("Key Observation:")
print("=" * 80)
print("Our Python Kabsch on CSV landmarks produces tilted faces.")
print("But C++ Kabsch on THE SAME landmarks produces upright faces.")
print()
print("Possible explanations:")
print("  1. C++ applies a correction after Kabsch")
print("  2. C++ uses a different reference shape than pdm.mean_shape")
print("  3. C++ pre-processes landmarks before Kabsch")
print("  4. Our Kabsch implementation has a subtle bug")
print()
print("Let's test explanation #4: Verify our Kabsch matches C++ numerically")
print("=" * 80)

# Test a simple known case
print("\nTest Case: Identity transform (landmarks = reference)")
source_test = aligner.reference_shape[aligner.RIGID_INDICES]
dest_test = aligner.reference_shape[aligner.RIGID_INDICES]
scale_rot_test = aligner._align_shapes_with_scale(source_test, dest_test)
angle_test = np.arctan2(scale_rot_test[1,0], scale_rot_test[0,0]) * 180 / np.pi

print(f"  Rotation angle: {angle_test:.6f}°")
print(f"  Scale-rot matrix:")
print(scale_rot_test)

if abs(angle_test) < 1e-6 and np.allclose(scale_rot_test, np.eye(2), atol=1e-6):
    print("  ✓ Kabsch correctly produces identity for identical inputs")
else:
    print("  ✗ PROBLEM: Kabsch doesn't produce identity!")

print("\n" + "=" * 80)
print("Next Debug Step:")
print("=" * 80)
print("Compare our scale_rot matrix values to C++ AlignShapesWithScale output")
print("Need to instrument C++ code to print the matrix...")
print("=" * 80)
