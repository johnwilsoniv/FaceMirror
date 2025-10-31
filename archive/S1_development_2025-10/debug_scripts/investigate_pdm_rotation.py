#!/usr/bin/env python3
"""
Investigate: If PDM mean shape is rotated 45° CCW, why does C++ get upright (0°) faces?

Hypothesis: C++ applies a de-rotation to the reference shape before alignment
"""

import numpy as np
from pdm_parser import PDMParser

print("=" * 70)
print("PDM Mean Shape Rotation Investigation")
print("=" * 70)

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_2d = mean_shape_scaled[:136]
reference_shape = mean_shape_2d.reshape(68, 2)

print("\n[1/4] Measuring PDM mean shape orientation...")

# Key facial features for orientation
nose_tip = reference_shape[30]  # Landmark 30 = nose tip
chin = reference_shape[8]       # Landmark 8 = chin
left_eye = reference_shape[36]  # Landmark 36 = left eye outer corner
right_eye = reference_shape[45] # Landmark 45 = right eye outer corner

# Compute vectors
nose_to_chin = chin - nose_tip
eye_axis = right_eye - left_eye

# Compute angles
vertical_angle = np.arctan2(nose_to_chin[0], nose_to_chin[1]) * 180 / np.pi
horizontal_angle = np.arctan2(eye_axis[1], eye_axis[0]) * 180 / np.pi

print(f"  Nose-to-chin vector angle from vertical: {vertical_angle:.2f}°")
print(f"  Eye axis angle from horizontal: {horizontal_angle:.2f}°")
print(f"  Overall face rotation (CCW): ~{vertical_angle:.0f}°")

print("\n[2/4] What if C++ rotates the reference to be upright?...")

# Rotate reference shape to make it upright (de-rotate by face angle)
de_rotation_angle = -vertical_angle * np.pi / 180
rotation_matrix = np.array([
    [np.cos(de_rotation_angle), -np.sin(de_rotation_angle)],
    [np.sin(de_rotation_angle),  np.cos(de_rotation_angle)]
])

# Center the shape, rotate, then uncenter
center = reference_shape.mean(axis=0)
reference_centered = reference_shape - center
reference_rotated = (rotation_matrix @ reference_centered.T).T + center

# Check if it's now upright
nose_tip_r = reference_rotated[30]
chin_r = reference_rotated[8]
nose_to_chin_r = chin_r - nose_tip_r
vertical_angle_r = np.arctan2(nose_to_chin_r[0], nose_to_chin_r[1]) * 180 / np.pi

print(f"  After de-rotation: nose-to-chin angle = {vertical_angle_r:.2f}°")
print(f"  De-rotation needed: {-vertical_angle:.2f}°")

print("\n[3/4] Testing alignment with de-rotated reference...")

# Load a test frame
import pandas as pd
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df[df['frame'] == 1].iloc[0]

x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)

# Rigid indices with eyes
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

def align_shapes_with_scale(src, dst):
    """Kabsch with scale"""
    n = src.shape[0]

    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)

    s_src = np.sqrt(np.sum(src_centered ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)

    src_norm = src_centered / s_src
    dst_norm = dst_centered / s_dst

    U, S, Vt = np.linalg.svd(src_norm.T @ dst_norm)

    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    if d > 0:
        corr[1, 1] = 1
    else:
        corr[1, 1] = -1

    R = Vt.T @ corr @ U.T
    scale = s_dst / s_src

    return scale * R

# Test 1: Original reference (rotated PDM)
src = landmarks_68[RIGID_INDICES]
dst_original = reference_shape[RIGID_INDICES]
scale_rot_original = align_shapes_with_scale(src, dst_original)
angle_original = np.arctan2(scale_rot_original[1,0], scale_rot_original[0,0]) * 180 / np.pi

# Test 2: De-rotated reference (upright)
dst_derotated = reference_rotated[RIGID_INDICES]
scale_rot_derotated = align_shapes_with_scale(src, dst_derotated)
angle_derotated = np.arctan2(scale_rot_derotated[1,0], scale_rot_derotated[0,0]) * 180 / np.pi

print(f"  With original PDM reference:    {angle_original:.2f}°")
print(f"  With de-rotated reference:      {angle_derotated:.2f}°")
print(f"  Difference:                     {angle_derotated - angle_original:.2f}°")

print("\n[4/4] Checking if de-rotation explains C++ behavior...")

if abs(angle_derotated) < 5:
    print(f"  ✓✓✓ SUCCESS! De-rotated reference gives ~0° (upright faces)!")
    print(f"  ✓ This explains why C++ gets upright faces")
    print(f"  ✓ C++ must be applying this de-rotation to the PDM mean shape")
    print(f"\n  SOLUTION: Rotate reference shape by {-vertical_angle:.2f}° before alignment")
elif abs(angle_derotated - angle_original) > 10:
    print(f"  ~ De-rotation changes angle significantly")
    print(f"  ~ But doesn't get to 0° - may need different rotation")
else:
    print(f"  ✗ De-rotation doesn't significantly affect angle")
    print(f"  ✗ This is not the solution")

print("=" * 70)
