#!/usr/bin/env python3
"""
Test if our Python Kabsch produces EXACTLY the same numerical result
as we expect from C++, using the same test values.

This bypasses the build issue by directly testing the algorithm.
"""

import numpy as np
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner

# Load aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

print("=" * 80)
print("Testing Numerical Precision of Kabsch Implementation")
print("=" * 80)

# Test frame 493 (stable, eyes open)
frame_num = 493
row = df[df['frame'] == frame_num].iloc[0]

# Get CSV landmarks
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
csv_x = row[x_cols].values.astype(np.float32)
csv_y = row[y_cols].values.astype(np.float32)
csv_landmarks = np.stack([csv_x, csv_y], axis=1)  # (68, 2)

# Extract rigid points
source_rigid = aligner._extract_rigid_points(csv_landmarks)  # (24, 2)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)  # (24, 2)

print(f"\n[Step 1] Input shapes:")
print(f"  Source rigid: {source_rigid.shape}")
print(f"  Dest rigid: {dest_rigid.shape}")

# Manually step through the algorithm
n = source_rigid.shape[0]

# 1. Mean normalize
mean_src_x = np.mean(source_rigid[:, 0])
mean_src_y = np.mean(source_rigid[:, 1])
mean_dst_x = np.mean(dest_rigid[:, 0])
mean_dst_y = np.mean(dest_rigid[:, 1])

print(f"\n[Step 2] Means:")
print(f"  Source mean: ({mean_src_x:.6f}, {mean_src_y:.6f})")
print(f"  Dest mean: ({mean_dst_x:.6f}, {mean_dst_y:.6f})")

src_centered = source_rigid.copy()
src_centered[:, 0] -= mean_src_x
src_centered[:, 1] -= mean_src_y

dst_centered = dest_rigid.copy()
dst_centered[:, 0] -= mean_dst_x
dst_centered[:, 1] -= mean_dst_y

# 2. Compute scale
src_sq = src_centered ** 2
dst_sq = dst_centered ** 2

s_src = np.sqrt(np.sum(src_sq) / n)
s_dst = np.sqrt(np.sum(dst_sq) / n)

print(f"\n[Step 3] Scales:")
print(f"  s_src: {s_src:.6f}")
print(f"  s_dst: {s_dst:.6f}")
print(f"  scale ratio: {s_dst / s_src:.6f}")

# 3. Normalize
src_norm = src_centered / s_src
dst_norm = dst_centered / s_dst

# 4. SVD for rotation
H = src_norm.T @ dst_norm  # Covariance matrix
print(f"\n[Step 4] Covariance matrix H:")
print(f"  Shape: {H.shape}")
print(f"  H =")
print(f"    [{H[0,0]:10.6f}, {H[0,1]:10.6f}]")
print(f"    [{H[1,0]:10.6f}, {H[1,1]:10.6f}]")

U, S, Vt = np.linalg.svd(H)

print(f"\n[Step 5] SVD decomposition:")
print(f"  U =")
print(f"    [{U[0,0]:10.6f}, {U[0,1]:10.6f}]")
print(f"    [{U[1,0]:10.6f}, {U[1,1]:10.6f}]")
print(f"  S = [{S[0]:.6f}, {S[1]:.6f}]")
print(f"  Vt =")
print(f"    [{Vt[0,0]:10.6f}, {Vt[0,1]:10.6f}]")
print(f"    [{Vt[1,0]:10.6f}, {Vt[1,1]:10.6f}]")

# 5. Check for reflection
d = np.linalg.det(Vt.T @ U.T)
print(f"\n[Step 6] Reflection check:")
print(f"  det(Vt.T @ U.T) = {d:.6f}")

corr = np.eye(2)
if d > 0:
    corr[1, 1] = 1
    print(f"  → No reflection (d > 0), corr[1,1] = 1")
else:
    corr[1, 1] = -1
    print(f"  → Reflection detected (d <= 0), corr[1,1] = -1")

# 6. Compute rotation
R = Vt.T @ corr @ U.T

print(f"\n[Step 7] Rotation matrix R:")
print(f"  R =")
print(f"    [{R[0,0]:10.6f}, {R[0,1]:10.6f}]")
print(f"    [{R[1,0]:10.6f}, {R[1,1]:10.6f}]")

angle_rad = np.arctan2(R[1,0], R[0,0])
angle_deg = angle_rad * 180 / np.pi
print(f"  Rotation angle: {angle_deg:.6f}°")

# 7. Final scale-rot matrix
scale = s_dst / s_src
scale_rot = scale * R

print(f"\n[Step 8] Final scale-rot matrix:")
print(f"  scale_rot = {scale:.6f} * R =")
print(f"    [{scale_rot[0,0]:10.6f}, {scale_rot[0,1]:10.6f}]")
print(f"    [{scale_rot[1,0]:10.6f}, {scale_rot[1,1]:10.6f}]")
print(f"  Final rotation angle: {angle_deg:.6f}°")

# Compare to our function
scale_rot_function = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
angle_function = np.arctan2(scale_rot_function[1,0], scale_rot_function[0,0]) * 180 / np.pi

print(f"\n[Step 9] Verification:")
print(f"  Function result:")
print(f"    [{scale_rot_function[0,0]:10.6f}, {scale_rot_function[0,1]:10.6f}]")
print(f"    [{scale_rot_function[1,0]:10.6f}, {scale_rot_function[1,1]:10.6f}]")
print(f"  Function angle: {angle_function:.6f}°")

if np.allclose(scale_rot, scale_rot_function, atol=1e-6):
    print(f"  ✓ Manual calculation matches function")
else:
    print(f"  ✗ MISMATCH - function differs from manual calculation!")

print("\n" + "=" * 80)
print("Key Observations:")
print("=" * 80)
print(f"Python produces: {angle_function:.2f}° rotation")
print(f"C++ should produce: ~0° rotation (upright faces)")
print(f"\nDifference: {abs(angle_function):.2f}°")
print("\nIf our algorithm is correct but produces wrong rotation,")
print("then C++ must be doing something DIFFERENT than standard Kabsch.")
print("Possibilities:")
print("  1. C++ uses different input (not CSV landmarks?)")
print("  2. C++ applies a correction after Kabsch")
print("  3. C++ uses different reference shape at runtime")
print("=" * 80)
