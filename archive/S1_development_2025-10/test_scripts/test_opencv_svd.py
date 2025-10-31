#!/usr/bin/env python3
"""
Test if using OpenCV's SVD instead of numpy's SVD matches C++ rotation

This is THE critical test - if this works, we've found the root cause!
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Load data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Load PDM
from pdm_parser import PDMParser
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_2d = mean_shape_scaled[:136]
reference_shape = mean_shape_2d.reshape(68, 2)

# Rigid indices
RIGID_INDICES_WITH_EYES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]
RIGID_INDICES_NO_EYES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35]

def align_shapes_with_scale_NUMPY(src, dst):
    """Original implementation using numpy SVD"""
    n = src.shape[0]

    # Mean normalize
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)

    # RMS scale
    s_src = np.sqrt(np.sum(src_centered ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)

    src_norm = src_centered / s_src
    dst_norm = dst_centered / s_dst

    # Kabsch with NUMPY SVD
    U, S, Vt = np.linalg.svd(src_norm.T @ dst_norm)

    # Check for reflection
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    if d > 0:
        corr[1, 1] = 1
    else:
        corr[1, 1] = -1

    R = Vt.T @ corr @ U.T
    scale = s_dst / s_src

    return scale * R

def align_shapes_with_scale_OPENCV(src, dst):
    """NEW implementation using OpenCV SVD"""
    n = src.shape[0]

    # Mean normalize
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)

    # RMS scale
    s_src = np.sqrt(np.sum(src_centered ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)

    src_norm = src_centered / s_src
    dst_norm = dst_centered / s_dst

    # Kabsch with OPENCV SVD
    H = (src_norm.T @ dst_norm).astype(np.float32)
    w, u, vt = cv2.SVDecomp(H)

    # Check for reflection
    d = np.linalg.det(vt.T @ u.T)
    corr = np.eye(2, dtype=np.float32)
    if d > 0:
        corr[1, 1] = 1
    else:
        corr[1, 1] = -1

    R = vt.T @ corr @ u.T
    scale = s_dst / s_src

    return scale * R

print("=" * 70)
print("SVD Hypothesis Test: numpy vs OpenCV")
print("=" * 70)

# Test on key frames
test_frames = [1, 493, 617, 863]

print("\n[1/3] Comparing SVD outputs on frame 1...")
row = df[df['frame'] == 1].iloc[0]

# Extract landmarks
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)

# Test with eyes (24 points)
src = landmarks_68[RIGID_INDICES_WITH_EYES]
dst = reference_shape[RIGID_INDICES_WITH_EYES]

# Mean normalize
n = src.shape[0]
src_centered = src - src.mean(axis=0)
dst_centered = dst - dst.mean(axis=0)
s_src = np.sqrt(np.sum(src_centered ** 2) / n)
s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)
src_norm = src_centered / s_src
dst_norm = dst_centered / s_dst

# Compute H matrix
H = src_norm.T @ dst_norm

# Compare SVD outputs
U_np, S_np, Vt_np = np.linalg.svd(H)
H_cv = H.astype(np.float32)
w_cv, u_cv, vt_cv = cv2.SVDecomp(H_cv)

print("\nSingular values:")
print(f"  numpy:  {S_np}")
print(f"  OpenCV: {w_cv.flatten()}")
print(f"  Difference: {np.abs(S_np - w_cv.flatten()).max():.10f}")

print("\nU matrix column signs:")
for i in range(2):
    sign_np = "+" if U_np[0, i] >= 0 else "-"
    sign_cv = "+" if u_cv[0, i] >= 0 else "-"
    match = "✓" if sign_np == sign_cv else "✗"
    print(f"  Column {i}: numpy={sign_np}, OpenCV={sign_cv} {match}")

print("\nV^T matrix row signs:")
for i in range(2):
    sign_np = "+" if Vt_np[i, 0] >= 0 else "-"
    sign_cv = "+" if vt_cv[i, 0] >= 0 else "-"
    match = "✓" if sign_np == sign_cv else "✗"
    print(f"  Row {i}: numpy={sign_np}, OpenCV={sign_cv} {match}")

print("\n[2/3] Testing rotation angles (WITH eyes - 24 points)...")
print("-" * 70)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Extract landmarks
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Extract rigid points
    src = landmarks_68[RIGID_INDICES_WITH_EYES]
    dst = reference_shape[RIGID_INDICES_WITH_EYES]

    # Compute rotation with numpy SVD
    scale_rot_np = align_shapes_with_scale_NUMPY(src, dst)
    angle_np = np.arctan2(scale_rot_np[1,0], scale_rot_np[0,0]) * 180 / np.pi

    # Compute rotation with OpenCV SVD
    scale_rot_cv = align_shapes_with_scale_OPENCV(src, dst)
    angle_cv = np.arctan2(scale_rot_cv[1,0], scale_rot_cv[0,0]) * 180 / np.pi

    diff = angle_cv - angle_np

    print(f"Frame {frame_num:4d}: numpy={angle_np:7.2f}°, OpenCV={angle_cv:7.2f}°, diff={diff:6.2f}°")

print("\n[3/3] Testing rotation angles (WITHOUT eyes - 16 points)...")
print("-" * 70)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Extract landmarks
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Extract rigid points (no eyes)
    src = landmarks_68[RIGID_INDICES_NO_EYES]
    dst = reference_shape[RIGID_INDICES_NO_EYES]

    # Compute rotation with numpy SVD
    scale_rot_np = align_shapes_with_scale_NUMPY(src, dst)
    angle_np = np.arctan2(scale_rot_np[1,0], scale_rot_np[0,0]) * 180 / np.pi

    # Compute rotation with OpenCV SVD
    scale_rot_cv = align_shapes_with_scale_OPENCV(src, dst)
    angle_cv = np.arctan2(scale_rot_cv[1,0], scale_rot_cv[0,0]) * 180 / np.pi

    diff = angle_cv - angle_np

    print(f"Frame {frame_num:4d}: numpy={angle_np:7.2f}°, OpenCV={angle_cv:7.2f}°, diff={diff:6.2f}°")

print("\n" + "=" * 70)
print("Conclusion:")
print("=" * 70)

# Check if OpenCV gives significantly different angles
angles_np_with_eyes = []
angles_cv_with_eyes = []
for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    src = landmarks_68[RIGID_INDICES_WITH_EYES]
    dst = reference_shape[RIGID_INDICES_WITH_EYES]

    scale_rot_np = align_shapes_with_scale_NUMPY(src, dst)
    scale_rot_cv = align_shapes_with_scale_OPENCV(src, dst)

    angles_np_with_eyes.append(np.arctan2(scale_rot_np[1,0], scale_rot_np[0,0]) * 180 / np.pi)
    angles_cv_with_eyes.append(np.arctan2(scale_rot_cv[1,0], scale_rot_cv[0,0]) * 180 / np.pi)

mean_diff = np.mean(np.array(angles_cv_with_eyes) - np.array(angles_np_with_eyes))

if abs(mean_diff) < 1.0:
    print("  ✗ SVD implementations give nearly identical results")
    print(f"  → Mean difference: {mean_diff:.2f}° (too small to explain -5° offset)")
    print("  → SVD sign ambiguity is NOT the root cause")
    print("  → Need to investigate other hypotheses")
elif abs(mean_diff) > 2.0:
    print("  ✓ SVD implementations give different results!")
    print(f"  → Mean difference: {mean_diff:.2f}°")
    if abs(np.mean(angles_cv_with_eyes)) < abs(np.mean(angles_np_with_eyes)):
        print("  → OpenCV SVD gives angles closer to 0° (more upright)")
        print("  → HYPOTHESIS CONFIRMED: Use cv2.SVDecomp to match C++!")
    else:
        print("  → But OpenCV doesn't improve rotation")
        print("  → Need different approach")
else:
    print(f"  ~ Moderate difference: {mean_diff:.2f}°")
    print("  → May explain part of the issue")

print("=" * 70)
