#!/usr/bin/env python3
"""
Diagnose why rotation varies per frame
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Test frames
test_frames = [1, 617, 740]

print("=" * 70)
print("Rotation Analysis Per Frame")
print("=" * 70)

from pdm_parser import PDMParser
from openface22_face_aligner import OpenFace22FaceAligner

pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
reference_shape = mean_shape_scaled[:136].reshape(68, 2)

RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Extract landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Extract rigid points
    src_rigid = landmarks_68[RIGID_INDICES]
    dst_rigid = reference_shape[RIGID_INDICES]

    # Compute transform
    n = src_rigid.shape[0]

    # Mean normalize
    mean_src = src_rigid.mean(axis=0)
    mean_dst = dst_rigid.mean(axis=0)
    src_norm = src_rigid - mean_src
    dst_norm = dst_rigid - mean_dst

    # RMS scale
    s_src = np.sqrt(np.sum(src_norm ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_norm ** 2) / n)

    src_norm_scaled = src_norm / s_src
    dst_norm_scaled = dst_norm / s_dst

    # Kabsch
    U, S, Vt = np.linalg.svd(src_norm_scaled.T @ dst_norm_scaled)
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    corr[1, 1] = 1 if d > 0 else -1
    R = Vt.T @ corr @ U.T

    scale = s_dst / s_src
    scale_rot = scale * R

    # Angle from matrix (no transpose)
    angle_no_transpose = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi

    # Angle from matrix (with transpose)
    scale_rot_T = scale_rot.T
    angle_with_transpose = np.arctan2(scale_rot_T[1,0], scale_rot_T[0,0]) * 180 / np.pi

    # Check what OpenFace CSV says about head pose
    pose_Rx = row['pose_Rx']  # Roll
    pose_Ry = row['pose_Ry']  # Pitch
    pose_Rz = row['pose_Rz']  # Yaw

    print(f"\nFrame {frame_num}:")
    print(f"  Head pose from OpenFace CSV:")
    print(f"    Roll (Rx):  {pose_Rx:.2f}° (rotation around X-axis, head tilt)")
    print(f"    Pitch (Ry): {pose_Ry:.2f}° (rotation around Y-axis, up/down)")
    print(f"    Yaw (Rz):   {pose_Rz:.2f}° (rotation around Z-axis, left/right)")
    print(f"  Computed alignment rotation:")
    print(f"    Without transpose: {angle_no_transpose:.2f}°")
    print(f"    With transpose:    {angle_with_transpose:.2f}°")
    print(f"  Scale factor: {scale:.4f}")

print("\n" + "=" * 70)
print("Analysis:")
print("  If C++ faces are always upright, the alignment should REMOVE the")
print("  head rotation (roll), not preserve it. The rotation angle should")
print("  be approximately equal to -pose_Rx to counteract the head tilt.")
print("=" * 70)
