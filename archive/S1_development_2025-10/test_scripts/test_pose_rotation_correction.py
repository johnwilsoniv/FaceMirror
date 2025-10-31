#!/usr/bin/env python3
"""
Test if using pose_Rz to pre-rotate landmarks produces upright faces

Hypothesis: C++ pre-rotates landmarks by -pose_Rz before Kabsch alignment
This would compensate for the inherent face rotation detected by CLNF
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from pdm_parser import PDMParser

# Load data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cpp_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_2d = mean_shape_scaled[:136]
reference_shape = mean_shape_2d.reshape(68, 2)

# Rigid indices
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

def rotate_points(points, angle_rad, center=None):
    """Rotate points around center by angle in radians"""
    if center is None:
        center = points.mean(axis=0)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    centered = points - center
    rotated = (R @ centered.T).T
    return rotated + center

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

print("=" * 70)
print("Testing Pose Rotation Correction Hypothesis")
print("=" * 70)

# Test key frames
test_frames = [1, 493, 617, 863]

print("\n[TEST A] Pre-rotate landmarks by -pose_Rz before alignment:")
print("-" * 70)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Get pose roll
    pose_Rz = row['pose_Rz']  # in radians

    # PRE-ROTATE landmarks by -pose_Rz (counter-clockwise to upright them)
    landmarks_corrected = rotate_points(landmarks_68, -pose_Rz)

    # Now do alignment with corrected landmarks
    src = landmarks_corrected[RIGID_INDICES]
    dst = reference_shape[RIGID_INDICES]

    scale_rot = align_shapes_with_scale(src, dst)
    output_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi

    print(f"Frame {frame_num:4d}: pose_Rz={pose_Rz*180/np.pi:+6.2f}° → output angle = {output_angle:+7.2f}°")

print("\n[TEST B] Rotate reference shape by +pose_Rz before alignment:")
print("-" * 70)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get landmarks
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    # Get pose roll
    pose_Rz = row['pose_Rz']  # in radians

    # ROTATE reference shape by +pose_Rz
    reference_corrected = rotate_points(reference_shape, pose_Rz)

    # Now do alignment
    src = landmarks_68[RIGID_INDICES]
    dst = reference_corrected[RIGID_INDICES]

    scale_rot = align_shapes_with_scale(src, dst)
    output_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi

    print(f"Frame {frame_num:4d}: pose_Rz={pose_Rz*180/np.pi:+6.2f}° → output angle = {output_angle:+7.2f}°")

print("\n[BASELINE] Current Python (no pose correction):")
print("-" * 70)

baseline_angles = []
for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    src = landmarks_68[RIGID_INDICES]
    dst = reference_shape[RIGID_INDICES]

    scale_rot = align_shapes_with_scale(src, dst)
    output_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi
    baseline_angles.append(output_angle)

    pose_Rz = row['pose_Rz']
    print(f"Frame {frame_num:4d}: pose_Rz={pose_Rz*180/np.pi:+6.2f}° → output angle = {output_angle:+7.2f}°")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("Baseline Python:")
print(f"  Mean: {np.mean(baseline_angles):+7.2f}°")
print(f"  Std:  {np.std(baseline_angles):7.2f}°")
print(f"  Expression sensitivity (493 vs 617): {abs(baseline_angles[1] - baseline_angles[2]):.2f}°")
print()
print("C++ Target:")
print("  Mean: ~0°")
print("  Std:  <2°")
print("  Expression sensitivity: ~0°")
print("=" * 70)
