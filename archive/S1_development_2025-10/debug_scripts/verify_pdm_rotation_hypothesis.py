#!/usr/bin/env python3
"""
Verify that params_global rotation 'unrotates' the PDM mean shape orientation

Theory:
- PDM mean_shape is rotated ~120° CCW in its canonical frame
- CalcShape2D applies params_global rotation to mean_shape → detected_landmarks
- detected_landmarks should be more upright than mean_shape
- AlignFace then aligns detected_landmarks to mean_shape, which should give near-zero rotation
"""

import numpy as np
import pandas as pd
from pdm_parser import PDMParser

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_3d = mean_shape_scaled.reshape(68, 3)  # Keep 3D

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

def euler_to_rotation_matrix(rx, ry, rz):
    """
    Convert Euler angles to 3D rotation matrix
    Using XYZ convention: R = Rx * Ry * Rz
    """
    s1, s2, s3 = np.sin(rx), np.sin(ry), np.sin(rz)
    c1, c2, c3 = np.cos(rx), np.cos(ry), np.cos(rz)

    R = np.array([
        [c2*c3,           -c2*s3,          s2],
        [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
        [s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3,  c1*c2]
    ])
    return R

def measure_2d_orientation(points_2d):
    """Measure face orientation from 2D landmarks"""
    # Eye axis angle
    left_eye = points_2d[36]
    right_eye = points_2d[45]
    eye_vec = right_eye - left_eye
    eye_angle = np.arctan2(eye_vec[1], eye_vec[0]) * 180 / np.pi

    # Nose-chin angle
    nose_tip = points_2d[30]
    chin = points_2d[8]
    nose_vec = chin - nose_tip
    nose_angle = np.arctan2(nose_vec[0], nose_vec[1]) * 180 / np.pi

    return eye_angle, nose_angle

print("=" * 70)
print("Verifying PDM Rotation Hypothesis")
print("=" * 70)

# Measure PDM mean_shape orientation (2D projection, no rotation)
mean_shape_2d = mean_shape_3d[:, :2]  # Just x, y
eye_angle_pdm, nose_angle_pdm = measure_2d_orientation(mean_shape_2d)

print(f"\n[1] PDM mean_shape (canonical, no rotation):")
print(f"    Eye axis:    {eye_angle_pdm:+7.2f}° from horizontal")
print(f"    Nose-chin:   {nose_angle_pdm:+7.2f}° from vertical")
print(f"    Face appears rotated ~{nose_angle_pdm:.0f}° from upright")

# Test key frames
test_frames = [1, 493, 617, 863]

print(f"\n[2] After applying params_global rotation to mean_shape:")
print("-" * 70)

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get params_global
    scale = row['p_scale']
    rx = row['p_rx']  # Note: These are PDM params, might differ from pose_Rx
    ry = row['p_ry']
    rz = row['p_rz']
    tx = row['p_tx']
    ty = row['p_ty']

    # Build rotation matrix
    R = euler_to_rotation_matrix(rx, ry, rz)

    # Apply rotation to 3D mean shape (no deformation, params_local = 0)
    rotated_3d = scale * (R @ mean_shape_3d.T).T

    # Project to 2D (weak perspective: just take x, y)
    rotated_2d = rotated_3d[:, :2]
    rotated_2d[:, 0] += tx
    rotated_2d[:, 1] += ty

    # Measure orientation
    eye_angle, nose_angle = measure_2d_orientation(rotated_2d)

    print(f"Frame {frame_num:4d}: p_rz={rz*180/np.pi:+6.2f}° → "
          f"eye={eye_angle:+6.2f}°, nose={nose_angle:+6.2f}° "
          f"(face rotated ~{nose_angle:.0f}° from upright)")

print("\n[3] Key Insight:")
print("-" * 70)
print("If params_global rotation successfully 'unrotates' the PDM:")
print("  - PDM mean shape: ~120° rotated")
print("  - After params_global rotation: ~0-10° rotated")
print("  - Then AlignFace aligns rotated shape back to PDM mean shape")
print("  - Final output should be upright (~0° rotation)")
print()
print("If rotated shapes are still significantly tilted (>10°), then")
print("params_global does NOT fully compensate for PDM canonical orientation")
print("=" * 70)
