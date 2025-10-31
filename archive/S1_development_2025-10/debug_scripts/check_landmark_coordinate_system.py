#!/usr/bin/env python3
"""
Check if detected landmarks have an implicit rotation we're not accounting for

Maybe the CLNF detector outputs landmarks in a rotated coordinate system?
Or maybe there's a pose-based rotation we should apply to landmarks before alignment?
"""

import numpy as np
import pandas as pd

df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

print("=" * 70)
print("Checking Landmark Coordinate System and Pose Parameters")
print("=" * 70)

# Check frames 1, 493, 617, 863
test_frames = [1, 493, 617, 863]

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get pose parameters
    pose_tx = row['p_tx']
    pose_ty = row['p_ty']
    pose_rx = row['p_rx']  # Pitch (rotation around X axis)
    pose_ry = row['p_ry']  # Yaw (rotation around Y axis)
    pose_rz = row['p_rz']  # Roll (rotation around Z axis)
    pose_scale = row['p_scale']

    # Also get world space pose
    pose_Tx = row['pose_Tx']
    pose_Ty = row['pose_Ty']
    pose_Tz = row['pose_Tz']
    pose_Rx = row['pose_Rx']
    pose_Ry = row['pose_Ry']
    pose_Rz = row['pose_Rz']

    # Get some key landmarks to measure orientation
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)

    # Measure face orientation from landmarks
    left_eye = np.array([x[36], y[36]])
    right_eye = np.array([x[45], y[45]])
    nose_tip = np.array([x[30], y[30]])
    chin = np.array([x[8], y[8]])

    eye_vec = right_eye - left_eye
    eye_angle = np.arctan2(eye_vec[1], eye_vec[0]) * 180 / np.pi

    nose_vec = chin - nose_tip
    nose_angle = np.arctan2(nose_vec[0], nose_vec[1]) * 180 / np.pi

    print(f"\nFrame {frame_num}:")
    print(f"  PDM pose (p_rx, p_ry, p_rz): ({pose_rx:.4f}, {pose_ry:.4f}, {pose_rz:.4f})")
    print(f"  World pose (pose_Rx, pose_Ry, pose_Rz): ({pose_Rx:.4f}, {pose_Ry:.4f}, {pose_Rz:.4f})")
    print(f"  Pose scale: {pose_scale:.4f}")
    print(f"  Eye axis angle from horizontal: {eye_angle:.2f}°")
    print(f"  Nose-chin angle from vertical: {nose_angle:.2f}°")
    print(f"  Face appears rotated: ~{nose_angle:.0f}° from upright")

print("\n" + "=" * 70)
print("Key Question:")
print("=" * 70)
print("If pose_rz (roll) represents face rotation, should we:")
print("  A) Apply pose_rz rotation to landmarks before alignment?")
print("  B) Apply pose_rz rotation to reference shape?")
print("  C) Use pose_rz to correct the final rotation?")
print("  D) Ignore pose_rz (current implementation)?")
print()
print("C++ must be doing something with these pose parameters")
print("that we're not doing in Python!")
print("=" * 70)
