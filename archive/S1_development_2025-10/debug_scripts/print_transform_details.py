#!/usr/bin/env python3
"""
Print detailed transform information from Python to manually compare with C++
"""

import numpy as np
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner

# Load aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load first frame data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]

# Extract landmarks
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)

pose_tx = row['p_tx']
pose_ty = row['p_ty']

print("="*70)
print("Python Transform Matrices (Frame 1)")
print("="*70)

# Extract rigid points
source_rigid = aligner._extract_rigid_points(landmarks_68)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)

print("\n[1] PDM Reference Shape (first 5 rigid points):")
for i in range(5):
    print(f"  Point {i}: ({dest_rigid[i,0]:10.6f}, {dest_rigid[i,1]:10.6f})")

print("\n[2] Detected Landmarks (first 5 rigid points):")
for i in range(5):
    print(f"  Point {i}: ({source_rigid[i,0]:10.6f}, {source_rigid[i,1]:10.6f})")

# Compute transform
scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)

print("\n[3] scale_rot_matrix:")
print(f"  [[{scale_rot[0,0]:12.9f}, {scale_rot[0,1]:12.9f}],")
print(f"   [{scale_rot[1,0]:12.9f}, {scale_rot[1,1]:12.9f}]]")

print(f"\n[4] params_global tx,ty: {pose_tx:.6f}, {pose_ty:.6f}")

# Transform T
T = np.array([pose_tx, pose_ty], dtype=np.float32)
T_transformed = scale_rot @ T
print(f"[5] T after transform: {T_transformed[0]:.6f}, {T_transformed[1]:.6f}")

# Build warp matrix
warp_matrix = aligner._build_warp_matrix(scale_rot, pose_tx, pose_ty)

print(f"\n[6] warp_matrix:")
print(f"  [[{warp_matrix[0,0]:12.9f}, {warp_matrix[0,1]:12.9f}, {warp_matrix[0,2]:12.9f}],")
print(f"   [{warp_matrix[1,0]:12.9f}, {warp_matrix[1,1]:12.9f}, {warp_matrix[1,2]:12.9f}]]")

print("\n" + "="*70)
print("Copy this output to compare with OpenFace C++ debug output")
print("="*70)

# Print what the transform should do
print("\n[7] Transform Effect:")
center = landmarks_68.mean(axis=0)
center_transformed = scale_rot @ center + warp_matrix[:, 2]
print(f"Face center: ({center[0]:.2f}, {center[1]:.2f}) → ({center_transformed[0]:.2f}, {center_transformed[1]:.2f})")
print(f"Expected: around (56, 56)")

# Check determinant
det = np.linalg.det(scale_rot)
angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi
scale_val = np.sqrt(np.abs(det))
print(f"\n[8] Transform Properties:")
print(f"Determinant (scale²): {det:.9f}")
print(f"Scale factor: {scale_val:.6f}")
print(f"Rotation angle: {angle:.2f}°")
