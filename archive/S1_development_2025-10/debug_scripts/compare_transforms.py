#!/usr/bin/env python3
"""
Compare Python vs expected C++ transform values
"""

import numpy as np
import cv2
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

print("=" * 70)
print("Transform Analysis")
print("=" * 70)

# Extract rigid points
source_rigid = aligner._extract_rigid_points(landmarks_68)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)

print("\n[1] Rigid Points Statistics")
print(f"Source (image landmarks):")
print(f"  Mean: ({source_rigid[:,0].mean():.2f}, {source_rigid[:,1].mean():.2f})")
print(f"  Std:  ({source_rigid[:,0].std():.2f}, {source_rigid[:,1].std():.2f})")
print(f"Dest (reference shape):")
print(f"  Mean: ({dest_rigid[:,0].mean():.4f}, {dest_rigid[:,1].mean():.4f})")
print(f"  Std:  ({dest_rigid[:,0].std():.4f}, {dest_rigid[:,1].std():.4f})")

# Compute transform
scale_rot_matrix = aligner._align_shapes_with_scale(source_rigid, dest_rigid)

print("\n[2] Scale-Rotation Matrix")
print(scale_rot_matrix)
scale_value = np.sqrt(np.abs(np.linalg.det(scale_rot_matrix)))
print(f"Estimated scale: {scale_value:.6f}")

# Build warp matrix
warp_matrix = aligner._build_warp_matrix(scale_rot_matrix, pose_tx, pose_ty)

print("\n[3] Warp Matrix (for cv2.warpAffine)")
print(warp_matrix)
print(f"Translation: ({warp_matrix[0,2]:.2f}, {warp_matrix[1,2]:.2f})")

# Test: where does the center of the face landmarks end up?
face_center = landmarks_68.mean(axis=0)
face_center_transformed = scale_rot_matrix @ face_center + warp_matrix[:, 2]
print(f"\n[4] Face Center Transformation")
print(f"Original face center: ({face_center[0]:.2f}, {face_center[1]:.2f})")
print(f"Transformed to: ({face_center_transformed[0]:.2f}, {face_center_transformed[1]:.2f})")
print(f"Expected (for 112x112): around (56, 56)")

# Test a few landmark points
print(f"\n[5] Sample Landmark Transformations")
for i in [0, 27, 33, 48]:  # Jaw, nose tip, nose bridge, lip
    pt = landmarks_68[i]
    pt_transformed = scale_rot_matrix @ pt + warp_matrix[:, 2]
    print(f"  Landmark {i:2d}: ({pt[0]:6.1f}, {pt[1]:6.1f}) → ({pt_transformed[0]:6.2f}, {pt_transformed[1]:6.2f})")

print("\n" + "=" * 70)
