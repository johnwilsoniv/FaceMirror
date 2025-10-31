#!/usr/bin/env python3
"""
Test alignment with rigid points vs all points
"""

import numpy as np
import cv2
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner
from scipy.stats import pearsonr

# Load data
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]
frame_num = int(row['frame'])

# Extract landmarks and pose
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)
pose_tx = row['p_tx']
pose_ty = row['p_ty']

# Load frame
video_file = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
ret, frame = cap.read()
cap.release()

# Load C++ aligned face
cpp_aligned_file = f"pyfhog_validation_output/IMG_0942_left_mirrored_aligned/frame_det_00_{frame_num:06d}.bmp"
cpp_aligned = cv2.imread(cpp_aligned_file)

print("="*70)
print("Testing: Rigid Points vs All Points")
print("="*70)

# Test 1: Current (24 rigid points)
source_rigid = aligner._extract_rigid_points(landmarks_68)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
scale_rot_rigid = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
warp_rigid = aligner._build_warp_matrix(scale_rot_rigid, pose_tx, pose_ty)
aligned_rigid = cv2.warpAffine(frame, warp_rigid, (112, 112), flags=cv2.INTER_LINEAR)

mse_rigid = np.mean((aligned_rigid.astype(float) - cpp_aligned.astype(float)) ** 2)
corr_rigid, _ = pearsonr(aligned_rigid.flatten(), cpp_aligned.flatten())

print(f"\n[1] Using 24 RIGID points:")
print(f"  MSE:  {mse_rigid:.2f}")
print(f"  Corr: {corr_rigid:.6f}")

# Test 2: All 68 points
scale_rot_all = aligner._align_shapes_with_scale(landmarks_68, aligner.reference_shape)
warp_all = aligner._build_warp_matrix(scale_rot_all, pose_tx, pose_ty)
aligned_all = cv2.warpAffine(frame, warp_all, (112, 112), flags=cv2.INTER_LINEAR)

mse_all = np.mean((aligned_all.astype(float) - cpp_aligned.astype(float)) ** 2)
corr_all, _ = pearsonr(aligned_all.flatten(), cpp_aligned.flatten())

print(f"\n[2] Using ALL 68 points:")
print(f"  MSE:  {mse_all:.2f}")
print(f"  Corr: {corr_all:.6f}")

print("\n" + "="*70)
if mse_all < mse_rigid:
    print("✓ ALL points gives BETTER match!")
    print("  → OpenFace may have used rigid=false")
else:
    print("✓ RIGID points gives BETTER match")
    print("  → OpenFace used rigid=true as expected")
print("="*70)

# Save comparison
comparison = np.hstack([aligned_rigid, aligned_all, cpp_aligned])
cv2.imwrite("rigid_vs_all_comparison.png", comparison)
print("\nSaved comparison to: rigid_vs_all_comparison.png")
