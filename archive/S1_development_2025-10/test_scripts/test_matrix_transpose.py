#!/usr/bin/env python3
"""
Test if transposing the rotation matrix fixes the rotation issue
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
print("Testing: Matrix Transpose")
print("="*70)

# Compute transform
source_rigid = aligner._extract_rigid_points(landmarks_68)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)

print(f"\nOriginal scale_rot:\n{scale_rot}")

# Test 1: Original (current implementation)
warp_orig = aligner._build_warp_matrix(scale_rot, pose_tx, pose_ty)
aligned_orig = cv2.warpAffine(frame, warp_orig, (112, 112), flags=cv2.INTER_LINEAR)
mse_orig = np.mean((aligned_orig.astype(float) - cpp_aligned.astype(float)) ** 2)
corr_orig, _ = pearsonr(aligned_orig.flatten(), cpp_aligned.flatten())

print(f"\n[1] Original: MSE={mse_orig:.2f}, r={corr_orig:.6f}")

# Test 2: Transpose scale_rot
scale_rot_T = scale_rot.T
warp_T = aligner._build_warp_matrix(scale_rot_T, pose_tx, pose_ty)
aligned_T = cv2.warpAffine(frame, warp_T, (112, 112), flags=cv2.INTER_LINEAR)
mse_T = np.mean((aligned_T.astype(float) - cpp_aligned.astype(float)) ** 2)
corr_T, _ = pearsonr(aligned_T.flatten(), cpp_aligned.flatten())

print(f"[2] Transposed scale_rot: MSE={mse_T:.2f}, r={corr_T:.6f}")

# Test 3: Negate off-diagonal elements (fixes reflection)
scale_rot_neg = scale_rot.copy()
scale_rot_neg[0, 1] *= -1
scale_rot_neg[1, 0] *= -1
warp_neg = aligner._build_warp_matrix(scale_rot_neg, pose_tx, pose_ty)
aligned_neg = cv2.warpAffine(frame, warp_neg, (112, 112), flags=cv2.INTER_LINEAR)
mse_neg = np.mean((aligned_neg.astype(float) - cpp_aligned.astype(float)) ** 2)
corr_neg, _ = pearsonr(aligned_neg.flatten(), cpp_aligned.flatten())

print(f"[3] Negated off-diagonals: MSE={mse_neg:.2f}, r={corr_neg:.6f}")

# Find best
results = [
    ("Original", mse_orig, corr_orig, aligned_orig),
    ("Transposed", mse_T, corr_T, aligned_T),
    ("Negated off-diag", mse_neg, corr_neg, aligned_neg)
]

best = min(results, key=lambda x: x[1])
print(f"\n{'='*70}")
print(f"Best: {best[0]} (MSE={best[1]:.2f}, r={best[2]:.6f})")
print("="*70)

# Save comparison
comparison = np.hstack([results[0][3], results[1][3], results[2][3], cpp_aligned])
cv2.imwrite("matrix_variants_comparison.png", comparison)
print("\nSaved: matrix_variants_comparison.png")
print("  [Original | Transposed | Neg off-diag | C++]")
