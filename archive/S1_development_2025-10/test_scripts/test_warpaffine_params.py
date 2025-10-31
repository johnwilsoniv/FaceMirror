#!/usr/bin/env python3
"""
Test different cv2.warpAffine parameters to match OpenFace C++ exactly
"""

import cv2
import numpy as np
import pandas as pd
from openface22_face_aligner import OpenFace22FaceAligner

# Initialize aligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

# Load validation CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]

# Extract landmarks and pose
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)
pose_tx = row['p_tx']
pose_ty = row['p_ty']

# Load frame
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

# Get warp matrix from aligner
source_rigid = aligner._extract_rigid_points(landmarks_68)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
warp_matrix = aligner._build_warp_matrix(scale_rot, pose_tx, pose_ty)

# Load C++ aligned face
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

print("=" * 70)
print("Testing cv2.warpAffine Parameters")
print("=" * 70)

# Test different interpolation methods
interpolations = [
    (cv2.INTER_LINEAR, "INTER_LINEAR (default)"),
    (cv2.INTER_NEAREST, "INTER_NEAREST"),
    (cv2.INTER_CUBIC, "INTER_CUBIC"),
    (cv2.INTER_LANCZOS4, "INTER_LANCZOS4"),
]

print("\n[1] Testing interpolation methods:")
print("-" * 70)
best_corr = 0
best_method = None

for interp, name in interpolations:
    aligned = cv2.warpAffine(frame, warp_matrix, (112, 112), flags=interp)

    # Compute correlation
    corr = np.corrcoef(cpp_aligned.flatten(), aligned.flatten())[0, 1]
    mse = np.mean((cpp_aligned.astype(float) - aligned.astype(float)) ** 2)

    print(f"{name:30s}: r={corr:.6f}, MSE={mse:.2f}")

    if corr > best_corr:
        best_corr = corr
        best_method = (interp, name)

print(f"\n✓ Best: {best_method[1]} with r={best_corr:.6f}")

# Test with different border modes
print("\n[2] Testing border modes (with INTER_LINEAR):")
print("-" * 70)

border_modes = [
    (cv2.BORDER_CONSTANT, "BORDER_CONSTANT (default)"),
    (cv2.BORDER_REPLICATE, "BORDER_REPLICATE"),
    (cv2.BORDER_REFLECT, "BORDER_REFLECT"),
    (cv2.BORDER_WRAP, "BORDER_WRAP"),
    (cv2.BORDER_REFLECT_101, "BORDER_REFLECT_101"),
]

for border, name in border_modes:
    aligned = cv2.warpAffine(frame, warp_matrix, (112, 112),
                            flags=cv2.INTER_LINEAR,
                            borderMode=border)

    corr = np.corrcoef(cpp_aligned.flatten(), aligned.flatten())[0, 1]
    mse = np.mean((cpp_aligned.astype(float) - aligned.astype(float)) ** 2)

    print(f"{name:30s}: r={corr:.6f}, MSE={mse:.2f}")

# Test matrix precision
print("\n[3] Testing matrix precision:")
print("-" * 70)

# Current matrix (float32)
aligned_f32 = cv2.warpAffine(frame, warp_matrix.astype(np.float32), (112, 112), flags=cv2.INTER_LINEAR)
corr_f32 = np.corrcoef(cpp_aligned.flatten(), aligned_f32.flatten())[0, 1]

# Double precision (float64)
aligned_f64 = cv2.warpAffine(frame, warp_matrix.astype(np.float64), (112, 112), flags=cv2.INTER_LINEAR)
corr_f64 = np.corrcoef(cpp_aligned.flatten(), aligned_f64.flatten())[0, 1]

print(f"float32 matrix: r={corr_f32:.6f}")
print(f"float64 matrix: r={corr_f64:.6f}")
print(f"Difference: {abs(corr_f64 - corr_f32):.9f}")

# Print the exact warp matrix being used
print("\n[4] Warp Matrix Values:")
print("-" * 70)
print(f"[[{warp_matrix[0,0]:18.15f}, {warp_matrix[0,1]:18.15f}, {warp_matrix[0,2]:18.15f}],")
print(f" [{warp_matrix[1,0]:18.15f}, {warp_matrix[1,1]:18.15f}, {warp_matrix[1,2]:18.15f}]]")

# Test small adjustments to translation
print("\n[5] Testing small translation adjustments:")
print("-" * 70)

for dx in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    for dy in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        wm_test = warp_matrix.copy()
        wm_test[0, 2] += dx
        wm_test[1, 2] += dy

        aligned = cv2.warpAffine(frame, wm_test, (112, 112), flags=cv2.INTER_LINEAR)
        corr = np.corrcoef(cpp_aligned.flatten(), aligned.flatten())[0, 1]

        if corr > 0.75:  # Only print if better than baseline
            print(f"  dx={dx:+5.1f}, dy={dy:+5.1f}: r={corr:.6f}")

print("\n" + "=" * 70)
print("Parameter testing complete")
print("=" * 70)
