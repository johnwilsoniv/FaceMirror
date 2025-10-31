#!/usr/bin/env python3
"""
Test different warpAffine parameters to see which matches C++ best
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

# Compute warp matrix (same as in aligner)
source_rigid = aligner._extract_rigid_points(landmarks_68)
dest_rigid = aligner._extract_rigid_points(aligner.reference_shape)
scale_rot_matrix = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
warp_matrix = aligner._build_warp_matrix(scale_rot_matrix, pose_tx, pose_ty)

print("=" * 70)
print("Testing Different warpAffine Parameters")
print("=" * 70)

# Test different interpolation methods
methods = [
    ("INTER_LINEAR", cv2.INTER_LINEAR),
    ("INTER_CUBIC", cv2.INTER_CUBIC),
    ("INTER_LANCZOS4", cv2.INTER_LANCZOS4),
    ("INTER_LINEAR + WARP_INVERSE_MAP", cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
]

results = []
for name, flags in methods:
    aligned = cv2.warpAffine(frame, warp_matrix, (112, 112), flags=flags)

    # Compute MSE and correlation
    mse = np.mean((aligned.astype(float) - cpp_aligned.astype(float)) ** 2)
    corr, _ = pearsonr(aligned.flatten(), cpp_aligned.flatten())

    results.append((name, mse, corr))
    print(f"{name:40s}: MSE={mse:8.2f}, r={corr:.6f}")

# Find best
best = min(results, key=lambda x: x[1])
print("\n" + "=" * 70)
print(f"Best method: {best[0]} (MSE={best[1]:.2f}, r={best[2]:.6f})")
print("=" * 70)
