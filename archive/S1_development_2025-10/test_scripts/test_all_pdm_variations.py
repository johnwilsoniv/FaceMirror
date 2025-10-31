#!/usr/bin/env python3
"""
Test different ways of interpreting the PDM mean shape
"""

import numpy as np
import cv2
import pandas as pd
from pdm_parser import PDMParser
from scipy.stats import pearsonr

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")

print("="*70)
print("Testing PDM Mean Shape Interpretations")
print("="*70)

# Different interpretations
variations = {}

# 1. Current: reshape(68, 3), take [:, :2]
mean_1 = (pdm.mean_shape.reshape(68, 3) * 0.7)[:, :2]
variations["Current (68,3)[:,:2]"] = mean_1

# 2. Original wrong way: take first 136, then reshape
mean_2 = (pdm.mean_shape * 0.7)[:136].reshape(68, 2)
variations["First 136 then reshape"] = mean_2

# 3. Reshape (68, 3), take [:, :2], then TRANSPOSE
mean_3 = ((pdm.mean_shape.reshape(68, 3) * 0.7)[:, :2]).T.T  # Same as mean_1
variations["(68,3)[:,:2] transposed"] = ((pdm.mean_shape.reshape(68, 3) * 0.7)[:, :2])

# 4. Try X,Y swapped
mean_4 = np.zeros((68, 2))
mean_4[:, 0] = (pdm.mean_shape.reshape(68, 3) * 0.7)[:, 1]  # Y as X
mean_4[:, 1] = (pdm.mean_shape.reshape(68, 3) * 0.7)[:, 0]  # X as Y
variations["X and Y swapped"] = mean_4

# 5. Try X negated
mean_5 = (pdm.mean_shape.reshape(68, 3) * 0.7)[:, :2].copy()
mean_5[:, 0] *= -1
variations["X negated"] = mean_5

# 6. Try Y negated
mean_6 = (pdm.mean_shape.reshape(68, 3) * 0.7)[:, :2].copy()
mean_6[:, 1] *= -1
variations["Y negated"] = mean_6

# 7. Try both negated
mean_7 = (pdm.mean_shape.reshape(68, 3) * 0.7)[:, :2].copy()
mean_7 *= -1
variations["Both negated"] = mean_7

print("\nPDM Mean Shape Statistics:")
for name, mean_shape in variations.items():
    print(f"\n{name}:")
    print(f"  X range: [{mean_shape[:,0].min():.2f}, {mean_shape[:,0].max():.2f}]")
    print(f"  Y range: [{mean_shape[:,1].min():.2f}, {mean_shape[:,1].max():.2f}]")
    print(f"  Center: ({mean_shape[:,0].mean():.2f}, {mean_shape[:,1].mean():.2f})")

print("\n" + "="*70)
print("Now testing which gives best alignment...")
print("="*70)

# Load test frame data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]
frame_num = int(row['frame'])

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

# Load C++ aligned
cpp_aligned = cv2.imread(f"pyfhog_validation_output/IMG_0942_left_mirrored_aligned/frame_det_00_{frame_num:06d}.bmp")

# Test each variation (using simplified aligner logic)
from openface22_face_aligner import OpenFace22FaceAligner
aligner = OpenFace22FaceAligner("In-the-wild_aligned_PDM_68.txt")

RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

results = []
for name, ref_shape in variations.items():
    source_rigid = landmarks_68[RIGID_INDICES]
    dest_rigid = ref_shape[RIGID_INDICES]

    scale_rot = aligner._align_shapes_with_scale(source_rigid, dest_rigid)
    warp_matrix = aligner._build_warp_matrix(scale_rot, pose_tx, pose_ty)
    aligned = cv2.warpAffine(frame, warp_matrix, (112, 112), flags=cv2.INTER_LINEAR)

    mse = np.mean((aligned.astype(float) - cpp_aligned.astype(float)) ** 2)
    corr, _ = pearsonr(aligned.flatten(), cpp_aligned.flatten())

    results.append((name, mse, corr))
    print(f"{name:30s}: MSE={mse:8.2f}, r={corr:.6f}")

# Find best
best = min(results, key=lambda x: x[1])
print(f"\n{'='*70}")
print(f"BEST: {best[0]}")
print(f"  MSE={best[1]:.2f}, r={best[2]:.6f}")
print("="*70)
