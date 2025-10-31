#!/usr/bin/env python3
"""
Test if alignment works with just scale + translation (no rotation)
"""

import numpy as np
import pandas as pd
import cv2

# Load test data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)

test_frames = [1, 617, 740]

from pdm_parser import PDMParser
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
reference_shape = mean_shape_scaled[:136].reshape(68, 2)

RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

print("=" * 70)
print("Testing Scale + Translation Only (No Rotation)")
print("=" * 70)

# Load C++ aligned for comparison
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

def compute_scale_only(src, dst):
    """Compute scale factor only, no rotation"""
    n = src.shape[0]
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)
    s_src = np.sqrt(np.sum(src_centered ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)
    return s_dst / s_src

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()

    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    p_tx = row['p_tx']
    p_ty = row['p_ty']

    src_rigid = landmarks_68[RIGID_INDICES]
    dst_rigid = reference_shape[RIGID_INDICES]

    # Compute scale only
    scale = compute_scale_only(src_rigid, dst_rigid)

    # Build warp matrix with scale but NO rotation
    warp = np.zeros((2, 3), dtype=np.float32)
    warp[0, 0] = scale  # Scale X
    warp[1, 1] = scale  # Scale Y
    # No off-diagonal elements = no rotation

    # Transform translation
    T = np.array([p_tx * scale, p_ty * scale])
    warp[0, 2] = -T[0] + 112/2 + 2
    warp[1, 2] = -T[1] + 112/2 - 2

    aligned = cv2.warpAffine(frame, warp, (112, 112), flags=cv2.INTER_LINEAR)

    print(f"\nFrame {frame_num}:")
    print(f"  Scale factor: {scale:.6f}")
    print(f"  Rotation: 0.00° (no rotation)")

    if frame_num == 1:
        corr = np.corrcoef(cpp_aligned.flatten(), aligned.flatten())[0, 1]
        print(f"  Correlation with C++: {corr:.6f}")

        # Save comparison
        vis = np.hstack([cpp_aligned, aligned])
        cv2.imwrite("test_no_rotation_frame1.png", vis)

cap.release()

print("\n" + "=" * 70)
print("If this gives good results, OpenFace may not be using rotation at all")
print("=" * 70)
