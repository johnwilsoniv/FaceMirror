#!/usr/bin/env python3
"""
Test if we need matrix inverse for warpAffine

Key insight: cv2.warpAffine uses INVERSE mapping:
  dst_pixel = src_image[M @ dst_coords]
So to map source landmarks to destination positions, M should be INVERSE of landmark transform
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

def align_shapes_with_scale(src, dst):
    n = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_norm = src - mean_src
    dst_norm = dst - mean_dst
    s_src = np.sqrt(np.sum(src_norm ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_norm ** 2) / n)
    src_norm_scaled = src_norm / s_src
    dst_norm_scaled = dst_norm / s_dst
    U, S, Vt = np.linalg.svd(src_norm_scaled.T @ dst_norm_scaled)
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    corr[1, 1] = 1 if d > 0 else -1
    R = Vt.T @ corr @ U.T
    scale = s_dst / s_src
    return scale * R

def build_warp_matrix_no_pose_transform(scale_rot):
    """Build warp without transforming pose through scale_rot"""
    warp = np.zeros((2, 3), dtype=np.float32)
    warp[:2, :2] = scale_rot
    # Simple centering without pose transform
    warp[0, 2] = 112/2
    warp[1, 2] = 112/2
    return warp

print("=" * 70)
print("Testing Matrix Inverse for warpAffine")
print("=" * 70)

# Load C++ aligned for comparison
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()

    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    src_rigid = landmarks_68[RIGID_INDICES]
    dst_rigid = reference_shape[RIGID_INDICES]

    # Compute forward transform (src->dst)
    scale_rot_forward = align_shapes_with_scale(src_rigid, dst_rigid)

    # Test 1: Forward transform as-is
    warp_1 = build_warp_matrix_no_pose_transform(scale_rot_forward)
    aligned_1 = cv2.warpAffine(frame, warp_1, (112, 112), flags=cv2.INTER_LINEAR)
    angle_1 = np.arctan2(scale_rot_forward[1,0], scale_rot_forward[0,0]) * 180 / np.pi

    # Test 2: Inverse of forward transform
    scale_rot_inv = np.linalg.inv(scale_rot_forward)
    warp_2 = build_warp_matrix_no_pose_transform(scale_rot_inv)
    aligned_2 = cv2.warpAffine(frame, warp_2, (112, 112), flags=cv2.INTER_LINEAR)
    angle_2 = np.arctan2(scale_rot_inv[1,0], scale_rot_inv[0,0]) * 180 / np.pi

    if frame_num == 1:
        corr_1 = np.corrcoef(cpp_aligned.flatten(), aligned_1.flatten())[0, 1]
        corr_2 = np.corrcoef(cpp_aligned.flatten(), aligned_2.flatten())[0, 1]

        print(f"\nFrame {frame_num}:")
        print(f"  Forward transform:      angle={angle_1:6.2f}°, r={corr_1:.6f}")
        print(f"  Inverse transform:      angle={angle_2:6.2f}°, r={corr_2:.6f}")
    else:
        print(f"\nFrame {frame_num}:")
        print(f"  Forward transform:      angle={angle_1:6.2f}°")
        print(f"  Inverse transform:      angle={angle_2:6.2f}°")

cap.release()

print("\n" + "=" * 70)
print("Key: rotation angle should be CONSISTENT across all frames for upright faces")
print("=" * 70)
