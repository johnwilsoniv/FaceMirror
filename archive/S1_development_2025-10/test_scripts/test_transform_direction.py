#!/usr/bin/env python3
"""
Test if we need the inverse transform direction for warpAffine
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Load test data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Load video
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cap = cv2.VideoCapture(video_path)

# Test frames
test_frames = [1, 617, 740]

from pdm_parser import PDMParser
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
reference_shape = mean_shape_scaled[:136].reshape(68, 2)

RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

def align_shapes_with_scale(src, dst):
    """Current implementation"""
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

def build_warp_matrix(scale_rot, tx, ty):
    warp = np.zeros((2, 3), dtype=np.float32)
    warp[:2, :2] = scale_rot
    T = scale_rot @ np.array([tx, ty])
    warp[0, 2] = -T[0] + 112/2 + 2
    warp[1, 2] = -T[1] + 112/2 - 2
    return warp

print("=" * 70)
print("Testing Transform Direction")
print("=" * 70)

# Load C++ aligned for comparison
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()

    # Extract landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    pose_tx = row['p_tx']
    pose_ty = row['p_ty']

    src_rigid = landmarks_68[RIGID_INDICES]
    dst_rigid = reference_shape[RIGID_INDICES]

    # Test 1: Current (src->dst, then transpose)
    scale_rot_1 = align_shapes_with_scale(src_rigid, dst_rigid).T
    warp_1 = build_warp_matrix(scale_rot_1, pose_tx, pose_ty)
    aligned_1 = cv2.warpAffine(frame, warp_1, (112, 112), flags=cv2.INTER_LINEAR)
    angle_1 = np.arctan2(scale_rot_1[1,0], scale_rot_1[0,0]) * 180 / np.pi

    # Test 2: Reverse direction (dst->src, no transpose)
    scale_rot_2 = align_shapes_with_scale(dst_rigid, src_rigid)
    warp_2 = build_warp_matrix(scale_rot_2, pose_tx, pose_ty)
    aligned_2 = cv2.warpAffine(frame, warp_2, (112, 112), flags=cv2.INTER_LINEAR)
    angle_2 = np.arctan2(scale_rot_2[1,0], scale_rot_2[0,0]) * 180 / np.pi

    # Test 3: Reverse direction + transpose
    scale_rot_3 = align_shapes_with_scale(dst_rigid, src_rigid).T
    warp_3 = build_warp_matrix(scale_rot_3, pose_tx, pose_ty)
    aligned_3 = cv2.warpAffine(frame, warp_3, (112, 112), flags=cv2.INTER_LINEAR)
    angle_3 = np.arctan2(scale_rot_3[1,0], scale_rot_3[0,0]) * 180 / np.pi

    if frame_num == 1:
        # Only compute correlation for frame 1
        corr_1 = np.corrcoef(cpp_aligned.flatten(), aligned_1.flatten())[0, 1]
        corr_2 = np.corrcoef(cpp_aligned.flatten(), aligned_2.flatten())[0, 1]
        corr_3 = np.corrcoef(cpp_aligned.flatten(), aligned_3.flatten())[0, 1]

        print(f"\nFrame {frame_num} (with C++ comparison):")
        print(f"  Method 1 (src->dst + transpose): angle={angle_1:6.2f}°, r={corr_1:.6f}")
        print(f"  Method 2 (dst->src, no transp):  angle={angle_2:6.2f}°, r={corr_2:.6f}")
        print(f"  Method 3 (dst->src + transpose): angle={angle_3:6.2f}°, r={corr_3:.6f}")

        # Save visual comparison
        vis = np.hstack([cpp_aligned, aligned_1, aligned_2, aligned_3])
        cv2.putText(vis, "C++", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "M1", (117, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "M2", (229, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "M3", (341, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(f"transform_direction_test_frame_{frame_num}.png", vis)
    else:
        print(f"\nFrame {frame_num}:")
        print(f"  Method 1 (src->dst + transpose): angle={angle_1:6.2f}°")
        print(f"  Method 2 (dst->src, no transp):  angle={angle_2:6.2f}°")
        print(f"  Method 3 (dst->src + transpose): angle={angle_3:6.2f}°")

cap.release()

print("\n" + "=" * 70)
print("If rotation angles are consistent across frames, that method is correct")
print("=" * 70)
