#!/usr/bin/env python3
"""
Test if rotation matrix needs to be transposed or inverted
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Load validation CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
row = df.iloc[0]

# Load PDM
from pdm_parser import PDMParser
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
reference_shape = mean_shape_scaled[:136].reshape(68, 2)

# Extract landmarks
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

# Load C++ aligned
comparison = cv2.imread("alignment_validation_output/frame_0001_comparison.png")
cpp_aligned = comparison[:, 112:224]

# Rigid indices
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

def extract_rigid(landmarks):
    return landmarks[RIGID_INDICES]

def align_shapes_with_scale(src, dst):
    """Current implementation"""
    n = src.shape[0]

    # Mean normalize
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_norm = src - mean_src
    dst_norm = dst - mean_dst

    # RMS scale
    s_src = np.sqrt(np.sum(src_norm ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_norm ** 2) / n)

    src_norm /= s_src
    dst_norm /= s_dst

    # Kabsch
    U, S, Vt = np.linalg.svd(src_norm.T @ dst_norm)
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    corr[1, 1] = 1 if d > 0 else -1
    R = Vt.T @ corr @ U.T

    scale = s_dst / s_src
    return scale * R

def build_warp_matrix(scale_rot, tx, ty, offset_x=2, offset_y=-2):
    """Build warp matrix with offsets"""
    warp = np.zeros((2, 3), dtype=np.float32)
    warp[:2, :2] = scale_rot
    T = scale_rot @ np.array([tx, ty])
    warp[0, 2] = -T[0] + 112/2 + offset_x
    warp[1, 2] = -T[1] + 112/2 + offset_y
    return warp

# Extract rigid points
src_rigid = extract_rigid(landmarks_68)
dst_rigid = extract_rigid(reference_shape)

print("=" * 70)
print("Testing Rotation Matrix Variations")
print("=" * 70)

# Test 1: Current implementation
scale_rot_current = align_shapes_with_scale(src_rigid, dst_rigid)
warp_current = build_warp_matrix(scale_rot_current, pose_tx, pose_ty)
aligned_current = cv2.warpAffine(frame, warp_current, (112, 112), flags=cv2.INTER_LINEAR)
corr_current = np.corrcoef(cpp_aligned.flatten(), aligned_current.flatten())[0, 1]

angle_current = np.arctan2(scale_rot_current[1,0], scale_rot_current[0,0]) * 180 / np.pi
print(f"\n[1] Current implementation:")
print(f"  Rotation angle: {angle_current:.2f}°")
print(f"  Correlation: {corr_current:.6f}")

# Test 2: Transpose the rotation matrix
scale_rot_transpose = scale_rot_current.T
warp_transpose = build_warp_matrix(scale_rot_transpose, pose_tx, pose_ty)
aligned_transpose = cv2.warpAffine(frame, warp_transpose, (112, 112), flags=cv2.INTER_LINEAR)
corr_transpose = np.corrcoef(cpp_aligned.flatten(), aligned_transpose.flatten())[0, 1]

angle_transpose = np.arctan2(scale_rot_transpose[1,0], scale_rot_transpose[0,0]) * 180 / np.pi
print(f"\n[2] Transposed rotation matrix:")
print(f"  Rotation angle: {angle_transpose:.2f}°")
print(f"  Correlation: {corr_transpose:.6f}")
if corr_transpose > corr_current:
    print(f"  ✓ BETTER by {corr_transpose - corr_current:.6f}!")

# Test 3: Inverse rotation
scale_rot_inv = np.linalg.inv(scale_rot_current)
warp_inv = build_warp_matrix(scale_rot_inv, pose_tx, pose_ty)
aligned_inv = cv2.warpAffine(frame, warp_inv, (112, 112), flags=cv2.INTER_LINEAR)
corr_inv = np.corrcoef(cpp_aligned.flatten(), aligned_inv.flatten())[0, 1]

angle_inv = np.arctan2(scale_rot_inv[1,0], scale_rot_inv[0,0]) * 180 / np.pi
print(f"\n[3] Inverted rotation matrix:")
print(f"  Rotation angle: {angle_inv:.2f}°")
print(f"  Correlation: {corr_inv:.6f}")
if corr_inv > corr_current:
    print(f"  ✓ BETTER by {corr_inv - corr_current:.6f}!")

# Test 4: Swap src and dst in alignment
scale_rot_swapped = align_shapes_with_scale(dst_rigid, src_rigid)
warp_swapped = build_warp_matrix(scale_rot_swapped, pose_tx, pose_ty)
aligned_swapped = cv2.warpAffine(frame, warp_swapped, (112, 112), flags=cv2.INTER_LINEAR)
corr_swapped = np.corrcoef(cpp_aligned.flatten(), aligned_swapped.flatten())[0, 1]

angle_swapped = np.arctan2(scale_rot_swapped[1,0], scale_rot_swapped[0,0]) * 180 / np.pi
print(f"\n[4] Swapped src/dst in align_shapes:")
print(f"  Rotation angle: {angle_swapped:.2f}°")
print(f"  Correlation: {corr_swapped:.6f}")
if corr_swapped > corr_current:
    print(f"  ✓ BETTER by {corr_swapped - corr_current:.6f}!")

# Test 5: Negate rotation component
scale = np.sqrt(np.abs(np.linalg.det(scale_rot_current)))
scale_rot_negated = scale_rot_current.copy()
scale_rot_negated[0,1] = -scale_rot_negated[0,1]
scale_rot_negated[1,0] = -scale_rot_negated[1,0]
warp_negated = build_warp_matrix(scale_rot_negated, pose_tx, pose_ty)
aligned_negated = cv2.warpAffine(frame, warp_negated, (112, 112), flags=cv2.INTER_LINEAR)
corr_negated = np.corrcoef(cpp_aligned.flatten(), aligned_negated.flatten())[0, 1]

angle_negated = np.arctan2(scale_rot_negated[1,0], scale_rot_negated[0,0]) * 180 / np.pi
print(f"\n[5] Negated off-diagonal (rotate opposite direction):")
print(f"  Rotation angle: {angle_negated:.2f}°")
print(f"  Correlation: {corr_negated:.6f}")
if corr_negated > corr_current:
    print(f"  ✓ BETTER by {corr_negated - corr_current:.6f}!")

# Save comparison images
vis = np.zeros((112 * 2, 112 * 3, 3), dtype=np.uint8)
vis[:112, :112] = cpp_aligned
vis[:112, 112:224] = aligned_current
vis[:112, 224:336] = aligned_transpose
vis[112:, :112] = aligned_inv
vis[112:, 112:224] = aligned_swapped
vis[112:, 224:336] = aligned_negated

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(vis, "C++", (5, 15), font, 0.4, (255, 255, 255), 1)
cv2.putText(vis, f"Current ({angle_current:.1f}deg)", (117, 15), font, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"Transposed ({angle_transpose:.1f}deg)", (229, 15), font, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"Inverted ({angle_inv:.1f}deg)", (5, 127), font, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"Swapped ({angle_swapped:.1f}deg)", (117, 127), font, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"Negated ({angle_negated:.1f}deg)", (229, 127), font, 0.3, (255, 255, 255), 1)

cv2.imwrite("rotation_test_comparison.png", vis)
print(f"\n✓ Saved rotation_test_comparison.png")

print("\n" + "=" * 70)
