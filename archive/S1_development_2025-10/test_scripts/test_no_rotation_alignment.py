#!/usr/bin/env python3
"""
Test: What if PDM-reconstructed landmarks don't need rotation?

Hypothesis: Since CSV landmarks are PDM-reconstructed (CalcShape2D output),
they're already in canonical orientation. Maybe we only need scale + translation!
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from pdm_parser import PDMParser

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_2d = mean_shape_scaled[:136].reshape(68, 2)

# Load CSV and video
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"

# Rigid indices
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

def align_with_scale_only(src, dst):
    """Compute ONLY scale (no rotation) between point sets"""
    n = src.shape[0]

    # Center both
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)

    # Compute scale
    s_src = np.sqrt(np.sum(src_centered ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)
    scale = s_dst / s_src

    # Return IDENTITY rotation with scale
    return scale * np.eye(2)

def align_face_no_rotation(image, landmarks, pose_tx, pose_ty):
    """Align face using only scale + translation (NO rotation)"""
    # Extract rigid points
    source_rigid = landmarks[RIGID_INDICES]
    dest_rigid = mean_shape_2d[RIGID_INDICES]

    # Compute scale-only transform
    scale_matrix = align_with_scale_only(source_rigid, dest_rigid)

    # Build warp matrix
    warp_matrix = np.zeros((2, 3), dtype=np.float32)
    warp_matrix[:2, :2] = scale_matrix

    # Transform pose translation
    T = np.array([pose_tx, pose_ty], dtype=np.float32)
    T_transformed = scale_matrix @ T

    # Centering translation
    warp_matrix[0, 2] = -T_transformed[0] + 112/2
    warp_matrix[1, 2] = -T_transformed[1] + 112/2

    # Apply warp
    aligned = cv2.warpAffine(image, warp_matrix, (112, 112), flags=cv2.INTER_LINEAR)

    return aligned, scale_matrix

print("=" * 80)
print("Test: Alignment with Scale Only (NO Rotation)")
print("=" * 80)

cap = cv2.VideoCapture(video_path)
cpp_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

test_frames = [1, 493, 617, 863]

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    if not ret:
        continue

    # Get landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks = np.stack([x, y], axis=1)

    # Get pose
    pose_tx = row['p_tx']
    pose_ty = row['p_ty']

    # Align with NO rotation
    python_no_rot, scale_mat = align_face_no_rotation(frame, landmarks, pose_tx, pose_ty)

    # Load C++ aligned
    cpp_file = cpp_dir / f"frame_det_00_{frame_num:06d}.bmp"
    cpp_aligned = cv2.imread(str(cpp_file))

    # Compute rotation angle (should be 0° if scale-only)
    angle = np.arctan2(scale_mat[1,0], scale_mat[0,0]) * 180 / np.pi

    # Compare
    expression = "eyes closed" if frame_num == 617 else ""
    print(f"\nFrame {frame_num:4d} {expression:>12}")
    print(f"  Scale matrix rotation: {angle:.2f}° (should be 0°)")
    print(f"  Scale factor: {scale_mat[0,0]:.4f}")

    # Visual comparison
    comparison = np.hstack([cpp_aligned, python_no_rot])
    output_file = f"test_no_rotation_frame_{frame_num}.png"
    cv2.imwrite(output_file, comparison)
    print(f"  Saved: {output_file}")

cap.release()

print("\n" + "=" * 80)
print("Results:")
print("=" * 80)
print("Check the output images:")
print("  - Left: C++ aligned (ground truth)")
print("  - Right: Python with scale-only (no rotation)")
print()
print("If they look similar:")
print("  ✓ Hypothesis CONFIRMED - PDM-reconstructed landmarks don't need rotation!")
print("  → Fix: Use scale + translation only")
print()
print("If they look different:")
print("  ✗ Hypothesis REJECTED - Still need rotation computation")
print("  → Next: Implement CalcParams to re-fit PDM")
print("=" * 80)
