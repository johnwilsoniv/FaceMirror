#!/usr/bin/env python3
"""
Full PDM Reconstruction: Replicate CalcShape3D + CalcShape2D

This implements the complete OpenFace PDM reconstruction pipeline:
1. CalcShape3D: mean_shape + principal_components @ params_local
2. CalcShape2D: Apply 3D rotation, scale, translate, project to 2D

Goal: Verify that this produces landmarks identical to CSV
"""

import numpy as np
import pandas as pd
from pdm_parser import PDMParser

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

def euler_to_rotation_matrix(rx, ry, rz):
    """Convert Euler angles to 3D rotation matrix (XYZ convention)"""
    s1, s2, s3 = np.sin(rx), np.sin(ry), np.sin(rz)
    c1, c2, c3 = np.cos(rx), np.cos(ry), np.cos(rz)

    R = np.array([
        [c2*c3,           -c2*s3,          s2],
        [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
        [s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3,  c1*c2]
    ])
    return R

def calc_shape_3d(mean_shape, principal_components, params_local):
    """
    CalcShape3D: Reconstruct 3D shape from PCA parameters

    From PDM.cpp line 153:
        out_shape = mean_shape + princ_comp * p_local;

    Args:
        mean_shape: (204, 1) - PDM mean shape [x0..x67, y0..y67, z0..z67]
        principal_components: (204, 34) - PCA basis vectors
        params_local: (34,) - Expression/identity parameters

    Returns:
        shape_3d: (204, 1) - Reconstructed 3D shape
    """
    # Ensure params_local is column vector
    if params_local.ndim == 1:
        params_local = params_local.reshape(-1, 1)

    # Matrix multiplication: mean_shape + (principal_components @ params_local)
    shape_3d = mean_shape + principal_components @ params_local

    return shape_3d

def calc_shape_2d(shape_3d, params_global):
    """
    CalcShape2D: Project 3D shape to 2D with rotation, scale, translation

    From PDM.cpp lines 159-188:
        1. Extract scale, rotation (rx, ry, rz), translation (tx, ty)
        2. Build 3D rotation matrix from Euler angles
        3. For each point: rotate in 3D, scale, project to 2D, translate

    Args:
        shape_3d: (204, 1) - 3D shape from CalcShape3D
        params_global: [scale, rx, ry, rz, tx, ty]

    Returns:
        shape_2d: (136,) - 2D landmarks [x0..x67, y0..y67]
    """
    n = 68

    # Extract params_global
    scale = params_global[0]
    rx = params_global[1]
    ry = params_global[2]
    rz = params_global[3]
    tx = params_global[4]
    ty = params_global[5]

    # Build rotation matrix
    R = euler_to_rotation_matrix(rx, ry, rz)

    # Reshape shape_3d to (68, 3) format
    # PDM format: [x0..x67, y0..y67, z0..z67]
    shape_x = shape_3d[0:n, 0]
    shape_y = shape_3d[n:2*n, 0]
    shape_z = shape_3d[2*n:3*n, 0]
    shape_3d_points = np.stack([shape_x, shape_y, shape_z], axis=1)  # (68, 3)

    # Apply rotation and scale
    # From PDM.cpp line 185-186:
    # out_shape(i,0) = s * (R[0,0]*x + R[0,1]*y + R[0,2]*z) + tx
    # out_shape(i+n,0) = s * (R[1,0]*x + R[1,1]*y + R[1,2]*z) + ty
    rotated_scaled = scale * (R @ shape_3d_points.T).T  # (68, 3)

    # Project to 2D (take first two rows of rotation result)
    out_x = rotated_scaled[:, 0] + tx
    out_y = rotated_scaled[:, 1] + ty

    # Return in (68, 2) format
    return np.stack([out_x, out_y], axis=1)

print("=" * 80)
print("Full PDM Reconstruction Test")
print("=" * 80)

# Test frame 493 (stable, eyes open)
test_frames = [1, 493, 617, 863]

for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get CSV landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    csv_x = row[x_cols].values.astype(np.float32)
    csv_y = row[y_cols].values.astype(np.float32)
    csv_landmarks = np.stack([csv_x, csv_y], axis=1)  # (68, 2)

    # Get params_global
    params_global = np.array([
        row['p_scale'],
        row['p_rx'],
        row['p_ry'],
        row['p_rz'],
        row['p_tx'],
        row['p_ty']
    ])

    # Get params_local (p_0 through p_33)
    params_local = np.array([row[f'p_{i}'] for i in range(34)], dtype=np.float32)

    # Reconstruct using PDM
    # Step 1: CalcShape3D
    shape_3d = calc_shape_3d(pdm.mean_shape, pdm.princ_comp, params_local)

    # Step 2: CalcShape2D
    reconstructed_landmarks = calc_shape_2d(shape_3d, params_global)

    # Compare
    diff = csv_landmarks - reconstructed_landmarks
    rmse_x = np.sqrt(np.mean(diff[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(diff[:, 1] ** 2))
    max_diff_x = np.max(np.abs(diff[:, 0]))
    max_diff_y = np.max(np.abs(diff[:, 1]))

    expression = "eyes closed" if frame_num == 617 else ""

    print(f"\nFrame {frame_num:4d} {expression:>12}")
    print(f"  RMSE: X={rmse_x:6.3f} px, Y={rmse_y:6.3f} px")
    print(f"  Max:  X={max_diff_x:6.3f} px, Y={max_diff_y:6.3f} px")

    if rmse_x < 0.1 and rmse_y < 0.1:
        print(f"  ✓ Perfect match! PDM reconstruction is correct.")
    elif rmse_x < 1.0 and rmse_y < 1.0:
        print(f"  ✓ Close match (< 1 px). Minor numerical differences.")
    elif rmse_x < 5.0 and rmse_y < 5.0:
        print(f"  ~ Reasonable match (< 5 px). Possible convention difference.")
    else:
        print(f"  ✗ Poor match. PDM reconstruction incorrect.")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("If all frames show RMSE < 0.1 px:")
print("  → We've perfectly replicated CalcShape3D + CalcShape2D")
print("  → CSV landmarks are indeed PDM-reconstructed")
print("  → We fully understand the C++ pipeline")
print("\nIf RMSE is larger:")
print("  → Check rotation convention (XYZ vs ZYX Euler angles)")
print("  → Check coordinate system (different origin/flip)")
print("  → Check if there's a post-processing step we're missing")
print("=" * 80)
