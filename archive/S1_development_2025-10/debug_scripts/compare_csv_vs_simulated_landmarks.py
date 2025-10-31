#!/usr/bin/env python3
"""
Compare CSV landmarks with simulated PDM reconstruction

Theory:
- CSV should contain landmarks from CalcShape2D(params_global + params_local)
- We can simulate CalcShape2D by applying 3D rotation to mean_shape
- If CSV matches simulation → we understand the transform
- If CSV differs → there's something else going on
"""

import numpy as np
import pandas as pd
from pdm_parser import PDMParser

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7  # (204, 1)

# PDM format: [x0...x67, y0...y67, z0...z67]
n = 68
mean_shape_x = mean_shape_scaled[0:n].flatten()
mean_shape_y = mean_shape_scaled[n:2*n].flatten()
mean_shape_z = mean_shape_scaled[2*n:3*n].flatten()
mean_shape_3d = np.stack([mean_shape_x, mean_shape_y, mean_shape_z], axis=1)  # (68, 3)

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

def simulate_calc_shape_2d(mean_shape_3d, scale, R, tx, ty):
    """Simulate CalcShape2D: rotate 3D, project to 2D"""
    # Rotate and scale in 3D
    rotated = scale * (R @ mean_shape_3d.T).T

    # Project to 2D (weak perspective)
    points_2d = rotated[:, :2]

    # Add translation
    points_2d[:, 0] += tx
    points_2d[:, 1] += ty

    return points_2d

print("=" * 80)
print("Comparing CSV Landmarks vs Simulated CalcShape2D")
print("=" * 80)

# Test frame 493 (stable, eyes open)
frame_num = 493
row = df[df['frame'] == frame_num].iloc[0]

# Get CSV landmarks
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
csv_x = row[x_cols].values.astype(np.float32)
csv_y = row[y_cols].values.astype(np.float32)
csv_landmarks = np.stack([csv_x, csv_y], axis=1)  # (68, 2)

# Get params_global
scale = row['p_scale']
rx = row['p_rx']
ry = row['p_ry']
rz = row['p_rz']
tx = row['p_tx']
ty = row['p_ty']

print(f"\nFrame {frame_num}:")
print(f"  params_global: scale={scale:.4f}, rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}, tx={tx:.1f}, ty={ty:.1f}")

# Simulate CalcShape2D (assuming params_local = 0, no expression deformation)
R_3d = euler_to_rotation_matrix(rx, ry, rz)
simulated_landmarks = simulate_calc_shape_2d(mean_shape_3d, scale, R_3d, tx, ty)

# Compare
print("\n" + "-" * 80)
print("Comparison (first 10 landmarks):")
print("-" * 80)
print(f"{'Landmark':>10} {'CSV X':>10} {'Sim X':>10} {'Diff X':>10} {'CSV Y':>10} {'Sim Y':>10} {'Diff Y':>10}")
print("-" * 80)

for i in range(10):
    csv_x_val = csv_landmarks[i, 0]
    csv_y_val = csv_landmarks[i, 1]
    sim_x_val = simulated_landmarks[i, 0]
    sim_y_val = simulated_landmarks[i, 1]
    diff_x = csv_x_val - sim_x_val
    diff_y = csv_y_val - sim_y_val

    print(f"{i:>10} {csv_x_val:>10.2f} {sim_x_val:>10.2f} {diff_x:>10.2f} {csv_y_val:>10.2f} {sim_y_val:>10.2f} {diff_y:>10.2f}")

# Overall statistics
diff_all = csv_landmarks - simulated_landmarks
mean_diff = np.mean(np.abs(diff_all), axis=0)
max_diff = np.max(np.abs(diff_all), axis=0)
rmse = np.sqrt(np.mean(diff_all ** 2, axis=0))

print("-" * 80)
print(f"{'Statistic':>10} {'X':>15} {'Y':>15}")
print("-" * 80)
print(f"{'Mean |Δ|':>10} {mean_diff[0]:>15.2f} {mean_diff[1]:>15.2f}")
print(f"{'Max |Δ|':>10} {max_diff[0]:>15.2f} {max_diff[1]:>15.2f}")
print(f"{'RMSE':>10} {rmse[0]:>15.2f} {rmse[1]:>15.2f}")

print("\n" + "=" * 80)
print("Interpretation:")
print("=" * 80)

if rmse[0] < 5 and rmse[1] < 5:
    print("✓ CSV landmarks ≈ simulated CalcShape2D (RMSE < 5 pixels)")
    print("→ CSV contains PDM-reconstructed landmarks as expected")
    print("→ params_local ≈ 0 (neutral expression) or small deformation")
else:
    print("✗ CSV landmarks ≠ simulated CalcShape2D (RMSE > 5 pixels)")
    print(f"  RMSE: ({rmse[0]:.1f}, {rmse[1]:.1f}) pixels")
    print("\nPossible reasons:")
    print("  1. params_local != 0 (expression deformation not accounted for)")
    print("  2. Different rotation convention (check Euler angle order)")
    print("  3. Different coordinate system (origin, flip, etc.)")
    print("  4. CSV landmarks are NOT from CalcShape2D")

print("=" * 80)
