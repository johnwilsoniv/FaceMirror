#!/usr/bin/env python3
"""
Test how 3D rotation affects 2D Kabsch alignment

Theory:
- CSV landmarks have 3D rotation (from params_global) + 2D projection
- PDM mean_shape is in canonical 3D frame, we use 2D projection
- Kabsch on 2D projections computes a 2D rotation
- Does this 2D rotation relate predictably to the 3D rotation?

Test:
1. Take PDM mean_shape 3D
2. Apply known 3D rotation (params_global)
3. Project both to 2D
4. Run Kabsch alignment
5. See what 2D rotation we get
"""

import numpy as np
import pandas as pd
from pdm_parser import PDMParser

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7  # (204, 1)

# CRITICAL: PDM format is [x0, x1, ..., x67, y0, y1, ..., y67, z0, z1, ..., z67]
# NOT [x0, y0, z0, x1, y1, z1, ...]
# So we need to reshape carefully
n = 68
mean_shape_x = mean_shape_scaled[0:n].flatten()      # x coordinates
mean_shape_y = mean_shape_scaled[n:2*n].flatten()    # y coordinates
mean_shape_z = mean_shape_scaled[2*n:3*n].flatten()  # z coordinates
mean_shape_3d = np.stack([mean_shape_x, mean_shape_y, mean_shape_z], axis=1)  # (68, 3)

# Load CSV
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")

# Rigid indices
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

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

def project_3d_to_2d(points_3d, scale, R, tx, ty):
    """Apply 3D rotation, scale, and project to 2D (weak perspective)"""
    # Rotate in 3D
    rotated = scale * (R @ points_3d.T).T

    # Project to 2D (take x, y coordinates)
    points_2d = rotated[:, :2]

    # Add translation
    points_2d[:, 0] += tx
    points_2d[:, 1] += ty

    return points_2d

def align_shapes_with_scale(src, dst):
    """Kabsch with scale"""
    n = src.shape[0]
    src_centered = src - src.mean(axis=0)
    dst_centered = dst - dst.mean(axis=0)
    s_src = np.sqrt(np.sum(src_centered ** 2) / n)
    s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)
    src_norm = src_centered / s_src
    dst_norm = dst_centered / s_dst
    U, S, Vt = np.linalg.svd(src_norm.T @ dst_norm)
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    if d > 0:
        corr[1, 1] = 1
    else:
        corr[1, 1] = -1
    R = Vt.T @ corr @ U.T
    scale = s_dst / s_src
    return scale * R

print("=" * 80)
print("Testing 3D Rotation → 2D Kabsch Relationship")
print("=" * 80)

# Reference: PDM mean_shape projected to 2D (no rotation)
# Just the x, y coordinates (z is discarded for 2D projection)
mean_shape_2d_reference = mean_shape_3d[:, :2].copy()  # (68, 2) - x, y only

test_frames = [1, 493, 617, 863]

print("\nFrames:")
print(f"{'Frame':>6} {'p_rz(deg)':>10} {'Kabsch 2D':>12} {'Difference':>12} {'Expression':>12}")
print("-" * 80)

results = []
for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]

    # Get params_global
    scale = row['p_scale']
    rx = row['p_rx']
    ry = row['p_ry']
    rz = row['p_rz']  # Roll - the main rotation we care about
    tx = row['p_tx']
    ty = row['p_ty']

    # Build 3D rotation matrix
    R_3d = euler_to_rotation_matrix(rx, ry, rz)

    # Simulate what CalcShape2D does: rotate 3D mean_shape and project
    simulated_landmarks = project_3d_to_2d(mean_shape_3d, scale, R_3d, tx, ty)

    # Now align these simulated landmarks back to mean_shape 2D reference
    src = simulated_landmarks[RIGID_INDICES]
    dst = mean_shape_2d_reference[RIGID_INDICES]

    scale_rot = align_shapes_with_scale(src, dst)
    kabsch_2d_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi

    # Compare to p_rz
    p_rz_deg = rz * 180 / np.pi
    difference = kabsch_2d_angle - p_rz_deg

    # Check if this is an expression frame
    expression_label = "eyes closed" if frame_num == 617 else ""

    results.append({
        'frame': frame_num,
        'p_rz_deg': p_rz_deg,
        'kabsch_2d': kabsch_2d_angle,
        'difference': difference,
        'expression': expression_label
    })

    print(f"{frame_num:>6} {p_rz_deg:>10.2f} {kabsch_2d_angle:>12.2f} {difference:>12.2f} {expression_label:>12}")

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)

diffs = [r['difference'] for r in results]
mean_diff = np.mean(diffs)
std_diff = np.std(diffs)

print(f"\nDifference statistics:")
print(f"  Mean:  {mean_diff:+7.2f}°")
print(f"  Std:   {std_diff:7.2f}°")
print(f"  Range: {min(diffs):+7.2f}° to {max(diffs):+7.2f}°")

print("\nInterpretation:")
if abs(mean_diff) < 1.0 and std_diff < 1.0:
    print("  ✓ Kabsch 2D angle ≈ -p_rz")
    print("  → 2D Kabsch inverts the 3D roll rotation applied by CalcShape2D")
elif abs(mean_diff + 120) < 5.0:  # Testing if offset by PDM canonical rotation
    print(f"  ✓ Consistent offset of ~{mean_diff:.0f}°")
    print("  → Might be related to PDM canonical orientation (~120°)")
elif std_diff < 2.0:
    print(f"  ~ Consistent offset of {mean_diff:+.1f}°")
    print("  → Might need to add this correction")
else:
    print("  ✗ Relationship is not straightforward")
    print("  → 3D→2D projection + Kabsch doesn't simply invert rotation")

# Expression sensitivity check
frame_493_result = results[1]
frame_617_result = results[2]
expression_change = abs(frame_617_result['kabsch_2d'] - frame_493_result['kabsch_2d'])

print(f"\nExpression sensitivity (493 vs 617):")
print(f"  Change in Kabsch angle: {expression_change:.2f}°")
if expression_change < 2.0:
    print("  ✓ Expression-invariant (< 2°)")
else:
    print(f"  ✗ Expression-sensitive ({expression_change:.2f}° change)")

print("\n" + "=" * 80)
print("Next Step:")
print("=" * 80)
print("Compare to our actual Python alignment results:")
print("  Python on CSV landmarks: -8.79°, -4.27°, +2.17°, -2.89°")
angles_str = ', '.join([f"{r['kabsch_2d']:+.2f}°" for r in results])
print(f"  Simulated (PDM→2D→Kabsch): {angles_str}")
print("\nIf these match, we've confirmed the coordinate transform is correct.")
print("If they don't match, there's something else different between our implementation")
print("and C++ AlignFace.")
print("=" * 80)
