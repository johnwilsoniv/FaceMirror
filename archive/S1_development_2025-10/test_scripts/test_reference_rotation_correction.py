#!/usr/bin/env python3
"""
Test applying various rotations to the reference shape to find what makes Python match C++

The external reviewer suggests C++ is upright the reference shape before alignment.
Let's test rotations from -180° to +180° to find which one produces upright faces.
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from pdm_parser import PDMParser

# Load data
df = pd.read_csv("of22_validation/IMG_0942_left_mirrored.csv")
video_path = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
cpp_dir = Path("pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

# Load PDM
pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
mean_shape_scaled = pdm.mean_shape * 0.7
mean_shape_2d = mean_shape_scaled[:136]
reference_shape_original = mean_shape_2d.reshape(68, 2)

# Rigid indices
RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

def rotate_points(points, angle_deg):
    """Rotate points around their center"""
    angle_rad = angle_deg * np.pi / 180
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    center = points.mean(axis=0)
    centered = points - center
    rotated = (R @ centered.T).T
    return rotated + center

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

print("=" * 70)
print("Testing Reference Shape Rotations to Find C++ Match")
print("=" * 70)

# Test frame 493 (eyes open, stable)
row = df[df['frame'] == 493].iloc[0]
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks_68 = np.stack([x, y], axis=1)

src = landmarks_68[RIGID_INDICES]

print("\n[1/2] Testing rotations from -180° to +180° in 10° increments...")
print("Looking for rotation that gives output angle closest to 0°")
print()

best_angle = None
best_correction = None
min_output_angle = 999

test_corrections = range(-180, 181, 10)
for correction_deg in test_corrections:
    # Apply rotation to reference
    reference_rotated = rotate_points(reference_shape_original, correction_deg)
    dst = reference_rotated[RIGID_INDICES]

    # Compute alignment
    scale_rot = align_shapes_with_scale(src, dst)
    output_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi

    if abs(output_angle) < abs(min_output_angle):
        min_output_angle = output_angle
        best_correction = correction_deg
        best_angle = output_angle

    if correction_deg % 30 == 0:  # Print every 30 degrees
        print(f"  Reference rotation {correction_deg:+4d}° → output angle {output_angle:+7.2f}°")

print(f"\n  ★ BEST: Reference rotation {best_correction:+4d}° → output angle {best_angle:+7.2f}°")

print("\n[2/2] Testing best rotation on all frames...")
print("-" * 70)

reference_best = rotate_points(reference_shape_original, best_correction)

cap = cv2.VideoCapture(video_path)
test_frames = [1, 493, 617, 863]

output_angles = []
for frame_num in test_frames:
    row = df[df['frame'] == frame_num].iloc[0]
    x = row[x_cols].values.astype(np.float32)
    y = row[y_cols].values.astype(np.float32)
    landmarks_68 = np.stack([x, y], axis=1)

    src = landmarks_68[RIGID_INDICES]
    dst = reference_best[RIGID_INDICES]

    scale_rot = align_shapes_with_scale(src, dst)
    output_angle = np.arctan2(scale_rot[1,0], scale_rot[0,0]) * 180 / np.pi
    output_angles.append(output_angle)

    print(f"Frame {frame_num:4d}: output angle = {output_angle:+7.2f}°")

cap.release()

# Check stability and expression sensitivity
angles_array = np.array(output_angles)
mean_angle = np.mean(angles_array)
std_angle = np.std(angles_array)

# Expression sensitivity (frame 617 vs others)
frame_617_angle = output_angles[2]  # Frame 617 is index 2
frame_493_angle = output_angles[1]  # Frame 493 is index 1
expression_change = abs(frame_617_angle - frame_493_angle)

print("\n" + "=" * 70)
print("Results:")
print("=" * 70)
print(f"Best reference correction: {best_correction:+d}°")
print(f"Output rotation:")
print(f"  Mean:  {mean_angle:+7.2f}°")
print(f"  Std:   {std_angle:7.2f}°")
print(f"  Range: {angles_array.min():+7.2f}° to {angles_array.max():+7.2f}°")
print(f"Expression sensitivity (493 vs 617): {expression_change:.2f}°")

print("\n" + "=" * 70)
print("Comparison to Original:")
print("=" * 70)
print("Original Python (no correction):")
print("  Mean: -3.45°, Std: 4.51°, Expression: 6.44°")
print(f"\nWith {best_correction:+d}° reference correction:")
print(f"  Mean: {mean_angle:+.2f}°, Std: {std_angle:.2f}°, Expression: {expression_change:.2f}°")

if abs(mean_angle) < 2.0 and std_angle < 3.0:
    print("\n  ✓✓✓ SUCCESS! This correction produces near-upright, stable faces!")
    print(f"  ✓ Apply {best_correction:+d}° rotation to reference shape in Python")
elif abs(mean_angle) < 5.0:
    print(f"\n  ~ IMPROVEMENT! Closer to upright than before")
    print(f"  ~ May need additional fine-tuning")
else:
    print(f"\n  ✗ No significant improvement")
    print(f"  → Need different approach")

print("=" * 70)
