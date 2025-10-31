#!/usr/bin/env python3
"""
Direct CalcParams Output Validation
Compare Python CalcParams outputs vs C++ baseline from CSV
"""

import cv2
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from pdm_parser import PDMParser
from calc_params import CalcParams

VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PDM_PATH = "In-the-wild_aligned_PDM_68.txt"

print("="*80)
print("CalcParams Direct Output Validation")
print("="*80)

# Load components
print("\n1. Loading PDM and CalcParams...")
pdm = PDMParser(PDM_PATH)
calc_params = CalcParams(pdm)

# Load CSV baseline
df = pd.read_csv(CSV_PATH)
print(f"  Loaded {len(df)} frames from CSV")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)

# Test on 50 frames
TEST_FRAMES = 50
print(f"\n2. Testing CalcParams on {TEST_FRAMES} frames...")

# Storage for parameters
cpp_pose = []  # 6 params: [scale, rx, ry, rz, tx, ty]
py_pose = []

cpp_shape = []  # 34 params
py_shape = []

for frame_idx in range(TEST_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    csv_row = df.iloc[frame_idx]

    # Get landmarks from CSV
    landmarks_2d = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        landmarks_2d[i, 0] = csv_row[f'x_{i}']
        landmarks_2d[i, 1] = csv_row[f'y_{i}']

    # Get C++ CalcParams output from CSV
    # CSV columns: p_scale, p_rx, p_ry, p_rz, p_tx, p_ty (CalcParams pose)
    # CSV columns: p_0, p_1, ..., p_33 (34 shape parameters)
    cpp_pose_params = np.array([
        csv_row['p_scale'],  # scale
        csv_row['p_rx'],      # rx
        csv_row['p_ry'],      # ry
        csv_row['p_rz'],      # rz
        csv_row['p_tx'],      # tx
        csv_row['p_ty']       # ty
    ], dtype=np.float32)

    cpp_shape_params = np.array([csv_row[f'p_{i}'] for i in range(34)], dtype=np.float32)

    # Run Python CalcParams
    try:
        py_pose_params, py_shape_params = calc_params.calc_params(landmarks_2d)

        cpp_pose.append(cpp_pose_params)
        py_pose.append(py_pose_params)
        cpp_shape.append(cpp_shape_params)
        py_shape.append(py_shape_params)

    except Exception as e:
        print(f"  Frame {frame_idx}: CalcParams failed - {e}")
        continue

    if (frame_idx + 1) % 10 == 0:
        print(f"  Processed {frame_idx + 1}/{TEST_FRAMES}...")

cap.release()

# Convert to arrays
cpp_pose = np.array(cpp_pose)
py_pose = np.array(py_pose)
cpp_shape = np.array(cpp_shape)
py_shape = np.array(py_shape)

print(f"\n3. Collected {len(cpp_pose)} valid frames")

# Analyze pose parameters
print("\n" + "="*80)
print("POSE PARAMETER COMPARISON")
print("="*80)

pose_names = ['scale', 'rx', 'ry', 'rz', 'tx', 'ty']
print(f"\n{'Param':<10} {'C++ Mean':<12} {'Py Mean':<12} {'RMSE':<12} {'r':<8}")
print("-"*60)

for i, name in enumerate(pose_names):
    cpp_vals = cpp_pose[:, i]
    py_vals = py_pose[:, i]

    rmse = np.sqrt(np.mean((cpp_vals - py_vals)**2))
    r, p = pearsonr(cpp_vals, py_vals)

    print(f"{name:<10} {cpp_vals.mean():>11.4f} {py_vals.mean():>11.4f} {rmse:>11.6f} {r:>7.4f}")

# Analyze shape parameters
print("\n" + "="*80)
print("SHAPE PARAMETER COMPARISON (first 10 of 34)")
print("="*80)

print(f"\n{'Param':<10} {'C++ Mean':<12} {'Py Mean':<12} {'RMSE':<12} {'r':<8}")
print("-"*60)

for i in range(min(10, 34)):
    cpp_vals = cpp_shape[:, i]
    py_vals = py_shape[:, i]

    rmse = np.sqrt(np.mean((cpp_vals - py_vals)**2))

    if cpp_vals.std() > 1e-6 and py_vals.std() > 1e-6:
        r, p = pearsonr(cpp_vals, py_vals)
    else:
        r = 0.0

    print(f"p_{i:<8} {cpp_vals.mean():>11.4f} {py_vals.mean():>11.4f} {rmse:>11.6f} {r:>7.4f}")

# Overall shape parameter correlation
print(f"\nRemaining shape parameters (p_10 to p_33):")
for i in range(10, 34):
    cpp_vals = cpp_shape[:, i]
    py_vals = py_shape[:, i]
    rmse = np.sqrt(np.mean((cpp_vals - py_vals)**2))
    if cpp_vals.std() > 1e-6 and py_vals.std() > 1e-6:
        r, p = pearsonr(cpp_vals, py_vals)
    else:
        r = 0.0
    print(f"  p_{i}: RMSE={rmse:>8.4f}, r={r:>6.3f}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Pose parameter accuracy
pose_rmse = []
pose_corr = []
for i in range(6):
    cpp_vals = cpp_pose[:, i]
    py_vals = py_pose[:, i]
    pose_rmse.append(np.sqrt(np.mean((cpp_vals - py_vals)**2)))
    r, p = pearsonr(cpp_vals, py_vals)
    pose_corr.append(r)

print(f"\nPose Parameters:")
print(f"  Mean RMSE: {np.mean(pose_rmse):.6f}")
print(f"  Mean correlation: {np.mean(pose_corr):.4f}")

# Shape parameter accuracy
shape_rmse = []
shape_corr = []
for i in range(34):
    cpp_vals = cpp_shape[:, i]
    py_vals = py_shape[:, i]
    shape_rmse.append(np.sqrt(np.mean((cpp_vals - py_vals)**2)))
    if cpp_vals.std() > 1e-6 and py_vals.std() > 1e-6:
        r, p = pearsonr(cpp_vals, py_vals)
        shape_corr.append(r)

print(f"\nShape Parameters:")
print(f"  Mean RMSE: {np.mean(shape_rmse):.6f}")
print(f"  Mean correlation: {np.mean(shape_corr):.4f}")

# Decision criteria
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

pose_match = np.mean(pose_corr) > 0.99 and np.mean(pose_rmse) < 0.01
shape_match = np.mean(shape_corr) > 0.90

if pose_match and shape_match:
    print("\n✅ SUCCESS: Python CalcParams matches C++ baseline!")
    print("   The problem is NOT in CalcParams output.")
    print("   Issue must be in downstream processing (alignment, HOG, etc.)")
elif pose_match and not shape_match:
    print("\n⚠️  PARTIAL: Pose params match, but shape params diverge")
    print("   → Shape parameter optimization needs tuning")
elif not pose_match and shape_match:
    print("\n⚠️  PARTIAL: Shape params match, but pose params diverge")
    print("   → Pose parameter optimization needs tuning")
else:
    print("\n❌ MISMATCH: Python CalcParams produces different outputs than C++")
    print("   → Need to debug CalcParams optimization algorithm")
    print(f"\n   Pose correlation: {np.mean(pose_corr):.4f} (target > 0.99)")
    print(f"   Shape correlation: {np.mean(shape_corr):.4f} (target > 0.90)")

print("\n" + "="*80)
