#!/usr/bin/env python3
"""
Test bounding-box-based scale/translation estimation (like C++ OpenFace).
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')

print("="*80)
print("BOUNDING BOX SCALE ESTIMATION TEST")
print("="*80)

MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
CPP_LANDMARKS = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

# Load C++ landmarks
cpp_landmarks = np.load(CPP_LANDMARKS)
print(f"\nOriginal C++ landmarks (first 3):")
for i in range(3):
    print(f"  Landmark {i}: ({cpp_landmarks[i, 0]:.2f}, {cpp_landmarks[i, 1]:.2f})")

# Load PDM
from pyfacelm.clnf.pdm import PointDistributionModel

pdm_path = Path(MODEL_DIR) / "pdms" / "In-the-wild_aligned_PDM_68.txt"
pdm = PointDistributionModel(pdm_path)

print(f"\n{'='*80}")
print("OLD METHOD (NORM-BASED)")
print(f"{'='*80}")

# Old method
mean_shape_3d = pdm.mean_shape.reshape(pdm.n_landmarks, 3)
mean_shape_2d = mean_shape_3d[:, :2]

landmarks_centered = cpp_landmarks - np.mean(cpp_landmarks, axis=0)
mean_centered = mean_shape_2d - np.mean(mean_shape_2d, axis=0)

scale_old = np.linalg.norm(landmarks_centered) / (np.linalg.norm(mean_centered) + 1e-8)
translation_old = np.mean(cpp_landmarks, axis=0) - scale_old * np.mean(mean_shape_2d, axis=0)

print(f"\nOld scale: {scale_old:.6f}")
print(f"Old translation: ({translation_old[0]:.2f}, {translation_old[1]:.2f})")

# Test reconstruction with zero params
zero_params = np.zeros(pdm.n_modes)
reconstructed_old = pdm.params_to_landmarks_2d(zero_params, scale_old, translation_old)
error_old = np.linalg.norm(reconstructed_old - cpp_landmarks, axis=1)
print(f"\nMean shape error (old method): {np.mean(error_old):.2f} pixels")

print(f"\n{'='*80}")
print("NEW METHOD (BOUNDING-BOX-BASED, like C++)")
print(f"{'='*80}")

# New method: bounding box based
# 1. Get bounding box of input landmarks
min_x = np.min(cpp_landmarks[:, 0])
max_x = np.max(cpp_landmarks[:, 0])
min_y = np.min(cpp_landmarks[:, 1])
max_y = np.max(cpp_landmarks[:, 1])

input_width = max_x - min_x
input_height = max_y - min_y
input_center_x = (min_x + max_x) / 2.0
input_center_y = (min_y + max_y) / 2.0

print(f"\nInput bounding box:")
print(f"  x: [{min_x:.2f}, {max_x:.2f}] width: {input_width:.2f}")
print(f"  y: [{min_y:.2f}, {max_y:.2f}] height: {input_height:.2f}")
print(f"  center: ({input_center_x:.2f}, {input_center_y:.2f})")

# 2. Get bounding box of mean shape (zero params)
mean_min_x = np.min(mean_shape_2d[:, 0])
mean_max_x = np.max(mean_shape_2d[:, 0])
mean_min_y = np.min(mean_shape_2d[:, 1])
mean_max_y = np.max(mean_shape_2d[:, 1])

mean_width = mean_max_x - mean_min_x
mean_height = mean_max_y - mean_min_y
mean_center_x = (mean_min_x + mean_max_x) / 2.0
mean_center_y = (mean_min_y + mean_max_y) / 2.0

print(f"\nMean shape bounding box:")
print(f"  x: [{mean_min_x:.2f}, {mean_max_x:.2f}] width: {mean_width:.2f}")
print(f"  y: [{mean_min_y:.2f}, {mean_max_y:.2f}] height: {mean_height:.2f}")
print(f"  center: ({mean_center_x:.2f}, {mean_center_y:.2f})")

# 3. Compute scale as average of width and height ratios
scale_new = ((input_width / mean_width) + (input_height / mean_height)) / 2.0

# 4. Translation is input center - scale * mean center
translation_new = np.array([
    input_center_x - scale_new * mean_center_x,
    input_center_y - scale_new * mean_center_y
])

print(f"\nNew scale: {scale_new:.6f}")
print(f"New translation: ({translation_new[0]:.2f}, {translation_new[1]:.2f})")

# Test reconstruction with zero params
reconstructed_new = pdm.params_to_landmarks_2d(zero_params, scale_new, translation_new)
error_new = np.linalg.norm(reconstructed_new - cpp_landmarks, axis=1)
print(f"\nMean shape error (new method): {np.mean(error_new):.2f} pixels")

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

print(f"\n                    Old Method    New Method")
print(f"  Scale:            {scale_old:10.6f}  {scale_new:10.6f}")
print(f"  Translation X:    {translation_old[0]:10.2f}  {translation_new[0]:10.2f}")
print(f"  Translation Y:    {translation_old[1]:10.2f}  {translation_new[1]:10.2f}")
print(f"  Mean shape error: {np.mean(error_old):10.2f}  {np.mean(error_new):10.2f}")

improvement = np.mean(error_old) - np.mean(error_new)
print(f"\nImprovement: {improvement:.2f} pixels")

if np.mean(error_new) < 50:
    print(f"\n✓ New method is MUCH better! Mean shape error < 50px")
    print(f"  This should give a good starting point for optimization.")
elif np.mean(error_new) < np.mean(error_old):
    print(f"\n⚠️  New method is better but still not great.")
    print(f"   May need iterative refinement.")
else:
    print(f"\n✗ New method is worse! Need to debug further.")

print(f"\n{'='*80}")
print("VISUALIZATION")
print(f"{'='*80}")

print(f"\nFirst 5 landmarks comparison:")
print(f"  Idx   Original          Old Method         New Method         New Error")
for i in range(5):
    orig = cpp_landmarks[i]
    old = reconstructed_old[i]
    new = reconstructed_new[i]
    err = error_new[i]
    print(f"  {i:2d}   ({orig[0]:7.1f}, {orig[1]:7.1f})  ({old[0]:7.1f}, {old[1]:7.1f})  ({new[0]:7.1f}, {new[1]:7.1f})  {err:6.2f}px")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
