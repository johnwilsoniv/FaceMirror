#!/usr/bin/env python3
"""
Test hypothesis: The depth prior from mean_shape needs to be in the
same coordinate system as the back-projected landmarks_3d_xy.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')

print("="*80)
print("PDM DEPTH PRIOR HYPOTHESIS TEST")
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
print("CURRENT IMPLEMENTATION (BUGGY)")
print(f"{'='*80}")

# Current implementation
params, scale, translation = pdm.landmarks_to_params_2d(cpp_landmarks)
print(f"\nScale: {scale:.6f}")
print(f"Translation: ({translation[0]:.2f}, {translation[1]:.2f})")
print(f"Params (first 5): {params[:5]}")

# Reconstruct
params_clamped = pdm.clamp_params(params, n_std=3.0)
reconstructed_buggy = pdm.params_to_landmarks_2d(params_clamped, scale, translation)

error_buggy = np.linalg.norm(reconstructed_buggy - cpp_landmarks, axis=1)
print(f"\nBuggy version error: {np.mean(error_buggy):.2f} pixels")

print(f"\n{'='*80}")
print("MANUAL IMPLEMENTATION WITH ANALYSIS")
print(f"{'='*80}")

# Manually reproduce the bug to understand it
mean_shape_3d = pdm.mean_shape.reshape(pdm.n_landmarks, 3)
mean_shape_2d = mean_shape_3d[:, :2]
depth_prior = mean_shape_3d[:, 2]

print(f"\nMean shape 2D (first 3 landmarks):")
for i in range(3):
    print(f"  Landmark {i}: ({mean_shape_2d[i, 0]:.4f}, {mean_shape_2d[i, 1]:.4f})")

print(f"\nDepth prior (z-coordinates, first 3):")
for i in range(3):
    print(f"  Landmark {i}: z={depth_prior[i]:.4f}")

# Center both shapes
landmarks_centered = cpp_landmarks - np.mean(cpp_landmarks, axis=0)
mean_centered = mean_shape_2d - np.mean(mean_shape_2d, axis=0)

# Compute scale
scale_manual = np.linalg.norm(landmarks_centered) / (np.linalg.norm(mean_centered) + 1e-8)
translation_manual = np.mean(cpp_landmarks, axis=0) - scale_manual * np.mean(mean_shape_2d, axis=0)

print(f"\nManual scale: {scale_manual:.6f}")
print(f"Manual translation: ({translation_manual[0]:.2f}, {translation_manual[1]:.2f})")

# Back-project to 3D
landmarks_3d_xy = (cpp_landmarks - translation_manual) / scale_manual

print(f"\nBack-projected xy (first 3):")
for i in range(3):
    print(f"  Landmark {i}: ({landmarks_3d_xy[i, 0]:.4f}, {landmarks_3d_xy[i, 1]:.4f})")
    print(f"    Mean shape xy: ({mean_shape_2d[i, 0]:.4f}, {mean_shape_2d[i, 1]:.4f})")
    print(f"    Difference: ({landmarks_3d_xy[i, 0] - mean_shape_2d[i, 0]:.4f}, {landmarks_3d_xy[i, 1] - mean_shape_2d[i, 1]:.4f})")

# Combine with depth prior (this is the bug!)
shape_3d_buggy = np.column_stack([landmarks_3d_xy, depth_prior])

print(f"\n3D shape (first 3):")
for i in range(3):
    print(f"  Landmark {i}: ({shape_3d_buggy[i, 0]:.4f}, {shape_3d_buggy[i, 1]:.4f}, {shape_3d_buggy[i, 2]:.4f})")
    print(f"    Mean shape 3D: ({mean_shape_3d[i, 0]:.4f}, {mean_shape_3d[i, 1]:.4f}, {mean_shape_3d[i, 2]:.4f})")

# Project to parameter space
shape_flat_buggy = shape_3d_buggy.reshape(-1)
params_manual = pdm.eigenvectors.T @ (shape_flat_buggy - pdm.mean_shape)

print(f"\nManual params (first 5): {params_manual[:5]}")
print(f"Match pdm.landmarks_to_params_2d? {np.allclose(params_manual, params)}")

# What should the params be if we had perfect reconstruction?
print(f"\n{'='*80}")
print("WHAT IF WE USE ZERO PARAMS?")
print(f"{'='*80}")

# Zero params = mean shape
zero_params = np.zeros_like(params)
mean_landmarks = pdm.params_to_landmarks_2d(zero_params, scale_manual, translation_manual)

print(f"\nMean shape projected (first 3):")
for i in range(3):
    print(f"  Landmark {i}: ({mean_landmarks[i, 0]:.2f}, {mean_landmarks[i, 1]:.2f})")
    print(f"    Original: ({cpp_landmarks[i, 0]:.2f}, {cpp_landmarks[i, 1]:.2f})")
    print(f"    Error: {np.linalg.norm(mean_landmarks[i] - cpp_landmarks[i]):.2f}px")

error_mean = np.linalg.norm(mean_landmarks - cpp_landmarks, axis=1)
print(f"\nMean shape error: {np.mean(error_mean):.2f} pixels")

print(f"\n{'='*80}")
print("HYPOTHESIS")
print(f"{'='*80}")

print("""
The bug is likely that we're mixing coordinate systems:
1. landmarks_3d_xy are in "unscaled mean-shape space"
2. depth_prior is in "mean-shape space"

These should match! If we back-project xy, we should also use a properly
scaled/transformed depth prior.

But wait... the z-coordinates in the mean shape are already in mean-shape space.
So using them directly should be correct!

Unless... the scale estimation is wrong? Or the alignment is wrong?

Let me check what the C++ code does for this conversion.
""")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
