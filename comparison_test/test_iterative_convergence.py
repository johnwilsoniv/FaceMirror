#!/usr/bin/env python3
"""
Test PDM iterative refinement convergence.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')

print("="*80)
print("PDM ITERATIVE CONVERGENCE TEST")
print("="*80)

MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
CPP_LANDMARKS = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

# Load C++ landmarks
cpp_landmarks = np.load(CPP_LANDMARKS)

# Load PDM
from pyfacelm.clnf.pdm import PointDistributionModel

pdm_path = Path(MODEL_DIR) / "pdms" / "In-the-wild_aligned_PDM_68.txt"
pdm = PointDistributionModel(pdm_path)

# Manually run the iterations with debug output
mean_shape_3d = pdm.mean_shape.reshape(pdm.n_landmarks, 3)
mean_shape_2d = mean_shape_3d[:, :2]

# Input bounding box
min_x = np.min(cpp_landmarks[:, 0])
max_x = np.max(cpp_landmarks[:, 0])
min_y = np.min(cpp_landmarks[:, 1])
max_y = np.max(cpp_landmarks[:, 1])
input_width = max_x - min_x
input_height = max_y - min_y
input_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

# Mean shape bounding box
mean_min_x = np.min(mean_shape_2d[:, 0])
mean_max_x = np.max(mean_shape_2d[:, 0])
mean_min_y = np.min(mean_shape_2d[:, 1])
mean_max_y = np.max(mean_shape_2d[:, 1])
mean_width = mean_max_x - mean_min_x
mean_height = mean_max_y - mean_min_y
mean_center = np.array([(mean_min_x + mean_max_x) / 2.0, (mean_min_y + mean_max_y) / 2.0])

# Initial scale (average of width and height ratios)
scale = ((input_width / mean_width) + (input_height / mean_height)) / 2.0

# Initial translation
translation = input_center - scale * mean_center

# Start with zero params (mean shape)
params = np.zeros(pdm.n_modes, dtype=np.float32)

print(f"\nInitial estimate:")
print(f"  Scale: {scale:.6f}")
print(f"  Translation: ({translation[0]:.2f}, {translation[1]:.2f})")

# Regularization
regularization = 1.0 / (pdm.eigenvalues + 1e-8)
reg_matrix = np.diag(regularization)

print(f"\nRunning iterative refinement (max 100 iterations)...")
print(f"{'Iter':>4}  {'RMSE':>10}  {'ΔError':>10}  {'||Δp||':>10}")

prev_error = float('inf')

for iteration in range(100):
    # Project current params to 2D
    projected = pdm.params_to_landmarks_2d(params, scale, translation)

    # Compute error
    error = cpp_landmarks - projected  # (n_landmarks, 2)
    error_flat = error.flatten()  # (n_landmarks*2,)
    mean_error = np.sqrt(np.mean(error_flat ** 2))

    # Compute delta
    delta_error = prev_error - mean_error

    # Compute Jacobian
    jacobian = pdm._compute_jacobian_2d(params, scale, translation)

    # Gauss-Newton update
    JtJ = jacobian.T @ jacobian + reg_matrix
    Jt_error = jacobian.T @ error_flat

    try:
        delta_params = np.linalg.solve(JtJ, Jt_error)
    except np.linalg.LinAlgError:
        delta_params = np.linalg.lstsq(JtJ, Jt_error, rcond=None)[0]

    delta_norm = np.linalg.norm(delta_params)

    # Print progress
    if iteration % 10 == 0 or iteration < 10:
        print(f"{iteration:4d}  {mean_error:10.4f}  {delta_error:10.4f}  {delta_norm:10.4f}")

    # Check convergence
    if abs(delta_error) < 0.01:
        print(f"\n✓ Converged at iteration {iteration}")
        print(f"  Final RMSE: {mean_error:.4f} pixels")
        break
    prev_error = mean_error

    # Update parameters
    params += delta_params

    if iteration == 99:
        print(f"\n⚠️  Reached max iterations (100)")
        print(f"  Final RMSE: {mean_error:.4f} pixels")

print(f"\nFinal params (first 10): {params[:10]}")

# Test reconstruction
reconstructed = pdm.params_to_landmarks_2d(params, scale, translation)
final_error = np.linalg.norm(reconstructed - cpp_landmarks, axis=1)
print(f"\nFinal mean error: {np.mean(final_error):.2f} pixels")
print(f"Final max error: {np.max(final_error):.2f} pixels")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
