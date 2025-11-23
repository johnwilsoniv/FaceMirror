#!/usr/bin/env python3
"""
Debug fit_to_landmarks to find X direction inversion.

Compare Python fit_to_landmarks with C++ CalcParams iteration by iteration.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.core.pdm import PDM
from pyclnf.clnf import CLNF

# Test with the eye landmarks after eye model refinement
# From the debug progress doc:
# - C++ Eye_8 final: (390.30, 831.46)
# - C++ landmark 36 final after CalcParams: (391.5, 830.2)

def debug_fit_to_landmarks():
    """Debug fit_to_landmarks vs C++ CalcParams."""

    print("="*70)
    print("DEBUG: fit_to_landmarks vs C++ CalcParams")
    print("="*70)

    # Load PDM
    pdm = PDM("pyclnf/models/exported_pdm")
    print(f"PDM: {pdm.n_points} landmarks, {pdm.n_modes} modes")

    # Use test landmarks from frame 30 (from previous debug)
    # These are the landmarks AFTER eye model refinement but BEFORE CalcParams
    # We need to get these from actual test data

    # For now, let's test with a simple case: create landmarks from known params
    # and see if fit_to_landmarks recovers the same params

    print("\n" + "="*70)
    print("TEST 1: Round-trip test (params -> landmarks -> fit_to_landmarks)")
    print("="*70)

    # Create test parameters
    test_params = np.zeros(pdm.n_params, dtype=np.float32)
    test_params[0] = 3.3  # scale
    test_params[1] = -0.1  # pitch
    test_params[2] = 0.15  # yaw
    test_params[3] = -0.08  # roll
    test_params[4] = 425.0  # tx
    test_params[5] = 820.0  # ty
    # Keep shape params at 0

    print(f"\nOriginal params:")
    print(f"  scale={test_params[0]:.6f}")
    print(f"  rot=({test_params[1]:.6f}, {test_params[2]:.6f}, {test_params[3]:.6f})")
    print(f"  tx={test_params[4]:.6f}, ty={test_params[5]:.6f}")

    # Generate landmarks
    landmarks = pdm.params_to_landmarks_2d(test_params)
    print(f"\nGenerated landmarks shape: {landmarks.shape}")
    print(f"  Landmark 36: ({landmarks[36, 0]:.4f}, {landmarks[36, 1]:.4f})")

    # Fit back using fit_to_landmarks
    rotation_init = test_params[1:4]  # Pass known rotation
    recovered_params = pdm.fit_to_landmarks(landmarks, rotation=rotation_init)

    print(f"\nRecovered params:")
    print(f"  scale={recovered_params[0]:.6f}")
    print(f"  rot=({recovered_params[1]:.6f}, {recovered_params[2]:.6f}, {recovered_params[3]:.6f})")
    print(f"  tx={recovered_params[4]:.6f}, ty={recovered_params[5]:.6f}")

    # Compare
    param_diff = recovered_params[:6] - test_params[:6]
    print(f"\nParameter differences:")
    print(f"  scale: {param_diff[0]:.6f}")
    print(f"  rot: ({param_diff[1]:.6f}, {param_diff[2]:.6f}, {param_diff[3]:.6f})")
    print(f"  tx: {param_diff[4]:.6f}, ty: {param_diff[5]:.6f}")

    # Generate landmarks from recovered params
    recovered_landmarks = pdm.params_to_landmarks_2d(recovered_params)
    lm_error = np.sqrt(np.sum((recovered_landmarks - landmarks)**2, axis=1))
    print(f"\nLandmark recovery error:")
    print(f"  Mean: {lm_error.mean():.6f} px")
    print(f"  Max: {lm_error.max():.6f} px")
    print(f"  Landmark 36 error: {lm_error[36]:.6f} px")

    print("\n" + "="*70)
    print("TEST 2: Perturb landmarks and check fit direction")
    print("="*70)

    # Perturb landmark 36 in +X direction (rightward)
    perturbed_landmarks = landmarks.copy()
    perturbed_landmarks[36, 0] += 2.0  # Move 2px right

    print(f"\nPerturbed landmark 36: ({perturbed_landmarks[36, 0]:.4f}, {perturbed_landmarks[36, 1]:.4f})")
    print(f"  Moved +2.0 px in X (rightward)")

    # Fit to perturbed landmarks
    fitted_params = pdm.fit_to_landmarks(perturbed_landmarks, rotation=rotation_init)
    fitted_landmarks = pdm.params_to_landmarks_2d(fitted_params)

    print(f"\nFitted landmark 36: ({fitted_landmarks[36, 0]:.4f}, {fitted_landmarks[36, 1]:.4f})")

    # Check direction of movement
    original_x = landmarks[36, 0]
    fitted_x = fitted_landmarks[36, 0]
    movement = fitted_x - original_x

    print(f"\nMovement analysis:")
    print(f"  Original X: {original_x:.4f}")
    print(f"  Target X:   {perturbed_landmarks[36, 0]:.4f} (+2.0)")
    print(f"  Fitted X:   {fitted_x:.4f}")
    print(f"  Actual movement: {movement:+.4f} px")

    if movement > 0:
        print(f"  ✓ Correct direction (rightward)")
    else:
        print(f"  ✗ WRONG direction (leftward) - THIS IS THE BUG!")

    print("\n" + "="*70)
    print("TEST 3: Debug fit_to_landmarks iteration by iteration")
    print("="*70)

    # Add detailed logging to fit_to_landmarks
    debug_fit_to_landmarks_detailed(pdm, perturbed_landmarks, rotation_init)


def debug_fit_to_landmarks_detailed(pdm, landmarks, rotation_init):
    """Run fit_to_landmarks with detailed iteration logging."""

    n = pdm.n_points
    m = pdm.n_modes

    landmarks = landmarks.reshape(-1, 2)

    # Build visibility mask (all visible)
    visi_count = n

    # Extract landmark locations in blocked format
    landmark_locs_vis = np.zeros((visi_count * 2, 1), dtype=np.float32)
    for i in range(n):
        landmark_locs_vis[i] = landmarks[i, 0]
        landmark_locs_vis[i + visi_count] = landmarks[i, 1]

    # Compute initial parameters from bounding box
    min_x, max_x = landmarks[:, 0].min(), landmarks[:, 0].max()
    min_y, max_y = landmarks[:, 1].min(), landmarks[:, 1].max()

    width = abs(max_x - min_x)
    height = abs(max_y - min_y)

    # Get model bbox
    neutral_params = np.zeros(pdm.n_params, dtype=np.float32)
    neutral_params[0] = 1.0
    neutral_lm = pdm.params_to_landmarks_2d(neutral_params)
    model_width = neutral_lm[:, 0].max() - neutral_lm[:, 0].min()
    model_height = neutral_lm[:, 1].max() - neutral_lm[:, 1].min()

    scaling = ((width / model_width) + (height / model_height)) / 2.0
    translation = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float32)

    # Initialize
    loc_params = np.zeros(m, dtype=np.float32)
    glob_params = np.array([scaling, rotation_init[0], rotation_init[1],
                            rotation_init[2], translation[0], translation[1]], dtype=np.float32)

    print(f"\nInitial params:")
    print(f"  scale={glob_params[0]:.6f}")
    print(f"  rot=({glob_params[1]:.6f}, {glob_params[2]:.6f}, {glob_params[3]:.6f})")
    print(f"  tx={glob_params[4]:.6f}, ty={glob_params[5]:.6f}")

    # Build regularization
    reg_factor = 1.0
    regularisations = np.zeros(6 + m, dtype=np.float32)
    regularisations[6:] = reg_factor / pdm.eigen_values.flatten()
    reg_diag = np.diag(regularisations)

    W = np.eye(visi_count * 2, dtype=np.float32)

    # Subsample M and V (all visible)
    M = pdm.mean_shape.flatten().reshape(-1, 1)
    V = pdm.princ_comp

    for iteration in range(10):  # Just first 10 iterations
        # Compute current shape
        shape_3d = M + V @ loc_params.reshape(-1, 1)
        shape_3d = shape_3d.reshape(3, n).T

        R = pdm._euler_to_rotation_matrix(glob_params[1:4])
        R_2d = R[:2, :]

        curr_shape_2d = glob_params[0] * (shape_3d @ R_2d.T)
        curr_shape_2d[:, 0] += glob_params[4]
        curr_shape_2d[:, 1] += glob_params[5]

        # Flatten for comparison (blocked format)
        curr_shape = np.zeros((n * 2, 1), dtype=np.float32)
        curr_shape[:n, 0] = curr_shape_2d[:, 0]
        curr_shape[n:, 0] = curr_shape_2d[:, 1]

        # Compute error
        error_resid = landmark_locs_vis - curr_shape
        error = np.linalg.norm(error_resid)

        # Log landmark 36 position and error
        lm36_curr = curr_shape_2d[36]
        lm36_target = landmarks[36]
        lm36_error = error_resid[36, 0], error_resid[36 + n, 0]

        if iteration < 5:
            print(f"\nIteration {iteration}:")
            print(f"  Error norm: {error:.6f}")
            print(f"  Landmark 36:")
            print(f"    Current:  ({lm36_curr[0]:.4f}, {lm36_curr[1]:.4f})")
            print(f"    Target:   ({lm36_target[0]:.4f}, {lm36_target[1]:.4f})")
            print(f"    Residual: ({lm36_error[0]:.4f}, {lm36_error[1]:.4f})")

        # Compute Jacobian
        J = pdm._compute_jacobian_subsampled(loc_params, glob_params, M, V, n)

        J_w_t = J.T @ W
        J_w_t_m = J_w_t @ error_resid
        J_w_t_m[6:] = J_w_t_m[6:] - reg_diag[6:, 6:] @ loc_params.reshape(-1, 1)

        Hessian = J_w_t @ J + reg_diag

        try:
            param_update = np.linalg.solve(Hessian, J_w_t_m).flatten()
        except:
            break

        if iteration < 5:
            print(f"  Param update (before damping):")
            print(f"    scale: {param_update[0]:.6f}")
            print(f"    rot: ({param_update[1]:.6f}, {param_update[2]:.6f}, {param_update[3]:.6f})")
            print(f"    tx: {param_update[4]:.6f}, ty: {param_update[5]:.6f}")

        # Apply damping
        param_update *= 0.75

        # Update parameters
        full_params = np.concatenate([glob_params, loc_params])
        full_params = pdm.update_params(full_params, param_update)

        glob_params = full_params[:6]
        loc_params = full_params[6:]

        if iteration < 5:
            print(f"  Updated params:")
            print(f"    scale={glob_params[0]:.6f}")
            print(f"    tx={glob_params[4]:.6f}, ty={glob_params[5]:.6f}")

    # Final result
    final_landmarks = pdm.params_to_landmarks_2d(np.concatenate([glob_params, loc_params]))
    print(f"\nFinal landmark 36: ({final_landmarks[36, 0]:.4f}, {final_landmarks[36, 1]:.4f})")
    print(f"Target landmark 36: ({landmarks[36, 0]:.4f}, {landmarks[36, 1]:.4f})")


if __name__ == "__main__":
    debug_fit_to_landmarks()
