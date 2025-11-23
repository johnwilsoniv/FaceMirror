#!/usr/bin/env python3
"""
Direct comparison: Python fit_to_landmarks vs C++ CalcParams

This test:
1. Takes landmarks from Python eye refinement output
2. Runs Python fit_to_landmarks
3. Compares iteration-by-iteration convergence

The goal is to understand why Python fit_to_landmarks doesn't produce
the same rightward movement as C++ CalcParams after eye refinement.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))

from pyclnf.core.pdm import PDM


def test_fit_comparison():
    """Compare Python fit_to_landmarks with detailed iteration logging."""

    print("="*70)
    print("FIT_TO_LANDMARKS ITERATION COMPARISON")
    print("="*70)

    pdm = PDM("pyclnf/models/exported_pdm")
    print(f"PDM: {pdm.n_points} landmarks, {pdm.n_modes} modes")

    # Use actual eye-refined landmarks from debug output
    # These are the post-refinement landmarks that need to be fitted
    eye_refined = np.array([
        [389.9033, 831.9913],  # 36 - moved left by eye model
        [405.3546, 814.0696],  # 37
        [437.0951, 809.2116],  # 38
        [458.5762, 822.8660],  # 39
        [438.3975, 833.8026],  # 40
        [410.3564, 838.8933],  # 41
    ])

    # Get full 68-point landmarks by starting from typical main model output
    # and updating just the eye landmarks
    init_params = np.zeros(pdm.n_params, dtype=np.float32)
    init_params[0] = 3.37  # scale
    init_params[1] = -0.117  # pitch
    init_params[2] = 0.175   # yaw
    init_params[3] = -0.1    # roll
    init_params[4] = 425.0   # tx
    init_params[5] = 820.0   # ty

    main_landmarks = pdm.params_to_landmarks_2d(init_params)

    # Update eye landmarks with refined values
    target_landmarks = main_landmarks.copy()
    for i, eye_lm in enumerate(eye_refined):
        target_landmarks[36 + i] = eye_lm

    print(f"\nInput landmark 36: ({target_landmarks[36, 0]:.4f}, {target_landmarks[36, 1]:.4f})")
    print(f"Main model had 36: ({main_landmarks[36, 0]:.4f}, {main_landmarks[36, 1]:.4f})")
    print(f"Eye refinement delta: ({target_landmarks[36, 0] - main_landmarks[36, 0]:.4f}, "
          f"{target_landmarks[36, 1] - main_landmarks[36, 1]:.4f})")

    # Run fit_to_landmarks with detailed iteration logging
    print("\n" + "="*70)
    print("Running fit_to_landmarks with iteration logging")
    print("="*70)

    fitted_params = fit_to_landmarks_detailed(pdm, target_landmarks)

    fitted_landmarks = pdm.params_to_landmarks_2d(fitted_params)

    print(f"\nFinal landmark 36: ({fitted_landmarks[36, 0]:.4f}, {fitted_landmarks[36, 1]:.4f})")

    # Compare movements
    eye_delta = target_landmarks[36] - main_landmarks[36]
    fit_delta = fitted_landmarks[36] - main_landmarks[36]

    print(f"\nMovement analysis for landmark 36:")
    print(f"  Eye refinement requested: ({eye_delta[0]:+.4f}, {eye_delta[1]:+.4f})")
    print(f"  fit_to_landmarks produced: ({fit_delta[0]:+.4f}, {fit_delta[1]:+.4f})")
    print(f"  Recovery ratio: X={fit_delta[0]/eye_delta[0]*100:.1f}%, Y={fit_delta[1]/eye_delta[1]*100:.1f}%")

    # Parameter comparison
    print(f"\nParameter comparison:")
    print(f"  Initial: scale={init_params[0]:.4f}, tx={init_params[4]:.4f}, ty={init_params[5]:.4f}")
    print(f"  Fitted:  scale={fitted_params[0]:.4f}, tx={fitted_params[4]:.4f}, ty={fitted_params[5]:.4f}")


def fit_to_landmarks_detailed(pdm, landmarks: np.ndarray):
    """Fit PDM parameters with detailed iteration logging."""

    n = pdm.n_points
    m = pdm.n_modes

    landmarks = landmarks.reshape(-1, 2)

    # All visible
    visi_count = n

    # Landmark locations in blocked format
    landmark_locs_vis = np.zeros((visi_count * 2, 1), dtype=np.float32)
    for i in range(n):
        landmark_locs_vis[i] = landmarks[i, 0]
        landmark_locs_vis[i + visi_count] = landmarks[i, 1]

    # Initial params from bbox
    min_x, max_x = landmarks[:, 0].min(), landmarks[:, 0].max()
    min_y, max_y = landmarks[:, 1].min(), landmarks[:, 1].max()

    width = abs(max_x - min_x)
    height = abs(max_y - min_y)

    # Model bbox
    neutral_params = np.zeros(pdm.n_params, dtype=np.float32)
    neutral_params[0] = 1.0
    neutral_lm = pdm.params_to_landmarks_2d(neutral_params)
    model_width = neutral_lm[:, 0].max() - neutral_lm[:, 0].min()
    model_height = neutral_lm[:, 1].max() - neutral_lm[:, 1].min()

    scaling = ((width / model_width) + (height / model_height)) / 2.0
    translation = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float32)

    # Initialize
    loc_params = np.zeros(m, dtype=np.float32)
    glob_params = np.array([scaling, 0.0, 0.0, 0.0, translation[0], translation[1]], dtype=np.float32)

    print(f"\nInitial from bbox:")
    print(f"  scale={glob_params[0]:.6f}")
    print(f"  tx={glob_params[4]:.6f}, ty={glob_params[5]:.6f}")

    # Regularization
    reg_factor = 1.0
    regularisations = np.zeros(6 + m, dtype=np.float32)
    regularisations[6:] = reg_factor / pdm.eigen_values.flatten()
    reg_diag = np.diag(regularisations)

    W = np.eye(visi_count * 2, dtype=np.float32)
    M = pdm.mean_shape.flatten().reshape(-1, 1)
    V = pdm.princ_comp

    prev_error = float('inf')
    not_improved_in = 0

    # Track error and landmark 36 position each iteration
    print(f"\n{'Iter':>4} {'Error':>10} {'Δ Error':>10} {'LM36 X':>10} {'LM36 Y':>10} {'Scale':>10} {'TX':>10}")
    print("-" * 80)

    for iteration in range(1000):
        # Current shape
        shape_3d = M + V @ loc_params.reshape(-1, 1)
        shape_3d = shape_3d.reshape(3, n).T

        rotation_init = glob_params[1:4]
        R = pdm._euler_to_rotation_matrix(rotation_init)
        R_2d = R[:2, :]

        curr_shape_2d = glob_params[0] * (shape_3d @ R_2d.T)
        curr_shape_2d[:, 0] += glob_params[4]
        curr_shape_2d[:, 1] += glob_params[5]

        # Flatten
        curr_shape = np.zeros((n * 2, 1), dtype=np.float32)
        curr_shape[:n, 0] = curr_shape_2d[:, 0]
        curr_shape[n:, 0] = curr_shape_2d[:, 1]

        error_resid = landmark_locs_vis - curr_shape
        error = np.linalg.norm(error_resid)

        delta_error = prev_error - error if prev_error != float('inf') else 0

        # Log iteration (first 10, then every 10)
        if iteration < 10 or iteration % 10 == 0:
            print(f"{iteration:4d} {error:10.4f} {delta_error:10.4f} "
                  f"{curr_shape_2d[36, 0]:10.4f} {curr_shape_2d[36, 1]:10.4f} "
                  f"{glob_params[0]:10.6f} {glob_params[4]:10.4f}")

        # Check convergence - use RELATIVE like C++ (0.1% improvement)
        # C++: if(0.999 * currError < error) not_improved_in++;
        if prev_error * 0.999 < error:
            not_improved_in += 1
            if not_improved_in >= 3:  # C++ uses 3, Python was using 5
                print(f"\nConverged at iteration {iteration}: no 0.1% improvement for 3 iterations")
                break
        else:
            not_improved_in = 0

        prev_error = error

        # Jacobian
        J = pdm._compute_jacobian_subsampled(loc_params, glob_params, M, V, n)

        J_w_t = J.T @ W
        J_w_t_m = J_w_t @ error_resid
        J_w_t_m[6:] = J_w_t_m[6:] - reg_diag[6:, 6:] @ loc_params.reshape(-1, 1)

        Hessian = J_w_t @ J + reg_diag

        try:
            param_update = np.linalg.solve(Hessian, J_w_t_m).flatten()
        except:
            break

        # Damping
        param_update *= 0.75

        # Update
        full_params = np.concatenate([glob_params, loc_params])
        full_params = pdm.update_params(full_params, param_update)

        glob_params = full_params[:6]
        loc_params = full_params[6:]

    print(f"\nFinal: {iteration} iterations, error={error:.4f}")

    return np.concatenate([glob_params, loc_params])


if __name__ == "__main__":
    test_fit_comparison()
