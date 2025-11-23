#!/usr/bin/env python3
"""
Detailed debug of fit_to_landmarks convergence.

Compare input landmarks to output landmarks to understand why
X direction doesn't change much.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.core.pdm import PDM

def test_fit_to_landmarks_detailed():
    """Test fit_to_landmarks with eye model output positions."""

    print("="*70)
    print("DETAILED FIT_TO_LANDMARKS DEBUG")
    print("="*70)

    pdm = PDM("pyclnf/models/exported_pdm")
    print(f"PDM: {pdm.n_points} landmarks, {pdm.n_modes} modes")

    # Simulate the situation after eye refinement
    # Start from main model output (these would be the initial landmarks)
    # Then simulate eye model moving some landmarks

    # Create initial params (simulating main model output)
    init_params = np.zeros(pdm.n_params, dtype=np.float32)
    init_params[0] = 3.37  # scale
    init_params[1] = -0.117  # pitch
    init_params[2] = 0.175   # yaw
    init_params[3] = -0.1    # roll
    init_params[4] = 425.0   # tx
    init_params[5] = 820.0   # ty

    # Get main model landmarks
    main_landmarks = pdm.params_to_landmarks_2d(init_params)

    print(f"\nMain model landmark 36: ({main_landmarks[36, 0]:.4f}, {main_landmarks[36, 1]:.4f})")

    # Simulate eye model moving landmarks 36-41
    # Move landmark 36 leftward (like Python eye model does)
    eye_refined = main_landmarks.copy()
    eye_refined[36] = [main_landmarks[36, 0] - 0.6, main_landmarks[36, 1] + 4.0]

    # Also move other eye landmarks slightly
    for i in range(37, 42):
        eye_refined[i, 1] += 3.0  # Move down

    print(f"Eye-refined landmark 36: ({eye_refined[36, 0]:.4f}, {eye_refined[36, 1]:.4f})")
    print(f"  Movement: ({eye_refined[36, 0] - main_landmarks[36, 0]:.4f}, {eye_refined[36, 1] - main_landmarks[36, 1]:.4f})")

    # Now call fit_to_landmarks
    print("\n" + "="*70)
    print("Calling fit_to_landmarks")
    print("="*70)

    # Call with detailed logging
    fitted_params = fit_to_landmarks_debug(pdm, eye_refined)

    fitted_landmarks = pdm.params_to_landmarks_2d(fitted_params)

    print(f"\nFitted landmark 36: ({fitted_landmarks[36, 0]:.4f}, {fitted_landmarks[36, 1]:.4f})")

    # Compare movements
    eye_movement = eye_refined[36] - main_landmarks[36]
    fit_movement = fitted_landmarks[36] - main_landmarks[36]

    print(f"\nMovement comparison:")
    print(f"  Eye model moved 36 by: ({eye_movement[0]:+.4f}, {eye_movement[1]:+.4f})")
    print(f"  fit_to_landmarks gave: ({fit_movement[0]:+.4f}, {fit_movement[1]:+.4f})")

    # Check parameter changes
    print(f"\nParameter comparison:")
    print(f"  Initial: scale={init_params[0]:.4f}, tx={init_params[4]:.4f}, ty={init_params[5]:.4f}")
    print(f"  Fitted:  scale={fitted_params[0]:.4f}, tx={fitted_params[4]:.4f}, ty={fitted_params[5]:.4f}")


def fit_to_landmarks_debug(pdm, landmarks: np.ndarray):
    """Fit with detailed iteration logging."""

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
    print(f"  rotation=(0, 0, 0)")
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

        # Check convergence
        if prev_error - error < 0.0001:
            not_improved_in += 1
            if not_improved_in >= 5:
                print(f"\nConverged at iteration {iteration}: error={error:.4f}")
                break
        else:
            not_improved_in = 0

        prev_error = error

        # Log first few and every 50th iteration
        if iteration < 5 or iteration % 50 == 0:
            lm36_curr = curr_shape_2d[36]
            lm36_target = landmarks[36]
            err_36 = np.sqrt((lm36_curr[0] - lm36_target[0])**2 + (lm36_curr[1] - lm36_target[1])**2)
            print(f"Iter {iteration}: error={error:.4f}, lm36=({lm36_curr[0]:.4f}, {lm36_curr[1]:.4f}), err_36={err_36:.4f}")

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

    return np.concatenate([glob_params, loc_params])


if __name__ == "__main__":
    test_fit_to_landmarks_detailed()
