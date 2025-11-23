#!/usr/bin/env python3
"""
Debug CalcParams with initial rotation from main model to see if it converges correctly.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def run_calcparams(pdm, target_points, eye_indices, init_rotation, label):
    """Run CalcParams with given initial rotation."""
    print(f"\n{'='*70}")
    print(f"CalcParams with {label}")
    print(f"Initial rotation: ({init_rotation[0]:.6f}, {init_rotation[1]:.6f}, {init_rotation[2]:.6f})")
    print("="*70)

    # Get mean shape for visible landmarks
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X_all = mean_flat[:n]
    Y_all = mean_flat[n:2*n]
    mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])

    # Compute initial scale from bounding boxes
    target_min_x, target_max_x = np.min(target_points[:, 0]), np.max(target_points[:, 0])
    target_min_y, target_max_y = np.min(target_points[:, 1]), np.max(target_points[:, 1])
    target_width = target_max_x - target_min_x
    target_height = target_max_y - target_min_y
    target_center_x = (target_min_x + target_max_x) / 2.0
    target_center_y = (target_min_y + target_max_y) / 2.0

    mean_min_x, mean_max_x = np.min(mean_2d[:, 0]), np.max(mean_2d[:, 0])
    mean_min_y, mean_max_y = np.min(mean_2d[:, 1]), np.max(mean_2d[:, 1])
    mean_width = mean_max_x - mean_min_x
    mean_height = mean_max_y - mean_min_y
    mean_center_x = (mean_min_x + mean_max_x) / 2.0
    mean_center_y = (mean_min_y + mean_max_y) / 2.0

    scale = ((target_width / mean_width) + (target_height / mean_height)) / 2.0

    # Corrected translation (accounting for model center)
    tx = target_center_x - scale * mean_center_x
    ty = target_center_y - scale * mean_center_y

    # Initialize parameters
    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[1] = init_rotation[0]
    params[2] = init_rotation[1]
    params[3] = init_rotation[2]
    params[4] = tx
    params[5] = ty

    print(f"Initial: scale={scale:.4f}, tx={tx:.4f}, ty={ty:.4f}")

    # Run iteration
    reg_factor = 1.0
    W = np.eye(12)

    # Build regularization
    reg_diag = np.zeros(pdm.n_params)
    for i in range(pdm.n_modes):
        if pdm.eigen_values.flatten()[i] > 1e-10:
            reg_diag[6+i] = reg_factor / pdm.eigen_values.flatten()[i]
        else:
            reg_diag[6+i] = 1e10
    regularisation = np.diag(reg_diag)

    prev_error = float('inf')
    not_improved = 0

    for iteration in range(1000):
        # Get current model points
        current_2d = pdm.params_to_landmarks_2d(params)
        current_points = current_2d[eye_indices]

        # Compute error residual
        error_resid = (target_points - current_points).flatten()
        error = np.linalg.norm(error_resid)

        # Check convergence
        if error >= 0.999 * prev_error:
            not_improved += 1
            if not_improved >= 3:
                break
        else:
            not_improved = 0
        prev_error = error

        # Get full Jacobian
        J_full = pdm.compute_jacobian(params)

        # Extract rows for our 6 landmarks
        J = np.zeros((12, pdm.n_params))
        for i, eye_idx in enumerate(eye_indices):
            J[2*i] = J_full[2*eye_idx]
            J[2*i+1] = J_full[2*eye_idx+1]

        # Solve
        J_w_t = J.T @ W
        J_w_t_m = J_w_t @ error_resid
        J_w_t_m[6:] = J_w_t_m[6:] - regularisation[6:, 6:] @ params[6:]
        Hessian = J_w_t @ J + regularisation

        try:
            param_update = np.linalg.solve(Hessian, J_w_t_m)
        except np.linalg.LinAlgError:
            break

        param_update *= 0.75
        params = pdm.update_params(params, param_update)
        params = pdm.clamp_params(params)

    print(f"Converged at iteration {iteration}")
    print(f"Final error: {error:.6f}")
    print(f"Final: scale={params[0]:.6f}, rot=({params[1]:.6f},{params[2]:.6f},{params[3]:.6f})")
    print(f"       tx={params[4]:.6f}, ty={params[5]:.6f}")

    # Compute differences from C++
    cpp_params = [3.533595, -0.220360, -0.104840, -0.101141, 602.636292, 807.187805]
    print(f"\nDiff from C++: scale={abs(params[0]-cpp_params[0]):.4f}, "
          f"rot=({abs(params[1]-cpp_params[1]):.4f},{abs(params[2]-cpp_params[2]):.4f},{abs(params[3]-cpp_params[3]):.4f})")
    print(f"               tx={abs(params[4]-cpp_params[4]):.4f}, ty={abs(params[5]-cpp_params[5]):.4f}")

    return params, error

def main():
    print("=" * 70)
    print("TESTING DIFFERENT INITIAL ROTATIONS FOR RIGHT EYE CALCPARAMS")
    print("=" * 70)

    # Load right eye PDM
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')

    # Eye indices for 6 visible landmarks
    eye_indices = [8, 10, 12, 14, 16, 18]

    # C++ right eye target landmarks
    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062),
        43: (583.5312, 812.9091),
        44: (611.5075, 810.8795),
        45: (635.8052, 825.4882),
        46: (615.4611, 834.7896),
        47: (587.8980, 837.3765)
    }

    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    # Test 1: Zero rotation
    params1, err1 = run_calcparams(pdm, target_points, eye_indices,
                                   [0.0, 0.0, 0.0], "zero rotation")

    # Test 2: C++ final rotation (cheat - using known answer)
    params2, err2 = run_calcparams(pdm, target_points, eye_indices,
                                   [-0.220360, -0.104840, -0.101141], "C++ final rotation")

    # Test 3: Partial rotation (interpolate)
    params3, err3 = run_calcparams(pdm, target_points, eye_indices,
                                   [-0.11, -0.05, -0.05], "partial rotation")

    # Test 4: Try a main model rotation (typical face tilt)
    # Main model for Shorty has some head pose
    params4, err4 = run_calcparams(pdm, target_points, eye_indices,
                                   [-0.15, -0.10, -0.10], "typical head pose")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Zero rotation:      ty={params1[5]:.2f}, error={err1:.4f}")
    print(f"C++ final rot:      ty={params2[5]:.2f}, error={err2:.4f}")
    print(f"Partial rot:        ty={params3[5]:.2f}, error={err3:.4f}")
    print(f"Typical head pose:  ty={params4[5]:.2f}, error={err4:.4f}")
    print(f"\nC++ target:         ty=807.19")

if __name__ == '__main__':
    main()
