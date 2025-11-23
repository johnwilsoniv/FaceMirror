#!/usr/bin/env python3
"""
Detailed iteration-by-iteration trace of CalcParams to find divergence from C++.
Prints values that can be compared directly with C++ debug output.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 80)
    print("DETAILED CALCPARAMS ITERATION TRACE - RIGHT EYE")
    print("=" * 80)

    # Load right eye PDM
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')

    # Eye indices for 6 visible landmarks
    eye_indices = [8, 10, 12, 14, 16, 18]

    # Target landmarks
    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062),
        43: (583.5312, 812.9091),
        44: (611.5075, 810.8795),
        45: (635.8052, 825.4882),
        46: (615.4611, 834.7896),
        47: (587.8980, 837.3765)
    }

    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    # Get mean shape
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X_all = mean_flat[:n]
    Y_all = mean_flat[n:2*n]
    mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])

    # Compute initial params (with mean center correction)
    target_min_x, target_max_x = np.min(target_points[:, 0]), np.max(target_points[:, 0])
    target_min_y, target_max_y = np.min(target_points[:, 1]), np.max(target_points[:, 1])
    target_width = target_max_x - target_min_x
    target_height = target_max_y - target_min_y

    mean_min_x, mean_max_x = np.min(mean_2d[:, 0]), np.max(mean_2d[:, 0])
    mean_min_y, mean_max_y = np.min(mean_2d[:, 1]), np.max(mean_2d[:, 1])
    mean_width = mean_max_x - mean_min_x
    mean_height = mean_max_y - mean_min_y

    scale = ((target_width / mean_width) + (target_height / mean_height)) / 2.0

    tx = (target_min_x + target_max_x) / 2.0
    ty = (target_min_y + target_max_y) / 2.0
    mean_center_x = (mean_min_x + mean_max_x) / 2.0
    mean_center_y = (mean_min_y + mean_max_y) / 2.0
    tx = tx - scale * mean_center_x
    ty = ty - scale * mean_center_y

    # Initialize parameters
    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[4] = tx
    params[5] = ty

    print(f"\n=== INITIAL STATE ===")
    print(f"scale={scale:.6f}")
    print(f"tx={tx:.6f}, ty={ty:.6f}")
    print(f"mean_center=({mean_center_x:.6f}, {mean_center_y:.6f})")

    # Setup optimization
    reg_factor = 1.0  # Use C++ value for comparison
    W = np.eye(12)

    reg_diag = np.zeros(pdm.n_params)
    for i in range(pdm.n_modes):
        if pdm.eigen_values.flatten()[i] > 1e-10:
            reg_diag[6+i] = reg_factor / pdm.eigen_values.flatten()[i]
        else:
            reg_diag[6+i] = 1e10
    regularisation = np.diag(reg_diag)

    print(f"\nFirst 3 regularization diagonals (shape): {reg_diag[6:9]}")

    prev_error = float('inf')
    not_improved = 0

    # Run iterations with detailed output
    for iteration in range(20):  # Only first 20 iterations
        # Get current model points
        current_2d = pdm.params_to_landmarks_2d(params)
        current_points = current_2d[eye_indices]

        # Compute error residual
        error_resid = (target_points - current_points).flatten()
        error = np.linalg.norm(error_resid)

        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}")

        print(f"\nCurrent params:")
        print(f"  scale={params[0]:.8f}")
        print(f"  rot=({params[1]:.8f}, {params[2]:.8f}, {params[3]:.8f})")
        print(f"  tx={params[4]:.8f}, ty={params[5]:.8f}")
        print(f"  shape[0:3]=({params[6]:.8f}, {params[7]:.8f}, {params[8]:.8f})")

        print(f"\nCurrent 2D landmarks (first 3 visible):")
        for i in range(3):
            idx = eye_indices[i]
            print(f"  Eye_{idx}: ({current_points[i, 0]:.6f}, {current_points[i, 1]:.6f})")

        print(f"\nError residual (first 6 = first 3 points x,y):")
        for i in range(6):
            print(f"  [{i}]: {error_resid[i]:.8f}")

        print(f"\nTotal error: {error:.8f}")

        # Check convergence
        if error >= 0.999 * prev_error:
            not_improved += 1
            if not_improved >= 3:
                print(f"\n*** CONVERGED ***")
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

        print(f"\nJacobian (first row, first 6 cols - rigid):")
        print(f"  {J[0, :6]}")

        # Weighted Jacobian
        J_w_t = J.T @ W

        # Projection of residuals onto jacobians
        J_w_t_m = J_w_t @ error_resid

        print(f"\nJ_w_t_m before reg (first 8):")
        print(f"  {J_w_t_m[:8]}")

        # Add regularization term for shape params
        reg_term = regularisation[6:, 6:] @ params[6:]
        J_w_t_m[6:] = J_w_t_m[6:] - reg_term

        print(f"\nReg term (first 3): {reg_term[:3]}")
        print(f"J_w_t_m after reg (first 8):")
        print(f"  {J_w_t_m[:8]}")

        # Compute Hessian
        Hessian = J_w_t @ J + regularisation

        print(f"\nHessian diagonal (first 8):")
        print(f"  {np.diag(Hessian)[:8]}")

        # Solve for parameter update
        try:
            param_update = np.linalg.solve(Hessian, J_w_t_m)
        except np.linalg.LinAlgError:
            print("SOLVE FAILED!")
            break

        print(f"\nParam update before damping:")
        print(f"  scale={param_update[0]:.8f}")
        print(f"  rot=({param_update[1]:.8f}, {param_update[2]:.8f}, {param_update[3]:.8f})")
        print(f"  tx={param_update[4]:.8f}, ty={param_update[5]:.8f}")

        # Damping
        param_update *= 0.75

        print(f"\nParam update after 0.75 damping:")
        print(f"  scale={param_update[0]:.8f}")
        print(f"  rot=({param_update[1]:.8f}, {param_update[2]:.8f}, {param_update[3]:.8f})")
        print(f"  tx={param_update[4]:.8f}, ty={param_update[5]:.8f}")

        # Update parameters
        params = pdm.update_params(params, param_update)
        params = pdm.clamp_params(params)

    # Final results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Python converged to:")
    print(f"  scale={params[0]:.6f}")
    print(f"  rot=({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
    print(f"  tx={params[4]:.6f}, ty={params[5]:.6f}")
    print(f"  shape={params[6:11]}")

    print(f"\nC++ target:")
    print(f"  scale=3.533595")
    print(f"  rot=(-0.220360, -0.104840, -0.101141)")
    print(f"  tx=602.636292, ty=807.187805")

if __name__ == '__main__':
    main()
