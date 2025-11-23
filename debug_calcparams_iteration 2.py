#!/usr/bin/env python3
"""
Debug CalcParams iteration step-by-step for right eye to find where divergence occurs.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("CALCPARAMS ITERATION DEBUG - RIGHT EYE")
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

    # Target points in eye model order
    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    # Get mean shape for visible landmarks
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X_all = mean_flat[:n]
    Y_all = mean_flat[n:2*n]
    Z_all = mean_flat[2*n:3*n]

    mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])
    mean_3d = np.column_stack([X_all[eye_indices], Y_all[eye_indices], Z_all[eye_indices]])

    print("\nMean shape 3D for 6 visible landmarks:")
    for i, idx in enumerate(eye_indices):
        print(f"  Eye_{idx}: X={mean_3d[i, 0]:.4f}, Y={mean_3d[i, 1]:.4f}, Z={mean_3d[i, 2]:.4f}")

    # Compute initial params (like my current implementation)
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

    # Compute initial params
    scale = ((target_width / mean_width) + (target_height / mean_height)) / 2.0
    tx = target_center_x
    ty = target_center_y

    # Try with corrected translation (like C++ first CalcParams)
    tx_corrected = tx - scale * mean_center_x
    ty_corrected = ty - scale * mean_center_y

    print(f"\nInitial params (uncorrected):")
    print(f"  scale={scale:.6f}, tx={tx:.6f}, ty={ty:.6f}")
    print(f"\nInitial params (corrected for model center):")
    print(f"  scale={scale:.6f}, tx={tx_corrected:.6f}, ty={ty_corrected:.6f}")

    # Initialize parameters with corrected translation
    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[4] = tx_corrected
    params[5] = ty_corrected

    # Run iteration (matching C++ CalcParams)
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

    print("\n--- ITERATION ---")
    for iteration in range(50):  # Only show first 50 iterations
        # Get current model points
        current_2d = pdm.params_to_landmarks_2d(params)
        current_points = current_2d[eye_indices]

        # Compute error residual (target - current)
        error_resid = (target_points - current_points).flatten()
        error = np.linalg.norm(error_resid)

        # Print every iteration for first 10, then every 10
        if iteration < 10 or iteration % 10 == 0:
            print(f"\nIteration {iteration}:")
            print(f"  error={error:.6f}")
            print(f"  scale={params[0]:.6f}, rot=({params[1]:.6f},{params[2]:.6f},{params[3]:.6f})")
            print(f"  tx={params[4]:.6f}, ty={params[5]:.6f}")

            # Print landmark errors
            if iteration < 3:
                for i, idx in enumerate(eye_indices):
                    err = np.sqrt((target_points[i, 0] - current_points[i, 0])**2 +
                                  (target_points[i, 1] - current_points[i, 1])**2)
                    print(f"    Eye_{idx}: err={err:.2f}px")

        # Check convergence
        if error >= 0.999 * prev_error:
            not_improved += 1
            if not_improved >= 3:
                print(f"\nConverged at iteration {iteration}")
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

        # Weighted Jacobian
        J_w_t = J.T @ W

        # Projection of residuals onto jacobians
        J_w_t_m = J_w_t @ error_resid

        # Add regularization term for shape params
        J_w_t_m[6:] = J_w_t_m[6:] - regularisation[6:, 6:] @ params[6:]

        # Compute Hessian
        Hessian = J_w_t @ J + regularisation

        # Solve for parameter update
        try:
            param_update = np.linalg.solve(Hessian, J_w_t_m)
        except np.linalg.LinAlgError:
            print("Solve failed!")
            break

        # Damping (C++ uses 0.75)
        param_update *= 0.75

        # Update parameters
        params = pdm.update_params(params, param_update)
        params = pdm.clamp_params(params)

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Python converged to:")
    print(f"  scale={params[0]:.6f}, rot=({params[1]:.6f},{params[2]:.6f},{params[3]:.6f})")
    print(f"  tx={params[4]:.6f}, ty={params[5]:.6f}")

    print(f"\nC++ converged to:")
    print(f"  scale=3.533595, rot=(-0.220360,-0.104840,-0.101141)")
    print(f"  tx=602.636292, ty=807.187805")

    print(f"\nDifference:")
    print(f"  scale: {abs(params[0] - 3.533595):.6f}")
    print(f"  rot_x: {abs(params[1] - (-0.220360)):.6f}")
    print(f"  rot_y: {abs(params[2] - (-0.104840)):.6f}")
    print(f"  rot_z: {abs(params[3] - (-0.101141)):.6f}")
    print(f"  tx: {abs(params[4] - 602.636292):.6f}")
    print(f"  ty: {abs(params[5] - 807.187805):.6f}")

if __name__ == '__main__':
    main()
