#!/usr/bin/env python3
"""
Test if initializing with C++ rotation changes the convergence path.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def run_calcparams(pdm, target_points, eye_indices, initial_rot=None, reg_factor=1.0):
    """Run CalcParams with optional initial rotation."""
    # Get mean shape
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X_all = mean_flat[:n]
    Y_all = mean_flat[n:2*n]
    mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])

    # Compute initial params
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

    if initial_rot is not None:
        params[1] = initial_rot[0]
        params[2] = initial_rot[1]
        params[3] = initial_rot[2]

    # Setup optimization
    W = np.eye(12)
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
        current_2d = pdm.params_to_landmarks_2d(params)
        current_points = current_2d[eye_indices]
        error_resid = (target_points - current_points).flatten()
        error = np.linalg.norm(error_resid)

        if error >= 0.999 * prev_error:
            not_improved += 1
            if not_improved >= 3:
                break
        else:
            not_improved = 0
        prev_error = error

        J_full = pdm.compute_jacobian(params)
        J = np.zeros((12, pdm.n_params))
        for i, eye_idx in enumerate(eye_indices):
            J[2*i] = J_full[2*eye_idx]
            J[2*i+1] = J_full[2*eye_idx+1]

        J_w_t = J.T @ W
        J_w_t_m = J_w_t @ error_resid
        reg_term = regularisation[6:, 6:] @ params[6:]
        J_w_t_m[6:] = J_w_t_m[6:] - reg_term
        Hessian = J_w_t @ J + regularisation

        try:
            param_update = np.linalg.solve(Hessian, J_w_t_m)
        except np.linalg.LinAlgError:
            break

        param_update *= 0.75
        params = pdm.update_params(params, param_update)
        params = pdm.clamp_params(params)

    return params, error, iteration

def main():
    print("=" * 70)
    print("TESTING INITIAL ROTATION EFFECT ON CONVERGENCE")
    print("=" * 70)

    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')
    eye_indices = [8, 10, 12, 14, 16, 18]

    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062), 43: (583.5312, 812.9091),
        44: (611.5075, 810.8795), 45: (635.8052, 825.4882),
        46: (615.4611, 834.7896), 47: (587.8980, 837.3765)
    }
    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    cpp_rot = np.array([-0.220360, -0.104840, -0.101141])
    cpp_target = {'scale': 3.533595, 'tx': 602.636292, 'ty': 807.187805}

    print("\nC++ target: scale=3.5336, tx=602.64, ty=807.19")
    print("-" * 70)

    # Test 1: No initial rotation, reg_factor=1.0
    params, error, iters = run_calcparams(pdm, target_points, eye_indices,
                                           initial_rot=None, reg_factor=1.0)
    print(f"\n1. No init rot, reg=1.0:")
    print(f"   scale={params[0]:.4f}, rot=({params[1]:.4f},{params[2]:.4f},{params[3]:.4f})")
    print(f"   tx={params[4]:.4f}, ty={params[5]:.4f}")
    print(f"   error={error:.4f}, iters={iters}, ty_diff={abs(params[5]-cpp_target['ty']):.2f}")

    # Test 2: C++ rotation, reg_factor=1.0
    params, error, iters = run_calcparams(pdm, target_points, eye_indices,
                                           initial_rot=cpp_rot, reg_factor=1.0)
    print(f"\n2. C++ init rot, reg=1.0:")
    print(f"   scale={params[0]:.4f}, rot=({params[1]:.4f},{params[2]:.4f},{params[3]:.4f})")
    print(f"   tx={params[4]:.4f}, ty={params[5]:.4f}")
    print(f"   error={error:.4f}, iters={iters}, ty_diff={abs(params[5]-cpp_target['ty']):.2f}")

    # Test 3: C++ rotation, higher reg_factor
    for reg in [10.0, 50.0, 100.0, 500.0]:
        params, error, iters = run_calcparams(pdm, target_points, eye_indices,
                                               initial_rot=cpp_rot, reg_factor=reg)
        print(f"\n3. C++ init rot, reg={reg}:")
        print(f"   scale={params[0]:.4f}, rot=({params[1]:.4f},{params[2]:.4f},{params[3]:.4f})")
        print(f"   tx={params[4]:.4f}, ty={params[5]:.4f}")
        print(f"   error={error:.4f}, iters={iters}, ty_diff={abs(params[5]-cpp_target['ty']):.2f}")

    # Test 4: Try VERY high regularization to force minimal shape usage
    print("\n" + "=" * 70)
    print("Testing very high regularization (force zero shape params)")
    print("=" * 70)

    for reg in [1000.0, 5000.0, 10000.0]:
        params, error, iters = run_calcparams(pdm, target_points, eye_indices,
                                               initial_rot=cpp_rot, reg_factor=reg)
        print(f"\nreg={reg}:")
        print(f"   scale={params[0]:.4f}, tx={params[4]:.4f}, ty={params[5]:.4f}")
        print(f"   shape[0:3]=({params[6]:.4f},{params[7]:.4f},{params[8]:.4f})")
        print(f"   error={error:.4f}, ty_diff={abs(params[5]-cpp_target['ty']):.2f}")

if __name__ == '__main__':
    main()
