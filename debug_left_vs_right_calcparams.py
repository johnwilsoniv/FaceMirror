#!/usr/bin/env python3
"""
Compare left and right eye CalcParams to understand why right eye diverges.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def run_calcparams(pdm, target_points, eye_indices, side):
    """Run CalcParams and return final params."""
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
    tx = target_center_x - scale * mean_center_x
    ty = target_center_y - scale * mean_center_y

    # Initialize parameters
    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[4] = tx
    params[5] = ty

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
        J_w_t_m[6:] = J_w_t_m[6:] - regularisation[6:, 6:] @ params[6:]
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
    print("LEFT vs RIGHT EYE CALCPARAMS COMPARISON")
    print("=" * 70)

    # Load both eye PDMs
    left_pdm = EyePDM('pyclnf/models/exported_eye_pdm_left')
    right_pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')

    eye_indices = [8, 10, 12, 14, 16, 18]

    # Target landmarks
    CPP_LEFT_EYE = {
        36: (392.1590, 847.6613), 37: (410.0039, 828.3166),
        38: (436.9223, 826.1841), 39: (461.9583, 842.8420),
        40: (438.4380, 850.4288), 41: (411.4089, 853.9998)
    }
    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062), 43: (583.5312, 812.9091),
        44: (611.5075, 810.8795), 45: (635.8052, 825.4882),
        46: (615.4611, 834.7896), 47: (587.8980, 837.3765)
    }

    # C++ results
    cpp_left = {'scale': 3.362953, 'rot': (-0.226434, -0.072403, 0.113299),
                'tx': 427.058319, 'ty': 839.989624}
    cpp_right = {'scale': 3.533595, 'rot': (-0.220360, -0.104840, -0.101141),
                 'tx': 602.636292, 'ty': 807.187805}

    # Run CalcParams for both eyes
    left_target = np.array([CPP_LEFT_EYE[i] for i in [36, 37, 38, 39, 40, 41]])
    right_target = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    left_params, left_err, left_iter = run_calcparams(left_pdm, left_target, eye_indices, 'left')
    right_params, right_err, right_iter = run_calcparams(right_pdm, right_target, eye_indices, 'right')

    print("\n--- LEFT EYE ---")
    print(f"Converged at iteration {left_iter}, error={left_err:.6f}")
    print(f"Python: scale={left_params[0]:.6f}, rot=({left_params[1]:.6f},{left_params[2]:.6f},{left_params[3]:.6f})")
    print(f"        tx={left_params[4]:.6f}, ty={left_params[5]:.6f}")
    print(f"C++:    scale={cpp_left['scale']:.6f}, rot=({cpp_left['rot'][0]:.6f},{cpp_left['rot'][1]:.6f},{cpp_left['rot'][2]:.6f})")
    print(f"        tx={cpp_left['tx']:.6f}, ty={cpp_left['ty']:.6f}")
    print(f"Diff:   tx={abs(left_params[4]-cpp_left['tx']):.4f}, ty={abs(left_params[5]-cpp_left['ty']):.4f}")

    print("\n--- RIGHT EYE ---")
    print(f"Converged at iteration {right_iter}, error={right_err:.6f}")
    print(f"Python: scale={right_params[0]:.6f}, rot=({right_params[1]:.6f},{right_params[2]:.6f},{right_params[3]:.6f})")
    print(f"        tx={right_params[4]:.6f}, ty={right_params[5]:.6f}")
    print(f"C++:    scale={cpp_right['scale']:.6f}, rot=({cpp_right['rot'][0]:.6f},{cpp_right['rot'][1]:.6f},{cpp_right['rot'][2]:.6f})")
    print(f"        tx={cpp_right['tx']:.6f}, ty={cpp_right['ty']:.6f}")
    print(f"Diff:   tx={abs(right_params[4]-cpp_right['tx']):.4f}, ty={abs(right_params[5]-cpp_right['ty']):.4f}")

    # Compare 3D mean shapes
    print("\n--- 3D MEAN SHAPE COMPARISON ---")
    for side, pdm_obj in [('LEFT', left_pdm), ('RIGHT', right_pdm)]:
        mean_flat = pdm_obj.mean_shape.flatten()
        n = pdm_obj.n_points
        X_all = mean_flat[:n]
        Y_all = mean_flat[n:2*n]
        Z_all = mean_flat[2*n:3*n]

        print(f"\n{side} eye 3D coordinates (Eye_8, Eye_14):")
        for idx in [8, 14]:
            print(f"  Eye_{idx}: X={X_all[idx]:.4f}, Y={Y_all[idx]:.4f}, Z={Z_all[idx]:.4f}")

        # Compute Z range
        visible_Z = Z_all[eye_indices]
        print(f"  Z range for visible: [{visible_Z.min():.4f}, {visible_Z.max():.4f}], diff={visible_Z.max()-visible_Z.min():.4f}")

if __name__ == '__main__':
    main()
