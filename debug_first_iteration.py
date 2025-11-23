#!/usr/bin/env python3
"""
Deep dive into first iteration to find exactly where Python differs from C++.
Compare specific matrix elements and intermediate values.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 80)
    print("FIRST ITERATION DEEP DIVE - RIGHT EYE")
    print("=" * 80)

    # Load right eye PDM
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')

    # Print PDM info
    print(f"\nPDM Info:")
    print(f"  n_points: {pdm.n_points}")
    print(f"  n_modes: {pdm.n_modes}")
    print(f"  eigenvalues: {pdm.eigen_values.flatten()}")

    # Eye indices
    eye_indices = [8, 10, 12, 14, 16, 18]

    # Target landmarks
    target_points = np.array([
        [560.8530, 833.9062],  # Eye_8
        [583.5312, 812.9091],  # Eye_10
        [611.5075, 810.8795],  # Eye_12
        [635.8052, 825.4882],  # Eye_14
        [615.4611, 834.7896],  # Eye_16
        [587.8980, 837.3765],  # Eye_18
    ])

    # Get mean shape for visible landmarks
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X_all = mean_flat[:n]
    Y_all = mean_flat[n:2*n]
    Z_all = mean_flat[2*n:3*n]

    print(f"\nMean shape 3D for visible landmarks:")
    for i, idx in enumerate(eye_indices):
        print(f"  Eye_{idx}: X={X_all[idx]:.6f}, Y={Y_all[idx]:.6f}, Z={Z_all[idx]:.6f}")

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

    # WITHOUT mean center correction (like C++ second CalcParams)
    print(f"\nInitial params (without center correction):")
    print(f"  scale={scale:.6f}, tx={tx:.6f}, ty={ty:.6f}")

    # Initialize parameters
    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[4] = tx
    params[5] = ty

    # Get current 2D landmarks
    current_2d = pdm.params_to_landmarks_2d(params)
    current_points = current_2d[eye_indices]

    print(f"\nInitial 2D landmarks:")
    for i, idx in enumerate(eye_indices):
        print(f"  Eye_{idx}: target=({target_points[i, 0]:.4f}, {target_points[i, 1]:.4f}), "
              f"current=({current_points[i, 0]:.4f}, {current_points[i, 1]:.4f})")

    # Error residual
    error_resid = (target_points - current_points).flatten()
    print(f"\nError residual (12 values, interleaved x,y):")
    print(f"  {error_resid}")

    # Jacobian
    J_full = pdm.compute_jacobian(params)
    J = np.zeros((12, pdm.n_params))
    for i, eye_idx in enumerate(eye_indices):
        J[2*i] = J_full[2*eye_idx]
        J[2*i+1] = J_full[2*eye_idx+1]

    print(f"\nJacobian for Eye_8 (row 0 = x, row 1 = y):")
    print(f"  Row 0 (x): scale={J[0,0]:.6f}, rot=({J[0,1]:.6f}, {J[0,2]:.6f}, {J[0,3]:.6f}), tx={J[0,4]:.6f}")
    print(f"  Row 1 (y): scale={J[1,0]:.6f}, rot=({J[1,1]:.6f}, {J[1,2]:.6f}, {J[1,3]:.6f}), ty={J[1,5]:.6f}")

    # What does the Jacobian tell us?
    # J[1,5] = 1.0 means increasing ty by 1 increases landmark y by 1
    # J[1,1] is the derivative of y w.r.t. rot_x
    print(f"\n  Interpretation:")
    print(f"    d(Eye_8_y)/d(ty) = {J[1,5]:.2f} (directly adds to y)")
    print(f"    d(Eye_8_y)/d(rot_x) = {J[1,1]:.6f}")
    print(f"    d(Eye_8_y)/d(rot_y) = {J[1,2]:.6f}")
    print(f"    d(Eye_8_y)/d(rot_z) = {J[1,3]:.6f}")

    # Compute J^T @ error
    W = np.eye(12)
    J_w_t = J.T @ W
    J_w_t_m = J_w_t @ error_resid

    print(f"\nJ^T @ W @ error_resid (gradient, before reg):")
    print(f"  scale gradient: {J_w_t_m[0]:.6f}")
    print(f"  rot gradient: ({J_w_t_m[1]:.6f}, {J_w_t_m[2]:.6f}, {J_w_t_m[3]:.6f})")
    print(f"  tx gradient: {J_w_t_m[4]:.6f}")
    print(f"  ty gradient: {J_w_t_m[5]:.6f}")

    # The gradient tells us the direction to move params
    # Positive gradient means increase parameter to reduce error
    print(f"\n  Interpretation:")
    print(f"    ty gradient = {J_w_t_m[5]:.2f} means ", end="")
    if J_w_t_m[5] > 0:
        print("INCREASE ty to reduce error")
    else:
        print("DECREASE ty to reduce error")

    # But wait - we need the Hessian inverse
    # The actual update is: delta_p = H^-1 @ gradient
    reg_diag = np.zeros(pdm.n_params)
    for i in range(pdm.n_modes):
        if pdm.eigen_values.flatten()[i] > 1e-10:
            reg_diag[6+i] = 1.0 / pdm.eigen_values.flatten()[i]
        else:
            reg_diag[6+i] = 1e10
    regularisation = np.diag(reg_diag)

    Hessian = J_w_t @ J + regularisation

    print(f"\nHessian diagonal (first 6):")
    print(f"  {np.diag(Hessian)[:6]}")

    # Solve
    param_update = np.linalg.solve(Hessian, J_w_t_m)

    print(f"\nParam update (before damping):")
    print(f"  scale: {param_update[0]:.6f}")
    print(f"  rot: ({param_update[1]:.6f}, {param_update[2]:.6f}, {param_update[3]:.6f})")
    print(f"  tx: {param_update[4]:.6f}")
    print(f"  ty: {param_update[5]:.6f}")

    # Key insight: why is ty update positive (pushing up) instead of negative (pushing down)?
    # Let's trace through the calculation manually for ty

    print(f"\n{'='*80}")
    print("MANUAL CHECK: Why is ty update positive?")
    print("="*80)

    # The error residuals for y components
    y_errors = error_resid[1::2]  # Every other element starting from 1
    print(f"\nY errors (target_y - current_y):")
    for i, idx in enumerate(eye_indices):
        print(f"  Eye_{idx}: {y_errors[i]:.4f} ({'+' if y_errors[i] > 0 else ''} means current is {'below' if y_errors[i] > 0 else 'above'} target)")

    # Sum of y errors
    print(f"\nSum of y errors: {np.sum(y_errors):.4f}")

    # If sum is positive, landmarks are below target, so we need to increase ty
    # If sum is negative, landmarks are above target, so we need to decrease ty

    # Let's also check the mean center approach
    target_center_y = np.mean(target_points[:, 1])
    current_center_y = np.mean(current_points[:, 1])
    print(f"\nCenter Y comparison:")
    print(f"  Target center Y: {target_center_y:.4f}")
    print(f"  Current center Y: {current_center_y:.4f}")
    print(f"  Difference: {target_center_y - current_center_y:.4f}")

if __name__ == '__main__':
    main()
