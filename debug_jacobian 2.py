#!/usr/bin/env python3
"""
Debug the Jacobian computation to find the sign issue.
"""

import numpy as np
import sys
import os
os.chdir('/Users/johnwilsoniv/Documents/SplitFace Open3')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

from core.eye_pdm import EyePDM

def main():
    # Load eye PDM
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_pdm_left'
    left_eye_pdm = EyePDM(model_dir)

    # Use the initial params from our debug output
    # Initial global params: scale=3.371204 rot=(-0.118319,0.176098,-0.099366) tx=425.031629 ty=820.112023
    params = np.zeros(left_eye_pdm.n_params)
    params[0] = 3.371204  # scale
    params[1] = -0.118319  # rx
    params[2] = 0.176098   # ry
    params[3] = -0.099366  # rz
    params[4] = 425.031629 # tx
    params[5] = 820.112023 # ty

    # Compute Jacobian
    J = left_eye_pdm.compute_jacobian(params)

    print("=" * 70)
    print("JACOBIAN ANALYSIS")
    print("=" * 70)

    print(f"\nJacobian shape: {J.shape}")
    print(f"Expected: ({2 * left_eye_pdm.n_points}, {left_eye_pdm.n_params})")

    # Look at landmark 8 (eye_8)
    # Row 16 = x component of landmark 8
    # Row 17 = y component of landmark 8
    lm_idx = 8

    print(f"\n--- Landmark {lm_idx} Jacobian ---")
    print(f"  J[{2*lm_idx},:6] (∂x{lm_idx}/∂params) = {J[2*lm_idx, :6]}")
    print(f"  J[{2*lm_idx+1},:6] (∂y{lm_idx}/∂params) = {J[2*lm_idx+1, :6]}")

    # Use mean-shift from our debug
    # Python Iter 0 Eye_8: ms=(1.154733, -0.032818)
    ms_x = 1.154733
    ms_y = -0.032818

    # Create full mean-shift vector (interleaved)
    mean_shift = np.zeros(2 * left_eye_pdm.n_points)

    # Set only landmark 8 to isolate its effect
    mean_shift[2*lm_idx] = ms_x
    mean_shift[2*lm_idx + 1] = ms_y

    # Extract rigid Jacobian
    J_rigid = J[:, :6]

    # Compute J'W*mean_shift (W=I)
    b = J_rigid.T @ mean_shift

    print(f"\nMean-shift for lm{lm_idx}: ({ms_x}, {ms_y})")
    print(f"\nJ'.T @ mean_shift (only lm{lm_idx} contribution):")
    print(f"  For scale:   {b[0]:.6f}")
    print(f"  For rx:      {b[1]:.6f}")
    print(f"  For ry:      {b[2]:.6f}")
    print(f"  For rz:      {b[3]:.6f}")
    print(f"  For tx:      {b[4]:.6f}")
    print(f"  For ty:      {b[5]:.6f}")

    # The contribution to scale from this landmark
    # ∂x/∂s * ms_x + ∂y/∂s * ms_y
    scale_contrib = J[2*lm_idx, 0] * ms_x + J[2*lm_idx+1, 0] * ms_y
    print(f"\nManual check: ∂x/∂s * ms_x + ∂y/∂s * ms_y = {scale_contrib:.6f}")

    # Now compute what the mean-shift SHOULD be to make the landmark move to target
    # Target: C++ moves lm36 from (391.1, 827.6) to (391.5, 830.2)
    # That's movement (+0.4, +2.6)

    # The relationship is: Δlandmarks = J @ Δparams
    # So: Δparams = solve(J'J, J' @ target_movement)

    target_dx = 0.4  # C++ moves right
    target_dy = 2.6  # C++ moves down

    target = np.zeros(2 * left_eye_pdm.n_points)
    target[2*lm_idx] = target_dx
    target[2*lm_idx + 1] = target_dy

    # Solve for delta_p that would achieve this target
    A = J_rigid.T @ J_rigid
    b_target = J_rigid.T @ target

    try:
        delta_p_for_target = np.linalg.solve(A, b_target)
        print(f"\n--- Delta_p to achieve target ({target_dx}, {target_dy}) ---")
        print(f"  delta_scale: {delta_p_for_target[0]:.6f}")
        print(f"  delta_rx:    {delta_p_for_target[1]:.6f}")
        print(f"  delta_ry:    {delta_p_for_target[2]:.6f}")
        print(f"  delta_rz:    {delta_p_for_target[3]:.6f}")
        print(f"  delta_tx:    {delta_p_for_target[4]:.6f}")
        print(f"  delta_ty:    {delta_p_for_target[5]:.6f}")

        # Apply and check
        # Δlandmarks = J @ Δparams
        landmark_change = J_rigid @ delta_p_for_target
        actual_dx = landmark_change[2*lm_idx]
        actual_dy = landmark_change[2*lm_idx + 1]
        print(f"\n  Resulting movement for lm{lm_idx}: ({actual_dx:.4f}, {actual_dy:.4f})")

    except Exception as e:
        print(f"Solve failed: {e}")

    # Key insight: The mean-shift tells us where to move TO, not the delta
    # Mean-shift (1.15, -0.03) means move right, slightly up
    # But C++ wants to move right and down (+0.4, +2.6)
    # There's a mismatch!

    print("\n" + "=" * 70)
    print("KEY OBSERVATION")
    print("=" * 70)
    print(f"\nMean-shift direction: ({ms_x:.2f}, {ms_y:.2f}) = right, up")
    print(f"Target movement:      ({target_dx:.2f}, {target_dy:.2f}) = right, down")
    print("\nThe Y directions are opposite!")
    print("This suggests the mean-shift Y component might have wrong sign")

if __name__ == '__main__':
    main()
