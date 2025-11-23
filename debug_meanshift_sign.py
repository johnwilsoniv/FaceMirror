#!/usr/bin/env python3
"""
Debug script to trace mean-shift sign error in eye refinement.

This tests whether the mean-shift direction from response map peak
matches the actual landmark movement direction.
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel

# Load test data
image = cv2.imread('comparison_frame_0030.jpg')
if image is None:
    image = cv2.imread('archive/results/test_frame.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ground truth C++ landmarks for test
cpp_landmarks = np.array([
    [399.2, 824.9],   # 36
    [417.5, 809.8],   # 37
    [444.3, 806.7],   # 38
    [468.8, 822.2],   # 39
    [445.2, 828.4],   # 40
    [417.9, 830.2],   # 41
])

# Create eye model
eye_model = HierarchicalEyeModel()
eye_model.load_model('models/clnf_models/main_clnf_general.txt', 'left_eye_28')
pdm = eye_model.pdm

# Mapping from main landmark indices to eye indices
mapping = {36: 8, 37: 10, 38: 12, 39: 14, 40: 16, 41: 18}

# Create eye model input landmarks
eye_input = np.zeros((6, 2))
for i, (main_idx, eye_idx) in enumerate(mapping.items()):
    eye_input[i] = cpp_landmarks[i]

print("=== Mean-Shift Sign Debug ===")
print()
print("Input landmarks (from C++ as ground truth):")
for i, (main_idx, eye_idx) in enumerate(mapping.items()):
    print(f"  Eye_{eye_idx} (main {main_idx}): ({eye_input[i, 0]:.4f}, {eye_input[i, 1]:.4f})")

# Initialize eye model with these landmarks
try:
    # Fit initial parameters
    params = pdm.fit_to_landmarks(eye_input, mapping)

    # Get initial 28 eye landmarks
    initial_landmarks = pdm.calc_shape(params)

    print()
    print("Initial fitted params:")
    print(f"  scale: {params[0]:.6f}")
    print(f"  rot: ({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
    print(f"  tx, ty: ({params[4]:.6f}, {params[5]:.6f})")

    print()
    print("Initial 28 eye landmarks (subset):")
    for idx in [0, 8, 10, 12, 14, 16, 18]:
        print(f"  Eye_{idx}: ({initial_landmarks[idx, 0]:.4f}, {initial_landmarks[idx, 1]:.4f})")

    # Run ONE iteration of refinement
    ws = 3
    eye_model._current_window_size = ws

    # Compute response maps for all landmarks
    response_maps = eye_model._compute_eye_response_maps(gray, initial_landmarks, ws)

    # Print response map info for landmark 8 (mapped from main 36)
    lm_idx = 8
    if lm_idx in response_maps:
        resp = response_maps[lm_idx]
        print()
        print(f"=== Response Map for Eye_{lm_idx} ===")
        print(f"Shape: {resp.shape}")
        print(f"Min: {resp.min():.6f}, Max: {resp.max():.6f}")
        print()
        print("Response map values (3x3):")
        for i in range(3):
            row_vals = [f"{resp[i, j]:.4f}" for j in range(3)]
            print(f"  Row {i}: {', '.join(row_vals)}")

        # Find peak
        peak_idx = np.unravel_index(np.argmax(resp), resp.shape)
        center = (ws - 1) / 2.0
        print()
        print(f"Peak at (row={peak_idx[0]}, col={peak_idx[1]})")
        print(f"Center: {center}")
        print(f"Peak offset from center: dx={peak_idx[1] - center}, dy={peak_idx[0] - center}")

        # Compute mean-shift using KDE
        a_kde = -0.5 / (1.0 * 1.0)  # sigma = 1.0
        total_weight = 0.0
        mx = 0.0
        my = 0.0

        for ii in range(ws):
            for jj in range(ws):
                dist_sq = (ii - center)**2 + (jj - center)**2
                kde_weight = np.exp(a_kde * dist_sq)
                weight = resp[ii, jj] * kde_weight
                total_weight += weight
                mx += weight * jj
                my += weight * ii

        ms_x = (mx / total_weight) - center
        ms_y = (my / total_weight) - center

        print()
        print("KDE Mean-shift computation:")
        print(f"  mx = {mx:.6f}")
        print(f"  my = {my:.6f}")
        print(f"  total_weight = {total_weight:.6f}")
        print(f"  ms_x = {mx/total_weight:.6f} - {center} = {ms_x:.6f}")
        print(f"  ms_y = {my/total_weight:.6f} - {center} = {ms_y:.6f}")

        if ms_x > 0:
            print(f"\n  → Mean-shift says move RIGHT by {ms_x:.2f} pixels")
        else:
            print(f"\n  → Mean-shift says move LEFT by {-ms_x:.2f} pixels")
        if ms_y > 0:
            print(f"  → Mean-shift says move DOWN by {ms_y:.2f} pixels")
        else:
            print(f"  → Mean-shift says move UP by {-ms_y:.2f} pixels")

    # Now compute full mean-shift vector
    mean_shift = eye_model._compute_eye_mean_shift(initial_landmarks, response_maps, eye_model.patch_experts)

    print()
    print(f"Full mean-shift vector for Eye_8:")
    print(f"  mean_shift[16] (X) = {mean_shift[16]:.6f}")
    print(f"  mean_shift[17] (Y) = {mean_shift[17]:.6f}")

    # Solve for parameter update
    J = pdm.compute_jacobian(params)
    J_rigid = J[:, :6]
    n_points = pdm.n_points
    W = np.eye(2 * n_points)
    A = J_rigid.T @ W @ J_rigid
    b = J_rigid.T @ W @ mean_shift
    delta_p_rigid = np.linalg.solve(A, b)

    print()
    print("Rigid parameter update (before damping):")
    print(f"  Δscale = {delta_p_rigid[0]:.6f}")
    print(f"  Δrx = {delta_p_rigid[1]:.6f}")
    print(f"  Δry = {delta_p_rigid[2]:.6f}")
    print(f"  Δrz = {delta_p_rigid[3]:.6f}")
    print(f"  Δtx = {delta_p_rigid[4]:.6f}")
    print(f"  Δty = {delta_p_rigid[5]:.6f}")

    # Apply update
    delta_p = np.zeros(pdm.n_params)
    delta_p[:6] = delta_p_rigid * 0.5  # damping

    new_params = params + delta_p
    new_landmarks = pdm.calc_shape(new_params)

    # Check how landmark 8 moved
    old_pos = initial_landmarks[8]
    new_pos = new_landmarks[8]
    actual_dx = new_pos[0] - old_pos[0]
    actual_dy = new_pos[1] - old_pos[1]

    print()
    print("Actual landmark Eye_8 movement:")
    print(f"  Old: ({old_pos[0]:.4f}, {old_pos[1]:.4f})")
    print(f"  New: ({new_pos[0]:.4f}, {new_pos[1]:.4f})")
    print(f"  Delta: dx={actual_dx:.4f}, dy={actual_dy:.4f}")

    if actual_dx > 0:
        print(f"\n  → Landmark MOVED RIGHT by {actual_dx:.2f} pixels")
    else:
        print(f"\n  → Landmark MOVED LEFT by {-actual_dx:.2f} pixels")
    if actual_dy > 0:
        print(f"  → Landmark MOVED DOWN by {actual_dy:.2f} pixels")
    else:
        print(f"  → Landmark MOVED UP by {-actual_dy:.2f} pixels")

    # Check for sign mismatch
    print()
    print("=== SIGN CONSISTENCY CHECK ===")
    x_match = (ms_x > 0 and actual_dx > 0) or (ms_x < 0 and actual_dx < 0) or abs(ms_x) < 0.01
    y_match = (ms_y > 0 and actual_dy > 0) or (ms_y < 0 and actual_dy < 0) or abs(ms_y) < 0.01

    if x_match:
        print("X direction: ✓ CONSISTENT")
    else:
        print("X direction: ✗ SIGN ERROR - mean-shift and movement have OPPOSITE signs!")

    if y_match:
        print("Y direction: ✓ CONSISTENT")
    else:
        print("Y direction: ✗ SIGN ERROR - mean-shift and movement have OPPOSITE signs!")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
