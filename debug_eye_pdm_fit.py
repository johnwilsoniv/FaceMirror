#!/usr/bin/env python3
"""
Debug eye PDM fitting - compare Python vs C++ parameter initialization.
"""

import sys
sys.path.insert(0, 'pyclnf')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM
from pyclnf.core.eye_patch_expert import LEFT_EYE_MAPPING

def main():
    # C++ landmarks for left eye (from CSV)
    cpp_left_eye = np.array([
        [399.5, 827.5],   # LM36 - outer corner
        [417.9, 809.4],   # LM37 - upper outer
        [444.1, 807.6],   # LM38 - upper inner
        [467.8, 822.7],   # LM39 - inner corner
        [444.8, 830.1],   # LM40 - lower inner
        [418.7, 833.1],   # LM41 - lower outer
    ])

    print("=== C++ Input Landmarks ===")
    for i, (x, y) in enumerate(cpp_left_eye):
        print(f"LM{36+i}: ({x:.4f}, {y:.4f})")

    # Load eye PDM
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_left')

    print(f"\n=== Eye PDM Info ===")
    print(f"n_points: {pdm.n_points}")
    print(f"n_modes: {pdm.n_modes}")
    print(f"n_params: {pdm.n_params}")

    # Eye indices in 28-point model that correspond to the 6 visible landmarks
    eye_indices = list(LEFT_EYE_MAPPING.values())  # [8, 10, 12, 14, 16, 18]
    print(f"\nEye indices (28-pt model): {eye_indices}")

    # Step 1: Compute bounding box scaling (like C++ CalcParams)
    # Get mean shape for visible landmarks
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X_all = mean_flat[:n]
    Y_all = mean_flat[n:2*n]

    mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])

    print(f"\n=== Mean Shape (visible landmarks) ===")
    for i, idx in enumerate(eye_indices):
        print(f"Eye_{idx}: ({mean_2d[i, 0]:.6f}, {mean_2d[i, 1]:.6f})")

    # Compute bounding boxes
    target_min_x, target_max_x = np.min(cpp_left_eye[:, 0]), np.max(cpp_left_eye[:, 0])
    target_min_y, target_max_y = np.min(cpp_left_eye[:, 1]), np.max(cpp_left_eye[:, 1])
    target_width = target_max_x - target_min_x
    target_height = target_max_y - target_min_y

    mean_min_x, mean_max_x = np.min(mean_2d[:, 0]), np.max(mean_2d[:, 0])
    mean_min_y, mean_max_y = np.min(mean_2d[:, 1]), np.max(mean_2d[:, 1])
    mean_width = mean_max_x - mean_min_x
    mean_height = mean_max_y - mean_min_y

    print(f"\n=== Bounding Box Comparison ===")
    print(f"Target: width={target_width:.4f}, height={target_height:.4f}")
    print(f"Mean:   width={mean_width:.6f}, height={mean_height:.6f}")

    # Scale
    scale = ((target_width / mean_width) + (target_height / mean_height)) / 2.0
    print(f"\nScale = ((target_w/mean_w) + (target_h/mean_h)) / 2")
    print(f"Scale = (({target_width:.4f}/{mean_width:.6f}) + ({target_height:.4f}/{mean_height:.6f})) / 2")
    print(f"Scale = ({target_width/mean_width:.6f} + {target_height/mean_height:.6f}) / 2")
    print(f"Scale = {scale:.6f}")

    # Translation
    tx = (target_min_x + target_max_x) / 2.0
    ty = (target_min_y + target_max_y) / 2.0
    print(f"\nTranslation: tx={tx:.4f}, ty={ty:.4f}")

    # Initial params
    params = np.zeros(pdm.n_params)
    params[0] = scale
    params[4] = tx
    params[5] = ty

    print(f"\n=== Initial Params (before optimization) ===")
    print(f"scale={params[0]:.6f}")
    print(f"rot=({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
    print(f"trans=({params[4]:.6f}, {params[5]:.6f})")
    print(f"shape params: {params[6:]}")

    # Project back to 2D
    landmarks_2d = pdm.params_to_landmarks_2d(params)

    print(f"\n=== Projected Landmarks (initial params) ===")
    for idx in eye_indices:
        print(f"Eye_{idx}: ({landmarks_2d[idx, 0]:.4f}, {landmarks_2d[idx, 1]:.4f})")

    print(f"\nEye_8 (our target): ({landmarks_2d[8, 0]:.4f}, {landmarks_2d[8, 1]:.4f})")

    # Compute error before optimization
    target_subset = cpp_left_eye
    projected_subset = landmarks_2d[eye_indices]
    error_before = np.mean(np.linalg.norm(projected_subset - target_subset, axis=1))
    print(f"Mean error before optimization: {error_before:.4f} pixels")

    # Now run the actual fitting from eye_patch_expert
    from pyclnf.core.eye_patch_expert import HierarchicalEyeModel

    eye_model = HierarchicalEyeModel('pyclnf/models')

    # Create fake main landmarks (68 points) with only left eye filled
    main_landmarks = np.zeros((68, 2))
    for i, main_idx in enumerate(sorted(LEFT_EYE_MAPPING.keys())):
        main_landmarks[main_idx] = cpp_left_eye[i]

    # Call _fit_eye_shape directly
    fitted_params = eye_model._fit_eye_shape(cpp_left_eye, LEFT_EYE_MAPPING, 'left')

    if fitted_params is not None:
        print(f"\n=== Fitted Params (after optimization) ===")
        print(f"scale={fitted_params[0]:.6f}")
        print(f"rot=({fitted_params[1]:.6f}, {fitted_params[2]:.6f}, {fitted_params[3]:.6f})")
        print(f"trans=({fitted_params[4]:.6f}, {fitted_params[5]:.6f})")
        print(f"shape params: {fitted_params[6:]}")

        # Project fitted params
        fitted_landmarks = pdm.params_to_landmarks_2d(fitted_params)

        print(f"\n=== Projected Landmarks (fitted params) ===")
        for idx in eye_indices:
            print(f"Eye_{idx}: ({fitted_landmarks[idx, 0]:.4f}, {fitted_landmarks[idx, 1]:.4f})")

        print(f"\nEye_8 (our target): ({fitted_landmarks[8, 0]:.4f}, {fitted_landmarks[8, 1]:.4f})")
        print(f"C++ LM36 input:     (399.5000, 827.5000)")
        print(f"Difference: ({fitted_landmarks[8, 0] - 399.5:.4f}, {fitted_landmarks[8, 1] - 827.5:.4f})")

        # Compute error after optimization
        fitted_subset = fitted_landmarks[eye_indices]
        error_after = np.mean(np.linalg.norm(fitted_subset - target_subset, axis=1))
        print(f"\nMean error after optimization: {error_after:.4f} pixels")

if __name__ == '__main__':
    main()
