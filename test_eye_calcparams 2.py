#!/usr/bin/env python3
"""Test CalcParams for eye model against C++ results."""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

# C++ debug values for left eye
cpp_input_landmarks = {
    8: (392.1590, 847.6613),
    10: (410.0039, 828.3166),
    12: (436.9223, 826.1841),
    14: (461.9583, 842.8420),
    16: (438.4380, 850.4288),
    18: (411.4089, 853.9998)
}

cpp_fitted_params = {
    'scale': 3.362953,
    'rot_x': -0.103207,
    'rot_y': 0.175851,
    'rot_z': -0.114345,
    'tx': 426.041504,
    'ty': 839.989624
}

cpp_initial_landmarks = {
    8: (392.3565, 847.6952),
    10: (409.5100, 828.6759),
    12: (437.5647, 825.9781),
    14: (461.7177, 842.6740),
    16: (438.4697, 850.8372),
    18: (411.2716, 853.5720)
}

def main():
    print("=" * 60)
    print("EYE CALCPARAMS COMPARISON TEST")
    print("=" * 60)

    # Load Python eye PDM
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_left')

    print(f"\nPDM loaded: {pdm.n_points} points, {pdm.n_modes} modes")

    # Extract target points from C++ input
    eye_indices = [8, 10, 12, 14, 16, 18]
    target_points = np.array([cpp_input_landmarks[i] for i in eye_indices])

    print("\nInput landmarks from C++ (6 points):")
    for i, idx in enumerate(eye_indices):
        print(f"  {idx}: ({target_points[i, 0]:.4f}, {target_points[i, 1]:.4f})")

    # Get mean shape for the 6 visible landmarks
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X_all = mean_flat[:n]
    Y_all = mean_flat[n:2*n]

    mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])

    print("\nMean shape (6 visible landmarks):")
    for i, idx in enumerate(eye_indices):
        print(f"  {idx}: ({mean_2d[i, 0]:.4f}, {mean_2d[i, 1]:.4f})")

    # Compute bounding boxes
    target_min_x = np.min(target_points[:, 0])
    target_max_x = np.max(target_points[:, 0])
    target_min_y = np.min(target_points[:, 1])
    target_max_y = np.max(target_points[:, 1])
    target_width = target_max_x - target_min_x
    target_height = target_max_y - target_min_y

    mean_min_x = np.min(mean_2d[:, 0])
    mean_max_x = np.max(mean_2d[:, 0])
    mean_min_y = np.min(mean_2d[:, 1])
    mean_max_y = np.max(mean_2d[:, 1])
    mean_width = mean_max_x - mean_min_x
    mean_height = mean_max_y - mean_min_y

    print(f"\nTarget bbox: x=[{target_min_x:.2f}, {target_max_x:.2f}], y=[{target_min_y:.2f}, {target_max_y:.2f}]")
    print(f"  width={target_width:.4f}, height={target_height:.4f}")
    print(f"Mean bbox: x=[{mean_min_x:.4f}, {mean_max_x:.4f}], y=[{mean_min_y:.4f}, {mean_max_y:.4f}]")
    print(f"  width={mean_width:.4f}, height={mean_height:.4f}")

    # Compute scale
    scale_x = target_width / mean_width if mean_width > 0 else 1.0
    scale_y = target_height / mean_height if mean_height > 0 else 1.0
    scale = (scale_x + scale_y) / 2.0

    print(f"\nScale calculation:")
    print(f"  scale_x = {target_width:.4f} / {mean_width:.4f} = {scale_x:.6f}")
    print(f"  scale_y = {target_height:.4f} / {mean_height:.4f} = {scale_y:.6f}")
    print(f"  scale = {scale:.6f}")

    # Compute translation
    target_center_x = (target_min_x + target_max_x) / 2.0
    target_center_y = (target_min_y + target_max_y) / 2.0
    mean_center_x = (mean_min_x + mean_max_x) / 2.0
    mean_center_y = (mean_min_y + mean_max_y) / 2.0

    tx = target_center_x - mean_center_x * scale
    ty = target_center_y - mean_center_y * scale

    print(f"\nTranslation calculation:")
    print(f"  target_center = ({target_center_x:.4f}, {target_center_y:.4f})")
    print(f"  mean_center = ({mean_center_x:.4f}, {mean_center_y:.4f})")
    print(f"  tx = {target_center_x:.4f} - {mean_center_x:.4f} * {scale:.4f} = {tx:.6f}")
    print(f"  ty = {target_center_y:.4f} - {mean_center_y:.4f} * {scale:.4f} = {ty:.6f}")

    # Initial params
    params = np.zeros(6 + pdm.n_modes)
    params[0] = scale
    params[4] = tx
    params[5] = ty

    print(f"\n" + "=" * 60)
    print("COMPARISON WITH C++")
    print("=" * 60)
    print(f"\nPython CalcParams result:")
    print(f"  scale: {scale:.6f}")
    print(f"  rot: (0, 0, 0)")
    print(f"  tx, ty: ({tx:.6f}, {ty:.6f})")

    print(f"\nC++ CalcParams result:")
    print(f"  scale: {cpp_fitted_params['scale']:.6f}")
    print(f"  rot: ({cpp_fitted_params['rot_x']:.6f}, {cpp_fitted_params['rot_y']:.6f}, {cpp_fitted_params['rot_z']:.6f})")
    print(f"  tx, ty: ({cpp_fitted_params['tx']:.6f}, {cpp_fitted_params['ty']:.6f})")

    print(f"\nDifferences:")
    print(f"  scale: {abs(scale - cpp_fitted_params['scale']):.6f}")
    print(f"  tx: {abs(tx - cpp_fitted_params['tx']):.6f}")
    print(f"  ty: {abs(ty - cpp_fitted_params['ty']):.6f}")

    # Compute landmarks with Python's CalcParams result
    py_landmarks = pdm.params_to_landmarks_2d(params)

    print(f"\nPython landmarks at visible indices:")
    for idx in eye_indices:
        py = py_landmarks[idx]
        cpp = cpp_initial_landmarks[idx]
        diff = np.sqrt((py[0] - cpp[0])**2 + (py[1] - cpp[1])**2)
        print(f"  {idx}: Py=({py[0]:.4f}, {py[1]:.4f}), C++=({cpp[0]:.4f}, {cpp[1]:.4f}), diff={diff:.4f}")

if __name__ == '__main__':
    main()
