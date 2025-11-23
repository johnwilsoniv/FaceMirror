#!/usr/bin/env python3
"""
Debug bounding box center calculation for eye CalcParams.
Tests hypothesis that mean shape center offset is causing the 19px error.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("BOUNDING BOX CENTER DEBUG")
    print("=" * 70)

    # Load both eye PDMs
    left_pdm = EyePDM('pyclnf/models/exported_eye_pdm_left')
    right_pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')

    # Eye indices for 6 visible landmarks
    eye_indices = [8, 10, 12, 14, 16, 18]

    # C++ target landmarks
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

    # C++ final params
    cpp_left_params = {
        'scale': 3.362953,
        'tx': 427.058319,
        'ty': 839.989624
    }

    cpp_right_params = {
        'scale': 3.533595,
        'tx': 602.636292,
        'ty': 807.187805
    }

    for side, pdm, target_dict, cpp_params in [
        ('LEFT', left_pdm, CPP_LEFT_EYE, cpp_left_params),
        ('RIGHT', right_pdm, CPP_RIGHT_EYE, cpp_right_params)
    ]:
        print(f"\n{'='*70}")
        print(f"{side} EYE ANALYSIS")
        print("=" * 70)

        # Get target points
        target_points = np.array([target_dict[i] for i in sorted(target_dict.keys())])

        # Get mean shape for visible landmarks
        mean_flat = pdm.mean_shape.flatten()
        n = pdm.n_points
        X_all = mean_flat[:n]
        Y_all = mean_flat[n:2*n]
        mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])

        # Target bounding box
        target_min_x, target_max_x = np.min(target_points[:, 0]), np.max(target_points[:, 0])
        target_min_y, target_max_y = np.min(target_points[:, 1]), np.max(target_points[:, 1])
        target_width = target_max_x - target_min_x
        target_height = target_max_y - target_min_y
        target_center_x = (target_min_x + target_max_x) / 2.0
        target_center_y = (target_min_y + target_max_y) / 2.0

        # Mean bounding box
        mean_min_x, mean_max_x = np.min(mean_2d[:, 0]), np.max(mean_2d[:, 0])
        mean_min_y, mean_max_y = np.min(mean_2d[:, 1]), np.max(mean_2d[:, 1])
        mean_width = mean_max_x - mean_min_x
        mean_height = mean_max_y - mean_min_y
        mean_center_x = (mean_min_x + mean_max_x) / 2.0
        mean_center_y = (mean_min_y + mean_max_y) / 2.0

        print(f"\nMean shape center: ({mean_center_x:.4f}, {mean_center_y:.4f})")
        print(f"Mean shape range: X=[{mean_min_x:.4f}, {mean_max_x:.4f}], Y=[{mean_min_y:.4f}, {mean_max_y:.4f}]")
        print(f"Mean shape size: {mean_width:.4f} x {mean_height:.4f}")

        print(f"\nTarget center: ({target_center_x:.4f}, {target_center_y:.4f})")
        print(f"Target range: X=[{target_min_x:.4f}, {target_max_x:.4f}], Y=[{target_min_y:.4f}, {target_max_y:.4f}]")
        print(f"Target size: {target_width:.4f} x {target_height:.4f}")

        # Current (broken) calculation
        scale = ((target_width / mean_width) + (target_height / mean_height)) / 2.0
        tx_broken = target_center_x
        ty_broken = target_center_y

        # Fixed calculation - account for mean center offset
        tx_fixed = target_center_x - scale * mean_center_x
        ty_fixed = target_center_y - scale * mean_center_y

        print(f"\nScale: {scale:.6f}")
        print(f"\nBROKEN calculation (not accounting for mean center):")
        print(f"  tx = target_center_x = {tx_broken:.6f}")
        print(f"  ty = target_center_y = {ty_broken:.6f}")

        print(f"\nFIXED calculation (accounting for mean center):")
        print(f"  tx = target_center_x - scale * mean_center_x")
        print(f"     = {target_center_x:.4f} - {scale:.4f} * {mean_center_x:.4f}")
        print(f"     = {tx_fixed:.6f}")
        print(f"  ty = target_center_y - scale * mean_center_y")
        print(f"     = {target_center_y:.4f} - {scale:.4f} * {mean_center_y:.4f}")
        print(f"     = {ty_fixed:.6f}")

        print(f"\nC++ final params:")
        print(f"  scale: {cpp_params['scale']:.6f}")
        print(f"  tx: {cpp_params['tx']:.6f}")
        print(f"  ty: {cpp_params['ty']:.6f}")

        print(f"\nDifference from C++ (BROKEN):")
        print(f"  scale: {abs(scale - cpp_params['scale']):.6f}")
        print(f"  tx: {abs(tx_broken - cpp_params['tx']):.6f}")
        print(f"  ty: {abs(ty_broken - cpp_params['ty']):.6f}")

        print(f"\nDifference from C++ (FIXED):")
        print(f"  scale: {abs(scale - cpp_params['scale']):.6f}")
        print(f"  tx: {abs(tx_fixed - cpp_params['tx']):.6f}")
        print(f"  ty: {abs(ty_fixed - cpp_params['ty']):.6f}")

        # Show what the model would produce with fixed initial params
        print(f"\n--- Testing fixed initial params ---")
        params = np.zeros(pdm.n_params)
        params[0] = scale
        params[4] = tx_fixed
        params[5] = ty_fixed

        initial_lm = pdm.params_to_landmarks_2d(params)
        visible_lm = initial_lm[eye_indices]

        print(f"Visible landmarks with fixed initial (scale={scale:.4f}, tx={tx_fixed:.4f}, ty={ty_fixed:.4f}):")
        for i, idx in enumerate(eye_indices):
            target = target_points[i]
            fitted = visible_lm[i]
            diff = np.sqrt((target[0]-fitted[0])**2 + (target[1]-fitted[1])**2)
            print(f"  Eye_{idx}: target=({target[0]:.2f}, {target[1]:.2f}), "
                  f"fitted=({fitted[0]:.2f}, {fitted[1]:.2f}), diff={diff:.2f}px")

if __name__ == '__main__':
    main()
