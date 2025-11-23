#!/usr/bin/env python3
"""
Test the full eye refinement pipeline to verify landmark accuracy.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING, RIGHT_EYE_MAPPING

def main():
    print("=" * 70)
    print("FULL EYE REFINEMENT PIPELINE TEST")
    print("=" * 70)

    # Load eye model
    eye_model = HierarchicalEyeModel('pyclnf/models')

    # Target landmarks (6 visible per eye from C++ debug)
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

    # Convert to target arrays
    left_target = np.array([CPP_LEFT_EYE[i] for i in [36, 37, 38, 39, 40, 41]])
    right_target = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    # Fit left eye
    left_params = eye_model._fit_eye_shape(left_target, LEFT_EYE_MAPPING, 'left', main_rotation=None)

    # Get full 28 landmarks for left eye
    left_pdm = eye_model.pdm['left']
    left_lm = left_pdm.params_to_landmarks_2d(left_params)

    # Compute error on 6 visible landmarks
    eye_indices = [8, 10, 12, 14, 16, 18]
    left_visible = left_lm[eye_indices]

    print("\n--- LEFT EYE ---")
    print(f"Params: scale={left_params[0]:.4f}, rot=({left_params[1]:.4f},{left_params[2]:.4f},{left_params[3]:.4f})")
    print(f"        tx={left_params[4]:.4f}, ty={left_params[5]:.4f}")

    print("\n6 visible landmarks error:")
    total_err = 0
    for i, main_idx in enumerate([36, 37, 38, 39, 40, 41]):
        dx = left_target[i, 0] - left_visible[i, 0]
        dy = left_target[i, 1] - left_visible[i, 1]
        err = np.sqrt(dx*dx + dy*dy)
        total_err += err*err
        print(f"  {main_idx}: target=({left_target[i,0]:.2f},{left_target[i,1]:.2f}) "
              f"fit=({left_visible[i,0]:.2f},{left_visible[i,1]:.2f}) err={err:.2f}px")

    left_total = np.sqrt(total_err)
    print(f"Total L2 error: {left_total:.4f}px")

    # Fit right eye
    right_params = eye_model._fit_eye_shape(right_target, RIGHT_EYE_MAPPING, 'right', main_rotation=None)

    # Get full 28 landmarks for right eye
    right_pdm = eye_model.pdm['right']
    right_lm = right_pdm.params_to_landmarks_2d(right_params)
    right_visible = right_lm[eye_indices]

    print("\n--- RIGHT EYE ---")
    print(f"Params: scale={right_params[0]:.4f}, rot=({right_params[1]:.4f},{right_params[2]:.4f},{right_params[3]:.4f})")
    print(f"        tx={right_params[4]:.4f}, ty={right_params[5]:.4f}")

    print("\n6 visible landmarks error:")
    total_err = 0
    for i, main_idx in enumerate([42, 43, 44, 45, 46, 47]):
        dx = right_target[i, 0] - right_visible[i, 0]
        dy = right_target[i, 1] - right_visible[i, 1]
        err = np.sqrt(dx*dx + dy*dy)
        total_err += err*err
        print(f"  {main_idx}: target=({right_target[i,0]:.2f},{right_target[i,1]:.2f}) "
              f"fit=({right_visible[i,0]:.2f},{right_visible[i,1]:.2f}) err={err:.2f}px")

    right_total = np.sqrt(total_err)
    print(f"Total L2 error: {right_total:.4f}px")

    # Show full 28 landmarks for right eye
    print("\n--- RIGHT EYE FULL 28 LANDMARKS ---")
    print("(All points in the eye model)")
    for i in range(28):
        marker = " *" if i in eye_indices else ""
        print(f"  {i:2d}: ({right_lm[i,0]:8.2f}, {right_lm[i,1]:8.2f}){marker}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Left eye fit error:  {left_total:.4f}px")
    print(f"Right eye fit error: {right_total:.4f}px")
    print(f"Average per-landmark: {(left_total + right_total) / 12:.4f}px")

    # Check if this is acceptable
    if (left_total + right_total) / 12 < 0.5:
        print("\n✓ Excellent fit accuracy (<0.5px per landmark)")
    elif (left_total + right_total) / 12 < 1.0:
        print("\n✓ Good fit accuracy (<1.0px per landmark)")
    else:
        print("\n⚠ Moderate fit accuracy - may need investigation")

if __name__ == '__main__':
    main()
