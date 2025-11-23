#!/usr/bin/env python3
"""
Test the updated _fit_eye_shape with mean center correction and 1500 iterations.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING, RIGHT_EYE_MAPPING

def main():
    print("=" * 70)
    print("TESTING UPDATED CalcParams (mean center correction + 1500 iterations)")
    print("=" * 70)

    # Load eye model
    eye_model = HierarchicalEyeModel('pyclnf/models')

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

    # Test LEFT eye
    left_target = np.array([CPP_LEFT_EYE[i] for i in [36, 37, 38, 39, 40, 41]])
    left_params = eye_model._fit_eye_shape(left_target, LEFT_EYE_MAPPING, 'left', main_rotation=None)

    print("\n--- LEFT EYE ---")
    print(f"Python: scale={left_params[0]:.6f}, rot=({left_params[1]:.6f},{left_params[2]:.6f},{left_params[3]:.6f})")
    print(f"        tx={left_params[4]:.6f}, ty={left_params[5]:.6f}")
    print(f"C++:    scale={cpp_left['scale']:.6f}, rot=({cpp_left['rot'][0]:.6f},{cpp_left['rot'][1]:.6f},{cpp_left['rot'][2]:.6f})")
    print(f"        tx={cpp_left['tx']:.6f}, ty={cpp_left['ty']:.6f}")
    print(f"Diff:   tx={abs(left_params[4]-cpp_left['tx']):.4f}, ty={abs(left_params[5]-cpp_left['ty']):.4f}")

    # Compute 6-landmark error
    left_pdm = eye_model.pdm['left']
    left_lm = left_pdm.params_to_landmarks_2d(left_params)
    eye_indices = [8, 10, 12, 14, 16, 18]
    left_visible = left_lm[eye_indices]
    left_error = np.sqrt(sum((left_target[i, 0]-left_visible[i, 0])**2 +
                             (left_target[i, 1]-left_visible[i, 1])**2
                             for i in range(6)))
    print(f"6-landmark error: {left_error:.4f}px")

    # Test RIGHT eye - try with C++ rotation as initial guess
    right_target = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])
    cpp_rotation = np.array([-0.220360, -0.104840, -0.101141])
    right_params = eye_model._fit_eye_shape(right_target, RIGHT_EYE_MAPPING, 'right', main_rotation=cpp_rotation)

    print("\n--- RIGHT EYE ---")
    print(f"Python: scale={right_params[0]:.6f}, rot=({right_params[1]:.6f},{right_params[2]:.6f},{right_params[3]:.6f})")
    print(f"        tx={right_params[4]:.6f}, ty={right_params[5]:.6f}")
    print(f"C++:    scale={cpp_right['scale']:.6f}, rot=({cpp_right['rot'][0]:.6f},{cpp_right['rot'][1]:.6f},{cpp_right['rot'][2]:.6f})")
    print(f"        tx={cpp_right['tx']:.6f}, ty={cpp_right['ty']:.6f}")
    print(f"Diff:   tx={abs(right_params[4]-cpp_right['tx']):.4f}, ty={abs(right_params[5]-cpp_right['ty']):.4f}")

    # Compute 6-landmark error
    right_pdm = eye_model.pdm['right']
    right_lm = right_pdm.params_to_landmarks_2d(right_params)
    right_visible = right_lm[eye_indices]
    right_error = np.sqrt(sum((right_target[i, 0]-right_visible[i, 0])**2 +
                              (right_target[i, 1]-right_visible[i, 1])**2
                              for i in range(6)))
    print(f"6-landmark error: {right_error:.4f}px")

    # Print shape parameters
    print("\n--- Shape Parameters ---")
    print(f"Left shape params: {left_params[6:11]}")
    print(f"Right shape params: {right_params[6:11]}")

if __name__ == '__main__':
    main()
