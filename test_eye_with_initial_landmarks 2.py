#!/usr/bin/env python3
"""
Test Python eye refinement using C++ INITIAL (pre-refinement) landmarks.
This is the proper apples-to-apples comparison.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING

def main():
    # C++ initial (pre-refinement) eye landmarks
    cpp_initial = {
        36: np.array([399.954468, 824.986633]),  # Eye_8
        37: np.array([417.933777, 806.527588]),  # Eye_10
        38: np.array([445.997620, 804.549561]),  # Eye_12
        39: np.array([469.824554, 821.693481]),  # Eye_14
        40: np.array([446.415588, 828.713074]),  # Eye_16
        41: np.array([419.007080, 830.748474]),  # Eye_18
    }
    
    # C++ final (post-refinement) landmarks
    cpp_final = {
        36: np.array([399.5, 827.5]),
        37: np.array([417.9, 809.4]),
        38: np.array([444.1, 807.6]),
        39: np.array([467.8, 822.7]),
        40: np.array([444.8, 830.1]),
        41: np.array([418.7, 833.1]),
    }
    
    # Load image
    image = cv2.imread('comparison_frame_0000.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create main landmarks array with initial positions
    main_landmarks = np.zeros((68, 2))
    for idx, pos in cpp_initial.items():
        main_landmarks[idx] = pos
    
    # Load eye model
    eye_model = HierarchicalEyeModel('pyclnf/models')
    
    # Run eye refinement
    print("=== Testing Python Eye Refinement with C++ Initial Landmarks ===\n")
    
    print("Input (C++ initial/pre-refinement):")
    for idx in sorted(cpp_initial.keys()):
        print(f"  LM{idx}: ({cpp_initial[idx][0]:.4f}, {cpp_initial[idx][1]:.4f})")
    
    # Refine left eye
    refined_landmarks = eye_model.refine_eye_landmarks(
        gray,
        main_landmarks.copy(),
        'left',
        np.eye(3),  # identity rotation
        1.0  # scale
    )
    
    print("\nPython output (after refinement):")
    for idx in sorted(cpp_initial.keys()):
        print(f"  LM{idx}: ({refined_landmarks[idx, 0]:.4f}, {refined_landmarks[idx, 1]:.4f})")
    
    print("\nC++ output (final/post-refinement):")
    for idx in sorted(cpp_final.keys()):
        print(f"  LM{idx}: ({cpp_final[idx][0]:.4f}, {cpp_final[idx][1]:.4f})")
    
    # Compare errors
    print("\n=== Error Comparison ===")
    print("\nPython vs C++ final:")
    errors_py_cpp = []
    for idx in sorted(cpp_final.keys()):
        diff = refined_landmarks[idx] - cpp_final[idx]
        error = np.linalg.norm(diff)
        errors_py_cpp.append(error)
        print(f"  LM{idx}: diff=({diff[0]:+.2f}, {diff[1]:+.2f}) err={error:.2f}px")
    
    print(f"\nMean error (Python vs C++ final): {np.mean(errors_py_cpp):.3f}px")
    
    # Also show how much each moved
    print("\n=== Movement Analysis ===")
    print("\nC++ refinement moved landmarks by:")
    for idx in sorted(cpp_initial.keys()):
        diff = cpp_final[idx] - cpp_initial[idx]
        print(f"  LM{idx}: ({diff[0]:+.2f}, {diff[1]:+.2f})")
    
    print("\nPython refinement moved landmarks by:")
    for idx in sorted(cpp_initial.keys()):
        diff = refined_landmarks[idx] - cpp_initial[idx]
        print(f"  LM{idx}: ({diff[0]:+.2f}, {diff[1]:+.2f})")

if __name__ == '__main__':
    main()
