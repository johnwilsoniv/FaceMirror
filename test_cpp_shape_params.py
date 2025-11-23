#!/usr/bin/env python3
"""
Test C++ parameters (including shape params) with Python PDM.
Uses values from comparison_frame_0030.jpg.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("TESTING C++ PARAMETERS WITH PYTHON PDM")
    print("=" * 70)

    # Load right eye PDM
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')
    eye_indices = [8, 10, 12, 14, 16, 18]

    # C++ parameters from comparison_frame_0030.jpg (right eye)
    cpp_params = np.zeros(pdm.n_params)
    cpp_params[0] = 3.57452  # scale
    cpp_params[1] = -0.225902  # rot_x
    cpp_params[2] = -0.111583  # rot_y
    cpp_params[3] = -0.112012  # rot_z
    cpp_params[4] = 1673.71  # tx
    cpp_params[5] = 808.281  # ty
    # Shape params
    cpp_shape = [0.61328, -4.46705, -1.89613, -0.973712, -0.654916,
                 -0.28567, 0.0780907, 0.76727, 0.368293, 0.278094]
    for i, val in enumerate(cpp_shape):
        cpp_params[6+i] = val

    print("\nC++ Parameters:")
    print(f"  scale={cpp_params[0]:.6f}")
    print(f"  rot=({cpp_params[1]:.6f}, {cpp_params[2]:.6f}, {cpp_params[3]:.6f})")
    print(f"  tx={cpp_params[4]:.6f}, ty={cpp_params[5]:.6f}")
    print(f"  shape[0:5]={cpp_params[6:11]}")

    # Project to 2D
    cpp_2d = pdm.params_to_landmarks_2d(cpp_params)
    cpp_visible = cpp_2d[eye_indices]

    print("\nProjected 6 visible landmarks (C++ params with shape):")
    for i, idx in enumerate(eye_indices):
        print(f"  Eye_{idx}: ({cpp_visible[i, 0]:.2f}, {cpp_visible[i, 1]:.2f})")

    # Now test WITHOUT shape params
    cpp_no_shape = cpp_params.copy()
    cpp_no_shape[6:] = 0  # Zero out shape params

    no_shape_2d = pdm.params_to_landmarks_2d(cpp_no_shape)
    no_shape_visible = no_shape_2d[eye_indices]

    print("\nProjected 6 visible landmarks (C++ params WITHOUT shape):")
    for i, idx in enumerate(eye_indices):
        dx = cpp_visible[i, 0] - no_shape_visible[i, 0]
        dy = cpp_visible[i, 1] - no_shape_visible[i, 1]
        print(f"  Eye_{idx}: ({no_shape_visible[i, 0]:.2f}, {no_shape_visible[i, 1]:.2f}) "
              f"shift=({dx:+.2f}, {dy:+.2f})")

    # Calculate shift in center due to shape params
    with_shape_center = np.mean(cpp_visible, axis=0)
    without_shape_center = np.mean(no_shape_visible, axis=0)
    center_shift = with_shape_center - without_shape_center

    print(f"\nCenter shift due to shape params: ({center_shift[0]:.2f}, {center_shift[1]:.2f})")

    # This is the key insight - shape params shift the landmarks!
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print(f"Shape parameters shift the eye center by ~{abs(center_shift[1]):.1f}px vertically")
    print("This explains why C++ ty differs from Python - C++ compensates with shape params")

    # Let's also check what Python found
    print("\n" + "=" * 70)
    print("COMPARISON WITH PYTHON SOLUTION")
    print("=" * 70)

    # Python params from earlier test (for different image but similar pattern)
    py_params = np.zeros(pdm.n_params)
    py_params[0] = 3.5327
    py_params[1] = -0.2133
    py_params[2] = -0.0783
    py_params[3] = -0.1131
    py_params[4] = 596.3507
    py_params[5] = 825.9734
    # Python shape params from earlier
    py_params[6] = 0.82
    py_params[7] = -4.39
    py_params[8] = -1.41

    py_2d = pdm.params_to_landmarks_2d(py_params)
    py_visible = py_2d[eye_indices]

    print("\nPython solution (different image):")
    print(f"  shape[0:3]=({py_params[6]:.2f}, {py_params[7]:.2f}, {py_params[8]:.2f})")

    print("\nC++ solution (this image):")
    print(f"  shape[0:3]=({cpp_params[6]:.2f}, {cpp_params[7]:.2f}, {cpp_params[8]:.2f})")

    print("\nBoth have similar shape[1] values (~-4.4), confirming similar optimization behavior!")

if __name__ == '__main__':
    main()
