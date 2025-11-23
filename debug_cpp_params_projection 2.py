#!/usr/bin/env python3
"""
Check what landmarks C++ params produce with Python PDM.
This will reveal if the PDM model itself is different.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("C++ PARAMETERS WITH PYTHON PDM")
    print("=" * 70)

    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')
    eye_indices = [8, 10, 12, 14, 16, 18]

    # Target landmarks
    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062), 43: (583.5312, 812.9091),
        44: (611.5075, 810.8795), 45: (635.8052, 825.4882),
        46: (615.4611, 834.7896), 47: (587.8980, 837.3765)
    }
    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    # C++ parameters (global only, no shape)
    cpp_params = np.zeros(pdm.n_params)
    cpp_params[0] = 3.533595  # scale
    cpp_params[1] = -0.220360  # rot_x
    cpp_params[2] = -0.104840  # rot_y
    cpp_params[3] = -0.101141  # rot_z
    cpp_params[4] = 602.636292  # tx
    cpp_params[5] = 807.187805  # ty

    # Project with C++ params
    cpp_2d = pdm.params_to_landmarks_2d(cpp_params)
    cpp_visible = cpp_2d[eye_indices]

    print("\nC++ params (no shape params):")
    print(f"scale={cpp_params[0]:.6f}")
    print(f"rot=({cpp_params[1]:.6f}, {cpp_params[2]:.6f}, {cpp_params[3]:.6f})")
    print(f"tx={cpp_params[4]:.6f}, ty={cpp_params[5]:.6f}")

    print("\nProjected landmarks vs targets:")
    total_error = 0
    for i, idx in enumerate(eye_indices):
        dx = target_points[i, 0] - cpp_visible[i, 0]
        dy = target_points[i, 1] - cpp_visible[i, 1]
        err = np.sqrt(dx*dx + dy*dy)
        total_error += err*err
        print(f"  Eye_{idx}: target=({target_points[i,0]:.2f}, {target_points[i,1]:.2f}) "
              f"proj=({cpp_visible[i,0]:.2f}, {cpp_visible[i,1]:.2f}) "
              f"err=({dx:.2f}, {dy:.2f})")

    print(f"\nTotal error (L2): {np.sqrt(total_error):.4f}px")

    # What if the C++ mean shape is different?
    # Let's compute what translation would be needed to match
    print("\n" + "=" * 70)
    print("ANALYZING TRANSLATION DIFFERENCE")
    print("=" * 70)

    # Current center with C++ params
    cpp_center_x = np.mean(cpp_visible[:, 0])
    cpp_center_y = np.mean(cpp_visible[:, 1])

    # Target center
    target_center_x = np.mean(target_points[:, 0])
    target_center_y = np.mean(target_points[:, 1])

    print(f"\nC++ projection center: ({cpp_center_x:.4f}, {cpp_center_y:.4f})")
    print(f"Target center: ({target_center_x:.4f}, {target_center_y:.4f})")
    print(f"Difference: ({target_center_x-cpp_center_x:.4f}, {target_center_y-cpp_center_y:.4f})")

    # What tx, ty would we need?
    adjusted_tx = cpp_params[4] + (target_center_x - cpp_center_x)
    adjusted_ty = cpp_params[5] + (target_center_y - cpp_center_y)
    print(f"\nIf we adjust tx/ty to match centers:")
    print(f"  adjusted_tx={adjusted_tx:.4f}, adjusted_ty={adjusted_ty:.4f}")

    # Test adjusted params
    adj_params = cpp_params.copy()
    adj_params[4] = adjusted_tx
    adj_params[5] = adjusted_ty

    adj_2d = pdm.params_to_landmarks_2d(adj_params)
    adj_visible = adj_2d[eye_indices]

    print("\nAdjusted projection vs targets:")
    total_error = 0
    for i, idx in enumerate(eye_indices):
        dx = target_points[i, 0] - adj_visible[i, 0]
        dy = target_points[i, 1] - adj_visible[i, 1]
        err = np.sqrt(dx*dx + dy*dy)
        total_error += err*err
        print(f"  Eye_{idx}: target=({target_points[i,0]:.2f}, {target_points[i,1]:.2f}) "
              f"proj=({adj_visible[i,0]:.2f}, {adj_visible[i,1]:.2f}) "
              f"err=({dx:.2f}, {dy:.2f})")

    print(f"\nAdjusted total error (L2): {np.sqrt(total_error):.4f}px")

    # Let's also check what Python converges to
    print("\n" + "=" * 70)
    print("PYTHON CONVERGED SOLUTION")
    print("=" * 70)

    # Python params (from earlier test)
    py_params = np.zeros(pdm.n_params)
    py_params[0] = 3.5327
    py_params[1] = -0.2133
    py_params[2] = -0.0783
    py_params[3] = -0.1131
    py_params[4] = 596.3507
    py_params[5] = 825.9734

    py_2d = pdm.params_to_landmarks_2d(py_params)
    py_visible = py_2d[eye_indices]

    print(f"\nPython params:")
    print(f"scale={py_params[0]:.6f}")
    print(f"rot=({py_params[1]:.6f}, {py_params[2]:.6f}, {py_params[3]:.6f})")
    print(f"tx={py_params[4]:.6f}, ty={py_params[5]:.6f}")

    print("\nPython projection vs targets:")
    total_error = 0
    for i, idx in enumerate(eye_indices):
        dx = target_points[i, 0] - py_visible[i, 0]
        dy = target_points[i, 1] - py_visible[i, 1]
        err = np.sqrt(dx*dx + dy*dy)
        total_error += err*err
        print(f"  Eye_{idx}: target=({target_points[i,0]:.2f}, {target_points[i,1]:.2f}) "
              f"proj=({py_visible[i,0]:.2f}, {py_visible[i,1]:.2f}) "
              f"err=({dx:.2f}, {dy:.2f})")

    print(f"\nPython total error (L2): {np.sqrt(total_error):.4f}px")
    print(f"Python center: ({np.mean(py_visible[:,0]):.4f}, {np.mean(py_visible[:,1]):.4f})")

if __name__ == '__main__':
    main()
