#!/usr/bin/env python3
"""
Compare C++ and Python CalcParams solutions for right eye.
Check if C++ solution fits the 6 landmarks better.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("COMPARING C++ vs PYTHON SOLUTIONS FOR RIGHT EYE")
    print("=" * 70)

    # Load right eye PDM
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')

    # Eye indices for 6 visible landmarks
    eye_indices = [8, 10, 12, 14, 16, 18]

    # Target landmarks
    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062),
        43: (583.5312, 812.9091),
        44: (611.5075, 810.8795),
        45: (635.8052, 825.4882),
        46: (615.4611, 834.7896),
        47: (587.8980, 837.3765)
    }

    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    # Python solution (from our CalcParams)
    python_params = np.zeros(pdm.n_params)
    python_params[0] = 3.532667  # scale
    python_params[1] = -0.213260  # rx
    python_params[2] = -0.078328  # ry
    python_params[3] = -0.113085  # rz
    python_params[4] = 596.350656  # tx
    python_params[5] = 825.973357  # ty

    # C++ solution
    cpp_params = np.zeros(pdm.n_params)
    cpp_params[0] = 3.533595  # scale
    cpp_params[1] = -0.220360  # rx
    cpp_params[2] = -0.104840  # ry
    cpp_params[3] = -0.101141  # rz
    cpp_params[4] = 602.636292  # tx
    cpp_params[5] = 807.187805  # ty

    # Get 2D landmarks for both solutions
    python_lm = pdm.params_to_landmarks_2d(python_params)
    cpp_lm = pdm.params_to_landmarks_2d(cpp_params)

    python_visible = python_lm[eye_indices]
    cpp_visible = cpp_lm[eye_indices]

    # Compute errors
    python_errors = []
    cpp_errors = []

    print("\n--- Visible Landmark Comparison ---")
    print(f"{'Lm':>4} {'Target':>20} {'Python':>20} {'C++':>20} {'Py Err':>10} {'C++ Err':>10}")

    for i, idx in enumerate(eye_indices):
        target = target_points[i]
        py = python_visible[i]
        cpp = cpp_visible[i]

        py_err = np.sqrt((target[0]-py[0])**2 + (target[1]-py[1])**2)
        cpp_err = np.sqrt((target[0]-cpp[0])**2 + (target[1]-cpp[1])**2)

        python_errors.append(py_err)
        cpp_errors.append(cpp_err)

        print(f"{idx:>4} ({target[0]:>7.2f},{target[1]:>7.2f}) ({py[0]:>7.2f},{py[1]:>7.2f}) ({cpp[0]:>7.2f},{cpp[1]:>7.2f}) {py_err:>10.4f} {cpp_err:>10.4f}")

    print(f"\nTotal error: Python={np.sqrt(sum(e**2 for e in python_errors)):.4f}, C++={np.sqrt(sum(e**2 for e in cpp_errors)):.4f}")
    print(f"Mean error:  Python={np.mean(python_errors):.4f}, C++={np.mean(cpp_errors):.4f}")

    # Compare full 28 landmarks
    print("\n--- Full 28 Landmark Comparison ---")
    print("Landmarks where C++ and Python differ by more than 5px:")

    diffs = []
    for i in range(28):
        diff = np.sqrt((python_lm[i, 0] - cpp_lm[i, 0])**2 +
                       (python_lm[i, 1] - cpp_lm[i, 1])**2)
        diffs.append(diff)
        if diff > 5:
            print(f"  Eye_{i:2d}: Python=({python_lm[i, 0]:>7.2f},{python_lm[i, 1]:>7.2f}) "
                  f"C++=({cpp_lm[i, 0]:>7.2f},{cpp_lm[i, 1]:>7.2f}) diff={diff:.2f}px")

    print(f"\nMax diff: {max(diffs):.2f}px at Eye_{np.argmax(diffs)}")
    print(f"Mean diff: {np.mean(diffs):.2f}px")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Python 6-landmark error: {np.sqrt(sum(e**2 for e in python_errors)):.4f}px")
    print(f"C++ 6-landmark error:    {np.sqrt(sum(e**2 for e in cpp_errors)):.4f}px")
    print(f"28-landmark mean diff:   {np.mean(diffs):.2f}px")

    if np.sqrt(sum(e**2 for e in cpp_errors)) < np.sqrt(sum(e**2 for e in python_errors)):
        print("\n*** C++ solution fits 6 landmarks better ***")
    else:
        print("\n*** Python solution fits 6 landmarks better ***")

if __name__ == '__main__':
    main()
