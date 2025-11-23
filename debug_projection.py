#!/usr/bin/env python3
"""
Debug projection formula - verify Python matches C++.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_pdm import EyePDM

def euler_to_rotation_matrix(euler):
    """Python rotation matrix (should match C++)."""
    s1, s2, s3 = np.sin(euler[0]), np.sin(euler[1]), np.sin(euler[2])
    c1, c2, c3 = np.cos(euler[0]), np.cos(euler[1]), np.cos(euler[2])

    R = np.array([
        [c2 * c3,              -c2 * s3,             s2],
        [c1 * s3 + c3 * s1 * s2,  c1 * c3 - s1 * s2 * s3,  -c2 * s1],
        [s1 * s3 - c1 * c3 * s2,  c3 * s1 + c1 * s2 * s3,   c1 * c2]
    ], dtype=np.float32)

    return R

def project_cpp_style(X, Y, Z, R, s, tx, ty):
    """Project 3D point to 2D using C++ formula."""
    x_2d = s * (R[0,0]*X + R[0,1]*Y + R[0,2]*Z) + tx
    y_2d = s * (R[1,0]*X + R[1,1]*Y + R[1,2]*Z) + ty
    return x_2d, y_2d

def project_python_style(X, Y, Z, R, s, tx, ty):
    """Project 3D point to 2D using Python formula (shape @ R.T)."""
    point = np.array([X, Y, Z])
    rotated = point @ R.T  # This should equal R @ point
    x_2d = s * rotated[0] + tx
    y_2d = s * rotated[1] + ty
    return x_2d, y_2d

def main():
    print("=" * 70)
    print("PROJECTION FORMULA DEBUG")
    print("=" * 70)

    # Test with C++ params
    s = 3.533595
    rx, ry, rz = -0.220360, -0.104840, -0.101141
    tx, ty = 602.636292, 807.187805

    R = euler_to_rotation_matrix(np.array([rx, ry, rz]))

    print("Rotation matrix R:")
    print(R)

    # Load PDM and get mean shape point 8
    pdm = EyePDM('pyclnf/models/exported_eye_pdm_right')
    mean_flat = pdm.mean_shape.flatten()
    n = pdm.n_points
    X = mean_flat[8]
    Y = mean_flat[8 + n]
    Z = mean_flat[8 + 2*n]

    print(f"\nPoint 8 in mean shape: X={X:.4f}, Y={Y:.4f}, Z={Z:.4f}")

    # Project using both methods
    cpp_x, cpp_y = project_cpp_style(X, Y, Z, R, s, tx, ty)
    py_x, py_y = project_python_style(X, Y, Z, R, s, tx, ty)

    print(f"\nC++ style projection: ({cpp_x:.4f}, {cpp_y:.4f})")
    print(f"Python style projection: ({py_x:.4f}, {py_y:.4f})")

    # Also verify shape @ R.T equals R @ shape for vectors
    point = np.array([X, Y, Z])
    method1 = point @ R.T
    method2 = R @ point

    print(f"\nVerifying rotation:")
    print(f"point @ R.T = {method1}")
    print(f"R @ point   = {method2}")
    print(f"Match: {np.allclose(method1, method2)}")

    # Now let's check what the PDM produces
    cpp_params = np.zeros(pdm.n_params)
    cpp_params[0] = s
    cpp_params[1] = rx
    cpp_params[2] = ry
    cpp_params[3] = rz
    cpp_params[4] = tx
    cpp_params[5] = ty

    pdm_lm = pdm.params_to_landmarks_2d(cpp_params)
    print(f"\nPDM produces for point 8: ({pdm_lm[8, 0]:.4f}, {pdm_lm[8, 1]:.4f})")

    # Expected target
    print(f"Target: (560.8530, 833.9062)")

if __name__ == '__main__':
    main()
