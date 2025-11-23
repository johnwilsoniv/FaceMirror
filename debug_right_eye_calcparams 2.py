#!/usr/bin/env python3
"""
Debug right eye CalcParams to compare with C++.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, RIGHT_EYE_MAPPING

def main():
    print("=" * 70)
    print("RIGHT EYE CALCPARAMS DEBUG")
    print("=" * 70)

    # C++ right eye input landmarks (main landmarks 42-47)
    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062),
        43: (583.5312, 812.9091),
        44: (611.5075, 810.8795),
        45: (635.8052, 825.4882),
        46: (615.4611, 834.7896),
        47: (587.8980, 837.3765)
    }

    eye_model = HierarchicalEyeModel('pyclnf/models')
    pdm = eye_model.pdm['right']

    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])

    print(f"\nInput landmarks (6 points):")
    for i, (idx, lm) in enumerate(CPP_RIGHT_EYE.items()):
        print(f"  {idx} -> Eye_{RIGHT_EYE_MAPPING[idx]}: ({lm[0]:.4f}, {lm[1]:.4f})")

    # Fit eye shape
    params = eye_model._fit_eye_shape(target_points, RIGHT_EYE_MAPPING, 'right', main_rotation=None)

    # Extract global params
    scale = params[0]
    rot_x = params[1]
    rot_y = params[2]
    rot_z = params[3]
    tx = params[4]
    ty = params[5]

    print(f"\nPython fitted params_global:")
    print(f"  scale: {scale:.6f}")
    print(f"  rot: ({rot_x:.6f}, {rot_y:.6f}, {rot_z:.6f})")
    print(f"  tx, ty: ({tx:.6f}, {ty:.6f})")

    print(f"\nC++ right eye params_global (from debug):")
    print(f"  scale: 3.533595")
    print(f"  rot: (-0.220360, -0.104840, -0.101141)")
    print(f"  tx, ty: (602.636292, 807.187805)")

    # Compare
    print(f"\nDifferences:")
    print(f"  scale: {abs(scale - 3.533595):.6f}")
    print(f"  rot_x: {abs(rot_x - (-0.220360)):.6f}")
    print(f"  rot_y: {abs(rot_y - (-0.104840)):.6f}")
    print(f"  rot_z: {abs(rot_z - (-0.101141)):.6f}")
    print(f"  tx: {abs(tx - 602.636292):.6f}")
    print(f"  ty: {abs(ty - 807.187805):.6f}")

    # Get landmarks
    eye_landmarks = pdm.params_to_landmarks_2d(params)

    print(f"\nPython first 5 eye landmarks:")
    for i in range(5):
        print(f"  {i}: ({eye_landmarks[i, 0]:.4f}, {eye_landmarks[i, 1]:.4f})")

    print(f"\nC++ first 5 eye landmarks:")
    cpp_landmarks = [
        (579.7636, 809.1275),
        (583.4733, 793.8094),
        (597.3139, 786.0415),
        (613.1776, 790.3743),
        (621.7717, 804.2696)
    ]
    for i, lm in enumerate(cpp_landmarks):
        print(f"  {i}: ({lm[0]:.4f}, {lm[1]:.4f})")
        diff = np.sqrt((eye_landmarks[i, 0] - lm[0])**2 + (eye_landmarks[i, 1] - lm[1])**2)
        print(f"      diff: {diff:.4f} px")

if __name__ == '__main__':
    main()
