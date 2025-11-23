#!/usr/bin/env python3
"""
Debug eye model by using C++ inputs and comparing step-by-step.
This isolates the eye model from main CLNF differences.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pathlib import Path

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING, RIGHT_EYE_MAPPING
from pyclnf.core.eye_pdm import EyePDM

# C++ debug values for frame 0 (from /tmp/cpp_eye_init_debug.txt)
CPP_LEFT_EYE_INPUT = {
    36: (392.1590, 847.6613),
    37: (410.0039, 828.3166),
    38: (436.9223, 826.1841),
    39: (461.9583, 842.8420),
    40: (438.4380, 850.4288),
    41: (411.4089, 853.9998)
}

CPP_LEFT_EYE_CALCPARAMS = {
    'scale': 3.362953,
    'rot': (-0.103207, 0.175851, -0.114345),
    'tx': 426.041504,
    'ty': 839.989624,
    'local': [-1.211064, -2.991485, -2.288383, 0.105293, -0.435301]  # First 5
}

CPP_LEFT_EYE_INITIAL_LANDMARKS = {
    8: (392.3565, 847.6952),
    10: (409.5100, 828.6759),
    12: (437.5647, 825.9781),
    14: (461.7177, 842.6740),
    16: (438.4697, 850.8372),
    18: (411.2716, 853.5720)
}

CPP_LEFT_EYE_POST_REFINEMENT = {
    36: (392.6315, 850.2545),
    37: (410.4917, 831.1846),
    38: (437.1271, 828.7042),
    39: (461.4543, 844.0116),
    40: (438.4069, 852.4236),
    41: (411.8793, 856.0948)
}

def main():
    print("=" * 70)
    print("EYE MODEL ITERATION-BY-ITERATION DEBUG")
    print("=" * 70)

    # Load video frame
    video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
    ret, frame = video.read()
    video.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print(f"\nImage size: {frame.shape[1]}x{frame.shape[0]}")

    # Load eye model
    model_dir = 'pyclnf/models'
    eye_model = HierarchicalEyeModel(model_dir)
    pdm = eye_model.pdm['left']

    print(f"Eye PDM: {pdm.n_points} points, {pdm.n_modes} modes")

    # Step 1: Test CalcParams with C++ input
    print("\n" + "=" * 70)
    print("STEP 1: CalcParams Comparison")
    print("=" * 70)

    # Convert C++ input to array
    main_indices = [36, 37, 38, 39, 40, 41]
    eye_indices = [8, 10, 12, 14, 16, 18]
    target_points = np.array([CPP_LEFT_EYE_INPUT[i] for i in main_indices])

    print("\nC++ Input landmarks (main indices):")
    for i, idx in enumerate(main_indices):
        print(f"  {idx}: ({target_points[i, 0]:.4f}, {target_points[i, 1]:.4f})")

    # Call _fit_eye_shape to get CalcParams result
    params = eye_model._fit_eye_shape(target_points, LEFT_EYE_MAPPING, 'left', main_rotation=None)

    print("\nPython CalcParams result:")
    print(f"  scale: {params[0]:.6f}")
    print(f"  rot: ({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
    print(f"  tx, ty: ({params[4]:.6f}, {params[5]:.6f})")
    print(f"  local[0:5]: [{params[6]:.6f}, {params[7]:.6f}, {params[8]:.6f}, {params[9]:.6f}, {params[10]:.6f}]")

    print("\nC++ CalcParams result:")
    print(f"  scale: {CPP_LEFT_EYE_CALCPARAMS['scale']:.6f}")
    print(f"  rot: ({CPP_LEFT_EYE_CALCPARAMS['rot'][0]:.6f}, {CPP_LEFT_EYE_CALCPARAMS['rot'][1]:.6f}, {CPP_LEFT_EYE_CALCPARAMS['rot'][2]:.6f})")
    print(f"  tx, ty: ({CPP_LEFT_EYE_CALCPARAMS['tx']:.6f}, {CPP_LEFT_EYE_CALCPARAMS['ty']:.6f})")
    print(f"  local[0:5]: {CPP_LEFT_EYE_CALCPARAMS['local']}")

    print("\nDifferences:")
    print(f"  scale: {abs(params[0] - CPP_LEFT_EYE_CALCPARAMS['scale']):.6f}")
    print(f"  rot_x: {abs(params[1] - CPP_LEFT_EYE_CALCPARAMS['rot'][0]):.6f}")
    print(f"  rot_y: {abs(params[2] - CPP_LEFT_EYE_CALCPARAMS['rot'][1]):.6f}")
    print(f"  rot_z: {abs(params[3] - CPP_LEFT_EYE_CALCPARAMS['rot'][2]):.6f}")
    print(f"  tx: {abs(params[4] - CPP_LEFT_EYE_CALCPARAMS['tx']):.6f}")
    print(f"  ty: {abs(params[5] - CPP_LEFT_EYE_CALCPARAMS['ty']):.6f}")

    # Step 2: Compare initial eye landmarks
    print("\n" + "=" * 70)
    print("STEP 2: Initial Eye Landmarks Comparison")
    print("=" * 70)

    py_landmarks = pdm.params_to_landmarks_2d(params)

    print("\nVisible eye landmarks (indices 8, 10, 12, 14, 16, 18):")
    print("  Idx  | Python               | C++                  | Diff")
    print("  -----|----------------------|----------------------|--------")

    total_diff = 0
    for eye_idx in eye_indices:
        py = py_landmarks[eye_idx]
        cpp = CPP_LEFT_EYE_INITIAL_LANDMARKS[eye_idx]
        diff = np.sqrt((py[0] - cpp[0])**2 + (py[1] - cpp[1])**2)
        total_diff += diff
        print(f"  {eye_idx:4d} | ({py[0]:8.4f}, {py[1]:8.4f}) | ({cpp[0]:8.4f}, {cpp[1]:8.4f}) | {diff:.4f}")

    print(f"\n  Mean landmark difference: {total_diff / len(eye_indices):.4f} px")

    # Step 3: Run eye refinement with C++ input and compare
    print("\n" + "=" * 70)
    print("STEP 3: Eye Refinement (using C++ input landmarks)")
    print("=" * 70)

    # Create main_landmarks array with C++ values
    main_landmarks = np.zeros((68, 2))
    for main_idx, pos in CPP_LEFT_EYE_INPUT.items():
        main_landmarks[main_idx] = pos

    # Also need to fill in other landmarks for the eye model to work
    # For now, just use the eye landmarks

    # Run eye refinement
    refined = eye_model.refine_eye_landmarks(
        gray, main_landmarks.copy(), 'left',
        main_rotation=None, main_scale=1.0
    )

    print("\nRefinement results:")
    print("  Idx  | Input                | Python refined       | C++ refined          | Py vs C++")
    print("  -----|----------------------|----------------------|----------------------|----------")

    for main_idx in main_indices:
        inp = CPP_LEFT_EYE_INPUT[main_idx]
        py_ref = refined[main_idx]
        cpp_ref = CPP_LEFT_EYE_POST_REFINEMENT[main_idx]
        diff = np.sqrt((py_ref[0] - cpp_ref[0])**2 + (py_ref[1] - cpp_ref[1])**2)
        print(f"  {main_idx:4d} | ({inp[0]:8.4f}, {inp[1]:8.4f}) | ({py_ref[0]:8.4f}, {py_ref[1]:8.4f}) | ({cpp_ref[0]:8.4f}, {cpp_ref[1]:8.4f}) | {diff:.4f}")

    # Step 4: Compare refinement changes
    print("\n" + "=" * 70)
    print("STEP 4: Refinement Changes Comparison")
    print("=" * 70)

    print("\n  Idx  | Python Δ             | C++ Δ                | Direction match?")
    print("  -----|----------------------|----------------------|------------------")

    for main_idx in main_indices:
        inp = CPP_LEFT_EYE_INPUT[main_idx]
        py_ref = refined[main_idx]
        cpp_ref = CPP_LEFT_EYE_POST_REFINEMENT[main_idx]

        py_dx = py_ref[0] - inp[0]
        py_dy = py_ref[1] - inp[1]
        cpp_dx = cpp_ref[0] - inp[0]
        cpp_dy = cpp_ref[1] - inp[1]

        # Check if directions match
        x_match = (py_dx * cpp_dx) > 0 if abs(cpp_dx) > 0.1 else True
        y_match = (py_dy * cpp_dy) > 0 if abs(cpp_dy) > 0.1 else True
        match_str = "✓" if (x_match and y_match) else "✗ WRONG"

        print(f"  {main_idx:4d} | ({py_dx:+7.4f}, {py_dy:+7.4f}) | ({cpp_dx:+7.4f}, {cpp_dy:+7.4f}) | {match_str}")

if __name__ == '__main__':
    main()
