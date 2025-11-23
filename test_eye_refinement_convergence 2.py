#!/usr/bin/env python3
"""
Test eye refinement convergence by comparing Python output to C++ ground truth.

This test:
1. Runs Python eye refinement on left eye
2. Compares output to C++ final landmarks
3. Measures if Python improves or worsens the landmarks
"""

import numpy as np
import cv2
import subprocess
import re
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel

def get_cpp_final_landmarks():
    """Get C++ final eye landmarks from debug output."""
    # Parse C++ debug file for final eye landmarks
    with open('/tmp/cpp_eye_model_detailed.txt', 'r') as f:
        content = f.read()

    # Find the final left eye landmarks (Eye_8 around x=390)
    # Look for "Updated mapped eyelid landmarks" after left eye processing
    matches = re.findall(
        r'Updated mapped eyelid landmarks:\s*'
        r'Eye_8: \(([\d.]+), ([\d.]+)\)\s*'
        r'Eye_10: \(([\d.]+), ([\d.]+)\)\s*'
        r'Eye_12: \(([\d.]+), ([\d.]+)\)\s*'
        r'Eye_14: \(([\d.]+), ([\d.]+)\)\s*'
        r'Eye_16: \(([\d.]+), ([\d.]+)\)\s*'
        r'Eye_18: \(([\d.]+), ([\d.]+)\)',
        content
    )

    # Find left eye (Eye_8.x < 500)
    left_eye_final = None
    for m in matches:
        eye_8_x = float(m[0])
        if eye_8_x < 500:  # Left eye
            left_eye_final = {
                8: (float(m[0]), float(m[1])),
                10: (float(m[2]), float(m[3])),
                12: (float(m[4]), float(m[5])),
                14: (float(m[6]), float(m[7])),
                16: (float(m[8]), float(m[9])),
                18: (float(m[10]), float(m[11]))
            }

    return left_eye_final

def run_python_eye_refinement():
    """Run Python eye refinement and return final landmarks."""
    # Load video
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load eye model
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models'
    eye_model = HierarchicalEyeModel(model_dir)

    # Get initial landmarks from main model (approximate)
    # Using C++ initial left eye positions
    main_landmarks = np.zeros((68, 2), dtype=np.float32)

    # Left eye landmarks (36-41) from C++ initial
    main_landmarks[36] = [391.3150, 827.5953]  # Eye_8
    main_landmarks[37] = [408.7730, 808.2344]  # Eye_10
    main_landmarks[38] = [436.9240, 805.9266]  # Eye_12
    main_landmarks[39] = [460.8869, 823.4746]  # Eye_14
    main_landmarks[40] = [437.5075, 831.1237]  # Eye_16
    main_landmarks[41] = [410.1848, 833.5138]  # Eye_18

    # Run eye refinement
    refined_landmarks = eye_model.refine_eye_landmarks(
        gray, main_landmarks, 'left',
        main_rotation=np.array([-0.118319, 0.176098, -0.099366]),
        main_scale=3.371204
    )

    # Extract refined eye landmarks
    python_final = {
        8: tuple(refined_landmarks[36]),
        10: tuple(refined_landmarks[37]),
        12: tuple(refined_landmarks[38]),
        14: tuple(refined_landmarks[39]),
        16: tuple(refined_landmarks[40]),
        18: tuple(refined_landmarks[41])
    }

    return python_final, main_landmarks

def main():
    print("=" * 70)
    print("EYE REFINEMENT CONVERGENCE TEST")
    print("=" * 70)

    # Get C++ final landmarks
    cpp_final = get_cpp_final_landmarks()
    if cpp_final is None:
        print("ERROR: Could not find C++ left eye final landmarks")
        return

    print("\nC++ Final Eye Landmarks (Left Eye):")
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        print(f"  Eye_{eye_idx}: {cpp_final[eye_idx]}")

    # Run Python refinement
    python_final, initial = run_python_eye_refinement()

    print("\nPython Final Eye Landmarks:")
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        print(f"  Eye_{eye_idx}: ({python_final[eye_idx][0]:.4f}, {python_final[eye_idx][1]:.4f})")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    main_to_eye = {36: 8, 37: 10, 38: 12, 39: 14, 40: 16, 41: 18}

    total_initial_error = 0
    total_python_error = 0
    total_cpp_error = 0

    print("\n| Main | Eye | Initial → Python | Initial → C++ | Py vs C++ |")
    print("|------|-----|------------------|---------------|-----------|")

    for main_idx, eye_idx in main_to_eye.items():
        # Initial position
        init_x, init_y = initial[main_idx]

        # Python final
        py_x, py_y = python_final[eye_idx]

        # C++ final
        cpp_x, cpp_y = cpp_final[eye_idx]

        # Errors (comparing to C++ as ground truth)
        init_err = np.sqrt((init_x - cpp_x)**2 + (init_y - cpp_y)**2)
        py_err = np.sqrt((py_x - cpp_x)**2 + (py_y - cpp_y)**2)

        total_initial_error += init_err
        total_python_error += py_err

        # Direction indicator
        if py_err < init_err:
            direction = "✅ improved"
        elif py_err > init_err * 1.5:
            direction = "❌ worse"
        else:
            direction = "→ similar"

        print(f"| {main_idx:4d} | {eye_idx:3d} | {init_err:16.2f} | {init_err:13.2f} | {py_err:6.2f} {direction} |")

    avg_initial = total_initial_error / 6
    avg_python = total_python_error / 6

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAverage error to C++ ground truth:")
    print(f"  Initial: {avg_initial:.2f} px")
    print(f"  Python:  {avg_python:.2f} px")

    if avg_python < avg_initial:
        print(f"\n✅ Python IMPROVED landmarks by {avg_initial - avg_python:.2f} px")
    else:
        print(f"\n❌ Python WORSENED landmarks by {avg_python - avg_initial:.2f} px")

    # Show the movement direction
    print("\n" + "=" * 70)
    print("MOVEMENT ANALYSIS")
    print("=" * 70)

    print("\n| Eye | Initial → Python | Should be (to reach C++) |")
    print("|-----|------------------|--------------------------|")

    for main_idx, eye_idx in main_to_eye.items():
        init_x, init_y = initial[main_idx]
        py_x, py_y = python_final[eye_idx]
        cpp_x, cpp_y = cpp_final[eye_idx]

        py_dx = py_x - init_x
        py_dy = py_y - init_y

        target_dx = cpp_x - init_x
        target_dy = cpp_y - init_y

        # Check if directions match
        x_match = "✅" if (py_dx * target_dx > 0 or abs(target_dx) < 0.5) else "❌"
        y_match = "✅" if (py_dy * target_dy > 0 or abs(target_dy) < 0.5) else "❌"

        print(f"| {eye_idx:3d} | ({py_dx:+6.2f}, {py_dy:+6.2f}) | ({target_dx:+6.2f}, {target_dy:+6.2f}) | {x_match} {y_match} |")

if __name__ == '__main__':
    main()
