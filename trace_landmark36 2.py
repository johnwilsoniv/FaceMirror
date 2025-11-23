#!/usr/bin/env python3
"""
Trace landmark 36 (eye landmark 8) step by step through eye refinement.
Compare each step with C++ to find where divergence occurs.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import subprocess
import os
import pandas as pd

def get_cpp_landmarks(image_path: str):
    """Run C++ FeatureExtraction and get landmarks."""
    out_dir = '/tmp/trace_lm36'
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir,
        '-of', 'trace'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    csv_path = os.path.join(out_dir, 'trace.csv')
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = df[f'x_{i}'].iloc[0]
        landmarks[i, 1] = df[f'y_{i}'].iloc[0]

    return landmarks


def main():
    print("=" * 70)
    print("LANDMARK 36 STEP-BY-STEP TRACE")
    print("=" * 70)

    # Load test frame
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    frame_idx = 30

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return

    image_path = "/tmp/shorty_frame_30.jpg"
    cv2.imwrite(image_path, frame)

    # Get C++ reference
    print("\n[Step 0: C++ Reference]")
    cpp_landmarks = get_cpp_landmarks(image_path)
    if cpp_landmarks is None:
        print("Failed to get C++ landmarks")
        return

    cpp_36 = cpp_landmarks[36]
    print(f"  C++ landmark 36: ({cpp_36[0]:.4f}, {cpp_36[1]:.4f})")

    # Run Python pipeline step by step
    print("\n[Step 1: Python Initialization]")

    from pyclnf.clnf import CLNF

    # First, run WITHOUT eye refinement to get initial position
    clnf_no_eye = CLNF(
        'pyclnf/models',
        regularization=35,
        use_eye_refinement=False
    )
    result_no_eye = clnf_no_eye.detect_and_fit(frame)
    if result_no_eye is None or result_no_eye[0] is None:
        print("Failed without eye refinement")
        return

    landmarks_no_eye = result_no_eye[0]
    py_36_init = landmarks_no_eye[36]

    print(f"  Python initial landmark 36: ({py_36_init[0]:.4f}, {py_36_init[1]:.4f})")
    print(f"  Diff from C++: ({py_36_init[0] - cpp_36[0]:.4f}, {py_36_init[1] - cpp_36[1]:.4f})")
    print(f"  Error: {np.sqrt((py_36_init[0]-cpp_36[0])**2 + (py_36_init[1]-cpp_36[1])**2):.4f} px")

    # Now run WITH eye refinement and trace each step
    print("\n[Step 2: Python Eye Refinement]")

    # Load a fresh CLNF with eye refinement
    clnf_eye = CLNF(
        'pyclnf/models',
        regularization=35,
        use_eye_refinement=True
    )

    # Run detection and main model fitting
    result_eye = clnf_eye.detect_and_fit(frame)
    if result_eye is None or result_eye[0] is None:
        print("Failed with eye refinement")
        return

    landmarks_eye = result_eye[0]
    py_36_final = landmarks_eye[36]

    print(f"  Python final landmark 36: ({py_36_final[0]:.4f}, {py_36_final[1]:.4f})")
    print(f"  Diff from C++: ({py_36_final[0] - cpp_36[0]:.4f}, {py_36_final[1] - cpp_36[1]:.4f})")
    print(f"  Error: {np.sqrt((py_36_final[0]-cpp_36[0])**2 + (py_36_final[1]-cpp_36[1])**2):.4f} px")

    # Movement analysis
    print("\n[Step 3: Movement Analysis]")

    movement = py_36_final - py_36_init
    needed = cpp_36 - py_36_init

    print(f"  Python moved: ({movement[0]:.4f}, {movement[1]:.4f})")
    print(f"  Needed to move: ({needed[0]:.4f}, {needed[1]:.4f})")
    print(f"  Movement error: ({movement[0] - needed[0]:.4f}, {movement[1] - needed[1]:.4f})")

    # Check ratio
    if abs(needed[0]) > 0.1:
        ratio_x = movement[0] / needed[0]
        print(f"  X ratio (actual/needed): {ratio_x:.4f}")
    if abs(needed[1]) > 0.1:
        ratio_y = movement[1] / needed[1]
        print(f"  Y ratio (actual/needed): {ratio_y:.4f}")

    print("\n[Step 4: Checking Debug Files]")

    # Check eye response maps
    try:
        with open('/tmp/python_eye_response_maps.txt', 'r') as f:
            content = f.read()
            # Find landmark 8 (which is main model 36)
            if 'Eye landmark 8' in content:
                print("  Found eye landmark 8 response map in debug file")
                # Extract relevant section
                lines = content.split('\n')
                in_lm8 = False
                for line in lines:
                    if 'Eye landmark 8' in line:
                        in_lm8 = True
                    if in_lm8:
                        print(f"    {line}")
                        if line.strip() == '' and in_lm8:
                            break
    except FileNotFoundError:
        print("  No response map debug file found")

    print("\n" + "=" * 70)
    print("TRACE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
