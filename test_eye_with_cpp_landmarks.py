#!/usr/bin/env python3
"""
Test Python eye refinement using C++ landmarks as starting point.
This isolates eye refinement differences from main model differences.
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
    """Run C++ FeatureExtraction to get landmarks."""
    print("=== Getting C++ Landmarks ===")

    out_dir = '/tmp/openface_cpp_lm'
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"C++ failed: {result.stderr}")
        return None

    # Parse CSV for landmarks
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(out_dir, f'{base_name}.csv')

    if not os.path.exists(csv_path):
        print(f"Error: C++ output not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    landmarks = np.zeros((68, 2))
    for i in range(68):
        for x_col, y_col in [(f'x_{i}', f'y_{i}'), (f' x_{i}', f' y_{i}')]:
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = df[x_col].iloc[0]
                landmarks[i, 1] = df[y_col].iloc[0]
                break

    return landmarks

def run_python_eye_refinement_with_landmarks(image: np.ndarray, cpp_landmarks: np.ndarray):
    """Run Python eye refinement starting from C++ landmarks."""
    print("\n=== Running Python Eye Refinement with C++ Landmarks ===")

    from pyclnf.clnf import CLNF
    from pyclnf.core.eye_patch_expert import HierarchicalEyeModel

    # Initialize eye model directly
    eye_model = HierarchicalEyeModel('pyclnf/models')

    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Extract left and right eye landmarks from C++ output
    # Left eye: landmarks 36-41, Right eye: landmarks 42-47
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))

    print(f"\nC++ Left eye landmarks (36-41):")
    for i in left_eye_indices:
        print(f"  LM{i}: ({cpp_landmarks[i, 0]:.4f}, {cpp_landmarks[i, 1]:.4f})")

    print(f"\nC++ Right eye landmarks (42-47):")
    for i in right_eye_indices:
        print(f"  LM{i}: ({cpp_landmarks[i, 0]:.4f}, {cpp_landmarks[i, 1]:.4f})")

    # Clear trace file
    trace_path = '/tmp/eye8_trace_python.txt'
    with open(trace_path, 'w') as f:
        f.write("=== PYTHON Eye_8 Trace (Using C++ Landmarks) ===\n")
        f.write(f"Input LM36 from C++: ({cpp_landmarks[36, 0]:.6f}, {cpp_landmarks[36, 1]:.6f})\n\n")

    # Refine eyes using the eye model
    # Use identity rotation and scale=1.0 (C++ doesn't pass these explicitly)
    main_rotation = np.eye(3)
    main_scale = 1.0

    refined_landmarks = cpp_landmarks.copy()

    # Refine left eye (36-41)
    print("\nRefining LEFT eye...")
    refined_landmarks = eye_model.refine_eye_landmarks(
        gray,
        refined_landmarks,
        'left',
        main_rotation,
        main_scale
    )

    # Refine right eye (42-47)
    print("Refining RIGHT eye...")
    refined_landmarks = eye_model.refine_eye_landmarks(
        gray,
        refined_landmarks,
        'right',
        main_rotation,
        main_scale
    )

    return refined_landmarks

def compare_eye_landmarks(cpp_landmarks, python_landmarks):
    """Compare eye landmarks between C++ and Python."""
    print("\n=== Eye Landmark Comparison ===")

    # Left eye (36-41)
    print("\nLeft Eye (36-41):")
    left_errors = []
    for i in range(36, 42):
        diff = python_landmarks[i] - cpp_landmarks[i]
        error = np.linalg.norm(diff)
        left_errors.append(error)
        print(f"  LM{i}: C++({cpp_landmarks[i, 0]:.2f}, {cpp_landmarks[i, 1]:.2f}) "
              f"Py({python_landmarks[i, 0]:.2f}, {python_landmarks[i, 1]:.2f}) "
              f"Diff({diff[0]:+.2f}, {diff[1]:+.2f}) Err={error:.2f}px")

    # Right eye (42-47)
    print("\nRight Eye (42-47):")
    right_errors = []
    for i in range(42, 48):
        diff = python_landmarks[i] - cpp_landmarks[i]
        error = np.linalg.norm(diff)
        right_errors.append(error)
        print(f"  LM{i}: C++({cpp_landmarks[i, 0]:.2f}, {cpp_landmarks[i, 1]:.2f}) "
              f"Py({python_landmarks[i, 0]:.2f}, {python_landmarks[i, 1]:.2f}) "
              f"Diff({diff[0]:+.2f}, {diff[1]:+.2f}) Err={error:.2f}px")

    print(f"\nLeft eye mean error: {np.mean(left_errors):.3f}px")
    print(f"Right eye mean error: {np.mean(right_errors):.3f}px")
    print(f"Left/Right ratio: {np.mean(left_errors)/np.mean(right_errors):.2f}x")

def main():
    image_path = 'comparison_frame_0000.jpg'

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    print(f"Using image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # Get C++ landmarks
    cpp_landmarks = get_cpp_landmarks(image_path)
    if cpp_landmarks is None:
        return

    # Run Python eye refinement with C++ landmarks
    python_landmarks = run_python_eye_refinement_with_landmarks(image, cpp_landmarks)
    if python_landmarks is None:
        return

    # Compare results
    compare_eye_landmarks(cpp_landmarks, python_landmarks)

    # Show the trace
    print("\n=== Python Eye_8 Trace ===")
    trace_path = '/tmp/eye8_trace_python.txt'
    if os.path.exists(trace_path):
        with open(trace_path, 'r') as f:
            content = f.read()
            # Show first part of trace
            lines = content.split('\n')
            for line in lines[:100]:
                print(line)
            if len(lines) > 100:
                print(f"... ({len(lines) - 100} more lines)")

if __name__ == '__main__':
    main()
