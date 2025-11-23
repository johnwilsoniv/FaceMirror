#!/usr/bin/env python3
"""
Test eye refinement to debug why it makes results worse.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import subprocess
import os

def get_cpp_landmarks(image_path: str) -> np.ndarray:
    """Run C++ FeatureExtraction and parse landmarks."""
    import pandas as pd

    out_dir = '/tmp/openface_verify'
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse CSV output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(out_dir, f'{base_name}.csv')

    if not os.path.exists(csv_path):
        print(f"Error: C++ output not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    landmarks = np.zeros((68, 2))
    for i in range(68):
        x_col = f'x_{i}'
        y_col = f'y_{i}'
        if x_col in df.columns and y_col in df.columns:
            landmarks[i, 0] = df[x_col].iloc[0]
            landmarks[i, 1] = df[y_col].iloc[0]
        else:
            x_col = f' x_{i}'
            y_col = f' y_{i}'
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = df[x_col].iloc[0]
                landmarks[i, 1] = df[y_col].iloc[0]

    return landmarks

def main():
    print("=" * 70)
    print("EYE REFINEMENT DIAGNOSTIC TEST")
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

    # Save for C++ processing
    image_path = "/tmp/shorty_frame_30.jpg"
    cv2.imwrite(image_path, frame)

    # Get C++ reference
    print("\nRunning C++ FeatureExtraction...")
    cpp_landmarks = get_cpp_landmarks(image_path)
    if cpp_landmarks is None:
        return

    # Test WITHOUT eye refinement
    print("\nRunning Python WITHOUT eye refinement...")
    from pyclnf.clnf import CLNF

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

    # Test WITH eye refinement
    print("Running Python WITH eye refinement...")
    clnf_eye = CLNF(
        'pyclnf/models',
        regularization=35,
        use_eye_refinement=True
    )
    result_eye = clnf_eye.detect_and_fit(frame)
    if result_eye is None or result_eye[0] is None:
        print("Failed with eye refinement")
        return
    landmarks_eye = result_eye[0]

    # Compute errors
    errors_no_eye = np.sqrt(np.sum((cpp_landmarks - landmarks_no_eye)**2, axis=1))
    errors_eye = np.sqrt(np.sum((cpp_landmarks - landmarks_eye)**2, axis=1))

    # Overall comparison
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)
    print(f"\nWithout eye refinement:")
    print(f"  Mean: {np.mean(errors_no_eye):.2f}px")
    print(f"  Eyes: {np.mean(errors_no_eye[36:48]):.2f}px")

    print(f"\nWith eye refinement:")
    print(f"  Mean: {np.mean(errors_eye):.2f}px")
    print(f"  Eyes: {np.mean(errors_eye[36:48]):.2f}px")

    print(f"\nDifference (eye refine - no refine):")
    print(f"  Mean: {np.mean(errors_eye) - np.mean(errors_no_eye):+.2f}px")
    print(f"  Eyes: {np.mean(errors_eye[36:48]) - np.mean(errors_no_eye[36:48]):+.2f}px")

    # Eye landmark detail
    print("\n" + "=" * 70)
    print("EYE LANDMARK DETAIL")
    print("=" * 70)

    eye_names = {
        36: 'R outer', 37: 'R up-out', 38: 'R up-in',
        39: 'R inner', 40: 'R lo-in', 41: 'R lo-out',
        42: 'L inner', 43: 'L up-in', 44: 'L up-out',
        45: 'L outer', 46: 'L lo-out', 47: 'L lo-in'
    }

    print("\nRight eye (36-41):")
    print(f"{'Idx':<4} {'Name':<10} {'No-Eye':<8} {'Eye':<8} {'Diff':<8} {'C++':^20} {'NoEye':^20} {'Eye':^20}")
    for i in range(36, 42):
        diff = errors_eye[i] - errors_no_eye[i]
        cpp = f"({cpp_landmarks[i,0]:.1f}, {cpp_landmarks[i,1]:.1f})"
        no_eye = f"({landmarks_no_eye[i,0]:.1f}, {landmarks_no_eye[i,1]:.1f})"
        eye = f"({landmarks_eye[i,0]:.1f}, {landmarks_eye[i,1]:.1f})"
        print(f"{i:<4} {eye_names[i]:<10} {errors_no_eye[i]:<8.2f} {errors_eye[i]:<8.2f} {diff:+.2f}     {cpp:<20} {no_eye:<20} {eye:<20}")

    print("\nLeft eye (42-47):")
    for i in range(42, 48):
        diff = errors_eye[i] - errors_no_eye[i]
        cpp = f"({cpp_landmarks[i,0]:.1f}, {cpp_landmarks[i,1]:.1f})"
        no_eye = f"({landmarks_no_eye[i,0]:.1f}, {landmarks_no_eye[i,1]:.1f})"
        eye = f"({landmarks_eye[i,0]:.1f}, {landmarks_eye[i,1]:.1f})"
        print(f"{i:<4} {eye_names[i]:<10} {errors_no_eye[i]:<8.2f} {errors_eye[i]:<8.2f} {diff:+.2f}     {cpp:<20} {no_eye:<20} {eye:<20}")

    # Analyze movement direction
    print("\n" + "=" * 70)
    print("MOVEMENT ANALYSIS")
    print("=" * 70)

    print("\nHow eye refinement moved landmarks relative to no-refinement:")
    for i in range(36, 48):
        dx = landmarks_eye[i, 0] - landmarks_no_eye[i, 0]
        dy = landmarks_eye[i, 1] - landmarks_no_eye[i, 1]

        # Ideal direction would be towards C++
        ideal_dx = cpp_landmarks[i, 0] - landmarks_no_eye[i, 0]
        ideal_dy = cpp_landmarks[i, 1] - landmarks_no_eye[i, 1]

        # Check if movement was in right direction
        correct_x = (dx * ideal_dx) > 0 if abs(ideal_dx) > 0.1 else True
        correct_y = (dy * ideal_dy) > 0 if abs(ideal_dy) > 0.1 else True

        status = "✓" if correct_x and correct_y else "✗"
        print(f"  {i} ({eye_names[i]}): moved ({dx:+.2f}, {dy:+.2f}), needed ({ideal_dx:+.2f}, {ideal_dy:+.2f}) {status}")

    # Check debug files
    print("\n" + "=" * 70)
    print("DEBUG FILE LOCATIONS")
    print("=" * 70)
    print("\n/tmp/python_eye_model_debug.txt - Pre/post refinement landmarks")
    print("/tmp/python_eye_init_debug.txt - Eye PDM initialization")
    print("/tmp/python_eye_response_maps.txt - Response maps")

if __name__ == '__main__':
    main()
