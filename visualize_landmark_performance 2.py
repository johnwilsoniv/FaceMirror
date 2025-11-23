#!/usr/bin/env python3
"""
Visualize landmark performance comparing Python vs C++.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import subprocess
import os
import pandas as pd

def get_cpp_landmarks(image_path: str) -> np.ndarray:
    """Run C++ FeatureExtraction and parse landmarks."""
    out_dir = '/tmp/openface_verify'
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(out_dir, f'{base_name}.csv')

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    landmarks = np.zeros((68, 2))
    for i in range(68):
        x_col = f' x_{i}' if f' x_{i}' in df.columns else f'x_{i}'
        y_col = f' y_{i}' if f' y_{i}' in df.columns else f'y_{i}'
        if x_col in df.columns and y_col in df.columns:
            landmarks[i, 0] = df[x_col].iloc[0]
            landmarks[i, 1] = df[y_col].iloc[0]

    return landmarks

def get_python_landmarks(image: np.ndarray) -> np.ndarray:
    """Run Python pipeline and get landmarks."""
    from pyclnf.clnf import CLNF

    clnf = CLNF(
        'pyclnf/models',
        regularization=40,
        use_eye_refinement=True
    )

    result = clnf.detect_and_fit(image)
    if result is None or result[0] is None:
        return None

    return result[0]

def main():
    # Load frame from video
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    frame_idx = 30

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame")
        return

    # Save frame for C++
    image_path = "/tmp/shorty_frame_30.jpg"
    cv2.imwrite(image_path, frame)

    # Get landmarks
    print("Getting C++ landmarks...")
    cpp_lm = get_cpp_landmarks(image_path)
    print("Getting Python landmarks...")
    py_lm = get_python_landmarks(frame)

    if cpp_lm is None or py_lm is None:
        print("Failed to get landmarks")
        return

    # Compute errors
    errors = np.sqrt(np.sum((cpp_lm - py_lm)**2, axis=1))

    # Create visualization
    vis = frame.copy()

    # Define regions with colors
    regions = {
        'Jaw': (list(range(17)), (200, 200, 200)),
        'R Eyebrow': (list(range(17, 22)), (255, 200, 100)),
        'L Eyebrow': (list(range(22, 27)), (255, 200, 100)),
        'Nose': (list(range(27, 36)), (100, 255, 100)),
        'R Eye': (list(range(36, 42)), (0, 100, 255)),  # Red-ish for worse
        'L Eye': (list(range(42, 48)), (255, 100, 0)),  # Blue-ish for better
        'Outer Lip': (list(range(48, 60)), (200, 100, 255)),
        'Inner Lip': (list(range(60, 68)), (200, 100, 255)),
    }

    # Draw landmarks
    for region_name, (indices, color) in regions.items():
        for i in indices:
            # C++ landmark (ground truth) - small circle
            cx, cy = int(cpp_lm[i, 0]), int(cpp_lm[i, 1])
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)  # Green = C++

            # Python landmark - circle with size based on error
            px, py = int(py_lm[i, 0]), int(py_lm[i, 1])
            err = errors[i]

            # Color based on error magnitude
            if err < 1.0:
                pt_color = (0, 255, 0)  # Green - good
            elif err < 2.0:
                pt_color = (0, 255, 255)  # Yellow - moderate
            else:
                pt_color = (0, 0, 255)  # Red - bad

            cv2.circle(vis, (px, py), 4, pt_color, 2)

            # Draw line from C++ to Python to show error
            if err > 0.5:
                cv2.line(vis, (cx, cy), (px, py), (0, 0, 255), 1)

    # Add text annotations for eye regions
    # Right eye (36-41)
    r_eye_center = np.mean(cpp_lm[36:42], axis=0).astype(int)
    r_eye_err = np.mean(errors[36:42])
    cv2.putText(vis, f"R Eye: {r_eye_err:.2f}px",
                (r_eye_center[0] - 60, r_eye_center[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    # Left eye (42-47)
    l_eye_center = np.mean(cpp_lm[42:48], axis=0).astype(int)
    l_eye_err = np.mean(errors[42:48])
    cv2.putText(vis, f"L Eye: {l_eye_err:.2f}px",
                (l_eye_center[0] - 60, l_eye_center[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    # Overall stats
    cv2.putText(vis, f"Overall: {np.mean(errors):.2f}px",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Eyes: {np.mean(errors[36:48]):.2f}px",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Legend
    cv2.putText(vis, "Green dot = C++ (ground truth)",
                (20, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, "Circle = Python (green<1px, yellow<2px, red>2px)",
                (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, "Red line = error vector",
                (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save visualization
    output_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/landmark_performance.png"
    cv2.imwrite(output_path, vis)
    print(f"\nVisualization saved to: {output_path}")

    # Print detailed stats
    print("\n=== Per-Landmark Errors ===")
    print("\nRight Eye (36-41):")
    for i in range(36, 42):
        print(f"  LM{i}: {errors[i]:.2f}px")

    print("\nLeft Eye (42-47):")
    for i in range(42, 48):
        print(f"  LM{i}: {errors[i]:.2f}px")

    print(f"\nRight eye mean: {np.mean(errors[36:42]):.2f}px")
    print(f"Left eye mean: {np.mean(errors[42:48]):.2f}px")
    print(f"Overall mean: {np.mean(errors):.2f}px")

if __name__ == '__main__':
    main()
