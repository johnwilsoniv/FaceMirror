#!/usr/bin/env python3
"""
Visualize landmark error between Python and C++ OpenFace.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import pandas as pd
from pyclnf.clnf import CLNF

def load_openface_landmarks(csv_path: str, frame_num: int) -> np.ndarray:
    """Load OpenFace landmarks for a specific frame."""
    df = pd.read_csv(csv_path)
    row = df[df['frame'] == frame_num].iloc[0]

    landmarks = np.zeros((68, 2))
    for i in range(68):
        for x_col, y_col in [(f'x_{i}', f'y_{i}'), (f' x_{i}', f' y_{i}')]:
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = row[x_col]
                landmarks[i, 1] = row[y_col]
                break
    return landmarks

def main():
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/IMG_0942.MOV'
    csv_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/IMG_0942.csv'

    # Use frame 100 for visualization
    frame_num = 100

    # Load video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame")
        return

    # Get C++ landmarks
    cpp_lm = load_openface_landmarks(csv_path, frame_num)

    # Get Python landmarks
    print("Running Python CLNF...")
    clnf = CLNF('pyclnf/pyclnf/models', regularization=40)
    result = clnf.detect_and_fit(frame)

    if result is None or result[0] is None:
        print("Python CLNF failed to detect face")
        return

    py_lm = result[0]

    # Compute errors
    errors = np.sqrt(np.sum((cpp_lm - py_lm)**2, axis=1))

    # Create visualization
    vis = frame.copy()

    # Draw C++ landmarks in green
    for i, (x, y) in enumerate(cpp_lm):
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Draw Python landmarks in red
    for i, (x, y) in enumerate(py_lm):
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Draw lines between corresponding landmarks for high-error points
    for i in range(68):
        if errors[i] > 10:  # Only show high errors
            pt1 = (int(cpp_lm[i, 0]), int(cpp_lm[i, 1]))
            pt2 = (int(py_lm[i, 0]), int(py_lm[i, 1]))
            cv2.line(vis, pt1, pt2, (255, 0, 255), 1)

    # Add text
    cv2.putText(vis, f"Green: C++ OpenFace", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"Red: Python CLNF", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, f"Mean error: {np.mean(errors):.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save
    output_path = 'landmark_error_visualization.png'
    cv2.imwrite(output_path, vis)
    print(f"\nSaved visualization to: {output_path}")

    # Print per-region errors
    print(f"\nFrame {frame_num} errors:")
    print(f"  Jaw (0-16): {np.mean(errors[0:17]):.1f}px")
    print(f"  Eyebrows (17-26): {np.mean(errors[17:27]):.1f}px")
    print(f"  Nose (27-35): {np.mean(errors[27:36]):.1f}px")
    print(f"  Eyes (36-47): {np.mean(errors[36:48]):.1f}px")
    print(f"  Mouth (48-67): {np.mean(errors[48:68]):.1f}px")

    print(f"\nWorst landmarks:")
    worst = np.argsort(errors)[-5:][::-1]
    for idx in worst:
        print(f"  Landmark {idx}: {errors[idx]:.1f}px")

if __name__ == '__main__':
    main()
