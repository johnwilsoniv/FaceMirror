#!/usr/bin/env python3
"""
End-to-end test of eye refinement comparing Python to C++ landmarks.

Run both C++ and Python on the same frame and compare eye landmarks.
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.clnf import CLNF

# Paths
VIDEO_PATH = Path("Patient Data/Normal Cohort/Shorty.mov")
CPP_BIN = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
OUTPUT_DIR = Path("/tmp/eye_refinement_test")

def run_cpp(frame_path):
    """Run C++ OpenFace and return landmarks."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    cmd = [
        CPP_BIN,
        "-f", str(frame_path),
        "-out_dir", str(OUTPUT_DIR),
        "-2Dfp"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Load CSV output
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    if not csv_files:
        return None

    import pandas as pd
    df = pd.read_csv(csv_files[0])

    # Extract landmarks
    x_cols = [f"x_{i}" for i in range(68)]
    y_cols = [f"y_{i}" for i in range(68)]

    x_vals = df[x_cols].values[0]
    y_vals = df[y_cols].values[0]

    return np.column_stack([x_vals, y_vals])

def main():
    print("="*70)
    print("Eye Refinement End-to-End Test")
    print("="*70)

    # Load video frame
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_idx = 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {frame_idx}")
        return

    # Save frame for C++
    frame_path = OUTPUT_DIR / f"frame_{frame_idx}.png"
    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(frame_path), frame)

    # Run C++ OpenFace
    print(f"\nRunning C++ OpenFace on frame {frame_idx}...")
    cpp_landmarks = run_cpp(frame_path)

    if cpp_landmarks is None:
        print("Error: C++ failed")
        return

    print(f"C++ landmarks loaded: {cpp_landmarks.shape}")

    # Get bbox from C++ landmarks
    bbox_margin = 30
    bbox_x = cpp_landmarks[:, 0].min() - bbox_margin
    bbox_y = cpp_landmarks[:, 1].min() - bbox_margin
    bbox_w = cpp_landmarks[:, 0].max() - bbox_x + bbox_margin
    bbox_h = cpp_landmarks[:, 1].max() - bbox_y + bbox_margin
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)

    # Run Python WITHOUT eye refinement
    print("\nRunning Python CLNF WITHOUT eye refinement...")
    clnf_no_eye = CLNF(
        model_dir="pyclnf/models",
        detector=None,
        max_iterations=40,
        regularization=25,
        sigma=1.75,
        use_eye_refinement=False
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    py_lm_no_eye, _ = clnf_no_eye.fit(gray, bbox, return_params=True)

    # Run Python WITH eye refinement
    print("Running Python CLNF WITH eye refinement...")
    clnf_with_eye = CLNF(
        model_dir="pyclnf/models",
        detector=None,
        max_iterations=40,
        regularization=25,
        sigma=1.75,
        use_eye_refinement=True
    )

    py_lm_with_eye, _ = clnf_with_eye.fit(gray, bbox, return_params=True)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON: Eye Landmarks (36-47)")
    print("="*70)

    print(f"\n{'Landmark':<10} {'C++':<25} {'Py (no eye)':<25} {'Py (eye)':<25} {'Δ no eye':<12} {'Δ eye':<12}")
    print("-"*110)

    eye_indices = list(range(36, 48))

    errors_no_eye = []
    errors_with_eye = []

    for i in eye_indices:
        cpp = cpp_landmarks[i]
        py_no = py_lm_no_eye[i]
        py_eye = py_lm_with_eye[i]

        err_no = np.sqrt(np.sum((py_no - cpp)**2))
        err_eye = np.sqrt(np.sum((py_eye - cpp)**2))

        errors_no_eye.append(err_no)
        errors_with_eye.append(err_eye)

        cpp_str = f"({cpp[0]:.2f}, {cpp[1]:.2f})"
        py_no_str = f"({py_no[0]:.2f}, {py_no[1]:.2f})"
        py_eye_str = f"({py_eye[0]:.2f}, {py_eye[1]:.2f})"

        print(f"{i:<10} {cpp_str:<25} {py_no_str:<25} {py_eye_str:<25} {err_no:>8.2f} px  {err_eye:>8.2f} px")

    print("-"*110)
    print(f"{'Mean':<10} {'':<25} {'':<25} {'':<25} {np.mean(errors_no_eye):>8.2f} px  {np.mean(errors_with_eye):>8.2f} px")

    # Check if eye refinement helped
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    improved = np.mean(errors_with_eye) < np.mean(errors_no_eye)
    improvement = np.mean(errors_no_eye) - np.mean(errors_with_eye)

    if improved:
        print(f"\n✓ Eye refinement IMPROVED results by {improvement:.2f} px mean")
    else:
        print(f"\n✗ Eye refinement WORSENED results by {-improvement:.2f} px mean")

    # Check X direction for landmark 36
    print("\nLandmark 36 detail:")
    cpp_36 = cpp_landmarks[36]
    py_no_36 = py_lm_no_eye[36]
    py_eye_36 = py_lm_with_eye[36]

    eye_movement_x = py_eye_36[0] - py_no_36[0]
    needed_movement_x = cpp_36[0] - py_no_36[0]

    print(f"  C++ position:       ({cpp_36[0]:.2f}, {cpp_36[1]:.2f})")
    print(f"  Python (no eye):    ({py_no_36[0]:.2f}, {py_no_36[1]:.2f})")
    print(f"  Python (with eye):  ({py_eye_36[0]:.2f}, {py_eye_36[1]:.2f})")
    print(f"  Needed X movement:  {needed_movement_x:+.2f} px")
    print(f"  Actual X movement:  {eye_movement_x:+.2f} px")

    if np.sign(eye_movement_x) == np.sign(needed_movement_x) or abs(needed_movement_x) < 0.1:
        print(f"  ✓ X direction correct")
    else:
        print(f"  ✗ X DIRECTION INVERTED!")

if __name__ == "__main__":
    main()
