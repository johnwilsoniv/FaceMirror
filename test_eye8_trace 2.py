#!/usr/bin/env python3
"""
Test script to run both C++ and Python pipelines with eye refinement
to generate Eye_8 trace files for comparison.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
import subprocess
import os

def run_cpp_pipeline(image_path: str):
    """Run C++ FeatureExtraction to generate Eye_8 trace."""
    print("\n=== Running C++ Pipeline ===")

    out_dir = '/tmp/openface_eye8_test'
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
    import pandas as pd
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(out_dir, f'{base_name}.csv')

    if not os.path.exists(csv_path):
        print(f"Error: C++ output not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    landmarks = np.zeros((68, 2))
    for i in range(68):
        # Try different column name formats
        for x_col, y_col in [(f'x_{i}', f'y_{i}'), (f' x_{i}', f' y_{i}')]:
            if x_col in df.columns and y_col in df.columns:
                landmarks[i, 0] = df[x_col].iloc[0]
                landmarks[i, 1] = df[y_col].iloc[0]
                break

    print(f"C++ LM36 (left eye outer): ({landmarks[36, 0]:.4f}, {landmarks[36, 1]:.4f})")
    print(f"C++ trace saved to: /tmp/eye8_trace_cpp.txt")

    return landmarks

def run_python_pipeline(image: np.ndarray):
    """Run Python pipeline with eye refinement to generate Eye_8 trace."""
    print("\n=== Running Python Pipeline ===")

    from pyclnf.clnf import CLNF

    # Enable eye refinement
    clnf = CLNF(
        'pyclnf/models',
        regularization=40,
        use_eye_refinement=True
    )

    result = clnf.detect_and_fit(image)
    if result is None or result[0] is None:
        print("Python face detection/fitting failed")
        return None

    landmarks = result[0]
    print(f"Python LM36 (left eye outer): ({landmarks[36, 0]:.4f}, {landmarks[36, 1]:.4f})")
    print(f"Python trace saved to: /tmp/eye8_trace_python.txt")

    return landmarks

def compare_traces():
    """Compare the Eye_8 trace files."""
    print("\n=== Comparing Eye_8 Traces ===")

    cpp_trace_path = '/tmp/eye8_trace_cpp.txt'
    python_trace_path = '/tmp/eye8_trace_python.txt'

    if not os.path.exists(cpp_trace_path):
        print(f"Error: C++ trace not found at {cpp_trace_path}")
        return

    if not os.path.exists(python_trace_path):
        print(f"Error: Python trace not found at {python_trace_path}")
        return

    print(f"\nC++ trace ({cpp_trace_path}):")
    print("=" * 60)
    with open(cpp_trace_path, 'r') as f:
        print(f.read())

    print(f"\nPython trace ({python_trace_path}):")
    print("=" * 60)
    with open(python_trace_path, 'r') as f:
        print(f.read())

def main():
    # Use comparison frame (known to have faces)
    image_path = 'comparison_frame_0000.jpg'

    if not os.path.exists(image_path):
        # Try test image
        image_path = 'test_image.jpg'

    if not os.path.exists(image_path):
        print(f"Error: No test image found")
        return

    print(f"Using test image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # Convert to grayscale for Python
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run both pipelines (Python CLNF expects grayscale or BGR)
    cpp_landmarks = run_cpp_pipeline(image_path)
    python_landmarks = run_python_pipeline(image)  # Pass BGR, CLNF will convert

    if cpp_landmarks is not None and python_landmarks is not None:
        # Compare LM36 (Eye_8 maps to this)
        diff = python_landmarks[36] - cpp_landmarks[36]
        error = np.linalg.norm(diff)

        print("\n=== LM36 Comparison ===")
        print(f"C++:    ({cpp_landmarks[36, 0]:.4f}, {cpp_landmarks[36, 1]:.4f})")
        print(f"Python: ({python_landmarks[36, 0]:.4f}, {python_landmarks[36, 1]:.4f})")
        print(f"Diff:   ({diff[0]:.4f}, {diff[1]:.4f})")
        print(f"Error:  {error:.4f} pixels")

    # Compare traces
    compare_traces()

if __name__ == '__main__':
    main()
