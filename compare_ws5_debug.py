#!/usr/bin/env python3
"""
Compare WS=5 and eye model debugging between C++ and Python.

This script runs both C++ OpenFace and Python pyclnf on the same frame
with window size 5 enabled, then compares the debug output to understand
why WS=5 causes overfitting in Python.
"""

import subprocess
import cv2
import numpy as np
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pyclnf import CLNF

# Paths
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
OPENFACE_BIN = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

def run_cpp_openface(frame_path, frame_idx):
    """Run C++ OpenFace on a single frame."""
    print("\n" + "="*60)
    print("Running C++ OpenFace with WS=5 debugging...")
    print("="*60)

    # Extract frame to temp file
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return None

    # Save frame
    cv2.imwrite(frame_path, frame)

    # Run OpenFace
    cmd = [
        OPENFACE_BIN,
        '-f', frame_path,
        '-out_dir', '/tmp',
        '-pose', '-2Dfp', '-3Dfp',
        '-verbose'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"OpenFace error: {result.stderr}")
        return None

    print("C++ OpenFace completed")
    return True

def run_python_pyclnf(frame_idx):
    """Run Python pyclnf on a frame with WS=5 enabled."""
    print("\n" + "="*60)
    print("Running Python pyclnf with WS=5 debugging...")
    print("="*60)

    # Load frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize CLNF with WS=5 enabled
    clnf = CLNF(
        model_dir="pyclnf/models",
        max_iterations=40,
        window_sizes=[11, 9, 7, 5],  # Include WS=5 for debugging
        use_eye_refinement=True  # Enable eye model debugging
    )

    # Use a known bbox (same as C++ would use)
    # For frame 0, use this approximate bbox
    bbox = np.array([349.354, 722.107, 395.049, 427.220])

    # Run CLNF
    landmarks, info = clnf.fit(gray, bbox)

    print(f"Python completed: {info['iterations']} iterations")
    return landmarks

def compare_debug_files():
    """Compare the debug output files."""
    print("\n" + "="*60)
    print("COMPARISON OF WS=5 DEBUG OUTPUT")
    print("="*60)

    # Read C++ WS=5 debug
    cpp_ws5_path = Path('/tmp/cpp_ws5_debug.txt')
    if cpp_ws5_path.exists():
        print("\n--- C++ WS=5 Debug ---")
        print(cpp_ws5_path.read_text()[:2000])
    else:
        print("\nNo C++ WS=5 debug file found")

    # Read Python WS=5 debug
    py_ws5_path = Path('/tmp/python_ws5_debug.txt')
    if py_ws5_path.exists():
        print("\n--- Python WS=5 Debug ---")
        print(py_ws5_path.read_text()[:2000])
    else:
        print("\nNo Python WS=5 debug file found")

    # Read C++ eye model debug
    cpp_eye_path = Path('/tmp/cpp_eye_model_debug.txt')
    if cpp_eye_path.exists():
        print("\n--- C++ Eye Model Debug ---")
        print(cpp_eye_path.read_text()[:1000])
    else:
        print("\nNo C++ eye model debug file found")

    # Read Python eye model debug
    py_eye_path = Path('/tmp/python_eye_model_debug.txt')
    if py_eye_path.exists():
        print("\n--- Python Eye Model Debug ---")
        print(py_eye_path.read_text()[:1000])
    else:
        print("\nNo Python eye model debug file found")

def main():
    """Main comparison function."""
    frame_idx = 0
    frame_path = '/tmp/test_frame.png'

    print("="*60)
    print("WS=5 and Eye Model Debugging Comparison")
    print("="*60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Frame: {frame_idx}")

    # Clear old debug files
    for f in ['/tmp/cpp_ws5_debug.txt', '/tmp/python_ws5_debug.txt',
              '/tmp/cpp_eye_model_debug.txt', '/tmp/python_eye_model_debug.txt']:
        Path(f).unlink(missing_ok=True)

    # Run C++ first
    run_cpp_openface(frame_path, frame_idx)

    # Run Python
    run_python_pyclnf(frame_idx)

    # Compare debug files
    compare_debug_files()

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)

if __name__ == "__main__":
    main()
