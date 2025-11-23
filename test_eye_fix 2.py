#!/usr/bin/env python3
"""
Test the eye refinement fix.

Compares Python CLNF with eye refinement to C++ ground truth.
"""

import numpy as np
from pathlib import Path
import sys
import subprocess
import tempfile
import cv2

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf import CLNF
from pymtcnn import MTCNN


def get_cpp_landmarks(image_path: str) -> np.ndarray:
    """Get ground truth landmarks from C++ OpenFace."""
    openface_bin = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run([
            openface_bin,
            "-f", image_path,
            "-out_dir", tmpdir,
            "-2Dfp"
        ], capture_output=True, text=True)

        csv_file = Path(tmpdir) / (Path(image_path).stem + ".csv")
        if csv_file.exists():
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                landmarks = np.zeros((68, 2), dtype=np.float32)
                for i in range(68):
                    landmarks[i, 0] = float(row[f"x_{i}"])
                    landmarks[i, 1] = float(row[f"y_{i}"])
                return landmarks
    return None


def test_eye_fix():
    """Test eye refinement fix."""

    print("="*70)
    print("EYE REFINEMENT FIX TEST")
    print("="*70)

    # Use first frame from video
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read video frame")
        return

    # Save frame for OpenFace
    frame_path = "/tmp/test_frame.png"
    cv2.imwrite(frame_path, frame)

    # Get C++ ground truth
    print("\n1. Getting C++ OpenFace landmarks...")
    cpp_landmarks = get_cpp_landmarks(frame_path)
    if cpp_landmarks is None:
        print("ERROR: Could not get C++ landmarks")
        return
    print(f"   C++ LM36: ({cpp_landmarks[36, 0]:.4f}, {cpp_landmarks[36, 1]:.4f})")

    # Initialize detector
    detector = MTCNN()
    faces, _ = detector.detect(frame)
    if len(faces) == 0:
        print("ERROR: No face detected")
        return
    bbox_np = faces[0]

    # Test WITH eye refinement
    print("\n2. Python CLNF WITH eye refinement (using fix)...")
    clnf_eye = CLNF(
        model_dir="pyclnf/models",
        max_iterations=40,
        use_eye_refinement=True,
        debug_mode=False
    )
    py_eye_landmarks, _ = clnf_eye.fit(frame, bbox_np)
    error_eye = np.mean(np.linalg.norm(py_eye_landmarks - cpp_landmarks, axis=1))
    error_36_eye = np.linalg.norm(py_eye_landmarks[36] - cpp_landmarks[36])
    print(f"   Python LM36: ({py_eye_landmarks[36, 0]:.4f}, {py_eye_landmarks[36, 1]:.4f})")
    print(f"   Error to C++: LM36={error_36_eye:.4f} px, Mean={error_eye:.4f} px")

    # Test WITHOUT eye refinement
    print("\n3. Python CLNF WITHOUT eye refinement...")
    clnf_no_eye = CLNF(
        model_dir="pyclnf/models",
        max_iterations=40,
        use_eye_refinement=False,
        debug_mode=False
    )
    py_no_eye_landmarks, _ = clnf_no_eye.fit(frame, bbox_np)
    error_no_eye = np.mean(np.linalg.norm(py_no_eye_landmarks - cpp_landmarks, axis=1))
    error_36_no_eye = np.linalg.norm(py_no_eye_landmarks[36] - cpp_landmarks[36])
    print(f"   Python LM36: ({py_no_eye_landmarks[36, 0]:.4f}, {py_no_eye_landmarks[36, 1]:.4f})")
    print(f"   Error to C++: LM36={error_36_no_eye:.4f} px, Mean={error_no_eye:.4f} px")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nC++ LM36:                ({cpp_landmarks[36, 0]:.4f}, {cpp_landmarks[36, 1]:.4f})")
    print(f"Python with eye ref:     ({py_eye_landmarks[36, 0]:.4f}, {py_eye_landmarks[36, 1]:.4f}) - Error: {error_36_eye:.4f} px")
    print(f"Python without eye ref:  ({py_no_eye_landmarks[36, 0]:.4f}, {py_no_eye_landmarks[36, 1]:.4f}) - Error: {error_36_no_eye:.4f} px")

    print(f"\nMean error across all 68 landmarks:")
    print(f"  With eye refinement:    {error_eye:.4f} px")
    print(f"  Without eye refinement: {error_no_eye:.4f} px")

    if error_eye < error_no_eye:
        print(f"\n✓ Eye refinement IMPROVED accuracy by {error_no_eye - error_eye:.4f} px")
    else:
        print(f"\n⚠️  Eye refinement DECREASED accuracy by {error_eye - error_no_eye:.4f} px")

    # Check X direction for landmark 36
    cpp_36_x = cpp_landmarks[36, 0]
    py_eye_36_x = py_eye_landmarks[36, 0]
    py_no_eye_36_x = py_no_eye_landmarks[36, 0]

    print(f"\nLM36 X-direction analysis:")
    print(f"  C++ target X:    {cpp_36_x:.4f}")
    print(f"  Python no-eye X: {py_no_eye_36_x:.4f} (delta to C++: {py_no_eye_36_x - cpp_36_x:+.4f})")
    print(f"  Python eye X:    {py_eye_36_x:.4f} (delta to C++: {py_eye_36_x - cpp_36_x:+.4f})")

    # Did eye refinement move in the right direction?
    needed_movement = cpp_36_x - py_no_eye_36_x
    actual_movement = py_eye_36_x - py_no_eye_36_x
    print(f"\n  Needed X movement: {needed_movement:+.4f}")
    print(f"  Actual X movement: {actual_movement:+.4f}")

    if needed_movement * actual_movement > 0:
        print(f"  ✓ Eye refinement moved in correct direction!")
    else:
        print(f"  ⚠️  Eye refinement moved in WRONG direction!")


if __name__ == "__main__":
    test_eye_fix()
