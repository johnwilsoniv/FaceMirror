#!/usr/bin/env python3
"""
Test all eye landmarks accuracy with and without refinement.
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


def test_eye_landmarks():
    """Test all eye landmarks."""

    print("="*70)
    print("ALL EYE LANDMARKS ACCURACY TEST")
    print("="*70)

    # Load test frame
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read video frame")
        return

    # Get C++ ground truth
    frame_path = "/tmp/test_frame.png"
    cv2.imwrite(frame_path, frame)
    cpp_landmarks = get_cpp_landmarks(frame_path)
    if cpp_landmarks is None:
        print("ERROR: Could not get C++ landmarks")
        return

    # Initialize detector
    detector = MTCNN()
    faces, _ = detector.detect(frame)
    bbox_np = faces[0]

    # Get Python landmarks with and without eye refinement
    clnf_eye = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=True, debug_mode=False)
    clnf_no_eye = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=False, debug_mode=False)

    py_eye, _ = clnf_eye.fit(frame, bbox_np)
    py_no_eye, _ = clnf_no_eye.fit(frame, bbox_np)

    # Analyze left eye (36-41)
    print("\n" + "="*70)
    print("LEFT EYE (landmarks 36-41)")
    print("="*70)
    print(f"\n{'LM':>4} {'C++ X':>10} {'No-eye X':>10} {'Eye X':>10} {'Err no-eye':>10} {'Err eye':>10} {'Improved':>10}")
    print("-"*74)

    for i in range(36, 42):
        err_no_eye = np.linalg.norm(py_no_eye[i] - cpp_landmarks[i])
        err_eye = np.linalg.norm(py_eye[i] - cpp_landmarks[i])
        improved = "✓" if err_eye < err_no_eye else "✗"
        print(f"{i:4d} {cpp_landmarks[i, 0]:10.4f} {py_no_eye[i, 0]:10.4f} {py_eye[i, 0]:10.4f} "
              f"{err_no_eye:10.4f} {err_eye:10.4f} {improved:>10}")

    # Mean for left eye
    err_no_eye_left = np.mean([np.linalg.norm(py_no_eye[i] - cpp_landmarks[i]) for i in range(36, 42)])
    err_eye_left = np.mean([np.linalg.norm(py_eye[i] - cpp_landmarks[i]) for i in range(36, 42)])
    print("-"*74)
    print(f"{'Mean':>4} {'-':>10} {'-':>10} {'-':>10} {err_no_eye_left:10.4f} {err_eye_left:10.4f}")

    # Analyze right eye (42-47)
    print("\n" + "="*70)
    print("RIGHT EYE (landmarks 42-47)")
    print("="*70)
    print(f"\n{'LM':>4} {'C++ X':>10} {'No-eye X':>10} {'Eye X':>10} {'Err no-eye':>10} {'Err eye':>10} {'Improved':>10}")
    print("-"*74)

    for i in range(42, 48):
        err_no_eye = np.linalg.norm(py_no_eye[i] - cpp_landmarks[i])
        err_eye = np.linalg.norm(py_eye[i] - cpp_landmarks[i])
        improved = "✓" if err_eye < err_no_eye else "✗"
        print(f"{i:4d} {cpp_landmarks[i, 0]:10.4f} {py_no_eye[i, 0]:10.4f} {py_eye[i, 0]:10.4f} "
              f"{err_no_eye:10.4f} {err_eye:10.4f} {improved:>10}")

    # Mean for right eye
    err_no_eye_right = np.mean([np.linalg.norm(py_no_eye[i] - cpp_landmarks[i]) for i in range(42, 48)])
    err_eye_right = np.mean([np.linalg.norm(py_eye[i] - cpp_landmarks[i]) for i in range(42, 48)])
    print("-"*74)
    print(f"{'Mean':>4} {'-':>10} {'-':>10} {'-':>10} {err_no_eye_right:10.4f} {err_eye_right:10.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nLeft eye mean error:  {err_no_eye_left:.4f} → {err_eye_left:.4f} px "
          f"({'↓ IMPROVED' if err_eye_left < err_no_eye_left else '↑ WORSE'})")
    print(f"Right eye mean error: {err_no_eye_right:.4f} → {err_eye_right:.4f} px "
          f"({'↓ IMPROVED' if err_eye_right < err_no_eye_right else '↑ WORSE'})")


if __name__ == "__main__":
    test_eye_landmarks()
