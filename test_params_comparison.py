#!/usr/bin/env python3
"""
Compare Python and C++ face parameters.

Check if there's a systematic scale/translation difference.
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


def test_params():
    """Compare face geometry parameters."""

    print("="*70)
    print("FACE GEOMETRY COMPARISON")
    print("="*70)

    # Load test frame
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    # Get C++ ground truth
    frame_path = "/tmp/test_frame.png"
    cv2.imwrite(frame_path, frame)
    cpp = get_cpp_landmarks(frame_path)

    # Get Python landmarks
    detector = MTCNN()
    faces, _ = detector.detect(frame)
    bbox_np = faces[0]

    clnf = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=True, debug_mode=False)
    py, info = clnf.fit(frame, bbox_np, return_params=True)
    params = info['params']

    # Compute face geometry from landmarks
    print("\n" + "="*70)
    print("FACE WIDTH (ear to ear)")
    print("="*70)
    # Jaw contour points 0 and 16 are ear-to-ear
    py_width = py[16, 0] - py[0, 0]
    cpp_width = cpp[16, 0] - cpp[0, 0]
    print(f"Python: {py_width:.2f} px")
    print(f"C++:    {cpp_width:.2f} px")
    print(f"Ratio:  {py_width/cpp_width:.4f}")

    print("\n" + "="*70)
    print("EYE SEPARATION (left to right eye)")
    print("="*70)
    # Eye centers
    py_left_eye = np.mean(py[36:42], axis=0)
    py_right_eye = np.mean(py[42:48], axis=0)
    cpp_left_eye = np.mean(cpp[36:42], axis=0)
    cpp_right_eye = np.mean(cpp[42:48], axis=0)

    py_eye_sep = py_right_eye[0] - py_left_eye[0]
    cpp_eye_sep = cpp_right_eye[0] - cpp_left_eye[0]

    print(f"Python: {py_eye_sep:.2f} px")
    print(f"C++:    {cpp_eye_sep:.2f} px")
    print(f"Ratio:  {py_eye_sep/cpp_eye_sep:.4f}")

    print("\n" + "="*70)
    print("FACE CENTER (nose tip)")
    print("="*70)
    # Nose tip is landmark 30
    print(f"Python: ({py[30, 0]:.2f}, {py[30, 1]:.2f})")
    print(f"C++:    ({cpp[30, 0]:.2f}, {cpp[30, 1]:.2f})")
    print(f"Delta:  ({py[30, 0] - cpp[30, 0]:.2f}, {py[30, 1] - cpp[30, 1]:.2f})")

    print("\n" + "="*70)
    print("EYE CENTERS")
    print("="*70)
    print(f"\nLeft eye center:")
    print(f"  Python: ({py_left_eye[0]:.2f}, {py_left_eye[1]:.2f})")
    print(f"  C++:    ({cpp_left_eye[0]:.2f}, {cpp_left_eye[1]:.2f})")
    print(f"  Delta:  ({py_left_eye[0] - cpp_left_eye[0]:.2f}, {py_left_eye[1] - cpp_left_eye[1]:.2f})")

    print(f"\nRight eye center:")
    print(f"  Python: ({py_right_eye[0]:.2f}, {py_right_eye[1]:.2f})")
    print(f"  C++:    ({cpp_right_eye[0]:.2f}, {cpp_right_eye[1]:.2f})")
    print(f"  Delta:  ({py_right_eye[0] - cpp_right_eye[0]:.2f}, {py_right_eye[1] - cpp_right_eye[1]:.2f})")

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    # Check if issue is narrower face
    if py_width < cpp_width and py_eye_sep < cpp_eye_sep:
        print("\n⚠️  Python face is NARROWER than C++")
        print(f"   Face width ratio: {py_width/cpp_width:.4f}")
        print(f"   Eye separation ratio: {py_eye_sep/cpp_eye_sep:.4f}")
    elif py_width > cpp_width:
        print("\n⚠️  Python face is WIDER than C++")
    else:
        print("\n✓ Face widths are similar")

    # Check if eyes are shifted
    left_eye_x_delta = py_left_eye[0] - cpp_left_eye[0]
    right_eye_x_delta = py_right_eye[0] - cpp_right_eye[0]

    if left_eye_x_delta < 0 and right_eye_x_delta > 0:
        print(f"\n⚠️  Eyes are shifted INWARD (narrower)")
        print(f"   Left eye shifted left by {abs(left_eye_x_delta):.2f} px")
        print(f"   Right eye shifted right by {abs(right_eye_x_delta):.2f} px")


if __name__ == "__main__":
    test_params()
