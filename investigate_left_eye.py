#!/usr/bin/env python3
"""
Investigate why left eye has higher error than right eye.

Possible causes:
1. Mirrored patch experts for left side landmarks
2. Asymmetric response maps
3. Eye model refinement differences
4. Main model optimization bias
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
from pyclnf.core.pdm import PDM
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


def investigate():
    """Deep dive into left eye error."""

    print("="*70)
    print("LEFT EYE INVESTIGATION")
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

    # Test 1: Compare left vs right eye with and without eye refinement
    print("\n" + "="*70)
    print("1. LEFT vs RIGHT EYE COMPARISON")
    print("="*70)

    clnf_eye = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=True, debug_mode=False)
    clnf_no_eye = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=False, debug_mode=False)

    py_eye, _ = clnf_eye.fit(frame, bbox_np)
    py_no_eye, _ = clnf_no_eye.fit(frame, bbox_np)

    # Detailed per-landmark comparison
    print("\nLeft Eye (36-41) - per landmark:")
    print(f"{'LM':>4} {'No-eye err':>12} {'Eye err':>12} {'Δ':>10} {'Direction':>12}")
    print("-"*54)

    left_improvements = []
    for i in range(36, 42):
        err_no = np.linalg.norm(py_no_eye[i] - cpp[i])
        err_eye = np.linalg.norm(py_eye[i] - cpp[i])
        delta = err_no - err_eye
        left_improvements.append(delta)
        direction = "improved" if delta > 0 else "worse"
        print(f"{i:4d} {err_no:12.4f} {err_eye:12.4f} {delta:+10.4f} {direction:>12}")

    print("\nRight Eye (42-47) - per landmark:")
    print(f"{'LM':>4} {'No-eye err':>12} {'Eye err':>12} {'Δ':>10} {'Direction':>12}")
    print("-"*54)

    right_improvements = []
    for i in range(42, 48):
        err_no = np.linalg.norm(py_no_eye[i] - cpp[i])
        err_eye = np.linalg.norm(py_eye[i] - cpp[i])
        delta = err_no - err_eye
        right_improvements.append(delta)
        direction = "improved" if delta > 0 else "worse"
        print(f"{i:4d} {err_no:12.4f} {err_eye:12.4f} {delta:+10.4f} {direction:>12}")

    print(f"\nMean improvement: Left={np.mean(left_improvements):.4f}, Right={np.mean(right_improvements):.4f}")

    # Test 2: Check error direction (X vs Y)
    print("\n" + "="*70)
    print("2. ERROR DIRECTION ANALYSIS (X vs Y components)")
    print("="*70)

    print("\nLeft Eye X and Y errors:")
    print(f"{'LM':>4} {'X err':>10} {'Y err':>10} {'Total':>10} {'Dominant':>10}")
    print("-"*50)

    for i in range(36, 42):
        x_err = py_eye[i, 0] - cpp[i, 0]
        y_err = py_eye[i, 1] - cpp[i, 1]
        total = np.sqrt(x_err**2 + y_err**2)
        dominant = "X" if abs(x_err) > abs(y_err) else "Y"
        print(f"{i:4d} {x_err:+10.4f} {y_err:+10.4f} {total:10.4f} {dominant:>10}")

    print("\nRight Eye X and Y errors:")
    print(f"{'LM':>4} {'X err':>10} {'Y err':>10} {'Total':>10} {'Dominant':>10}")
    print("-"*50)

    for i in range(42, 48):
        x_err = py_eye[i, 0] - cpp[i, 0]
        y_err = py_eye[i, 1] - cpp[i, 1]
        total = np.sqrt(x_err**2 + y_err**2)
        dominant = "X" if abs(x_err) > abs(y_err) else "Y"
        print(f"{i:4d} {x_err:+10.4f} {y_err:+10.4f} {total:10.4f} {dominant:>10}")

    # Test 3: Asymmetry analysis
    print("\n" + "="*70)
    print("3. X-DIRECTION ASYMMETRY ANALYSIS")
    print("="*70)

    # Calculate mean X error for each eye
    left_x_errors = [py_eye[i, 0] - cpp[i, 0] for i in range(36, 42)]
    right_x_errors = [py_eye[i, 0] - cpp[i, 0] for i in range(42, 48)]

    print(f"\nMean X error:")
    print(f"  Left eye:  {np.mean(left_x_errors):+.4f} px (all landmarks shifted LEFT)")
    print(f"  Right eye: {np.mean(right_x_errors):+.4f} px (mostly shifted RIGHT)")
    print(f"\nThis indicates both eyes are shifted INWARD toward the nose")

    # Calculate how much the eye centers differ
    py_left_center = np.mean(py_eye[36:42], axis=0)
    py_right_center = np.mean(py_eye[42:48], axis=0)
    cpp_left_center = np.mean(cpp[36:42], axis=0)
    cpp_right_center = np.mean(cpp[42:48], axis=0)

    py_eye_sep = py_right_center[0] - py_left_center[0]
    cpp_eye_sep = cpp_right_center[0] - cpp_left_center[0]

    print(f"\nEye separation:")
    print(f"  Python: {py_eye_sep:.2f} px")
    print(f"  C++:    {cpp_eye_sep:.2f} px")
    print(f"  Difference: {py_eye_sep - cpp_eye_sep:.2f} px")

    if py_eye_sep > cpp_eye_sep:
        print(f"\n⚠️  Python eyes are {py_eye_sep - cpp_eye_sep:.2f} px WIDER apart")
    else:
        print(f"\n⚠️  Python eyes are {cpp_eye_sep - py_eye_sep:.2f} px CLOSER together")

    # Test 4: Compare main model (no eye refinement) asymmetry
    print("\n" + "="*70)
    print("4. MAIN MODEL ASYMMETRY (before eye refinement)")
    print("="*70)

    left_no_eye = np.mean([np.linalg.norm(py_no_eye[i] - cpp[i]) for i in range(36, 42)])
    right_no_eye = np.mean([np.linalg.norm(py_no_eye[i] - cpp[i]) for i in range(42, 48)])

    print(f"\nMain model mean error:")
    print(f"  Left eye:  {left_no_eye:.4f} px")
    print(f"  Right eye: {right_no_eye:.4f} px")
    print(f"  Asymmetry: {left_no_eye - right_no_eye:.4f} px")

    if left_no_eye > right_no_eye:
        print(f"\n⚠️  Main model already has LEFT eye bias before eye refinement!")

    # Test 5: Eye refinement improvement ratio
    print("\n" + "="*70)
    print("5. EYE REFINEMENT EFFECTIVENESS")
    print("="*70)

    left_eye = np.mean([np.linalg.norm(py_eye[i] - cpp[i]) for i in range(36, 42)])
    right_eye = np.mean([np.linalg.norm(py_eye[i] - cpp[i]) for i in range(42, 48)])

    left_improvement = (left_no_eye - left_eye) / left_no_eye * 100
    right_improvement = (right_no_eye - right_eye) / right_no_eye * 100

    print(f"\nEye refinement improvement:")
    print(f"  Left eye:  {left_no_eye:.4f} → {left_eye:.4f} px ({left_improvement:.1f}% reduction)")
    print(f"  Right eye: {right_no_eye:.4f} → {right_eye:.4f} px ({right_improvement:.1f}% reduction)")

    if left_improvement < right_improvement:
        print(f"\n⚠️  Eye refinement is LESS effective on left eye!")
        print(f"    Right eye improved {right_improvement - left_improvement:.1f}% more")


if __name__ == "__main__":
    investigate()
