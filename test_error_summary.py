#!/usr/bin/env python3
"""
Clear error summary comparing Python to C++ ground truth.
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


def main():
    """Print clear error summary."""

    print("="*70)
    print("PYTHON vs C++ ERROR SUMMARY")
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

    # With eye refinement
    clnf_eye = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=True, debug_mode=False)
    py_eye, _ = clnf_eye.fit(frame, bbox_np)

    # Without eye refinement
    clnf_no_eye = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=False, debug_mode=False)
    py_no_eye, _ = clnf_no_eye.fit(frame, bbox_np)

    # Compute errors
    errors_eye = np.linalg.norm(py_eye - cpp, axis=1)
    errors_no_eye = np.linalg.norm(py_no_eye - cpp, axis=1)

    print("\n" + "="*70)
    print("ERROR METRICS (comparing to C++ OpenFace ground truth)")
    print("="*70)

    print("\n┌─────────────────────────┬──────────────┬──────────────┐")
    print("│ Metric                  │  With Eye Ref│ Without Eye  │")
    print("├─────────────────────────┼──────────────┼──────────────┤")
    print(f"│ Mean error (all 68)     │ {np.mean(errors_eye):>10.4f} px│ {np.mean(errors_no_eye):>10.4f} px│")
    print(f"│ Max error               │ {np.max(errors_eye):>10.4f} px│ {np.max(errors_no_eye):>10.4f} px│")
    print(f"│ Median error            │ {np.median(errors_eye):>10.4f} px│ {np.median(errors_no_eye):>10.4f} px│")
    print("├─────────────────────────┼──────────────┼──────────────┤")

    # By region
    left_eye_err = np.mean(errors_eye[36:42])
    left_eye_err_no = np.mean(errors_no_eye[36:42])
    right_eye_err = np.mean(errors_eye[42:48])
    right_eye_err_no = np.mean(errors_no_eye[42:48])

    print(f"│ Left eye (36-41)        │ {left_eye_err:>10.4f} px│ {left_eye_err_no:>10.4f} px│")
    print(f"│ Right eye (42-47)       │ {right_eye_err:>10.4f} px│ {right_eye_err_no:>10.4f} px│")
    print("└─────────────────────────┴──────────────┴──────────────┘")

    print("\n" + "="*70)
    print("COMPARISON TO TARGET")
    print("="*70)

    target = 0.5
    print(f"\nTarget: < {target} px mean error")
    print(f"Current: {np.mean(errors_eye):.4f} px")
    print(f"Gap: {np.mean(errors_eye) - target:.4f} px")

    if np.mean(errors_eye) < target:
        print("\n✓ TARGET ACHIEVED!")
    else:
        print(f"\n⚠️  Need to reduce error by {(np.mean(errors_eye) - target) / np.mean(errors_eye) * 100:.1f}%")

    print("\n" + "="*70)
    print("KEY POINTS")
    print("="*70)
    print("\n• C++ error is 0 px (comparing to itself)")
    print(f"• Python error is {np.mean(errors_eye):.4f} px (comparing to C++)")
    print(f"• Eye refinement improved error by {np.mean(errors_no_eye) - np.mean(errors_eye):.4f} px")
    print(f"• Worst region: Left eye at {left_eye_err:.4f} px")


if __name__ == "__main__":
    main()
