#!/usr/bin/env python3
"""
Test all 68 landmarks accuracy grouped by face region.
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


def test_all_landmarks():
    """Test all landmarks by face region."""

    # Face regions based on 68-point landmark model
    regions = {
        'Jaw (0-16)': range(0, 17),
        'Left Eyebrow (17-21)': range(17, 22),
        'Right Eyebrow (22-26)': range(22, 27),
        'Nose Bridge (27-30)': range(27, 31),
        'Nose Tip (31-35)': range(31, 36),
        'Left Eye (36-41)': range(36, 42),
        'Right Eye (42-47)': range(42, 48),
        'Outer Lip (48-59)': range(48, 60),
        'Inner Lip (60-67)': range(60, 68),
    }

    print("="*70)
    print("LANDMARK ACCURACY BY FACE REGION")
    print("="*70)

    # Load test frame
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    # Get C++ ground truth
    frame_path = "/tmp/test_frame.png"
    cv2.imwrite(frame_path, frame)
    cpp_landmarks = get_cpp_landmarks(frame_path)

    # Initialize detector
    detector = MTCNN()
    faces, _ = detector.detect(frame)
    bbox_np = faces[0]

    # Get Python landmarks with eye refinement
    clnf = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=True, debug_mode=False)
    py_landmarks, _ = clnf.fit(frame, bbox_np)

    # Calculate errors by region
    print(f"\n{'Region':>25} {'Count':>6} {'Mean Err':>10} {'Max Err':>10} {'Max LM':>8}")
    print("-"*65)

    all_errors = []
    for name, indices in regions.items():
        errors = [np.linalg.norm(py_landmarks[i] - cpp_landmarks[i]) for i in indices]
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        max_idx = list(indices)[np.argmax(errors)]
        all_errors.extend(errors)
        print(f"{name:>25} {len(indices):6d} {mean_err:10.4f} {max_err:10.4f} {max_idx:8d}")

    print("-"*65)
    print(f"{'TOTAL':>25} {68:6d} {np.mean(all_errors):10.4f} {np.max(all_errors):10.4f}")

    # Identify worst landmarks
    print("\n" + "="*70)
    print("TOP 10 WORST LANDMARKS")
    print("="*70)

    errors = [(i, np.linalg.norm(py_landmarks[i] - cpp_landmarks[i])) for i in range(68)]
    errors.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'LM':>4} {'Error':>10} {'Python':>20} {'C++':>20}")
    print("-"*60)
    for i, err in errors[:10]:
        print(f"{i:4d} {err:10.4f} ({py_landmarks[i, 0]:8.2f}, {py_landmarks[i, 1]:8.2f}) "
              f"({cpp_landmarks[i, 0]:8.2f}, {cpp_landmarks[i, 1]:8.2f})")


if __name__ == "__main__":
    test_all_landmarks()
