#!/usr/bin/env python3
"""
Visualize Python vs C++ landmarks on the face image.
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


def visualize():
    """Create visualization of Python vs C++ landmarks."""

    print("="*70)
    print("LANDMARK VISUALIZATION")
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
    cpp = get_cpp_landmarks(frame_path)
    if cpp is None:
        print("ERROR: Could not get C++ landmarks")
        return

    # Get Python landmarks
    detector = MTCNN()
    faces, _ = detector.detect(frame)
    bbox_np = faces[0]

    clnf = CLNF(model_dir="pyclnf/models", max_iterations=40, use_eye_refinement=True, debug_mode=False)
    py, _ = clnf.fit(frame, bbox_np)

    # Create visualization
    vis = frame.copy()

    # Draw all landmarks
    for i in range(68):
        # C++ in green
        cv2.circle(vis, (int(cpp[i, 0]), int(cpp[i, 1])), 2, (0, 255, 0), -1)
        # Python in red
        cv2.circle(vis, (int(py[i, 0]), int(py[i, 1])), 2, (0, 0, 255), -1)

    # Draw lines connecting same landmarks (shows error)
    for i in range(68):
        pt1 = (int(cpp[i, 0]), int(cpp[i, 1]))
        pt2 = (int(py[i, 0]), int(py[i, 1]))
        cv2.line(vis, pt1, pt2, (255, 255, 0), 1)  # Yellow line shows error

    # Add legend
    cv2.putText(vis, "Green: C++ OpenFace", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "Red: Python pyclnf", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, "Yellow: Error vectors", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Calculate and display error
    mean_error = np.mean(np.linalg.norm(py - cpp, axis=1))
    cv2.putText(vis, f"Mean error: {mean_error:.2f} px", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save full visualization
    output_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/landmark_comparison_full.png"
    cv2.imwrite(output_path, vis)
    print(f"\nFull face saved to: {output_path}")

    # Create zoomed eye region visualization
    # Left eye region
    left_eye_center = np.mean(cpp[36:42], axis=0)
    margin = 80
    x1 = int(max(0, left_eye_center[0] - margin))
    y1 = int(max(0, left_eye_center[1] - margin))
    x2 = int(min(frame.shape[1], left_eye_center[0] + margin))
    y2 = int(min(frame.shape[0], left_eye_center[1] + margin))

    eye_vis = frame[y1:y2, x1:x2].copy()

    # Scale up for better visibility
    scale = 3
    eye_vis = cv2.resize(eye_vis, (eye_vis.shape[1] * scale, eye_vis.shape[0] * scale), interpolation=cv2.INTER_LINEAR)

    # Draw left eye landmarks with larger markers
    for i in range(36, 42):
        # Adjust coordinates for crop and scale
        cpp_pt = ((int(cpp[i, 0]) - x1) * scale, (int(cpp[i, 1]) - y1) * scale)
        py_pt = ((int(py[i, 0]) - x1) * scale, (int(py[i, 1]) - y1) * scale)

        # C++ in green
        cv2.circle(eye_vis, cpp_pt, 5, (0, 255, 0), -1)
        cv2.putText(eye_vis, str(i), (cpp_pt[0] + 8, cpp_pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Python in red
        cv2.circle(eye_vis, py_pt, 5, (0, 0, 255), -1)

        # Error line
        cv2.line(eye_vis, cpp_pt, py_pt, (255, 255, 0), 2)

    # Add legend to eye view
    cv2.putText(eye_vis, "LEFT EYE ZOOM (3x)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(eye_vis, "Green=C++, Red=Python", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    eye_output_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/landmark_comparison_left_eye.png"
    cv2.imwrite(eye_output_path, eye_vis)
    print(f"Left eye zoom saved to: {eye_output_path}")

    # Print error summary
    print("\n" + "="*70)
    print("ERROR SUMMARY")
    print("="*70)
    print(f"\nMean error (all 68): {mean_error:.4f} px")

    left_eye_err = np.mean([np.linalg.norm(py[i] - cpp[i]) for i in range(36, 42)])
    right_eye_err = np.mean([np.linalg.norm(py[i] - cpp[i]) for i in range(42, 48)])
    print(f"Left eye mean error:  {left_eye_err:.4f} px")
    print(f"Right eye mean error: {right_eye_err:.4f} px")


if __name__ == "__main__":
    visualize()
