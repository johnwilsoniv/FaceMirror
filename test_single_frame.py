#!/usr/bin/env python3
"""
Single frame comparison: C++ OpenFace (video mode) vs Python pyCLNF (20 iterations - FULL QUALITY).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

import cv2
import numpy as np
import subprocess
import csv
import time
from pyclnf import CLNF


def run_cpp_openface_video_mode(image_path, output_dir):
    """Run C++ OpenFace in video mode (-wild flag)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_binary = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(cpp_binary),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-2Dfp",
        "-wild"  # Video mode
    ]

    print("\n[1/3] Running C++ OpenFace (video mode)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse CSV
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise FileNotFoundError("No CSV from C++ OpenFace")

    csv_path = csv_files[0]

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_row = next(reader)

    # Extract landmarks
    x_indices = [i for i, h in enumerate(header) if h.strip().startswith('x_') and h.strip()[2:].isdigit()]
    y_indices = [i for i, h in enumerate(header) if h.strip().startswith('y_') and h.strip()[2:].isdigit()]

    x_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))
    y_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))

    landmarks = []
    for x_idx, y_idx in zip(x_indices, y_indices):
        x = float(data_row[x_idx])
        y = float(data_row[y_idx])
        landmarks.append([x, y])

    print(f"     ✓ C++ OpenFace: {len(landmarks)} landmarks")
    return np.array(landmarks)


# REMOVED - No longer needed, CLNF will use RetinaFace detector directly


def calculate_error(landmarks, cpp_landmarks):
    """Calculate per-landmark errors."""
    distances = np.sqrt(np.sum((landmarks - cpp_landmarks)**2, axis=1))

    return {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'max': np.max(distances),
        'std': np.std(distances),
        'distances': distances
    }


def visualize_comparison(image, cpp_landmarks, py_landmarks, output_path):
    """Create side-by-side comparison visualization."""
    print("\n[3/3] Creating visualization...")

    h, w = image.shape[:2]

    # Create side-by-side image
    vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
    vis[:, :w] = image.copy()
    vis[:, w:] = image.copy()

    # Draw C++ landmarks (left, green)
    for x, y in cpp_landmarks:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Draw Python landmarks (right, blue)
    for x, y in py_landmarks:
        cv2.circle(vis, (int(x) + w, int(y)), 2, (255, 0, 0), -1)

    # Add labels
    cv2.putText(vis, "C++ OpenFace (video mode)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis, "Python pyCLNF", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imwrite(str(output_path), vis)
    print(f"     ✓ Saved to: {output_path}")

    return vis


def main():
    print("\n" + "="*70)
    print("SINGLE FRAME COMPARISON")
    print("="*70)
    print("C++ OpenFace (video mode) vs Python pyCLNF")
    print("="*70)

    # Use calibration frame (same as successful 8.23px test)
    image_path = Path("calibration_frames/patient1_frame1.jpg")

    if not image_path.exists():
        print(f"ERROR: Calibration frame not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    print(f"\nTest image: {image_path.name}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")

    # Run C++ OpenFace
    cpp_output_dir = Path("test_output/single_frame/cpp_output")
    cpp_landmarks = run_cpp_openface_video_mode(image_path, cpp_output_dir)

    # Run Python pyCLNF using RetinaFace detector (default)
    # This will use the calibrated RetinaFace correction factor
    print("\n[2/3] Running Python pyCLNF with RetinaFace detector...")
    print("     (Using calibrated RetinaFace correction)")

    import time
    start_time = time.time()

    # Initialize CLNF with RetinaFace detector (default)
    clnf = CLNF(max_iterations=20, convergence_threshold=0.01, detector="retinaface")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face using RetinaFace (with V2 correction)
    corrected_bboxes = clnf.detector.detect_and_correct(image)
    if len(corrected_bboxes) == 0:
        raise RuntimeError("RetinaFace did not detect any faces")

    # Use first detected face bbox (already corrected by RetinaFace V2)
    bbox = corrected_bboxes[0]  # (x, y, w, h)
    print(f"     RetinaFace detected face: bbox={bbox}")

    # Fit CLNF with detected bbox
    py_landmarks, py_info = clnf.fit(gray, bbox, return_params=True)

    elapsed = time.time() - start_time
    print(f"     ✓ Python pyCLNF: {len(py_landmarks)} landmarks ({py_info['iterations']} iterations, {elapsed:.2f}s)")

    # Calculate accuracy
    error = calculate_error(py_landmarks, cpp_landmarks)

    # Visualize
    vis_path = Path("test_output/single_frame/comparison.jpg")
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_comparison(image, cpp_landmarks, py_landmarks, vis_path)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nAccuracy:")
    print(f"  Mean error:     {error['mean']:.3f}px")
    print(f"  Median error:   {error['median']:.3f}px")
    print(f"  Max error:      {error['max']:.3f}px")
    print(f"  Std deviation:  {error['std']:.3f}px")

    print(f"\nConvergence:")
    print(f"  Iterations:     {py_info['iterations']}")
    print(f"  Converged:      {py_info['converged']}")
    print(f"  Final update:   {py_info['final_update']:.6f}")

    print(f"\nComparison to previous results:")
    print(f"  vs PyMTCNN (16.4px):    {((16.4 - error['mean'])/16.4)*100:+.1f}% better")
    print(f"  vs Previous (8.23px):   {((8.23 - error['mean'])/8.23)*100:+.1f}% change")

    print("\n" + "="*70)

    if error['mean'] < 5.0:
        print("✅ EXCELLENT accuracy!")
    elif error['mean'] < 10.0:
        print("✅ GOOD accuracy")
    else:
        print("⚠️  Higher error than expected")

    print(f"\nVisualization saved to: {vis_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
