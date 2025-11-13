#!/usr/bin/env python3
"""
Test Tier 2 Model Integration

Tests the integrated Tier 2 bbox correction model on the difficult IMG_0422 frame
that previously had 45.9px center offset with V2 correction.

Expected results with Tier 2 model:
- Significantly reduced center offset (~51% improvement vs raw RetinaFace)
- Better pyCLNF convergence
- Improved final landmark accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))
sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

import cv2
import numpy as np
import subprocess
import csv

def run_cpp_openface_video_mode(image_path, output_dir):
    """Run C++ OpenFace in video mode for ground truth."""
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

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse CSV
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
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

    return np.array(landmarks)


def calculate_error(landmarks, cpp_landmarks):
    """Calculate per-landmark errors."""
    distances = np.sqrt(np.sum((landmarks - cpp_landmarks)**2, axis=1))

    return {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'max': np.max(distances),
        'std': np.std(distances),
    }


def main():
    print("="*80)
    print("TIER 2 MODEL INTEGRATION TEST")
    print("="*80)
    print("Testing on difficult frame: IMG_0422 (previously 45.9px center offset)")
    print()

    # Load test image
    image_path = Path("calibration_frames/patient1_frame1.jpg")
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"Test image: {image_path.name}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")
    print()

    # Run C++ OpenFace for ground truth
    print("[1/2] Running C++ OpenFace (ground truth)...")
    cpp_output_dir = Path("test_output/tier2_test/cpp_output")
    cpp_landmarks = run_cpp_openface_video_mode(image_path, cpp_output_dir)
    print(f"     ✓ C++ OpenFace: {len(cpp_landmarks)} landmarks")
    print()

    # Run Python pyCLNF with Tier 2 model
    print("[2/2] Running Python pyCLNF with Tier 2 bbox correction...")
    from pyclnf import CLNF

    # Initialize CLNF with RetinaFace + Tier 2 correction (default)
    clnf = CLNF(max_iterations=10, convergence_threshold=0.005, detector="retinaface")

    # Detect and correct bbox
    corrected_bboxes = clnf.detector.detect_and_correct(image)
    if len(corrected_bboxes) == 0:
        raise RuntimeError("RetinaFace did not detect any faces")

    bbox = corrected_bboxes[0]
    print(f"     Detected bbox: {bbox}")

    # Fit CLNF
    py_landmarks, py_info = clnf.fit(gray, bbox, return_params=True)
    print(f"     ✓ pyCLNF: {len(py_landmarks)} landmarks ({py_info['iterations']} iterations)")
    print()

    # Calculate accuracy
    error = calculate_error(py_landmarks, cpp_landmarks)

    # Results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nLandmark Accuracy:")
    print(f"  Mean error:     {error['mean']:.3f}px")
    print(f"  Median error:   {error['median']:.3f}px")
    print(f"  Max error:      {error['max']:.3f}px")
    print(f"  Std deviation:  {error['std']:.3f}px")

    print(f"\nConvergence:")
    print(f"  Iterations:     {py_info['iterations']}")
    print(f"  Converged:      {py_info['converged']}")
    print(f"  Final update:   {py_info['final_update']:.6f}")

    print(f"\nComparison:")
    print(f"  Previous V2 (IMG_0422):  ~27.5px mean error")
    print(f"  Tier 2 Model:            {error['mean']:.2f}px mean error")
    if error['mean'] < 27.5:
        improvement = (27.5 - error['mean']) / 27.5 * 100
        print(f"  Improvement:             {improvement:.1f}%")

    print("\n" + "="*80)

    if error['mean'] < 10.0:
        print("✅ EXCELLENT - Tier 2 model working well!")
    elif error['mean'] < 15.0:
        print("✅ GOOD - Significant improvement over V2")
    else:
        print("⚠️  Model loaded but may need further tuning")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
