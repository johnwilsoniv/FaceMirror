#!/usr/bin/env python3
"""
Direct comparison: OLD parameters vs NEW parameters on the SAME frame.
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


def run_cpp_openface(image_path, output_dir):
    """Run C++ OpenFace."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_binary = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [str(cpp_binary), "-f", str(image_path), "-out_dir", str(output_dir), "-2Dfp", "-wild"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV from C++ OpenFace")

    with open(csv_files[0], 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_row = next(reader)

    x_indices = [i for i, h in enumerate(header) if h.strip().startswith('x_') and h.strip()[2:].isdigit()]
    y_indices = [i for i, h in enumerate(header) if h.strip().startswith('y_') and h.strip()[2:].isdigit()]
    x_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))
    y_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))

    landmarks = [[float(data_row[x_idx]), float(data_row[y_idx])] for x_idx, y_idx in zip(x_indices, y_indices)]
    return np.array(landmarks)


def calculate_error(landmarks, cpp_landmarks):
    """Calculate error metrics."""
    distances = np.sqrt(np.sum((landmarks - cpp_landmarks)**2, axis=1))
    return {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'max': np.max(distances),
    }


def main():
    print("\n" + "="*80)
    print("A/B TEST: OLD PARAMETERS vs NEW PARAMETERS")
    print("="*80)

    # Use first frame
    image_path = Path("test_output/baseline_fast/frames/IMG_0422_frame_0000.jpg")

    if not image_path.exists():
        print(f"\nExtracting test frame...")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", "Patient Data/Normal Cohort/IMG_0422.MOV",
            "-vframes", "1", "-y", str(image_path)
        ], capture_output=True)

    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"\nTest image: {image_path.name}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")

    # Get C++ OpenFace ground truth
    print("\n[1/3] Running C++ OpenFace (ground truth)...")
    cpp_output_dir = Path("test_output/param_comparison/cpp_output")
    cpp_landmarks = run_cpp_openface(image_path, cpp_output_dir)
    print(f"     ✓ C++ OpenFace: {len(cpp_landmarks)} landmarks")

    # Estimate bbox from C++ landmarks
    x_min, y_min = cpp_landmarks.min(axis=0)
    x_max, y_max = cpp_landmarks.max(axis=0)
    margin = 0.1
    width = x_max - x_min
    height = y_max - y_min
    x_min -= margin * width
    y_min -= margin * height
    width *= (1 + 2*margin)
    height *= (1 + 2*margin)
    bbox = (x_min, y_min, width, height)

    # Test OLD parameters
    print("\n[2/3] Running pyCLNF with OLD parameters...")
    print("     (max_iterations=10, convergence_threshold=0.005)")

    t0 = time.time()
    clnf_old = CLNF(max_iterations=10, convergence_threshold=0.005, detector=None)
    old_landmarks, old_info = clnf_old.fit(gray, bbox, return_params=True)
    old_time = time.time() - t0

    old_error = calculate_error(old_landmarks, cpp_landmarks)
    print(f"     ✓ Completed in {old_time:.2f}s")
    print(f"       Iterations: {old_info['iterations']}, Converged: {old_info['converged']}")
    print(f"       Mean error: {old_error['mean']:.3f}px")

    # Test NEW parameters
    print("\n[3/3] Running pyCLNF with NEW parameters...")
    print("     (max_iterations=20, convergence_threshold=0.01)")

    t0 = time.time()
    clnf_new = CLNF(max_iterations=20, convergence_threshold=0.01, detector=None)
    new_landmarks, new_info = clnf_new.fit(gray, bbox, return_params=True)
    new_time = time.time() - t0

    new_error = calculate_error(new_landmarks, cpp_landmarks)
    print(f"     ✓ Completed in {new_time:.2f}s")
    print(f"       Iterations: {new_info['iterations']}, Converged: {new_info['converged']}")
    print(f"       Mean error: {new_error['mean']:.3f}px")

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nOLD Parameters (max_iter=10, threshold=0.005):")
    print(f"  Mean error:   {old_error['mean']:.3f}px")
    print(f"  Median error: {old_error['median']:.3f}px")
    print(f"  Max error:    {old_error['max']:.3f}px")
    print(f"  Runtime:      {old_time:.2f}s")
    print(f"  Iterations:   {old_info['iterations']}")

    print(f"\nNEW Parameters (max_iter=20, threshold=0.01):")
    print(f"  Mean error:   {new_error['mean']:.3f}px")
    print(f"  Median error: {new_error['median']:.3f}px")
    print(f"  Max error:    {new_error['max']:.3f}px")
    print(f"  Runtime:      {new_time:.2f}s")
    print(f"  Iterations:   {new_info['iterations']}")

    print(f"\nChange:")
    error_change = new_error['mean'] - old_error['mean']
    error_pct = (error_change / old_error['mean']) * 100
    print(f"  Error delta:  {error_change:+.3f}px ({error_pct:+.1f}%)")
    print(f"  Runtime delta: {new_time - old_time:+.2f}s ({((new_time/old_time)-1)*100:+.1f}%)")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if new_error['mean'] < old_error['mean']:
        improvement = ((old_error['mean'] - new_error['mean']) / old_error['mean']) * 100
        print(f"✅ NEW parameters are BETTER by {improvement:.1f}%")
    elif new_error['mean'] > old_error['mean']:
        degradation = ((new_error['mean'] - old_error['mean']) / old_error['mean']) * 100
        print(f"❌ NEW parameters are WORSE by {degradation:.1f}%")
        print(f"   The C++ OpenFace parameters cause pyCLNF to diverge!")
        print(f"   RECOMMENDATION: Revert to OLD parameters or investigate why.")
    else:
        print(f"➡️  No significant difference")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
