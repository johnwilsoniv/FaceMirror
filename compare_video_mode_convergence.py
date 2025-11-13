#!/usr/bin/env python3
"""
Compare C++ OpenFace (video mode) vs Python pyCLNF convergence.
Focus on:
1. Iteration counts
2. Convergence criteria
3. Coordinate system consistency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

import cv2
import numpy as np
import subprocess
import csv
from pyclnf import CLNF
from pyclnf.core.pdm import PDM


def run_cpp_video_mode(image_path, output_dir):
    """Run C++ OpenFace in video mode (no face checking, similar to our use)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_binary = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    # Video mode flags: -wild (for video mode optimization)
    cmd = [
        str(cpp_binary),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-2Dfp", "-pdmparams", "-tracked",
        "-wild"  # Use in-the-wild/video mode settings
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\n=== C++ OpenFace Output ===")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print("="*50)

    # Parse CSV
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV from C++")

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

    landmarks = np.array(landmarks)

    # Extract PDM parameters if available
    params = None
    param_indices = [i for i, h in enumerate(header) if h.strip().startswith('p_') and h.strip()[2:].isdigit()]
    if param_indices:
        param_indices.sort(key=lambda i: int(header[i].strip().split('_')[1]))
        params = np.array([float(data_row[i]) for i in param_indices])
        print(f"\nC++ PDM parameters ({len(params)} values):")
        print(f"  Scale (p_0): {params[0]:.6f}")
        if len(params) > 1:
            print(f"  Rotation: {params[1:4]}")
            print(f"  Translation: {params[4:6]}")
            print(f"  Shape params (first 5): {params[6:11]}")

    # Check for success/confidence
    success_idx = header.index('success') if 'success' in header else None
    confidence_idx = header.index('confidence') if 'confidence' in header else None

    success = float(data_row[success_idx]) if success_idx else 1.0
    confidence = float(data_row[confidence_idx]) if confidence_idx else 1.0

    return {
        'landmarks': landmarks,
        'params': params,
        'success': success,
        'confidence': confidence,
        'stdout': result.stdout
    }


def run_python_video_mode(image_path, bbox):
    """Run Python with explicit video mode settings."""
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize with video mode settings (weight_multiplier=0.0)
    clnf = CLNF(
        weight_multiplier=0.0,  # Video mode
        sigma=1.5,
        max_iterations=10,
        convergence_threshold=0.005,
        detector=None
    )

    # Get initial params for debugging
    init_params = clnf.pdm.init_params(bbox)
    print(f"\nPython initial PDM parameters:")
    print(f"  Scale (p_0): {init_params[0]:.6f}")
    print(f"  Rotation: {init_params[1:4]}")
    print(f"  Translation: {init_params[4:6]}")
    print(f"  Shape params (first 5): {init_params[6:11]}")

    # Fit
    landmarks, info = clnf.fit(gray, bbox, return_params=True)

    final_params = info['params']
    print(f"\nPython final PDM parameters:")
    print(f"  Scale (p_0): {final_params[0]:.6f}")
    print(f"  Rotation: {final_params[1:4]}")
    print(f"  Translation: {final_params[4:6]}")
    print(f"  Shape params (first 5): {final_params[6:11]}")

    return {
        'landmarks': landmarks,
        'init_params': init_params,
        'final_params': final_params,
        'info': info
    }


def check_coordinate_systems():
    """Check for coordinate system mismatches."""
    print("\n" + "="*70)
    print("COORDINATE SYSTEM VERIFICATION")
    print("="*70)

    # Load PDM
    pdm = PDM("pyclnf/models/exported_pdm")

    print(f"\n1. Mean shape storage:")
    print(f"   Raw shape: {pdm.mean_shape.shape}")
    print(f"   Expected: (204,) for [all X, all Y, all Z]")

    # Verify reshape
    mean_3d = pdm.mean_shape.reshape(3, -1)
    print(f"   Reshaped: {mean_3d.shape} (should be (3, 68))")

    # Check first few points
    print(f"\n2. First 3 landmarks in mean shape:")
    for i in range(3):
        x, y, z = mean_3d[0, i], mean_3d[1, i], mean_3d[2, i]
        print(f"   Landmark {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    # Verify 2D projection
    print(f"\n3. Testing 2D projection:")
    test_params = np.zeros(pdm.n_params)
    test_params[0] = 1.0  # Unit scale
    landmarks_2d = pdm.params_to_landmarks_2d(test_params)
    print(f"   Output shape: {landmarks_2d.shape} (should be (68, 2))")
    print(f"   First 3 landmarks:")
    for i in range(3):
        print(f"     Landmark {i}: x={landmarks_2d[i, 0]:.3f}, y={landmarks_2d[i, 1]:.3f}")

    # Check principal components
    print(f"\n4. Principal components:")
    print(f"   princ_comp shape: {pdm.princ_comp.shape}")
    print(f"   Expected: (204, n_modes) for dimension-major format")

    # Verify princ_comp is column-major (each column is a mode)
    print(f"   First mode norm: {np.linalg.norm(pdm.princ_comp[:, 0]):.3f}")
    print(f"   Second mode norm: {np.linalg.norm(pdm.princ_comp[:, 1]):.3f}")

    print("\n" + "="*70)


def analyze_convergence_details(cpp_data, py_data):
    """Detailed convergence analysis."""
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)

    # Compare parameters
    if cpp_data['params'] is not None and py_data['final_params'] is not None:
        cpp_params = cpp_data['params']
        py_params = py_data['final_params']

        print(f"\nParameter comparison (C++ vs Python):")
        print(f"  Number of params: C++={len(cpp_params)}, Python={len(py_params)}")

        # Compare key parameters
        param_names = ['Scale', 'Rx', 'Ry', 'Rz', 'Tx', 'Ty']
        for i, name in enumerate(param_names):
            if i < len(cpp_params) and i < len(py_params):
                diff = abs(cpp_params[i] - py_params[i])
                pct = (diff / abs(cpp_params[i]) * 100) if cpp_params[i] != 0 else 0
                print(f"  {name:6s}: C++={cpp_params[i]:10.6f}, Py={py_params[i]:10.6f}, "
                      f"diff={diff:8.6f} ({pct:6.2f}%)")

        # Compare shape parameters
        if len(cpp_params) > 6 and len(py_params) > 6:
            shape_diff = np.abs(cpp_params[6:] - py_params[6:])
            print(f"\n  Shape parameters:")
            print(f"    Mean diff: {np.mean(shape_diff):.6f}")
            print(f"    Max diff: {np.max(shape_diff):.6f}")
            print(f"    Std diff: {np.std(shape_diff):.6f}")

    # Landmark differences
    cpp_lm = cpp_data['landmarks']
    py_lm = py_data['landmarks']

    distances = np.sqrt(np.sum((py_lm - cpp_lm)**2, axis=1))
    print(f"\nLandmark errors:")
    print(f"  Mean: {np.mean(distances):.3f}px")
    print(f"  Median: {np.median(distances):.3f}px")
    print(f"  Max: {np.max(distances):.3f}px")
    print(f"  Std: {np.std(distances):.3f}px")

    # Find worst landmarks
    worst = np.argsort(distances)[-10:][::-1]
    print(f"\n  Worst 10 landmarks:")
    for idx in worst:
        print(f"    Landmark {idx:2d}: {distances[idx]:6.2f}px")

    # Check for systematic biases
    diff_vec = py_lm - cpp_lm
    mean_diff = np.mean(diff_vec, axis=0)
    print(f"\n  Systematic bias:")
    print(f"    Mean X offset: {mean_diff[0]:+.3f}px")
    print(f"    Mean Y offset: {mean_diff[1]:+.3f}px")

    # Check iteration info
    print(f"\nIteration info:")
    print(f"  Python iterations: {py_data['info']['iterations']}")
    print(f"  Python converged: {py_data['info']['converged']}")
    print(f"  Python final update: {py_data['info']['final_update']:.6f}")

    print("\n" + "="*70)


def main():
    print("\n🔍 VIDEO MODE CONVERGENCE ANALYSIS")
    print("="*70)

    # Test image
    image_path = Path("Patient Data/Normal Cohort/IMG_0434.MOV_frame_0000.jpg")

    # Step 1: Check coordinate systems
    check_coordinate_systems()

    # Step 2: Run C++ in video mode
    print("\n📹 Running C++ OpenFace in VIDEO mode (-wild flag)...")
    cpp_output_dir = Path("test_output/video_mode_comparison/cpp_output")
    cpp_data = run_cpp_video_mode(image_path, cpp_output_dir)
    print(f"   ✅ C++ landmarks: {len(cpp_data['landmarks'])} points")
    print(f"   Success: {cpp_data['success']}")
    print(f"   Confidence: {cpp_data['confidence']:.3f}")

    # Step 3: Run Python in video mode with C++ bbox
    print("\n🐍 Running Python pyCLNF in VIDEO mode...")
    # Use C++ bbox estimate from landmarks
    cpp_lm = cpp_data['landmarks']
    x_min, y_min = cpp_lm.min(axis=0)
    x_max, y_max = cpp_lm.max(axis=0)
    margin = 0.1
    width = x_max - x_min
    height = y_max - y_min
    x_min -= margin * width
    y_min -= margin * height
    width *= (1 + 2*margin)
    height *= (1 + 2*margin)
    bbox = (x_min, y_min, width, height)

    py_data = run_python_video_mode(image_path, bbox)
    print(f"   ✅ Python landmarks: {len(py_data['landmarks'])} points")

    # Step 4: Analyze convergence
    analyze_convergence_details(cpp_data, py_data)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
