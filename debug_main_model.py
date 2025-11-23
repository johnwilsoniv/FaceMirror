#!/usr/bin/env python3
"""
Debug main CLNF model to understand why Python achieves 4.75px error vs C++ 0.65px.

Focus on single-frame comparison with detailed logging.
"""

import cv2
import numpy as np
from pathlib import Path

# Add pyclnf to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pyclnf.clnf import CLNF
from pyclnf.core.pdm import PDM

# Paths
MODEL_DIR = Path("pyclnf/models")
VIDEO_PATH = Path("Patient Data/Normal Cohort/Shorty.mov")
TRACE_DIR = Path("/tmp/clnf_debug")
TRACE_DIR.mkdir(exist_ok=True)

def run_cpp_trace(frame_path, output_dir):
    """Run C++ with detailed tracing enabled."""
    import subprocess

    cpp_bin = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
    trace_file = TRACE_DIR / "cpp_trace.txt"

    cmd = [
        cpp_bin,
        "-f", str(frame_path),
        "-out_dir", str(output_dir),
        "-2Dfp",
        "-verbose"
    ]

    # Set env to enable tracing
    env = {"OPENFACE_TRACE_FILE": str(trace_file)}

    result = subprocess.run(cmd, capture_output=True, text=True, env={**dict(__import__('os').environ), **env})

    return trace_file

def load_cpp_final_landmarks(csv_path):
    """Load C++ output landmarks from CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Extract x and y columns
    x_cols = [f" x_{i}" for i in range(68)]
    y_cols = [f" y_{i}" for i in range(68)]

    x_vals = df[x_cols].values[0]
    y_vals = df[y_cols].values[0]

    return np.column_stack([x_vals, y_vals])

def compare_initialization(py_params, cpp_trace_file):
    """Compare initial parameters between Python and C++."""
    print("\n" + "="*60)
    print("INITIALIZATION COMPARISON")
    print("="*60)

    print(f"\nPython initial params:")
    print(f"  scale={py_params[0]:.6f}")
    print(f"  rot=({py_params[1]:.6f}, {py_params[2]:.6f}, {py_params[3]:.6f})")
    print(f"  tx={py_params[4]:.6f}, ty={py_params[5]:.6f}")

    # Parse first line of C++ trace for initial params
    if cpp_trace_file.exists():
        with open(cpp_trace_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 12:
                    print(f"\nC++ initial params (iter 0):")
                    print(f"  scale={float(parts[6]):.6f}")
                    print(f"  rot=({float(parts[7]):.6f}, {float(parts[8]):.6f}, {float(parts[9]):.6f})")
                    print(f"  tx={float(parts[10]):.6f}, ty={float(parts[11]):.6f}")
                    break

def compare_landmarks(py_lm, cpp_lm, label=""):
    """Compute error between Python and C++ landmarks."""
    errors = np.sqrt(np.sum((py_lm - cpp_lm)**2, axis=1))

    print(f"\n{label} Error:")
    print(f"  Mean: {errors.mean():.4f} px")
    print(f"  Max:  {errors.max():.4f} px (landmark {errors.argmax()})")
    print(f"  Min:  {errors.min():.4f} px")

    # Show worst landmarks
    worst_idx = np.argsort(errors)[-5:][::-1]
    print(f"  Worst 5 landmarks:")
    for i in worst_idx:
        print(f"    Landmark {i}: {errors[i]:.4f} px")
        print(f"      Python: ({py_lm[i,0]:.2f}, {py_lm[i,1]:.2f})")
        print(f"      C++:    ({cpp_lm[i,0]:.2f}, {cpp_lm[i,1]:.2f})")
        diff = py_lm[i] - cpp_lm[i]
        print(f"      Diff:   ({diff[0]:.2f}, {diff[1]:.2f})")

    return errors

def main():
    print("="*70)
    print("MAIN MODEL DEBUG - Comparing Python vs C++ CLNF")
    print("="*70)

    # Load video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Cannot open {VIDEO_PATH}")
        return

    # Get frame 30
    frame_idx = 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {frame_idx}")
        return

    # Save frame for C++
    frame_path = TRACE_DIR / f"frame_{frame_idx}.png"
    cv2.imwrite(str(frame_path), frame)

    # Run C++ OpenFace
    print(f"\nRunning C++ on frame {frame_idx}...")
    output_dir = TRACE_DIR / "cpp_output"
    output_dir.mkdir(exist_ok=True)

    cpp_trace = run_cpp_trace(frame_path, output_dir)

    # Load C++ results
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        print("Error: No C++ CSV output found")
        return

    cpp_landmarks = load_cpp_final_landmarks(csv_files[0])
    print(f"Loaded C++ landmarks: shape={cpp_landmarks.shape}")

    # Get bbox from C++ or detect with Python
    # For now, use approximate bbox from landmarks
    bbox_x = cpp_landmarks[:, 0].min() - 20
    bbox_y = cpp_landmarks[:, 1].min() - 30
    bbox_w = cpp_landmarks[:, 0].max() - bbox_x + 20
    bbox_h = cpp_landmarks[:, 1].max() - bbox_y + 30
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
    print(f"Using bbox: {bbox}")

    # Initialize Python CLNF
    print("\nInitializing Python CLNF...")
    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=40,
        convergence_threshold=0.5,
        regularization=25,
        sigma=1.75,
        debug_mode=True  # Enable debug output
    )

    # Get initial params
    pdm = clnf.pdm
    init_params = pdm.init_params(bbox)
    init_landmarks = pdm.params_to_landmarks_2d(init_params)

    # Compare initialization
    compare_initialization(init_params, cpp_trace)
    compare_landmarks(init_landmarks, cpp_landmarks, "INITIAL")

    # Run Python CLNF
    print("\n" + "="*60)
    print("RUNNING PYTHON OPTIMIZATION")
    print("="*60)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    py_landmarks, info = clnf.fit(gray, bbox, return_params=True)

    if py_landmarks is None:
        print("Error: Python fit failed")
        return

    # Compare final results
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    compare_landmarks(py_landmarks, cpp_landmarks, "FINAL")

    print(f"\nPython info:")
    print(f"  Converged: {info['converged']}")
    print(f"  Total iterations: {info['iterations']}")

    # Analyze by landmark type
    print("\n" + "="*60)
    print("ERROR BY LANDMARK REGION")
    print("="*60)

    errors = np.sqrt(np.sum((py_landmarks - cpp_landmarks)**2, axis=1))

    regions = {
        'Jawline (0-16)': list(range(0, 17)),
        'Left eyebrow (17-21)': list(range(17, 22)),
        'Right eyebrow (22-26)': list(range(22, 27)),
        'Nose (27-35)': list(range(27, 36)),
        'Left eye (36-41)': list(range(36, 42)),
        'Right eye (42-47)': list(range(42, 48)),
        'Outer mouth (48-59)': list(range(48, 60)),
        'Inner mouth (60-67)': list(range(60, 68))
    }

    for name, indices in regions.items():
        region_errors = errors[indices]
        print(f"{name}: mean={region_errors.mean():.2f}px, max={region_errors.max():.2f}px")

if __name__ == "__main__":
    main()
