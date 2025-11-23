#!/usr/bin/env python3
"""
Compare actual calculation data between C++ and Python CLNF.

Captures and compares:
1. Mean-shift vectors
2. Update magnitudes
3. Parameter changes
4. Jacobian properties
"""

import numpy as np
import cv2
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf import CLNF
from pyclnf.core.pdm import PDM

def run_cpp_debug(frame_path, bbox):
    """Run C++ with iteration tracing enabled."""
    # C++ already writes trace to /tmp/clnf_iteration_traces/cpp_trace.txt
    bin_path = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        bin_path,
        "-f", str(frame_path),
        "-out_dir", "/tmp/clnf_iteration_traces/cpp_debug",
        "-2Dfp",
        "-pose"
    ]

    subprocess.run(cmd, capture_output=True)

    # Parse trace file
    trace_data = []
    with open("/tmp/clnf_iteration_traces/cpp_trace.txt", 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 11:
                continue

            trace_data.append({
                'iteration': int(parts[0]),
                'phase': parts[1],
                'window_size': int(parts[2]),
                'mean_shift_norm': float(parts[3]),
                'update_magnitude': float(parts[4]),
                'scale': float(parts[5]),
                'tx': float(parts[9]),
                'ty': float(parts[10]),
            })

    return trace_data

def main():
    print("=" * 100)
    print("NUMERICAL COMPARISON: C++ vs Python CLNF")
    print("=" * 100)

    # Setup
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    frame_idx = 160

    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    # Save frame for C++
    frame_path = "/tmp/debug_frame.png"
    cv2.imwrite(frame_path, frame)

    # Get bbox from C++
    bbox_file = Path("/tmp/cpp_init_bbox.txt")
    with open(bbox_file, 'r') as f:
        for line in f:
            if line.startswith('bbox:'):
                parts = line.split()[1:]
                bbox = np.array([float(p) for p in parts])

    print(f"\nFrame: {frame_idx}")
    print(f"Bbox: {bbox}")

    # Run C++
    print("\n" + "=" * 100)
    print("Running C++ OpenFace...")
    print("=" * 100)
    cpp_trace = run_cpp_debug(frame_path, bbox)

    # Run Python with detailed debug
    print("\n" + "=" * 100)
    print("Running Python pyclnf...")
    print("=" * 100)

    MODEL_DIR = Path("pyclnf/models")
    pdm = PDM(str(MODEL_DIR / "exported_pdm"))

    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=40,
        convergence_threshold=0.01,
        regularization=25,
        window_sizes=[11, 9, 7, 5],
        debug_mode=True  # Enable detailed output
    )

    # Run optimization
    landmarks, info = clnf.fit(frame, bbox, return_params=True)

    # Get iteration history
    py_history = info.get('iteration_history', [])

    print("\n" + "=" * 100)
    print("ITERATION-BY-ITERATION COMPARISON")
    print("=" * 100)

    print(f"\n{'Iter':<5} {'Phase':<10} {'WS':<4} {'C++ MS':<12} {'Py MS':<12} {'Ratio':<8} {'C++ Upd':<12} {'Py Upd':<12} {'Ratio':<8}")
    print("-" * 95)

    # Compare first few iterations
    for i in range(min(20, len(cpp_trace), len(py_history))):
        cpp = cpp_trace[i]
        py = py_history[i]

        ms_ratio = py['mean_shift_norm'] / cpp['mean_shift_norm'] if cpp['mean_shift_norm'] > 0 else 0
        upd_ratio = py['update_magnitude'] / cpp['update_magnitude'] if cpp['update_magnitude'] > 0 else 0

        print(f"{i:<5} {cpp['phase']:<10} {cpp['window_size']:<4} "
              f"{cpp['mean_shift_norm']:<12.2f} {py['mean_shift_norm']:<12.2f} {ms_ratio:<8.2f} "
              f"{cpp['update_magnitude']:<12.2f} {py['update_magnitude']:<12.2f} {upd_ratio:<8.2f}")

    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Analyze first rigid iteration (most important)
    if len(cpp_trace) > 0 and len(py_history) > 0:
        cpp0 = cpp_trace[0]
        py0 = py_history[0]

        print(f"\nFirst iteration analysis:")
        print(f"  Mean-shift ratio (Py/C++): {py0['mean_shift_norm']/cpp0['mean_shift_norm']:.3f}")
        print(f"  Update ratio (Py/C++):     {py0['update_magnitude']/cpp0['update_magnitude']:.3f}")

        # The update should be proportional to mean-shift
        # If MS ratio > Update ratio, Python's Jacobian/solver is less effective
        ms_ratio = py0['mean_shift_norm'] / cpp0['mean_shift_norm']
        upd_ratio = py0['update_magnitude'] / cpp0['update_magnitude']

        if ms_ratio > upd_ratio * 1.2:
            print(f"\n⚠️  Python has higher mean-shift but lower update!")
            print(f"    This suggests Jacobian or solver differences")
            print(f"    Expected update ratio: ~{ms_ratio:.2f}, Actual: {upd_ratio:.2f}")
        elif ms_ratio < upd_ratio * 0.8:
            print(f"\n⚠️  Python has lower mean-shift but higher update!")
            print(f"    This suggests different response map quality")

    # Check parameter evolution
    print("\n" + "=" * 100)
    print("PARAMETER EVOLUTION (first 10 iterations)")
    print("=" * 100)

    print(f"\n{'Iter':<5} {'C++ scale':<12} {'Py scale':<12} {'C++ tx':<12} {'Py tx':<12} {'C++ ty':<12} {'Py ty':<12}")
    print("-" * 75)

    for i in range(min(10, len(cpp_trace), len(py_history))):
        cpp = cpp_trace[i]
        py_params = py_history[i]['params']

        print(f"{i:<5} {cpp['scale']:<12.4f} {py_params[0]:<12.4f} "
              f"{cpp['tx']:<12.2f} {py_params[4]:<12.2f} "
              f"{cpp['ty']:<12.2f} {py_params[5]:<12.2f}")

if __name__ == "__main__":
    main()
