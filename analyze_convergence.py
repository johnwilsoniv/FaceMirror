#!/usr/bin/env python3
"""
CLNF Convergence Analysis

Analyzes iteration-by-iteration pixel error to determine if Python CLNF
is converging or oscillating compared to C++ OpenFace ground truth.

Features:
- Converts parameters to landmarks for pixel error computation
- Tracks convergence vs oscillation patterns
- Supports multiple frames from video
- Generates visualization plots
"""

import numpy as np
import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
import cv2

# Add local modules to path
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

import matplotlib.pyplot as plt
from pyclnf.core.pdm import PDM
from pyclnf import CLNF
from pymtcnn import MTCNN

# ============================================================================
# Configuration
# ============================================================================
OPENFACE_BIN = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
MODEL_DIR = Path(__file__).parent / "pyclnf" / "models"
PDM_DIR = MODEL_DIR / "exported_pdm"
TRACE_DIR = Path("/tmp/clnf_iteration_traces")
OUTPUT_DIR = Path(__file__).parent / "convergence_analysis"

# ============================================================================
# Helper Functions
# ============================================================================

def params_to_array(params):
    """Convert params to numpy array in correct order.

    Handles both:
    - dict format (from C++ trace): {'scale': ..., 'rot_x': ..., 'local': [...]}
    - numpy array format (from Python): [scale, tx, ty, wx, wy, wz, q0, ..., qm]
    """
    if isinstance(params, np.ndarray):
        return params
    elif isinstance(params, dict):
        return np.array([
            params['scale'],
            params['rot_x'],
            params['rot_y'],
            params['rot_z'],
            params['trans_x'],
            params['trans_y'],
        ] + params['local'])
    else:
        raise ValueError(f"Unknown params format: {type(params)}")


def load_cpp_trace(trace_file):
    """Load C++ iteration trace, filtering to 68-landmark model only."""
    iterations = []

    with open(trace_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()

            # Filter: 68-landmark model has 34 local params
            # Format: iter phase ws ms_norm update_mag jwtm_norm scale rx ry rz tx ty p0..p33
            # Total parts should be 12 + 34 = 46
            if len(parts) < 46:
                continue

            # Check if we have exactly 34 local params (not eye models)
            # Format: iter phase ws ms_norm update_mag jwtm_norm scale rx ry rz tx ty p0...p33
            # Indices: 0    1     2  3       4          5         6     7  8  9  10 11 12...45
            local_params = [float(p) for p in parts[12:12+34]]
            if len(local_params) != 34:
                continue

            iteration_data = {
                'iteration': int(parts[0]),
                'phase': parts[1],
                'window_size': int(parts[2]),
                'mean_shift_norm': float(parts[3]),
                'update_magnitude': float(parts[4]),
                'params': {
                    'scale': float(parts[6]),
                    'rot_x': float(parts[7]),
                    'rot_y': float(parts[8]),
                    'rot_z': float(parts[9]),
                    'trans_x': float(parts[10]),
                    'trans_y': float(parts[11]),
                    'local': local_params
                }
            }
            iterations.append(iteration_data)

    return iterations


def load_python_trace(trace_file):
    """Load Python iteration trace."""
    with open(trace_file, 'r') as f:
        data = json.load(f)
    return data['iterations']


def compute_pixel_error(landmarks1, landmarks2):
    """Compute per-landmark Euclidean distance and statistics."""
    distances = np.sqrt(np.sum((landmarks1 - landmarks2)**2, axis=1))
    return {
        'mean': distances.mean(),
        'max': distances.max(),
        'std': distances.std(),
        'per_landmark': distances
    }


def run_cpp_on_frame(image_path, output_dir):
    """Run C++ OpenFace on a single frame and return bbox + traces."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous traces
    trace_file = TRACE_DIR / "cpp_trace.txt"
    if trace_file.exists():
        trace_file.unlink()

    # Run FeatureExtraction
    cmd = [
        OPENFACE_BIN,
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-2Dfp",  # Output 2D landmarks
        "-pose",  # Output pose
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"C++ FeatureExtraction failed: {result.stderr}")
        return None, None

    # Parse CSV output for bbox and final landmarks
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        return None, None

    csv_file = csv_files[0]

    # Read landmarks from CSV
    import pandas as pd
    df = pd.read_csv(csv_file)

    if len(df) == 0:
        return None, None

    row = df.iloc[0]

    # Extract 2D landmarks (x_0, x_1, ..., x_67, y_0, y_1, ..., y_67)
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = row[x_cols[i]]
        landmarks[i, 1] = row[y_cols[i]]

    # Load iteration trace
    cpp_iterations = []
    if trace_file.exists():
        cpp_iterations = load_cpp_trace(trace_file)

    # Compute bbox from C++ iteration 0 parameters (the actual initialization)
    # This ensures Python uses the same starting point as C++
    bbox = None
    if cpp_iterations:
        # Load PDM to convert params to landmarks
        pdm = PDM(str(PDM_DIR))

        # Get iteration 0 parameters
        iter0 = cpp_iterations[0]
        params = params_to_array(iter0['params'])

        # C++ and Python PDM use same order: [s, rx, ry, rz, tx, ty, q0...]
        # Compute initial landmarks from iteration 0 params
        init_landmarks = pdm.params_to_landmarks_2d(params)

        # Compute bbox from initial landmarks
        x_min, y_min = init_landmarks.min(axis=0)
        x_max, y_max = init_landmarks.max(axis=0)

        # Add padding to match MTCNN-style bbox
        width = x_max - x_min
        height = y_max - y_min
        padding = 0.1

        bbox = np.array([
            x_min - width * padding,
            y_min - height * padding,
            width * (1 + 2 * padding),
            height * (1 + 2 * padding)
        ])
        print(f"    Bbox from C++ iter0: {bbox}")

    if bbox is None:
        # Fallback: estimate from final landmarks
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        padding = 0.1
        bbox = np.array([
            x_min - width * padding,
            y_min - height * padding,
            width * (1 + 2 * padding),
            height * (1 + 2 * padding)
        ])
        print(f"    Bbox from final landmarks (fallback): {bbox}")

    return bbox, cpp_iterations, landmarks


def run_python_on_frame(image, bbox, clnf):
    """Run Python CLNF on a frame and return iteration history."""
    # Clear previous trace
    trace_file = TRACE_DIR / "python_trace.json"

    # Fit with iteration tracking
    landmarks, info = clnf.fit(image, bbox, return_params=True)

    if landmarks is None:
        return None, None

    # Extract iteration history
    if 'iteration_history' not in info:
        print("Warning: iteration_history not in info dict")
        return landmarks, []

    return landmarks, info['iteration_history']


def analyze_convergence(cpp_iterations, python_iterations, pdm, cpp_final_landmarks):
    """
    Analyze convergence patterns for both C++ and Python.

    Returns dict with analysis results.
    """
    results = {
        'cpp': {
            'errors': [],
            'oscillations': 0,
            'iterations': []
        },
        'python': {
            'errors': [],
            'oscillations': 0,
            'iterations': []
        }
    }

    # Analyze C++ convergence
    prev_error = None
    for i, iter_data in enumerate(cpp_iterations):
        params = params_to_array(iter_data['params'])
        landmarks = pdm.params_to_landmarks_2d(params)
        error_info = compute_pixel_error(landmarks, cpp_final_landmarks)

        results['cpp']['errors'].append(error_info['mean'])
        results['cpp']['iterations'].append({
            'idx': i,
            'phase': iter_data['phase'],
            'window_size': iter_data['window_size'],
            'error': error_info['mean'],
            'max_error': error_info['max'],
            'delta': error_info['mean'] - prev_error if prev_error is not None else 0,
            'is_oscillation': prev_error is not None and error_info['mean'] > prev_error
        })

        if prev_error is not None and error_info['mean'] > prev_error:
            results['cpp']['oscillations'] += 1

        prev_error = error_info['mean']

    # Analyze Python convergence
    prev_error = None
    for i, iter_data in enumerate(python_iterations):
        params = params_to_array(iter_data['params'])
        landmarks = pdm.params_to_landmarks_2d(params)
        error_info = compute_pixel_error(landmarks, cpp_final_landmarks)

        results['python']['errors'].append(error_info['mean'])
        results['python']['iterations'].append({
            'idx': i,
            'phase': iter_data.get('phase', 'unknown'),
            'window_size': iter_data.get('window_size', 0),
            'error': error_info['mean'],
            'max_error': error_info['max'],
            'delta': error_info['mean'] - prev_error if prev_error is not None else 0,
            'is_oscillation': prev_error is not None and error_info['mean'] > prev_error
        })

        if prev_error is not None and error_info['mean'] > prev_error:
            results['python']['oscillations'] += 1

        prev_error = error_info['mean']

    return results


def plot_convergence(results, frame_idx, output_path):
    """Generate convergence plot comparing C++ and Python."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Error over iterations
    ax1 = axes[0]

    cpp_errors = results['cpp']['errors']
    py_errors = results['python']['errors']

    ax1.plot(range(len(cpp_errors)), cpp_errors, 'b-o', label='C++ → C++ final', markersize=4)
    ax1.plot(range(len(py_errors)), py_errors, 'r-o', label='Python → C++ final', markersize=4)

    # Mark oscillations
    for iter_info in results['cpp']['iterations']:
        if iter_info['is_oscillation']:
            ax1.axvline(x=iter_info['idx'], color='blue', alpha=0.3, linestyle='--')

    for iter_info in results['python']['iterations']:
        if iter_info['is_oscillation']:
            ax1.axvline(x=iter_info['idx'], color='red', alpha=0.3, linestyle='--')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Pixel Error')
    ax1.set_title(f'Frame {frame_idx}: Convergence Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add window size annotations
    if results['python']['iterations']:
        prev_ws = None
        for iter_info in results['python']['iterations']:
            ws = iter_info['window_size']
            if ws != prev_ws:
                ax1.axvline(x=iter_info['idx'], color='gray', alpha=0.5, linestyle=':')
                ax1.text(iter_info['idx'], ax1.get_ylim()[1], f'WS={ws}',
                        rotation=90, va='top', fontsize=8)
                prev_ws = ws

    # Plot 2: Error delta (change per iteration)
    ax2 = axes[1]

    cpp_deltas = [it['delta'] for it in results['cpp']['iterations']]
    py_deltas = [it['delta'] for it in results['python']['iterations']]

    ax2.bar(np.arange(len(cpp_deltas)) - 0.2, cpp_deltas, 0.4, label='C++ Δ error', color='blue', alpha=0.7)
    ax2.bar(np.arange(len(py_deltas)) + 0.2, py_deltas, 0.4, label='Python Δ error', color='red', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Δ Pixel Error')
    ax2.set_title('Error Change Per Iteration (positive = oscillation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_iteration_table(results):
    """Print detailed iteration-by-iteration comparison table."""
    print("\n" + "=" * 100)
    print("ITERATION-BY-ITERATION ANALYSIS")
    print("=" * 100)

    # Determine max iterations
    max_iters = max(len(results['cpp']['iterations']), len(results['python']['iterations']))

    print(f"\n{'Iter':<5} {'Phase':<10} {'WS':<4} {'C++ Error':<12} {'Py Error':<12} {'Py Δ':<10} {'Status':<15}")
    print("-" * 80)

    for i in range(max_iters):
        cpp_iter = results['cpp']['iterations'][i] if i < len(results['cpp']['iterations']) else None
        py_iter = results['python']['iterations'][i] if i < len(results['python']['iterations']) else None

        if py_iter:
            phase = py_iter['phase']
            ws = py_iter['window_size']
            cpp_err = f"{cpp_iter['error']:.2f} px" if cpp_iter else "N/A"
            py_err = f"{py_iter['error']:.2f} px"
            py_delta = f"{py_iter['delta']:+.2f}" if i > 0 else "-"

            if py_iter['is_oscillation']:
                status = "↑ OSCILLATION"
            elif py_iter['delta'] < -0.1:
                status = "↓ converging"
            else:
                status = "→ stable"

            print(f"{i:<5} {phase:<10} {ws:<4} {cpp_err:<12} {py_err:<12} {py_delta:<10} {status:<15}")


def print_summary(results, frame_idx):
    """Print convergence summary statistics."""
    print("\n" + "=" * 100)
    print(f"CONVERGENCE SUMMARY - Frame {frame_idx}")
    print("=" * 100)

    cpp_errors = results['cpp']['errors']
    py_errors = results['python']['errors']

    print(f"\nC++ Convergence:")
    print(f"  Initial error: {cpp_errors[0]:.2f} px")
    print(f"  Final error:   {cpp_errors[-1]:.2f} px")
    print(f"  Reduction:     {cpp_errors[0] - cpp_errors[-1]:.2f} px ({100*(cpp_errors[0] - cpp_errors[-1])/cpp_errors[0]:.1f}%)")
    print(f"  Oscillations:  {results['cpp']['oscillations']}")
    print(f"  Total iters:   {len(cpp_errors)}")

    print(f"\nPython Convergence:")
    print(f"  Initial error: {py_errors[0]:.2f} px")
    print(f"  Final error:   {py_errors[-1]:.2f} px")
    print(f"  Reduction:     {py_errors[0] - py_errors[-1]:.2f} px ({100*(py_errors[0] - py_errors[-1])/py_errors[0]:.1f}%)")
    print(f"  Oscillations:  {results['python']['oscillations']}")
    print(f"  Total iters:   {len(py_errors)}")

    print(f"\nComparison:")
    print(f"  Python final error to C++ final: {py_errors[-1]:.2f} px")

    if results['python']['oscillations'] > results['cpp']['oscillations']:
        print(f"  ⚠️  Python has MORE oscillations than C++ ({results['python']['oscillations']} vs {results['cpp']['oscillations']})")
    elif results['python']['oscillations'] == 0 and results['cpp']['oscillations'] == 0:
        print(f"  ✓ Both converge monotonically")
    else:
        print(f"  ✓ Similar oscillation patterns")


def analyze_video(video_path, num_frames=10):
    """Analyze convergence across multiple frames from a video."""
    print("=" * 100)
    print("CLNF CONVERGENCE ANALYSIS")
    print("=" * 100)
    print(f"\nVideo: {video_path}")
    print(f"Frames to analyze: {num_frames}")
    print()

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    # Load PDM for parameter-to-landmark conversion
    print("Loading PDM model...")
    pdm = PDM(str(PDM_DIR))
    print(f"  PDM loaded: {pdm.n_points} landmarks, {pdm.n_modes} modes")

    # Initialize CLNF with more iterations to test convergence depth
    print("Initializing Python CLNF...")
    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=40,  # Match C++ which does 10 per phase × 4 windows = 40
        convergence_threshold=0.5,
        regularization=25,  # Match C++ video mode reg_factor
        window_sizes=[11, 9, 7],  # Skip WS=5 which causes overfitting
        debug_mode=False
    )
    print(f"  CLNF initialized (max_iterations=40)")
    print()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")

    # Select frames to analyze (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    print(f"Analyzing frames: {frame_indices.tolist()}")
    print()

    # Aggregate results
    all_results = []

    for frame_idx in frame_indices:
        print(f"\n{'='*100}")
        print(f"FRAME {frame_idx}")
        print(f"{'='*100}")

        # Extract frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Error: Could not read frame {frame_idx}")
            continue

        # Save frame temporarily
        temp_frame_path = TRACE_DIR / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(temp_frame_path), frame)

        # Run C++ on frame
        print("  Running C++ OpenFace...")
        cpp_output_dir = TRACE_DIR / f"cpp_output_{frame_idx}"
        bbox, cpp_iterations, cpp_final_landmarks = run_cpp_on_frame(
            temp_frame_path, cpp_output_dir
        )

        if bbox is None or cpp_iterations is None or len(cpp_iterations) == 0:
            print(f"  Error: C++ failed on frame {frame_idx}")
            continue

        print(f"    Bbox: {bbox}")
        print(f"    C++ iterations: {len(cpp_iterations)}")

        # Run Python on frame with same bbox
        print("  Running Python pyclnf...")
        py_landmarks, py_iterations = run_python_on_frame(frame, bbox, clnf)

        if py_landmarks is None or len(py_iterations) == 0:
            print(f"  Error: Python failed on frame {frame_idx}")
            continue

        print(f"    Python iterations: {len(py_iterations)}")

        # Analyze convergence
        print("  Analyzing convergence...")
        results = analyze_convergence(cpp_iterations, py_iterations, pdm, cpp_final_landmarks)

        # Print results
        print_iteration_table(results)
        print_summary(results, frame_idx)

        # Generate plot
        plot_path = OUTPUT_DIR / f"convergence_frame_{frame_idx}.png"
        plot_convergence(results, frame_idx, plot_path)
        print(f"\n  Plot saved: {plot_path}")

        all_results.append({
            'frame_idx': frame_idx,
            'results': results
        })

    cap.release()

    # Print aggregate summary
    if all_results:
        print("\n" + "=" * 100)
        print("AGGREGATE SUMMARY ACROSS ALL FRAMES")
        print("=" * 100)

        total_cpp_osc = sum(r['results']['cpp']['oscillations'] for r in all_results)
        total_py_osc = sum(r['results']['python']['oscillations'] for r in all_results)

        final_errors = [r['results']['python']['errors'][-1] for r in all_results]

        print(f"\nTotal C++ oscillations:    {total_cpp_osc}")
        print(f"Total Python oscillations: {total_py_osc}")
        print(f"\nPython final errors:")
        print(f"  Mean: {np.mean(final_errors):.2f} px")
        print(f"  Max:  {np.max(final_errors):.2f} px")
        print(f"  Min:  {np.min(final_errors):.2f} px")

        if total_py_osc > total_cpp_osc * 1.5:
            print(f"\n⚠️  ISSUE: Python has significantly more oscillations than C++")
            print(f"    This indicates the Python optimizer may be overshooting")
        elif np.mean(final_errors) > 3.0:
            print(f"\n⚠️  ISSUE: Python final error is high (>{3.0} px mean)")
            print(f"    This indicates Python is not converging to C++ result")
        else:
            print(f"\n✓ Python convergence appears reasonable")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    analyze_video(video_path, num_frames=10)
