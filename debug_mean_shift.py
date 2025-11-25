#!/usr/bin/env python3
"""
Debug Mean-Shift Vector Comparison Tool

Captures and compares mean-shift vectors between C++ OpenFace and Python CLNF
to identify the source of convergence divergence.
"""

import numpy as np
import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Add local modules to path
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf import CLNF
from pyclnf.core.pdm import PDM

# ============================================================================
# Configuration
# ============================================================================
OPENFACE_BIN = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
MODEL_DIR = Path(__file__).parent / "pyclnf" / "pyclnf" / "models"
PDM_DIR = MODEL_DIR / "exported_pdm"
TRACE_DIR = Path("/tmp/clnf_iteration_traces")
OUTPUT_DIR = Path(__file__).parent / "mean_shift_analysis"

# ============================================================================
# Helper Functions
# ============================================================================

def params_to_array(params):
    """Convert params to numpy array in correct order."""
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
    """Load C++ iteration trace with enhanced parsing."""
    iterations = []

    with open(trace_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()

            # Filter to 68-landmark model (34 local params)
            if len(parts) < 46:
                continue

            local_params = [float(p) for p in parts[12:12+34]]
            if len(local_params) != 34:
                continue

            iteration_data = {
                'iteration': int(parts[0]),
                'phase': parts[1],
                'window_size': int(parts[2]),
                'mean_shift_norm': float(parts[3]),
                'update_magnitude': float(parts[4]),
                'jwtm_norm': float(parts[5]),
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


def extract_mean_shift_diagnostics(clnf, image, bbox, pdm, max_iterations=10):
    """
    Run Python CLNF with detailed mean-shift diagnostics.

    Returns:
        diagnostics: Dict with iteration-by-iteration mean-shift info
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Run with enhanced tracing
    old_debug_mode = clnf.optimizer.debug_mode
    old_max_iter = clnf.optimizer.max_iterations

    clnf.optimizer.debug_mode = True
    clnf.optimizer.max_iterations = max_iterations

    # Fit and get iteration history
    landmarks, info = clnf.fit(gray, bbox, return_params=True)

    # Restore settings
    clnf.optimizer.debug_mode = old_debug_mode
    clnf.optimizer.max_iterations = old_max_iter

    # Process iteration history
    diagnostics = {
        'iterations': [],
        'final_landmarks': landmarks,
        'final_params': info['params'] if 'params' in info else None
    }

    if 'iteration_history' in info:
        for iter_data in info['iteration_history']:
            # Compute landmarks for this iteration's params
            iter_landmarks = pdm.params_to_landmarks_2d(iter_data['params'])

            diagnostics['iterations'].append({
                'iteration': iter_data['iteration'],
                'phase': iter_data['phase'],
                'window_size': iter_data['window_size'],
                'mean_shift_norm': iter_data['mean_shift_norm'],
                'mean_shift_mean': iter_data['mean_shift_mean'],
                'update_magnitude': iter_data['update_magnitude'],
                'jacobian_norm': iter_data['jacobian_norm'],
                'regularization': iter_data['regularization'],
                'params': iter_data['params'],
                'landmarks': iter_landmarks
            })

    return diagnostics


def compare_iterations(cpp_iterations, py_diagnostics, pdm):
    """
    Compare C++ and Python iterations in detail.

    Returns comparison dict with divergence analysis.
    """
    comparison = {
        'iterations': [],
        'divergence_point': None,
        'max_divergence': 0.0
    }

    for i in range(min(len(cpp_iterations), len(py_diagnostics['iterations']))):
        cpp_iter = cpp_iterations[i]
        py_iter = py_diagnostics['iterations'][i]

        # Convert params to landmarks
        cpp_params = params_to_array(cpp_iter['params'])
        cpp_landmarks = pdm.params_to_landmarks_2d(cpp_params)

        py_landmarks = py_iter['landmarks']

        # Compute differences
        landmark_diff = np.linalg.norm(cpp_landmarks - py_landmarks, axis=1)
        mean_error = landmark_diff.mean()
        max_error = landmark_diff.max()

        # Parameter differences
        param_diff = cpp_params - py_iter['params']

        iter_comparison = {
            'iteration': i,
            'cpp_phase': cpp_iter['phase'],
            'py_phase': py_iter['phase'],
            'cpp_window_size': cpp_iter['window_size'],
            'py_window_size': py_iter['window_size'],
            'cpp_mean_shift_norm': cpp_iter['mean_shift_norm'],
            'py_mean_shift_norm': py_iter['mean_shift_norm'],
            'cpp_update_mag': cpp_iter['update_magnitude'],
            'py_update_mag': py_iter['update_magnitude'],
            'landmark_error_mean': mean_error,
            'landmark_error_max': max_error,
            'param_diff_norm': np.linalg.norm(param_diff),
            'scale_diff': param_diff[0],
            'rotation_diff': param_diff[1:4],
            'translation_diff': param_diff[4:6]
        }

        comparison['iterations'].append(iter_comparison)

        # Track max divergence
        if mean_error > comparison['max_divergence']:
            comparison['max_divergence'] = mean_error

        # Identify divergence point (first iteration with >1px error)
        if comparison['divergence_point'] is None and mean_error > 1.0:
            comparison['divergence_point'] = i

    return comparison


def plot_diagnostic_comparison(comparison, output_path):
    """Generate detailed diagnostic plots."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    iterations = [c['iteration'] for c in comparison['iterations']]

    # Plot 1: Mean landmark error
    ax = axes[0, 0]
    landmark_errors = [c['landmark_error_mean'] for c in comparison['iterations']]
    ax.plot(iterations, landmark_errors, 'r-o', label='Mean Error')
    if comparison['divergence_point'] is not None:
        ax.axvline(x=comparison['divergence_point'], color='red', alpha=0.3,
                  linestyle='--', label=f'Divergence (iter {comparison["divergence_point"]})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Landmark Error (px)')
    ax.set_title('Landmark Error: Python vs C++')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean-shift norms
    ax = axes[0, 1]
    cpp_ms_norms = [c['cpp_mean_shift_norm'] for c in comparison['iterations']]
    py_ms_norms = [c['py_mean_shift_norm'] for c in comparison['iterations']]
    ax.plot(iterations, cpp_ms_norms, 'b-o', label='C++ Mean-Shift', markersize=6)
    ax.plot(iterations, py_ms_norms, 'r-s', label='Python Mean-Shift', markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean-Shift Norm')
    ax.set_title('Mean-Shift Vector Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Update magnitudes
    ax = axes[1, 0]
    cpp_updates = [c['cpp_update_mag'] for c in comparison['iterations']]
    py_updates = [c['py_update_mag'] for c in comparison['iterations']]
    ax.plot(iterations, cpp_updates, 'b-o', label='C++ Update', markersize=6)
    ax.plot(iterations, py_updates, 'r-s', label='Python Update', markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Update Magnitude')
    ax.set_title('Parameter Update Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Scale parameter difference
    ax = axes[1, 1]
    scale_diffs = [c['scale_diff'] for c in comparison['iterations']]
    ax.plot(iterations, scale_diffs, 'g-o', label='Scale Diff')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Scale Parameter Difference')
    ax.set_title('Scale Parameter: Python - C++')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Rotation differences
    ax = axes[2, 0]
    rot_x_diffs = [c['rotation_diff'][0] for c in comparison['iterations']]
    rot_y_diffs = [c['rotation_diff'][1] for c in comparison['iterations']]
    rot_z_diffs = [c['rotation_diff'][2] for c in comparison['iterations']]
    ax.plot(iterations, rot_x_diffs, 'r-', label='Rot X', alpha=0.7)
    ax.plot(iterations, rot_y_diffs, 'g-', label='Rot Y', alpha=0.7)
    ax.plot(iterations, rot_z_diffs, 'b-', label='Rot Z', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rotation Difference (radians)')
    ax.set_title('Rotation Parameters: Python - C++')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Phase comparison
    ax = axes[2, 1]
    phases = []
    phase_colors = []
    for c in comparison['iterations']:
        if c['cpp_phase'] == 'rigid':
            phases.append(0.5)
            phase_colors.append('blue')
        else:
            phases.append(1.5)
            phase_colors.append('green')

    ax.scatter(iterations, phases, c=phase_colors, s=50, label='C++ Phase')

    phases_py = []
    phase_colors_py = []
    for c in comparison['iterations']:
        if c['py_phase'] == 'rigid':
            phases_py.append(0.4)
            phase_colors_py.append('red')
        else:
            phases_py.append(1.4)
            phase_colors_py.append('orange')

    ax.scatter(iterations, phases_py, c=phase_colors_py, s=50, marker='s', label='Python Phase')

    ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Rigid', 'Non-Rigid'])
    ax.set_xlabel('Iteration')
    ax.set_title('Optimization Phase Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Mean-Shift Diagnostic Comparison\nMax Divergence: {comparison["max_divergence"]:.2f}px at iter {comparison["divergence_point"]}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Diagnostic plot saved to: {output_path}")


def print_divergence_analysis(comparison):
    """Print detailed divergence analysis."""
    print("\n" + "=" * 100)
    print("MEAN-SHIFT DIVERGENCE ANALYSIS")
    print("=" * 100)

    if comparison['divergence_point'] is not None:
        div_iter = comparison['iterations'][comparison['divergence_point']]
        print(f"\nDivergence detected at iteration {comparison['divergence_point']}:")
        print(f"  Phase: C++={div_iter['cpp_phase']}, Python={div_iter['py_phase']}")
        print(f"  Window Size: C++={div_iter['cpp_window_size']}, Python={div_iter['py_window_size']}")
        print(f"  Mean-Shift Norm: C++={div_iter['cpp_mean_shift_norm']:.4f}, Python={div_iter['py_mean_shift_norm']:.4f}")
        print(f"  Update Magnitude: C++={div_iter['cpp_update_mag']:.4f}, Python={div_iter['py_update_mag']:.4f}")
        print(f"  Landmark Error: {div_iter['landmark_error_mean']:.2f}px (max: {div_iter['landmark_error_max']:.2f}px)")

    print("\n" + "-" * 50)
    print("Iteration-by-Iteration Comparison:")
    print("-" * 50)
    print(f"{'Iter':<5} {'Phase':<10} {'C++ MS':<12} {'Py MS':<12} {'C++ Upd':<12} {'Py Upd':<12} {'Error':<10}")
    print("-" * 80)

    for c in comparison['iterations'][:20]:  # Show first 20 iterations
        phase = f"{c['cpp_phase'][:3]}/{c['py_phase'][:3]}"
        print(f"{c['iteration']:<5} {phase:<10} "
              f"{c['cpp_mean_shift_norm']:<12.4f} {c['py_mean_shift_norm']:<12.4f} "
              f"{c['cpp_update_mag']:<12.4f} {c['py_update_mag']:<12.4f} "
              f"{c['landmark_error_mean']:<10.2f}")

    print("\n" + "-" * 50)
    print("Key Observations:")
    print("-" * 50)

    # Check for phase mismatch
    phase_mismatch = False
    for c in comparison['iterations']:
        if c['cpp_phase'] != c['py_phase']:
            phase_mismatch = True
            break

    if phase_mismatch:
        print("⚠️  Phase mismatch detected between C++ and Python")

    # Check for mean-shift magnitude differences
    ms_ratio_issues = []
    for c in comparison['iterations'][:10]:
        if c['cpp_mean_shift_norm'] > 0:
            ratio = c['py_mean_shift_norm'] / c['cpp_mean_shift_norm']
            if ratio > 2.0 or ratio < 0.5:
                ms_ratio_issues.append((c['iteration'], ratio))

    if ms_ratio_issues:
        print(f"⚠️  Large mean-shift differences detected:")
        for iter_num, ratio in ms_ratio_issues[:3]:
            print(f"    Iteration {iter_num}: Python/C++ ratio = {ratio:.2f}")

    # Check for update magnitude differences
    update_ratio_issues = []
    for c in comparison['iterations'][:10]:
        if c['cpp_update_mag'] > 0:
            ratio = c['py_update_mag'] / c['cpp_update_mag']
            if ratio > 2.0 or ratio < 0.5:
                update_ratio_issues.append((c['iteration'], ratio))

    if update_ratio_issues:
        print(f"⚠️  Large update magnitude differences detected:")
        for iter_num, ratio in update_ratio_issues[:3]:
            print(f"    Iteration {iter_num}: Python/C++ ratio = {ratio:.2f}")


def analyze_single_frame(image_path, frame_name="test"):
    """Analyze a single frame with detailed mean-shift comparison."""
    print(f"\nAnalyzing frame: {frame_name}")
    print("-" * 50)

    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    # Load PDM
    print("Loading PDM model...")
    pdm = PDM(str(PDM_DIR))

    # Initialize Python CLNF
    print("Initializing Python CLNF...")
    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=20,  # Limit for focused analysis
        convergence_threshold=0.5,
        regularization=25,
        window_sizes=[11, 9, 7],
        debug_mode=False
    )

    # Load image
    image = cv2.imread(str(image_path))

    # Run C++ OpenFace
    print("Running C++ OpenFace...")
    cpp_output_dir = TRACE_DIR / f"cpp_output_{frame_name}"
    cpp_output_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous traces
    cpp_trace_file = TRACE_DIR / "cpp_trace.txt"
    if cpp_trace_file.exists():
        cpp_trace_file.unlink()

    cmd = [
        OPENFACE_BIN,
        "-f", str(image_path),
        "-out_dir", str(cpp_output_dir),
        "-2Dfp", "-pose"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"C++ FeatureExtraction failed: {result.stderr}")
        return None

    # Load C++ trace
    if not cpp_trace_file.exists():
        print("C++ trace file not found")
        return None

    cpp_iterations = load_cpp_trace(cpp_trace_file)
    print(f"  Loaded {len(cpp_iterations)} C++ iterations")

    # Get bbox from C++ initialization
    if not cpp_iterations:
        print("No C++ iterations found")
        return None

    iter0 = cpp_iterations[0]
    params0 = params_to_array(iter0['params'])
    init_landmarks = pdm.params_to_landmarks_2d(params0)

    x_min, y_min = init_landmarks.min(axis=0)
    x_max, y_max = init_landmarks.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    padding = 0.1

    bbox = np.array([
        x_min - width * padding,
        y_min - height * padding,
        width * (1 + 2 * padding),
        height * (1 + 2 * padding)
    ])

    print(f"  Bbox from C++ init: {bbox}")

    # Run Python with diagnostics
    print("Running Python CLNF with diagnostics...")
    py_diagnostics = extract_mean_shift_diagnostics(clnf, image, bbox, pdm, max_iterations=20)
    print(f"  Captured {len(py_diagnostics['iterations'])} Python iterations")

    # Compare iterations
    print("Comparing iterations...")
    comparison = compare_iterations(cpp_iterations, py_diagnostics, pdm)

    # Generate plots
    plot_path = OUTPUT_DIR / f"mean_shift_comparison_{frame_name}.png"
    plot_diagnostic_comparison(comparison, plot_path)

    # Print analysis
    print_divergence_analysis(comparison)

    # Save comparison data
    json_path = OUTPUT_DIR / f"mean_shift_comparison_{frame_name}.json"
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_comparison = {
            'iterations': comparison['iterations'],
            'divergence_point': comparison['divergence_point'],
            'max_divergence': float(comparison['max_divergence'])
        }
        json.dump(json_comparison, f, indent=2)
    print(f"\nComparison data saved to: {json_path}")

    return comparison


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Use first frame from test video
    test_frame = TRACE_DIR / "frame_0.jpg"

    if not test_frame.exists():
        print(f"Test frame not found: {test_frame}")
        print("Extracting frame from video...")

        video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(test_frame), frame)
            print(f"Frame extracted to: {test_frame}")
        cap.release()

    if test_frame.exists():
        analyze_single_frame(test_frame, "frame_0")
    else:
        print("Could not extract test frame")