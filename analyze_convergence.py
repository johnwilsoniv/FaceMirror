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
from pyclnf.core.eye_pdm import EyePDM
from pyclnf import CLNF
from pymtcnn import MTCNN

# ============================================================================
# Configuration
# ============================================================================
OPENFACE_BIN = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
MODEL_DIR = Path(__file__).parent / "pyclnf" / "pyclnf" / "models"
PDM_DIR = MODEL_DIR / "exported_pdm"
EYE_PDM_LEFT_DIR = MODEL_DIR / "exported_eye_pdm_left"
EYE_PDM_RIGHT_DIR = MODEL_DIR / "exported_eye_pdm_right"
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


def load_cpp_trace(trace_file, include_eyes=False):
    """Load C++ iteration trace, optionally including eye model iterations.

    Args:
        trace_file: Path to C++ trace file
        include_eyes: If True, also return eye iterations separately

    Returns:
        If include_eyes=False: list of face iterations
        If include_eyes=True: (face_iterations, eye_iterations)
    """
    face_iterations = []
    eye_iterations = []

    with open(trace_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()

            if len(parts) < 12:  # Not enough for basic parameters
                continue

            # Extract basic iteration info (common to face and eye models)
            iteration_base = {
                'iteration': int(parts[0]),
                'phase': parts[1],
                'window_size': int(parts[2]),
                'mean_shift_norm': float(parts[3]),
                'update_magnitude': float(parts[4]),
                'jwtm_norm': float(parts[5]),
                'scale': float(parts[6]),
                'rot_x': float(parts[7]),
                'rot_y': float(parts[8]),
                'rot_z': float(parts[9]),
                'trans_x': float(parts[10]),
                'trans_y': float(parts[11]),
            }

            # Determine model type by number of local parameters
            n_local_params = len(parts) - 12

            if n_local_params == 34:  # Face model (68 landmarks)
                local_params = [float(p) for p in parts[12:12+34]]
                iteration_data = {
                    **iteration_base,
                    'model_type': 'face',
                    'params': {
                        'scale': iteration_base['scale'],
                        'rot_x': iteration_base['rot_x'],
                        'rot_y': iteration_base['rot_y'],
                        'rot_z': iteration_base['rot_z'],
                        'trans_x': iteration_base['trans_x'],
                        'trans_y': iteration_base['trans_y'],
                        'local': local_params
                    }
                }
                face_iterations.append(iteration_data)

            elif n_local_params == 10 and include_eyes:  # Eye model (28 landmarks per eye)
                local_params = [float(p) for p in parts[12:12+10]]

                # Determine which eye based on tx position
                # The eye model tx indicates the eye center in image coordinates
                # Right eye has higher tx (right side of image), left eye has lower tx
                # Use face center tx (from last face iteration) as reference
                face_center_tx = 500  # Default fallback
                if face_iterations:
                    face_center_tx = face_iterations[-1]['trans_x']

                if iteration_base['trans_x'] > face_center_tx:
                    eye_side = 'right'
                else:
                    eye_side = 'left'

                iteration_data = {
                    **iteration_base,
                    'model_type': 'eye',
                    'eye_side': eye_side,
                    'params': {
                        'scale': iteration_base['scale'],
                        'rot_x': iteration_base['rot_x'],
                        'rot_y': iteration_base['rot_y'],
                        'rot_z': iteration_base['rot_z'],
                        'trans_x': iteration_base['trans_x'],
                        'trans_y': iteration_base['trans_y'],
                        'local': local_params
                    }
                }
                eye_iterations.append(iteration_data)

    if include_eyes:
        return face_iterations, eye_iterations
    else:
        return face_iterations


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


def compute_final_combined_error(py_landmarks, cpp_final_landmarks):
    """
    Compute final 68-point error AFTER eye refinement.

    This is the TRUE final error that matters for evaluation:
    - Landmarks 0-35: Face model (cheeks, nose, etc.)
    - Landmarks 36-41: LEFT eye (refined by eye model)
    - Landmarks 42-47: RIGHT eye (refined by eye model)
    - Landmarks 48-67: Mouth, jawline (face model)
    """
    # Face landmarks excluding eyes
    face_idx = list(range(36)) + list(range(48, 68))

    return {
        'total_68': compute_pixel_error(py_landmarks, cpp_final_landmarks),
        'face_only': compute_pixel_error(py_landmarks[face_idx], cpp_final_landmarks[face_idx]),
        'left_eye': compute_pixel_error(py_landmarks[36:42], cpp_final_landmarks[36:42]),
        'right_eye': compute_pixel_error(py_landmarks[42:48], cpp_final_landmarks[42:48])
    }


def run_cpp_on_frame(image_path, output_dir, include_eyes=False):
    """Run C++ OpenFace on a single frame and return bbox + traces.

    Args:
        image_path: Path to input image
        output_dir: Directory for C++ outputs
        include_eyes: If True, also return eye refinement iterations

    Returns:
        bbox: Face bounding box
        face_iterations: Main face model iterations
        eye_iterations: Eye refinement iterations (if include_eyes=True)
        final_landmarks: Final refined landmarks from CSV
    """
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
    eye_iterations = []
    if trace_file.exists():
        if include_eyes:
            cpp_iterations, eye_iterations = load_cpp_trace(trace_file, include_eyes=True)
        else:
            cpp_iterations = load_cpp_trace(trace_file, include_eyes=False)

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

    if include_eyes:
        return bbox, cpp_iterations, eye_iterations, landmarks
    else:
        return bbox, cpp_iterations, landmarks


def run_python_on_frame(image, bbox, clnf):
    """Run Python CLNF on a frame and return iteration history."""
    # Clear previous trace
    trace_file = TRACE_DIR / "python_trace.json"

    # Fit with iteration tracking
    landmarks, info = clnf.fit(image, bbox, return_params=True)

    if landmarks is None:
        return None, None, None

    # Extract iteration history
    if 'iteration_history' not in info:
        print("Warning: iteration_history not in info dict")
        return landmarks, [], []

    # Extract eye iteration history
    eye_history = info.get('eye_iteration_history', [])

    return landmarks, info['iteration_history'], eye_history


def analyze_eye_convergence(cpp_eye_iterations, python_eye_iterations, cpp_final_landmarks):
    """
    Analyze eye-specific convergence patterns.

    Args:
        cpp_eye_iterations: List of C++ eye iteration data
        python_eye_iterations: List of Python eye iteration data
        cpp_final_landmarks: Final C++ landmarks (68 points) for reference

    Returns:
        Dict with eye-specific analysis results
    """
    # NOTE: We use C++ eye final state (28-point landmarks) as ground truth instead of
    # face model landmarks. This ensures we compare in the same eye-centered coordinate frame.

    results = {
        'left_eye': {
            'cpp': {'iterations': [], 'window_sizes': [], 'phases': [], 'errors': []},
            'python': {'iterations': [], 'window_sizes': [], 'phases': [], 'errors': []}
        },
        'right_eye': {
            'cpp': {'iterations': [], 'window_sizes': [], 'phases': [], 'errors': []},
            'python': {'iterations': [], 'window_sizes': [], 'phases': [], 'errors': []}
        },
        'summary': {}
    }

    # Load EyePDM models for converting C++ params to landmarks
    eye_pdm_left = None
    eye_pdm_right = None
    try:
        if EYE_PDM_LEFT_DIR.exists():
            eye_pdm_left = EyePDM(str(EYE_PDM_LEFT_DIR))
        if EYE_PDM_RIGHT_DIR.exists():
            eye_pdm_right = EyePDM(str(EYE_PDM_RIGHT_DIR))
    except Exception as e:
        print(f"Warning: Could not load EyePDM models: {e}")

    # Separate C++ iterations by eye
    cpp_left = [i for i in cpp_eye_iterations if i.get('eye_side') == 'left']
    cpp_right = [i for i in cpp_eye_iterations if i.get('eye_side') == 'right']

    # Separate Python iterations by eye
    py_left = [i for i in python_eye_iterations if i.get('eye_side') == 'left']
    py_right = [i for i in python_eye_iterations if i.get('eye_side') == 'right']

    # Compute C++ eye final state as ground truth (last iteration's 28-point landmarks)
    # This ensures we compare in the same eye-centered coordinate frame
    cpp_left_final_lms = None
    cpp_right_final_lms = None

    if cpp_left and eye_pdm_left is not None:
        cpp_left_final_params = params_to_array(cpp_left[-1]['params'])
        cpp_left_final_lms = eye_pdm_left.params_to_landmarks_2d(cpp_left_final_params)

    if cpp_right and eye_pdm_right is not None:
        cpp_right_final_params = params_to_array(cpp_right[-1]['params'])
        cpp_right_final_lms = eye_pdm_right.params_to_landmarks_2d(cpp_right_final_params)

    # Analyze left eye - C++
    if cpp_left:
        results['left_eye']['cpp']['iterations'] = len(cpp_left)
        results['left_eye']['cpp']['window_sizes'] = sorted(set(i['window_size'] for i in cpp_left))
        results['left_eye']['cpp']['phases'] = sorted(set(i['phase'] for i in cpp_left))

        # Compute C++ eye errors using EyePDM - compare against C++ final state (same coord frame)
        if eye_pdm_left is not None and cpp_left_final_lms is not None:
            cpp_left_errors = []
            for iter_data in cpp_left:
                # Convert C++ params to eye landmarks using EyePDM
                params = params_to_array(iter_data['params'])
                eye_lms_28 = eye_pdm_left.params_to_landmarks_2d(params)
                # Compare ALL 28 points against C++ final state (same coordinate frame!)
                error = np.mean(np.linalg.norm(eye_lms_28 - cpp_left_final_lms, axis=1))
                cpp_left_errors.append(error)
            results['left_eye']['cpp']['errors'] = cpp_left_errors

    # Analyze left eye - Python
    if py_left:
        results['left_eye']['python']['iterations'] = len(py_left)
        results['left_eye']['python']['window_sizes'] = sorted(set(i['window_size'] for i in py_left))
        results['left_eye']['python']['phases'] = sorted(set(i['phase'] for i in py_left))

        # Compute eye landmark errors - compare against C++ final state
        if py_left[-1].get('landmarks') and cpp_left_final_lms is not None:
            errors = []
            for iter_data in py_left:
                if 'landmarks' in iter_data:
                    eye_lms = np.array(iter_data['landmarks'])
                    # Compare ALL 28 points against C++ final state (same coordinate frame!)
                    error = np.mean(np.linalg.norm(eye_lms - cpp_left_final_lms, axis=1))
                    errors.append(error)
            results['left_eye']['python']['errors'] = errors

    # Analyze right eye - C++
    if cpp_right:
        results['right_eye']['cpp']['iterations'] = len(cpp_right)
        results['right_eye']['cpp']['window_sizes'] = sorted(set(i['window_size'] for i in cpp_right))
        results['right_eye']['cpp']['phases'] = sorted(set(i['phase'] for i in cpp_right))

        # Compute C++ eye errors using EyePDM - compare against C++ final state (same coord frame)
        if eye_pdm_right is not None and cpp_right_final_lms is not None:
            cpp_right_errors = []
            for iter_data in cpp_right:
                # Convert C++ params to eye landmarks using EyePDM
                params = params_to_array(iter_data['params'])
                eye_lms_28 = eye_pdm_right.params_to_landmarks_2d(params)
                # Compare ALL 28 points against C++ final state (same coordinate frame!)
                error = np.mean(np.linalg.norm(eye_lms_28 - cpp_right_final_lms, axis=1))
                cpp_right_errors.append(error)
            results['right_eye']['cpp']['errors'] = cpp_right_errors

    # Analyze right eye - Python
    if py_right:
        results['right_eye']['python']['iterations'] = len(py_right)
        results['right_eye']['python']['window_sizes'] = sorted(set(i['window_size'] for i in py_right))
        results['right_eye']['python']['phases'] = sorted(set(i['phase'] for i in py_right))

        # Compute eye landmark errors - compare against C++ final state
        if py_right[-1].get('landmarks') and cpp_right_final_lms is not None:
            errors = []
            for iter_data in py_right:
                if 'landmarks' in iter_data:
                    eye_lms = np.array(iter_data['landmarks'])
                    # Compare ALL 28 points against C++ final state (same coordinate frame!)
                    error = np.mean(np.linalg.norm(eye_lms - cpp_right_final_lms, axis=1))
                    errors.append(error)
            results['right_eye']['python']['errors'] = errors

    # Compute summary statistics
    results['summary'] = {
        'cpp_total_eye_iterations': len(cpp_eye_iterations),
        'python_total_eye_iterations': len(python_eye_iterations),
        'cpp_left_iterations': results['left_eye']['cpp']['iterations'],
        'python_left_iterations': results['left_eye']['python']['iterations'],
        'cpp_right_iterations': results['right_eye']['cpp']['iterations'],
        'python_right_iterations': results['right_eye']['python']['iterations'],
        'iteration_ratio': len(python_eye_iterations) / len(cpp_eye_iterations) if cpp_eye_iterations else 0,
        'symmetry_score': abs(results['left_eye']['python']['iterations'] -
                            results['right_eye']['python']['iterations'])
    }

    # Add Python convergence metrics if we have error data
    if results['left_eye']['python'].get('errors'):
        left_errors = results['left_eye']['python']['errors']
        if left_errors:
            results['summary']['python_left_final_error'] = left_errors[-1] if left_errors else 0
            results['summary']['python_left_convergence'] = (left_errors[0] - left_errors[-1]) / left_errors[0] if left_errors[0] > 0 else 0

    if results['right_eye']['python'].get('errors'):
        right_errors = results['right_eye']['python']['errors']
        if right_errors:
            results['summary']['python_right_final_error'] = right_errors[-1] if right_errors else 0
            results['summary']['python_right_convergence'] = (right_errors[0] - right_errors[-1]) / right_errors[0] if right_errors[0] > 0 else 0

    # Add C++ convergence metrics if we have error data
    if results['left_eye']['cpp'].get('errors'):
        cpp_left_errors = results['left_eye']['cpp']['errors']
        if cpp_left_errors:
            results['summary']['cpp_left_final_error'] = cpp_left_errors[-1] if cpp_left_errors else 0
            results['summary']['cpp_left_convergence'] = (cpp_left_errors[0] - cpp_left_errors[-1]) / cpp_left_errors[0] if cpp_left_errors[0] > 0 else 0

    if results['right_eye']['cpp'].get('errors'):
        cpp_right_errors = results['right_eye']['cpp']['errors']
        if cpp_right_errors:
            results['summary']['cpp_right_final_error'] = cpp_right_errors[-1] if cpp_right_errors else 0
            results['summary']['cpp_right_convergence'] = (cpp_right_errors[0] - cpp_right_errors[-1]) / cpp_right_errors[0] if cpp_right_errors[0] > 0 else 0

    return results


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


def plot_convergence_with_eyes(results, eye_results, frame_idx, output_path, final_error=None):
    """Create 4-panel plot showing face and eye convergence.

    Args:
        results: Face convergence results
        eye_results: Eye convergence results
        frame_idx: Frame index
        output_path: Path to save the plot
        final_error: Optional dict with post-eye-refinement 68-point error
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Frame {frame_idx}: Face and Eye Convergence Analysis', fontsize=16, fontweight='bold')

    # Panel 1: Face convergence (top left)
    ax1 = axes[0, 0]
    if results['cpp']['errors']:
        ax1.plot(results['cpp']['errors'], 'b-', label='C++', linewidth=2)
    if results['python']['errors']:
        ax1.plot(results['python']['errors'], 'r-', label='Python', linewidth=2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Error (pixels)')
    ax1.set_title('Face Model Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Add phase transitions
    for i, phase in enumerate(results['python']['iterations']):
        if i > 0 and phase['phase'] != results['python']['iterations'][i-1]['phase']:
            ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
            ax1.text(i, ax1.get_ylim()[1]*0.8, phase['phase'], rotation=90, va='top')

    # Mark transition to eye model (after face model completes)
    face_iter_count = len(results['python']['iterations'])
    if face_iter_count > 0:
        ax1.axvline(x=face_iter_count - 0.5, color='purple', linestyle='--',
                    linewidth=2.5, label='Eye Model', alpha=0.8)
        ylim = ax1.get_ylim()
        ax1.text(face_iter_count + 0.5, ylim[1] * 0.7, 'Eye Model →',
                fontsize=9, fontweight='bold', color='purple', va='top')

        # Add post-eye-refinement final error point (after eye model runs)
        if final_error is not None:
            # Plot Python's final 68-point error after eye refinement
            post_eye_x = face_iter_count + 3  # Position slightly after Eye Model line
            py_final = final_error['total_68']['mean']
            ax1.scatter([post_eye_x], [py_final], color='red', s=150, marker='*',
                       zorder=5, label=f'Python Post-Eye ({py_final:.2f}px)')
            # C++ reference (by definition 0, but show for comparison)
            ax1.scatter([post_eye_x], [results['cpp']['errors'][-1]], color='blue', s=150, marker='*',
                       zorder=5, label=f'C++ Final ({results["cpp"]["errors"][-1]:.2f}px)')
            # Extend x-axis to show the post-eye points
            ax1.set_xlim(right=post_eye_x + 2)

        ax1.legend()  # Update legend to include Eye Model line and post-eye points

    # Panel 2: Face error change (top right)
    ax2 = axes[0, 1]
    if results['python']['errors']:
        errors = results['python']['errors']
        error_changes = [0] + [errors[i] - errors[i-1] for i in range(1, len(errors))]
        colors = ['r' if e > 0 else 'b' for e in error_changes]
        ax2.bar(range(len(error_changes)), error_changes, color=colors, alpha=0.6)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error Change (pixels)')
    ax2.set_title('Error Change Per Iteration (positive = oscillation)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Left eye convergence (bottom left)
    ax3 = axes[1, 0]
    has_left_data = False

    # Plot C++ left eye errors (blue)
    if eye_results['left_eye']['cpp'].get('errors'):
        cpp_left_errors = eye_results['left_eye']['cpp']['errors']
        ax3.plot(cpp_left_errors, 'b-', label='C++ Left Eye', linewidth=2)
        has_left_data = True

    # Plot Python left eye errors (green)
    if eye_results['left_eye']['python'].get('errors'):
        py_left_errors = eye_results['left_eye']['python']['errors']
        ax3.plot(py_left_errors, 'g-', label='Python Left Eye', linewidth=2)
        has_left_data = True

    if has_left_data:
        ax3.set_xlabel('Eye Iteration')
        ax3.set_ylabel('Mean Error (pixels)')
        ax3.set_title('Left Eye Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Mark window size transitions for Python
        py_left = [i for i in eye_results.get('python_iterations', []) if i.get('eye_side') == 'left']
        if py_left:
            prev_ws = None
            for i, iter_data in enumerate(py_left):
                if iter_data['window_size'] != prev_ws and prev_ws is not None:
                    ax3.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
                    ax3.text(i, ax3.get_ylim()[0]*1.1, f"ws={iter_data['window_size']}", rotation=90)
                prev_ws = iter_data['window_size']
    else:
        ax3.text(0.5, 0.5, 'No left eye data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Left Eye Convergence')

    # Panel 4: Right eye convergence (bottom right)
    ax4 = axes[1, 1]
    has_right_data = False

    # Plot C++ right eye errors (blue)
    if eye_results['right_eye']['cpp'].get('errors'):
        cpp_right_errors = eye_results['right_eye']['cpp']['errors']
        ax4.plot(cpp_right_errors, 'b-', label='C++ Right Eye', linewidth=2)
        has_right_data = True

    # Plot Python right eye errors (magenta)
    if eye_results['right_eye']['python'].get('errors'):
        py_right_errors = eye_results['right_eye']['python']['errors']
        ax4.plot(py_right_errors, 'm-', label='Python Right Eye', linewidth=2)
        has_right_data = True

    if has_right_data:
        ax4.set_xlabel('Eye Iteration')
        ax4.set_ylabel('Mean Error (pixels)')
        ax4.set_title('Right Eye Convergence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Mark window size transitions for Python
        py_right = [i for i in eye_results.get('python_iterations', []) if i.get('eye_side') == 'right']
        if py_right:
            prev_ws = None
            for i, iter_data in enumerate(py_right):
                if iter_data['window_size'] != prev_ws and prev_ws is not None:
                    ax4.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
                    ax4.text(i, ax4.get_ylim()[0]*1.1, f"ws={iter_data['window_size']}", rotation=90)
                prev_ws = iter_data['window_size']
    else:
        ax4.text(0.5, 0.5, 'No right eye data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Right Eye Convergence')

    # Add summary text
    fig.text(0.5, 0.02,
            f"Face: {results['python']['iterations'][-1]} iterations | "
            f"Eyes: {eye_results['summary']['python_total_eye_iterations']} iterations | "
            f"Efficiency: Python uses {eye_results['summary']['iteration_ratio']:.0%} of C++ eye iterations",
            ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_diagnostic_analysis(py_iterations, frame_idx, output_path):
    """Generate diagnostic plots for late-stage convergence analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Frame {frame_idx}: Late-Stage Convergence Diagnostics', fontsize=14, fontweight='bold')

    # Filter iterations with diagnostic data
    nonrigid_iters = [it for it in py_iterations if it['phase'] == 'nonrigid' and it.get('hessian_cond') is not None]

    if not nonrigid_iters:
        plt.text(0.5, 0.5, 'No diagnostic data available', ha='center', va='center')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Panel 1: Hessian condition number
    ax1 = axes[0, 0]
    hessian_cond = [it['hessian_cond'] for it in nonrigid_iters]
    ax1.plot(range(len(hessian_cond)), hessian_cond, 'r-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Nonrigid Iteration')
    ax1.set_ylabel('Condition Number')
    ax1.set_title('Hessian Matrix Conditioning')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Panel 2: Regularization ratio
    ax2 = axes[0, 1]
    reg_ratio = [it['reg_ratio'] for it in nonrigid_iters if it['reg_ratio'] is not None]
    ax2.plot(range(len(reg_ratio)), reg_ratio, 'b-o', linewidth=2, markersize=4)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Reg = Data Term')
    ax2.set_xlabel('Nonrigid Iteration')
    ax2.set_ylabel('Reg Term / Data Term')
    ax2.set_title('Regularization Dominance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Mean-shift decay
    ax3 = axes[1, 0]
    all_ms = [it['mean_shift_norm'] for it in py_iterations]
    ax3.plot(range(len(all_ms)), all_ms, 'g-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Mean-Shift Norm')
    ax3.set_title('Mean-Shift Decay')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Mark phase transitions
    for i, it in enumerate(py_iterations):
        if i > 0 and it['phase'] != py_iterations[i-1]['phase']:
            ax3.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    # Panel 4: Per-landmark mean-shift at final iteration
    ax4 = axes[1, 1]
    if 'per_landmark_ms' in nonrigid_iters[-1]:
        final_ms = nonrigid_iters[-1]['per_landmark_ms']

        # Color by face region
        colors = []
        for i in range(len(final_ms)):
            if i < 17:
                colors.append('gray')      # Jaw
            elif i < 27:
                colors.append('orange')    # Eyebrows
            elif i < 36:
                colors.append('green')     # Nose
            elif i < 48:
                colors.append('blue')      # Eyes
            else:
                colors.append('red')       # Mouth

        ax4.bar(range(len(final_ms)), final_ms, color=colors, alpha=0.7)
        ax4.set_xlabel('Landmark Index')
        ax4.set_ylabel('Mean-Shift Magnitude')
        ax4.set_title('Final Per-Landmark Mean-Shift')
        ax4.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', label='Jaw (0-16)'),
            Patch(facecolor='orange', label='Brows (17-26)'),
            Patch(facecolor='green', label='Nose (27-35)'),
            Patch(facecolor='blue', label='Eyes (36-47)'),
            Patch(facecolor='red', label='Mouth (48-67)'),
        ]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Add summary text
    final_cond = hessian_cond[-1] if hessian_cond else 0
    final_reg = reg_ratio[-1] if reg_ratio else 0
    fig.text(0.5, 0.02,
            f"Final Hessian Cond: {final_cond:.2e} | Final Reg Ratio: {final_reg:.4f} | "
            f"Reg {'dominates' if final_reg > 1 else 'does NOT dominate'} data term",
            ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


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


def print_eye_summary(eye_results):
    """Print eye convergence summary."""
    summary = eye_results['summary']

    print("\n" + "=" * 100)
    print("EYE CONVERGENCE SUMMARY")
    print("=" * 100)

    # Iteration counts
    print(f"\nIteration Counts:")
    print(f"  C++ Left Eye:    {summary['cpp_left_iterations']} iterations")
    print(f"  Python Left Eye: {summary['python_left_iterations']} iterations")
    print(f"  C++ Right Eye:   {summary['cpp_right_iterations']} iterations")
    print(f"  Python Right Eye: {summary['python_right_iterations']} iterations")

    # Efficiency
    if summary['cpp_total_eye_iterations'] > 0:
        print(f"\nEfficiency:")
        print(f"  Python uses {summary['iteration_ratio']:.1%} of C++ iterations")
        print(f"  C++: {summary['cpp_total_eye_iterations']} total, Python: {summary['python_total_eye_iterations']} total")

    # Window sizes
    if eye_results['left_eye']['python']['window_sizes']:
        print(f"\nWindow Sizes Used:")
        print(f"  Left Eye:  {eye_results['left_eye']['python']['window_sizes']}")
        print(f"  Right Eye: {eye_results['right_eye']['python']['window_sizes']}")

    # Convergence metrics - C++
    if 'cpp_left_final_error' in summary or 'cpp_right_final_error' in summary:
        print(f"\nC++ Convergence Quality:")
        if 'cpp_left_final_error' in summary:
            print(f"  Left Eye Final Error:  {summary.get('cpp_left_final_error', 0):.2f} px")
        if 'cpp_right_final_error' in summary:
            print(f"  Right Eye Final Error: {summary.get('cpp_right_final_error', 0):.2f} px")
        if 'cpp_left_convergence' in summary:
            print(f"  Left Eye Convergence:  {summary.get('cpp_left_convergence', 0):.1%}")
        if 'cpp_right_convergence' in summary:
            print(f"  Right Eye Convergence: {summary.get('cpp_right_convergence', 0):.1%}")

    # Convergence metrics - Python
    if 'python_left_final_error' in summary or 'python_right_final_error' in summary:
        print(f"\nPython Convergence Quality:")
        if 'python_left_final_error' in summary:
            print(f"  Left Eye Final Error:  {summary.get('python_left_final_error', 0):.2f} px")
        if 'python_right_final_error' in summary:
            print(f"  Right Eye Final Error: {summary.get('python_right_final_error', 0):.2f} px")
        if 'python_left_convergence' in summary:
            print(f"  Left Eye Convergence:  {summary.get('python_left_convergence', 0):.1%}")
        if 'python_right_convergence' in summary:
            print(f"  Right Eye Convergence: {summary.get('python_right_convergence', 0):.1%}")

    # Symmetry
    print(f"\nSymmetry:")
    print(f"  Iteration difference between eyes: {summary['symmetry_score']}")
    if summary['symmetry_score'] == 0:
        print("  ✓ Perfect symmetry in iteration counts")
    elif summary['symmetry_score'] <= 2:
        print("  ✓ Good symmetry")
    else:
        print("  ⚠️ Asymmetric convergence")


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

    # Initialize CLNF with iterations matching C++ (~40 total)
    print("Initializing Python CLNF...")
    clnf = CLNF(
        model_dir=str(MODEL_DIR),
        detector=None,
        max_iterations=20,  # Distributed across windows: ~40 total to match C++
        convergence_threshold=1e-6,  # Disabled - no early stopping
        regularization=25,  # C++ default
        window_sizes=[11, 9, 7],  # Skip WS=5 which causes overfitting
        debug_mode=False  # Disable debug for clean performance comparison
    )
    print(f"  CLNF initialized (max_iterations={clnf.optimizer.max_iterations} per phase)")
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
        bbox, cpp_iterations, cpp_eye_iterations, cpp_final_landmarks = run_cpp_on_frame(
            temp_frame_path, cpp_output_dir, include_eyes=True
        )

        if bbox is None or cpp_iterations is None or len(cpp_iterations) == 0:
            print(f"  Error: C++ failed on frame {frame_idx}")
            continue

        print(f"    Bbox: {bbox}")
        print(f"    C++ face iterations: {len(cpp_iterations)}")
        print(f"    C++ eye iterations: {len(cpp_eye_iterations)}")

        # Run Python on frame with same bbox
        print("  Running Python pyclnf...")
        py_landmarks, py_iterations, py_eye_iterations = run_python_on_frame(frame, bbox, clnf)

        if py_landmarks is None or len(py_iterations) == 0:
            print(f"  Error: Python failed on frame {frame_idx}")
            continue

        print(f"    Python face iterations: {len(py_iterations)}")
        print(f"    Python eye iterations: {len(py_eye_iterations)}")

        # Save Python eye iterations to trace file for debugging
        if py_eye_iterations:
            python_eye_trace = TRACE_DIR / "python_eye_trace.txt"
            with open(python_eye_trace, 'w') as f:
                f.write("# Python Eye Model Trace\n")
                for iter_data in py_eye_iterations:
                    # Format similar to C++ trace for easy comparison
                    params_str = ' '.join(f"{p:.8f}" for p in iter_data['params']['global'] + iter_data['params']['local'])
                    f.write(f"{iter_data['iteration']} {iter_data['window_size']} {iter_data['phase']} {iter_data['eye_side']} {params_str}\n")
            print(f"    Saved Python eye trace to {python_eye_trace}")

        # Analyze face convergence
        print("  Analyzing face convergence...")
        results = analyze_convergence(cpp_iterations, py_iterations, pdm, cpp_final_landmarks)

        # Analyze eye convergence
        print("  Analyzing eye convergence...")
        eye_results = analyze_eye_convergence(cpp_eye_iterations, py_eye_iterations, cpp_final_landmarks)

        # Print face results
        print_iteration_table(results)
        print_summary(results, frame_idx)

        # Print eye convergence summary
        print_eye_summary(eye_results)

        # Print final combined error (after all refinement)
        final_error = compute_final_combined_error(py_landmarks, cpp_final_landmarks)
        print("\n" + "=" * 100)
        print("FINAL COMBINED ERROR (After Eye Refinement)")
        print("=" * 100)
        print(f"\n  Total 68-point error: {final_error['total_68']['mean']:.2f} px (max: {final_error['total_68']['max']:.2f} px)")
        print(f"  Face (excl eyes):     {final_error['face_only']['mean']:.2f} px")
        print(f"  Left eye (36-41):     {final_error['left_eye']['mean']:.2f} px")
        print(f"  Right eye (42-47):    {final_error['right_eye']['mean']:.2f} px")

        # Store eye results for plotting
        eye_results['python_iterations'] = py_eye_iterations

        # Generate enhanced plot with eye convergence
        plot_path = OUTPUT_DIR / f"convergence_frame_{frame_idx}.png"
        plot_convergence_with_eyes(results, eye_results, frame_idx, plot_path, final_error=final_error)
        print(f"\n  Enhanced plot saved: {plot_path}")

        # Generate diagnostic analysis plot for first frame only (to understand late-stage convergence)
        if frame_idx == 0:
            diag_path = OUTPUT_DIR / f"diagnostic_frame_{frame_idx}.png"
            plot_diagnostic_analysis(py_iterations, frame_idx, diag_path)
            print(f"  Diagnostic plot saved: {diag_path}")

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
