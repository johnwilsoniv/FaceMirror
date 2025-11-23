#!/usr/bin/env python3
"""
Compare C++ and Python response maps to find the Y-axis discrepancy.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import struct
import cv2
from pathlib import Path

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING
from pyclnf.core.eye_pdm import EyePDM

def read_cpp_response_map(filepath):
    """Read C++ binary response map (float32 values)."""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Determine size from filename (e.g., ws11 = 11x11)
    ws = int(filepath.stem.split('ws')[-1])
    expected_size = ws * ws * 4  # float32

    if len(data) != expected_size:
        print(f"Warning: {filepath} has {len(data)} bytes, expected {expected_size}")
        return None, ws

    values = struct.unpack(f'{ws*ws}f', data)
    return np.array(values).reshape(ws, ws), ws

def compute_mean_shift(response_map, sigma=1.0):
    """Compute mean-shift from response map (same as Python code)."""
    ws = response_map.shape[0]
    center = (ws - 1) / 2.0
    a_kde = -0.5 / (sigma * sigma)

    total_weight = 0.0
    mx = 0.0
    my = 0.0

    for ii in range(ws):
        for jj in range(ws):
            dist_sq = (ii - center)**2 + (jj - center)**2
            kde_weight = np.exp(a_kde * dist_sq)
            weight = response_map[ii, jj] * kde_weight

            total_weight += weight
            mx += weight * jj
            my += weight * ii

    if total_weight > 1e-10:
        ms_x = (mx / total_weight) - center
        ms_y = (my / total_weight) - center
    else:
        ms_x = ms_y = 0.0

    return ms_x, ms_y

def main():
    print("=" * 70)
    print("C++ vs PYTHON RESPONSE MAP COMPARISON")
    print("=" * 70)

    # Check for C++ eye model debug files
    cpp_debug_dir = Path('/tmp/cpp_eye_debug')
    if not cpp_debug_dir.exists():
        cpp_debug_dir = Path('/tmp/cpp_debug')

    print(f"\nChecking directory: {cpp_debug_dir}")

    # List available response maps
    response_files = list(cpp_debug_dir.glob('response_*.bin'))
    print(f"Found {len(response_files)} response map files")

    for rf in sorted(response_files)[:6]:
        print(f"  {rf.name}")

    # Read and analyze each response map
    print("\n" + "=" * 70)
    print("RESPONSE MAP ANALYSIS")
    print("=" * 70)

    for rf in sorted(response_files):
        response_map, ws = read_cpp_response_map(rf)
        if response_map is None:
            continue

        # Find peak position
        peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
        peak_row, peak_col = peak_idx
        center = (ws - 1) / 2.0

        # Compute mean-shift
        ms_x, ms_y = compute_mean_shift(response_map)

        print(f"\n{rf.name}:")
        print(f"  Size: {ws}x{ws}, center: {center:.1f}")
        print(f"  Range: [{response_map.min():.6f}, {response_map.max():.6f}]")
        print(f"  Peak at: row={peak_row}, col={peak_col} (offset from center: {peak_row-center:+.1f}, {peak_col-center:+.1f})")
        print(f"  Mean-shift: ({ms_x:+.4f}, {ms_y:+.4f})")

        # Show 3x3 around peak
        if ws >= 3:
            print(f"  3x3 around peak:")
            for r in range(max(0, peak_row-1), min(ws, peak_row+2)):
                row_str = "    "
                for c in range(max(0, peak_col-1), min(ws, peak_col+2)):
                    row_str += f"{response_map[r, c]:.4f} "
                print(row_str)

    # Now generate Python response maps for comparison
    print("\n" + "=" * 70)
    print("GENERATING PYTHON RESPONSE MAPS FOR EYE MODEL")
    print("=" * 70)

    # Load video frame
    video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
    ret, frame = video.read()
    video.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load eye model
    model_dir = 'pyclnf/models'
    eye_model = HierarchicalEyeModel(model_dir)
    pdm = eye_model.pdm['left']

    # C++ input landmarks
    main_indices = [36, 37, 38, 39, 40, 41]
    CPP_LEFT_EYE_INPUT = {
        36: (392.1590, 847.6613),
        37: (410.0039, 828.3166),
        38: (436.9223, 826.1841),
        39: (461.9583, 842.8420),
        40: (438.4380, 850.4288),
        41: (411.4089, 853.9998)
    }

    target_points = np.array([CPP_LEFT_EYE_INPUT[i] for i in main_indices])

    # Fit shape parameters
    params = eye_model._fit_eye_shape(target_points, LEFT_EYE_MAPPING, 'left', main_rotation=None)

    # Get initial eye landmarks
    eye_landmarks = pdm.params_to_landmarks_2d(params)

    print(f"\nInitial eye landmarks (key visible points):")
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        lm = eye_landmarks[eye_idx]
        print(f"  Eye_{eye_idx}: ({lm[0]:.2f}, {lm[1]:.2f})")

    # Compute response maps for first window size
    ws = eye_model.window_sizes[0]
    eye_model._current_window_size = ws
    patch_experts = eye_model.patch_experts_left[ws]

    print(f"\nComputing response maps for window size {ws}...")
    response_maps = eye_model._compute_eye_response_maps(gray, eye_landmarks, patch_experts)

    print("\nPython response maps for visible eye landmarks:")
    for lm_idx in [8, 10, 12, 14, 16, 18]:
        if lm_idx not in response_maps:
            continue

        rm = response_maps[lm_idx]
        peak_idx = np.unravel_index(np.argmax(rm), rm.shape)
        peak_row, peak_col = peak_idx
        center = (ws - 1) / 2.0

        ms_x, ms_y = compute_mean_shift(rm)

        print(f"\n  Eye_{lm_idx}:")
        print(f"    Range: [{rm.min():.6f}, {rm.max():.6f}]")
        print(f"    Peak at: row={peak_row}, col={peak_col} (offset: {peak_row-center:+.1f}, {peak_col-center:+.1f})")
        print(f"    Mean-shift: ({ms_x:+.4f}, {ms_y:+.4f})")

        # Show 3x3 around center
        half = ws // 2
        print(f"    3x3 around center:")
        for r in range(half-1, half+2):
            row_str = "      "
            for c in range(half-1, half+2):
                row_str += f"{rm[r, c]:.4f} "
            print(row_str)

    # Now let's check C++ eye model response maps if they exist
    print("\n" + "=" * 70)
    print("CHECKING FOR C++ EYE MODEL DEBUG")
    print("=" * 70)

    cpp_eye_debug = Path('/tmp/cpp_eye_model_debug.txt')
    if cpp_eye_debug.exists():
        print(f"\nFound {cpp_eye_debug}, reading...")
        with open(cpp_eye_debug) as f:
            content = f.read()
        print(content[:2000] if len(content) > 2000 else content)
    else:
        print("\nNo C++ eye model debug file found.")
        print("The existing debug files are for main CLNF, not eye model.")
        print("\nTo properly compare, we need to add debug output to C++ eye model code.")

if __name__ == '__main__':
    main()
