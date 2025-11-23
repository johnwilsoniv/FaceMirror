#!/usr/bin/env python3
"""
Debug eye model mean-shift computation step by step.
Verify Y-axis convention is correct.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pathlib import Path

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING
from pyclnf.core.eye_pdm import EyePDM

def compute_mean_shift_with_debug(response_map, sigma=1.0):
    """Compute mean-shift with detailed debug output."""
    ws = response_map.shape[0]
    center = (ws - 1) / 2.0
    a_kde = -0.5 / (sigma * sigma)

    total_weight = 0.0
    mx = 0.0
    my = 0.0

    print(f"  Response map shape: {ws}x{ws}, center: {center}")
    print(f"  Response range: [{response_map.min():.6f}, {response_map.max():.6f}]")

    # Find peak
    peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
    peak_row, peak_col = peak_idx
    print(f"  Peak at row={peak_row}, col={peak_col} (value={response_map[peak_row, peak_col]:.6f})")
    print(f"  Peak offset from center: row_off={peak_row - center:+.1f}, col_off={peak_col - center:+.1f}")

    # Compute weighted sums
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

    print(f"  Total weight: {total_weight:.6f}")
    print(f"  Weighted avg col: {mx/total_weight:.4f}, row: {my/total_weight:.4f}")
    print(f"  Mean-shift: ({ms_x:+.4f}, {ms_y:+.4f})")

    # Interpretation
    if ms_y < 0:
        print(f"  -> Y negative means move UP (toward lower row in image)")
    else:
        print(f"  -> Y positive means move DOWN (toward higher row in image)")

    return ms_x, ms_y

def main():
    print("=" * 70)
    print("EYE MODEL MEAN-SHIFT DEBUG")
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

    print("\nInitial eye landmarks:")
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        lm = eye_landmarks[eye_idx]
        print(f"  Eye_{eye_idx}: ({lm[0]:.2f}, {lm[1]:.2f})")

    # Get patch experts
    scale = eye_model.patch_scale
    ccnf = eye_model.ccnf['left']
    patch_experts = ccnf.get_all_patch_experts(scale)

    print(f"\nLoaded {len(patch_experts)} patch experts at scale {scale}")

    # Compute response maps for first window size
    ws = eye_model.window_sizes[0]
    eye_model._current_window_size = ws
    half_ws = ws // 2

    print(f"\nComputing response maps (window size {ws})...")
    response_maps = eye_model._compute_eye_response_maps(gray, eye_landmarks, patch_experts)

    # Analyze each visible landmark
    print("\n" + "=" * 70)
    print("DETAILED MEAN-SHIFT ANALYSIS")
    print("=" * 70)

    # C++ refinement results for comparison
    CPP_DELTA = {
        36: (+0.4725, +2.5932),
        37: (+0.4878, +2.8680),
        38: (+0.2048, +2.5201),
        39: (-0.5040, +1.1696),
        40: (-0.0311, +1.9948),
        41: (+0.4704, +2.0950)
    }

    eye_to_main = {8: 36, 10: 37, 12: 38, 14: 39, 16: 40, 18: 41}

    for eye_idx in [8, 10, 12, 14, 16, 18]:
        if eye_idx not in response_maps:
            print(f"\nEye_{eye_idx}: No response map!")
            continue

        main_idx = eye_to_main[eye_idx]
        lm = eye_landmarks[eye_idx]
        rm = response_maps[eye_idx]
        cpp_d = CPP_DELTA[main_idx]

        print(f"\nEye_{eye_idx} (main {main_idx}) at ({lm[0]:.2f}, {lm[1]:.2f}):")
        print(f"  C++ delta: ({cpp_d[0]:+.4f}, {cpp_d[1]:+.4f})")

        # Note: C++ moved this landmark DOWN (positive Y)
        if cpp_d[1] > 0:
            print(f"  -> C++ moved DOWN by {cpp_d[1]:.2f} pixels")
        else:
            print(f"  -> C++ moved UP by {abs(cpp_d[1]):.2f} pixels")

        ms_x, ms_y = compute_mean_shift_with_debug(rm)

        # Check if direction matches
        y_match = (ms_y > 0) == (cpp_d[1] > 0) if abs(cpp_d[1]) > 0.1 else True
        x_match = (ms_x > 0) == (cpp_d[0] > 0) if abs(cpp_d[0]) > 0.1 else True

        if y_match and x_match:
            print(f"  ✓ Direction matches C++")
        else:
            print(f"  ✗ DIRECTION MISMATCH!")
            if not y_match:
                print(f"    - Y direction wrong: Python wants {'+' if ms_y > 0 else '-'}, C++ moved {'+' if cpp_d[1] > 0 else '-'}")

        # Print 3x3 response map around center for visualization
        print(f"  3x3 around center:")
        for r in range(half_ws - 1, half_ws + 2):
            row_str = "    "
            for c in range(half_ws - 1, half_ws + 2):
                row_str += f"{rm[r, c]:.4f} "
            print(row_str)

    # Now let's check the actual mean-shift vector from the eye model
    print("\n" + "=" * 70)
    print("FULL MEAN-SHIFT VECTOR")
    print("=" * 70)

    mean_shift = eye_model._compute_eye_mean_shift(eye_landmarks, response_maps, patch_experts)

    print("\nMean-shift for visible landmarks:")
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        ms_x = mean_shift[2 * eye_idx]
        ms_y = mean_shift[2 * eye_idx + 1]
        print(f"  Eye_{eye_idx}: ({ms_x:+.4f}, {ms_y:+.4f})")

    # Test: create a synthetic response map with clear peak and verify
    print("\n" + "=" * 70)
    print("SANITY CHECK: Synthetic response map")
    print("=" * 70)

    test_rm = np.zeros((9, 9), dtype=np.float32)
    # Put peak at bottom-right (row 8, col 8) - should give positive ms_x and ms_y
    test_rm[8, 8] = 1.0
    print("\nTest: Peak at bottom-right (row=8, col=8)")
    print("  Expected: positive ms_x (right), positive ms_y (down)")
    compute_mean_shift_with_debug(test_rm)

    # Put peak at top-left (row 0, col 0) - should give negative ms_x and ms_y
    test_rm = np.zeros((9, 9), dtype=np.float32)
    test_rm[0, 0] = 1.0
    print("\nTest: Peak at top-left (row=0, col=0)")
    print("  Expected: negative ms_x (left), negative ms_y (up)")
    compute_mean_shift_with_debug(test_rm)

if __name__ == '__main__':
    main()
