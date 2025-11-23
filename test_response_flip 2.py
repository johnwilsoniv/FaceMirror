#!/usr/bin/env python3
"""
Test if flipping the response map vertically fixes the Y-axis issue.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pathlib import Path

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING
from pyclnf.core.eye_pdm import EyePDM

def compute_mean_shift(response_map, sigma=1.0):
    """Compute mean-shift from response map."""
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
    print("TEST RESPONSE MAP Y-FLIP")
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
    params = eye_model._fit_eye_shape(target_points, LEFT_EYE_MAPPING, 'left', main_rotation=None)
    eye_landmarks = pdm.params_to_landmarks_2d(params)

    # Get patch experts
    scale = eye_model.patch_scale
    ccnf = eye_model.ccnf['left']
    patch_experts = ccnf.get_all_patch_experts(scale)

    # Compute response maps
    ws = eye_model.window_sizes[0]
    eye_model._current_window_size = ws
    response_maps = eye_model._compute_eye_response_maps(gray, eye_landmarks, patch_experts)

    # C++ refinement deltas
    CPP_DELTA = {
        36: (+0.4725, +2.5932),
        37: (+0.4878, +2.8680),
        38: (+0.2048, +2.5201),
        39: (-0.5040, +1.1696),
        40: (-0.0311, +1.9948),
        41: (+0.4704, +2.0950)
    }

    eye_to_main = {8: 36, 10: 37, 12: 38, 14: 39, 16: 40, 18: 41}

    print("\nComparison: Original vs Flipped response maps\n")
    print("  Eye  |    C++ Δy   |  Orig ms_y  | Flipped ms_y | Flipped matches?")
    print("  -----|-------------|-------------|--------------|------------------")

    matches = 0
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        if eye_idx not in response_maps:
            continue

        main_idx = eye_to_main[eye_idx]
        rm = response_maps[eye_idx]
        cpp_dy = CPP_DELTA[main_idx][1]

        # Original mean-shift
        orig_ms_x, orig_ms_y = compute_mean_shift(rm)

        # Flipped response map (flip along Y-axis / flip rows)
        rm_flipped = np.flipud(rm)
        flip_ms_x, flip_ms_y = compute_mean_shift(rm_flipped)

        # Check if flipped matches C++ direction
        orig_match = (orig_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True
        flip_match = (flip_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True

        match_str = "✓" if flip_match else "✗"
        if flip_match:
            matches += 1

        print(f"  {eye_idx:3d}  | {cpp_dy:+10.4f} | {orig_ms_y:+10.4f}  | {flip_ms_y:+10.4f}   | {match_str}")

    print(f"\n  Matches with flip: {matches}/6")

    if matches >= 5:
        print("\n  *** Y-FLIP LIKELY NEEDED ***")
        print("  The response maps are inverted along Y-axis.")
        print("  Fix: Either flip response map after computing, or fix patch/weight storage.")
    else:
        print("\n  Flip doesn't help much. Issue might be elsewhere.")

    # Let's also test X-flip
    print("\n" + "=" * 70)
    print("ADDITIONAL TESTS")
    print("=" * 70)

    print("\nX-flip test (fliplr):")
    matches_x = 0
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        if eye_idx not in response_maps:
            continue

        main_idx = eye_to_main[eye_idx]
        rm = response_maps[eye_idx]
        cpp_dx = CPP_DELTA[main_idx][0]

        rm_flipped_x = np.fliplr(rm)
        flip_ms_x, flip_ms_y = compute_mean_shift(rm_flipped_x)

        x_match = (flip_ms_x > 0) == (cpp_dx > 0) if abs(cpp_dx) > 0.1 else True
        if x_match:
            matches_x += 1

    print(f"  X-direction matches after fliplr: {matches_x}/6")

    # Test both flips
    print("\nBoth X and Y flip test (flip both axes):")
    matches_both = 0
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        if eye_idx not in response_maps:
            continue

        main_idx = eye_to_main[eye_idx]
        rm = response_maps[eye_idx]
        cpp_dx = CPP_DELTA[main_idx][0]
        cpp_dy = CPP_DELTA[main_idx][1]

        rm_flipped = np.flip(rm)  # Flip both axes
        flip_ms_x, flip_ms_y = compute_mean_shift(rm_flipped)

        x_match = (flip_ms_x > 0) == (cpp_dx > 0) if abs(cpp_dx) > 0.1 else True
        y_match = (flip_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True
        if x_match and y_match:
            matches_both += 1

    print(f"  Both directions match after flip both: {matches_both}/6")

    # Transpose test
    print("\nTranspose test (swap rows and columns):")
    matches_trans = 0
    for eye_idx in [8, 10, 12, 14, 16, 18]:
        if eye_idx not in response_maps:
            continue

        main_idx = eye_to_main[eye_idx]
        rm = response_maps[eye_idx]
        cpp_dx = CPP_DELTA[main_idx][0]
        cpp_dy = CPP_DELTA[main_idx][1]

        rm_trans = rm.T
        trans_ms_x, trans_ms_y = compute_mean_shift(rm_trans)

        x_match = (trans_ms_x > 0) == (cpp_dx > 0) if abs(cpp_dx) > 0.1 else True
        y_match = (trans_ms_y > 0) == (cpp_dy > 0) if abs(cpp_dy) > 0.1 else True
        if x_match and y_match:
            matches_trans += 1

    print(f"  Both directions match after transpose: {matches_trans}/6")

if __name__ == '__main__':
    main()
