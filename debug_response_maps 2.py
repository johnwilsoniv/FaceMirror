#!/usr/bin/env python3
"""
Debug response maps for landmarks 0, 1, 8 to find where Python differs from C++.
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("DEBUG RESPONSE MAPS FOR LANDMARKS 0, 1, 8")
    print("=" * 70)

    # Load image
    image_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov'
    cap = cv2.VideoCapture(image_path)
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Initialize eye patch expert
    eye_expert = EyePatchExpert()

    # Load eye PDM
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_pdm_left'
    left_eye_pdm = EyePDM(model_dir)

    # Initial params from debug
    params = np.zeros(left_eye_pdm.n_params)
    params[0] = 3.371204  # scale
    params[1] = -0.118319  # rx
    params[2] = 0.176098   # ry
    params[3] = -0.099366  # rz
    params[4] = 425.031629 # tx
    params[5] = 820.112023 # ty

    # Get landmarks
    eye_landmarks = left_eye_pdm.params_to_landmarks_2d(params)

    print(f"\nLandmark positions:")
    for i in [0, 1, 8]:
        print(f"  Landmark {i}: ({eye_landmarks[i, 0]:.4f}, {eye_landmarks[i, 1]:.4f})")

    # Compute similarity transform
    sim_ref_to_img, sim_img_to_ref = eye_expert._compute_sim_transforms(
        eye_landmarks, params, left_eye_pdm, 'left'
    )

    print(f"\nsim_ref_to_img:")
    print(f"  [{sim_ref_to_img[0, 0]:.6f}, {sim_ref_to_img[0, 1]:.6f}]")
    print(f"  [{sim_ref_to_img[1, 0]:.6f}, {sim_ref_to_img[1, 1]:.6f}]")

    # Get patch experts for this window size
    window_size = 3
    eye_expert._current_window_size = window_size
    patch_experts = eye_expert._get_eye_patch_experts(window_size)

    print(f"\nPatch experts loaded: {len(patch_experts)} landmarks")

    # Compute response maps for landmarks 0, 1, 8
    debug_landmarks = [0, 1, 8]

    for lm_idx in debug_landmarks:
        if lm_idx not in patch_experts:
            print(f"\n--- Landmark {lm_idx}: NO PATCH EXPERT ---")
            continue

        expert = patch_experts[lm_idx]
        lm_pos = eye_landmarks[lm_idx]

        print(f"\n--- Landmark {lm_idx} ---")
        print(f"  Position: ({lm_pos[0]:.4f}, {lm_pos[1]:.4f})")

        # Extract patch (this is what Python does)
        # Need to transform landmark to get patch center
        half_ws = (window_size - 1) / 2.0

        # Get patch bounds in image
        # The patch is extracted around the landmark position
        # Transform: patch_coord = sim_ref_to_img @ ref_coord + landmark

        # Actually, let me trace the actual response computation
        response_map = eye_expert._compute_single_response(
            gray, lm_pos, expert, sim_ref_to_img, window_size
        )

        if response_map is not None:
            print(f"  Response map shape: {response_map.shape}")
            print(f"  Response map min: {response_map.min():.6f}")
            print(f"  Response map max: {response_map.max():.6f}")
            print(f"  Response map mean: {response_map.mean():.6f}")

            # Find peak
            peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
            center = (window_size - 1) / 2.0
            peak_offset_x = peak_idx[1] - center
            peak_offset_y = peak_idx[0] - center
            print(f"  Peak at: ({peak_idx[1]}, {peak_idx[0]})")
            print(f"  Peak offset from center: ({peak_offset_x:.1f}, {peak_offset_y:.1f})")

            # Print response map values
            print(f"  Response map values:")
            for row in range(response_map.shape[0]):
                row_str = "    "
                for col in range(response_map.shape[1]):
                    row_str += f"{response_map[row, col]:.4f} "
                print(row_str)

            # Compute mean-shift
            sigma = 1.0
            a_kde = -0.5 / (sigma * sigma)

            total_weight = 0.0
            mx = 0.0
            my = 0.0

            for ii in range(window_size):
                for jj in range(window_size):
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
                ms_x = 0.0
                ms_y = 0.0

            print(f"  KDE mean-shift (ref space): ({ms_x:.6f}, {ms_y:.6f})")

            # Transform to image space
            ms_img_x = ms_x * sim_ref_to_img[0, 0] + ms_y * sim_ref_to_img[0, 1]
            ms_img_y = ms_x * sim_ref_to_img[1, 0] + ms_y * sim_ref_to_img[1, 1]
            print(f"  Transformed mean-shift (img space): ({ms_img_x:.6f}, {ms_img_y:.6f})")
        else:
            print(f"  Response map: None")

    print("\n" + "=" * 70)
    print("Compare with C++ values from /tmp/cpp_eye_model_detailed.txt")
    print("=" * 70)

if __name__ == '__main__':
    main()
