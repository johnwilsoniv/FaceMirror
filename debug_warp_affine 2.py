#!/usr/bin/env python3
"""
Debug warpAffine to find why C++ and Python extract different patches.
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, align_shapes_with_scale
from pyclnf.core.eye_pdm import EyePDM

def main():
    print("=" * 70)
    print("DEBUG WARPAFFINE TRANSFORMATION")
    print("=" * 70)

    # Load image
    image_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov'
    cap = cv2.VideoCapture(image_path)
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print(f"Image shape: {gray.shape}")

    # Load eye PDM
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/models/exported_eye_pdm_left'
    left_eye_pdm = EyePDM(model_dir)

    # Initial params from C++ debug
    # Initial global params: scale=3.371204 rot=(-0.118319,0.176098,-0.099366) tx=425.031629 ty=820.112023
    params = np.zeros(left_eye_pdm.n_params)
    params[0] = 3.371204  # scale
    params[1] = -0.118319  # rx
    params[2] = 0.176098   # ry
    params[3] = -0.099366  # rz
    params[4] = 425.031629 # tx
    params[5] = 820.112023 # ty

    # Get landmarks
    eye_landmarks = left_eye_pdm.params_to_landmarks_2d(params)

    print(f"\nEye landmarks shape: {eye_landmarks.shape}")
    print(f"Landmark 8: ({eye_landmarks[8, 0]:.4f}, {eye_landmarks[8, 1]:.4f})")

    # Compute similarity transform (like _compute_sim_transforms)
    image_shape = eye_landmarks  # (28, 2)

    ref_params = params.copy()
    ref_params[0] = 1.0  # patch_scale
    ref_params[1:4] = 0  # no rotation
    ref_params[4:6] = 0  # no translation
    reference_shape = left_eye_pdm.params_to_landmarks_2d(ref_params)  # (28, 2)

    sim_img_to_ref = align_shapes_with_scale(image_shape, reference_shape)
    sim_ref_to_img = np.linalg.inv(sim_img_to_ref)

    print(f"\nsim_ref_to_img:")
    print(f"  [{sim_ref_to_img[0, 0]:.6f}, {sim_ref_to_img[0, 1]:.6f}]")
    print(f"  [{sim_ref_to_img[1, 0]:.6f}, {sim_ref_to_img[1, 1]:.6f}]")

    # Extract a1, b1 like C++ (note the sign!)
    a1 = sim_ref_to_img[0, 0]
    b1 = -sim_ref_to_img[0, 1]  # C++ uses -sim_ref_to_img(0,1)

    print(f"\na1 = {a1:.6f}")
    print(f"b1 = {b1:.6f} (negated from sim_ref_to_img[0,1] = {sim_ref_to_img[0, 1]:.6f})")

    # For window_size=3, patch=11x11: AOI = 13x13
    window_size = 3
    patch_size = 11
    aoi_size = window_size + patch_size - 1
    half_aoi = (aoi_size - 1) / 2.0

    print(f"\nwindow_size={window_size}, patch_size={patch_size}")
    print(f"aoi_size={aoi_size}, half_aoi={half_aoi}")

    # Test for landmark 8
    lm_idx = 8
    x, y = eye_landmarks[lm_idx]

    print(f"\n=== Landmark {lm_idx} ===")
    print(f"Position: ({x:.4f}, {y:.4f})")

    # C++ transformation matrix construction:
    # tx = landmark_x - a1 * half_aoi + b1 * half_aoi
    # ty = landmark_y - a1 * half_aoi - b1 * half_aoi
    tx = x - a1 * half_aoi + b1 * half_aoi
    ty = y - a1 * half_aoi - b1 * half_aoi

    print(f"\nTransformation:")
    print(f"  tx = {x:.4f} - {a1:.4f} * {half_aoi:.1f} + {b1:.4f} * {half_aoi:.1f} = {tx:.4f}")
    print(f"  ty = {y:.4f} - {a1:.4f} * {half_aoi:.1f} - {b1:.4f} * {half_aoi:.1f} = {ty:.4f}")

    # Build transformation matrix
    sim = np.array([[a1, -b1, tx],
                    [b1, a1, ty]], dtype=np.float32)

    print(f"\nwarpAffine matrix:")
    print(f"  [{sim[0, 0]:.6f}, {sim[0, 1]:.6f}, {sim[0, 2]:.6f}]")
    print(f"  [{sim[1, 0]:.6f}, {sim[1, 1]:.6f}, {sim[1, 2]:.6f}]")

    # Extract AOI
    area_of_interest = cv2.warpAffine(
        gray,
        sim,
        (aoi_size, aoi_size),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
    )

    print(f"\nExtracted AOI shape: {area_of_interest.shape}")
    print(f"AOI stats: min={area_of_interest.min():.1f}, max={area_of_interest.max():.1f}, mean={area_of_interest.mean():.1f}")

    print(f"\nAOI values (full 13x13):")
    for row in range(aoi_size):
        row_str = "  "
        for col in range(aoi_size):
            row_str += f"{area_of_interest[row, col]:6.1f} "
        print(row_str)

    # What the center pixel should correspond to
    center = aoi_size // 2
    print(f"\nCenter pixel [{center},{center}] = {area_of_interest[center, center]:.1f}")

    # Verify: inverse warp of center should give original landmark position
    # With WARP_INVERSE_MAP: dst[x,y] = src[M @ [x,y,1]]
    # So AOI[center,center] should come from image at landmark position
    src_x = sim[0, 0] * center + sim[0, 1] * center + sim[0, 2]
    src_y = sim[1, 0] * center + sim[1, 1] * center + sim[1, 2]
    print(f"Inverse transform: AOI center -> image ({src_x:.2f}, {src_y:.2f})")
    print(f"Expected landmark position: ({x:.2f}, {y:.2f})")
    print(f"Difference: ({src_x - x:.4f}, {src_y - y:.4f})")

    # Now compare with what we expect from C++ debug
    print("\n" + "=" * 70)
    print("Compare with C++ /tmp/cpp_eye_ccnf_neuron_debug.txt")
    print("C++ AOI (first 5x5):")
    print("  131.6 131.4 124.6 130.5 127.2")
    print("  132.7 130.9 135.5  88.8  82.5")
    print("  112.5  72.1  34.9  37.8  40.6")
    print("   56.4  40.1  58.7  71.9  42.5")
    print("   48.3  73.6  97.3 101.7  46.2")
    print("=" * 70)

if __name__ == '__main__':
    main()
