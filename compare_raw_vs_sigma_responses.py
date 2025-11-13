#!/usr/bin/env python3
"""
Compare raw neuron response magnitudes vs sigma-transformed responses.

Goal: Determine if the tiny response magnitudes are due to:
1. Weak raw neuron responses (problem in patch expert computation)
2. Over-attenuation by sigma transformation (problem in sigma formula or components)
"""

import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from pyclnf import CLNF
from pyclnf.core.patch_expert import CCNFModel


def analyze_raw_vs_sigma():
    """Compare response magnitudes before and after sigma transformation."""

    print("=" * 80)
    print("Raw vs Sigma-Transformed Response Analysis")
    print("=" * 80)

    # Load test image
    video_path = Path("Patient Data/Normal Cohort/IMG_0434.MOV")
    if not video_path.exists():
        print(f"Error: Test video not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to read video frame")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Set up face bbox
    face_size = min(width, height) // 2
    face_bbox = (
        width // 2 - face_size // 2,
        height // 2 - face_size // 2,
        face_size,
        face_size
    )

    print(f"\nImage: {gray.shape}")
    print(f"Face bbox: {face_bbox}")

    # Load CLNF model
    clnf = CLNF(model_dir="pyclnf/models", max_iterations=5)

    # Initialize parameters
    params = clnf.pdm.init_params(face_bbox)
    landmarks_2d = clnf.pdm.params_to_landmarks_2d(params)

    # Get patch experts for frontal view, scale 0.25
    view_idx = 0
    patch_scale = 0.25
    patch_experts = clnf._get_patch_experts(view_idx, patch_scale)

    print(f"\nPatch experts loaded: {len(patch_experts)} landmarks")

    # Test several landmarks
    test_landmarks = [30, 36, 48]  # nose, eye, mouth
    window_size = 11

    print(f"\n{'=' * 80}")
    print(f"Testing {len(test_landmarks)} landmarks with window_size={window_size}")
    print(f"{'=' * 80}\n")

    for lm_idx in test_landmarks:
        if lm_idx not in patch_experts:
            print(f"Skipping landmark {lm_idx} (no patch expert)")
            continue

        patch_expert = patch_experts[lm_idx]
        lm_x, lm_y = landmarks_2d[lm_idx]

        print(f"Landmark {lm_idx} at ({lm_x:.1f}, {lm_y:.1f}):")

        # ===================================================================
        # STEP 1: Compute RAW response map (sum of neuron responses)
        # ===================================================================
        half_window = window_size // 2
        raw_response_map = np.zeros((window_size, window_size), dtype=np.float32)

        for i in range(window_size):
            for j in range(window_size):
                # Calculate patch center
                search_x = lm_x + (j - half_window)
                search_y = lm_y + (i - half_window)

                # Extract patch
                patch_half_w = patch_expert.width // 2
                patch_half_h = patch_expert.height // 2

                x1 = int(search_x - patch_half_w)
                y1 = int(search_y - patch_half_h)
                x2 = int(search_x + patch_half_w + (patch_expert.width % 2))
                y2 = int(search_y + patch_half_h + (patch_expert.height % 2))

                if (x1 >= 0 and y1 >= 0 and
                    x2 <= gray.shape[1] and y2 <= gray.shape[0]):
                    patch = gray[y1:y2, x1:x2]

                    if patch.shape == (patch_expert.height, patch_expert.width):
                        raw_response_map[i, j] = patch_expert.compute_response(patch)

        # Print RAW response statistics
        print(f"  RAW response map (before sigma):")
        print(f"    Min:    {raw_response_map.min():.6f}")
        print(f"    Max:    {raw_response_map.max():.6f}")
        print(f"    Mean:   {raw_response_map.mean():.6f}")
        print(f"    Std:    {raw_response_map.std():.6f}")

        # ===================================================================
        # STEP 2: Apply sigma transformation
        # ===================================================================
        sigma_transformed = raw_response_map.copy()

        if clnf.ccnf.sigma_components and window_size in clnf.ccnf.sigma_components:
            sigma_comps = clnf.ccnf.sigma_components[window_size]
            Sigma = patch_expert.compute_sigma(sigma_comps, window_size=window_size)
            response_vec = sigma_transformed.reshape(-1, 1)
            response_transformed = Sigma @ response_vec
            sigma_transformed = response_transformed.reshape(sigma_transformed.shape)

            # Print sigma properties
            print(f"\n  Sigma matrix properties:")
            print(f"    Diagonal mean: {np.diag(Sigma).mean():.6f}")
            print(f"    Off-diagonal mean: {(Sigma.sum() - np.diag(Sigma).sum()) / (Sigma.size - len(Sigma)):.6f}")
            print(f"    Min value: {Sigma.min():.6f}")
            print(f"    Max value: {Sigma.max():.6f}")

        # Remove negative values (OpenFace step)
        min_val = sigma_transformed.min()
        if min_val < 0:
            sigma_transformed = sigma_transformed - min_val

        # Print SIGMA-TRANSFORMED response statistics
        print(f"\n  SIGMA-TRANSFORMED response map (after sigma, before normalization):")
        print(f"    Min:    {sigma_transformed.min():.6f}")
        print(f"    Max:    {sigma_transformed.max():.6f}")
        print(f"    Mean:   {sigma_transformed.mean():.6f}")
        print(f"    Std:    {sigma_transformed.std():.6f}")

        # ===================================================================
        # STEP 3: Calculate attenuation factor
        # ===================================================================
        raw_max = raw_response_map.max()
        sigma_max = sigma_transformed.max()

        if raw_max > 0:
            attenuation = raw_max / sigma_max
            print(f"\n  Attenuation by sigma transformation:")
            print(f"    Raw max: {raw_max:.6f}")
            print(f"    Sigma max: {sigma_max:.6f}")
            print(f"    Attenuation factor: {attenuation:.1f}x")

        # ===================================================================
        # STEP 4: Check neuron properties
        # ===================================================================
        print(f"\n  Patch expert properties:")
        print(f"    Number of neurons: {len(patch_expert.neurons)}")
        alphas = [n['alpha'] for n in patch_expert.neurons]
        print(f"    Neuron alphas - sum: {sum(alphas):.3f}, mean: {np.mean(alphas):.3f}, max: {max(alphas):.3f}")
        if hasattr(patch_expert, 'betas'):
            print(f"    Betas - sum: {sum(patch_expert.betas):.3f}, mean: {np.mean(patch_expert.betas):.3f}")

        print()

    # ===================================================================
    # Overall Analysis
    # ===================================================================
    print(f"{'=' * 80}")
    print(f"ANALYSIS")
    print(f"{'=' * 80}\n")

    print("Key observations:")
    print("1. If RAW responses are already tiny (<0.1), the problem is in neuron computation")
    print("2. If RAW responses are normal (>1.0) but sigma reduces them too much,")
    print("   the problem is in sigma transformation")
    print("3. Attenuation factor tells us how much sigma reduces response magnitudes")
    print()
    print("Expected OpenFace C++ behavior:")
    print("- Raw neuron responses should be in range ~1-10")
    print("- Sigma transformation should reduce by ~100-200x")
    print("- Final unnormalized responses should still be ~0.01-0.1 (large enough)")
    print()


if __name__ == "__main__":
    analyze_raw_vs_sigma()
