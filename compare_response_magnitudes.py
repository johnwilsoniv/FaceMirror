#!/usr/bin/env python3
"""
Compare response map magnitudes between PyCLNF and expected OpenFace C++ values.

Hypothesis: PyCLNF's unnormalized response maps have different magnitude ranges
than OpenFace C++, causing mean-shift to behave differently.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "pyclnf"))

from pyclnf import CLNF
from pyclnf.core.patch_expert import CCNFModel


def analyze_response_magnitudes():
    """Compare response magnitudes with and without normalization."""

    print("=" * 80)
    print("Response Magnitude Analysis: Why Does PyCLNF Need Normalization?")
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

    # Test several landmarks and collect response statistics
    test_landmarks = [30, 36, 48]  # nose, eye, mouth
    window_size = 11

    print(f"\n{'=' * 80}")
    print(f"Testing {len(test_landmarks)} landmarks with window_size={window_size}")
    print(f"{'=' * 80}\n")

    all_responses_normalized = []
    all_responses_unnormalized = []

    for lm_idx in test_landmarks:
        if lm_idx not in patch_experts:
            print(f"Skipping landmark {lm_idx} (no patch expert)")
            continue

        patch_expert = patch_experts[lm_idx]
        lm_x, lm_y = landmarks_2d[lm_idx]

        print(f"Landmark {lm_idx} at ({lm_x:.1f}, {lm_y:.1f}):")

        # Compute response map WITHOUT normalization
        # (Modified version of optimizer._compute_response_map)
        half_window = window_size // 2
        response_map = np.zeros((window_size, window_size), dtype=np.float32)

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
                        response_map[i, j] = patch_expert.compute_response(patch)

        # Apply sigma transformation if available
        if clnf.ccnf.sigma_components and window_size in clnf.ccnf.sigma_components:
            sigma_comps = clnf.ccnf.sigma_components[window_size]
            Sigma = patch_expert.compute_sigma(sigma_comps, window_size=window_size)
            response_vec = response_map.reshape(-1, 1)
            response_transformed = Sigma @ response_vec
            response_map = response_transformed.reshape(response_map.shape)

        # Remove negative values (OpenFace step)
        min_val = response_map.min()
        if min_val < 0:
            response_map = response_map - min_val

        # Store unnormalized version
        response_unnormalized = response_map.copy()
        all_responses_unnormalized.extend(response_unnormalized.flatten())

        # Normalize to [0, 1] (PyCLNF current approach)
        max_val = response_map.max()
        if max_val > 0:
            response_normalized = response_map / max_val
        else:
            response_normalized = response_map

        all_responses_normalized.extend(response_normalized.flatten())

        # Report statistics
        print(f"  Unnormalized response map:")
        print(f"    Min:    {response_unnormalized.min():.6f}")
        print(f"    Max:    {response_unnormalized.max():.6f}")
        print(f"    Mean:   {response_unnormalized.mean():.6f}")
        print(f"    Std:    {response_unnormalized.std():.6f}")
        print(f"    Median: {np.median(response_unnormalized):.6f}")

        print(f"  Normalized response map:")
        print(f"    Min:    {response_normalized.min():.6f}")
        print(f"    Max:    {response_normalized.max():.6f}")
        print(f"    Mean:   {response_normalized.mean():.6f}")
        print(f"    Std:    {response_normalized.std():.6f}")

        # Check peak sharpness
        flat_unnorm = response_unnormalized.flatten()
        flat_norm = response_normalized.flatten()

        peak_val_unnorm = flat_unnorm.max()
        second_highest_unnorm = np.partition(flat_unnorm, -2)[-2]
        sharpness_unnorm = peak_val_unnorm / (second_highest_unnorm + 1e-10)

        peak_val_norm = flat_norm.max()
        second_highest_norm = np.partition(flat_norm, -2)[-2]
        sharpness_norm = peak_val_norm / (second_highest_norm + 1e-10)

        print(f"  Peak sharpness (max/second_highest):")
        print(f"    Unnormalized: {sharpness_unnorm:.3f}")
        print(f"    Normalized:   {sharpness_norm:.3f}")
        print()

    # Overall statistics
    print(f"{'=' * 80}")
    print(f"Overall Statistics Across All Landmarks")
    print(f"{'=' * 80}\n")

    all_unnorm = np.array(all_responses_unnormalized)
    all_norm = np.array(all_responses_normalized)

    print(f"Unnormalized response values:")
    print(f"  Range:      [{all_unnorm.min():.6f}, {all_unnorm.max():.6f}]")
    print(f"  Mean:       {all_unnorm.mean():.6f}")
    print(f"  Std:        {all_unnorm.std():.6f}")
    print(f"  Percentiles:")
    print(f"    25th:     {np.percentile(all_unnorm, 25):.6f}")
    print(f"    50th:     {np.percentile(all_unnorm, 50):.6f}")
    print(f"    75th:     {np.percentile(all_unnorm, 75):.6f}")
    print(f"    95th:     {np.percentile(all_unnorm, 95):.6f}")
    print(f"    99th:     {np.percentile(all_unnorm, 99):.6f}")

    print(f"\nNormalized response values:")
    print(f"  Range:      [{all_norm.min():.6f}, {all_norm.max():.6f}]")
    print(f"  Mean:       {all_norm.mean():.6f}")
    print(f"  Std:        {all_norm.std():.6f}")

    # Key insight: check the absolute magnitude scale
    print(f"\n{'=' * 80}")
    print(f"KEY INSIGHT: Magnitude Scale")
    print(f"{'=' * 80}\n")

    print(f"Unnormalized response magnitudes after sigma transformation:")
    print(f"  Typical values: ~{all_unnorm.mean():.4f}")
    print(f"  Max across all landmarks: {all_unnorm.max():.4f}")
    print()

    # Check if these magnitudes are reasonable for KDE mean-shift
    print(f"Analysis:")
    if all_unnorm.max() < 0.1:
        print(f"  ⚠️  Response magnitudes are VERY SMALL (max={all_unnorm.max():.4f})")
        print(f"      This could cause numerical issues in mean-shift computation")
        print(f"      OpenFace C++ might handle small numbers differently (e.g., different precision)")
    elif all_unnorm.max() > 10.0:
        print(f"  ⚠️  Response magnitudes are VERY LARGE (max={all_unnorm.max():.4f})")
        print(f"      This could cause overflow/instability in mean-shift computation")
    else:
        print(f"  ✓  Response magnitudes appear reasonable ({all_unnorm.max():.4f})")

    print()

    # Compare variance
    print(f"Variance analysis:")
    print(f"  Unnormalized std/mean ratio: {all_unnorm.std() / (all_unnorm.mean() + 1e-10):.3f}")
    print(f"  Normalized std/mean ratio:   {all_norm.std() / (all_norm.mean() + 1e-10):.3f}")
    print()

    if all_unnorm.std() / (all_unnorm.mean() + 1e-10) < 0.5:
        print(f"  ⚠️  Unnormalized responses have LOW variance relative to mean")
        print(f"      This suggests response maps might be too flat (poor discrimination)")

    # Check expected OpenFace C++ behavior
    print(f"\n{'=' * 80}")
    print(f"Expected OpenFace C++ Behavior")
    print(f"{'=' * 80}\n")

    print(f"OpenFace C++ uses unnormalized response values directly in mean-shift.")
    print(f"For this to work properly, response magnitudes should be:")
    print(f"  1. Large enough to avoid precision loss (~0.01 to 10.0)")
    print(f"  2. Have sufficient dynamic range (peaks >> background)")
    print(f"  3. Consistent scale across different landmarks")
    print()

    print(f"Current PyCLNF unnormalized responses:")
    print(f"  1. Magnitude range: {all_unnorm.min():.4f} to {all_unnorm.max():.4f}")

    if all_unnorm.max() < 0.05:
        print(f"     → TOO SMALL! Likely causing precision/numerical issues")
        print(f"     → This explains why normalization helps (rescales to [0,1])")
    else:
        print(f"     → Appears reasonable")

    print(f"  2. Dynamic range (95th/50th percentile): {np.percentile(all_unnorm, 95) / (np.percentile(all_unnorm, 50) + 1e-10):.2f}x")

    if np.percentile(all_unnorm, 95) / (np.percentile(all_unnorm, 50) + 1e-10) < 2.0:
        print(f"     → LOW! Peaks are not distinctive enough")
    else:
        print(f"     → Appears reasonable")

    print()

    # Hypothesis
    print(f"{'=' * 80}")
    print(f"HYPOTHESIS")
    print(f"{'=' * 80}\n")

    if all_unnorm.max() < 0.05:
        print(f"PyCLNF's unnormalized response magnitudes are TOO SMALL (~{all_unnorm.max():.4f}).")
        print(f"This suggests one of:")
        print(f"  1. Sigma transformation is over-attenuating (but we verified the formula)")
        print(f"  2. Raw neuron responses are too weak (need to check neuron alphas/biases)")
        print(f"  3. Missing a scaling factor somewhere in the pipeline")
        print(f"  4. OpenFace C++ has different sigma components than what we exported")
        print()
        print(f"OpenFace C++ likely produces larger unnormalized response values,")
        print(f"making normalization unnecessary. We need to find why PyCLNF's are smaller.")
    else:
        print(f"Response magnitudes appear reasonable. The issue may be elsewhere.")
        print(f"Need to investigate:")
        print(f"  - Mean-shift computation (KDE weighting)")
        print(f"  - Jacobian calculation")
        print(f"  - Parameter update logic")

    print()


if __name__ == "__main__":
    analyze_response_magnitudes()
