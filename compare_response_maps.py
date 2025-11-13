#!/usr/bin/env python3
"""
Compare response map quality between PyCLNF and OpenFace C++.

This diagnostic measures response map "sharpness" by analyzing:
1. Peak values (higher = stronger response)
2. Peak offsets from center (smaller = better aligned)
3. Response map spread/variance (lower = sharper peak)
4. Signal-to-noise ratio (peak vs background)
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pathlib import Path
import pandas as pd

def analyze_response_map_quality(response_map, window_size):
    """
    Analyze response map sharpness and quality.

    Returns dict with:
    - peak_value: Maximum response value
    - peak_offset_x, peak_offset_y: Offset from center
    - peak_dist: Distance from center
    - variance: Variance of response map (lower = sharper)
    - snr: Signal-to-noise ratio (peak / mean_background)
    """
    center = (window_size - 1) / 2.0
    peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
    peak_y, peak_x = peak_idx

    peak_value = response_map[peak_y, peak_x]
    offset_x = peak_x - center
    offset_y = peak_y - center
    peak_dist = np.sqrt(offset_x**2 + offset_y**2)

    # Compute variance (lower = sharper peak)
    variance = np.var(response_map)

    # Compute SNR: peak vs mean of non-peak region
    mask = np.ones_like(response_map, dtype=bool)
    mask[peak_y, peak_x] = False
    background_mean = response_map[mask].mean()
    snr = peak_value / (background_mean + 1e-10)

    return {
        'peak_value': peak_value,
        'peak_offset_x': offset_x,
        'peak_offset_y': offset_y,
        'peak_dist': peak_dist,
        'variance': variance,
        'snr': snr
    }


def main():
    print("=" * 80)
    print("RESPONSE MAP QUALITY COMPARISON: PyCLNF vs OpenFace C++")
    print("=" * 80)
    print()

    # Load test frame
    video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_bbox = (241, 555, 532, 532)

    # Initialize PyCLNF with debug mode
    print("Initializing PyCLNF with single iteration to capture response maps...")
    clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)

    # Get initial parameters
    params = clnf.pdm.init_params(face_bbox)
    landmarks_init = clnf.pdm.params_to_landmarks_2d(params)

    print(f"  Initial landmarks range: x=[{landmarks_init[:, 0].min():.1f}, {landmarks_init[:, 0].max():.1f}], "
          f"y=[{landmarks_init[:, 1].min():.1f}, {landmarks_init[:, 1].max():.1f}]")
    print()

    # Get patch experts for window size 11
    window_size = 11
    scale_idx = clnf.window_to_scale[window_size]
    patch_scale = clnf.patch_scaling[scale_idx]
    patch_experts = clnf._get_patch_experts(view_idx=0, scale=patch_scale)

    print(f"Analyzing response maps for window size {window_size}")
    print(f"  Patch experts loaded: {len(patch_experts)}")
    print()

    # Analyze response maps for all landmarks
    qualities = []

    for lm_idx in range(68):
        if lm_idx not in patch_experts:
            continue

        patch_expert = patch_experts[lm_idx]
        lm_x, lm_y = landmarks_init[lm_idx]

        # Compute response map (without sigma transformation for cleaner analysis)
        response_map = clnf.optimizer._compute_response_map(
            gray, lm_x, lm_y, patch_expert, window_size,
            sim_img_to_ref=None,
            sim_ref_to_img=None,
            sigma_components=None  # Disable sigma for cleaner diagnostic
        )

        quality = analyze_response_map_quality(response_map, window_size)
        quality['landmark'] = lm_idx
        qualities.append(quality)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(qualities)

    print("=" * 80)
    print("RESPONSE MAP QUALITY STATISTICS")
    print("=" * 80)
    print()

    print("Peak Values (higher = stronger response):")
    print(f"  Mean:   {df['peak_value'].mean():.6f}")
    print(f"  Median: {df['peak_value'].median():.6f}")
    print(f"  Std:    {df['peak_value'].std():.6f}")
    print(f"  Min:    {df['peak_value'].min():.6f}")
    print(f"  Max:    {df['peak_value'].max():.6f}")
    print()

    print("Peak Distances from Center (smaller = better aligned):")
    print(f"  Mean:   {df['peak_dist'].mean():.2f} pixels")
    print(f"  Median: {df['peak_dist'].median():.2f} pixels")
    print(f"  Std:    {df['peak_dist'].std():.2f} pixels")
    print(f"  Max:    {df['peak_dist'].max():.2f} pixels")
    print()

    print("Response Map Variance (lower = sharper peak):")
    print(f"  Mean:   {df['variance'].mean():.6f}")
    print(f"  Median: {df['variance'].median():.6f}")
    print()

    print("Signal-to-Noise Ratio (peak / background):")
    print(f"  Mean:   {df['snr'].mean():.2f}x")
    print(f"  Median: {df['snr'].median():.2f}x")
    print(f"  Min:    {df['snr'].min():.2f}x")
    print()

    # Find problematic landmarks (large offsets or low SNR)
    print("=" * 80)
    print("PROBLEMATIC LANDMARKS (large offsets or low quality)")
    print("=" * 80)
    print()

    # Worst peak offsets
    worst_offsets = df.nlargest(10, 'peak_dist')
    print("Top 10 worst peak offsets:")
    print(f"{'LM':<4} {'Offset_x':<10} {'Offset_y':<10} {'Distance':<10} {'Peak':<10} {'SNR':<8}")
    print("-" * 60)
    for _, row in worst_offsets.iterrows():
        print(f"{int(row['landmark']):<4} {row['peak_offset_x']:>9.2f} {row['peak_offset_y']:>9.2f} "
              f"{row['peak_dist']:>9.2f} {row['peak_value']:>9.4f} {row['snr']:>7.2f}x")
    print()

    # Lowest SNR
    worst_snr = df.nsmallest(10, 'snr')
    print("Top 10 lowest SNR (weak response maps):")
    print(f"{'LM':<4} {'SNR':<10} {'Peak':<10} {'Variance':<12} {'Distance':<10}")
    print("-" * 50)
    for _, row in worst_snr.iterrows():
        print(f"{int(row['landmark']):<4} {row['snr']:>9.2f}x {row['peak_value']:>9.4f} "
              f"{row['variance']:>11.6f} {row['peak_dist']:>9.2f}")
    print()

    # Summary
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    large_offset_count = (df['peak_dist'] > 2.0).sum()
    low_snr_count = (df['snr'] < 5.0).sum()

    print(f"Landmarks with large peak offsets (>2px): {large_offset_count}/68 ({large_offset_count/68*100:.1f}%)")
    print(f"Landmarks with low SNR (<5x): {low_snr_count}/68 ({low_snr_count/68*100:.1f}%)")
    print()

    if large_offset_count > 20:
        print("⚠️  HIGH OFFSET RATE: Many response map peaks are misaligned.")
        print("   This suggests initialization is poor or patch experts don't match well.")
    elif large_offset_count > 10:
        print("⚠️  MODERATE OFFSET RATE: Some response maps are misaligned.")
        print("   May indicate specific landmarks with poor patch expert quality.")
    else:
        print("✓ Low offset rate: Most response maps are well-aligned.")
    print()

    if low_snr_count > 20:
        print("⚠️  HIGH LOW-SNR RATE: Many response maps have weak peaks.")
        print("   This suggests patch experts are not discriminative enough.")
    elif low_snr_count > 10:
        print("⚠️  MODERATE LOW-SNR RATE: Some response maps are weak.")
        print("   May indicate specific landmarks with poor patch expert training.")
    else:
        print("✓ Low low-SNR rate: Most response maps have strong peaks.")
    print()

    print("NEXT STEPS:")
    print("1. Compare these metrics against OpenFace C++ response maps")
    print("2. Check if response map quality degrades over iterations")
    print("3. Test if increasing patch scaling improves response map quality")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
