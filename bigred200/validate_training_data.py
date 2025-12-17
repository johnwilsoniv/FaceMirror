#!/usr/bin/env python3
"""
Validate Training Data against C++ OpenFace Reference

Compares:
- Landmarks (68 points)
- AU intensities (17 values)
"""
import argparse
import numpy as np
import pandas as pd
import h5py
import os

AU_NAMES = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
            'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
            'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

LANDMARK_REGIONS = {
    'jaw': list(range(0, 17)),
    'brows': list(range(17, 27)),
    'nose': list(range(27, 36)),
    'eyes': list(range(36, 48)),
    'mouth': list(range(48, 68))
}


def main():
    parser = argparse.ArgumentParser(description='Validate training data against C++ reference')
    parser.add_argument('--training-h5', required=True, help='Training data HDF5 file')
    parser.add_argument('--cpp-csv', required=True, help='C++ reference CSV file')
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING DATA VALIDATION vs C++ OpenFace")
    print("=" * 70)

    # Load training data
    print(f"\nLoading training data: {args.training_h5}")
    with h5py.File(args.training_h5, 'r') as f:
        py_landmarks = f['landmarks'][:]
        py_aus = f['au_intensities'][:]
        frame_indices = f['frame_indices'][:]
        print(f"  Frames: {len(frame_indices)}")
        print(f"  Landmarks shape: {py_landmarks.shape}")
        print(f"  AU shape: {py_aus.shape}")

    # Load C++ reference
    print(f"\nLoading C++ reference: {args.cpp_csv}")
    cpp_df = pd.read_csv(args.cpp_csv)
    cpp_df.columns = cpp_df.columns.str.strip()
    print(f"  Frames: {len(cpp_df)}")

    # Extract C++ landmarks
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    cpp_landmarks = np.zeros((len(cpp_df), 68, 2), dtype=np.float32)
    cpp_landmarks[:, :, 0] = cpp_df[x_cols].values
    cpp_landmarks[:, :, 1] = cpp_df[y_cols].values

    # Extract C++ AUs
    cpp_aus = cpp_df[AU_NAMES].values

    # Align by frame indices
    min_frames = min(len(frame_indices), len(cpp_df))
    py_landmarks = py_landmarks[:min_frames]
    py_aus = py_aus[:min_frames]
    cpp_landmarks = cpp_landmarks[:min_frames]
    cpp_aus = cpp_aus[:min_frames]

    print(f"\nComparing {min_frames} frames")

    # Landmark comparison
    print("\n" + "-" * 60)
    print("LANDMARK ACCURACY (pixels)")
    print("-" * 60)

    lm_diff = py_landmarks - cpp_landmarks
    lm_error = np.sqrt(lm_diff[:, :, 0]**2 + lm_diff[:, :, 1]**2)

    print(f"Overall: {lm_error.mean():.2f} +/- {lm_error.std():.2f} (max: {lm_error.max():.2f})")

    for region, indices in LANDMARK_REGIONS.items():
        region_error = lm_error[:, indices]
        print(f"  {region.capitalize():8s}: {region_error.mean():.2f} +/- {region_error.std():.2f}")

    # AU comparison
    print("\n" + "-" * 60)
    print("AU ACCURACY")
    print("-" * 60)
    print(f"{'AU':<10} {'Corr':>10} {'MAE':>10} {'Py Mean':>10} {'C++ Mean':>10}")
    print("-" * 60)

    correlations = []
    maes = []

    for i, au in enumerate(AU_NAMES):
        py_vals = py_aus[:, i]
        cpp_vals = cpp_aus[:, i]

        # Correlation
        if py_vals.std() > 0.001 and cpp_vals.std() > 0.001:
            corr = np.corrcoef(py_vals, cpp_vals)[0, 1]
        else:
            corr = np.nan

        mae = np.abs(py_vals - cpp_vals).mean()

        if not np.isnan(corr):
            correlations.append(corr)
        maes.append(mae)

        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
        print(f"{au:<10} {corr_str:>10} {mae:>10.3f} {py_vals.mean():>10.3f} {cpp_vals.mean():>10.3f}")

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Landmark Error: {lm_error.mean():.2f} px (target: <2.0 px)")

    if correlations:
        mean_corr = np.mean(correlations)
        passing = sum(1 for c in correlations if c >= 0.95)
        print(f"AU Correlation: {mean_corr:.3f} (target: >=0.95)")
        print(f"AUs Passing (>=0.95): {passing}/{len(correlations)}")

    print(f"AU MAE: {np.mean(maes):.3f}")

    # Pass/Fail
    print("\n" + "=" * 70)
    lm_pass = lm_error.mean() < 2.0
    au_pass = mean_corr >= 0.95 if correlations else False

    if lm_pass and au_pass:
        print("RESULT: PASS ✓")
    else:
        print("RESULT: FAIL ✗")
        if not lm_pass:
            print(f"  - Landmark error {lm_error.mean():.2f} >= 2.0 px")
        if not au_pass:
            print(f"  - AU correlation {mean_corr:.3f} < 0.95")
    print("=" * 70)


if __name__ == '__main__':
    main()
