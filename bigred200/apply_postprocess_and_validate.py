#!/usr/bin/env python3
"""
Apply post-processing to existing training data AUs and validate against C++ reference.
"""
import argparse
import numpy as np
import pandas as pd
import h5py
import os
import sys

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
    parser = argparse.ArgumentParser(description='Apply post-processing and validate')
    parser.add_argument('--training-h5', required=True, help='Training data HDF5 file')
    parser.add_argument('--cpp-csv', required=True, help='C++ reference CSV file')
    args = parser.parse_args()

    print("=" * 70)
    print("APPLY POST-PROCESSING AND VALIDATE")
    print("=" * 70)

    # Load training data (raw AUs)
    print(f"\nLoading training data: {args.training_h5}")
    with h5py.File(args.training_h5, 'r') as f:
        py_landmarks = f['landmarks'][:]
        raw_aus = f['au_intensities'][:]
        frame_indices = f['frame_indices'][:]
        print(f"  Frames: {len(frame_indices)}")
        print(f"  Landmarks shape: {py_landmarks.shape}")
        print(f"  Raw AU shape: {raw_aus.shape}")

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

    # Apply post-processing to raw AUs
    print("\nApplying post-processing to raw AUs...")
    sys.stdout.flush()

    # Dynamic AUs and their cutoff values (from OpenFace SVR models)
    # These are the percentile thresholds for baseline correction
    DYNAMIC_AU_CUTOFFS = {
        'AU01_r': 0.60,
        'AU02_r': 0.75,
        'AU04_r': 0.55,
        'AU05_r': 0.55,
        'AU06_r': 0.60,
        'AU07_r': 0.65,
        'AU09_r': 0.60,
        'AU10_r': 0.55,
        'AU12_r': 0.60,
        'AU14_r': 0.50,
        'AU15_r': 0.55,
        # 'AU17_r': 0.20,  # Skip AU17 - unusual weight distribution
        'AU20_r': 0.50,
        'AU23_r': 0.55,
        'AU25_r': 0.50,
        'AU26_r': 0.50,
        'AU45_r': 0.50,
    }

    # Create DataFrame with raw AU values
    au_df_data = {'frame': list(frame_indices), 'success': [True] * len(frame_indices)}
    for i, au_name in enumerate(AU_NAMES):
        au_df_data[au_name] = raw_aus[:, i].tolist()

    processed_df = pd.DataFrame(au_df_data)

    # Step 1: Apply cutoff adjustment for dynamic AUs
    print("  [1/2] Applying cutoff adjustment...")
    for au_col in AU_NAMES:
        if au_col in DYNAMIC_AU_CUTOFFS:
            model_cutoff = DYNAMIC_AU_CUTOFFS[au_col]
            au_values = processed_df[au_col].values.copy()

            # Match C++ - only use valid (non-zero) predictions for percentile
            valid_mask = au_values > 0.001
            valid_vals = au_values[valid_mask]

            if len(valid_vals) >= 10:
                sorted_vals = np.sort(valid_vals)
                cutoff_idx = int(len(sorted_vals) * model_cutoff)
                offset = sorted_vals[cutoff_idx]
                processed_df[au_col] = np.clip(au_values - offset, 0.0, 5.0)

    # Step 2: Apply temporal smoothing (3-frame moving average)
    print("  [2/2] Applying temporal smoothing...")
    for au_col in AU_NAMES:
        au_values = processed_df[au_col].values
        smoothed = np.convolve(au_values, np.ones(3)/3, mode='same')
        # Fix edges
        smoothed[0] = au_values[0]
        smoothed[-1] = au_values[-1]
        processed_df[au_col] = smoothed

    # Extract post-processed AU values
    py_aus = np.array([[processed_df.iloc[idx][au] for au in AU_NAMES]
                       for idx in range(len(processed_df))], dtype=np.float32)

    print(f"  Post-processed AU shape: {py_aus.shape}")

    # Align by frame indices
    min_frames = min(len(frame_indices), len(cpp_df))
    py_landmarks = py_landmarks[:min_frames]
    py_aus = py_aus[:min_frames]
    raw_aus = raw_aus[:min_frames]
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

    # AU comparison - RAW vs C++
    print("\n" + "-" * 60)
    print("AU ACCURACY - RAW (before post-processing)")
    print("-" * 60)
    print(f"{'AU':<10} {'Corr':>10} {'MAE':>10}")
    print("-" * 40)

    raw_correlations = []
    for i, au in enumerate(AU_NAMES):
        py_vals = raw_aus[:, i]
        cpp_vals = cpp_aus[:, i]
        if py_vals.std() > 0.001 and cpp_vals.std() > 0.001:
            corr = np.corrcoef(py_vals, cpp_vals)[0, 1]
            raw_correlations.append(corr)
        else:
            corr = np.nan
        mae = np.abs(py_vals - cpp_vals).mean()
        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
        print(f"{au:<10} {corr_str:>10} {mae:>10.3f}")

    if raw_correlations:
        print(f"\nRaw Mean Correlation: {np.mean(raw_correlations):.3f}")

    # AU comparison - POST-PROCESSED vs C++
    print("\n" + "-" * 60)
    print("AU ACCURACY - POST-PROCESSED")
    print("-" * 60)
    print(f"{'AU':<10} {'Corr':>10} {'MAE':>10} {'Py Mean':>10} {'C++ Mean':>10}")
    print("-" * 60)

    correlations = []
    maes = []

    for i, au in enumerate(AU_NAMES):
        py_vals = py_aus[:, i]
        cpp_vals = cpp_aus[:, i]

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
        print(f"AU Correlation (post-processed): {mean_corr:.3f} (target: >=0.95)")
        print(f"AUs Passing (>=0.95): {passing}/{len(correlations)}")

    print(f"AU MAE: {np.mean(maes):.3f}")

    # Pass/Fail
    print("\n" + "=" * 70)
    lm_pass = lm_error.mean() < 2.0
    au_pass = mean_corr >= 0.95 if correlations else False

    if lm_pass and au_pass:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        if not lm_pass:
            print(f"  - Landmark error {lm_error.mean():.2f} >= 2.0 px")
        if not au_pass:
            print(f"  - AU correlation {mean_corr:.3f} < 0.95")
    print("=" * 70)


if __name__ == '__main__':
    main()
