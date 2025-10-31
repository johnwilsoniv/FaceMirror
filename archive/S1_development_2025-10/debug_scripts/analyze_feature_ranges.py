#!/usr/bin/env python3
"""
Analyze the actual value ranges of HOG and geometric features
to determine appropriate histogram parameters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from openface22_hog_parser import OF22HOGParser


def extract_geometric_features(df_row):
    """Extract 238-dimensional geometric features from CSV row"""
    # Extract 3D landmarks (X, Y, Z for 68 points)
    X_cols = [f'X_{i}' for i in range(68)]
    Y_cols = [f'Y_{i}' for i in range(68)]
    Z_cols = [f'Z_{i}' for i in range(68)]

    landmarks_3d = np.concatenate([
        df_row[X_cols].values,
        df_row[Y_cols].values,
        df_row[Z_cols].values
    ])  # 204 dims

    # Extract PDM shape parameters
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df_row[pdm_cols].values  # 34 dims

    return np.concatenate([landmarks_3d, pdm_params])


def main():
    # Paths
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.hog"
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"

    print("="*80)
    print("Feature Range Analysis")
    print("="*80)

    # Parse HOG features
    print("\nLoading HOG features...")
    hog_parser = OF22HOGParser(hog_file)
    frame_indices, hog_features_all = hog_parser.parse()
    print(f"✓ Loaded {len(frame_indices)} frames with {hog_features_all.shape[1]} HOG features")

    # Load CSV
    print("\nLoading geometric features from CSV...")
    df = pd.read_csv(csv_file)

    # Extract all geometric features
    geom_features_all = []
    for i in range(len(df)):
        geom_feat = extract_geometric_features(df.iloc[i])
        geom_features_all.append(geom_feat)
    geom_features_all = np.array(geom_features_all)
    print(f"✓ Loaded {len(geom_features_all)} geometric feature vectors")

    # Analyze HOG features
    print("\n" + "="*80)
    print("HOG Features (4464 dims)")
    print("="*80)
    print(f"Min value:  {hog_features_all.min():.6f}")
    print(f"Max value:  {hog_features_all.max():.6f}")
    print(f"Mean:       {hog_features_all.mean():.6f}")
    print(f"Median:     {np.median(hog_features_all):.6f}")
    print(f"Std dev:    {hog_features_all.std():.6f}")
    print(f"1st percentile:  {np.percentile(hog_features_all, 1):.6f}")
    print(f"99th percentile: {np.percentile(hog_features_all, 99):.6f}")

    # Analyze geometric features
    print("\n" + "="*80)
    print("Geometric Features (238 dims)")
    print("="*80)
    print(f"Min value:  {geom_features_all.min():.6f}")
    print(f"Max value:  {geom_features_all.max():.6f}")
    print(f"Mean:       {geom_features_all.mean():.6f}")
    print(f"Median:     {np.median(geom_features_all):.6f}")
    print(f"Std dev:    {geom_features_all.std():.6f}")
    print(f"1st percentile:  {np.percentile(geom_features_all, 1):.6f}")
    print(f"99th percentile: {np.percentile(geom_features_all, 99):.6f}")

    # Analyze combined features
    combined_features = np.concatenate([hog_features_all, geom_features_all], axis=1)
    print("\n" + "="*80)
    print("Combined Features (4702 dims)")
    print("="*80)
    print(f"Min value:  {combined_features.min():.6f}")
    print(f"Max value:  {combined_features.max():.6f}")
    print(f"Mean:       {combined_features.mean():.6f}")
    print(f"Median:     {np.median(combined_features):.6f}")
    print(f"Std dev:    {combined_features.std():.6f}")
    print(f"1st percentile:  {np.percentile(combined_features, 1):.6f}")
    print(f"99th percentile: {np.percentile(combined_features, 99):.6f}")

    # Check if values are in [-3, 5] range
    in_range = np.sum((combined_features >= -3.0) & (combined_features <= 5.0))
    total = combined_features.size
    pct_in_range = 100.0 * in_range / total

    print("\n" + "="*80)
    print("Histogram Range Check ([-3, 5])")
    print("="*80)
    print(f"Values in range: {in_range:,} / {total:,} ({pct_in_range:.2f}%)")
    print(f"Values below -3: {np.sum(combined_features < -3.0):,} ({100.0 * np.sum(combined_features < -3.0) / total:.2f}%)")
    print(f"Values above 5:  {np.sum(combined_features > 5.0):,} ({100.0 * np.sum(combined_features > 5.0) / total:.2f}%)")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
