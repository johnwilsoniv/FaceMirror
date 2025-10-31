#!/usr/bin/env python3
"""
Analyze centered features (features - means) to determine
appropriate histogram parameters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from openface22_hog_parser import OF22HOGParser
from openface22_model_parser import OF22ModelParser


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
    models_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.hog"
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"

    print("="*80)
    print("Centered Feature Range Analysis")
    print("="*80)

    # Load a model to get the means
    print("\nLoading SVR model to get means...")
    parser = OF22ModelParser(models_dir)
    models = parser.load_all_models(use_recommended=True, use_combined=True)

    # Use AU12 as reference (it's working perfectly)
    model = models['AU12_r']
    means = model['means'].flatten()
    print(f"✓ Loaded means from {model['au_name']}: shape {means.shape}")

    # Parse HOG features
    print("\nLoading HOG features...")
    hog_parser = OF22HOGParser(hog_file)
    frame_indices, hog_features_all = hog_parser.parse()

    # Load CSV
    print("Loading geometric features from CSV...")
    df = pd.read_csv(csv_file)

    # Extract all features and center them
    centered_features_all = []
    for i in range(min(len(frame_indices), len(df))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i])

        # Concatenate to full feature vector
        full_feat = np.concatenate([hog_feat, geom_feat])

        # Center by subtracting means
        centered = full_feat - means

        centered_features_all.append(centered)

    centered_features_all = np.array(centered_features_all)
    print(f"✓ Centered {len(centered_features_all)} feature vectors")

    # Analyze centered features
    print("\n" + "="*80)
    print("CENTERED Features (after subtracting means)")
    print("="*80)
    print(f"Min value:  {centered_features_all.min():.6f}")
    print(f"Max value:  {centered_features_all.max():.6f}")
    print(f"Mean:       {centered_features_all.mean():.6f}")
    print(f"Median:     {np.median(centered_features_all):.6f}")
    print(f"Std dev:    {centered_features_all.std():.6f}")
    print(f"1st percentile:  {np.percentile(centered_features_all, 1):.6f}")
    print(f"99th percentile: {np.percentile(centered_features_all, 99):.6f}")

    # Check if values are in [-3, 5] range
    in_range = np.sum((centered_features_all >= -3.0) & (centered_features_all <= 5.0))
    total = centered_features_all.size
    pct_in_range = 100.0 * in_range / total

    print("\n" + "="*80)
    print("Histogram Range Check ([-3, 5])")
    print("="*80)
    print(f"Values in range: {in_range:,} / {total:,} ({pct_in_range:.2f}%)")
    print(f"Values below -3: {np.sum(centered_features_all < -3.0):,} ({100.0 * np.sum(centered_features_all < -3.0) / total:.2f}%)")
    print(f"Values above 5:  {np.sum(centered_features_all > 5.0):,} ({100.0 * np.sum(centered_features_all > 5.0) / total:.2f}%)")

    # Suggest better range
    print("\n" + "="*80)
    print("Suggested Histogram Parameters")
    print("="*80)
    min_val = centered_features_all.min()
    max_val = centered_features_all.max()
    print(f"Based on actual data:")
    print(f"  min_val: {min_val:.2f}")
    print(f"  max_val: {max_val:.2f}")

    # Use 1st to 99th percentile for more robust range
    p1 = np.percentile(centered_features_all, 1)
    p99 = np.percentile(centered_features_all, 99)
    print(f"\nBased on 1st-99th percentile:")
    print(f"  min_val: {p1:.2f}")
    print(f"  max_val: {p99:.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
