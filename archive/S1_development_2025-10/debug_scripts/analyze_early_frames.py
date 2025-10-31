#!/usr/bin/env python3
"""
Analyze early frame predictions to understand why frames 0-12 are zero in OpenFace.

Compare:
1. Running median values frame-by-frame
2. Raw predictions (before smoothing)
3. Feature vector statistics
"""

import numpy as np
import pandas as pd
from openface22_model_parser import OF22ModelParser
from openface22_hog_parser import OF22HOGParser
from pdm_parser import PDMParser
from histogram_median_tracker import DualHistogramMedianTracker

def extract_geometric_features(df_row, pdm_parser):
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df_row[pdm_cols].values
    return pdm_parser.extract_geometric_features(pdm_params)

def construct_full_feature_vector(hog_features, geom_features):
    return np.concatenate([hog_features, geom_features])

def main():
    print("="*80)
    print("EARLY FRAME ANALYSIS FOR AU15")
    print("="*80)

    # Load components
    models_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"
    pdm_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt"
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.hog"
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"

    parser = OF22ModelParser(models_dir)
    models = parser.load_all_models(use_recommended=True, use_combined=True)
    au15_model = models['AU15_r']

    pdm_parser = PDMParser(pdm_file)
    hog_parser = OF22HOGParser(hog_file)
    frame_indices, hog_features_all = hog_parser.parse()
    df = pd.read_csv(csv_file)

    # Initialize median tracker
    median_tracker = DualHistogramMedianTracker(
        hog_dim=4464, geom_dim=238,
        hog_bins=1000, hog_min=-0.005, hog_max=1.0,
        geom_bins=10000, geom_min=-60.0, geom_max=60.0
    )

    print(f"\nAnalyzing first 20 frames...")
    print(f"{'Frame':<6} {'Update?':<8} {'Median Mean':<14} {'Median Std':<14} {'Raw Pred':<12} {'OF2.2':<12} {'Error':<12}")
    print("-" * 95)

    for i in range(20):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        # Update median tracker
        update_histogram = (i % 2 == 1)
        median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)
        running_median = median_tracker.get_combined_median()

        # Construct features
        full_vector = construct_full_feature_vector(hog_feat, geom_feat)

        # Raw prediction (before smoothing, before clamping)
        centered = full_vector - au15_model['means'].flatten() - running_median
        pred_raw = np.dot(centered.reshape(1, -1), au15_model['support_vectors']) + au15_model['bias']
        pred_raw = float(pred_raw[0, 0])

        # Clamped prediction
        pred_clamped = np.clip(pred_raw, 0.0, 5.0)

        # OpenFace prediction
        of22_pred = df['AU15_r'].values[i]

        # Error
        error = pred_clamped - of22_pred

        print(f"{i:<6} {str(update_histogram):<8} {running_median.mean():<14.6f} {running_median.std():<14.6f} {pred_clamped:<12.4f} {of22_pred:<12.4f} {error:+12.4f}")

        # Special analysis for frame 0
        if i == 0:
            print(f"\n  Frame 0 details:")
            print(f"    Raw prediction (before clamp): {pred_raw:.6f}")
            print(f"    Clamped prediction: {pred_clamped:.6f}")
            print(f"    Running median min/max: [{running_median.min():.6f}, {running_median.max():.6f}]")
            print(f"    Feature vector mean: {full_vector.mean():.6f}")
            print(f"    Feature vector std: {full_vector.std():.6f}")
            print(f"    Model means mean: {au15_model['means'].mean():.6f}")
            print(f"    Centered features mean: {centered.mean():.6f}")
            print()

    # Check HOG median clamping
    print(f"\n" + "="*80)
    print("CHECKING HOG MEDIAN CLAMPING")
    print("="*80)
    hog_median = median_tracker.get_hog_median()
    geom_median = median_tracker.get_geom_median()
    print(f"HOG median min: {hog_median.min():.6f} (should be >= 0 after clamping)")
    print(f"HOG median max: {hog_median.max():.6f}")
    print(f"HOG median negative count: {(hog_median < 0).sum()} (should be 0)")
    print(f"Geom median min: {geom_median.min():.6f} (no clamping applied)")
    print(f"Geom median max: {geom_median.max():.6f}")

if __name__ == "__main__":
    main()
