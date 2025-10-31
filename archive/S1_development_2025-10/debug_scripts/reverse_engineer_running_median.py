#!/usr/bin/env python3
"""
Reverse-engineer OpenFace's running median by solving the prediction equation.

For dynamic models:
  prediction = (features - means - running_median) · SV + bias

Solving for running_median:
  running_median = features - means - (prediction - bias) / SV

This only works for individual features that have large SV values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from openface22_model_parser import OF22ModelParser
from openface22_hog_parser import OF22HOGParser
from pdm_parser import PDMParser
from histogram_median_tracker import DualHistogramMedianTracker
import matplotlib.pyplot as plt

def extract_geometric_features(df_row, pdm_parser):
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df_row[pdm_cols].values
    return pdm_parser.extract_geometric_features(pdm_params)

def construct_full_feature_vector(hog_features, geom_features):
    return np.concatenate([hog_features, geom_features])

def main():
    print("="*80)
    print("REVERSE ENGINEERING OPENFACE RUNNING MEDIAN")
    print("="*80)

    # Paths
    models_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"
    pdm_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt"
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.hog"
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"

    # Load components
    print("\nLoading components...")
    parser = OF22ModelParser(models_dir)
    models = parser.load_all_models(use_recommended=True, use_combined=True)
    au15_model = models['AU15_r']

    pdm_parser = PDMParser(pdm_file)
    hog_parser = OF22HOGParser(hog_file)
    frame_indices, hog_features_all = hog_parser.parse()
    df = pd.read_csv(csv_file)

    # Initialize our tracker
    median_tracker = DualHistogramMedianTracker(
        hog_dim=4464, geom_dim=238,
        hog_bins=1000, hog_min=-0.005, hog_max=1.0,
        geom_bins=10000, geom_min=-60.0, geom_max=60.0
    )

    our_medians = []

    # Process first 10 frames to see initialization
    print(f"\nComparing first 10 frames...")
    print(f"{'Frame':<6} {'Our Median Mean':<18} {'Our Median Std':<18}")
    print("-" * 50)

    for i in range(min(10, len(hog_features_all))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        update_histogram = (i % 2 == 1)
        median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)
        our_median = median_tracker.get_combined_median()
        our_medians.append(our_median.copy())

        print(f"{i:<6} {our_median.mean():<18.6f} {our_median.std():<18.6f}")

    # Now let's check if the first few medians make sense
    # by seeing if they lead to reasonable predictions
    print(f"\n" + "="*80)
    print("PREDICTION CHECK FOR FIRST 10 FRAMES")
    print("="*80)
    print(f"{'Frame':<6} {'Python Pred':<15} {'OF2.2 Pred':<15} {'Error':<15}")
    print("-" * 60)

    for i in range(10):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)
        full_vector = construct_full_feature_vector(hog_feat, geom_feat)

        our_median = our_medians[i]

        # Our prediction
        centered = full_vector - au15_model['means'].flatten() - our_median
        pred = np.dot(centered.reshape(1, -1), au15_model['support_vectors']) + au15_model['bias']
        pred = np.clip(float(pred[0, 0]), 0.0, 5.0)

        # OF2.2 prediction (before smoothing - we'd need to undo smoothing)
        of22_pred = df['AU15_r'].values[i]

        error = pred - of22_pred
        print(f"{i:<6} {pred:<15.4f} {of22_pred:<15.4f} {error:+15.4f}")

if __name__ == "__main__":
    main()
