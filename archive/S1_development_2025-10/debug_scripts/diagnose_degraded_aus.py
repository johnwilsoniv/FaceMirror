#!/usr/bin/env python3
"""
Diagnose AU20/AU23 Degradation with Two-Pass Processing

Compares single-pass vs two-pass predictions to understand why
AU20 and AU23 correlation DECREASED after implementing two-pass processing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from openface22_model_parser import OF22ModelParser
from openface22_hog_parser import OF22HOGParser
from histogram_median_tracker import DualHistogramMedianTracker
from pdm_parser import PDMParser
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def extract_geometric_features(df_row, pdm_parser):
    """Extract 238-dimensional geometric features from CSV row"""
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df_row[pdm_cols].values
    geom_features = pdm_parser.extract_geometric_features(pdm_params)
    return geom_features


def construct_full_feature_vector(hog_features, geom_features):
    """Construct complete 4702-dimensional feature vector"""
    return np.concatenate([hog_features, geom_features])


def predict_single_pass(models, hog_features_all, df, pdm_parser, au_names):
    """
    Single-pass prediction (original implementation)
    Uses running median as it evolves frame-by-frame
    """
    print("=" * 80)
    print("SINGLE-PASS PREDICTION (Original)")
    print("=" * 80)

    median_tracker = DualHistogramMedianTracker(
        hog_dim=4464,
        geom_dim=238,
        hog_bins=1000,
        hog_min=-0.005,
        hog_max=1.0,
        geom_bins=10000,
        geom_min=-60.0,
        geom_max=60.0
    )

    all_predictions = {au: [] for au in au_names}

    for i in range(min(len(hog_features_all), len(df))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        # Update running median
        update_histogram = (i % 2 == 1)
        median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)
        running_median = median_tracker.get_combined_median()

        # Predict for each AU
        full_vector = construct_full_feature_vector(hog_feat, geom_feat)

        for au_name in au_names:
            model = models[au_name]
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic:
                centered = full_vector - model['means'].flatten() - running_median
                pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
                pred = float(pred[0, 0])
            else:
                centered = full_vector - model['means'].flatten()
                pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
                pred = float(pred[0, 0])

            pred = np.clip(pred, 0.0, 5.0)
            all_predictions[au_name].append(pred)

    # Apply temporal smoothing
    for au_name in au_names:
        preds = np.array(all_predictions[au_name])
        smoothed = preds.copy()

        for i in range(1, len(preds) - 1):
            smoothed[i] = (preds[i-1] + preds[i] + preds[i+1]) / 3.0

        all_predictions[au_name] = smoothed

    return all_predictions


def predict_two_pass(models, hog_features_all, df, pdm_parser, au_names):
    """
    Two-pass prediction (new implementation)
    Pass 1: Build running median from all frames
    Pass 2: Reprocess first 3000 frames with final median
    """
    print("=" * 80)
    print("TWO-PASS PREDICTION (With Postprocessing)")
    print("=" * 80)

    median_tracker = DualHistogramMedianTracker(
        hog_dim=4464,
        geom_dim=238,
        hog_bins=1000,
        hog_min=-0.005,
        hog_max=1.0,
        geom_bins=10000,
        geom_min=-60.0,
        geom_max=60.0
    )

    # PASS 1: Build running median
    print("\nPass 1: Building running median...")
    running_medians_per_frame = []
    stored_features = []
    max_init_frames = min(3000, min(len(hog_features_all), len(df)))

    for i in range(min(len(hog_features_all), len(df))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        update_histogram = (i % 2 == 1)
        median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)

        running_medians_per_frame.append(median_tracker.get_combined_median().copy())

        if i < max_init_frames:
            stored_features.append((hog_feat.copy(), geom_feat.copy()))

    # PASS 2: Reprocess early frames with final median
    print(f"Pass 2: Reprocessing first {len(stored_features)} frames with final median...")
    final_median = median_tracker.get_combined_median()

    for i in range(len(stored_features)):
        running_medians_per_frame[i] = final_median.copy()

    # Now predict using stored running medians
    print("\nPredicting with two-pass running medians...")
    all_predictions = {au: [] for au in au_names}

    for i in range(min(len(hog_features_all), len(df))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        running_median = running_medians_per_frame[i]
        full_vector = construct_full_feature_vector(hog_feat, geom_feat)

        for au_name in au_names:
            model = models[au_name]
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic:
                centered = full_vector - model['means'].flatten() - running_median
                pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
                pred = float(pred[0, 0])
            else:
                centered = full_vector - model['means'].flatten()
                pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
                pred = float(pred[0, 0])

            pred = np.clip(pred, 0.0, 5.0)
            all_predictions[au_name].append(pred)

    # Apply temporal smoothing
    for au_name in au_names:
        preds = np.array(all_predictions[au_name])
        smoothed = preds.copy()

        for i in range(1, len(preds) - 1):
            smoothed[i] = (preds[i-1] + preds[i] + preds[i+1]) / 3.0

        all_predictions[au_name] = smoothed

    return all_predictions


def analyze_differences(single_pass, two_pass, of22_preds, au_name, output_dir):
    """Analyze where and why predictions differ"""

    single = np.array(single_pass[au_name])
    two = np.array(two_pass[au_name])
    of22 = of22_preds[au_name]

    # Calculate correlations
    r_single, _ = pearsonr(single, of22)
    r_two, _ = pearsonr(two, of22)

    # Find frames with largest differences
    diff = np.abs(two - single)
    top_diff_frames = np.argsort(diff)[-20:][::-1]

    print(f"\n{'='*80}")
    print(f"{au_name} - Detailed Analysis")
    print(f"{'='*80}")
    print(f"Single-pass correlation: {r_single:.6f}")
    print(f"Two-pass correlation:    {r_two:.6f}")
    print(f"Change:                  {r_two - r_single:+.6f}")

    if r_two < r_single:
        print(f"⚠️  DEGRADATION! Two-pass is WORSE than single-pass")
    else:
        print(f"✓ Improvement from two-pass processing")

    print(f"\nTop 20 frames with largest prediction differences:")
    print(f"{'Frame':<8} {'Single':<10} {'Two-Pass':<10} {'OF2.2':<10} {'Diff':<10} {'Single Err':<12} {'Two Err':<12}")
    print("-" * 80)

    for frame_idx in top_diff_frames:
        s = single[frame_idx]
        t = two[frame_idx]
        o = of22[frame_idx]
        d = t - s
        err_single = abs(s - o)
        err_two = abs(t - o)

        better_marker = "✓" if err_two < err_single else "✗"
        print(f"{frame_idx:<8} {s:<10.4f} {t:<10.4f} {o:<10.4f} {d:<+10.4f} {err_single:<12.4f} {err_two:<12.4f} {better_marker}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: All predictions over time
    ax = axes[0, 0]
    ax.plot(of22, label='OpenFace 2.2', color='black', linewidth=2, alpha=0.7)
    ax.plot(single, label='Single-pass', color='blue', alpha=0.6)
    ax.plot(two, label='Two-pass', color='red', alpha=0.6)
    ax.set_xlabel('Frame')
    ax.set_ylabel('AU Intensity')
    ax.set_title(f'{au_name} - Predictions Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: First 100 frames (early frame behavior)
    ax = axes[0, 1]
    frames = min(100, len(of22))
    ax.plot(of22[:frames], label='OpenFace 2.2', color='black', linewidth=2, alpha=0.7)
    ax.plot(single[:frames], label='Single-pass', color='blue', alpha=0.6)
    ax.plot(two[:frames], label='Two-pass', color='red', alpha=0.6)
    ax.set_xlabel('Frame')
    ax.set_ylabel('AU Intensity')
    ax.set_title(f'{au_name} - First 100 Frames (Early Frame Behavior)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Scatter - Single-pass vs OF2.2
    ax = axes[1, 0]
    ax.scatter(of22, single, alpha=0.3, s=10)
    max_val = max(of22.max(), single.max())
    min_val = min(of22.min(), single.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('OpenFace 2.2')
    ax.set_ylabel('Single-pass')
    ax.set_title(f'{au_name} - Single-pass (r={r_single:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 4: Scatter - Two-pass vs OF2.2
    ax = axes[1, 1]
    ax.scatter(of22, two, alpha=0.3, s=10)
    max_val = max(of22.max(), two.max())
    min_val = min(of22.min(), two.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('OpenFace 2.2')
    ax.set_ylabel('Two-pass')
    ax.set_title(f'{au_name} - Two-pass (r={r_two:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    output_path = Path(output_dir) / f'diagnosis_{au_name}.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved comparison plot to {output_path}")


def main():
    """Run diagnostic analysis"""

    # Paths
    models_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"
    pdm_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt"
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.hog"
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"
    output_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/degradation_analysis"

    Path(output_dir).mkdir(exist_ok=True)

    # Load models
    print("Loading SVR models...")
    parser = OF22ModelParser(models_dir)
    models = parser.load_all_models(use_recommended=True, use_combined=True)

    # Focus on problematic AUs
    problematic_aus = ['AU02_r', 'AU05_r', 'AU20_r', 'AU23_r']

    print(f"\nFocusing on problematic AUs: {problematic_aus}")

    # Load data
    print(f"\nLoading PDM from {Path(pdm_file).name}...")
    pdm_parser = PDMParser(pdm_file)

    print(f"\nLoading HOG features from {Path(hog_file).name}...")
    hog_parser = OF22HOGParser(hog_file)
    frame_indices, hog_features_all = hog_parser.parse()

    print(f"\nLoading OF2.2 predictions from {Path(csv_file).name}...")
    df = pd.read_csv(csv_file)

    # Get OpenFace predictions
    of22_preds = {}
    for au in problematic_aus:
        of22_preds[au] = df[au].values

    # Run single-pass predictions
    print("\n" + "="*80)
    print("Running SINGLE-PASS predictions...")
    print("="*80)
    single_pass = predict_single_pass(models, hog_features_all, df, pdm_parser, problematic_aus)

    # Run two-pass predictions
    print("\n" + "="*80)
    print("Running TWO-PASS predictions...")
    print("="*80)
    two_pass = predict_two_pass(models, hog_features_all, df, pdm_parser, problematic_aus)

    # Analyze differences for each AU
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    for au in problematic_aus:
        analyze_differences(single_pass, two_pass, of22_preds, au, output_dir)

    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
