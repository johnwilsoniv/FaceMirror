#!/usr/bin/env python3
"""
Frame-by-frame comparison of running median for AU15 debugging.

Strategy:
1. Extract OpenFace's "means" from the model (what they subtract)
2. Reverse-engineer what running median OF2.2 must have used
3. Compare with our running median frame-by-frame
4. Find where divergence starts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from openface22_model_parser import OF22ModelParser
from openface22_hog_parser import OF22HOGParser
from pdm_parser import PDMParser
from histogram_median_tracker import DualHistogramMedianTracker

def extract_geometric_features(df_row, pdm_parser):
    """Extract geometric features using PDM reconstruction"""
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df_row[pdm_cols].values
    return pdm_parser.extract_geometric_features(pdm_params)

def construct_full_feature_vector(hog_features, geom_features):
    """Construct complete 4702-dim feature vector"""
    return np.concatenate([hog_features, geom_features])

def main():
    print("="*80)
    print("AU15 RUNNING MEDIAN DEBUG")
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

    print(f"✓ Loaded AU15 model (dynamic, cutoff={au15_model['cutoff']})")
    print(f"✓ Loaded {len(frame_indices)} frames")

    # Initialize our running median tracker
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

    # Track predictions and running median per frame
    python_preds_raw = []
    of22_preds = df['AU15_r'].values

    our_running_medians = []
    implied_running_medians = []  # Reverse-engineered from OF2.2

    print("\nProcessing frames...")
    for i in range(min(len(hog_features_all), len(df))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        # Update our tracker
        update_histogram = (i % 2 == 1)
        median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)
        our_running_median = median_tracker.get_combined_median()
        our_running_medians.append(our_running_median.copy())

        # Construct features
        full_vector = construct_full_feature_vector(hog_feat, geom_feat)

        # Our prediction
        centered = full_vector - au15_model['means'].flatten() - our_running_median
        pred = np.dot(centered.reshape(1, -1), au15_model['support_vectors']) + au15_model['bias']
        pred = float(pred[0, 0])
        pred = np.clip(pred, 0.0, 5.0)
        python_preds_raw.append(pred)

        # Reverse-engineer implied running median from OF2.2
        # OF2.2 does: pred = (features - means - running_median) * SV + bias
        # So: running_median = features - means - (pred - bias) / SV
        of22_pred = of22_preds[i]
        # Solve for running_median:
        # pred = (features - means - rm) * SV + bias
        # pred - bias = (features - means - rm) * SV
        # (pred - bias) / SV = features - means - rm
        # rm = features - means - (pred - bias) / SV

        # But this only works if we know the prediction BEFORE smoothing
        # Skip reverse engineering for now, just compare our predictions

    python_preds_raw = np.array(python_preds_raw)

    # Apply temporal smoothing
    window_size = 3
    half_window = (window_size - 1) // 2
    python_preds = python_preds_raw.copy()

    for i in range(half_window, len(python_preds_raw) - half_window):
        window_sum = 0.0
        for w in range(-half_window, half_window + 1):
            window_sum += python_preds_raw[i + w]
        python_preds[i] = window_sum / window_size

    # Analyze running median statistics
    print("\n" + "="*80)
    print("RUNNING MEDIAN ANALYSIS")
    print("="*80)

    our_running_medians = np.array(our_running_medians)
    print(f"\nOur running median shape: {our_running_medians.shape}")
    print(f"Mean: {our_running_medians.mean():.6f}")
    print(f"Std:  {our_running_medians.std():.6f}")
    print(f"Min:  {our_running_medians.min():.6f}")
    print(f"Max:  {our_running_medians.max():.6f}")

    # Analyze prediction errors
    errors = python_preds - of22_preds[:len(python_preds)]
    from scipy.stats import pearsonr
    r, p = pearsonr(python_preds, of22_preds[:len(python_preds)])

    print(f"\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    print(f"\nCorrelation: r = {r:.6f}")
    print(f"Mean error: {errors.mean():.6f}")
    print(f"Std error:  {errors.std():.6f}")
    print(f"Max |error|: {np.abs(errors).max():.6f}")

    # Find worst frames
    worst_indices = np.argsort(np.abs(errors))[-20:][::-1]
    print(f"\nWorst 20 frames:")
    for idx in worst_indices:
        print(f"  Frame {idx}: Python={python_preds[idx]:.4f}, OF2.2={of22_preds[idx]:.4f}, Error={errors[idx]:+.4f}")

    # Plot running median convergence
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Running median mean over time
    frames = np.arange(len(our_running_medians))
    running_median_means = our_running_medians.mean(axis=1)

    axes[0].plot(frames, running_median_means, 'b-', linewidth=1)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Running Median Mean')
    axes[0].set_title('Running Median Convergence (Mean Across Dimensions)')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prediction comparison
    axes[1].plot(frames, of22_preds[:len(frames)], 'b-', label='OpenFace 2.2', alpha=0.7, linewidth=1.5)
    axes[1].plot(frames, python_preds, 'r-', label='Python', alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('AU15 Intensity')
    axes[1].set_title(f'AU15 Predictions (r={r:.6f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Error over time
    axes[2].plot(frames, errors, 'r-', alpha=0.7, linewidth=1)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[2].axhline(y=errors.mean(), color='b', linestyle=':', label=f'Mean: {errors.mean():+.4f}', linewidth=2)
    axes[2].fill_between(frames,
                         errors.mean() - errors.std(),
                         errors.mean() + errors.std(),
                         alpha=0.2, color='blue')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Error (Python - OF2.2)')
    axes[2].set_title('Prediction Error Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path('of22_validation/au15_debug.png')
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")

    # Analyze specific frames with large errors
    print(f"\n" + "="*80)
    print("DETAILED ANALYSIS OF WORST FRAMES")
    print("="*80)

    for idx in worst_indices[:5]:  # Top 5 worst
        print(f"\nFrame {idx}:")
        print(f"  Python pred: {python_preds[idx]:.4f}")
        print(f"  OF2.2 pred:  {of22_preds[idx]:.4f}")
        print(f"  Error:       {errors[idx]:+.4f}")
        print(f"  Running median mean: {our_running_medians[idx].mean():.6f}")
        print(f"  Running median std:  {our_running_medians[idx].std():.6f}")

if __name__ == "__main__":
    main()
