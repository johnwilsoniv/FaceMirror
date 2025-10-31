#!/usr/bin/env python3
"""
Deep diagnostic analysis of problematic AUs (r < 0.75)

Problematic AUs:
- AU02 (r=0.560) - Outer brow raiser - Dynamic
- AU05 (r=0.637) - Upper lid raiser - Dynamic
- AU15 (r=0.618) - Lip corner depressor - Dynamic
- AU20 (r=0.522) - Lip stretcher - Dynamic
- AU23 (r=0.723) - Lip tightener - Dynamic (borderline)

All are DYNAMIC models, suggesting issue with running median implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from openface22_model_parser import OF22ModelParser
from openface22_hog_parser import OF22HOGParser
from pdm_parser import PDMParser
from histogram_median_tracker import DualHistogramMedianTracker
from scipy.stats import pearsonr

def extract_geometric_features(df_row, pdm_parser):
    """Extract geometric features using PDM reconstruction"""
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df_row[pdm_cols].values
    return pdm_parser.extract_geometric_features(pdm_params)

def construct_full_feature_vector(hog_features, geom_features):
    """Construct complete 4702-dim feature vector"""
    return np.concatenate([hog_features, geom_features])

def diagnose_au(au_name, model, hog_features_all, df, pdm_parser):
    """Detailed diagnostic for a single AU"""

    print(f"\n{'='*80}")
    print(f"DIAGNOSING {au_name}")
    print(f"{'='*80}")

    is_dynamic = (model['model_type'] == 'dynamic')
    print(f"Model type: {model['model_type']}")
    print(f"Cutoff: {model['cutoff']}")
    print(f"Means shape: {model['means'].shape}")
    print(f"Support vectors shape: {model['support_vectors'].shape}")
    print(f"Bias: {model['bias']:.6f}")

    # Initialize median tracker
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

    # Compute predictions
    python_preds = []
    python_preds_raw = []  # Before smoothing
    running_medians_per_frame = []

    parser = OF22ModelParser("")

    for i in range(min(len(hog_features_all), len(df))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        # Update median tracker
        update_histogram = (i % 2 == 1)
        median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)
        running_median = median_tracker.get_combined_median()
        running_medians_per_frame.append(running_median.copy())

        # Construct features
        full_vector = construct_full_feature_vector(hog_feat, geom_feat)

        # Predict
        if is_dynamic:
            centered = full_vector - model['means'].flatten() - running_median
            pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
            pred = float(pred[0, 0])
        else:
            pred = parser.predict_au(full_vector, model)

        # Clamp predictions to [0, 5] range (matches OpenFace 2.2)
        pred = np.clip(pred, 0.0, 5.0)

        python_preds_raw.append(pred)

    python_preds_raw = np.array(python_preds_raw)

    # Apply temporal smoothing
    window_size = 3
    half_window = (window_size - 1) // 2
    python_preds_smoothed = python_preds_raw.copy()

    for i in range(half_window, len(python_preds_raw) - half_window):
        window_sum = 0.0
        for w in range(-half_window, half_window + 1):
            window_sum += python_preds_raw[i + w]
        python_preds_smoothed[i] = window_sum / window_size

    python_preds = python_preds_smoothed
    of22_preds = df[au_name].values[:len(python_preds)]

    # Calculate metrics
    r_raw, _ = pearsonr(python_preds_raw, of22_preds)
    r_smoothed, _ = pearsonr(python_preds, of22_preds)

    rmse_raw = np.sqrt(np.mean((python_preds_raw - of22_preds) ** 2))
    rmse_smoothed = np.sqrt(np.mean((python_preds - of22_preds) ** 2))

    print(f"\nCorrelation:")
    print(f"  Before smoothing: r = {r_raw:.6f}")
    print(f"  After smoothing:  r = {r_smoothed:.6f}")
    print(f"  Improvement: {r_smoothed - r_raw:+.6f}")

    print(f"\nRMSE:")
    print(f"  Before smoothing: {rmse_raw:.6f}")
    print(f"  After smoothing:  {rmse_smoothed:.6f}")

    # Analyze prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"  Python (raw):")
    print(f"    Mean: {python_preds_raw.mean():.6f}")
    print(f"    Std:  {python_preds_raw.std():.6f}")
    print(f"    Min:  {python_preds_raw.min():.6f}")
    print(f"    Max:  {python_preds_raw.max():.6f}")

    print(f"  OpenFace 2.2:")
    print(f"    Mean: {of22_preds.mean():.6f}")
    print(f"    Std:  {of22_preds.std():.6f}")
    print(f"    Min:  {of22_preds.min():.6f}")
    print(f"    Max:  {of22_preds.max():.6f}")

    # Compute bias
    bias = python_preds_raw.mean() - of22_preds.mean()
    print(f"\n  Systematic bias: {bias:+.6f} (Python - OF2.2)")

    # Analyze errors
    errors = python_preds - of22_preds
    print(f"\nError Analysis:")
    print(f"  Mean error: {errors.mean():+.6f}")
    print(f"  Std error:  {errors.std():.6f}")
    print(f"  Max |error|: {np.abs(errors).max():.6f}")

    # Find worst frames
    worst_indices = np.argsort(np.abs(errors))[-10:][::-1]
    print(f"\nWorst 10 frames (largest |error|):")
    for idx in worst_indices:
        print(f"  Frame {idx}: Python={python_preds[idx]:.4f}, OF2.2={of22_preds[idx]:.4f}, Error={errors[idx]:+.4f}")

    # Analyze running median
    if is_dynamic:
        print(f"\nRunning Median Analysis:")
        running_medians_array = np.array(running_medians_per_frame)
        print(f"  Shape: {running_medians_array.shape}")
        print(f"  Mean: {running_medians_array.mean():.6f}")
        print(f"  Std:  {running_medians_array.std():.6f}")
        print(f"  Min:  {running_medians_array.min():.6f}")
        print(f"  Max:  {running_medians_array.max():.6f}")

        # Check convergence
        print(f"\n  Running median convergence:")
        for frame_idx in [10, 50, 100, 200, 500, 1000]:
            if frame_idx < len(running_medians_array):
                median_val = running_medians_array[frame_idx].mean()
                print(f"    Frame {frame_idx}: mean={median_val:.6f}")

    # Create diagnostic plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Time series comparison
    frames = np.arange(len(python_preds))
    axes[0].plot(frames, of22_preds, 'b-', label='OpenFace 2.2', alpha=0.7, linewidth=1.5)
    axes[0].plot(frames, python_preds, 'r-', label='Python (smoothed)', alpha=0.7, linewidth=1.5)
    axes[0].plot(frames, python_preds_raw, 'g--', label='Python (raw)', alpha=0.5, linewidth=1)
    axes[0].set_xlabel('Frame', fontsize=11)
    axes[0].set_ylabel('AU Intensity', fontsize=11)
    axes[0].set_title(f'{au_name} - Time Series Comparison (r={r_smoothed:.4f})',
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    axes[1].scatter(of22_preds, python_preds, alpha=0.3, s=10)
    min_val = min(of22_preds.min(), python_preds.min())
    max_val = max(of22_preds.max(), python_preds.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match', linewidth=2)
    axes[1].set_xlabel('OpenFace 2.2 Prediction', fontsize=11)
    axes[1].set_ylabel('Python Prediction', fontsize=11)
    axes[1].set_title(f'{au_name} - Correlation Plot', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    # Plot 3: Error over time
    axes[2].plot(frames, errors, 'r-', alpha=0.7, linewidth=1)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[2].axhline(y=errors.mean(), color='b', linestyle=':', label=f'Mean error: {errors.mean():+.4f}', linewidth=2)
    axes[2].fill_between(frames,
                         errors.mean() - errors.std(),
                         errors.mean() + errors.std(),
                         alpha=0.2, color='blue', label=f'±1 std: {errors.std():.4f}')
    axes[2].set_xlabel('Frame', fontsize=11)
    axes[2].set_ylabel('Error (Python - OF2.2)', fontsize=11)
    axes[2].set_title(f'{au_name} - Prediction Error Over Time', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/diagnostics')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'diagnostic_{au_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'au_name': au_name,
        'r_raw': r_raw,
        'r_smoothed': r_smoothed,
        'rmse_raw': rmse_raw,
        'rmse_smoothed': rmse_smoothed,
        'bias': bias,
        'mean_error': errors.mean(),
        'std_error': errors.std()
    }

def main():
    """Run diagnostics on problematic AUs"""

    print("="*80)
    print("DEEP DIAGNOSTIC ANALYSIS OF PROBLEMATIC AUs")
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
    pdm_parser = PDMParser(pdm_file)
    hog_parser = OF22HOGParser(hog_file)
    frame_indices, hog_features_all = hog_parser.parse()
    df = pd.read_csv(csv_file)

    print(f"✓ Loaded {len(models)} AU models")
    print(f"✓ Loaded {len(frame_indices)} frames")

    # Problematic AUs
    problematic_aus = ['AU02_r', 'AU05_r', 'AU15_r', 'AU20_r', 'AU23_r']

    # Also analyze one working AU for comparison
    working_au = 'AU25_r'  # r=0.993

    print(f"\nAnalyzing {len(problematic_aus)} problematic AUs + 1 working AU")

    results = []

    # Analyze working AU first
    print(f"\n{'#'*80}")
    print(f"BASELINE: Analyzing working AU for comparison")
    print(f"{'#'*80}")
    result = diagnose_au(working_au, models[working_au], hog_features_all, df, pdm_parser)
    results.append(result)

    # Analyze problematic AUs
    for au_name in problematic_aus:
        result = diagnose_au(au_name, models[au_name], hog_features_all, df, pdm_parser)
        results.append(result)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nDiagnostic plots saved to: of22_validation/diagnostics/")

if __name__ == "__main__":
    main()
