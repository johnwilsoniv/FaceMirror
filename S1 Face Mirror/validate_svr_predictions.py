#!/usr/bin/env python3
"""
Validate Python SVR Predictions vs OpenFace 2.2

Compares Python SVR model predictions against OF2.2's C++ predictions
to verify correctness of the implementation.

Feature vector composition (4702 dims):
1. HOG features: 4464 dims (from .hog file)
2. 3D landmarks: 204 dims (X_0...X_67, Y_0...Y_67, Z_0...Z_67)
3. PDM params: 34 dims (p_0...p_33)
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
    """
    Extract 238-dimensional geometric features from CSV row

    Uses PDM reconstruction to match OpenFace 2.2's approach:
    - Reconstructs landmarks from PDM parameters (NOT raw landmarks)
    - Concatenates with PDM parameters

    Args:
        df_row: Single row from OF2.2 CSV output
        pdm_parser: PDMParser instance for landmark reconstruction

    Returns:
        numpy array of shape (238,) containing [reconstructed_landmarks, PDM_params]
    """
    # Extract PDM shape parameters
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df_row[pdm_cols].values  # 34 dims

    # Use PDM parser to extract geometric features (matches OF2.2)
    geom_features = pdm_parser.extract_geometric_features(pdm_params)

    assert geom_features.shape == (238,), f"Expected 238 dims, got {geom_features.shape}"

    return geom_features


def construct_full_feature_vector(hog_features, geom_features):
    """
    Construct complete 4702-dimensional feature vector

    Args:
        hog_features: (4464,) array from .hog file
        geom_features: (238,) array from CSV

    Returns:
        (4702,) array: [HOG, geometric]
    """
    assert hog_features.shape == (4464,), f"Expected 4464 HOG dims, got {hog_features.shape}"
    assert geom_features.shape == (238,), f"Expected 238 geom dims, got {geom_features.shape}"

    full_vector = np.concatenate([hog_features, geom_features])

    return full_vector


def validate_predictions(models, hog_file, csv_file, pdm_file, output_dir):
    """
    Validate Python SVR predictions against OF2.2 CSV output

    Args:
        models: Dictionary of loaded SVR models
        hog_file: Path to .hog file
        csv_file: Path to OF2.2 CSV output
        pdm_file: Path to PDM .txt file
        output_dir: Directory to save comparison plots
    """
    print("="*80)
    print("SVR Prediction Validation")
    print("="*80)

    # Load PDM for geometric feature reconstruction
    print(f"\nLoading PDM from {Path(pdm_file).name}...")
    pdm_parser = PDMParser(pdm_file)

    # Parse HOG file
    print(f"\nLoading HOG features from {Path(hog_file).name}...")
    hog_parser = OF22HOGParser(hog_file)
    frame_indices, hog_features_all = hog_parser.parse()
    print(f"✓ Loaded {len(frame_indices)} frames with {hog_features_all.shape[1]} HOG features")

    # Load CSV
    print(f"\nLoading OF2.2 predictions from {Path(csv_file).name}...")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} rows")

    # Verify alignment
    if len(frame_indices) != len(df):
        print(f"⚠️  Warning: HOG has {len(frame_indices)} frames, CSV has {len(df)} rows")

    # Extract AU columns from CSV
    au_cols_csv = sorted([c for c in df.columns if c.startswith('AU') and c.endswith('_r')])
    print(f"✓ Found {len(au_cols_csv)} AU intensity columns in CSV")

    # Create histogram-based running median tracker for dynamic models
    print("\nInitializing histogram-based running median tracker for dynamic models...")
    # Using OF2.2 actual parameters from FaceAnalyser.cpp:
    # HOG: num_bins=1000, min_val=-0.005, max_val=1.0
    # Geometric: num_bins=10000, min_val=-60, max_val=60
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

    # PASS 1: Build running median history frame-by-frame (online processing)
    print(f"\n{'='*80}")
    print("PASS 1: Building running median from video history (online processing)...")
    print(f"{'='*80}")

    running_medians_per_frame = []
    stored_features = []  # Store (hog, geom) for first 3000 frames
    max_init_frames = min(3000, min(len(frame_indices), len(df)))

    for i in range(min(len(frame_indices), len(df))):
        hog_feat = hog_features_all[i]
        geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

        # Update tracker (update histogram every 2nd frame like OF2.2)
        update_histogram = (i % 2 == 1)
        median_tracker.update(hog_feat, geom_feat, update_histogram=update_histogram)

        # Store the running median at this frame
        running_medians_per_frame.append(median_tracker.get_combined_median().copy())

        # Store features for postprocessing (first 3000 frames only)
        if i < max_init_frames:
            stored_features.append((hog_feat.copy(), geom_feat.copy()))

    print(f"✓ Running median calculated for {len(running_medians_per_frame)} frames")
    print(f"✓ Stored features for {len(stored_features)} early frames for postprocessing")

    # PASS 2: Postprocess early frames with final running median (offline processing)
    print(f"\n{'='*80}")
    print(f"PASS 2: Reprocessing first {len(stored_features)} frames with final median (offline)...")
    print(f"{'='*80}")

    final_median = median_tracker.get_combined_median()
    print(f"Final median stats: mean={final_median.mean():.6f}, std={final_median.std():.6f}")
    print(f"HOG median range: [{final_median[:4464].min():.6f}, {final_median[:4464].max():.6f}]")
    print(f"Geom median range: [{final_median[4464:].min():.6f}, {final_median[4464:].max():.6f}]")

    # Re-update early frames to use final median
    frames_updated = 0
    for i in range(len(stored_features)):
        # Replace running median for this frame with final median
        running_medians_per_frame[i] = final_median.copy()
        frames_updated += 1

    print(f"✓ Updated running median for first {frames_updated} frames to use final stable median")

    # Compare predictions for each AU
    print(f"\n{'='*80}")
    print("Comparing Python SVR vs OF2.2 predictions:")
    print(f"{'='*80}\n")

    results = []

    for au_name in sorted(models.keys()):
        if au_name not in au_cols_csv:
            print(f"⚠️  {au_name}: Not in CSV, skipping")
            continue

        model = models[au_name]
        is_dynamic = (model['model_type'] == 'dynamic')

        # Predict for all frames using pre-computed running medians
        python_predictions = []

        for i in range(min(len(frame_indices), len(df))):
            # Construct features
            hog_feat = hog_features_all[i]
            geom_feat = extract_geometric_features(df.iloc[i], pdm_parser)

            # Get running median for this frame
            running_median = running_medians_per_frame[i]

            # Construct full feature vector
            full_vector = construct_full_feature_vector(hog_feat, geom_feat)

            # Predict using Python SVR
            from openface22_model_parser import OF22ModelParser
            parser = OF22ModelParser("")  # Dummy parser for prediction method

            if is_dynamic:
                # Dynamic models: Subtract both means and running median
                centered = full_vector - model['means'].flatten() - running_median
                pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
                pred = float(pred[0, 0])
            else:
                # Static models: Use standard prediction
                pred = parser.predict_au(full_vector, model)

            # Clamp predictions to [0, 5] range (matches OpenFace 2.2)
            pred = np.clip(pred, 0.0, 5.0)

            python_predictions.append(pred)

        python_predictions = np.array(python_predictions)

        # Apply cutoff-based offset adjustment (OpenFace lines 605-630)
        # This shifts the neutral baseline by subtracting the cutoff percentile value
        if is_dynamic and model.get('cutoff', -1) != -1:
            cutoff = model['cutoff']
            # Sort predictions to find the cutoff percentile value
            sorted_preds = np.sort(python_predictions)
            cutoff_idx = int(len(sorted_preds) * cutoff)
            offset = sorted_preds[cutoff_idx]

            # Subtract offset from all predictions
            python_predictions = python_predictions - offset

            # Clamp to [0, 5] after offset adjustment
            python_predictions = np.clip(python_predictions, 0.0, 5.0)

        # Apply 3-frame moving average (matches OpenFace 2.2 offline processing)
        window_size = 3
        half_window = (window_size - 1) // 2
        python_predictions_smoothed = python_predictions.copy()

        for i in range(half_window, len(python_predictions) - half_window):
            window_sum = 0.0
            for w in range(-half_window, half_window + 1):
                window_sum += python_predictions[i + w]
            python_predictions_smoothed[i] = window_sum / window_size

        python_predictions = python_predictions_smoothed
        of22_predictions = df[au_name].values[:len(python_predictions)]

        # Calculate correlation
        r, p_value = pearsonr(python_predictions, of22_predictions)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((python_predictions - of22_predictions) ** 2))

        # Calculate mean absolute error
        mae = np.mean(np.abs(python_predictions - of22_predictions))

        model_type_str = "dynamic" if is_dynamic else "static"
        print(f"{au_name} ({model_type_str}):")
        print(f"  Correlation (r): {r:.6f} (p={p_value:.2e})")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")

        if r > 0.99:
            print(f"  ✓ EXCELLENT match!")
        elif r > 0.95:
            print(f"  ✓ Very good match")
        elif r > 0.90:
            print(f"  ~ Good match")
        else:
            print(f"  ✗ Poor match - investigation needed")

        print()

        results.append({
            'AU': au_name,
            'correlation': r,
            'p_value': p_value,
            'rmse': rmse,
            'mae': mae
        })

        # Create comparison plot
        plot_comparison(python_predictions, of22_predictions, au_name, r, output_dir)

    # Summary
    print(f"{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(results)
    avg_r = results_df['correlation'].mean()
    avg_rmse = results_df['rmse'].mean()
    avg_mae = results_df['mae'].mean()

    excellent = (results_df['correlation'] > 0.99).sum()
    very_good = ((results_df['correlation'] > 0.95) & (results_df['correlation'] <= 0.99)).sum()
    good = ((results_df['correlation'] > 0.90) & (results_df['correlation'] <= 0.95)).sum()
    poor = (results_df['correlation'] <= 0.90).sum()

    print(f"Tested {len(results_df)} AUs:")
    print(f"  Excellent (r > 0.99): {excellent}")
    print(f"  Very good (r > 0.95): {very_good}")
    print(f"  Good (r > 0.90): {good}")
    print(f"  Poor (r ≤ 0.90): {poor}")
    print(f"\nAverage metrics:")
    print(f"  Correlation: {avg_r:.6f}")
    print(f"  RMSE: {avg_rmse:.6f}")
    print(f"  MAE: {avg_mae:.6f}")

    if avg_r > 0.99:
        print(f"\n✓ SUCCESS: Python SVR implementation matches OF2.2!")
    elif avg_r > 0.95:
        print(f"\n✓ Python SVR implementation is very close to OF2.2")
    else:
        print(f"\n⚠️  Warning: Python SVR deviates from OF2.2, further investigation needed")

    return results_df


def plot_comparison(python_preds, of22_preds, au_name, r, output_dir):
    """Create scatter plot comparing predictions"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(of22_preds, python_preds, alpha=0.3, s=10)

    # Reference line
    max_val = max(of22_preds.max(), python_preds.max())
    min_val = min(of22_preds.min(), python_preds.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect match')

    ax.set_xlabel('OpenFace 2.2 Prediction', fontsize=12)
    ax.set_ylabel('Python SVR Prediction', fontsize=12)
    ax.set_title(f'{au_name} - Correlation: r={r:.6f}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / f'validation_{au_name}.png', dpi=150)
    plt.close()


def main():
    """Run validation"""

    # Paths
    models_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"
    pdm_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt"
    hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.hog"
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"
    output_dir = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/comparison_plots"

    # Load SVR models
    print("Loading SVR models...")
    parser = OF22ModelParser(models_dir)
    models = parser.load_all_models(use_recommended=True, use_combined=True)
    print()

    # Run validation
    results = validate_predictions(models, hog_file, csv_file, pdm_file, output_dir)

    print(f"\n{'='*80}")
    print(f"Validation complete! Plots saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
