#!/usr/bin/env python3
"""
Train Tier 2 BBox Correction Model

Trains a parametric regression model to adaptively correct RetinaFace bboxes
based on face size, aspect ratio, and image dimensions.

Tier 2 Features:
- Face size (sqrt of bbox area)
- Face aspect ratio (width/height)
- Face center position (x, y as fraction of image dimensions)
- Image dimensions (width, height)

Model: Linear Regression for interpretability and speed
Output: Center offset (dx, dy) and size scaling factors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def load_dataset(csv_path):
    """Load and prepare dataset."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def extract_features(df):
    """
    Extract Tier 2 features for parametric regression.

    Features:
    - rf_size: Face size (RetinaFace detection)
    - rf_aspect_ratio: Face aspect ratio
    - rf_center_x_norm: Normalized center x (0-1)
    - rf_center_y_norm: Normalized center y (0-1)
    - image_width: Image width
    - image_height: Image height
    - image_aspect_ratio: Image aspect ratio
    """
    features = pd.DataFrame({
        'rf_size': df['rf_size'],
        'rf_aspect_ratio': df['rf_aspect_ratio'],
        'rf_center_x_norm': df['rf_center_x'] / df['image_width'],
        'rf_center_y_norm': df['rf_center_y'] / df['image_height'],
        'image_width': df['image_width'],
        'image_height': df['image_height'],
        'image_aspect_ratio': df['image_width'] / df['image_height'],
    })

    # Targets: What we want to predict
    targets = pd.DataFrame({
        'center_offset_x': df['center_offset_x'],
        'center_offset_y': df['center_offset_y'],
        'width_correction': df['cpp_bbox_w'] - df['rf_bbox_w'],
        'height_correction': df['cpp_bbox_h'] - df['rf_bbox_h'],
    })

    return features, targets


def train_model(X_train, y_train, X_test, y_test):
    """
    Train Ridge regression model with cross-validation for optimal alpha.

    Ridge (L2 regularization) prevents overfitting on correlated features.
    """
    print("\nTraining Ridge Regression model...")

    # Standardize features (important for Ridge regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train separate models for each target (center_x, center_y, width, height)
    models = {}
    metrics = {}

    for target_name in y_train.columns:
        print(f"\n  Training model for: {target_name}")

        # Ridge with alpha=1.0 (moderate regularization)
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train[target_name])

        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_mae = mean_absolute_error(y_train[target_name], train_pred)
        test_mae = mean_absolute_error(y_test[target_name], test_pred)
        test_r2 = r2_score(y_test[target_name], test_pred)

        print(f"    Train MAE: {train_mae:.2f}px")
        print(f"    Test MAE:  {test_mae:.2f}px")
        print(f"    Test R²:   {test_r2:.3f}")

        models[target_name] = model
        metrics[target_name] = {
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2)
        }

        # Print feature importance (coefficients)
        print(f"    Feature importance:")
        for feat_name, coef in zip(X_train.columns, model.coef_):
            if abs(coef) > 0.01:  # Only show significant features
                print(f"      {feat_name:25s}: {coef:+.3f}")

    return models, scaler, metrics


def evaluate_overall_improvement(df, X_test, y_test, models, scaler, output_dir):
    """
    Evaluate overall bbox correction improvement.

    Compares:
    - Baseline (RetinaFace raw)
    - Tier 2 Model (our adaptive correction)
    """
    print("\n" + "="*80)
    print("OVERALL EVALUATION")
    print("="*80)

    # Get test set indices
    test_indices = X_test.index

    # Baseline: RetinaFace raw center offset
    baseline_center_offset = df.loc[test_indices, 'center_offset_total'].values

    # Tier 2 Model: Predicted corrections
    X_test_scaled = scaler.transform(X_test)

    pred_center_offset_x = models['center_offset_x'].predict(X_test_scaled)
    pred_center_offset_y = models['center_offset_y'].predict(X_test_scaled)

    # Apply corrections
    corrected_center_x = df.loc[test_indices, 'rf_center_x'] + pred_center_offset_x
    corrected_center_y = df.loc[test_indices, 'rf_center_y'] + pred_center_offset_y

    # Ground truth
    gt_center_x = df.loc[test_indices, 'cpp_center_x']
    gt_center_y = df.loc[test_indices, 'cpp_center_y']

    # Calculate residual errors
    residual_offset_x = gt_center_x - corrected_center_x
    residual_offset_y = gt_center_y - corrected_center_y
    residual_center_offset = np.sqrt(residual_offset_x**2 + residual_offset_y**2)

    # Statistics
    print(f"\nCenter Offset (Euclidean distance to ground truth):")
    print(f"  Baseline (RetinaFace raw):")
    print(f"    Mean:   {baseline_center_offset.mean():.2f}px")
    print(f"    Median: {np.median(baseline_center_offset):.2f}px")
    print(f"    Std:    {baseline_center_offset.std():.2f}px")
    print(f"    Max:    {baseline_center_offset.max():.2f}px")

    print(f"\n  Tier 2 Model (after correction):")
    print(f"    Mean:   {residual_center_offset.mean():.2f}px")
    print(f"    Median: {np.median(residual_center_offset):.2f}px")
    print(f"    Std:    {residual_center_offset.std():.2f}px")
    print(f"    Max:    {residual_center_offset.max():.2f}px")

    improvement = (baseline_center_offset.mean() - residual_center_offset.mean()) / baseline_center_offset.mean() * 100
    print(f"\n  Improvement: {improvement:.1f}%")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    axes[0].hist(baseline_center_offset, bins=50, alpha=0.5, label='Baseline (RetinaFace raw)', color='red')
    axes[0].hist(residual_center_offset, bins=50, alpha=0.5, label='Tier 2 Model (corrected)', color='green')
    axes[0].axvline(baseline_center_offset.mean(), color='red', linestyle='--', label=f'Baseline mean: {baseline_center_offset.mean():.1f}px')
    axes[0].axvline(residual_center_offset.mean(), color='green', linestyle='--', label=f'Model mean: {residual_center_offset.mean():.1f}px')
    axes[0].set_xlabel('Center Offset (px)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Center Offset Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Scatter plot: Baseline vs Model
    axes[1].scatter(baseline_center_offset, residual_center_offset, alpha=0.3)
    axes[1].plot([0, baseline_center_offset.max()], [0, baseline_center_offset.max()], 'r--', label='No improvement line')
    axes[1].set_xlabel('Baseline Center Offset (px)')
    axes[1].set_ylabel('Model Center Offset (px)')
    axes[1].set_title('Per-Sample Improvement')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'model_evaluation.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\n✓ Saved evaluation plot: {plot_path}")

    return {
        'baseline_mean': float(baseline_center_offset.mean()),
        'baseline_median': float(np.median(baseline_center_offset)),
        'baseline_std': float(baseline_center_offset.std()),
        'model_mean': float(residual_center_offset.mean()),
        'model_median': float(np.median(residual_center_offset)),
        'model_std': float(residual_center_offset.std()),
        'improvement_percent': float(improvement)
    }


def main():
    """Main training pipeline."""
    print("="*80)
    print("TIER 2 BBOX CORRECTION MODEL TRAINING")
    print("="*80)

    # Load dataset
    dataset_path = Path("bbox_dataset/bbox_dataset.csv")
    df = load_dataset(dataset_path)

    # Extract features and targets
    print("\nExtracting features...")
    X, y = extract_features(df)
    print(f"Features: {list(X.columns)}")
    print(f"Targets:  {list(y.columns)}")

    # Split into train/test (80/20)
    # Use stratified split by patient_id to avoid data leakage
    unique_patients = df['patient_id'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )

    train_mask = df['patient_id'].isin(train_patients)
    test_mask = df['patient_id'].isin(test_patients)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTrain set: {len(X_train)} samples ({len(train_patients)} patients)")
    print(f"Test set:  {len(X_test)} samples ({len(test_patients)} patients)")

    # Train model
    models, scaler, metrics = train_model(X_train, y_train, X_test, y_test)

    # Evaluate overall improvement
    output_dir = Path("bbox_correction_model")
    output_dir.mkdir(exist_ok=True)

    overall_metrics = evaluate_overall_improvement(df, X_test, y_test, models, scaler, output_dir)

    # Save models
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    model_data = {
        'models': models,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'target_names': list(y.columns),
        'metrics': metrics,
        'overall_metrics': overall_metrics
    }

    model_path = output_dir / "bbox_correction_model.pkl"
    joblib.dump(model_data, model_path)
    print(f"✓ Saved model: {model_path}")

    # Save metrics as JSON
    metrics_path = output_dir / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({'per_target': metrics, 'overall': overall_metrics}, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")

    print("\n" + "="*80)
    print("✅ MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel achieves {overall_metrics['improvement_percent']:.1f}% improvement")
    print(f"  Baseline: {overall_metrics['baseline_mean']:.1f}px → Model: {overall_metrics['model_mean']:.1f}px")
    print(f"\nNext step: Integrate model into RetinaFaceCorrectedDetector")


if __name__ == "__main__":
    main()
