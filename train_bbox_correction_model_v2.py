#!/usr/bin/env python3
"""
Train Tier 2 BBox Correction Model - V2 (REFORMULATED)

CRITICAL CHANGE from V1:
- V1 predicted width_correction and height_correction independently
  → Problem: init_scale = sqrt(width × height) needs them correlated!
  → Result: Only 0.4% improvement in init_scale error (12.75% → 12.70%)

- V2 predicts scale_ratio and aspect_ratio_adjustment directly
  → scale_ratio = cpp_scale / rf_scale (directly optimizes init_scale!)
  → aspect_ratio_adjustment = (cpp_w/cpp_h) - (rf_w/rf_h)
  → Expected: Massive improvement in init_scale error

Tier 2 Features (unchanged):
- Face size (sqrt of bbox area)
- Face aspect ratio (width/height)
- Face center position (x, y as fraction of image dimensions)
- Image dimensions (width, height)

Model: Ridge Regression for interpretability and speed
Output: Center offset (dx, dy), scale ratio, and aspect ratio adjustment
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

    Features (unchanged from V1):
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

    # Compute derived quantities
    rf_scale = np.sqrt(df['rf_bbox_w'] * df['rf_bbox_h'])
    cpp_scale = np.sqrt(df['cpp_bbox_w'] * df['cpp_bbox_h'])

    rf_aspect = df['rf_bbox_w'] / df['rf_bbox_h']
    cpp_aspect = df['cpp_bbox_w'] / df['cpp_bbox_h']

    # Targets: REFORMULATED for V2!
    targets = pd.DataFrame({
        'center_offset_x': df['center_offset_x'],
        'center_offset_y': df['center_offset_y'],
        'scale_ratio': cpp_scale / rf_scale,  # NEW! Directly optimizes init_scale
        'aspect_ratio_adjustment': cpp_aspect - rf_aspect,  # NEW! Adjusts w/h ratio
    })

    return features, targets


def train_model(X_train, y_train, X_test, y_test):
    """
    Train Ridge regression model with cross-validation for optimal alpha.

    Ridge (L2 regularization) prevents overfitting on correlated features.
    """
    print("\nTraining Ridge Regression model (V2 - Reformulated)...")

    # Standardize features (important for Ridge regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train separate models for each target
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

        # Different units for different targets
        if target_name in ['center_offset_x', 'center_offset_y']:
            unit = "px"
        elif target_name == 'scale_ratio':
            unit = ""  # Ratio (dimensionless)
        elif target_name == 'aspect_ratio_adjustment':
            unit = ""  # Ratio adjustment

        print(f"    Train MAE: {train_mae:.4f}{unit}")
        print(f"    Test MAE:  {test_mae:.4f}{unit}")
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


def apply_v2_correction(rf_bbox, predictions):
    """
    Apply V2 corrections to RetinaFace bbox.

    Args:
        rf_bbox: (x, y, w, h) - RetinaFace raw bbox
        predictions: dict with keys ['center_offset_x', 'center_offset_y',
                                     'scale_ratio', 'aspect_ratio_adjustment']

    Returns:
        corrected_bbox: (x, y, w, h) - Corrected bbox
    """
    rf_x, rf_y, rf_w, rf_h = rf_bbox

    # 1. Correct center
    rf_center_x = rf_x + rf_w / 2
    rf_center_y = rf_y + rf_h / 2

    corrected_center_x = rf_center_x + predictions['center_offset_x']
    corrected_center_y = rf_center_y + predictions['center_offset_y']

    # 2. Correct scale
    rf_scale = np.sqrt(rf_w * rf_h)
    corrected_scale = rf_scale * predictions['scale_ratio']

    # 3. Correct aspect ratio
    rf_aspect = rf_w / rf_h
    corrected_aspect = rf_aspect + predictions['aspect_ratio_adjustment']

    # 4. Convert (scale, aspect) back to (width, height)
    # scale = sqrt(w * h)
    # aspect = w / h
    # Therefore:
    #   w = scale * sqrt(aspect)
    #   h = scale / sqrt(aspect)
    corrected_w = corrected_scale * np.sqrt(corrected_aspect)
    corrected_h = corrected_scale / np.sqrt(corrected_aspect)

    # 5. Convert center-based to corner-based
    corrected_x = corrected_center_x - corrected_w / 2
    corrected_y = corrected_center_y - corrected_h / 2

    return (corrected_x, corrected_y, corrected_w, corrected_h)


def evaluate_overall_improvement(df, X_test, y_test, models, scaler, output_dir):
    """
    Evaluate overall bbox correction improvement with V2 metrics.

    Focuses on initialization quality metrics:
    - Init scale error (%)
    - Bbox IoU
    - Center offset (px)
    """
    print("\n" + "="*80)
    print("OVERALL EVALUATION (V2 - INITIALIZATION QUALITY FOCUS)")
    print("="*80)

    # Get test set indices
    test_indices = X_test.index

    # Get predictions
    X_test_scaled = scaler.transform(X_test)

    predictions = {}
    for target_name in y_test.columns:
        predictions[target_name] = models[target_name].predict(X_test_scaled)

    # Apply V2 corrections
    corrected_bboxes = []
    for i, idx in enumerate(test_indices):
        rf_bbox = (
            df.loc[idx, 'rf_bbox_x'],
            df.loc[idx, 'rf_bbox_y'],
            df.loc[idx, 'rf_bbox_w'],
            df.loc[idx, 'rf_bbox_h']
        )

        pred_dict = {k: v[i] for k, v in predictions.items()}
        corrected_bbox = apply_v2_correction(rf_bbox, pred_dict)
        corrected_bboxes.append(corrected_bbox)

    corrected_bboxes = np.array(corrected_bboxes)

    # Ground truth
    gt_bboxes = np.array([
        df.loc[test_indices, 'cpp_bbox_x'],
        df.loc[test_indices, 'cpp_bbox_y'],
        df.loc[test_indices, 'cpp_bbox_w'],
        df.loc[test_indices, 'cpp_bbox_h']
    ]).T

    rf_bboxes = np.array([
        df.loc[test_indices, 'rf_bbox_x'],
        df.loc[test_indices, 'rf_bbox_y'],
        df.loc[test_indices, 'rf_bbox_w'],
        df.loc[test_indices, 'rf_bbox_h']
    ]).T

    # === 1. INIT SCALE ERROR (PRIMARY METRIC) ===
    def compute_scale_error(bbox1, bbox2):
        scale1 = np.sqrt(bbox1[:, 2] * bbox1[:, 3])
        scale2 = np.sqrt(bbox2[:, 2] * bbox2[:, 3])
        return np.abs(scale1 - scale2) / scale2 * 100

    baseline_scale_errors = compute_scale_error(rf_bboxes, gt_bboxes)
    model_scale_errors = compute_scale_error(corrected_bboxes, gt_bboxes)

    print(f"\n1. INIT SCALE ERROR (%):")
    print(f"  Baseline (RetinaFace raw):")
    print(f"    Mean:   {baseline_scale_errors.mean():.2f}%")
    print(f"    Median: {np.median(baseline_scale_errors):.2f}%")
    print(f"    <3%:    {(baseline_scale_errors < 3).sum()}/{len(baseline_scale_errors)} ({(baseline_scale_errors < 3).mean()*100:.1f}%)")
    print(f"    <5%:    {(baseline_scale_errors < 5).sum()}/{len(baseline_scale_errors)} ({(baseline_scale_errors < 5).mean()*100:.1f}%)")

    print(f"\n  Tier 2 V2 Model (after correction):")
    print(f"    Mean:   {model_scale_errors.mean():.2f}%")
    print(f"    Median: {np.median(model_scale_errors):.2f}%")
    print(f"    <3%:    {(model_scale_errors < 3).sum()}/{len(model_scale_errors)} ({(model_scale_errors < 3).mean()*100:.1f}%)")
    print(f"    <5%:    {(model_scale_errors < 5).sum()}/{len(model_scale_errors)} ({(model_scale_errors < 5).mean()*100:.1f}%)")

    scale_improvement = (baseline_scale_errors.mean() - model_scale_errors.mean()) / baseline_scale_errors.mean() * 100
    print(f"\n  Improvement: {scale_improvement:.1f}%")

    # === 2. CENTER OFFSET ===
    def compute_center_offset(bbox1, bbox2):
        center1 = bbox1[:, :2] + bbox1[:, 2:] / 2
        center2 = bbox2[:, :2] + bbox2[:, 2:] / 2
        return np.sqrt(np.sum((center1 - center2)**2, axis=1))

    baseline_center_offset = compute_center_offset(rf_bboxes, gt_bboxes)
    model_center_offset = compute_center_offset(corrected_bboxes, gt_bboxes)

    print(f"\n2. CENTER OFFSET (px):")
    print(f"  Baseline: {baseline_center_offset.mean():.2f}px")
    print(f"  Model:    {model_center_offset.mean():.2f}px")
    print(f"  Improvement: {(baseline_center_offset.mean() - model_center_offset.mean()) / baseline_center_offset.mean() * 100:.1f}%")

    # === 3. BBOX IoU ===
    def compute_iou(bbox1, bbox2):
        x1_max = np.maximum(bbox1[:, 0], bbox2[:, 0])
        y1_max = np.maximum(bbox1[:, 1], bbox2[:, 1])
        x2_min = np.minimum(bbox1[:, 0] + bbox1[:, 2], bbox2[:, 0] + bbox2[:, 2])
        y2_min = np.minimum(bbox1[:, 1] + bbox1[:, 3], bbox2[:, 1] + bbox2[:, 3])

        inter_area = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)
        bbox1_area = bbox1[:, 2] * bbox1[:, 3]
        bbox2_area = bbox2[:, 2] * bbox2[:, 3]
        union_area = bbox1_area + bbox2_area - inter_area

        iou = inter_area / np.maximum(union_area, 1e-10)
        return iou

    baseline_iou = compute_iou(rf_bboxes, gt_bboxes)
    model_iou = compute_iou(corrected_bboxes, gt_bboxes)

    print(f"\n3. BBOX IoU:")
    print(f"  Baseline: {baseline_iou.mean():.3f}")
    print(f"  Model:    {model_iou.mean():.3f}")
    print(f"  Improvement: {(model_iou.mean() - baseline_iou.mean()) / baseline_iou.mean() * 100:.1f}%")

    # === GOALS CHECK ===
    print(f"\n" + "="*80)
    print("GOAL ACHIEVEMENT:")
    print("="*80)

    pct_under_3 = (model_scale_errors < 3).mean() * 100
    pct_under_5 = (model_scale_errors < 5).mean() * 100

    goal_90_under_3 = pct_under_3 >= 90
    goal_99_under_5 = pct_under_5 >= 99
    goal_iou_70 = model_iou.mean() >= 0.7

    print(f"  Init scale <3% on 90% of frames: {pct_under_3:.1f}% {'✅' if goal_90_under_3 else '❌'}")
    print(f"  Init scale <5% on 99% of frames: {pct_under_5:.1f}% {'✅' if goal_99_under_5 else '❌'}")
    print(f"  Bbox IoU >0.7 average:           {model_iou.mean():.3f} {'✅' if goal_iou_70 else '❌'}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Init scale error histogram
    axes[0, 0].hist(baseline_scale_errors, bins=50, alpha=0.5, label='Baseline', color='red')
    axes[0, 0].hist(model_scale_errors, bins=50, alpha=0.5, label='V2 Model', color='green')
    axes[0, 0].axvline(3, color='black', linestyle='--', linewidth=2, label='3% threshold')
    axes[0, 0].axvline(5, color='gray', linestyle='--', linewidth=2, label='5% threshold')
    axes[0, 0].set_xlabel('Init Scale Error (%)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Initialization Scale Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Bbox IoU histogram
    axes[0, 1].hist(baseline_iou, bins=50, alpha=0.5, label='Baseline', color='red')
    axes[0, 1].hist(model_iou, bins=50, alpha=0.5, label='V2 Model', color='green')
    axes[0, 1].axvline(0.7, color='black', linestyle='--', linewidth=2, label='0.7 threshold')
    axes[0, 1].set_xlabel('Bbox IoU')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Bbox IoU Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Scale error: Baseline vs Model
    axes[1, 0].scatter(baseline_scale_errors, model_scale_errors, alpha=0.3)
    axes[1, 0].plot([0, baseline_scale_errors.max()], [0, baseline_scale_errors.max()], 'r--', label='No improvement')
    axes[1, 0].axhline(3, color='green', linestyle='--', label='3% goal')
    axes[1, 0].set_xlabel('Baseline Scale Error (%)')
    axes[1, 0].set_ylabel('Model Scale Error (%)')
    axes[1, 0].set_title('Per-Sample Init Scale Error Improvement')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. Center offset improvement
    axes[1, 1].scatter(baseline_center_offset, model_center_offset, alpha=0.3)
    axes[1, 1].plot([0, baseline_center_offset.max()], [0, baseline_center_offset.max()], 'r--', label='No improvement')
    axes[1, 1].set_xlabel('Baseline Center Offset (px)')
    axes[1, 1].set_ylabel('Model Center Offset (px)')
    axes[1, 1].set_title('Per-Sample Center Offset Improvement')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'model_v2_evaluation.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\n✓ Saved evaluation plot: {plot_path}")

    return {
        'init_scale_error': {
            'baseline_mean': float(baseline_scale_errors.mean()),
            'baseline_median': float(np.median(baseline_scale_errors)),
            'model_mean': float(model_scale_errors.mean()),
            'model_median': float(np.median(model_scale_errors)),
            'improvement_percent': float(scale_improvement),
            'pct_under_3': float(pct_under_3),
            'pct_under_5': float(pct_under_5),
        },
        'center_offset': {
            'baseline_mean': float(baseline_center_offset.mean()),
            'model_mean': float(model_center_offset.mean()),
        },
        'bbox_iou': {
            'baseline_mean': float(baseline_iou.mean()),
            'model_mean': float(model_iou.mean()),
        },
        'goals_achieved': {
            'init_scale_under_3_on_90pct': bool(goal_90_under_3),
            'init_scale_under_5_on_99pct': bool(goal_99_under_5),
            'bbox_iou_over_0.7': bool(goal_iou_70),
        }
    }


def main():
    """Main training pipeline."""
    print("="*80)
    print("TIER 2 BBOX CORRECTION MODEL TRAINING - V2 (REFORMULATED)")
    print("="*80)
    print("\nKey Change: Predict scale_ratio directly instead of width/height!")
    print("Expected: Massive improvement in init_scale error\n")

    # Load dataset
    dataset_path = Path("bbox_dataset/bbox_dataset.csv")
    df = load_dataset(dataset_path)

    # Extract features and targets
    print("\nExtracting features (V2)...")
    X, y = extract_features(df)
    print(f"Features: {list(X.columns)}")
    print(f"Targets (NEW!):  {list(y.columns)}")

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
    output_dir = Path("bbox_correction_model_v2")
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
        'overall_metrics': overall_metrics,
        'version': 'v2',
        'formulation': 'scale_ratio + aspect_ratio_adjustment'
    }

    model_path = output_dir / "bbox_correction_model_v2.pkl"
    joblib.dump(model_data, model_path)
    print(f"✓ Saved model: {model_path}")

    # Save metrics as JSON
    metrics_path = output_dir / "model_v2_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({'per_target': metrics, 'overall': overall_metrics}, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")

    print("\n" + "="*80)
    print("✅ MODEL V2 TRAINING COMPLETE")
    print("="*80)

    # Summary
    print(f"\nInit Scale Error:")
    print(f"  Baseline: {overall_metrics['init_scale_error']['baseline_mean']:.2f}%")
    print(f"  V2 Model: {overall_metrics['init_scale_error']['model_mean']:.2f}%")
    print(f"  Improvement: {overall_metrics['init_scale_error']['improvement_percent']:.1f}%")

    print(f"\nGoal Achievement:")
    for goal_name, achieved in overall_metrics['goals_achieved'].items():
        print(f"  {goal_name}: {'✅' if achieved else '❌'}")

    print(f"\nNext step: Test V2 model on diverse frames including paralysis cases")


if __name__ == "__main__":
    main()
