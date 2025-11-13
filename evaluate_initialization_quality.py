#!/usr/bin/env python3
"""
Evaluate Bbox Correction Model - Initialization Quality Focus

Evaluates Tier 2 model performance using metrics that directly impact CLNF initialization:
1. Init Scale Error (%) - Most important for CLNF convergence
2. Bbox IoU - Standard metric for bbox quality
3. Component errors (x, y, w, h) - For debugging

Goal Thresholds:
- Init Scale Error: <3% on 90% of frames, <5% on 99% of frames
- Bbox IoU: >0.7 average, >0.6 on 90% of frames
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import joblib


def compute_init_scale_error(pred_bbox, gt_bbox):
    """
    Compute initialization scale error (%).

    Init scale is what CLNF uses for PDM initialization:
    scale = sqrt(bbox_width * bbox_height)

    Args:
        pred_bbox: (x, y, w, h)
        gt_bbox: (x, y, w, h)

    Returns:
        scale_error_pct: Percentage error in scale
    """
    pred_w, pred_h = pred_bbox[2], pred_bbox[3]
    gt_w, gt_h = gt_bbox[2], gt_bbox[3]

    pred_scale = np.sqrt(pred_w * pred_h)
    gt_scale = np.sqrt(gt_w * gt_h)

    scale_error_pct = abs(pred_scale - gt_scale) / gt_scale * 100

    return scale_error_pct


def compute_iou(pred_bbox, gt_bbox):
    """
    Compute Intersection over Union (IoU) for bboxes.

    Args:
        pred_bbox: (x, y, w, h)
        gt_bbox: (x, y, w, h)

    Returns:
        iou: IoU score [0, 1]
    """
    # Convert to (x1, y1, x2, y2)
    pred_x1, pred_y1 = pred_bbox[0], pred_bbox[1]
    pred_x2, pred_y2 = pred_x1 + pred_bbox[2], pred_y1 + pred_bbox[3]

    gt_x1, gt_y1 = gt_bbox[0], gt_bbox[1]
    gt_x2, gt_y2 = gt_x1 + gt_bbox[2], gt_y1 + gt_bbox[3]

    # Intersection
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    pred_area = pred_bbox[2] * pred_bbox[3]
    gt_area = gt_bbox[2] * gt_bbox[3]
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def evaluate_model(df, X_test, y_test, model_data):
    """
    Comprehensive evaluation focusing on initialization quality.
    """
    print("\n" + "="*80)
    print("INITIALIZATION QUALITY EVALUATION")
    print("="*80)

    # Get test indices
    test_indices = X_test.index

    # Scale features
    scaler = model_data['scaler']
    X_test_scaled = scaler.transform(X_test)

    # Predict corrections
    models = model_data['models']
    pred_center_offset_x = models['center_offset_x'].predict(X_test_scaled)
    pred_center_offset_y = models['center_offset_y'].predict(X_test_scaled)
    pred_width_correction = models['width_correction'].predict(X_test_scaled)
    pred_height_correction = models['height_correction'].predict(X_test_scaled)

    # Build predicted bboxes
    results = []

    for i, idx in enumerate(test_indices):
        # Ground truth (C++ MTCNN)
        gt_bbox = (
            df.loc[idx, 'cpp_bbox_x'],
            df.loc[idx, 'cpp_bbox_y'],
            df.loc[idx, 'cpp_bbox_w'],
            df.loc[idx, 'cpp_bbox_h']
        )

        # RetinaFace raw
        rf_bbox = (
            df.loc[idx, 'rf_bbox_x'],
            df.loc[idx, 'rf_bbox_y'],
            df.loc[idx, 'rf_bbox_w'],
            df.loc[idx, 'rf_bbox_h']
        )

        # Model prediction (apply corrections to RetinaFace)
        pred_center_x = df.loc[idx, 'rf_center_x'] + pred_center_offset_x[i]
        pred_center_y = df.loc[idx, 'rf_center_y'] + pred_center_offset_y[i]
        pred_w = df.loc[idx, 'rf_bbox_w'] + pred_width_correction[i]
        pred_h = df.loc[idx, 'rf_bbox_h'] + pred_height_correction[i]

        pred_bbox = (
            pred_center_x - pred_w / 2,
            pred_center_y - pred_h / 2,
            pred_w,
            pred_h
        )

        # Compute metrics
        # Baseline (RetinaFace raw)
        baseline_scale_error = compute_init_scale_error(rf_bbox, gt_bbox)
        baseline_iou = compute_iou(rf_bbox, gt_bbox)

        # Model prediction
        model_scale_error = compute_init_scale_error(pred_bbox, gt_bbox)
        model_iou = compute_iou(pred_bbox, gt_bbox)

        # Center offset (Euclidean)
        baseline_center_offset = np.sqrt(
            (df.loc[idx, 'rf_center_x'] - df.loc[idx, 'cpp_center_x'])**2 +
            (df.loc[idx, 'rf_center_y'] - df.loc[idx, 'cpp_center_y'])**2
        )
        model_center_offset = np.sqrt(
            (pred_center_x - df.loc[idx, 'cpp_center_x'])**2 +
            (pred_center_y - df.loc[idx, 'cpp_center_y'])**2
        )

        results.append({
            'patient_id': df.loc[idx, 'patient_id'],
            'video_name': df.loc[idx, 'video_name'],
            'frame_idx': df.loc[idx, 'frame_idx'],
            'face_size': df.loc[idx, 'cpp_size'],

            # Baseline (RetinaFace raw)
            'baseline_scale_error': baseline_scale_error,
            'baseline_iou': baseline_iou,
            'baseline_center_offset': baseline_center_offset,

            # Model prediction
            'model_scale_error': model_scale_error,
            'model_iou': model_iou,
            'model_center_offset': model_center_offset,

            # Component errors (for debugging)
            'pred_width_error': abs(pred_w - gt_bbox[2]),
            'pred_height_error': abs(pred_h - gt_bbox[3]),
        })

    results_df = pd.DataFrame(results)

    # Print statistics
    print("\n" + "-"*80)
    print("INIT SCALE ERROR (%)")
    print("-"*80)

    print("\nBaseline (RetinaFace raw):")
    print(f"  Mean:   {results_df['baseline_scale_error'].mean():.2f}%")
    print(f"  Median: {results_df['baseline_scale_error'].median():.2f}%")
    print(f"  90th percentile: {results_df['baseline_scale_error'].quantile(0.90):.2f}%")
    print(f"  99th percentile: {results_df['baseline_scale_error'].quantile(0.99):.2f}%")
    print(f"  Max:    {results_df['baseline_scale_error'].max():.2f}%")

    baseline_under_3pct = (results_df['baseline_scale_error'] < 3.0).sum() / len(results_df) * 100
    baseline_under_5pct = (results_df['baseline_scale_error'] < 5.0).sum() / len(results_df) * 100
    print(f"  <3%:    {baseline_under_3pct:.1f}% of frames")
    print(f"  <5%:    {baseline_under_5pct:.1f}% of frames")

    print("\nTier 2 Model:")
    print(f"  Mean:   {results_df['model_scale_error'].mean():.2f}%")
    print(f"  Median: {results_df['model_scale_error'].median():.2f}%")
    print(f"  90th percentile: {results_df['model_scale_error'].quantile(0.90):.2f}%")
    print(f"  99th percentile: {results_df['model_scale_error'].quantile(0.99):.2f}%")
    print(f"  Max:    {results_df['model_scale_error'].max():.2f}%")

    model_under_3pct = (results_df['model_scale_error'] < 3.0).sum() / len(results_df) * 100
    model_under_5pct = (results_df['model_scale_error'] < 5.0).sum() / len(results_df) * 100
    print(f"  <3%:    {model_under_3pct:.1f}% of frames")
    print(f"  <5%:    {model_under_5pct:.1f}% of frames")

    scale_improvement = (results_df['baseline_scale_error'].mean() - results_df['model_scale_error'].mean()) / results_df['baseline_scale_error'].mean() * 100
    print(f"\n  Improvement: {scale_improvement:.1f}%")

    # Goal check
    print("\n  Goal Check:")
    if model_under_3pct >= 90.0:
        print(f"    ✓ <3% on ≥90% frames: {model_under_3pct:.1f}% (PASS)")
    else:
        print(f"    ✗ <3% on ≥90% frames: {model_under_3pct:.1f}% (FAIL - need {90.0 - model_under_3pct:.1f}% more)")

    if model_under_5pct >= 99.0:
        print(f"    ✓ <5% on ≥99% frames: {model_under_5pct:.1f}% (PASS)")
    else:
        print(f"    ✗ <5% on ≥99% frames: {model_under_5pct:.1f}% (FAIL - need {99.0 - model_under_5pct:.1f}% more)")

    print("\n" + "-"*80)
    print("BBOX IoU")
    print("-"*80)

    print("\nBaseline (RetinaFace raw):")
    print(f"  Mean:   {results_df['baseline_iou'].mean():.3f}")
    print(f"  Median: {results_df['baseline_iou'].median():.3f}")
    print(f"  10th percentile: {results_df['baseline_iou'].quantile(0.10):.3f}")

    baseline_above_06 = (results_df['baseline_iou'] > 0.6).sum() / len(results_df) * 100
    baseline_above_07 = (results_df['baseline_iou'] > 0.7).sum() / len(results_df) * 100
    print(f"  >0.6:   {baseline_above_06:.1f}% of frames")
    print(f"  >0.7:   {baseline_above_07:.1f}% of frames")

    print("\nTier 2 Model:")
    print(f"  Mean:   {results_df['model_iou'].mean():.3f}")
    print(f"  Median: {results_df['model_iou'].median():.3f}")
    print(f"  10th percentile: {results_df['model_iou'].quantile(0.10):.3f}")

    model_above_06 = (results_df['model_iou'] > 0.6).sum() / len(results_df) * 100
    model_above_07 = (results_df['model_iou'] > 0.7).sum() / len(results_df) * 100
    print(f"  >0.6:   {model_above_06:.1f}% of frames")
    print(f"  >0.7:   {model_above_07:.1f}% of frames")

    iou_improvement = (results_df['model_iou'].mean() - results_df['baseline_iou'].mean()) / results_df['baseline_iou'].mean() * 100
    print(f"\n  Improvement: {iou_improvement:.1f}%")

    # Goal check
    print("\n  Goal Check:")
    if results_df['model_iou'].mean() >= 0.7:
        print(f"    ✓ Mean IoU ≥0.7: {results_df['model_iou'].mean():.3f} (PASS)")
    else:
        print(f"    ✗ Mean IoU ≥0.7: {results_df['model_iou'].mean():.3f} (FAIL)")

    if model_above_06 >= 90.0:
        print(f"    ✓ >0.6 on ≥90% frames: {model_above_06:.1f}% (PASS)")
    else:
        print(f"    ✗ >0.6 on ≥90% frames: {model_above_06:.1f}% (FAIL)")

    print("\n" + "-"*80)
    print("CENTER OFFSET (px)")
    print("-"*80)

    print("\nBaseline (RetinaFace raw):")
    print(f"  Mean:   {results_df['baseline_center_offset'].mean():.2f}px")
    print(f"  Median: {results_df['baseline_center_offset'].median():.2f}px")

    print("\nTier 2 Model:")
    print(f"  Mean:   {results_df['model_center_offset'].mean():.2f}px")
    print(f"  Median: {results_df['model_center_offset'].median():.2f}px")

    center_improvement = (results_df['baseline_center_offset'].mean() - results_df['model_center_offset'].mean()) / results_df['baseline_center_offset'].mean() * 100
    print(f"\n  Improvement: {center_improvement:.1f}%")

    # Identify worst cases
    print("\n" + "-"*80)
    print("WORST CASES (Top 10 by init scale error)")
    print("-"*80)

    worst_cases = results_df.nlargest(10, 'model_scale_error')
    for i, row in worst_cases.iterrows():
        print(f"\n  Patient {row['patient_id']:3d} | {row['video_name'][:20]:20s} | Frame {row['frame_idx']:4d}")
        print(f"    Face size: {row['face_size']:.1f}px")
        print(f"    Scale error: {row['model_scale_error']:.2f}%  (baseline: {row['baseline_scale_error']:.2f}%)")
        print(f"    IoU: {row['model_iou']:.3f}  (baseline: {row['baseline_iou']:.3f})")
        print(f"    Center offset: {row['model_center_offset']:.1f}px  (baseline: {row['baseline_center_offset']:.1f}px)")

    # Create visualizations
    output_dir = Path("bbox_correction_model")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Init scale error distribution
    axes[0, 0].hist(results_df['baseline_scale_error'], bins=50, alpha=0.5,
                    label='Baseline (RetinaFace raw)', color='red')
    axes[0, 0].hist(results_df['model_scale_error'], bins=50, alpha=0.5,
                    label='Tier 2 Model', color='green')
    axes[0, 0].axvline(3.0, color='black', linestyle='--', linewidth=2, label='3% threshold')
    axes[0, 0].axvline(5.0, color='gray', linestyle='--', linewidth=2, label='5% threshold')
    axes[0, 0].set_xlabel('Init Scale Error (%)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Initialization Scale Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: IoU distribution
    axes[0, 1].hist(results_df['baseline_iou'], bins=50, alpha=0.5,
                    label='Baseline', color='red')
    axes[0, 1].hist(results_df['model_iou'], bins=50, alpha=0.5,
                    label='Tier 2 Model', color='green')
    axes[0, 1].axvline(0.7, color='black', linestyle='--', linewidth=2, label='0.7 threshold')
    axes[0, 1].set_xlabel('Bbox IoU')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Bbox IoU Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Scale error vs face size
    axes[1, 0].scatter(results_df['face_size'], results_df['baseline_scale_error'],
                       alpha=0.3, s=10, label='Baseline', color='red')
    axes[1, 0].scatter(results_df['face_size'], results_df['model_scale_error'],
                       alpha=0.3, s=10, label='Tier 2 Model', color='green')
    axes[1, 0].axhline(3.0, color='black', linestyle='--', linewidth=2, label='3% threshold')
    axes[1, 0].axhline(5.0, color='gray', linestyle='--', linewidth=2, label='5% threshold')
    axes[1, 0].set_xlabel('Face Size (px)')
    axes[1, 0].set_ylabel('Init Scale Error (%)')
    axes[1, 0].set_title('Scale Error vs Face Size')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Baseline vs Model (scatter)
    axes[1, 1].scatter(results_df['baseline_scale_error'], results_df['model_scale_error'], alpha=0.3)
    max_val = max(results_df['baseline_scale_error'].max(), results_df['model_scale_error'].max())
    axes[1, 1].plot([0, max_val], [0, max_val], 'r--', label='No improvement line')
    axes[1, 1].axhline(3.0, color='green', linestyle='--', alpha=0.5, label='3% goal')
    axes[1, 1].axvline(3.0, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Baseline Scale Error (%)')
    axes[1, 1].set_ylabel('Model Scale Error (%)')
    axes[1, 1].set_title('Per-Sample Improvement')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'initialization_quality_evaluation.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\n✓ Saved evaluation plot: {plot_path}")

    # Save detailed results
    results_csv_path = output_dir / 'initialization_quality_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Saved detailed results: {results_csv_path}")

    # Summary metrics
    summary = {
        'scale_error': {
            'baseline_mean': float(results_df['baseline_scale_error'].mean()),
            'model_mean': float(results_df['model_scale_error'].mean()),
            'improvement_pct': float(scale_improvement),
            'model_under_3pct': float(model_under_3pct),
            'model_under_5pct': float(model_under_5pct),
            'goal_3pct_90frames': model_under_3pct >= 90.0,
            'goal_5pct_99frames': model_under_5pct >= 99.0,
        },
        'iou': {
            'baseline_mean': float(results_df['baseline_iou'].mean()),
            'model_mean': float(results_df['model_iou'].mean()),
            'improvement_pct': float(iou_improvement),
            'model_above_06pct': float(model_above_06),
            'model_above_07pct': float(model_above_07),
            'goal_mean_07': results_df['model_iou'].mean() >= 0.7,
            'goal_06_90frames': model_above_06 >= 90.0,
        },
        'center_offset': {
            'baseline_mean': float(results_df['baseline_center_offset'].mean()),
            'model_mean': float(results_df['model_center_offset'].mean()),
            'improvement_pct': float(center_improvement),
        }
    }

    return summary


def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("BBOX CORRECTION MODEL - INITIALIZATION QUALITY EVALUATION")
    print("="*80)

    # Load dataset
    dataset_path = Path("bbox_dataset/bbox_dataset.csv")
    df = pd.read_csv(dataset_path)
    print(f"\nLoaded {len(df)} samples from {dataset_path}")

    # Extract features (same as training)
    X = pd.DataFrame({
        'rf_size': df['rf_size'],
        'rf_aspect_ratio': df['rf_aspect_ratio'],
        'rf_center_x_norm': df['rf_center_x'] / df['image_width'],
        'rf_center_y_norm': df['rf_center_y'] / df['image_height'],
        'image_width': df['image_width'],
        'image_height': df['image_height'],
        'image_aspect_ratio': df['image_width'] / df['image_height'],
    })

    y = pd.DataFrame({
        'center_offset_x': df['center_offset_x'],
        'center_offset_y': df['center_offset_y'],
        'width_correction': df['cpp_bbox_w'] - df['rf_bbox_w'],
        'height_correction': df['cpp_bbox_h'] - df['rf_bbox_h'],
    })

    # Load trained model
    model_path = Path("bbox_correction_model/bbox_correction_model.pkl")
    model_data = joblib.load(model_path)
    print(f"Loaded model: {model_path}")

    # Get test set (same split as training)
    from sklearn.model_selection import train_test_split
    unique_patients = df['patient_id'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )

    test_mask = df['patient_id'].isin(test_patients)
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Test set: {len(X_test)} samples ({len(test_patients)} patients)")

    # Evaluate
    summary = evaluate_model(df, X_test, y_test, model_data)

    # Save summary
    summary_path = Path("bbox_correction_model/initialization_quality_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")

    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    goals_met = 0
    goals_total = 4

    if summary['scale_error']['goal_3pct_90frames']:
        goals_met += 1
    if summary['scale_error']['goal_5pct_99frames']:
        goals_met += 1
    if summary['iou']['goal_mean_07']:
        goals_met += 1
    if summary['iou']['goal_06_90frames']:
        goals_met += 1

    print(f"\nGoals achieved: {goals_met}/{goals_total}")

    if goals_met == goals_total:
        print("\n✅ TIER 2 MODEL IS SUFFICIENT")
        print("   All initialization quality goals met!")
        print("   → Ready for production deployment")
    elif goals_met >= 2:
        print("\n⚠️  TIER 2 MODEL IS DECENT")
        print(f"   {goals_met}/{goals_total} goals met, but could be better")
        print("   → Consider Tier 3 (polynomial features) for improvement")
    else:
        print("\n❌ TIER 2 MODEL NEEDS IMPROVEMENT")
        print(f"   Only {goals_met}/{goals_total} goals met")
        print("   → Proceed to Tier 3 (polynomial features)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
