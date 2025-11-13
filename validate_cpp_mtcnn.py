#!/usr/bin/env python3
"""
Validate C++ MTCNN Detector Against Ground Truth

Compares Python implementation using exact C++ weights to C++ OpenFace ground truth.
This should give us init_scale error ≈ 0% if implementation is correct.
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from cpp_mtcnn_detector import CPPMTCNNDetector
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_bbox_metrics(pred_bbox, gt_bbox):
    """Calculate metrics between predicted and ground truth bbox."""
    pred_x, pred_y, pred_w, pred_h = pred_bbox
    gt_x, gt_y, gt_w, gt_h = gt_bbox

    # Scale (sqrt of area)
    pred_scale = np.sqrt(pred_w * pred_h)
    gt_scale = np.sqrt(gt_w * gt_h)

    # Init scale error (what we're trying to minimize)
    init_scale_error = abs(pred_scale - gt_scale) / gt_scale * 100

    # Center offset
    pred_cx = pred_x + pred_w / 2
    pred_cy = pred_y + pred_h / 2
    gt_cx = gt_x + gt_w / 2
    gt_cy = gt_y + gt_h / 2
    center_offset = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)

    # Size differences
    width_diff = abs(pred_w - gt_w)
    height_diff = abs(pred_h - gt_h)

    return {
        'pred_scale': pred_scale,
        'gt_scale': gt_scale,
        'init_scale_error': init_scale_error,
        'center_offset': center_offset,
        'width_diff': width_diff,
        'height_diff': height_diff
    }


def main():
    print("="*80)
    print("C++ MTCNN DETECTOR VALIDATION")
    print("="*80)

    # Load bbox dataset
    print("\nLoading bbox dataset...")
    df = pd.read_csv("bbox_dataset/bbox_dataset.csv")
    print(f"  Loaded {len(df)} frames from {df['patient_id'].nunique()} patients")

    # Initialize detector
    print("\nInitializing C++ MTCNN detector...")
    detector = CPPMTCNNDetector()
    detector.min_face_size = 40  # Match C++ OpenFace default
    print("  ✓ Detector ready")

    # Validation results
    results = []
    failures = []

    # Sample subset for faster testing
    sample_size = min(50, len(df))  # Start with 50 frames
    df_sample = df.sample(n=sample_size, random_state=42)

    print(f"\nValidating on {sample_size} random frames...")

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        frame_path = row['frame_path']

        # Load image (already rotated by FFmpeg during dataset collection)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"  Warning: Could not load {frame_path}")
            failures.append({
                'frame_path': frame_path,
                'reason': 'Failed to load image'
            })
            continue

        # Run detector
        try:
            bboxes, landmarks = detector.detect(img)
        except Exception as e:
            print(f"  Warning: Detection failed for {frame_path}: {e}")
            failures.append({
                'frame_path': frame_path,
                'reason': f'Detection error: {e}'
            })
            continue

        # Check if we got a detection
        if len(bboxes) == 0:
            failures.append({
                'frame_path': frame_path,
                'reason': 'No face detected'
            })
            continue

        # Use first detection (should be highest confidence)
        pred_bbox = bboxes[0]
        gt_bbox = (row['cpp_bbox_x'], row['cpp_bbox_y'],
                   row['cpp_bbox_w'], row['cpp_bbox_h'])

        # Calculate metrics
        metrics = calculate_bbox_metrics(pred_bbox, gt_bbox)

        results.append({
            'patient_id': row['patient_id'],
            'frame_path': frame_path,
            'pred_x': pred_bbox[0],
            'pred_y': pred_bbox[1],
            'pred_w': pred_bbox[2],
            'pred_h': pred_bbox[3],
            'gt_x': gt_bbox[0],
            'gt_y': gt_bbox[1],
            'gt_w': gt_bbox[2],
            'gt_h': gt_bbox[3],
            **metrics
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"\nSuccessful detections: {len(results)}/{sample_size} ({len(results)/sample_size*100:.1f}%)")
    print(f"Failed detections: {len(failures)}")

    if len(results) > 0:
        print(f"\nInit Scale Error Statistics:")
        print(f"  Mean:   {results_df['init_scale_error'].mean():.2f}%")
        print(f"  Median: {results_df['init_scale_error'].median():.2f}%")
        print(f"  Std:    {results_df['init_scale_error'].std():.2f}%")
        print(f"  Min:    {results_df['init_scale_error'].min():.2f}%")
        print(f"  Max:    {results_df['init_scale_error'].max():.2f}%")

        # Percentage achieving goals
        below_1pct = (results_df['init_scale_error'] < 1.0).sum()
        below_3pct = (results_df['init_scale_error'] < 3.0).sum()
        below_5pct = (results_df['init_scale_error'] < 5.0).sum()

        print(f"\nPercentage achieving error thresholds:")
        print(f"  <1%:  {below_1pct}/{len(results)} ({below_1pct/len(results)*100:.1f}%)")
        print(f"  <3%:  {below_3pct}/{len(results)} ({below_3pct/len(results)*100:.1f}%)")
        print(f"  <5%:  {below_5pct}/{len(results)} ({below_5pct/len(results)*100:.1f}%)")

        print(f"\nBbox Alignment Metrics:")
        print(f"  Center offset:  {results_df['center_offset'].mean():.2f} ± {results_df['center_offset'].std():.2f} px")
        print(f"  Width diff:     {results_df['width_diff'].mean():.2f} ± {results_df['width_diff'].std():.2f} px")
        print(f"  Height diff:    {results_df['height_diff'].mean():.2f} ± {results_df['height_diff'].std():.2f} px")

        # Save results
        results_df.to_csv('cpp_mtcnn_validation_results.csv', index=False)
        print(f"\n✓ Saved detailed results to: cpp_mtcnn_validation_results.csv")

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Init scale error distribution
        axes[0, 0].hist(results_df['init_scale_error'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(results_df['init_scale_error'].mean(), color='red', linestyle='--',
                          label=f'Mean: {results_df["init_scale_error"].mean():.2f}%')
        axes[0, 0].set_xlabel('Init Scale Error (%)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Init Scale Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Predicted vs GT scale
        axes[0, 1].scatter(results_df['gt_scale'], results_df['pred_scale'], alpha=0.5)
        min_scale = min(results_df['gt_scale'].min(), results_df['pred_scale'].min())
        max_scale = max(results_df['gt_scale'].max(), results_df['pred_scale'].max())
        axes[0, 1].plot([min_scale, max_scale], [min_scale, max_scale], 'r--', label='Perfect match')
        axes[0, 1].set_xlabel('C++ OpenFace Scale (ground truth)')
        axes[0, 1].set_ylabel('Python C++ MTCNN Scale')
        axes[0, 1].set_title('Scale Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Center offset vs scale
        axes[1, 0].scatter(results_df['gt_scale'], results_df['center_offset'], alpha=0.5)
        axes[1, 0].set_xlabel('Face Scale')
        axes[1, 0].set_ylabel('Center Offset (pixels)')
        axes[1, 0].set_title('Bbox Center Alignment')
        axes[1, 0].grid(alpha=0.3)

        # Error by patient
        patient_errors = results_df.groupby('patient_id')['init_scale_error'].mean().sort_values()
        axes[1, 1].bar(range(len(patient_errors)), patient_errors.values)
        axes[1, 1].axhline(results_df['init_scale_error'].mean(), color='red', linestyle='--',
                          label='Overall mean')
        axes[1, 1].set_xlabel('Patient (sorted by error)')
        axes[1, 1].set_ylabel('Mean Init Scale Error (%)')
        axes[1, 1].set_title('Error by Patient')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('cpp_mtcnn_validation.png', dpi=150)
        print(f"✓ Saved visualization to: cpp_mtcnn_validation.png")

    if len(failures) > 0:
        print(f"\n⚠ Failures:")
        failure_reasons = {}
        for f in failures:
            reason = f['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
