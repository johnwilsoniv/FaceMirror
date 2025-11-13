#!/usr/bin/env python3
"""
Diagnose Why Scale Ratio is Hard to Predict

Investigates why scale_ratio has low R² (0.237) despite being the correct formulation.

Questions to answer:
1. What features actually correlate with scale_ratio?
2. Are there patient-specific patterns?
3. Are there outliers driving the error?
4. Is there a non-linear relationship?
5. Is the scale difference fundamentally noisy/unpredictable?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
df = pd.read_csv("bbox_dataset/bbox_dataset.csv")

# Compute scale_ratio
rf_scale = np.sqrt(df['rf_bbox_w'] * df['rf_bbox_h'])
cpp_scale = np.sqrt(df['cpp_bbox_w'] * df['cpp_bbox_h'])
scale_ratio = cpp_scale / rf_scale

df['rf_scale'] = rf_scale
df['cpp_scale'] = cpp_scale
df['scale_ratio'] = scale_ratio

# Features
features = {
    'rf_size': df['rf_size'],
    'rf_aspect_ratio': df['rf_aspect_ratio'],
    'rf_center_x_norm': df['rf_center_x'] / df['image_width'],
    'rf_center_y_norm': df['rf_center_y'] / df['image_height'],
    'image_width': df['image_width'],
    'image_height': df['image_height'],
    'image_aspect_ratio': df['image_width'] / df['image_height'],
}

print("="*80)
print("SCALE RATIO PREDICTABILITY ANALYSIS")
print("="*80)

# === 1. Basic statistics ===
print(f"\nScale Ratio Statistics:")
print(f"  Mean:   {scale_ratio.mean():.4f}")
print(f"  Median: {scale_ratio.median():.4f}")
print(f"  Std:    {scale_ratio.std():.4f}")
print(f"  Min:    {scale_ratio.min():.4f}")
print(f"  Max:    {scale_ratio.max():.4f}")
print(f"  Range:  {scale_ratio.max() - scale_ratio.min():.4f}")

# === 2. Correlation analysis ===
print(f"\nCorrelation with Scale Ratio:")
correlations = {}
for feat_name, feat_values in features.items():
    corr = stats.pearsonr(feat_values, scale_ratio)[0]
    correlations[feat_name] = corr
    print(f"  {feat_name:25s}: {corr:+.3f}")

# === 3. Patient-level analysis ===
print(f"\nPatient-Level Variance:")
patient_stats = df.groupby('patient_id')['scale_ratio'].agg(['mean', 'std', 'count'])
patient_stats = patient_stats[patient_stats['count'] >= 5]  # At least 5 frames

print(f"  Patients with >=5 frames: {len(patient_stats)}")
print(f"  Mean of patient means:    {patient_stats['mean'].mean():.4f}")
print(f"  Std of patient means:     {patient_stats['mean'].std():.4f}")
print(f"  Mean of patient stds:     {patient_stats['std'].mean():.4f}")

# Check if variance is more within-patient or between-patient
within_patient_var = patient_stats['std'].mean()**2
between_patient_var = patient_stats['mean'].std()**2
total_var = scale_ratio.std()**2

print(f"\n  Within-patient variance:  {within_patient_var:.6f}")
print(f"  Between-patient variance: {between_patient_var:.6f}")
print(f"  Total variance:           {total_var:.6f}")
print(f"  Explained by patient:     {between_patient_var / total_var * 100:.1f}%")

# === 4. Outlier analysis ===
print(f"\nOutlier Analysis:")
q1 = scale_ratio.quantile(0.25)
q3 = scale_ratio.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = (scale_ratio < lower_bound) | (scale_ratio > upper_bound)
print(f"  Outliers (IQR method): {outliers.sum()} ({outliers.mean()*100:.1f}%)")

if outliers.sum() > 0:
    print(f"\n  Top 5 outlier patients:")
    outlier_df = df[outliers].copy()
    outlier_counts = outlier_df['patient_id'].value_counts().head(5)
    for patient_id, count in outlier_counts.items():
        patient_data = df[df['patient_id'] == patient_id]
        print(f"    Patient {patient_id}: {count} outliers, scale_ratio range: {patient_data['scale_ratio'].min():.3f}-{patient_data['scale_ratio'].max():.3f}")

# === 5. Constant offset analysis ===
print(f"\nConstant Offset Analysis:")
mean_scale_ratio = scale_ratio.mean()
constant_offset_pred = np.full_like(scale_ratio, mean_scale_ratio)
constant_offset_mae = np.abs(scale_ratio - constant_offset_pred).mean()
print(f"  Constant offset (using mean): {mean_scale_ratio:.4f}")
print(f"  MAE with constant offset:     {constant_offset_mae:.4f}")
print(f"  V2 model MAE (from training): 0.1301")
print(f"  Improvement over constant:    {(constant_offset_mae - 0.1301) / constant_offset_mae * 100:.1f}%")

# === 6. Create visualizations ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1: Distribution
axes[0, 0].hist(scale_ratio, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(scale_ratio.mean(), color='red', linestyle='--', label=f'Mean: {scale_ratio.mean():.3f}')
axes[0, 0].axvline(scale_ratio.median(), color='green', linestyle='--', label=f'Median: {scale_ratio.median():.3f}')
axes[0, 0].set_xlabel('Scale Ratio (cpp_scale / rf_scale)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Scale Ratio Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 6.2: Strongest correlation
strongest_feat = max(correlations, key=lambda k: abs(correlations[k]))
axes[0, 1].scatter(features[strongest_feat], scale_ratio, alpha=0.3)
axes[0, 1].set_xlabel(strongest_feat)
axes[0, 1].set_ylabel('Scale Ratio')
axes[0, 1].set_title(f'Scale Ratio vs {strongest_feat} (r={correlations[strongest_feat]:+.3f})')
axes[0, 1].grid(alpha=0.3)

# 6.3: Patient-level variance
patient_means = patient_stats['mean'].sort_values()
axes[0, 2].bar(range(len(patient_means)), patient_means)
axes[0, 2].axhline(scale_ratio.mean(), color='red', linestyle='--', label='Overall mean')
axes[0, 2].set_xlabel('Patient (sorted by mean)')
axes[0, 2].set_ylabel('Mean Scale Ratio')
axes[0, 2].set_title('Patient-Level Mean Scale Ratios')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 6.4: rf_scale vs cpp_scale
axes[1, 0].scatter(rf_scale, cpp_scale, alpha=0.3)
axes[1, 0].plot([rf_scale.min(), rf_scale.max()], [rf_scale.min(), rf_scale.max()], 'r--', label='y=x (perfect match)')
axes[1, 0].set_xlabel('RetinaFace Scale')
axes[1, 0].set_ylabel('C++ MTCNN Scale')
axes[1, 0].set_title('Scale Comparison')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 6.5: Scale ratio by face size
axes[1, 1].scatter(rf_scale, scale_ratio, alpha=0.3)
axes[1, 1].axhline(1.0, color='red', linestyle='--', label='Ratio = 1.0 (same scale)')
axes[1, 1].set_xlabel('RetinaFace Scale')
axes[1, 1].set_ylabel('Scale Ratio')
axes[1, 1].set_title(f'Scale Ratio vs Face Size (r={correlations["rf_size"]:+.3f})')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 6.6: Residuals if we use constant offset
residuals = scale_ratio - mean_scale_ratio
axes[1, 2].scatter(rf_scale, residuals, alpha=0.3)
axes[1, 2].axhline(0, color='red', linestyle='--')
axes[1, 2].set_xlabel('RetinaFace Scale')
axes[1, 2].set_ylabel('Residual (actual - mean)')
axes[1, 2].set_title('Constant Offset Residuals')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scale_ratio_diagnostics.png', dpi=150)
print(f"\n✓ Saved diagnostic plots: scale_ratio_diagnostics.png")

# === 7. Check if V2 fixed correction would be better ===
print(f"\n" + "="*80)
print("COMPARISON: V2 Fixed Correction vs Adaptive Model")
print("="*80)

# V2 fixed correction parameters
V2_ALPHA = -0.01642482
V2_BETA = 0.23601291
V2_GAMMA = 0.99941800
V2_DELTA = 0.76624999

# Apply V2 fixed correction
v2_corrected_w = df['rf_bbox_w'] * V2_GAMMA
v2_corrected_h = df['rf_bbox_h'] * V2_DELTA
v2_corrected_scale = np.sqrt(v2_corrected_w * v2_corrected_h)

# Compute init scale error
v2_scale_error = np.abs(v2_corrected_scale - cpp_scale) / cpp_scale * 100

print(f"\nInit Scale Error:")
baseline_scale_error = (np.abs(rf_scale - cpp_scale) / cpp_scale * 100).mean()
print(f"  Baseline (RetinaFace raw):    {baseline_scale_error:.2f}%")
print(f"  V2 Fixed Correction:          {v2_scale_error.mean():.2f}%")
print(f"  V2 Adaptive Model:            ~12.24% (from training)")

print(f"\nPercentage achieving goals:")
print(f"  V2 Fixed - <3%:   {(v2_scale_error < 3).sum()}/{len(v2_scale_error)} ({(v2_scale_error < 3).mean()*100:.1f}%)")
print(f"  V2 Fixed - <5%:   {(v2_scale_error < 5).sum()}/{len(v2_scale_error)} ({(v2_scale_error < 5).mean()*100:.1f}%)")
print(f"  Adaptive - <3%:   28/230 (12.2%) [test set only]")
print(f"  Adaptive - <5%:   47/230 (20.4%) [test set only]")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"\nScale ratio is hard to predict because:")
print(f"  1. Low feature correlation (max r={max(correlations.values()):.3f})")
print(f"  2. Patient-specific variance explains only {between_patient_var / total_var * 100:.1f}% of total variance")
print(f"  3. Model barely beats constant offset ({(constant_offset_mae - 0.1301) / constant_offset_mae * 100:.1f}% improvement)")
print(f"\nRecommendation: Try polynomial features (Tier 3) or accept that scale is inherently noisy.")
