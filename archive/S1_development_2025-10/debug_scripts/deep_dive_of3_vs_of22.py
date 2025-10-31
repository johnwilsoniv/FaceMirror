#!/usr/bin/env python3
"""
Deep dive into OF3 vs OF2.2 differences
Investigating WHY correlations are so low
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("="*80)
print("Deep Dive: OpenFace 3.0 vs 2.2 AU Differences")
print("="*80)

# Load files
of3_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv')
of22_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP22.csv')

# Focus on left side for detailed analysis
print(f"\nAnalyzing LEFT side: {len(of3_left)} frames")

# Get common AUs
common_aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU12_r', 'AU20_r', 'AU45_r']

print("\n" + "="*80)
print("1. VALUE RANGE ANALYSIS")
print("="*80)

for au in common_aus:
    of3_vals = of3_left[au].dropna()
    of22_vals = of22_left[au].dropna()

    print(f"\n{au}:")
    print(f"  OpenFace 3.0: min={of3_vals.min():.4f}, max={of3_vals.max():.4f}, mean={of3_vals.mean():.4f}, std={of3_vals.std():.4f}")
    print(f"  OpenFace 2.2: min={of22_vals.min():.4f}, max={of22_vals.max():.4f}, mean={of22_vals.mean():.4f}, std={of22_vals.std():.4f}")
    print(f"  Ratio (OF3/OF2.2): {of3_vals.mean() / (of22_vals.mean() + 1e-10):.2f}x")

print("\n" + "="*80)
print("2. TEMPORAL PATTERN ANALYSIS")
print("="*80)

# Check if temporal patterns match (even if values don't)
for au in common_aus:
    of3_vals = of3_left[au].values
    of22_vals = of22_left[au].values

    # Normalize both to [0, 1] to remove scaling effects
    of3_norm = (of3_vals - np.nanmin(of3_vals)) / (np.nanmax(of3_vals) - np.nanmin(of3_vals) + 1e-10)
    of22_norm = (of22_vals - np.nanmin(of22_vals)) / (np.nanmax(of22_vals) - np.nanmin(of22_vals) + 1e-10)

    # Remove NaN
    mask = ~(np.isnan(of3_norm) | np.isnan(of22_norm))

    if mask.sum() > 1:
        corr, _ = stats.pearsonr(of3_norm[mask], of22_norm[mask])
        print(f"\n{au}: Correlation after normalization = {corr:.4f}")
        if corr > 0.5:
            print(f"  ✓ Temporal patterns MATCH well")
        elif corr > 0.3:
            print(f"  ⚠ Temporal patterns somewhat similar")
        else:
            print(f"  ✗ Temporal patterns DIVERGE")

print("\n" + "="*80)
print("3. ACTIVATION FREQUENCY ANALYSIS")
print("="*80)

# Check how often each AU is "active" (> threshold)
for au in common_aus:
    of3_vals = of3_left[au].dropna()
    of22_vals = of22_left[au].dropna()

    # Define "active" as > mean + 0.5*std (arbitrary but consistent)
    of3_threshold = of3_vals.mean() + 0.5 * of3_vals.std()
    of22_threshold = of22_vals.mean() + 0.5 * of22_vals.std()

    of3_active = (of3_vals > of3_threshold).sum()
    of22_active = (of22_vals > of22_threshold).sum()

    of3_pct = 100 * of3_active / len(of3_vals)
    of22_pct = 100 * of22_active / len(of22_vals)

    print(f"\n{au}:")
    print(f"  OF3: {of3_pct:.1f}% of frames active (threshold: {of3_threshold:.4f})")
    print(f"  OF2.2: {of22_pct:.1f}% of frames active (threshold: {of22_threshold:.4f})")

    if abs(of3_pct - of22_pct) < 10:
        print(f"  ✓ Similar activation frequency")
    elif abs(of3_pct - of22_pct) < 25:
        print(f"  ⚠ Moderately different activation")
    else:
        print(f"  ✗ Very different activation patterns")

print("\n" + "="*80)
print("4. PEAK DETECTION ANALYSIS")
print("="*80)

# Do both versions detect the same peaks/events?
for au in common_aus:
    of3_vals = of3_left[au].values
    of22_vals = of22_left[au].values

    # Find peaks (local maxima above threshold)
    def find_peaks(vals, threshold_factor=1.0):
        threshold = np.nanmean(vals) + threshold_factor * np.nanstd(vals)
        peaks = []
        for i in range(1, len(vals)-1):
            if np.isnan(vals[i]):
                continue
            if vals[i] > threshold and vals[i] > vals[i-1] and vals[i] > vals[i+1]:
                peaks.append(i)
        return peaks

    of3_peaks = find_peaks(of3_vals)
    of22_peaks = find_peaks(of22_vals)

    # Find overlapping peaks (within 5 frames)
    overlap_count = 0
    for of3_peak in of3_peaks:
        if any(abs(of3_peak - of22_peak) <= 5 for of22_peak in of22_peaks):
            overlap_count += 1

    if len(of3_peaks) > 0:
        overlap_pct = 100 * overlap_count / len(of3_peaks)
    else:
        overlap_pct = 0

    print(f"\n{au}:")
    print(f"  OF3 detected {len(of3_peaks)} peaks")
    print(f"  OF2.2 detected {len(of22_peaks)} peaks")
    print(f"  Overlap: {overlap_count} peaks ({overlap_pct:.1f}% of OF3 peaks)")

    if overlap_pct > 70:
        print(f"  ✓ Both detect similar events")
    elif overlap_pct > 40:
        print(f"  ⚠ Partial agreement on events")
    else:
        print(f"  ✗ Detect different events")

print("\n" + "="*80)
print("5. VISUALIZATION OF KEY AUs")
print("="*80)

# Create comparison plots
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('OpenFace 3.0 vs 2.2 - Temporal Comparison', fontsize=16)

for idx, au in enumerate(['AU01_r', 'AU12_r', 'AU20_r', 'AU45_r']):
    if idx >= len(axes.flatten()):
        break

    ax = axes.flatten()[idx]

    # Plot raw values
    ax.plot(of3_left['frame'], of3_left[au], label='OF 3.0', alpha=0.7, linewidth=2)
    ax.plot(of22_left['frame'], of22_left[au], label='OF 2.2', alpha=0.7, linewidth=2, linestyle='--')
    ax.set_xlabel('Frame')
    ax.set_ylabel(f'{au} Intensity')
    ax.set_title(f'{au} - Raw Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Save plot
plt.tight_layout()
output_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of3_vs_of22_temporal.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")
plt.close()

print("\n" + "="*80)
print("6. DIAGNOSIS")
print("="*80)

print("\nPossible reasons for low correlation:")
print("  1. Different training datasets (different ground truth labels)")
print("  2. Different AU intensity scales (OF2.2: 0-5, OF3: 0-?)")
print("  3. Different model sensitivities (OF3 may be more/less sensitive)")
print("  4. Different facial landmark inputs (OF3 uses 98pts, OF2.2 uses 68pts)")
print("  5. Fundamental algorithmic differences (SVM vs Deep Learning)")

print("\nTo investigate further:")
print("  - Check if OF3 outputs are in a different scale")
print("  - Verify both are using the same FACS AU definitions")
print("  - Look at OF3 documentation for expected AU ranges")
print("  - Test on videos with known/labeled AUs")

print("\n" + "="*80)
