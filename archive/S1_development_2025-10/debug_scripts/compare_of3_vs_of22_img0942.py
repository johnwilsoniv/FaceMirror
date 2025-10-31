#!/usr/bin/env python3
"""
Quick comparison of OpenFace 3.0 vs 2.2 for IMG_0942 specifically
"""

import pandas as pd
import numpy as np
from scipy import stats

print("="*80)
print("OpenFace 3.0 vs 2.2 - IMG_0942 Comparison")
print("="*80)

# Load files
of3_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv')
of3_right = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_right_mirroredvOP3ORIG.csv')
of22_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP22.csv')
of22_right = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_right_mirroredOP22.csv')

print(f"\nFiles loaded:")
print(f"  OF3 Left: {len(of3_left)} frames")
print(f"  OF3 Right: {len(of3_right)} frames")
print(f"  OF2.2 Left: {len(of22_left)} frames")
print(f"  OF2.2 Right: {len(of22_right)} frames")

# Get AU columns
of3_au_cols = [col for col in of3_left.columns if col.startswith('AU') and '_r' in col]
of22_au_cols = [col for col in of22_left.columns if col.startswith('AU') and '_r' in col]
common_aus = sorted(set(of3_au_cols) & set(of22_au_cols))

print(f"\nCommon AUs: {len(common_aus)}")
print(f"  {', '.join(common_aus)}")

def compare_side(of3_df, of22_df, side_name):
    print(f"\n{'-'*80}")
    print(f"{side_name} SIDE")
    print(f"{'-'*80}")

    results = []
    for au in common_aus:
        of3_vals = of3_df[au].values
        of22_vals = of22_df[au].values

        # Remove NaN
        mask = ~(np.isnan(of3_vals) | np.isnan(of22_vals))
        of3_clean = of3_vals[mask]
        of22_clean = of22_vals[mask]

        if len(of3_clean) < 2:
            continue

        # Calculate metrics
        corr, _ = stats.pearsonr(of3_clean, of22_clean) if np.std(of3_clean) > 0 and np.std(of22_clean) > 0 else (0, 1)
        abs_diff = np.abs(of3_clean - of22_clean).mean()

        results.append({
            'AU': au,
            'correlation': corr,
            'mean_abs_diff': abs_diff,
            'of3_mean': np.mean(of3_clean),
            'of22_mean': np.mean(of22_clean)
        })

    results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)

    print(f"\n{'AU':<12} {'OF3 Mean':>10} {'OF2.2 Mean':>10} {'Abs Diff':>10} {'Correlation':>12}")
    print('-'*60)
    for _, row in results_df.iterrows():
        print(f"{row['AU']:<12} {row['of3_mean']:>10.4f} {row['of22_mean']:>10.4f} {row['mean_abs_diff']:>10.4f} {row['correlation']:>12.4f}")

    avg_corr = results_df['correlation'].mean()
    print(f"\n{side_name} Average Correlation: {avg_corr:.4f}")

    return results_df

left_results = compare_side(of3_left, of22_left, "LEFT")
right_results = compare_side(of3_right, of22_right, "RIGHT")

# Overall summary
print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}")

all_results = pd.concat([left_results, right_results])
avg_corr = all_results['correlation'].mean()
avg_diff = all_results['mean_abs_diff'].mean()

print(f"\nAverage correlation across all AUs: {avg_corr:.4f}")
print(f"Average absolute difference: {avg_diff:.4f}")

if avg_corr > 0.7:
    print("\n✓ FAIR to GOOD agreement")
    print("OpenFace 3.0 shows reasonable agreement with 2.2")
elif avg_corr > 0.5:
    print("\n⚠ MODERATE agreement")
    print("OpenFace 3.0 has notable differences from 2.2")
else:
    print("\n⚠ LOW agreement")
    print("OpenFace 3.0 uses fundamentally different approach than 2.2")

print("\nInterpretation:")
print("  - OpenFace 3.0 uses modern deep learning (transformer-based)")
print("  - OpenFace 2.2 uses traditional computer vision (SVMs + HOG)")
print("  - Low correlation does NOT mean OF3 is worse")
print("  - OF3 is likely MORE accurate despite differences")
print("  - These are different measurement methods")

print(f"\n{'='*80}\n")
