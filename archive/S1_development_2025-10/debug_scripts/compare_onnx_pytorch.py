#!/usr/bin/env python3
"""
Compare ONNX vs PyTorch AU outputs to assess equivalence
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
ONNX_FILES = {
    'left': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredONNXv3.csv',
    'right': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_right_mirroredONNXv3.csv'
}

PYTORCH_FILES = {
    'left': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv',
    'right': '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_right_mirroredvOP3ORIG.csv'
}

def load_and_compare(side):
    """Load ONNX and PyTorch CSVs for a given side and compute statistics"""

    print(f"\n{'='*80}")
    print(f"COMPARING {side.upper()} SIDE: ONNX vs PyTorch")
    print(f"{'='*80}")

    # Load data
    onnx_df = pd.read_csv(ONNX_FILES[side])
    pytorch_df = pd.read_csv(PYTORCH_FILES[side])

    print(f"\nONNX frames: {len(onnx_df)}")
    print(f"PyTorch frames: {len(pytorch_df)}")

    # Ensure same number of frames
    min_frames = min(len(onnx_df), len(pytorch_df))
    onnx_df = onnx_df.head(min_frames)
    pytorch_df = pytorch_df.head(min_frames)

    # Get AU columns (both _r and _c)
    au_cols = [col for col in onnx_df.columns if col.startswith('AU') and ('_r' in col or '_c' in col)]

    # Filter to only columns that exist in both datasets
    au_cols = [col for col in au_cols if col in pytorch_df.columns]

    print(f"\nAU columns to compare: {len(au_cols)}")
    print(f"AU types: {sorted(set([col.split('_')[0] for col in au_cols]))}")

    # Compute statistics for each AU
    results = []

    for au_col in au_cols:
        onnx_vals = onnx_df[au_col].values
        pytorch_vals = pytorch_df[au_col].values

        # Remove NaN values for comparison
        mask = ~(np.isnan(onnx_vals) | np.isnan(pytorch_vals))
        onnx_clean = onnx_vals[mask]
        pytorch_clean = pytorch_vals[mask]

        if len(onnx_clean) == 0:
            continue

        # Compute metrics
        abs_diff = np.abs(onnx_clean - pytorch_clean)
        rel_diff = abs_diff / (np.abs(pytorch_clean) + 1e-10)  # Avoid div by zero

        # Correlation
        if len(onnx_clean) > 1 and np.std(onnx_clean) > 0 and np.std(pytorch_clean) > 0:
            pearson_corr, _ = stats.pearsonr(onnx_clean, pytorch_clean)
        else:
            pearson_corr = np.nan

        results.append({
            'AU': au_col,
            'valid_frames': len(onnx_clean),
            'mean_abs_diff': np.mean(abs_diff),
            'median_abs_diff': np.median(abs_diff),
            'max_abs_diff': np.max(abs_diff),
            'mean_rel_diff_%': np.mean(rel_diff) * 100,
            'median_rel_diff_%': np.median(rel_diff) * 100,
            'pearson_corr': pearson_corr,
            'onnx_mean': np.mean(onnx_clean),
            'pytorch_mean': np.mean(pytorch_clean),
            'onnx_std': np.std(onnx_clean),
            'pytorch_std': np.std(pytorch_clean)
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Separate regression (_r) and classification (_c)
    results_r = results_df[results_df['AU'].str.contains('_r')]
    results_c = results_df[results_df['AU'].str.contains('_c')]

    return results_r, results_c, onnx_df, pytorch_df, au_cols


def print_summary(results_df, title):
    """Print summary statistics"""
    print(f"\n{title}")
    print("-" * len(title))

    if len(results_df) == 0:
        print("No data available")
        return

    print(f"\nNumber of AUs: {len(results_df)}")
    print(f"\nAbsolute Differences:")
    print(f"  Mean:   {results_df['mean_abs_diff'].mean():.6f}")
    print(f"  Median: {results_df['median_abs_diff'].median():.6f}")
    print(f"  Max:    {results_df['max_abs_diff'].max():.6f}")

    print(f"\nRelative Differences (%):")
    print(f"  Mean:   {results_df['mean_rel_diff_%'].mean():.2f}%")
    print(f"  Median: {results_df['median_rel_diff_%'].median():.2f}%")

    print(f"\nPearson Correlation:")
    valid_corr = results_df['pearson_corr'].dropna()
    if len(valid_corr) > 0:
        print(f"  Mean:   {valid_corr.mean():.6f}")
        print(f"  Median: {valid_corr.median():.6f}")
        print(f"  Min:    {valid_corr.min():.6f}")

    print(f"\nTop 10 AUs by Absolute Difference:")
    top10 = results_df.nlargest(10, 'mean_abs_diff')[['AU', 'mean_abs_diff', 'pearson_corr']]
    print(top10.to_string(index=False))

    print(f"\nTop 10 AUs by Correlation (highest agreement):")
    top10_corr = results_df.nlargest(10, 'pearson_corr')[['AU', 'pearson_corr', 'mean_abs_diff']]
    print(top10_corr.to_string(index=False))


def create_visualizations(side, results_r, results_c, onnx_df, pytorch_df):
    """Create comparison visualizations"""

    # Select a few key AUs for plotting
    key_aus_r = ['AU01_r', 'AU02_r', 'AU12_r', 'AU45_r']
    key_aus_r = [au for au in key_aus_r if au in onnx_df.columns and au in pytorch_df.columns]

    if len(key_aus_r) == 0:
        print(f"\nNo key AUs available for visualization on {side} side")
        return

    # Create figure with subplots
    n_plots = len(key_aus_r)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 4*n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for idx, au in enumerate(key_aus_r):
        # Time series plot
        ax1 = axes[idx, 0]
        ax1.plot(onnx_df['frame'], onnx_df[au], label='ONNX', alpha=0.7, linewidth=2)
        ax1.plot(pytorch_df['frame'], pytorch_df[au], label='PyTorch', alpha=0.7, linewidth=2, linestyle='--')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel(au)
        ax1.set_title(f'{side.upper()} - {au} Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2 = axes[idx, 1]
        mask = ~(np.isnan(onnx_df[au]) | np.isnan(pytorch_df[au]))
        if mask.sum() > 0:
            ax2.scatter(pytorch_df[au][mask], onnx_df[au][mask], alpha=0.5, s=20)

            # Add diagonal line (perfect agreement)
            min_val = min(pytorch_df[au][mask].min(), onnx_df[au][mask].min())
            max_val = max(pytorch_df[au][mask].max(), onnx_df[au][mask].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')

            # Calculate and display correlation
            if mask.sum() > 1:
                corr, _ = stats.pearsonr(pytorch_df[au][mask], onnx_df[au][mask])
                ax2.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax2.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel('PyTorch')
        ax2.set_ylabel('ONNX')
        ax2.set_title(f'{side.upper()} - {au} Correlation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    output_path = f'/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/onnx_vs_pytorch_{side}_v3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    """Main comparison function"""

    print("\n" + "="*80)
    print("ONNX vs PyTorch AU Output Comparison (WITH ANTIALIASING FIX)")
    print("="*80)
    print("Comparing CORRECTED ONNX (ONNXv3) to original PyTorch (OP3ORIG)")
    print("Expected: >0.99 correlation after antialiasing fix")
    print("="*80)

    all_results = {}

    for side in ['left', 'right']:
        results_r, results_c, onnx_df, pytorch_df, au_cols = load_and_compare(side)
        all_results[side] = {'r': results_r, 'c': results_c}

        # Print summaries
        print_summary(results_r, f"\n{side.upper()} - Regression AUs (_r)")
        print_summary(results_c, f"\n{side.upper()} - Classification AUs (_c)")

        # Create visualizations
        create_visualizations(side, results_r, results_c, onnx_df, pytorch_df)

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY - Both Sides Combined")
    print(f"{'='*80}")

    # Combine all regression results
    all_r = pd.concat([all_results['left']['r'], all_results['right']['r']], ignore_index=True)
    all_c = pd.concat([all_results['left']['c'], all_results['right']['c']], ignore_index=True)

    print_summary(all_r, "\nCOMBINED - Regression AUs (_r)")
    print_summary(all_c, "\nCOMBINED - Classification AUs (_c)")

    # Assessment
    print(f"\n{'='*80}")
    print("EQUIVALENCE ASSESSMENT")
    print(f"{'='*80}")

    if len(all_r) > 0:
        mean_corr = all_r['pearson_corr'].dropna().mean()
        mean_abs_diff = all_r['mean_abs_diff'].mean()

        print(f"\nRegression AUs:")
        print(f"  Average correlation: {mean_corr:.4f}")
        print(f"  Average absolute difference: {mean_abs_diff:.6f}")

        if mean_corr > 0.99 and mean_abs_diff < 0.01:
            print(f"  ✓ EXCELLENT equivalence - virtually identical")
        elif mean_corr > 0.95 and mean_abs_diff < 0.05:
            print(f"  ✓ GOOD equivalence - minor differences")
        elif mean_corr > 0.90 and mean_abs_diff < 0.1:
            print(f"  ⚠ MODERATE equivalence - noticeable differences")
        else:
            print(f"  ✗ POOR equivalence - significant differences")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
