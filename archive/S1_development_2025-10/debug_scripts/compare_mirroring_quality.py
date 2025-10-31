#!/usr/bin/env python3
"""
Compare OpenFace 3.0 AU predictions on different mirroring approaches

Tests hypothesis: Does OF2.2's higher-quality mirroring improve OF3.0 AU predictions?

Compares:
1. OF2.2 AUs (ground truth from original OF2.2 pipeline)
2. OF3.0 AUs on OF3.0-mirrored videos (original test)
3. OF3.0 AUs on OF2.2-mirrored videos (NEW - better mirroring)

If correlations improve in (3), mirroring was the problem.
If correlations stay poor, OF3.0 AU models are the problem.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def load_csv_data(csv_path):
    """Load CSV and extract AU columns"""
    df = pd.read_csv(csv_path)
    au_cols = sorted([c for c in df.columns if c.startswith('AU') and c.endswith('_r')])
    return df, au_cols

def calculate_correlation(series1, series2):
    """Calculate Pearson correlation, handling NaN values"""
    # Remove NaN values
    mask = ~(np.isnan(series1) | np.isnan(series2))
    if mask.sum() < 10:  # Need at least 10 valid points
        return np.nan, np.nan

    valid1 = series1[mask]
    valid2 = series2[mask]

    # Check for zero variance
    if valid1.std() < 1e-6 or valid2.std() < 1e-6:
        return np.nan, np.nan

    try:
        r_pearson, _ = pearsonr(valid1, valid2)
        r_spearman, _ = spearmanr(valid1, valid2)
        return r_pearson, r_spearman
    except:
        return np.nan, np.nan

def compare_distributions(df_of22, df_of3_old, df_of3_new, au_name, side):
    """Compare AU distributions across three approaches"""

    results = {
        'au': au_name,
        'side': side,
        'of22_mean': df_of22[au_name].mean(),
        'of22_std': df_of22[au_name].std(),
        'of3_old_mean': df_of3_old[au_name].mean(),
        'of3_old_std': df_of3_old[au_name].std(),
        'of3_new_mean': df_of3_new[au_name].mean(),
        'of3_new_std': df_of3_new[au_name].std(),
    }

    # Correlations: OF2.2 vs OF3.0 (old mirroring)
    r_old, rs_old = calculate_correlation(
        df_of22[au_name].values,
        df_of3_old[au_name].values
    )
    results['corr_old_pearson'] = r_old
    results['corr_old_spearman'] = rs_old

    # Correlations: OF2.2 vs OF3.0 (new mirroring)
    r_new, rs_new = calculate_correlation(
        df_of22[au_name].values,
        df_of3_new[au_name].values
    )
    results['corr_new_pearson'] = r_new
    results['corr_new_spearman'] = rs_new

    # Improvement
    if not np.isnan(r_old) and not np.isnan(r_new):
        results['improvement'] = r_new - r_old
    else:
        results['improvement'] = np.nan

    return results

def plot_comparison(df_of22, df_of3_old, df_of3_new, au_name, side, output_dir):
    """Create comparison plot showing all three approaches"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Temporal plot
    frames = df_of22['frame'].values if 'frame' in df_of22.columns else np.arange(len(df_of22))

    ax1.plot(frames, df_of22[au_name], 'b-', alpha=0.7, linewidth=1.5, label='OF2.2 (Ground Truth)')
    ax1.plot(frames, df_of3_old[au_name], 'r-', alpha=0.5, linewidth=1, label='OF3.0 (OF3.0 Mirroring)')
    ax1.plot(frames, df_of3_new[au_name], 'g-', alpha=0.5, linewidth=1, label='OF3.0 (OF2.2 Mirroring)')

    ax1.set_xlabel('Frame', fontsize=11)
    ax1.set_ylabel('AU Intensity', fontsize=11)
    ax1.set_title(f'{au_name} - {side} Side: Temporal Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Scatter plots: OF2.2 vs OF3.0
    ax2_left = ax2
    ax2_right = ax2.twinx()

    # Old mirroring (red)
    mask_old = ~(np.isnan(df_of22[au_name]) | np.isnan(df_of3_old[au_name]))
    r_old, _ = calculate_correlation(df_of22[au_name].values, df_of3_old[au_name].values)
    ax2_left.scatter(df_of22[au_name][mask_old], df_of3_old[au_name][mask_old],
                     c='red', alpha=0.3, s=10, label=f'OF3.0 Mirroring (r={r_old:.3f})')

    # New mirroring (green)
    mask_new = ~(np.isnan(df_of22[au_name]) | np.isnan(df_of3_new[au_name]))
    r_new, _ = calculate_correlation(df_of22[au_name].values, df_of3_new[au_name].values)
    ax2_right.scatter(df_of22[au_name][mask_new], df_of3_new[au_name][mask_new],
                      c='green', alpha=0.3, s=10, label=f'OF2.2 Mirroring (r={r_new:.3f})')

    # Reference line
    max_val = max(df_of22[au_name].max(), df_of3_old[au_name].max(), df_of3_new[au_name].max())
    ax2_left.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

    ax2_left.set_xlabel('OF2.2 AU Intensity', fontsize=11)
    ax2_left.set_ylabel('OF3.0 AU Intensity (OF3.0 Mirroring - Red)', fontsize=10, color='red')
    ax2_right.set_ylabel('OF3.0 AU Intensity (OF2.2 Mirroring - Green)', fontsize=10, color='green')
    ax2_left.set_title(f'{au_name} - {side} Side: Correlation Comparison', fontsize=12, fontweight='bold')

    # Legends
    lines1, labels1 = ax2_left.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2_left.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax2_left.grid(True, alpha=0.3)
    ax2_left.tick_params(axis='y', labelcolor='red')
    ax2_right.tick_params(axis='y', labelcolor='green')

    plt.tight_layout()

    output_file = output_dir / f'mirroring_comparison_{au_name}_{side}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file

def main():
    """Main comparison analysis"""

    data_dir = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22")
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror")

    print("=" * 80)
    print("Mirroring Quality Impact Analysis")
    print("=" * 80)
    print("\nHypothesis: Does OF2.2's higher-quality mirroring improve OF3.0 AU predictions?")
    print("\nComparing:")
    print("  1. OF2.2 AUs (ground truth)")
    print("  2. OF3.0 on OF3.0-mirrored videos (original)")
    print("  3. OF3.0 on OF2.2-mirrored videos (NEW)")
    print("\n" + "=" * 80 + "\n")

    # File paths
    files = {
        'left': {
            'of22': data_dir / 'IMG_0942_left_mirroredOP22.csv',
            'of3_old': data_dir / 'IMG_0942_left_mirroredONNXv3.csv',
            'of3_new': data_dir / 'IMG_0942_left_mirroredOP22_processedONNXv3.csv'
        },
        'right': {
            'of22': data_dir / 'IMG_0942_right_mirroredOP22.csv',
            'of3_old': data_dir / 'IMG_0942_right_mirroredONNXv3.csv',
            'of3_new': data_dir / 'IMG_0942_right_mirroredOP22_processedONNXv3.csv'
        }
    }

    # Check files exist
    for side, paths in files.items():
        for name, path in paths.items():
            if not path.exists():
                print(f"ERROR: Missing file: {path.name}")
                return

    print("✓ All input files found\n")

    # Process both sides
    all_results = []

    for side in ['left', 'right']:
        print(f"\nProcessing {side.upper()} side...")
        print("-" * 80)

        # Load data
        df_of22, of22_cols = load_csv_data(files[side]['of22'])
        df_of3_old, of3_old_cols = load_csv_data(files[side]['of3_old'])
        df_of3_new, of3_new_cols = load_csv_data(files[side]['of3_new'])

        # Only compare AUs present in ALL three datasets
        common_aus = set(of22_cols) & set(of3_old_cols) & set(of3_new_cols)
        au_cols = sorted(list(common_aus))

        print(f"  Loaded {len(df_of22)} frames")
        print(f"  Analyzing {len(au_cols)} AUs (common to all datasets)\n")

        # Compare each AU
        for au in au_cols:
            # Skip if AU is all NaN in any dataset
            if (df_of22[au].isna().all() or
                df_of3_old[au].isna().all() or
                df_of3_new[au].isna().all()):
                continue

            # Calculate statistics
            result = compare_distributions(df_of22, df_of3_old, df_of3_new, au, side)
            all_results.append(result)

            # Create comparison plot
            plot_file = plot_comparison(df_of22, df_of3_old, df_of3_new, au, side, output_dir)

            # Print summary
            r_old = result['corr_old_pearson']
            r_new = result['corr_new_pearson']
            improvement = result['improvement']

            print(f"  {au}:")
            print(f"    OF2.2:          mean={result['of22_mean']:.3f}, std={result['of22_std']:.3f}")
            print(f"    OF3.0 (old mir): mean={result['of3_old_mean']:.3f}, std={result['of3_old_std']:.3f}, r={r_old:.3f}")
            print(f"    OF3.0 (new mir): mean={result['of3_new_mean']:.3f}, std={result['of3_new_std']:.3f}, r={r_new:.3f}")
            if not np.isnan(improvement):
                status = "✓ IMPROVED" if improvement > 0.1 else "~ SIMILAR" if improvement > -0.1 else "✗ WORSE"
                print(f"    Improvement: {improvement:+.3f} {status}")
            print()

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80 + "\n")

    results_df = pd.DataFrame(all_results)

    # Count improvements
    improved = (results_df['improvement'] > 0.1).sum()
    similar = ((results_df['improvement'] >= -0.1) & (results_df['improvement'] <= 0.1)).sum()
    worse = (results_df['improvement'] < -0.1).sum()

    print(f"AUs Improved (Δr > 0.1):  {improved}/{len(results_df)}")
    print(f"AUs Similar (|Δr| ≤ 0.1): {similar}/{len(results_df)}")
    print(f"AUs Worse (Δr < -0.1):    {worse}/{len(results_df)}")

    # Average improvement
    avg_improvement = results_df['improvement'].mean()
    print(f"\nAverage correlation change: {avg_improvement:+.3f}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80 + "\n")

    if avg_improvement > 0.15:
        print("✓ MIRRORING WAS THE PROBLEM")
        print("  OF2.2's mirroring significantly improves OF3.0 AU predictions.")
        print("  Recommendation: Use OF2.2 mirroring pipeline with OF3.0 AU extraction.")
    elif avg_improvement > 0.05:
        print("~ MIRRORING PARTIALLY RESPONSIBLE")
        print("  OF2.2's mirroring shows modest improvement in OF3.0 AU predictions.")
        print("  Both mirroring and AU models may need investigation.")
    else:
        print("✗ AU MODELS ARE THE PROBLEM")
        print("  Mirroring quality does NOT explain OF3.0's poor AU predictions.")
        print("  The OF3.0 AU models themselves are clinically invalid.")
        print("  Recommendation: Proceed with OF2.2 AU model migration as planned.")

    print("\n" + "=" * 80)
    print(f"\nPlots saved to: {output_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
