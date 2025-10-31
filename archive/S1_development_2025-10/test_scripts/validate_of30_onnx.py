#!/usr/bin/env python3
"""
Validate OF3.0 ONNX Implementation vs Original

This script compares our ONNX implementation against the original
OpenFace 3.0 Python models to identify any discrepancies.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
sns.set_style("whitegrid")

class OF30Validator:
    """Validate ONNX implementation against original OF3.0"""

    def __init__(self, onnx_left, onnx_right, orig_left, orig_right):
        """
        Initialize validator

        Args:
            onnx_left: Path to ONNX left side CSV
            onnx_right: Path to ONNX right side CSV
            orig_left: Path to original OF3.0 left side CSV
            orig_right: Path to original OF3.0 right side CSV
        """
        print("="*80)
        print("OpenFace 3.0 ONNX Implementation Validation")
        print("="*80)
        print("\nLoading data...")

        self.onnx_left = pd.read_csv(onnx_left)
        self.onnx_right = pd.read_csv(onnx_right)
        self.orig_left = pd.read_csv(orig_left)
        self.orig_right = pd.read_csv(orig_right)

        print(f"  ONNX Left:  {len(self.onnx_left)} frames")
        print(f"  ONNX Right: {len(self.onnx_right)} frames")
        print(f"  Orig Left:  {len(self.orig_left)} frames")
        print(f"  Orig Right: {len(self.orig_right)} frames")

        # Get AU columns (intensity only)
        self.au_cols = sorted([c for c in self.onnx_left.columns if c.startswith('AU') and c.endswith('_r')])

        # Filter to AUs with values (not all NaN)
        self.active_aus = []
        for au in self.au_cols:
            if not self.onnx_left[au].isna().all() or not self.orig_left[au].isna().all():
                self.active_aus.append(au)

        print(f"\n  Active AUs: {len(self.active_aus)}")

    def compare_distributions(self, side='left'):
        """Compare value distributions between ONNX and original"""
        print("\n" + "="*80)
        print(f"Value Distribution Comparison ({side.upper()} side)")
        print("="*80)

        onnx_data = self.onnx_left if side == 'left' else self.onnx_right
        orig_data = self.orig_left if side == 'left' else self.orig_right

        print(f"\n{'AU':<10} {'ONNX Mean':<12} {'Orig Mean':<12} {'Diff %':<10} {'ONNX Std':<12} {'Orig Std':<12}")
        print("-"*80)

        results = []

        for au in self.active_aus:
            onnx_vals = onnx_data[au].dropna()
            orig_vals = orig_data[au].dropna()

            if len(onnx_vals) > 0 and len(orig_vals) > 0:
                onnx_mean = onnx_vals.mean()
                orig_mean = orig_vals.mean()
                onnx_std = onnx_vals.std()
                orig_std = orig_vals.std()

                # Calculate percentage difference
                if orig_mean > 0.01:
                    diff_pct = (onnx_mean - orig_mean) / orig_mean * 100
                else:
                    diff_pct = 0.0

                status = ""
                if abs(diff_pct) < 5:
                    status = "✓"
                elif abs(diff_pct) < 20:
                    status = "~"
                else:
                    status = "✗"

                print(f"{au:<10} {onnx_mean:<12.3f} {orig_mean:<12.3f} {diff_pct:>6.1f}% {status:<3} {onnx_std:<12.3f} {orig_std:<12.3f}")

                results.append({
                    'au': au,
                    'onnx_mean': onnx_mean,
                    'orig_mean': orig_mean,
                    'diff_pct': diff_pct,
                    'onnx_std': onnx_std,
                    'orig_std': orig_std
                })

        return results

    def calculate_correlations(self, side='left'):
        """Calculate frame-by-frame correlations"""
        print("\n" + "="*80)
        print(f"Frame-by-Frame Correlations ({side.upper()} side)")
        print("="*80)

        onnx_data = self.onnx_left if side == 'left' else self.onnx_right
        orig_data = self.orig_left if side == 'left' else self.orig_right

        # Align frames
        min_len = min(len(onnx_data), len(orig_data))
        onnx_data = onnx_data.iloc[:min_len]
        orig_data = orig_data.iloc[:min_len]

        print(f"\nUsing {min_len} frames for correlation analysis")
        print(f"\n{'AU':<10} {'Pearson r':<12} {'P-value':<12} {'Status':<15}")
        print("-"*60)

        corr_results = []

        for au in self.active_aus:
            onnx_vals = onnx_data[au]
            orig_vals = orig_data[au]

            # Remove NaN pairs
            valid_mask = (~onnx_vals.isna()) & (~orig_vals.isna())

            if valid_mask.sum() > 10:
                onnx_valid = onnx_vals[valid_mask]
                orig_valid = orig_vals[valid_mask]

                r, p = pearsonr(onnx_valid, orig_valid)

                # Interpret correlation
                if abs(r) >= 0.95:
                    status = "✓ Excellent"
                elif abs(r) >= 0.90:
                    status = "✓ Good"
                elif abs(r) >= 0.80:
                    status = "~ Acceptable"
                elif abs(r) >= 0.70:
                    status = "⚠ Weak"
                else:
                    status = "✗ Poor"

                print(f"{au:<10} {r:<12.3f} {p:<12.6f} {status:<15}")

                corr_results.append({
                    'au': au,
                    'pearson_r': r,
                    'p_value': p,
                    'n_pairs': valid_mask.sum()
                })
            else:
                print(f"{au:<10} {'N/A':<12} {'N/A':<12} {'Insufficient data':<15}")

        return corr_results

    def plot_au_comparison(self, au='AU12_r', save_path=None):
        """Plot AU timeseries comparison"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'{au} Comparison: ONNX vs Original OF3.0', fontsize=16, fontweight='bold')

        # Left side
        ax = axes[0]
        min_len = min(len(self.onnx_left), len(self.orig_left))
        frames = np.arange(min_len)

        if au in self.onnx_left.columns:
            ax.plot(frames, self.onnx_left[au].iloc[:min_len], label='ONNX', alpha=0.7, linewidth=2)
        if au in self.orig_left.columns:
            ax.plot(frames, self.orig_left[au].iloc[:min_len], label='Original', alpha=0.7, linewidth=2)

        ax.set_title('Left Side', fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('AU Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right side
        ax = axes[1]
        min_len = min(len(self.onnx_right), len(self.orig_right))
        frames = np.arange(min_len)

        if au in self.onnx_right.columns:
            ax.plot(frames, self.onnx_right[au].iloc[:min_len], label='ONNX', alpha=0.7, linewidth=2)
        if au in self.orig_right.columns:
            ax.plot(frames, self.orig_right[au].iloc[:min_len], label='Original', alpha=0.7, linewidth=2)

        ax.set_title('Right Side', fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('AU Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to: {save_path}")

        return fig

    def identify_discrepancies(self):
        """Identify specific frames with large discrepancies"""
        print("\n" + "="*80)
        print("Frame-Level Discrepancy Analysis")
        print("="*80)

        for side, onnx_data, orig_data in [
            ('left', self.onnx_left, self.orig_left),
            ('right', self.onnx_right, self.orig_right)
        ]:
            print(f"\n{side.upper()} SIDE:")
            print("-" * 40)

            for au in ['AU01_r', 'AU12_r', 'AU45_r']:  # Key AUs
                if au not in self.active_aus:
                    continue

                onnx_vals = onnx_data[au].values
                orig_vals = orig_data[au].values

                # Find frames with large differences
                min_len = min(len(onnx_vals), len(orig_vals))
                onnx_vals = onnx_vals[:min_len]
                orig_vals = orig_vals[:min_len]

                # Remove NaN
                valid_mask = (~np.isnan(onnx_vals)) & (~np.isnan(orig_vals))
                valid_onnx = onnx_vals[valid_mask]
                valid_orig = orig_vals[valid_mask]

                if len(valid_onnx) == 0:
                    continue

                # Calculate absolute differences
                diffs = np.abs(valid_onnx - valid_orig)
                mean_diff = diffs.mean()
                max_diff = diffs.max()
                max_diff_idx = diffs.argmax()

                # Get frame number
                valid_indices = np.where(valid_mask)[0]
                max_diff_frame = valid_indices[max_diff_idx] if len(valid_indices) > 0 else -1

                print(f"\n  {au}:")
                print(f"    Mean absolute difference: {mean_diff:.4f}")
                print(f"    Max difference: {max_diff:.4f} at frame {max_diff_frame}")
                if max_diff_frame >= 0:
                    print(f"      ONNX value: {onnx_vals[max_diff_frame]:.4f}")
                    print(f"      Orig value: {orig_vals[max_diff_frame]:.4f}")

    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        # Collect all correlations
        left_corrs = self.calculate_correlations('left')
        right_corrs = self.calculate_correlations('right')

        all_corrs = left_corrs + right_corrs

        if len(all_corrs) > 0:
            avg_r = np.mean([abs(c['pearson_r']) for c in all_corrs])

            excellent = sum(1 for c in all_corrs if abs(c['pearson_r']) >= 0.95)
            good = sum(1 for c in all_corrs if 0.90 <= abs(c['pearson_r']) < 0.95)
            acceptable = sum(1 for c in all_corrs if 0.80 <= abs(c['pearson_r']) < 0.90)
            weak = sum(1 for c in all_corrs if 0.70 <= abs(c['pearson_r']) < 0.80)
            poor = sum(1 for c in all_corrs if abs(c['pearson_r']) < 0.70)

            print(f"\nAverage correlation: {avg_r:.3f}")
            print(f"\nCorrelation breakdown:")
            print(f"  ✓ Excellent (r≥0.95): {excellent}/{len(all_corrs)}")
            print(f"  ✓ Good (0.90≤r<0.95): {good}/{len(all_corrs)}")
            print(f"  ~ Acceptable (0.80≤r<0.90): {acceptable}/{len(all_corrs)}")
            print(f"  ⚠ Weak (0.70≤r<0.80): {weak}/{len(all_corrs)}")
            print(f"  ✗ Poor (r<0.70): {poor}/{len(all_corrs)}")

            print("\n" + "="*80)
            print("RECOMMENDATION:")
            print("="*80)

            if avg_r >= 0.95:
                print("\n✓ EXCELLENT - ONNX implementation matches original OF3.0")
                print("  → Safe to proceed with OF2.2 migration")
            elif avg_r >= 0.90:
                print("\n✓ GOOD - Minor differences, but acceptable")
                print("  → Proceed with OF2.2 migration")
                print("  → Document known differences")
            elif avg_r >= 0.80:
                print("\n~ ACCEPTABLE - Some discrepancies present")
                print("  → Investigate discrepancies before proceeding")
                print("  → May need to fix ONNX implementation")
            else:
                print("\n✗ POOR - Significant discrepancies detected")
                print("  → DO NOT proceed with OF2.2 migration")
                print("  → Fix ONNX implementation first")
                print("  → Check preprocessing, model conversion, post-processing")


def main():
    """Main validation workflow"""

    base_path = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files")

    # Paths to CSV files
    onnx_left = base_path / "OP3 v OP 22" / "IMG_0942_left_mirroredONNXv3.csv"
    onnx_right = base_path / "OP3 v OP 22" / "IMG_0942_right_mirroredONNXv3.csv"
    orig_left = base_path / "Three-Way Comparison" / "IMG_0942_left_mirroredOP3ORIG.csv"
    orig_right = base_path / "Three-Way Comparison" / "IMG_0942_right_mirroredvOP3ORIG.csv"

    # Create validator
    validator = OF30Validator(onnx_left, onnx_right, orig_left, orig_right)

    # Run analyses
    validator.compare_distributions('left')
    validator.compare_distributions('right')

    left_corrs = validator.calculate_correlations('left')
    right_corrs = validator.calculate_correlations('right')

    validator.identify_discrepancies()

    # Generate plots
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror")

    for au in ['AU01_r', 'AU12_r', 'AU45_r']:
        if au in validator.active_aus:
            print(f"\nPlotting {au}...")
            save_path = output_dir / f"of30_validation_{au}.png"
            validator.plot_au_comparison(au, save_path=save_path)

    # Final summary
    validator.generate_summary_report()

    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
