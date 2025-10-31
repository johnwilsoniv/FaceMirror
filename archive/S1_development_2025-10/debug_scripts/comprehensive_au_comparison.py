#!/usr/bin/env python3
"""
Comprehensive OpenFace 2.2 vs 3.0 Comparison Analysis

This script performs detailed statistical and clinical validation of AU data from
OpenFace 2.2 (C++ binary) vs OpenFace 3.0 (Python/ONNX).

Goals:
1. Assess data quality and correlations between versions
2. Validate clinical meaningfulness of OF3.0 AUs
3. Test symmetry in non-paralyzed patient
4. Inform decision on OF2.2 Python migration vs OF3.0 adoption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class OpenFaceComparer:
    """Comprehensive comparison of OpenFace 2.2 vs 3.0"""

    def __init__(self, of22_left_path, of22_right_path, of30_left_path, of30_right_path):
        """Initialize with paths to all 4 CSV files"""
        print("="*80)
        print("OpenFace 2.2 vs 3.0 Comprehensive Analysis")
        print("="*80)
        print(f"\nLoading data...")

        # Load CSVs
        self.of22_left = pd.read_csv(of22_left_path)
        self.of22_right = pd.read_csv(of22_right_path)
        self.of30_left = pd.read_csv(of30_left_path)
        self.of30_right = pd.read_csv(of30_right_path)

        print(f"  OF2.2 Left:  {len(self.of22_left)} frames")
        print(f"  OF2.2 Right: {len(self.of22_right)} frames")
        print(f"  OF3.0 Left:  {len(self.of30_left)} frames")
        print(f"  OF3.0 Right: {len(self.of30_right)} frames")

        # Identify AU columns
        self.of22_au_cols = sorted([c for c in self.of22_left.columns if c.startswith('AU') and c.endswith('_r')])
        self.of30_au_cols = sorted([c for c in self.of30_left.columns if c.startswith('AU') and c.endswith('_r')])

        # Common AUs (present in both versions)
        self.common_aus = sorted(list(set(self.of22_au_cols) & set(self.of30_au_cols)))

        # Missing AUs (in OF2.2 but not OF3.0)
        self.missing_in_of30 = sorted(list(set(self.of22_au_cols) - set(self.of30_au_cols)))

        # New AUs (in OF3.0 but not OF2.2)
        self.new_in_of30 = sorted(list(set(self.of30_au_cols) - set(self.of22_au_cols)))

        print(f"\n  Common AUs: {len(self.common_aus)}")
        print(f"  Missing in OF3.0: {len(self.missing_in_of30)}")
        print(f"  New in OF3.0: {len(self.new_in_of30)}")

        # Results storage
        self.results = {
            'correlations': {},
            'distributions': {},
            'symmetry': {},
            'clinical_actions': {}
        }

    def analyze_au_availability(self):
        """Analyze which AUs are actually available (not all NaN) in OF3.0"""
        print("\n" + "="*80)
        print("ANALYSIS 1: AU Availability")
        print("="*80)

        print("\n📊 OpenFace 2.2 AUs (all have values):")
        for au in self.of22_au_cols:
            print(f"  ✓ {au}")

        print(f"\n📊 OpenFace 3.0 AUs:")

        # Check which OF3.0 AUs actually have values
        of30_available = []
        of30_all_nan = []

        for au in self.of30_au_cols:
            left_has_values = not self.of30_left[au].isna().all()
            right_has_values = not self.of30_right[au].isna().all()

            if left_has_values or right_has_values:
                of30_available.append(au)
                non_nan_pct_left = (~self.of30_left[au].isna()).sum() / len(self.of30_left) * 100
                non_nan_pct_right = (~self.of30_right[au].isna()).sum() / len(self.of30_right) * 100
                print(f"  ✓ {au}: {non_nan_pct_left:.1f}% left, {non_nan_pct_right:.1f}% right")
            else:
                of30_all_nan.append(au)
                print(f"  ✗ {au}: ALL NaN (never computed)")

        print(f"\n❌ AUs Missing in OF3.0 (present in OF2.2):")
        for au in self.missing_in_of30:
            print(f"  - {au}")

        self.results['availability'] = {
            'of30_available': of30_available,
            'of30_all_nan': of30_all_nan,
            'missing_in_of30': self.missing_in_of30
        }

        return of30_available

    def compare_distributions(self, side='left'):
        """Compare value distributions for common AUs"""
        print("\n" + "="*80)
        print(f"ANALYSIS 2: Value Distributions ({side.upper()} side)")
        print("="*80)

        of22_data = self.of22_left if side == 'left' else self.of22_right
        of30_data = self.of30_left if side == 'left' else self.of30_right

        print(f"\n{'AU':<10} {'OF2.2 Mean':<12} {'OF3.0 Mean':<12} {'Ratio':<10} {'OF2.2 Std':<12} {'OF3.0 Std':<12}")
        print("-"*70)

        dist_results = []

        for au in self.common_aus:
            # Get non-NaN values
            of22_vals = of22_data[au].dropna()
            of30_vals = of30_data[au].dropna()

            if len(of22_vals) > 0 and len(of30_vals) > 0:
                of22_mean = of22_vals.mean()
                of30_mean = of30_vals.mean()
                of22_std = of22_vals.std()
                of30_std = of30_vals.std()

                # Calculate ratio (handle near-zero values)
                if of30_mean > 0.001:
                    ratio = of22_mean / of30_mean
                    ratio_str = f"{ratio:.1f}x"
                else:
                    ratio_str = "N/A"
                    ratio = np.nan

                print(f"{au:<10} {of22_mean:<12.3f} {of30_mean:<12.3f} {ratio_str:<10} {of22_std:<12.3f} {of30_std:<12.3f}")

                dist_results.append({
                    'au': au,
                    'of22_mean': of22_mean,
                    'of30_mean': of30_mean,
                    'ratio': ratio,
                    'of22_std': of22_std,
                    'of30_std': of30_std
                })

        self.results['distributions'][side] = dist_results
        return dist_results

    def calculate_correlations(self, side='left'):
        """Calculate correlations between OF2.2 and OF3.0 for each AU"""
        print("\n" + "="*80)
        print(f"ANALYSIS 3: Correlations ({side.upper()} side)")
        print("="*80)

        of22_data = self.of22_left if side == 'left' else self.of22_right
        of30_data = self.of30_left if side == 'left' else self.of30_right

        # Align frames (use minimum length)
        min_len = min(len(of22_data), len(of30_data))
        of22_data = of22_data.iloc[:min_len]
        of30_data = of30_data.iloc[:min_len]

        print(f"\nUsing {min_len} frames for correlation analysis")
        print(f"\n{'AU':<10} {'Pearson r':<12} {'P-value':<12} {'Spearman ρ':<12} {'Interpretation':<20}")
        print("-"*80)

        corr_results = []

        for au in self.common_aus:
            # Get aligned values, drop NaN pairs
            of22_vals = of22_data[au]
            of30_vals = of30_data[au]

            # Create mask for valid pairs
            valid_mask = (~of22_vals.isna()) & (~of30_vals.isna())

            if valid_mask.sum() > 10:  # Need at least 10 valid pairs
                of22_valid = of22_vals[valid_mask]
                of30_valid = of30_vals[valid_mask]

                # Pearson correlation (linear relationship)
                pearson_r, pearson_p = pearsonr(of22_valid, of30_valid)

                # Spearman correlation (monotonic relationship)
                spearman_r, spearman_p = spearmanr(of22_valid, of30_valid)

                # Interpret
                if abs(pearson_r) >= 0.7:
                    interp = "✓ Strong"
                elif abs(pearson_r) >= 0.5:
                    interp = "~ Moderate"
                elif abs(pearson_r) >= 0.3:
                    interp = "⚠ Weak"
                else:
                    interp = "✗ Very weak/None"

                print(f"{au:<10} {pearson_r:<12.3f} {pearson_p:<12.4f} {spearman_r:<12.3f} {interp:<20}")

                corr_results.append({
                    'au': au,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'interpretation': interp,
                    'n_pairs': valid_mask.sum()
                })
            else:
                print(f"{au:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'Insufficient data':<20}")

        self.results['correlations'][side] = corr_results
        return corr_results

    def analyze_symmetry(self, version='of30'):
        """Analyze left vs right symmetry (should be similar for non-paralyzed patient)"""
        print("\n" + "="*80)
        print(f"ANALYSIS 4: Left vs Right Symmetry ({version.upper()})")
        print("="*80)
        print("\n(Non-paralyzed patient: expect high symmetry)")

        if version == 'of30':
            left_data = self.of30_left
            right_data = self.of30_right
            au_cols = self.of30_au_cols
        else:
            left_data = self.of22_left
            right_data = self.of22_right
            au_cols = self.of22_au_cols

        # Align frames
        min_len = min(len(left_data), len(right_data))
        left_data = left_data.iloc[:min_len]
        right_data = right_data.iloc[:min_len]

        print(f"\n{'AU':<10} {'Left Mean':<12} {'Right Mean':<12} {'Asymmetry':<12} {'L-R Corr':<12} {'Status':<15}")
        print("-"*80)

        symmetry_results = []

        for au in au_cols:
            left_vals = left_data[au].dropna()
            right_vals = right_data[au].dropna()

            if len(left_vals) > 0 and len(right_vals) > 0:
                left_mean = left_vals.mean()
                right_mean = right_vals.mean()

                # Calculate asymmetry ratio (should be ~1.0 for symmetric)
                if right_mean > 0.01:
                    asymmetry = left_mean / right_mean
                else:
                    asymmetry = np.nan

                # Calculate correlation between left and right (temporal pattern)
                valid_mask = (~left_data[au].isna()) & (~right_data[au].isna())
                if valid_mask.sum() > 10:
                    lr_corr, _ = pearsonr(left_data[au][valid_mask], right_data[au][valid_mask])
                else:
                    lr_corr = np.nan

                # Assess symmetry
                if not np.isnan(asymmetry):
                    if 0.8 <= asymmetry <= 1.25:
                        status = "✓ Symmetric"
                    elif 0.5 <= asymmetry <= 2.0:
                        status = "~ Mild asymmetry"
                    else:
                        status = "✗ Asymmetric"
                else:
                    status = "? Insufficient data"

                asym_str = f"{asymmetry:.2f}" if not np.isnan(asymmetry) else "N/A"
                lr_corr_str = f"{lr_corr:.2f}" if not np.isnan(lr_corr) else "N/A"

                print(f"{au:<10} {left_mean:<12.3f} {right_mean:<12.3f} {asym_str:<12} {lr_corr_str:<12} {status:<15}")

                symmetry_results.append({
                    'au': au,
                    'left_mean': left_mean,
                    'right_mean': right_mean,
                    'asymmetry_ratio': asymmetry,
                    'lr_correlation': lr_corr,
                    'status': status
                })

        self.results['symmetry'][version] = symmetry_results
        return symmetry_results

    def identify_clinical_actions(self):
        """Identify clinical actions (smile, brow raise) in the video"""
        print("\n" + "="*80)
        print("ANALYSIS 5: Clinical Action Detection")
        print("="*80)

        # Define clinical action signatures
        actions = {
            'smile': {
                'aus': ['AU12_r', 'AU06_r'],  # Lip corner puller, cheek raiser
                'threshold_percentile': 75  # Top 25% of values = active
            },
            'brow_raise': {
                'aus': ['AU01_r', 'AU02_r'],  # Inner/outer brow raiser
                'threshold_percentile': 75
            },
            'lip_stretch': {
                'aus': ['AU20_r'],  # Lip stretcher
                'threshold_percentile': 75
            }
        }

        print("\nDetecting clinical actions in video...")

        action_results = {}

        for action_name, config in actions.items():
            print(f"\n{action_name.upper().replace('_', ' ')}:")

            # Check both versions
            for version_name, left_data, right_data in [
                ('OF2.2', self.of22_left, self.of22_right),
                ('OF3.0', self.of30_left, self.of30_right)
            ]:
                print(f"\n  {version_name}:")

                action_detected = False
                au_activations = {}

                for au in config['aus']:
                    if au in left_data.columns:
                        # Calculate threshold for "active" frames
                        left_vals = left_data[au].dropna()
                        right_vals = right_data[au].dropna()

                        if len(left_vals) > 0:
                            threshold_left = np.percentile(left_vals, config['threshold_percentile'])
                            active_frames_left = (left_data[au] > threshold_left).sum()
                            active_pct_left = active_frames_left / len(left_data) * 100
                        else:
                            active_pct_left = 0

                        if len(right_vals) > 0:
                            threshold_right = np.percentile(right_vals, config['threshold_percentile'])
                            active_frames_right = (right_data[au] > threshold_right).sum()
                            active_pct_right = active_frames_right / len(right_data) * 100
                        else:
                            active_pct_right = 0

                        au_activations[au] = {
                            'left_active_pct': active_pct_left,
                            'right_active_pct': active_pct_right
                        }

                        if active_pct_left > 10 or active_pct_right > 10:
                            action_detected = True

                        print(f"    {au}: Left {active_pct_left:.1f}%, Right {active_pct_right:.1f}% active")
                    else:
                        print(f"    {au}: Not available (NaN)")

                action_results[f"{action_name}_{version_name.lower()}"] = {
                    'detected': action_detected,
                    'au_activations': au_activations
                }

        self.results['clinical_actions'] = action_results
        return action_results

    def plot_au_timeseries(self, au='AU12_r', save_path=None):
        """Plot AU timeseries comparison for visual inspection"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{au} Temporal Comparison: OF2.2 vs OF3.0', fontsize=16, fontweight='bold')

        # Left side - OF2.2 vs OF3.0
        ax = axes[0, 0]
        min_len = min(len(self.of22_left), len(self.of30_left))
        frames = np.arange(min_len)

        if au in self.of22_left.columns:
            ax.plot(frames, self.of22_left[au].iloc[:min_len], label='OF2.2', alpha=0.7, linewidth=2)
        if au in self.of30_left.columns:
            ax.plot(frames, self.of30_left[au].iloc[:min_len], label='OF3.0', alpha=0.7, linewidth=2)

        ax.set_title('Left Side', fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('AU Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right side - OF2.2 vs OF3.0
        ax = axes[0, 1]
        min_len = min(len(self.of22_right), len(self.of30_right))
        frames = np.arange(min_len)

        if au in self.of22_right.columns:
            ax.plot(frames, self.of22_right[au].iloc[:min_len], label='OF2.2', alpha=0.7, linewidth=2)
        if au in self.of30_right.columns:
            ax.plot(frames, self.of30_right[au].iloc[:min_len], label='OF3.0', alpha=0.7, linewidth=2)

        ax.set_title('Right Side', fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('AU Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Left vs Right - OF2.2
        ax = axes[1, 0]
        min_len = min(len(self.of22_left), len(self.of22_right))
        frames = np.arange(min_len)

        if au in self.of22_left.columns:
            ax.plot(frames, self.of22_left[au].iloc[:min_len], label='Left', alpha=0.7, linewidth=2)
            ax.plot(frames, self.of22_right[au].iloc[:min_len], label='Right', alpha=0.7, linewidth=2)

        ax.set_title('OF2.2: Left vs Right Symmetry', fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('AU Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Left vs Right - OF3.0
        ax = axes[1, 1]
        min_len = min(len(self.of30_left), len(self.of30_right))
        frames = np.arange(min_len)

        if au in self.of30_left.columns:
            ax.plot(frames, self.of30_left[au].iloc[:min_len], label='Left', alpha=0.7, linewidth=2)
            ax.plot(frames, self.of30_right[au].iloc[:min_len], label='Right', alpha=0.7, linewidth=2)

        ax.set_title('OF3.0: Left vs Right Symmetry', fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('AU Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to: {save_path}")

        return fig

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)

        print("\n📋 Key Findings:")
        print("-" * 40)

        # 1. AU Availability
        print("\n1. AU AVAILABILITY:")
        available = len(self.results['availability']['of30_available'])
        missing = len(self.results['availability']['missing_in_of30'])
        print(f"   - OF3.0 provides {available} AUs with values")
        print(f"   - OF3.0 missing {missing} AUs that OF2.2 has")
        print(f"   - Missing AUs: {', '.join(self.results['availability']['missing_in_of30'])}")

        # 2. Correlations
        print("\n2. VERSION CORRELATIONS:")
        if 'left' in self.results['correlations']:
            corrs = self.results['correlations']['left']
            strong = sum(1 for c in corrs if abs(c['pearson_r']) >= 0.7)
            moderate = sum(1 for c in corrs if 0.5 <= abs(c['pearson_r']) < 0.7)
            weak = sum(1 for c in corrs if abs(c['pearson_r']) < 0.5)

            print(f"   - Strong correlations (r≥0.7): {strong}/{len(corrs)} AUs")
            print(f"   - Moderate correlations (0.5≤r<0.7): {moderate}/{len(corrs)} AUs")
            print(f"   - Weak correlations (r<0.5): {weak}/{len(corrs)} AUs")

        # 3. Symmetry (for non-paralyzed patient)
        print("\n3. SYMMETRY ANALYSIS (Non-paralyzed patient):")
        for version in ['of30', 'of22']:
            if version in self.results['symmetry']:
                syms = self.results['symmetry'][version]
                symmetric = sum(1 for s in syms if '✓' in s['status'])
                print(f"   - {version.upper()}: {symmetric}/{len(syms)} AUs show good symmetry")

        # 4. Clinical Actions
        print("\n4. CLINICAL ACTIONS DETECTED:")
        if self.results['clinical_actions']:
            for action_key, result in self.results['clinical_actions'].items():
                if result['detected']:
                    print(f"   ✓ {action_key.replace('_', ' ').title()}")

        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)

        # Provide recommendations based on results
        if 'left' in self.results['correlations']:
            avg_corr = np.mean([abs(c['pearson_r']) for c in self.results['correlations']['left']])

            print(f"\nAverage correlation between OF2.2 and OF3.0: {avg_corr:.3f}")

            if avg_corr < 0.3:
                print("\n⚠️  WEAK CORRELATION - OF3.0 produces fundamentally different AU values")
                print("    → Recommendation: Investigate OF2.2 Python migration OR retrain models on OF3.0")
            elif avg_corr < 0.6:
                print("\n⚠️  MODERATE CORRELATION - Some agreement but significant differences")
                print("    → Recommendation: Clinical validation needed before using OF3.0")
            else:
                print("\n✓ STRONG CORRELATION - OF3.0 tracks OF2.2 reasonably well")
                print("    → Recommendation: OF3.0 may be suitable with recalibration")


def main():
    """Main analysis workflow"""

    # Paths to CSV files
    base_path = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22")

    of22_left = base_path / "IMG_0942_left_mirroredOP22.csv"
    of22_right = base_path / "IMG_0942_right_mirroredOP22.csv"
    of30_left = base_path / "IMG_0942_left_mirroredONNXv3.csv"
    of30_right = base_path / "IMG_0942_right_mirroredONNXv3.csv"

    # Create comparer
    comparer = OpenFaceComparer(of22_left, of22_right, of30_left, of30_right)

    # Run analyses
    comparer.analyze_au_availability()

    comparer.compare_distributions('left')
    comparer.compare_distributions('right')

    comparer.calculate_correlations('left')
    comparer.calculate_correlations('right')

    comparer.analyze_symmetry('of30')
    comparer.analyze_symmetry('of22')

    comparer.identify_clinical_actions()

    # Generate plots for key AUs
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror")

    for au in ['AU12_r', 'AU01_r', 'AU06_r']:
        if au in comparer.common_aus:
            print(f"\nPlotting {au}...")
            save_path = output_dir / f"comparison_{au}.png"
            comparer.plot_au_timeseries(au, save_path=save_path)

    # Final summary
    comparer.generate_summary_report()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
