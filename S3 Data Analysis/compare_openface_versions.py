"""
Compare OpenFace 2.2 and 3.0 datasets to understand feature changes and data distributions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# File paths
OF2_PATH = "/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/combined_results.csv"
OF3_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S3O Results/combined_results.csv"
OUTPUT_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis/openface_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_unique_aus(df):
    """Extract unique AU names from column headers"""
    au_cols = [col for col in df.columns if '_AU' in col and '_r' in col and 'Normalized' not in col]
    # Extract just the AU number (e.g., 'BC_Left AU01_r' -> 'AU01')
    aus = set()
    for col in au_cols:
        au_part = col.split('AU')[1].split('_')[0]
        aus.add(f'AU{au_part}')
    return sorted(aus)

def compare_au_availability(df_old, df_new):
    """Compare which AUs are available in each dataset"""
    aus_old = extract_unique_aus(df_old)
    aus_new = extract_unique_aus(df_new)

    print("=" * 80)
    print("AU AVAILABILITY COMPARISON")
    print("=" * 80)
    print(f"\nOpenFace 2.2 AUs ({len(aus_old)}): {', '.join(aus_old)}")
    print(f"\nOpenFace 3.0 AUs ({len(aus_new)}): {', '.join(aus_new)}")

    removed = set(aus_old) - set(aus_new)
    added = set(aus_new) - set(aus_old)
    common = set(aus_old) & set(aus_new)

    if removed:
        print(f"\n❌ REMOVED in 3.0 ({len(removed)}): {', '.join(sorted(removed))}")
    if added:
        print(f"\n✅ ADDED in 3.0 ({len(added)}): {', '.join(sorted(added))}")
    print(f"\n✓ COMMON ({len(common)}): {', '.join(sorted(common))}")

    return aus_old, aus_new, common

def get_au_columns_for_condition(df, condition, au_num, side=None):
    """Get AU column names for a specific condition"""
    cols = []
    if side:
        # Specific side
        raw_col = f"{condition}_{side} AU{au_num}_r"
        norm_col = f"{condition}_{side} AU{au_num}_r (Normalized)"
    else:
        # Both sides
        raw_col_left = f"{condition}_Left AU{au_num}_r"
        raw_col_right = f"{condition}_Right AU{au_num}_r"
        norm_col_left = f"{condition}_Left AU{au_num}_r (Normalized)"
        norm_col_right = f"{condition}_Right AU{au_num}_r (Normalized)"

        cols = [c for c in [raw_col_left, raw_col_right, norm_col_left, norm_col_right] if c in df.columns]
        return cols

    cols = [c for c in [raw_col, norm_col] if c in df.columns]
    return cols

def compare_au_distributions(df_old, df_new, au_num, condition='BC'):
    """Compare distributions of a specific AU between datasets"""
    # Get data for left and right sides
    cols_old_left = get_au_columns_for_condition(df_old, condition, au_num, 'Left')
    cols_old_right = get_au_columns_for_condition(df_old, condition, au_num, 'Right')
    cols_new_left = get_au_columns_for_condition(df_new, condition, au_num, 'Left')
    cols_new_right = get_au_columns_for_condition(df_new, condition, au_num, 'Right')

    if not cols_old_left or not cols_new_left:
        return None

    # Use raw values (not normalized)
    old_left = df_old[cols_old_left[0]].dropna()
    old_right = df_old[cols_old_right[0]].dropna()
    new_left = df_new[cols_new_left[0]].dropna()
    new_right = df_new[cols_new_right[0]].dropna()

    stats = {
        'AU': f'AU{au_num}',
        'Condition': condition,
        'OF2.2 Left Mean': old_left.mean(),
        'OF2.2 Left Std': old_left.std(),
        'OF2.2 Right Mean': old_right.mean(),
        'OF2.2 Right Std': old_right.std(),
        'OF3.0 Left Mean': new_left.mean(),
        'OF3.0 Left Std': new_left.std(),
        'OF3.0 Right Mean': new_right.mean(),
        'OF3.0 Right Std': new_right.std(),
        'Left Mean Delta': new_left.mean() - old_left.mean(),
        'Right Mean Delta': new_right.mean() - old_right.mean(),
    }

    return stats, (old_left, old_right), (new_left, new_right)

def analyze_au6_au7_relationship(df_old, df_new):
    """Analyze the relationship between AU6 and AU7 (critical for midface)"""
    print("\n" + "=" * 80)
    print("AU6 vs AU7 ANALYSIS (Critical for Midface Detection)")
    print("=" * 80)

    conditions = ['BC', 'SE', 'SO']  # Brow Cocked, Snarl, Soft Smile (conditions that activate midface)

    results = []
    for condition in conditions:
        # OpenFace 2.2: Get AU6 and AU7
        au6_old_left = df_old.get(f"{condition}_Left AU06_r")
        au7_old_left = df_old.get(f"{condition}_Left AU07_r")
        au6_old_right = df_old.get(f"{condition}_Right AU06_r")
        au7_old_right = df_old.get(f"{condition}_Right AU07_r")

        # OpenFace 3.0: Get AU6 and AU7
        au6_new_left = df_new.get(f"{condition}_Left AU06_r")
        au7_new_left = df_new.get(f"{condition}_Left AU07_r")
        au6_new_right = df_new.get(f"{condition}_Right AU06_r")
        au7_new_right = df_new.get(f"{condition}_Right AU07_r")

        if au6_old_left is not None and au7_old_left is not None:
            # Correlation between AU6 and AU7 in OLD dataset
            corr_old = au6_old_left.corr(au7_old_left) if len(au6_old_left.dropna()) > 1 else np.nan

            # Compare AU6 values between datasets
            au6_mean_old = au6_old_left.mean()
            au6_mean_new = au6_new_left.mean() if au6_new_left is not None else np.nan

            # Compare AU7 values between datasets
            au7_mean_old = au7_old_left.mean()
            au7_mean_new = au7_new_left.mean() if au7_new_left is not None else np.nan

            results.append({
                'Condition': condition,
                'AU6-AU7 Correlation (OF2.2)': corr_old,
                'AU6 Mean (OF2.2)': au6_mean_old,
                'AU6 Mean (OF3.0)': au6_mean_new,
                'AU6 Delta': au6_mean_new - au6_mean_old,
                'AU7 Mean (OF2.2)': au7_mean_old,
                'AU7 Mean (OF3.0)': au7_mean_new,
                'AU7 Delta': au7_mean_new - au7_mean_old,
            })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    # Save to CSV
    results_df.to_csv(OUTPUT_DIR / "au6_au7_comparison.csv", index=False)
    print(f"\n✓ Saved to: {OUTPUT_DIR / 'au6_au7_comparison.csv'}")

    return results_df

def analyze_au45_ear_relationship(df_old, df_new):
    """Analyze AU45 changes (traditional vs EAR-based)"""
    print("\n" + "=" * 80)
    print("AU45 ANALYSIS (Traditional vs EAR-based)")
    print("=" * 80)

    # Conditions where blink/eye closure is relevant
    conditions = ['BK', 'ES', 'ET']  # Blink, Eyes Shut, Eyes Tight

    results = []
    for condition in conditions:
        au45_old_left = df_old.get(f"{condition}_Left AU45_r")
        au45_new_left = df_new.get(f"{condition}_Left AU45_r")
        au45_old_right = df_old.get(f"{condition}_Right AU45_r")
        au45_new_right = df_new.get(f"{condition}_Right AU45_r")

        if au45_old_left is not None and au45_new_left is not None:
            results.append({
                'Condition': condition,
                'AU45 Mean (OF2.2 Left)': au45_old_left.mean(),
                'AU45 Mean (OF3.0 Left)': au45_new_left.mean(),
                'AU45 Mean (OF2.2 Right)': au45_old_right.mean(),
                'AU45 Mean (OF3.0 Right)': au45_new_right.mean(),
                'Left Delta': au45_new_left.mean() - au45_old_left.mean(),
                'Right Delta': au45_new_right.mean() - au45_old_right.mean(),
                'Correlation (Left)': au45_old_left.corr(au45_new_left) if len(au45_old_left.dropna()) > 1 else np.nan,
            })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_DIR / "au45_comparison.csv", index=False)
    print(f"\n✓ Saved to: {OUTPUT_DIR / 'au45_comparison.csv'}")

    return results_df

def compare_patient_overlap(df_old, df_new):
    """Check which patients are in both datasets"""
    print("\n" + "=" * 80)
    print("PATIENT OVERLAP ANALYSIS")
    print("=" * 80)

    patients_old = set(df_old['Patient ID'].dropna())
    patients_new = set(df_new['Patient ID'].dropna())

    overlap = patients_old & patients_new
    only_old = patients_old - patients_new
    only_new = patients_new - patients_old

    print(f"\nTotal patients in OF2.2: {len(patients_old)}")
    print(f"Total patients in OF3.0: {len(patients_new)}")
    print(f"Overlap: {len(overlap)}")
    print(f"Only in OF2.2: {len(only_old)}")
    print(f"Only in OF3.0: {len(only_new)}")

    if overlap:
        print(f"\n✓ Can do direct comparison on {len(overlap)} shared patients")
        print(f"Shared patients: {sorted(overlap)}")

    return overlap

def generate_distribution_plots(df_old, df_new, common_aus, overlap_patients):
    """Generate distribution comparison plots for common AUs"""
    print("\n" + "=" * 80)
    print("GENERATING DISTRIBUTION PLOTS")
    print("=" * 80)

    # Filter to only overlapping patients for fair comparison
    if overlap_patients:
        df_old_filtered = df_old[df_old['Patient ID'].isin(overlap_patients)].copy()
        df_new_filtered = df_new[df_new['Patient ID'].isin(overlap_patients)].copy()
    else:
        df_old_filtered = df_old.copy()
        df_new_filtered = df_new.copy()

    # Focus on key conditions
    conditions = ['BC', 'SE', 'SO', 'ES']  # Brow Cocked, Snarl, Soft Smile, Eyes Shut

    # Plot for each AU
    for au in common_aus:
        au_num = au.replace('AU', '')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{au} Distribution Comparison: OpenFace 2.2 vs 3.0', fontsize=14, fontweight='bold')

        for idx, condition in enumerate(conditions):
            ax = axes[idx // 2, idx % 2]

            # Get left side data (raw values)
            col_old = f"{condition}_Left {au}_r"
            col_new = f"{condition}_Left {au}_r"

            if col_old in df_old_filtered.columns and col_new in df_new_filtered.columns:
                old_vals = df_old_filtered[col_old].dropna()
                new_vals = df_new_filtered[col_new].dropna()

                if len(old_vals) > 0 and len(new_vals) > 0:
                    # Create histogram
                    ax.hist(old_vals, bins=20, alpha=0.5, label=f'OF2.2 (μ={old_vals.mean():.2f})', color='blue')
                    ax.hist(new_vals, bins=20, alpha=0.5, label=f'OF3.0 (μ={new_vals.mean():.2f})', color='orange')
                    ax.set_xlabel(f'{au} Intensity')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Condition: {condition}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / f'{au}_distribution_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path.name}")

def main():
    print("\n" + "=" * 80)
    print("OPENFACE 2.2 vs 3.0 COMPREHENSIVE COMPARISON")
    print("=" * 80)

    # Load datasets
    print("\nLoading datasets...")
    df_old = pd.read_csv(OF2_PATH)
    df_new = pd.read_csv(OF3_PATH)
    print(f"✓ OpenFace 2.2: {len(df_old)} patients")
    print(f"✓ OpenFace 3.0: {len(df_new)} patients")

    # Compare AU availability
    aus_old, aus_new, common_aus = compare_au_availability(df_old, df_new)

    # Check patient overlap
    overlap = compare_patient_overlap(df_old, df_new)

    # Analyze AU6 vs AU7 (critical for midface)
    au6_au7_analysis = analyze_au6_au7_relationship(df_old, df_new)

    # Analyze AU45 changes (traditional vs EAR)
    au45_analysis = analyze_au45_ear_relationship(df_old, df_new)

    # Generate comprehensive statistics for all common AUs
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE AU STATISTICS")
    print("=" * 80)

    all_stats = []
    for au in common_aus:
        au_num = au.replace('AU', '')
        for condition in ['BC', 'SE', 'SO', 'ES', 'BK']:
            result = compare_au_distributions(df_old, df_new, au_num, condition)
            if result:
                stats, _, _ = result
                all_stats.append(stats)

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(OUTPUT_DIR / "comprehensive_au_statistics.csv", index=False)
        print(f"\n✓ Saved comprehensive statistics to: {OUTPUT_DIR / 'comprehensive_au_statistics.csv'}")

        # Show biggest changes
        stats_df['Avg_Mean_Delta'] = (abs(stats_df['Left Mean Delta']) + abs(stats_df['Right Mean Delta'])) / 2
        biggest_changes = stats_df.nlargest(10, 'Avg_Mean_Delta')[['AU', 'Condition', 'Left Mean Delta', 'Right Mean Delta', 'Avg_Mean_Delta']]
        print("\nTop 10 Biggest AU Changes:")
        print(biggest_changes.to_string(index=False))

    # Generate distribution plots
    generate_distribution_plots(df_old, df_new, list(common_aus)[:5], overlap)  # Plot first 5 AUs to avoid too many plots

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nKey Files:")
    print(f"  - au6_au7_comparison.csv: Critical midface AU comparison")
    print(f"  - au45_comparison.csv: Eye closure/blink comparison")
    print(f"  - comprehensive_au_statistics.csv: Full AU statistics")
    print(f"  - *.png: Distribution comparison plots")

if __name__ == "__main__":
    main()
