#!/usr/bin/env python3
"""
Compare OpenFace 3.0 (with skip_face_detection fix) vs OpenFace 2.2 results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# File paths
OF30_LEFT = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_left_mirrored.csv")
OF30_RIGHT = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data/IMG_0942_right_mirrored.csv")
OF22_LEFT = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22/IMG_0942_left_mirroredOP22.csv")
OF22_RIGHT = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/OP3 v OP 22/IMG_0942_right_mirroredOP22.csv")

OUTPUT_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/comparison_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("OPENFACE 3.0 vs OPENFACE 2.2 COMPARISON")
print("=" * 80)
print()

# Load CSVs
print("Loading CSV files...")
df_of30_left = pd.read_csv(OF30_LEFT)
df_of30_right = pd.read_csv(OF30_RIGHT)
df_of22_left = pd.read_csv(OF22_LEFT)
df_of22_right = pd.read_csv(OF22_RIGHT)

print(f"✓ OF3.0 Left:  {len(df_of30_left)} frames")
print(f"✓ OF3.0 Right: {len(df_of30_right)} frames")
print(f"✓ OF2.2 Left:  {len(df_of22_left)} frames")
print(f"✓ OF2.2 Right: {len(df_of22_right)} frames")
print()

# Key AUs to compare
key_aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU20_r', 'AU25_r']

def compare_side(df_of30, df_of22, side_name):
    """Compare OF3.0 vs OF2.2 for one side"""

    print(f"\n{'='*80}")
    print(f"{side_name.upper()} SIDE COMPARISON")
    print(f"{'='*80}\n")

    # Find common AUs
    available_aus = [au for au in key_aus if au in df_of30.columns and au in df_of22.columns]

    if not available_aus:
        print(f"⚠ No common AUs found for {side_name}")
        return None

    print(f"Comparing {len(available_aus)} AUs: {', '.join(available_aus)}\n")

    # Align lengths
    min_len = min(len(df_of30), len(df_of22))

    # Compute statistics for each AU
    stats = []

    for au in available_aus:
        of30_values = df_of30[au].values[:min_len]
        of22_values = df_of22[au].values[:min_len]

        # Remove NaNs
        valid_mask = ~np.isnan(of30_values) & ~np.isnan(of22_values)

        if np.sum(valid_mask) < 10:
            print(f"{au}: Insufficient valid data")
            continue

        of30_valid = of30_values[valid_mask]
        of22_valid = of22_values[valid_mask]

        # Compute metrics
        correlation = np.corrcoef(of30_valid, of22_valid)[0, 1]
        mae = np.mean(np.abs(of30_valid - of22_valid))
        rmse = np.sqrt(np.mean((of30_valid - of22_valid) ** 2))

        of30_mean = np.mean(of30_valid)
        of30_max = np.max(of30_valid)
        of30_std = np.std(of30_valid)

        of22_mean = np.mean(of22_valid)
        of22_max = np.max(of22_valid)
        of22_std = np.std(of22_valid)

        stats.append({
            'AU': au,
            'Correlation': correlation,
            'MAE': mae,
            'RMSE': rmse,
            'OF30_Mean': of30_mean,
            'OF30_Max': of30_max,
            'OF30_Std': of30_std,
            'OF22_Mean': of22_mean,
            'OF22_Max': of22_max,
            'OF22_Std': of22_std,
        })

        # Print summary
        print(f"{au}:")
        print(f"  OF2.2: mean={of22_mean:.3f}, max={of22_max:.3f}, std={of22_std:.3f}")
        print(f"  OF3.0: mean={of30_mean:.3f}, max={of30_max:.3f}, std={of30_std:.3f}")
        print(f"  Correlation: r={correlation:.3f}")
        print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        if correlation > 0.8:
            print(f"  ✓ EXCELLENT correlation")
        elif correlation > 0.6:
            print(f"  ✓ GOOD correlation")
        elif correlation > 0.4:
            print(f"  ⚠ MODERATE correlation")
        else:
            print(f"  ✗ POOR correlation")
        print()

    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

    for idx, au in enumerate(available_aus[:8]):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        if au not in df_of30.columns or au not in df_of22.columns:
            ax.axis('off')
            continue

        of30_values = df_of30[au].values[:min_len]
        of22_values = df_of22[au].values[:min_len]
        frames = np.arange(min_len)

        ax.plot(frames, of22_values, 'b-', label='OF2.2', linewidth=2, alpha=0.7)
        ax.plot(frames, of30_values, 'r-', label='OF3.0', linewidth=2, alpha=0.7)

        ax.set_title(au, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Intensity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add correlation text
        valid_mask = ~np.isnan(of30_values) & ~np.isnan(of22_values)
        if np.sum(valid_mask) > 0:
            corr = np.corrcoef(of30_values[valid_mask], of22_values[valid_mask])[0, 1]
            color = 'green' if corr > 0.7 else 'orange' if corr > 0.5 else 'red'
            ax.text(0.98, 0.98, f'r={corr:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.5),
                   fontsize=10, fontweight='bold')

    plt.suptitle(f'OF2.2 vs OF3.0 (FIXED) - {side_name.upper()} Side',
                fontsize=14, fontweight='bold')

    output_path = OUTPUT_DIR / f"comparison_{side_name.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization: {output_path}\n")

    return pd.DataFrame(stats)


# Compare both sides
stats_left = compare_side(df_of30_left, df_of22_left, "Left")
stats_right = compare_side(df_of30_right, df_of22_right, "Right")

# Create detailed AU12_r comparison (the problematic AU)
print("\n" + "="*80)
print("DETAILED AU12_r ANALYSIS (Most Important)")
print("="*80 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('AU12_r Detailed Comparison: OF2.2 vs OF3.0 (FIXED)', fontsize=14, fontweight='bold')

for idx, (df_of30, df_of22, side) in enumerate([
    (df_of30_left, df_of22_left, "Left"),
    (df_of30_right, df_of22_right, "Right")
]):
    if 'AU12_r' not in df_of30.columns or 'AU12_r' not in df_of22.columns:
        continue

    min_len = min(len(df_of30), len(df_of22))
    of30_au12 = df_of30['AU12_r'].values[:min_len]
    of22_au12 = df_of22['AU12_r'].values[:min_len]
    frames = np.arange(min_len)

    # Temporal plot
    ax_temporal = axes[0, idx]
    ax_temporal.plot(frames, of22_au12, 'b-', label='OF2.2', linewidth=2)
    ax_temporal.plot(frames, of30_au12, 'r-', label='OF3.0 (FIXED)', linewidth=2, alpha=0.7)
    ax_temporal.set_xlabel('Frame', fontsize=11)
    ax_temporal.set_ylabel('AU12_r Intensity', fontsize=11)
    ax_temporal.set_title(f'{side} Side - Temporal Pattern', fontsize=12, fontweight='bold')
    ax_temporal.legend(fontsize=10)
    ax_temporal.grid(True, alpha=0.3)

    # Add statistics
    valid = ~np.isnan(of30_au12) & ~np.isnan(of22_au12)
    if np.sum(valid) > 0:
        corr = np.corrcoef(of30_au12[valid], of22_au12[valid])[0, 1]
        mae = np.mean(np.abs(of30_au12[valid] - of22_au12[valid]))

        stats_text = f'r={corr:.3f}\nMAE={mae:.3f}'
        color = 'green' if corr > 0.7 else 'orange' if corr > 0.5 else 'red'
        ax_temporal.text(0.02, 0.98, stats_text,
                        transform=ax_temporal.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                        fontsize=11, fontweight='bold')

    # Scatter plot
    ax_scatter = axes[1, idx]
    valid = ~np.isnan(of30_au12) & ~np.isnan(of22_au12)
    ax_scatter.scatter(of22_au12[valid], of30_au12[valid], alpha=0.5, s=20)

    # Perfect correlation line
    max_val = max(np.max(of22_au12[valid]), np.max(of30_au12[valid]))
    ax_scatter.plot([0, max_val], [0, max_val], 'k--', label='Perfect correlation', linewidth=2)

    ax_scatter.set_xlabel('OF2.2 AU12_r', fontsize=11)
    ax_scatter.set_ylabel('OF3.0 AU12_r', fontsize=11)
    ax_scatter.set_title(f'{side} Side - Correlation', fontsize=12, fontweight='bold')
    ax_scatter.legend(fontsize=10)
    ax_scatter.grid(True, alpha=0.3)

    # Print detailed stats
    print(f"{side} Side AU12_r:")
    print(f"  OF2.2: mean={np.mean(of22_au12[valid]):.3f}, max={np.max(of22_au12[valid]):.3f}")
    print(f"  OF3.0: mean={np.mean(of30_au12[valid]):.3f}, max={np.max(of30_au12[valid]):.3f}")
    print(f"  Correlation: r={corr:.3f}")
    print(f"  MAE: {mae:.3f}")
    print()

plt.tight_layout()
output_path = OUTPUT_DIR / "au12_detailed_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved AU12_r detailed comparison: {output_path}\n")

# Summary statistics table
print("="*80)
print("SUMMARY - AVERAGE CORRELATIONS")
print("="*80)

if stats_left is not None:
    avg_corr_left = stats_left['Correlation'].mean()
    print(f"Left Side:  Average correlation = {avg_corr_left:.3f}")

if stats_right is not None:
    avg_corr_right = stats_right['Correlation'].mean()
    print(f"Right Side: Average correlation = {avg_corr_right:.3f}")

print()
print("="*80)
print("CONCLUSION")
print("="*80)

if stats_left is not None and stats_right is not None:
    avg_corr = (avg_corr_left + avg_corr_right) / 2

    print(f"\nOverall average correlation: {avg_corr:.3f}")
    print()

    if avg_corr > 0.75:
        print("✓ EXCELLENT: OF3.0 is producing results very similar to OF2.2!")
        print("  The skip_face_detection fix is working properly.")
    elif avg_corr > 0.6:
        print("✓ GOOD: OF3.0 shows strong correlation with OF2.2")
        print("  Results are usable, but some minor differences remain.")
    elif avg_corr > 0.4:
        print("⚠ MODERATE: OF3.0 shows moderate correlation with OF2.2")
        print("  Further investigation may be needed.")
    else:
        print("✗ POOR: OF3.0 results still differ significantly from OF2.2")
        print("  Additional debugging required - the RetinaFace fix alone wasn't enough.")

print()
print(f"All visualizations saved to: {OUTPUT_DIR}")
print("="*80)
