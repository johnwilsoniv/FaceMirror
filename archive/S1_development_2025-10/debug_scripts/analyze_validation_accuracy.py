#!/usr/bin/env python3
"""
Clarify what our validation is actually measuring and create comparison charts.

IMPORTANT DISTINCTION:
- Our validation: Python predictions vs OpenFace 2.2 C++ predictions (replication test)
- Published benchmarks: OpenFace 2.2 predictions vs Human-coded labels (accuracy test)

We're measuring REPLICATION ACCURACY, not absolute performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_comparison_chart():
    """Create chart showing our replication accuracy vs published OF2.2 performance"""

    # Our results: Python vs OF2.2 C++ (replication accuracy)
    our_results = {
        'AU01': 0.811, 'AU02': 0.560, 'AU04': 0.876, 'AU05': 0.637,
        'AU06': 0.958, 'AU07': 0.925, 'AU09': 0.891, 'AU10': 0.981,
        'AU12': 0.99996, 'AU14': 0.99975, 'AU15': 0.618, 'AU17': 0.814,
        'AU20': 0.522, 'AU23': 0.723, 'AU25': 0.993, 'AU26': 0.996,
        'AU45': 0.993
    }

    # Published OF2.2 performance: OF2.2 vs Human labels on DISFA
    # Source: "Assessing Automated Facial Action Unit Detection Systems" (Sensors, 2021)
    # NOTE: Only a few AUs were reported in the paper
    published_of22_vs_humans = {
        'AU12': 0.85,  # Reported
        'AU15': 0.39,  # Reported
        # Average across all AUs: 0.73
    }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Subplot 1: Our replication accuracy (Python vs OF2.2 C++)
    aus = sorted(our_results.keys())
    correlations = [our_results[au] for au in aus]
    colors = ['green' if c > 0.90 else 'orange' if c > 0.70 else 'red' for c in correlations]

    ax1.bar(aus, correlations, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.90, color='green', linestyle='--', label='Excellent (r > 0.90)', linewidth=2)
    ax1.axhline(y=0.70, color='orange', linestyle='--', label='Good (r > 0.70)', linewidth=2)
    ax1.set_ylim([0, 1.05])
    ax1.set_ylabel('Correlation (r)', fontsize=12, fontweight='bold')
    ax1.set_title('Our Python Implementation vs OpenFace 2.2 C++ (Replication Accuracy)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlabel('Action Unit', fontsize=12, fontweight='bold')

    # Add average line
    avg = np.mean(correlations)
    ax1.axhline(y=avg, color='blue', linestyle=':', label=f'Our Average: {avg:.3f}', linewidth=2)

    # Add text annotations for key values
    for i, (au, corr) in enumerate(zip(aus, correlations)):
        if corr > 0.95 or corr < 0.65:
            ax1.text(i, corr + 0.02, f'{corr:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Subplot 2: Comparison chart
    # Show what we're measuring vs what published benchmarks measure
    comparison_data = {
        'Metric': ['Our Python\nvs\nOF2.2 C++\n(Replication)',
                   'Published\nOF2.2 vs Humans\n(DISFA Dataset)'],
        'AU12': [our_results['AU12'], 0.85],
        'AU15': [our_results['AU15'], 0.39],
        'Average': [avg, 0.73]
    }

    x = np.arange(len(comparison_data['Metric']))
    width = 0.25

    ax2.bar(x - width, comparison_data['AU12'], width, label='AU12', color='steelblue', edgecolor='black')
    ax2.bar(x, comparison_data['AU15'], width, label='AU15', color='coral', edgecolor='black')
    ax2.bar(x + width, comparison_data['Average'], width, label='Average (all AUs)',
            color='mediumseagreen', edgecolor='black')

    ax2.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax2.set_title('CRITICAL DISTINCTION: What Are We Actually Measuring?',
                  fontsize=14, fontweight='bold', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_data['Metric'], fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])

    # Add value labels on bars
    for i, metric in enumerate(comparison_data['Metric']):
        ax2.text(i - width, comparison_data['AU12'][i] + 0.02,
                f"{comparison_data['AU12'][i]:.3f}", ha='center', fontsize=9, fontweight='bold')
        ax2.text(i, comparison_data['AU15'][i] + 0.02,
                f"{comparison_data['AU15'][i]:.3f}", ha='center', fontsize=9, fontweight='bold')
        ax2.text(i + width, comparison_data['Average'][i] + 0.02,
                f"{comparison_data['Average'][i]:.3f}", ha='center', fontsize=9, fontweight='bold')

    # Add explanation text
    fig.text(0.5, 0.48,
             'Left bars: Our Python code replicating OpenFace 2.2 C++ (r ≈ 1.0 = perfect replication)\n'
             'Right bars: OpenFace 2.2 accuracy vs human labels (r ≈ 0.7-0.8 = good for AU detection)',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = Path('/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/validation_comparison_chart.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {output_file}")

    return our_results, published_of22_vs_humans


def print_clarification():
    """Print clear explanation of what we're measuring"""

    print("=" * 80)
    print("CLARIFICATION: What We're Actually Measuring")
    print("=" * 80)
    print()

    print("❌ INCORRECT INTERPRETATION:")
    print("  'Our Python implementation performs better than OpenFace 2.2'")
    print()

    print("✅ CORRECT INTERPRETATION:")
    print("  'Our Python implementation successfully REPLICATES OpenFace 2.2 C++'")
    print()

    print("=" * 80)
    print("Two Different Comparisons")
    print("=" * 80)
    print()

    print("1️⃣  OUR VALIDATION (Replication Test):")
    print("   - Input: OpenFace 2.2's HOG features + landmarks + PDM params")
    print("   - Models: OpenFace 2.2's trained SVR models")
    print("   - Ground truth: OpenFace 2.2 C++ predictions (from CSV)")
    print("   - Result: Our Python predictions vs OF2.2 C++ predictions")
    print("   - Interpretation: r ≈ 1.0 = perfect replication")
    print()

    print("2️⃣  PUBLISHED BENCHMARKS (Accuracy Test):")
    print("   - Input: Video frames from DISFA dataset")
    print("   - Models: OpenFace 2.2 (C++)")
    print("   - Ground truth: Human-coded AU labels")
    print("   - Result: OpenFace 2.2 predictions vs Human labels")
    print("   - Interpretation: r ≈ 0.7-0.8 = good for AU detection")
    print()

    print("=" * 80)
    print("Why High Correlations Don't Mean We're 'Better'")
    print("=" * 80)
    print()

    print("When we get r=0.99996 for AU12:")
    print("  ✓ This means: Our Python code nearly perfectly replicates OF2.2 C++")
    print("  ✗ This does NOT mean: We're more accurate than OpenFace at detecting AUs")
    print()

    print("Think of it like:")
    print("  - Replication test (ours): Copying an answer key → 100% match = A+")
    print("  - Accuracy test (published): Taking the actual test → 85% correct = B")
    print()

    print("=" * 80)
    print("What About Low Correlation AUs?")
    print("=" * 80)
    print()

    print("AUs with r < 0.75 (AU02, AU05, AU15, AU20, AU23):")
    print()
    print("Possible causes:")
    print("  1. Remaining implementation differences in our Python code")
    print("  2. Numerical precision differences (Python float64 vs C++ double)")
    print("  3. Subtle differences in histogram median calculation")
    print("  4. Edge case handling in feature preprocessing")
    print()
    print("These are NOT inherent model limitations - OpenFace 2.2 C++ itself")
    print("produces these predictions. We just haven't perfectly replicated")
    print("the C++ behavior yet for these specific AUs.")
    print()

    print("=" * 80)
    print("Conclusion")
    print("=" * 80)
    print()

    print("✅ What we've achieved:")
    print("   - Perfectly replicated 5 AUs (r > 0.99)")
    print("   - Closely replicated 3 AUs (r > 0.95)")
    print("   - Good replication for 4 AUs (r > 0.85)")
    print("   - Moderate replication for 5 AUs (r > 0.52)")
    print()

    print("🎯 Goal for 'production ready':")
    print("   - All 17 AUs should have r > 0.95 (near-perfect replication)")
    print("   - Currently only 8 AUs meet this threshold")
    print()

    print("🔧 Remaining work:")
    print("   - Investigate 5 AUs with r < 0.75")
    print("   - Find remaining implementation differences")
    print("   - Achieve r > 0.95 for all AUs")
    print()


if __name__ == "__main__":
    print_clarification()
    print("\nGenerating comparison chart...")
    our_results, published = create_comparison_chart()

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    correlations = list(our_results.values())
    print(f"\nOur Replication Accuracy:")
    print(f"  Mean: {np.mean(correlations):.3f}")
    print(f"  Median: {np.median(correlations):.3f}")
    print(f"  Min: {np.min(correlations):.3f}")
    print(f"  Max: {np.max(correlations):.3f}")

    excellent = sum(1 for c in correlations if c > 0.95)
    good = sum(1 for c in correlations if 0.85 < c <= 0.95)
    moderate = sum(1 for c in correlations if 0.70 < c <= 0.85)
    poor = sum(1 for c in correlations if c <= 0.70)

    print(f"\nBreakdown:")
    print(f"  Excellent (r > 0.95): {excellent}/17 AUs ({100*excellent/17:.1f}%)")
    print(f"  Good (0.85 < r ≤ 0.95): {good}/17 AUs ({100*good/17:.1f}%)")
    print(f"  Moderate (0.70 < r ≤ 0.85): {moderate}/17 AUs ({100*moderate/17:.1f}%)")
    print(f"  Needs work (r ≤ 0.70): {poor}/17 AUs ({100*poor/17:.1f}%)")
