#!/usr/bin/env python3
"""
Find the correct AU ordering by testing which arrangement
gives the best correlation with OpenFace 2.2 labels.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from itertools import permutations

# The 8 AUs that OpenFace 3.0 outputs (DISFA subset)
# We know these are the AUs, but not their order
DISFA_8_AUS = ['AU01', 'AU02', 'AU04', 'AU06', 'AU12', 'AU15', 'AU20', 'AU25']


def load_csvs():
    """Load OF3 and OF2.2 CSVs"""
    of3_csv = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv"
    of22_csv = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP22.csv"

    df_of3 = pd.read_csv(of3_csv)
    df_of22 = pd.read_csv(of22_csv)

    # Extract the 8 AU columns from OF3 (with current incorrect mapping)
    current_mapping = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU20_r', 'AU25_r']
    of3_data = df_of3[current_mapping].values  # Shape: (N, 8)

    # Get OF2.2 labels for the 8 AUs
    of22_data = {}
    for au in DISFA_8_AUS:
        col = f'{au}_r'
        if col in df_of22.columns:
            of22_data[au] = df_of22[col].values

    return of3_data, of22_data


def compute_total_correlation(of3_data, of22_data, ordering):
    """
    Compute total correlation for a given AU ordering.

    Args:
        of3_data: (N, 8) array of OF3 outputs
        of22_data: dict of AU name -> (N,) array
        ordering: list of 8 AU names in index order

    Returns:
        total_abs_corr: sum of absolute correlations
        correlations: dict of AU -> correlation
    """
    total_abs_corr = 0.0
    correlations = {}

    for idx, au_name in enumerate(ordering):
        if au_name not in of22_data:
            correlations[au_name] = 0.0
            continue

        of3_values = of3_data[:, idx]
        of22_values = of22_data[au_name]

        # Remove NaN values
        mask = ~(np.isnan(of3_values) | np.isnan(of22_values))

        if mask.sum() > 10 and np.std(of3_values[mask]) > 1e-10 and np.std(of22_values[mask]) > 1e-10:
            corr, _ = pearsonr(of3_values[mask], of22_values[mask])
            correlations[au_name] = corr
            total_abs_corr += abs(corr)
        else:
            correlations[au_name] = 0.0

    return total_abs_corr, correlations


def find_best_ordering_greedy(of3_data, of22_data):
    """
    Use greedy approach to find best ordering.

    For each OF3 index, find which AU it correlates best with.
    """
    print("="*80)
    print("GREEDY SEARCH FOR BEST AU ORDERING")
    print("="*80)

    used_aus = set()
    best_ordering = []
    all_correlations = []

    for idx in range(8):
        best_au = None
        best_corr = 0.0

        of3_values = of3_data[:, idx]

        print(f"\nOF3 Index {idx}:")
        print(f"  Testing correlations with each AU...")

        au_scores = []
        for au_name in DISFA_8_AUS:
            if au_name in used_aus or au_name not in of22_data:
                continue

            of22_values = of22_data[au_name]

            # Remove NaN values
            mask = ~(np.isnan(of3_values) | np.isnan(of22_values))

            if mask.sum() > 10 and np.std(of3_values[mask]) > 1e-10 and np.std(of22_values[mask]) > 1e-10:
                corr, _ = pearsonr(of3_values[mask], of22_values[mask])
                au_scores.append((au_name, abs(corr), corr))
            else:
                au_scores.append((au_name, 0.0, 0.0))

        # Sort by absolute correlation
        au_scores.sort(key=lambda x: x[1], reverse=True)

        # Print top 3
        print("  Top 3 matches:")
        for i, (au, abs_corr, corr) in enumerate(au_scores[:3]):
            print(f"    {i+1}. {au}: r={corr:.3f} (|r|={abs_corr:.3f})")

        # Pick best available AU
        for au_name, abs_corr, corr in au_scores:
            if au_name not in used_aus:
                best_au = au_name
                best_corr = corr
                break

        if best_au is None:
            # No valid AU found, use placeholder
            best_au = f'UNKNOWN_{idx}'
            best_corr = 0.0

        best_ordering.append(best_au)
        all_correlations.append(best_corr)
        used_aus.add(best_au)

        print(f"  ✓ SELECTED: {best_au} (r={best_corr:.3f})")

    return best_ordering, all_correlations


def main():
    print("Loading CSVs...")
    of3_data, of22_data = load_csvs()

    print(f"OF3 data shape: {of3_data.shape}")
    print(f"OF2.2 AUs: {list(of22_data.keys())}")

    # Use greedy search (faster than testing all permutations)
    best_ordering, correlations = find_best_ordering_greedy(of3_data, of22_data)

    print("\n" + "="*80)
    print("BEST AU ORDERING FOUND")
    print("="*80)

    print("\nCorrected OpenFace 3.0 AU mapping:")
    print("self.of3_au_mapping = {")
    for idx, au_name in enumerate(best_ordering):
        if au_name.startswith('UNKNOWN'):
            print(f"    {idx}: '{au_name}_r',  # No clear match")
        else:
            au_num = au_name.replace('AU', '')
            corr = correlations[idx]
            inverted = " - INVERTED!" if corr < 0 else ""
            print(f"    {idx}: 'AU{au_num}_r',  # r={corr:.3f}{inverted}")
    print("}")

    print("\n" + "="*80)
    print("CORRELATION SUMMARY")
    print("="*80)
    for idx, (au_name, corr) in enumerate(zip(best_ordering, correlations)):
        status = "✓" if abs(corr) > 0.3 else "⚠️" if abs(corr) > 0.1 else "❌"
        inverted = " (INVERTED)" if corr < 0 else ""
        print(f"{status} Index {idx} -> {au_name}: r={corr:.3f}{inverted}")

    total_abs_corr = sum(abs(c) for c in correlations)
    print(f"\nTotal |correlation|: {total_abs_corr:.3f}")
    print(f"Average |correlation|: {total_abs_corr/8:.3f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
