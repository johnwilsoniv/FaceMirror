#!/usr/bin/env python3
"""
Compare OpenFace 2.2 vs 3.0 CSV outputs

This script analyzes the differences between OpenFace 2.2 and 3.0 AU extraction results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Directories
COMBINED_DATA = Path.home() / "Documents/SplitFace/S1O Processed Files/Combined Data"
OF22_TEST = Path.home() / "Documents/SplitFace/S1O Processed Files/OpenFace 2.2 Test"


def compare_csv_files(of3_path, of2_path):
    """
    Compare two CSV files from OpenFace 2.2 and 3.0

    Args:
        of3_path: Path to OpenFace 3.0 CSV
        of2_path: Path to OpenFace 2.2 CSV
    """
    print(f"\n{'='*80}")
    print(f"Comparing: {of3_path.name}")
    print(f"{'='*80}\n")

    # Read CSVs
    try:
        of3_df = pd.read_csv(of3_path)
        print(f"✓ OpenFace 3.0 CSV loaded: {len(of3_df)} frames")
    except Exception as e:
        print(f"✗ Failed to load OpenFace 3.0 CSV: {e}")
        return

    try:
        of2_df = pd.read_csv(of2_path)
        print(f"✓ OpenFace 2.2 CSV loaded: {len(of2_df)} frames")
    except Exception as e:
        print(f"✗ Failed to load OpenFace 2.2 CSV: {e}")
        return

    # Compare dimensions
    print(f"\nDimensions:")
    print(f"  OpenFace 3.0: {of3_df.shape} (frames × columns)")
    print(f"  OpenFace 2.2: {of2_df.shape} (frames × columns)")

    if of3_df.shape[0] != of2_df.shape[0]:
        print(f"  ⚠️  WARNING: Different number of frames!")

    # Compare columns
    of3_cols = set(of3_df.columns)
    of2_cols = set(of2_df.columns)

    common_cols = of3_cols & of2_cols
    of3_only = of3_cols - of2_cols
    of2_only = of2_cols - of3_cols

    print(f"\nColumns:")
    print(f"  Common columns:     {len(common_cols)}")
    print(f"  OpenFace 3.0 only:  {len(of3_only)}")
    print(f"  OpenFace 2.2 only:  {len(of2_only)}")

    if of3_only:
        print(f"\n  Columns only in OpenFace 3.0:")
        for col in sorted(of3_only):
            print(f"    - {col}")

    if of2_only:
        print(f"\n  Columns only in OpenFace 2.2:")
        for col in sorted(of2_only):
            print(f"    - {col}")

    # Extract AU columns
    of3_au_cols = [c for c in of3_df.columns if c.startswith('AU') and '_r' in c]
    of2_au_cols = [c for c in of2_df.columns if c.startswith('AU') and '_r' in c]

    print(f"\nAction Units:")
    print(f"  OpenFace 3.0: {len(of3_au_cols)} AUs")
    print(f"  OpenFace 2.2: {len(of2_au_cols)} AUs")

    print(f"\n  OpenFace 3.0 AUs: {', '.join(sorted(of3_au_cols))}")
    print(f"  OpenFace 2.2 AUs: {', '.join(sorted(of2_au_cols))}")

    # Compare common AU values
    common_au_cols = sorted(set(of3_au_cols) & set(of2_au_cols))

    if common_au_cols:
        print(f"\nComparing {len(common_au_cols)} common AUs...")
        print(f"\n{'AU':<12} {'OF3.0 Mean':<12} {'OF2.2 Mean':<12} {'Correlation':<12} {'Status'}")
        print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")

        issues = []

        for au in common_au_cols:
            # Get values (exclude NaN)
            of3_vals = of3_df[au].dropna()
            of2_vals = of2_df[au].dropna()

            if len(of3_vals) == 0 or len(of2_vals) == 0:
                print(f"{au:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} All NaN")
                issues.append(f"{au}: All values are NaN")
                continue

            of3_mean = of3_vals.mean()
            of2_mean = of2_vals.mean()

            # Compute correlation (if both have valid values at same indices)
            min_len = min(len(of3_df), len(of2_df))
            of3_aligned = of3_df[au][:min_len]
            of2_aligned = of2_df[au][:min_len]

            # Remove rows where either is NaN
            valid_mask = ~(of3_aligned.isna() | of2_aligned.isna())
            if valid_mask.sum() < 2:
                corr = np.nan
                corr_str = "N/A"
            else:
                corr = of3_aligned[valid_mask].corr(of2_aligned[valid_mask])
                corr_str = f"{corr:.3f}"

            # Determine status
            if np.isnan(corr):
                status = "⚠️  Insufficient data"
                issues.append(f"{au}: Insufficient data for correlation")
            elif corr < 0.5:
                status = "❌ Low correlation"
                issues.append(f"{au}: Low correlation ({corr:.3f})")
            elif corr < 0.8:
                status = "⚠️  Moderate correlation"
            else:
                status = "✓ Good correlation"

            print(f"{au:<12} {of3_mean:<12.3f} {of2_mean:<12.3f} {corr_str:<12} {status}")

        if issues:
            print(f"\n⚠️  Issues detected:")
            for issue in issues:
                print(f"  - {issue}")

    # Check for success column
    if 'success' in of3_df.columns and 'success' in of2_df.columns:
        of3_success_rate = (of3_df['success'] == 1).mean() * 100
        of2_success_rate = (of2_df['success'] == 1).mean() * 100

        print(f"\nSuccess Rate:")
        print(f"  OpenFace 3.0: {of3_success_rate:.1f}%")
        print(f"  OpenFace 2.2: {of2_success_rate:.1f}%")

        if abs(of3_success_rate - of2_success_rate) > 5:
            print(f"  ⚠️  WARNING: Success rates differ by >5%")

    print(f"\n{'='*80}\n")


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("OpenFace 2.2 vs 3.0 CSV Comparison")
    print("="*80)

    # Check directories exist
    if not COMBINED_DATA.exists():
        print(f"ERROR: Combined Data directory not found: {COMBINED_DATA}")
        return

    if not OF22_TEST.exists():
        print(f"ERROR: OpenFace 2.2 Test directory not found: {OF22_TEST}")
        print(f"Run run_openface22_comparison.py first!")
        return

    # Find matching CSV files
    of3_csvs = list(COMBINED_DATA.glob("*.csv"))
    of2_csvs = list(OF22_TEST.glob("*.csv"))

    if not of3_csvs:
        print(f"No CSV files found in: {COMBINED_DATA}")
        return

    if not of2_csvs:
        print(f"No CSV files found in: {OF22_TEST}")
        return

    print(f"\nFound {len(of3_csvs)} OpenFace 3.0 CSV(s)")
    print(f"Found {len(of2_csvs)} OpenFace 2.2 CSV(s)")

    # Compare matching files
    compared_count = 0

    for of3_csv in sorted(of3_csvs):
        # Find corresponding OF2.2 CSV
        of2_csv = OF22_TEST / of3_csv.name

        if of2_csv.exists():
            compare_csv_files(of3_csv, of2_csv)
            compared_count += 1
        else:
            print(f"\n⚠️  No OpenFace 2.2 match for: {of3_csv.name}")

    if compared_count == 0:
        print("\nNo matching CSV files found for comparison!")
    else:
        print(f"Compared {compared_count} file pair(s)")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
