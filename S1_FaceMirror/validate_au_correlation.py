#!/usr/bin/env python3
"""
Validate AU correlation between PyFaceAU (Python) and C++ OpenFace outputs.

Computes Pearson correlation for each AU and checks if >=15/17 exceed 0.95 threshold.

IMPORTANT: Use the actual PyFaceAU output (validation_output/cpp_vs_python_test/),
NOT openface_output/ which may contain copied C++ data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Paths - use actual PyFaceAU output, not copied data
CPP_REF = Path(__file__).parent.parent / "archive/test_artifacts_root/cpp_reference_output/IMG_0942.csv"
PYTHON_OUTPUT = Path(__file__).parent.parent / "validation_output/cpp_vs_python_test/IMG_0942.csv"

# 17 Action Units (regression values)
AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
    'AU25_r', 'AU26_r', 'AU45_r'
]

THRESHOLD = 0.95
PASS_COUNT = 15  # Need at least 15/17 to pass


def load_and_align_data(cpp_path, py_path):
    """Load CSVs and align by frame number."""
    cpp_df = pd.read_csv(cpp_path)
    py_df = pd.read_csv(py_path)

    # Align by frame column
    cpp_indexed = cpp_df.set_index('frame')
    py_indexed = py_df.set_index('frame')
    common_frames = cpp_indexed.index.intersection(py_indexed.index)

    cpp_aligned = cpp_indexed.loc[common_frames]
    py_aligned = py_indexed.loc[common_frames]

    return cpp_aligned[AU_COLUMNS].values, py_aligned[AU_COLUMNS].values, len(common_frames)


def compute_correlations(cpp_data, python_data, clamp=True):
    """Compute Pearson correlation for each AU.

    Args:
        cpp_data: C++ AU values
        python_data: Python AU values
        clamp: If True, clamp Python values to [0, 5] to match C++ behavior
    """
    correlations = {}

    for i, au in enumerate(AU_COLUMNS):
        cpp_col = cpp_data[:, i].astype(float)
        py_col = python_data[:, i].astype(float)

        # Remove NaN/Inf
        valid = np.isfinite(cpp_col) & np.isfinite(py_col)
        cpp_valid = cpp_col[valid]
        py_valid = py_col[valid]

        if len(cpp_valid) < 10:
            correlations[au] = np.nan
            continue

        # Clamp Python to [0, 5] to match C++ OpenFace behavior
        if clamp:
            py_valid = np.clip(py_valid, 0, 5)

        # Skip if either column has zero variance
        if np.std(cpp_valid) < 1e-10 or np.std(py_valid) < 1e-10:
            correlations[au] = np.nan
            continue

        r, p = stats.pearsonr(cpp_valid, py_valid)
        correlations[au] = r

    return correlations


def main():
    print("=" * 60)
    print("AU CORRELATION VALIDATION")
    print("=" * 60)
    print(f"C++ Reference: {CPP_REF}")
    print(f"Python Output: {PYTHON_OUTPUT}")
    print()

    # Load and align data by frame number
    cpp_data, python_data, common_frames = load_and_align_data(CPP_REF, PYTHON_OUTPUT)
    print(f"Common frames: {common_frames}")
    print()

    # Compute correlations (with clamping to match C++ behavior)
    correlations = compute_correlations(cpp_data, python_data, clamp=True)

    # Display results
    print(f"{'AU':<10} {'Correlation':>12} {'Status':>10}")
    print("-" * 35)

    passed = 0
    failed = []

    for au, r in correlations.items():
        if np.isnan(r):
            status = "NO VAR"
        elif r >= THRESHOLD:
            status = "PASS ✓"
            passed += 1
        else:
            status = "FAIL ✗"
            failed.append((au, r))

        r_str = f"{r:.4f}" if not np.isnan(r) else "N/A"
        print(f"{au:<10} {r_str:>12} {status:>10}")

    print("-" * 35)
    print()

    # Summary
    total = len([r for r in correlations.values() if not np.isnan(r)])
    print(f"SUMMARY: {passed}/{total} AUs above {THRESHOLD} threshold")

    if passed >= PASS_COUNT:
        print(f"VALIDATION: PASSED ({passed}>={PASS_COUNT})")
    else:
        print(f"VALIDATION: FAILED ({passed}<{PASS_COUNT})")
        print("\nFailed AUs:")
        for au, r in failed:
            print(f"  {au}: {r:.4f}")

    print("=" * 60)

    return passed >= PASS_COUNT


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
