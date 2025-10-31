"""
Check which AUs are actually functional (have non-NaN values) in OpenFace 3.0 vs 2.2
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Sample a few raw OpenFace files
OF3_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Combined Data")
OF2_DIR = Path("/Users/johnwilsoniv/Documents/open2GR/1_Face_Mirror")

def check_au_functionality(csv_path, openface_version):
    """Check which AUs have actual values vs NaN"""
    try:
        df = pd.read_csv(csv_path)

        # Get AU columns (intensity values _r)
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

        results = {}
        for au_col in sorted(au_cols):
            values = df[au_col]
            non_nan_count = values.notna().sum()
            total_count = len(values)
            non_nan_pct = (non_nan_count / total_count * 100) if total_count > 0 else 0

            # Check if AU has meaningful values
            if non_nan_count > 0:
                non_zero_count = (values[values.notna()] > 0.01).sum()
                mean_val = values[values.notna()].mean()
                max_val = values[values.notna()].max()
            else:
                non_zero_count = 0
                mean_val = 0
                max_val = 0

            results[au_col.replace('_r', '')] = {
                'non_nan_pct': non_nan_pct,
                'mean': mean_val,
                'max': max_val,
                'non_zero_count': non_zero_count,
                'functional': non_nan_pct > 50  # Consider functional if >50% non-NaN
            }

        return results
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return {}

print("=" * 100)
print("ANALYZING FUNCTIONAL AUs IN RAW OPENFACE OUTPUT")
print("=" * 100)

# Check OpenFace 3.0 files
print("\n### OpenFace 3.0 (from S1 Face Mirror output) ###\n")
of3_files = list(OF3_DIR.glob("*.csv"))[:5]  # Sample 5 files

all_of3_results = {}
for csv_file in of3_files:
    print(f"Analyzing: {csv_file.name}")
    results = check_au_functionality(csv_file, "3.0")

    for au, data in results.items():
        if au not in all_of3_results:
            all_of3_results[au] = []
        all_of3_results[au].append(data['functional'])

print("\n" + "=" * 100)
print("OpenFace 3.0 FUNCTIONAL AU SUMMARY")
print("=" * 100)

functional_aus_3 = []
non_functional_aus_3 = []

for au in sorted(all_of3_results.keys()):
    functionality_rate = sum(all_of3_results[au]) / len(all_of3_results[au]) * 100
    status = "✓ FUNCTIONAL" if functionality_rate >= 50 else "✗ NON-FUNCTIONAL"
    print(f"{au:6s}: {status:20s} ({functionality_rate:5.1f}% of files)")

    if functionality_rate >= 50:
        functional_aus_3.append(au)
    else:
        non_functional_aus_3.append(au)

print(f"\n{'='*100}")
print(f"OPENFACE 3.0 SUMMARY:")
print(f"  Functional AUs ({len(functional_aus_3)}): {', '.join(functional_aus_3)}")
print(f"  Non-Functional AUs ({len(non_functional_aus_3)}): {', '.join(non_functional_aus_3)}")

# Check if OpenFace 2.2 files exist
if OF2_DIR.exists():
    print(f"\n\n### OpenFace 2.2 (from old dataset) ###\n")

    # Try to find processed files
    of2_files = []
    for subdir in OF2_DIR.iterdir():
        if subdir.is_dir():
            csv_files = list(subdir.glob("*_mirrored.csv"))
            of2_files.extend(csv_files[:2])  # 2 from each subdir
            if len(of2_files) >= 5:
                break

    if of2_files:
        all_of2_results = {}
        for csv_file in of2_files[:5]:
            print(f"Analyzing: {csv_file.name}")
            results = check_au_functionality(csv_file, "2.2")

            for au, data in results.items():
                if au not in all_of2_results:
                    all_of2_results[au] = []
                all_of2_results[au].append(data['functional'])

        print("\n" + "=" * 100)
        print("OpenFace 2.2 FUNCTIONAL AU SUMMARY")
        print("=" * 100)

        functional_aus_2 = []
        non_functional_aus_2 = []

        for au in sorted(all_of2_results.keys()):
            functionality_rate = sum(all_of2_results[au]) / len(all_of2_results[au]) * 100
            status = "✓ FUNCTIONAL" if functionality_rate >= 50 else "✗ NON-FUNCTIONAL"
            print(f"{au:6s}: {status:20s} ({functionality_rate:5.1f}% of files)")

            if functionality_rate >= 50:
                functional_aus_2.append(au)
            else:
                non_functional_aus_2.append(au)

        print(f"\n{'='*100}")
        print(f"OPENFACE 2.2 SUMMARY:")
        print(f"  Functional AUs ({len(functional_aus_2)}): {', '.join(functional_aus_2)}")
        print(f"  Non-Functional AUs ({len(non_functional_aus_2)}): {', '.join(non_functional_aus_2)}")

        # Compare
        print(f"\n{'='*100}")
        print("COMPARISON: FUNCTIONAL AUs")
        print(f"{'='*100}")

        lost_aus = set(functional_aus_2) - set(functional_aus_3)
        gained_aus = set(functional_aus_3) - set(functional_aus_2)
        maintained_aus = set(functional_aus_2) & set(functional_aus_3)

        print(f"\n✓ MAINTAINED ({len(maintained_aus)}): {', '.join(sorted(maintained_aus))}")
        if lost_aus:
            print(f"\n✗ LOST in 3.0 ({len(lost_aus)}): {', '.join(sorted(lost_aus))}")
        if gained_aus:
            print(f"\n✓ GAINED in 3.0 ({len(gained_aus)}): {', '.join(sorted(gained_aus))}")
    else:
        print("Could not find OpenFace 2.2 processed files")
else:
    print(f"\nOpenFace 2.2 directory not found: {OF2_DIR}")

print(f"\n{'='*100}")
print("ANALYSIS COMPLETE")
print(f"{'='*100}")
