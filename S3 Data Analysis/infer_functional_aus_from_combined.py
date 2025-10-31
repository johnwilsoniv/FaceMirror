"""
Infer which AUs were functional in OpenFace 2.2 by analyzing the combined_results.csv
"""

import pandas as pd
import numpy as np

OF2_PATH = "/Users/johnwilsoniv/Documents/open2GR/3_Data_Analysis/combined_results.csv"
OF3_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S3O Results/combined_results.csv"

def analyze_au_functionality_from_combined(csv_path, version_name):
    """Check which AUs have meaningful values in combined results"""
    df = pd.read_csv(csv_path)

    # Get all AU columns (non-normalized raw values)
    au_cols = [col for col in df.columns if ' AU' in col and '_r' in col and 'Normalized' not in col]

    # Extract unique AUs
    aus = {}
    for col in au_cols:
        au_num = col.split('AU')[1].split('_')[0]
        au_name = f'AU{au_num}'
        if au_name not in aus:
            aus[au_name] = []
        aus[au_name].append(col)

    print(f"\n{'='*100}")
    print(f"{version_name} - AU FUNCTIONALITY ANALYSIS FROM COMBINED RESULTS")
    print(f"{'='*100}\n")

    functional_aus = []
    non_functional_aus = []

    for au_name in sorted(aus.keys()):
        cols = aus[au_name]

        # Combine all values for this AU across all conditions/sides
        all_values = pd.concat([df[col] for col in cols], ignore_index=True)

        # Calculate statistics
        non_nan_count = all_values.notna().sum()
        total_count = len(all_values)
        non_nan_pct = (non_nan_count / total_count * 100) if total_count > 0 else 0

        if non_nan_count > 0:
            non_zero_count = (all_values[all_values.notna()] > 0.01).sum()
            non_zero_pct = (non_zero_count / non_nan_count * 100) if non_nan_count > 0 else 0
            mean_val = all_values[all_values.notna()].mean()
            max_val = all_values[all_values.notna()].max()
        else:
            non_zero_count = 0
            non_zero_pct = 0
            mean_val = 0
            max_val = 0

        # Consider functional if >50% non-NaN AND mean > 0.01
        is_functional = non_nan_pct > 50 and mean_val > 0.01

        status = "✓ FUNCTIONAL" if is_functional else "✗ NON-FUNCTIONAL"
        print(f"{au_name:6s}: {status:20s} | Non-NaN: {non_nan_pct:5.1f}% | Mean: {mean_val:6.3f} | Max: {max_val:6.3f} | NonZero: {non_zero_pct:5.1f}%")

        if is_functional:
            functional_aus.append(au_name)
        else:
            non_functional_aus.append(au_name)

    print(f"\n{'='*100}")
    print(f"{version_name} SUMMARY:")
    print(f"  Functional AUs ({len(functional_aus)}): {', '.join(functional_aus)}")
    print(f"  Non-Functional AUs ({len(non_functional_aus)}): {', '.join(non_functional_aus)}")
    print(f"{'='*100}")

    return functional_aus, non_functional_aus

# Analyze both versions
print("\n" + "="*100)
print("COMPARING AU FUNCTIONALITY: OpenFace 2.2 vs 3.0")
print("="*100)

functional_2, non_functional_2 = analyze_au_functionality_from_combined(OF2_PATH, "OpenFace 2.2")
functional_3, non_functional_3 = analyze_au_functionality_from_combined(OF3_PATH, "OpenFace 3.0")

# Compare
print(f"\n{'='*100}")
print("COMPARISON SUMMARY")
print(f"{'='*100}\n")

lost_aus = set(functional_2) - set(functional_3)
gained_aus = set(functional_3) - set(functional_2)
maintained_aus = set(functional_2) & set(functional_3)

print(f"✓ MAINTAINED ({len(maintained_aus)}): {', '.join(sorted(maintained_aus))}")
if lost_aus:
    print(f"\n✗ LOST in OpenFace 3.0 ({len(lost_aus)}): {', '.join(sorted(lost_aus))}")
if gained_aus:
    print(f"\n✓ GAINED in OpenFace 3.0 ({len(gained_aus)}): {', '.join(sorted(gained_aus))}")

# Map to facial zones
print(f"\n{'='*100}")
print("IMPACT BY FACIAL ZONE")
print(f"{'='*100}\n")

# Based on manuscript and typical FACS mapping
upper_face_aus = ['AU01', 'AU02', 'AU04', 'AU05']
mid_face_aus = ['AU06', 'AU07', 'AU09', 'AU45']
lower_face_aus = ['AU10', 'AU12', 'AU14', 'AU15', 'AU16', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26']

def analyze_zone(zone_name, zone_aus, func_2, func_3):
    """Analyze impact on a specific facial zone"""
    zone_func_2 = [au for au in zone_aus if au in func_2]
    zone_func_3 = [au for au in zone_aus if au in func_3]
    zone_lost = set(zone_func_2) - set(zone_func_3)

    pct_retained = (len(zone_func_3) / len(zone_func_2) * 100) if zone_func_2 else 0

    print(f"{zone_name}:")
    print(f"  OpenFace 2.2: {len(zone_func_2)}/{len(zone_aus)} functional - {', '.join(zone_func_2)}")
    print(f"  OpenFace 3.0: {len(zone_func_3)}/{len(zone_aus)} functional - {', '.join(zone_func_3)}")
    if zone_lost:
        print(f"  ✗ LOST ({len(zone_lost)}): {', '.join(sorted(zone_lost))}")
    print(f"  Retention Rate: {pct_retained:.1f}%\n")

    return pct_retained

upper_retention = analyze_zone("UPPER FACE", upper_face_aus, functional_2, functional_3)
mid_retention = analyze_zone("MID FACE", mid_face_aus, functional_2, functional_3)
lower_retention = analyze_zone("LOWER FACE", lower_face_aus, functional_2, functional_3)

print(f"{'='*100}")
print("OVERALL IMPACT ASSESSMENT")
print(f"{'='*100}\n")
print(f"Upper Face Retention: {upper_retention:.1f}% - {'⚠ MODERATE IMPACT' if upper_retention < 75 else '✓ LOW IMPACT'}")
print(f"Mid Face Retention: {mid_retention:.1f}% - {'⚠ SEVERE IMPACT' if mid_retention < 50 else '⚠ MODERATE IMPACT' if mid_retention < 75 else '✓ LOW IMPACT'}")
print(f"Lower Face Retention: {lower_retention:.1f}% - {'⚠ SEVERE IMPACT' if lower_retention < 50 else '⚠ MODERATE IMPACT' if lower_retention < 75 else '✓ LOW IMPACT'}")
