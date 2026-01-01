#!/usr/bin/env python3
"""
diagnose_feature_discrepancy.py

Diagnostic script to compare feature extraction between:
1. Training pipeline (extract_features from zone modules via prepare_data_generalized)
2. Production pipeline (extract_features_for_detection from zone modules)

This helps identify why production accuracy differs from training accuracy.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add the current directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from paralysis_config import ZONE_CONFIG, INPUT_FILES, CLASS_NAMES
from paralysis_utils import _extract_base_au_features, standardize_paralysis_labels


def load_training_data() -> pd.DataFrame:
    """Load the combined_results.csv used for training."""
    results_path = INPUT_FILES.get('results_csv')
    if not results_path or not os.path.exists(results_path):
        print(f"ERROR: Results CSV not found at {results_path}")
        return None

    print(f"Loading training data from: {results_path}")
    df = pd.read_csv(results_path, low_memory=False)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def extract_features_training_style(row_data: pd.Series, side: str, zone: str) -> Dict[str, float]:
    """
    Extract features the way training does it.
    This mirrors what happens in prepare_data_generalized + zone-specific extract_features.
    """
    zone_config = ZONE_CONFIG[zone]
    actions = zone_config.get('actions', [])
    aus = zone_config.get('aus', [])
    feature_cfg = zone_config.get('feature_extraction', {})

    # Convert to single-row DataFrame (as training does)
    df_single = pd.DataFrame([row_data.to_dict()], index=[0])

    # Extract base features using the same helper as training
    base_features_df = _extract_base_au_features(df_single, side, actions, aus, feature_cfg, zone)

    # Convert to dict
    features = {}
    for col in base_features_df.columns:
        features[col] = base_features_df[col].iloc[0]

    # Add side_indicator (added by prepare_data_generalized after extract_features)
    features['side_indicator'] = 0 if side.lower() == 'left' else 1

    # Add zone-specific summary features
    if zone == 'lower':
        # From lower_face_features.py extract_features
        avg_au12_ratio_vals = []
        max_au12_pd_vals = []
        for act in actions:
            ratio_key = f"{act}_AU12_r_Asym_Ratio"
            pd_key = f"{act}_AU12_r_Asym_PercDiff"
            if ratio_key in features:
                avg_au12_ratio_vals.append(features[ratio_key])
            if pd_key in features:
                max_au12_pd_vals.append(features[pd_key])

        features['avg_AU12_Asym_Ratio'] = np.mean(avg_au12_ratio_vals) if avg_au12_ratio_vals else 1.0
        features['max_AU12_Asym_PercDiff'] = np.max(max_au12_pd_vals) if max_au12_pd_vals else 0.0

        bs_au12 = features.get('BS_AU12_r_Asym_Ratio', 1.0)
        bs_au25 = features.get('BS_AU25_r_Asym_Ratio', 1.0)
        features['BS_Asym_Ratio_Product_12_25'] = bs_au12 * bs_au25

    return features


def extract_features_production_style(row_data: pd.Series, side: str, zone: str) -> Dict[str, float]:
    """
    Extract features the way production does it.
    This calls extract_features_for_detection from the zone module.
    """
    import importlib

    module_name = f"{zone}_face_features"
    try:
        feature_module = importlib.import_module(module_name)
        feature_vector = feature_module.extract_features_for_detection(row_data.to_dict(), side, zone)

        if feature_vector is None:
            print(f"  WARNING: extract_features_for_detection returned None for {zone} {side}")
            return None

        # Load the feature list to map indices to names
        feature_list_path = ZONE_CONFIG[zone]['filenames'].get('feature_list')
        if feature_list_path and os.path.exists(feature_list_path):
            with open(feature_list_path, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]

            if len(feature_vector) == len(feature_names):
                return dict(zip(feature_names, feature_vector))
            else:
                print(f"  WARNING: Feature vector length ({len(feature_vector)}) != feature names ({len(feature_names)})")
                return {'_raw_vector': feature_vector}
        else:
            print(f"  WARNING: Feature list not found at {feature_list_path}")
            return {'_raw_vector': feature_vector}

    except Exception as e:
        print(f"  ERROR: Failed to extract production features: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_features(training_features: Dict[str, float],
                    production_features: Dict[str, float],
                    tolerance: float = 1e-6) -> Dict:
    """Compare two feature dictionaries and report differences."""

    if production_features is None:
        return {'error': 'Production features are None'}

    if '_raw_vector' in production_features:
        return {'error': 'Production returned raw vector without feature names'}

    results = {
        'matches': [],
        'mismatches': [],
        'missing_in_production': [],
        'extra_in_production': [],
        'production_zeros': []
    }

    training_keys = set(training_features.keys())
    production_keys = set(production_features.keys())

    # Features in training but not production
    results['missing_in_production'] = list(training_keys - production_keys)

    # Features in production but not training (shouldn't happen)
    results['extra_in_production'] = list(production_keys - training_keys)

    # Compare common features
    common_keys = training_keys & production_keys
    for key in sorted(common_keys):
        train_val = training_features[key]
        prod_val = production_features[key]

        # Track zeros in production
        if prod_val == 0.0 and train_val != 0.0:
            results['production_zeros'].append({
                'feature': key,
                'training_value': train_val,
                'production_value': prod_val
            })

        if abs(train_val - prod_val) < tolerance:
            results['matches'].append(key)
        else:
            results['mismatches'].append({
                'feature': key,
                'training_value': train_val,
                'production_value': prod_val,
                'difference': prod_val - train_val
            })

    return results


def analyze_patient(df: pd.DataFrame, patient_id: str, zone: str = 'lower'):
    """Analyze feature extraction for a specific patient."""

    print(f"\n{'='*80}")
    print(f"Analyzing Patient: {patient_id}, Zone: {zone}")
    print(f"{'='*80}")

    # Find the patient row
    patient_rows = df[df['Patient ID'].astype(str) == str(patient_id)]
    if patient_rows.empty:
        print(f"ERROR: Patient {patient_id} not found in data")
        return None

    row = patient_rows.iloc[0]
    print(f"Found patient row with index {patient_rows.index[0]}")

    results = {}
    for side in ['Left', 'Right']:
        print(f"\n--- {side} Side ---")

        # Extract using training method
        print("  Extracting features (training style)...")
        training_features = extract_features_training_style(row, side, zone)
        print(f"    Generated {len(training_features)} features")

        # Extract using production method
        print("  Extracting features (production style)...")
        production_features = extract_features_production_style(row, side, zone)
        if production_features:
            print(f"    Generated {len(production_features)} features")
        else:
            print("    FAILED to generate features")

        # Compare
        print("  Comparing features...")
        comparison = compare_features(training_features, production_features)

        if 'error' in comparison:
            print(f"    ERROR: {comparison['error']}")
        else:
            print(f"    Matches: {len(comparison['matches'])}")
            print(f"    Mismatches: {len(comparison['mismatches'])}")
            print(f"    Missing in production: {len(comparison['missing_in_production'])}")
            print(f"    Production zeros (training non-zero): {len(comparison['production_zeros'])}")

            if comparison['mismatches']:
                print("\n    TOP MISMATCHES:")
                for item in sorted(comparison['mismatches'],
                                   key=lambda x: abs(x['difference']),
                                   reverse=True)[:10]:
                    print(f"      {item['feature']}: train={item['training_value']:.4f}, "
                          f"prod={item['production_value']:.4f}, diff={item['difference']:.4f}")

            if comparison['missing_in_production']:
                print(f"\n    MISSING IN PRODUCTION: {comparison['missing_in_production'][:10]}")

            if comparison['production_zeros']:
                print(f"\n    PRODUCTION ZEROS (training non-zero):")
                for item in comparison['production_zeros'][:10]:
                    print(f"      {item['feature']}: train={item['training_value']:.4f}")

        results[side] = {
            'training_features': training_features,
            'production_features': production_features,
            'comparison': comparison
        }

    return results


def check_column_availability(df: pd.DataFrame, zone: str = 'lower'):
    """Check if expected AU columns are present in the data."""

    print(f"\n{'='*80}")
    print(f"Checking Column Availability for Zone: {zone}")
    print(f"{'='*80}")

    zone_config = ZONE_CONFIG[zone]
    actions = zone_config.get('actions', [])
    aus = zone_config.get('aus', [])

    print(f"Actions: {actions}")
    print(f"AUs: {aus}")

    missing_columns = []
    present_columns = []

    for action in actions:
        for side in ['Left', 'Right']:
            for au in aus:
                # Check raw column
                raw_col = f"{action}_{side} {au}"
                if raw_col in df.columns:
                    present_columns.append(raw_col)
                else:
                    missing_columns.append(raw_col)

                # Check normalized column
                norm_col = f"{action}_{side} {au} (Normalized)"
                if norm_col in df.columns:
                    present_columns.append(norm_col)
                else:
                    missing_columns.append(norm_col)

    print(f"\nPresent columns: {len(present_columns)}")
    print(f"Missing columns: {len(missing_columns)}")

    if missing_columns:
        print(f"\nMissing columns (first 20):")
        for col in missing_columns[:20]:
            print(f"  - {col}")

    return present_columns, missing_columns


def main():
    """Main diagnostic function."""

    print("="*80)
    print("FEATURE EXTRACTION DISCREPANCY DIAGNOSTIC")
    print("="*80)

    # Load data
    df = load_training_data()
    if df is None:
        return

    # Get available patient IDs
    patient_ids = df['Patient ID'].dropna().unique()
    print(f"\nFound {len(patient_ids)} unique patients")

    # Check column availability for each zone
    for zone in ['lower', 'mid', 'upper']:
        check_column_availability(df, zone)

    # Analyze a few sample patients
    print("\n" + "="*80)
    print("ANALYZING SAMPLE PATIENTS")
    print("="*80)

    sample_patients = list(patient_ids[:3])  # First 3 patients
    print(f"Analyzing patients: {sample_patients}")

    all_results = {}
    for patient_id in sample_patients:
        for zone in ['lower', 'mid', 'upper']:
            results = analyze_patient(df, patient_id, zone)
            all_results[(patient_id, zone)] = results

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    total_mismatches = 0
    total_zeros = 0
    total_missing = 0

    for (patient_id, zone), results in all_results.items():
        if results:
            for side in ['Left', 'Right']:
                if side in results and 'comparison' in results[side]:
                    comp = results[side]['comparison']
                    if 'error' not in comp:
                        total_mismatches += len(comp.get('mismatches', []))
                        total_zeros += len(comp.get('production_zeros', []))
                        total_missing += len(comp.get('missing_in_production', []))

    print(f"\nAcross all analyzed samples:")
    print(f"  Total feature mismatches: {total_mismatches}")
    print(f"  Total production zeros (where training non-zero): {total_zeros}")
    print(f"  Total missing in production: {total_missing}")

    if total_mismatches > 0 or total_zeros > 0 or total_missing > 0:
        print("\n>>> DISCREPANCIES DETECTED - This likely explains accuracy differences <<<")
    else:
        print("\n>>> No obvious discrepancies detected <<<")


if __name__ == "__main__":
    main()
