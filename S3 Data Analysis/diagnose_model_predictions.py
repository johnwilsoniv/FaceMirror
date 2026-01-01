#!/usr/bin/env python3
"""
diagnose_model_predictions.py

This script runs the saved models on data from combined_results.csv
and compares predictions against expert labels to measure actual accuracy.

This tests the FULL production pipeline on the TRAINING data source.
"""

import os
import sys
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from paralysis_config import ZONE_CONFIG, INPUT_FILES, CLASS_NAMES
from paralysis_detector import ParalysisDetector
from facial_au_constants import standardize_paralysis_label


def load_data():
    """Load combined results and expert key."""
    results_path = INPUT_FILES.get('results_csv')
    expert_path = INPUT_FILES.get('expert_key_csv')

    print(f"Loading results from: {results_path}")
    results_df = pd.read_csv(results_path, low_memory=False)
    print(f"  Loaded {len(results_df)} patients")

    print(f"Loading expert key from: {expert_path}")
    expert_df = pd.read_csv(expert_path, keep_default_na=False)
    expert_df = expert_df.set_index('Patient')
    print(f"  Loaded {len(expert_df)} expert labels")

    return results_df, expert_df


def run_predictions(results_df, expert_df):
    """Run predictions for all patients and compare against expert labels."""

    # Initialize detectors
    detectors = {}
    for zone in ['lower', 'mid', 'upper']:
        try:
            detectors[zone] = ParalysisDetector(zone=zone)
        except Exception as e:
            print(f"Failed to load {zone} detector: {e}")

    # Expert column mapping
    expert_cols = {
        'lower': {'left': 'Paralysis - Left Lower Face', 'right': 'Paralysis - Right Lower Face'},
        'mid': {'left': 'Paralysis - Left Mid Face', 'right': 'Paralysis - Right Mid Face'},
        'upper': {'left': 'Paralysis - Left Upper Face', 'right': 'Paralysis - Right Upper Face'}
    }

    # Reverse class name mapping
    class_to_num = {'Normal': 0, 'None': 0, 'Partial': 1, 'Incomplete': 1, 'Complete': 2}

    results = {zone: {'correct': 0, 'incorrect': 0, 'error': 0, 'missing_expert': 0, 'details': []}
               for zone in detectors.keys()}

    print("\n" + "="*80)
    print("RUNNING PREDICTIONS ON ALL PATIENTS")
    print("="*80)

    for idx, row in results_df.iterrows():
        patient_id = row.get('Patient ID', f'Row_{idx}')

        # Skip if patient not in expert key
        if patient_id not in expert_df.index:
            for zone in detectors:
                results[zone]['missing_expert'] += 2  # Both sides
            continue

        expert_row = expert_df.loc[patient_id]

        for zone, detector in detectors.items():
            for side in ['left', 'right']:
                # Get expert label
                expert_col = expert_cols[zone][side]
                if expert_col not in expert_row.index:
                    results[zone]['missing_expert'] += 1
                    continue

                expert_val = expert_row[expert_col]
                if pd.isna(expert_val) or expert_val == '':
                    results[zone]['missing_expert'] += 1
                    continue

                expert_std = standardize_paralysis_label(expert_val)
                expert_num = class_to_num.get(expert_std, -1)

                # Run prediction
                try:
                    result, confidence, details = detector.detect(row.to_dict(), side)
                    pred_num = details.get('raw_prediction', -1)

                    if expert_num == pred_num:
                        results[zone]['correct'] += 1
                    else:
                        results[zone]['incorrect'] += 1
                        results[zone]['details'].append({
                            'patient': patient_id,
                            'side': side,
                            'expert': expert_std,
                            'predicted': CLASS_NAMES.get(pred_num, 'Unknown'),
                            'confidence': confidence,
                            'probs': details.get('probabilities', [])
                        })

                except Exception as e:
                    results[zone]['error'] += 1
                    print(f"  Error predicting {patient_id} {side} {zone}: {e}")

    return results


def print_results(results):
    """Print accuracy summary."""
    print("\n" + "="*80)
    print("ACCURACY RESULTS (Model on combined_results.csv vs Expert Key)")
    print("="*80)

    # Paper-reported accuracies for comparison
    paper_accuracy = {'lower': 0.84, 'mid': 0.93, 'upper': 0.83}

    for zone in ['upper', 'mid', 'lower']:
        if zone not in results:
            continue

        r = results[zone]
        total = r['correct'] + r['incorrect']
        accuracy = r['correct'] / total if total > 0 else 0

        print(f"\n{zone.upper()} FACE:")
        print(f"  Correct: {r['correct']}")
        print(f"  Incorrect: {r['incorrect']}")
        print(f"  Errors: {r['error']}")
        print(f"  Missing expert labels: {r['missing_expert']}")
        print(f"  ACCURACY: {accuracy:.2%} (Paper reported: {paper_accuracy[zone]:.0%})")

        if r['incorrect'] > 0:
            print(f"\n  Sample misclassifications (first 10):")
            for d in r['details'][:10]:
                print(f"    {d['patient']} {d['side']}: Expert={d['expert']}, Pred={d['predicted']} (conf={d['confidence']:.2f})")


def main():
    print("="*80)
    print("MODEL PREDICTION DIAGNOSTIC")
    print("="*80)

    results_df, expert_df = load_data()
    results = run_predictions(results_df, expert_df)
    print_results(results)


if __name__ == "__main__":
    main()
