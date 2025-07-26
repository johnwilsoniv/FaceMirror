#!/usr/bin/env python3
"""
validate_improvements.py - Check if optimizations are properly implemented
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score


def validate_model_improvements(zone):
    """Validate improvements for a specific zone"""
    print(f"\nValidating {zone.upper()} face model...")

    # Check if model exists
    model_path = f"models/{zone}_face_model.pkl"
    if not os.path.exists(model_path):
        print(f"  ✗ Model not found: {model_path}")
        return None

    # Load model
    try:
        model = joblib.load(model_path)
        print(f"  ✓ Model loaded successfully")

        # Check if it's a StackingClassifier
        if hasattr(model, 'estimators_'):
            print(f"  ✓ Model is StackingClassifier with {len(model.estimators_)} base models")
        else:
            print(f"  ✗ Model is not StackingClassifier")

        # Check for calibration
        if 'CalibratedClassifierCV' in str(type(model)):
            print(f"  ✓ Model is calibrated")
        else:
            print(f"  ! Model is not calibrated")

        # Check for optimal thresholds
        threshold_path = f"models/{zone}_optimal_thresholds.pkl"
        if os.path.exists(threshold_path):
            thresholds = joblib.load(threshold_path)
            print(f"  ✓ Optimal thresholds found: {thresholds}")
        else:
            print(f"  ! No optimal thresholds saved")

    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None

    # Check configuration
    try:
        from paralysis_config import get_zone_config, get_performance_targets
        config = get_zone_config(zone)
        targets = get_performance_targets(zone)

        print(f"\n  Configuration checks:")
        print(f"    - SMOTEENN enabled: {config['training']['smote']['use_smoteenn_after']}")
        print(f"    - Threshold optimization: {config['training']['threshold_optimization']['enabled']}")
        print(f"    - Class weights: {config['training']['class_weights']}")
        print(f"    - Performance targets: {targets}")

    except Exception as e:
        print(f"  ✗ Error checking configuration: {e}")

    return True


def compare_results():
    """Compare current results with baseline"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    baseline = {
        'lower': {'accuracy': 0.84, 'f1_weighted': 0.82, 'f1_none': 0.92, 'f1_partial': 0.46, 'f1_complete': 0.81},
        'mid': {'accuracy': 0.93, 'f1_weighted': 0.92, 'f1_none': 0.98, 'f1_partial': 0.67, 'f1_complete': 0.83},
        'upper': {'accuracy': 0.83, 'f1_weighted': 0.83, 'f1_none': 0.88, 'f1_partial': 0.40, 'f1_complete': 0.86}
    }

    print(f"\n{'Zone':<10} {'Metric':<15} {'Baseline':<10} {'Current':<10} {'Change':<10}")
    print("-" * 60)

    # You would need to load actual current results here
    # This is just a template

    for zone in ['lower', 'mid', 'upper']:
        for metric in ['f1_partial']:
            baseline_val = baseline[zone][metric]
            # current_val = get_current_value(zone, metric)  # Implement this
            # change = current_val - baseline_val
            # print(f"{zone:<10} {metric:<15} {baseline_val:<10.3f} {current_val:<10.3f} {change:+10.3f}")


if __name__ == "__main__":
    print("Facial Paralysis Detection - Optimization Validation")
    print("=" * 60)

    # Check Python packages
    required_packages = ['sklearn', 'xgboost', 'imblearn', 'optuna']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")

    # Validate each zone
    for zone in ['lower', 'mid', 'upper']:
        validate_model_improvements(zone)

    # Compare results
    compare_results()