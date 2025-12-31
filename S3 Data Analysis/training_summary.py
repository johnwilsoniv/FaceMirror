"""
Training and Performance Summary Module

Generates comprehensive summaries of training runs and model performance,
auto-saving to files to avoid context bloat in conversations.
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def generate_training_summary(zone_name, zone_key, training_results, config):
    """
    Generate comprehensive training summary.

    Args:
        zone_name: Display name of zone (e.g., "Upper Face")
        zone_key: Zone key (e.g., "upper")
        training_results: Dictionary containing training results:
            - n_features_initial: Initial number of features
            - n_features_selected: Number of selected features
            - selected_features: List of selected feature names
            - optuna_study: Optuna study object (optional)
            - optimal_thresholds: Dictionary of optimized thresholds (optional)
            - shap_available: Whether SHAP analysis was performed
            - shap_method: SHAP computation method if available
            - smote_applied: Whether SMOTE was applied
            - calibration_method: Calibration method used (if any)
        config: Zone configuration dictionary

    Returns:
        str: Formatted training summary
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 70)
    lines.append(f"TRAINING SUMMARY: {zone_name}")
    lines.append(f"Zone Key: {zone_key}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("=" * 70)
    lines.append("")

    # Feature Engineering Summary
    lines.append("FEATURE ENGINEERING")
    lines.append("-" * 70)
    lines.append(f"Initial Features: {training_results.get('n_features_initial', 'N/A')}")
    lines.append(f"Selected Features: {training_results.get('n_features_selected', 'N/A')}")
    if training_results.get('n_features_initial') and training_results.get('n_features_selected'):
        reduction_pct = (1 - training_results['n_features_selected'] / training_results['n_features_initial']) * 100
        lines.append(f"Feature Reduction: {reduction_pct:.1f}%")
    lines.append("")

    # Top Selected Features (first 15)
    if training_results.get('selected_features'):
        lines.append("Top Selected Features (first 15):")
        for idx, feat in enumerate(training_results['selected_features'][:15], 1):
            lines.append(f"  {idx:2d}. {feat}")
        if len(training_results['selected_features']) > 15:
            lines.append(f"  ... and {len(training_results['selected_features']) - 15} more")
        lines.append("")

    # Data Augmentation Summary
    lines.append("DATA PROCESSING")
    lines.append("-" * 70)
    lines.append(f"SMOTE Applied: {'Yes' if training_results.get('smote_applied') else 'No'}")
    if training_results.get('smote_variant'):
        lines.append(f"SMOTE Variant: {training_results['smote_variant']}")
    if training_results.get('n_samples_original') and training_results.get('n_samples_after_smote'):
        lines.append(f"Samples (Original): {training_results['n_samples_original']}")
        lines.append(f"Samples (After SMOTE): {training_results['n_samples_after_smote']}")
        augmentation_pct = (training_results['n_samples_after_smote'] / training_results['n_samples_original'] - 1) * 100
        lines.append(f"Data Augmentation: +{augmentation_pct:.1f}%")
    lines.append("")

    # Hyperparameter Optimization Summary
    lines.append("HYPERPARAMETER OPTIMIZATION")
    lines.append("-" * 70)
    optuna_study = training_results.get('optuna_study')
    if optuna_study:
        lines.append(f"Method: Optuna (TPE Sampler)")
        lines.append(f"Trials Completed: {len(optuna_study.trials)}")
        if optuna_study.best_trial:
            lines.append(f"Best Trial: #{optuna_study.best_trial.number}")
            # Safety check: ensure best_value is a number, not a dict
            best_val = optuna_study.best_value
            if isinstance(best_val, (int, float)):
                lines.append(f"Best Score: {best_val:.4f}")
            else:
                lines.append(f"Best Score: {best_val}")
            lines.append("Best Parameters:")
            for param, value in optuna_study.best_params.items():
                if isinstance(value, float):
                    lines.append(f"  {param}: {value:.4g}")
                else:
                    lines.append(f"  {param}: {value}")
    else:
        lines.append("No hyperparameter optimization performed (using defaults)")
    lines.append("")

    # Model Configuration
    lines.append("MODEL CONFIGURATION")
    lines.append("-" * 70)
    lines.append(f"Base Model: XGBoost + VotingClassifier (RF, ET)")
    lines.append(f"Calibration: {training_results.get('calibration_method', 'None')}")
    if training_results.get('optimal_thresholds'):
        lines.append("Threshold Optimization: Enabled")
        lines.append("Optimized Thresholds:")
        for class_name, threshold in training_results['optimal_thresholds'].items():
            # Safety check: ensure threshold is a number
            if isinstance(threshold, (int, float)):
                lines.append(f"  {class_name}: {threshold:.3f}")
            else:
                lines.append(f"  {class_name}: {threshold}")
    else:
        lines.append("Threshold Optimization: Disabled")
    lines.append("")

    # Explainability Analysis
    lines.append("EXPLAINABILITY ANALYSIS")
    lines.append("-" * 70)
    if training_results.get('shap_available'):
        lines.append(f"SHAP Analysis: Completed")
        lines.append(f"SHAP Method: {training_results.get('shap_method', 'standard').upper()}")
        lines.append(f"SHAP Samples: {training_results.get('shap_n_samples', 'All')}")
    else:
        lines.append("SHAP Analysis: Not performed")
    lines.append("")

    # Hardware Configuration
    if training_results.get('hardware_info'):
        hw = training_results['hardware_info']
        lines.append("HARDWARE CONFIGURATION")
        lines.append("-" * 70)
        lines.append(f"Processor: {hw.get('processor', 'Unknown')}")
        lines.append(f"Architecture: {hw.get('architecture', 'Unknown')}")
        lines.append(f"CPU Cores: {hw.get('cpu_cores_physical', '?')} physical, {hw.get('cpu_cores_logical', '?')} logical")
        lines.append(f"Memory: {hw.get('memory_gb', '?')} GB")
        lines.append(f"XGBoost tree_method: {hw.get('recommended_tree_method', 'hist')}")
        lines.append(f"XGBoost n_jobs: {hw.get('recommended_n_jobs', -1)}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF TRAINING SUMMARY")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_performance_summary(zone_name, zone_key, y_true, y_pred, y_proba, class_names,
                                 performance_targets=None, ordinal_enabled=False):
    """
    Generate comprehensive performance summary.

    Args:
        zone_name: Display name of zone
        zone_key: Zone key
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        class_names: Dictionary mapping class indices to names
        performance_targets: Dictionary of target metrics (optional)
        ordinal_enabled: Whether ordinal classification was used (optional)

    Returns:
        str: Formatted performance summary
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 70)
    lines.append(f"PERFORMANCE SUMMARY: {zone_name}")
    lines.append(f"Zone Key: {zone_key}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("=" * 70)
    lines.append("")

    # Overall Metrics
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 70)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    lines.append(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Macro and weighted metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    lines.append(f"F1-Score (macro): {f1_macro:.4f}")
    lines.append(f"F1-Score (weighted): {f1_weighted:.4f}")
    lines.append(f"Precision (macro): {precision_macro:.4f}")
    lines.append(f"Recall (macro): {recall_macro:.4f}")
    lines.append("")

    # Performance Targets
    if performance_targets:
        lines.append("PERFORMANCE vs TARGETS")
        lines.append("-" * 70)
        targets_met = 0
        targets_total = 0

        # Check for various possible target keys
        if 'f1_macro' in performance_targets:
            target = performance_targets['f1_macro']
            if isinstance(target, (int, float)):
                met = f1_macro >= target
                targets_met += int(met)
                targets_total += 1
                status = "✓ MET" if met else "✗ BELOW TARGET"
                lines.append(f"F1 Macro: {f1_macro:.4f} (target: {target:.4f}) {status}")

        if 'overall_f1' in performance_targets:
            target = performance_targets['overall_f1']
            if isinstance(target, (int, float)):
                met = f1_weighted >= target
                targets_met += int(met)
                targets_total += 1
                status = "✓ MET" if met else "✗ BELOW TARGET"
                lines.append(f"F1 Weighted: {f1_weighted:.4f} (target: {target:.4f}) {status}")

        if 'accuracy' in performance_targets:
            target = performance_targets['accuracy']
            if isinstance(target, (int, float)):
                met = accuracy >= target
                targets_met += int(met)
                targets_total += 1
                status = "✓ MET" if met else "✗ BELOW TARGET"
                lines.append(f"Accuracy: {accuracy:.4f} (target: {target:.4f}) {status}")

        if 'balanced_accuracy' in performance_targets:
            target = performance_targets['balanced_accuracy']
            if isinstance(target, (int, float)):
                # Calculate balanced accuracy
                from sklearn.metrics import balanced_accuracy_score
                bal_acc = balanced_accuracy_score(y_true, y_pred)
                met = bal_acc >= target
                targets_met += int(met)
                targets_total += 1
                status = "✓ MET" if met else "✗ BELOW TARGET"
                lines.append(f"Balanced Accuracy: {bal_acc:.4f} (target: {target:.4f}) {status}")

        if targets_total > 0:
            lines.append(f"\nTargets Met: {targets_met}/{targets_total}")
        lines.append("")

    # Per-Class Performance
    lines.append("PER-CLASS PERFORMANCE")
    lines.append("-" * 70)
    lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 70)

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    for class_idx in sorted(class_names.keys()):
        class_name = class_names[class_idx]
        str_class_idx = str(class_idx)

        if str_class_idx in report_dict:
            metrics = report_dict[str_class_idx]
            # Safety check: ensure metrics are in expected format
            try:
                precision = float(metrics['precision']) if isinstance(metrics, dict) and 'precision' in metrics else 0.0
                recall = float(metrics['recall']) if isinstance(metrics, dict) and 'recall' in metrics else 0.0
                f1 = float(metrics['f1-score']) if isinstance(metrics, dict) and 'f1-score' in metrics else 0.0
                support = int(metrics['support']) if isinstance(metrics, dict) and 'support' in metrics else 0
                lines.append(f"{class_name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error formatting metrics for class {class_name}: {e}")
                lines.append(f"{class_name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    lines.append("")

    # Confusion Matrix
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 70)
    cm = confusion_matrix(y_true, y_pred)

    # Header
    header = "True \\ Pred  "
    for class_idx in sorted(class_names.keys()):
        header += f"{class_names[class_idx][:8]:>10s}"
    lines.append(header)
    lines.append("-" * 70)

    # Rows
    for i, class_idx in enumerate(sorted(class_names.keys())):
        row = f"{class_names[class_idx]:<13}"
        for j in range(len(class_names)):
            if i < cm.shape[0] and j < cm.shape[1]:
                row += f"{cm[i, j]:>10d}"
            else:
                row += f"{'0':>10s}"
        lines.append(row)

    lines.append("")

    # Ordinal Classification Metrics (if enabled)
    if ordinal_enabled and len(class_names) == 3:
        lines.append("ORDINAL CLASSIFICATION METRICS")
        lines.append("-" * 70)

        # Calculate ordinal-specific metrics
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # Mean Absolute Error (ordinal distance)
        mae = np.mean(np.abs(y_true_arr - y_pred_arr))
        lines.append(f"Mean Absolute Error (MAE): {mae:.4f}")

        # Adjacent Accuracy (predictions within 1 class)
        adjacent_correct = np.sum(np.abs(y_true_arr - y_pred_arr) <= 1)
        adjacent_accuracy = adjacent_correct / len(y_true_arr)
        lines.append(f"Adjacent Accuracy (±1 class): {adjacent_accuracy:.4f} ({adjacent_accuracy*100:.2f}%)")

        # Spearman correlation (ordinal association)
        try:
            from scipy.stats import spearmanr
            spearman_corr, spearman_p = spearmanr(y_true_arr, y_pred_arr)
            if not np.isnan(spearman_corr):
                lines.append(f"Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
        except Exception:
            pass

        # Off-by-2 errors (worst case: predicting Complete when None or vice versa)
        off_by_2 = np.sum(np.abs(y_true_arr - y_pred_arr) == 2)
        off_by_2_pct = off_by_2 / len(y_true_arr) * 100
        lines.append(f"Off-by-2 Errors (None↔Complete): {off_by_2} ({off_by_2_pct:.1f}%)")

        lines.append("")

    # Prediction Confidence Analysis
    lines.append("PREDICTION CONFIDENCE")
    lines.append("-" * 70)
    max_probs = np.max(y_proba, axis=1)
    lines.append(f"Mean Confidence: {np.mean(max_probs):.4f}")
    lines.append(f"Median Confidence: {np.median(max_probs):.4f}")
    lines.append(f"Min Confidence: {np.min(max_probs):.4f}")
    lines.append(f"Max Confidence: {np.max(max_probs):.4f}")

    # Low confidence predictions
    low_conf_threshold = 0.5
    low_conf_count = np.sum(max_probs < low_conf_threshold)
    lines.append(f"Low Confidence Predictions (<{low_conf_threshold}): {low_conf_count} ({low_conf_count/len(y_true)*100:.1f}%)")
    lines.append("")

    lines.append("=" * 70)
    lines.append("END OF PERFORMANCE SUMMARY")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_summary(summary_text, output_path, zone_name):
    """
    Save summary to file.

    Args:
        summary_text: Formatted summary text
        output_path: Path to save file
        zone_name: Zone name for logging
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(summary_text)
        logger.info(f"[{zone_name}] Summary saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"[{zone_name}] Failed to save summary: {e}")
        return False


if __name__ == "__main__":
    # Test summary generation
    print("Training Summary Module")
    print("=" * 70)
    print("This module generates auto-saved training and performance summaries")
    print("to avoid context bloat when discussing model performance.")
