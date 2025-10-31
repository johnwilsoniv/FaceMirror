"""
SHAP Analysis Module for Paralysis Detection Training Pipeline

Provides SHAP (SHapley Additive exPlanations) analysis for model interpretability.
Uses FastTreeSHAP when available for 1.5-2.7x faster computation.
"""

import logging
import numpy as np
import pandas as pd
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import SHAP and FastTreeSHAP
SHAP_AVAILABLE = False
FASTTREESHAP_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
    logger.debug("SHAP library loaded successfully")
except ImportError:
    logger.warning("SHAP library not available. Install with: pip install shap")

try:
    import fasttreeshap
    FASTTREESHAP_AVAILABLE = True
    logger.debug("FastTreeSHAP library loaded successfully (1.5-2.7x faster)")
except ImportError:
    logger.debug("FastTreeSHAP not available, will use standard SHAP TreeExplainer")


def compute_shap_values(model, X_data, feature_names, zone_name="Model", use_fasttreeshap=True):
    """
    Compute SHAP values for model explanations.

    Args:
        model: Trained XGBoost or tree-based model
        X_data: Input features (numpy array or pandas DataFrame)
        feature_names: List of feature names
        zone_name: Name of the zone for logging (e.g., "Upper Face")
        use_fasttreeshap: Whether to try using FastTreeSHAP (default: True)

    Returns:
        dict: {
            'shap_values': SHAP values array (n_samples, n_features, n_classes) or (n_samples, n_features),
            'base_values': Base values (expected model output),
            'explainer': The SHAP explainer object,
            'feature_names': Feature names used,
            'computation_method': 'fasttreeshap' or 'standard'
        }
        Returns None if SHAP computation fails.
    """
    if not SHAP_AVAILABLE:
        logger.error(f"[{zone_name}] SHAP library not available. Cannot compute SHAP values.")
        return None

    try:
        logger.info(f"[{zone_name}] Computing SHAP values for {X_data.shape[0]} samples with {X_data.shape[1]} features...")

        # Convert to numpy array if DataFrame
        if isinstance(X_data, pd.DataFrame):
            X_array = X_data.values
        else:
            X_array = X_data

        # Determine which explainer to use
        computation_method = 'standard'

        if use_fasttreeshap and FASTTREESHAP_AVAILABLE:
            try:
                logger.info(f"[{zone_name}] Using FastTreeSHAP for faster computation...")
                explainer = fasttreeshap.TreeExplainer(
                    model,
                    algorithm='v2',  # FastTreeSHAP v2 algorithm
                    n_jobs=-1,  # Use all available cores
                    shortcut=False  # More accurate
                )
                computation_method = 'fasttreeshap'
            except Exception as e:
                logger.warning(f"[{zone_name}] FastTreeSHAP failed ({e}), falling back to standard SHAP")
                explainer = shap.TreeExplainer(model)
        else:
            if use_fasttreeshap and not FASTTREESHAP_AVAILABLE:
                logger.info(f"[{zone_name}] FastTreeSHAP requested but not available, using standard SHAP")
            explainer = shap.TreeExplainer(model)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_array)
        base_values = explainer.expected_value

        logger.info(f"[{zone_name}] SHAP computation complete using {computation_method}")

        # For multiclass, shap_values is a list of arrays (one per class)
        # For binary, it's a single array
        if isinstance(shap_values, list):
            logger.info(f"[{zone_name}] Multiclass SHAP: {len(shap_values)} classes, shape per class: {shap_values[0].shape}")
        else:
            logger.info(f"[{zone_name}] Binary SHAP values shape: {shap_values.shape}")

        return {
            'shap_values': shap_values,
            'base_values': base_values,
            'explainer': explainer,
            'feature_names': feature_names,
            'computation_method': computation_method,
            'n_samples': X_array.shape[0],
            'n_features': X_array.shape[1]
        }

    except Exception as e:
        logger.error(f"[{zone_name}] Error computing SHAP values: {e}", exc_info=True)
        return None


def compute_shap_feature_importance(shap_result, class_names=None):
    """
    Compute feature importance from SHAP values.

    Args:
        shap_result: Dictionary returned from compute_shap_values()
        class_names: Dictionary mapping class indices to names (e.g., {0: 'None', 1: 'Partial', 2: 'Complete'})

    Returns:
        pd.DataFrame: Feature importance scores with columns:
            - feature: Feature name
            - mean_abs_shap: Mean absolute SHAP value across all samples and classes
            - For each class: mean_abs_shap_class_{i} (if multiclass)
    """
    if shap_result is None:
        return None

    shap_values = shap_result['shap_values']
    feature_names = shap_result['feature_names']

    # Handle multiclass vs binary
    if isinstance(shap_values, list):
        # Multiclass: compute mean absolute SHAP for each class
        n_classes = len(shap_values)
        importance_data = {'feature': feature_names}

        class_mean_abs_shaps = []
        for class_idx, class_shap_vals in enumerate(shap_values):
            mean_abs_shap_class = np.mean(np.abs(class_shap_vals), axis=0)
            class_name = class_names.get(class_idx, f"Class_{class_idx}") if class_names else f"Class_{class_idx}"
            importance_data[f'mean_abs_shap_{class_name}'] = mean_abs_shap_class
            class_mean_abs_shaps.append(mean_abs_shap_class)

        # Overall importance: average across all classes
        importance_data['mean_abs_shap_overall'] = np.mean(class_mean_abs_shaps, axis=0)

    else:
        # Binary classification
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        importance_data = {
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }

    importance_df = pd.DataFrame(importance_data)

    # Sort by overall importance (or single importance for binary)
    sort_col = 'mean_abs_shap_overall' if 'mean_abs_shap_overall' in importance_df.columns else 'mean_abs_shap'
    importance_df = importance_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    return importance_df


def save_shap_importance(shap_importance_df, output_path, zone_name="Model"):
    """
    Save SHAP-based feature importance to CSV.

    Args:
        shap_importance_df: DataFrame from compute_shap_feature_importance()
        output_path: Path to save CSV file
        zone_name: Zone name for logging
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shap_importance_df.to_csv(output_path, index=False)
        logger.info(f"[{zone_name}] SHAP feature importance saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"[{zone_name}] Failed to save SHAP importance: {e}")
        return False


def generate_shap_summary(shap_result, shap_importance_df, class_names=None):
    """
    Generate a text summary of SHAP analysis for logging or display.

    Args:
        shap_result: Dictionary from compute_shap_values()
        shap_importance_df: DataFrame from compute_shap_feature_importance()
        class_names: Dictionary mapping class indices to names

    Returns:
        str: Formatted summary text
    """
    if shap_result is None or shap_importance_df is None:
        return "SHAP analysis not available"

    lines = []
    lines.append("=" * 60)
    lines.append("SHAP FEATURE IMPORTANCE ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Computation method: {shap_result['computation_method'].upper()}")
    lines.append(f"Samples analyzed: {shap_result['n_samples']}")
    lines.append(f"Features analyzed: {shap_result['n_features']}")
    lines.append("")

    # Top features overall
    if 'mean_abs_shap_overall' in shap_importance_df.columns:
        sort_col = 'mean_abs_shap_overall'
    else:
        sort_col = 'mean_abs_shap'

    top_n = min(15, len(shap_importance_df))
    lines.append(f"TOP {top_n} MOST IMPORTANT FEATURES (by SHAP):")
    lines.append("-" * 60)

    for idx, row in shap_importance_df.head(top_n).iterrows():
        feature_name = row['feature']
        importance_val = row[sort_col]
        lines.append(f"{idx + 1:2d}. {feature_name:40s} {importance_val:.6f}")

    lines.append("=" * 60)

    return "\n".join(lines)


def run_full_shap_analysis(model, X_data, feature_names, zone_name, class_names,
                           save_path=None, use_fasttreeshap=True):
    """
    Run complete SHAP analysis workflow.

    Args:
        model: Trained model
        X_data: Input features
        feature_names: List of feature names
        zone_name: Zone name for logging
        class_names: Dictionary mapping class indices to names
        save_path: Path to save SHAP importance CSV (optional)
        use_fasttreeshap: Whether to use FastTreeSHAP if available

    Returns:
        dict: {
            'shap_result': Raw SHAP computation result,
            'importance_df': SHAP-based feature importance DataFrame,
            'summary_text': Formatted summary text
        }
        Returns None if SHAP analysis fails.
    """
    logger.info(f"[{zone_name}] Starting full SHAP analysis...")

    # Compute SHAP values
    shap_result = compute_shap_values(model, X_data, feature_names, zone_name, use_fasttreeshap)
    if shap_result is None:
        return None

    # Compute feature importance from SHAP
    importance_df = compute_shap_feature_importance(shap_result, class_names)
    if importance_df is None:
        return None

    # Generate summary text
    summary_text = generate_shap_summary(shap_result, importance_df, class_names)

    # Save if path provided
    if save_path:
        save_shap_importance(importance_df, save_path, zone_name)

    # Log summary
    logger.info(f"\n{summary_text}")

    return {
        'shap_result': shap_result,
        'importance_df': importance_df,
        'summary_text': summary_text
    }


if __name__ == "__main__":
    # Test SHAP availability
    import logging
    logging.basicConfig(level=logging.INFO)

    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"FastTreeSHAP available: {FASTTREESHAP_AVAILABLE}")

    if FASTTREESHAP_AVAILABLE:
        print("✓ FastTreeSHAP will be used (1.5-2.7x faster)")
    elif SHAP_AVAILABLE:
        print("○ Standard SHAP will be used (install fasttreeshap for speedup)")
    else:
        print("✗ SHAP not available (install with: pip install shap fasttreeshap)")
