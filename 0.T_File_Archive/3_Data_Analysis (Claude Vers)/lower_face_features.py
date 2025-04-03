"""
Feature extraction for lower face paralysis detection.
Standardized preprocessing and feature vector creation.
"""

import numpy as np
import pandas as pd
import logging
from lower_face_config import FEATURE_CONFIG, LOG_DIR
import os

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """
    Prepare dataset for ML model training by merging detection results with expert labels.

    Args:
        results_file (str): Path to combined results CSV
        expert_file (str): Path to expert labels CSV

    Returns:
        tuple: (features DataFrame, targets array)
    """
    # Load datasets
    logger.info("Loading datasets...")
    try:
        results_df = pd.read_csv(results_file)
        expert_df = pd.read_csv(expert_file)

        logger.info(f"Loaded {len(results_df)} detection results and {len(expert_df)} expert grades")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    # Rename columns for consistent joining
    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        'Paralysis - Left Lower Face': 'Expert_Left_Lower_Face',
        'Paralysis - Right Lower Face': 'Expert_Right_Lower_Face'
    })

    # Merge datasets on Patient ID
    merged_df = pd.merge(results_df, expert_df, on='Patient ID', how='inner')
    logger.info(f"Merged dataset contains {len(merged_df)} patients")

    # Create feature dataframes for left and right sides
    left_features = extract_features(merged_df, 'Left')
    right_features = extract_features(merged_df, 'Right')

    # Process targets for both sides
    left_targets = process_targets(merged_df['Expert_Left_Lower_Face'])
    right_targets = process_targets(merged_df['Expert_Right_Lower_Face'])

    # Combine sides into one dataset, adding a 'side' feature
    left_features['side'] = 0  # 0 for left
    right_features['side'] = 1  # 1 for right

    features = pd.concat([left_features, right_features], ignore_index=True)
    targets = np.concatenate([left_targets, right_targets])

    # Handle missing values
    features = features.fillna(0)

    logger.info(f"Final dataset: {len(features)} samples with {features.shape[1]} features")

    return features, targets


def extract_features(df, side):
    """
    Extract relevant features for lower face paralysis detection.
    Focuses on Big Smile (BS) action for lower face analysis.

    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'

    Returns:
        pandas.DataFrame: Features for the specified side
    """
    features = pd.DataFrame()

    # Use configured action (BS for lower face)
    action = FEATURE_CONFIG['actions'][0]

    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'

    # Build column names for AU values
    au12_col = f"{action}_{side} AU12_r"
    au25_col = f"{action}_{side} AU25_r"
    au12_norm_col = f"{action}_{side} AU12_r (Normalized)"
    au25_norm_col = f"{action}_{side} AU25_r (Normalized)"

    # Opposite side columns
    au12_opp_col = f"{action}_{opposite_side} AU12_r"
    au25_opp_col = f"{action}_{opposite_side} AU25_r"
    au12_opp_norm_col = f"{action}_{opposite_side} AU12_r (Normalized)"
    au25_opp_norm_col = f"{action}_{opposite_side} AU25_r (Normalized)"

    # 1. Raw AU values
    if au12_col in df.columns:
        features[f"{action}_AU12_raw"] = df[au12_col]
    else:
        features[f"{action}_AU12_raw"] = 0

    if au25_col in df.columns:
        features[f"{action}_AU25_raw"] = df[au25_col]
    else:
        features[f"{action}_AU25_raw"] = 0

    # 2. Normalized AU values (these are crucial for better generalization)
    if au12_norm_col in df.columns:
        features[f"{action}_AU12_norm"] = df[au12_norm_col]
    else:
        features[f"{action}_AU12_norm"] = features[f"{action}_AU12_raw"]

    if au25_norm_col in df.columns:
        features[f"{action}_AU25_norm"] = df[au25_norm_col]
    else:
        features[f"{action}_AU25_norm"] = features[f"{action}_AU25_raw"]

    # 3. Opposite side raw AU values
    if au12_opp_col in df.columns:
        features[f"{action}_opp_AU12_raw"] = df[au12_opp_col]
    else:
        features[f"{action}_opp_AU12_raw"] = 0

    if au25_opp_col in df.columns:
        features[f"{action}_opp_AU25_raw"] = df[au25_opp_col]
    else:
        features[f"{action}_opp_AU25_raw"] = 0

    # 4. Opposite side normalized AU values
    if au12_opp_norm_col in df.columns:
        features[f"{action}_opp_AU12_norm"] = df[au12_opp_norm_col]
    else:
        features[f"{action}_opp_AU12_norm"] = features[f"{action}_opp_AU12_raw"]

    if au25_opp_norm_col in df.columns:
        features[f"{action}_opp_AU25_norm"] = df[au25_opp_norm_col]
    else:
        features[f"{action}_opp_AU25_norm"] = features[f"{action}_opp_AU25_raw"]

    # 5. Asymmetry metrics
    # Ratio metrics (min/max)
    features[f"{action}_AU12_ratio"] = calculate_ratio(
        features[f"{action}_AU12_norm"],
        features[f"{action}_opp_AU12_norm"]
    )

    features[f"{action}_AU25_ratio"] = calculate_ratio(
        features[f"{action}_AU25_norm"],
        features[f"{action}_opp_AU25_norm"]
    )

    # Percent difference metrics
    features[f"{action}_AU12_percent_diff"] = calculate_percent_diff(
        features[f"{action}_AU12_norm"],
        features[f"{action}_opp_AU12_norm"]
    )

    features[f"{action}_AU25_percent_diff"] = calculate_percent_diff(
        features[f"{action}_AU25_norm"],
        features[f"{action}_opp_AU25_norm"]
    )

    # 6. Directional asymmetry (which side is weaker)
    features[f"{action}_AU12_is_weaker"] = (
            features[f"{action}_AU12_norm"] < features[f"{action}_opp_AU12_norm"]
    ).astype(int)

    features[f"{action}_AU25_is_weaker"] = (
            features[f"{action}_AU25_norm"] < features[f"{action}_opp_AU25_norm"]
    ).astype(int)

    # 7. Combined feature metrics
    # AU12 and AU25 interaction
    features[f"{action}_AU12_AU25_interaction"] = (
            features[f"{action}_AU12_norm"] * features[f"{action}_AU25_norm"]
    )

    # Asymmetry interaction (ratio product)
    features[f"{action}_ratio_product"] = (
            features[f"{action}_AU12_ratio"] * features[f"{action}_AU25_ratio"]
    )

    # Asymmetry differential (difference between AU12 and AU25 asymmetry)
    features[f"{action}_asymmetry_differential"] = abs(
        features[f"{action}_AU12_percent_diff"] - features[f"{action}_AU25_percent_diff"]
    )

    # 8. Additional features for borderline cases
    # AU intensity sum (overall expressiveness)
    features[f"{action}_AU_intensity_sum"] = (
            features[f"{action}_AU12_norm"] + features[f"{action}_AU25_norm"]
    )

    # AU intensity ratio (relative contribution of AU12 vs AU25)
    denominator = features[f"{action}_AU12_norm"] + features[f"{action}_AU25_norm"]
    features[f"{action}_AU12_contribution"] = np.where(
        denominator > 0,
        features[f"{action}_AU12_norm"] / denominator,
        0
    )

    # Average ratio
    features[f"{action}_avg_ratio"] = (
            features[f"{action}_AU12_ratio"] * FEATURE_CONFIG['au_weights']['AU12_r'] +
            features[f"{action}_AU25_ratio"] * FEATURE_CONFIG['au_weights']['AU25_r']
    )

    # Average percent difference
    features[f"{action}_avg_percent_diff"] = (
            features[f"{action}_AU12_percent_diff"] * FEATURE_CONFIG['au_weights']['AU12_r'] +
            features[f"{action}_AU25_percent_diff"] * FEATURE_CONFIG['au_weights']['AU25_r']
    )

    return features


def calculate_ratio(val1, val2):
    """
    Calculate ratio between values (min/max).

    Args:
        val1 (pandas.Series): First value series
        val2 (pandas.Series): Second value series

    Returns:
        pandas.Series: Ratio values
    """
    # Create copies to avoid warnings
    val1_copy = val1.copy()
    val2_copy = val2.copy()

    # Replace zeros with small value to avoid division by zero
    min_val = FEATURE_CONFIG['min_value']
    val1_copy = val1_copy.replace(0, min_val)
    val2_copy = val2_copy.replace(0, min_val)

    min_vals = pd.Series([min(a, b) for a, b in zip(val1_copy, val2_copy)])
    max_vals = pd.Series([max(a, b) for a, b in zip(val1_copy, val2_copy)])

    # Calculate ratio (capped at 1.0)
    ratio = min_vals / max_vals
    ratio[ratio > 1.0] = 1.0  # Cap at 1.0 in case of floating point issues

    return ratio


def calculate_percent_diff(val1, val2):
    """
    Calculate percent difference between values.

    Args:
        val1 (pandas.Series): First value series
        val2 (pandas.Series): Second value series

    Returns:
        pandas.Series: Percent difference values
    """
    # Create copies to avoid warnings
    val1_copy = val1.copy()
    val2_copy = val2.copy()

    # Calculate absolute difference
    abs_diff = abs(val1_copy - val2_copy)

    # Calculate average of the two values
    avg = (val1_copy + val2_copy) / 2

    # Replace zeros with small value to avoid division by zero
    avg = avg.replace(0, FEATURE_CONFIG['min_value'])

    # Calculate percent difference
    percent_diff = (abs_diff / avg) * 100

    # Cap at configured value for extreme differences
    percent_diff[percent_diff > FEATURE_CONFIG['percent_diff_cap']] = FEATURE_CONFIG['percent_diff_cap']

    return percent_diff


def process_targets(target_series):
    """
    Convert text labels to numerical targets.

    Args:
        target_series (pandas.Series): Series of text labels

    Returns:
        numpy.ndarray: Numerical target values
    """
    # Mapping for different possible label formats
    mapping = {
        'None': 0, 'no': 0, 'No': 0, 'N/A': 0, '': 0, 'normal': 0, 'Normal': 0,
        'Partial': 1, 'partial': 1, 'mild': 1, 'Mild': 1, 'moderate': 1, 'Moderate': 1,
        'Complete': 2, 'complete': 2, 'severe': 2, 'Severe': 2
    }

    # Map labels to numerical values
    processed = target_series.map(mapping)

    # Fill missing/NaN values with 0 (None)
    processed = processed.fillna(0)

    # Convert to int type
    return processed.astype(int).values


def extract_features_for_detection(info, side, zone, aus, values, other_values,
                                   values_normalized, other_values_normalized):
    """
    Extract features for detection from info dictionary.
    Used during real-time detection rather than training.

    Args:
        info (dict): Results dictionary
        side (str): Side being analyzed ('left' or 'right')
        zone (str): Zone being analyzed ('lower')
        aus (list): Action Units for this zone
        values (dict): AU values for this side
        other_values (dict): AU values for opposite side
        values_normalized (dict): Normalized AU values for this side
        other_values_normalized (dict): Normalized AU values for opposite side

    Returns:
        list: Feature vector for model input
    """
    # Only use Big Smile action
    action = FEATURE_CONFIG['actions'][0]

    # Initialize feature dictionary
    features = {}

    # Extract AU values
    au12_val = values.get('AU12_r', 0)
    au25_val = values.get('AU25_r', 0)
    au12_other = other_values.get('AU12_r', 0)
    au25_other = other_values.get('AU25_r', 0)

    features[f"{action}_AU12_raw"] = au12_val
    features[f"{action}_AU25_raw"] = au25_val
    features[f"{action}_opp_AU12_raw"] = au12_other
    features[f"{action}_opp_AU25_raw"] = au25_other

    # Normalized values
    au12_norm = values_normalized.get('AU12_r', au12_val) if values_normalized else au12_val
    au25_norm = values_normalized.get('AU25_r', au25_val) if values_normalized else au25_val
    au12_other_norm = other_values_normalized.get('AU12_r', au12_other) if other_values_normalized else au12_other
    au25_other_norm = other_values_normalized.get('AU25_r', au25_other) if other_values_normalized else au25_other

    features[f"{action}_AU12_norm"] = au12_norm
    features[f"{action}_AU25_norm"] = au25_norm
    features[f"{action}_opp_AU12_norm"] = au12_other_norm
    features[f"{action}_opp_AU25_norm"] = au25_other_norm

    # Calculate asymmetry metrics
    # Ratio metrics (min/max)
    features[f"{action}_AU12_ratio"] = min(au12_norm, au12_other_norm) / max(au12_norm, au12_other_norm,
                                                                             FEATURE_CONFIG['min_value'])
    features[f"{action}_AU25_ratio"] = min(au25_norm, au25_other_norm) / max(au25_norm, au25_other_norm,
                                                                             FEATURE_CONFIG['min_value'])

    # Percent difference metrics
    features[f"{action}_AU12_percent_diff"] = calculate_single_percent_diff(au12_norm, au12_other_norm)
    features[f"{action}_AU25_percent_diff"] = calculate_single_percent_diff(au25_norm, au25_other_norm)

    # Directional asymmetry (which side is weaker)
    features[f"{action}_AU12_is_weaker"] = 1 if au12_norm < au12_other_norm else 0
    features[f"{action}_AU25_is_weaker"] = 1 if au25_norm < au25_other_norm else 0

    # Enhanced features - AU interactions
    features[f"{action}_AU12_AU25_interaction"] = au12_norm * au25_norm
    features[f"{action}_ratio_product"] = features[f"{action}_AU12_ratio"] * features[f"{action}_AU25_ratio"]
    features[f"{action}_asymmetry_differential"] = abs(
        features[f"{action}_AU12_percent_diff"] - features[f"{action}_AU25_percent_diff"]
    )

    # Side indicator
    features["side"] = 1 if side.lower() == 'right' else 0

    # Additional features
    # AU intensity sum (overall expressiveness)
    features[f"{action}_AU_intensity_sum"] = au12_norm + au25_norm

    # AU intensity ratio (relative contribution of AU12 vs AU25)
    denominator = au12_norm + au25_norm
    features[f"{action}_AU12_contribution"] = au12_norm / denominator if denominator > 0 else 0

    # Average ratio
    features[f"{action}_avg_ratio"] = (
            features[f"{action}_AU12_ratio"] * FEATURE_CONFIG['au_weights']['AU12_r'] +
            features[f"{action}_AU25_ratio"] * FEATURE_CONFIG['au_weights']['AU25_r']
    )

    # Average percent difference
    features[f"{action}_avg_percent_diff"] = (
            features[f"{action}_AU12_percent_diff"] * FEATURE_CONFIG['au_weights']['AU12_r'] +
            features[f"{action}_AU25_percent_diff"] * FEATURE_CONFIG['au_weights']['AU25_r']
    )

    # Create feature vector
    return list(features.values())


def calculate_single_percent_diff(val1, val2):
    """Calculate percent difference between two single values."""
    if val1 == 0 and val2 == 0:
        return 0.0

    # Calculate absolute difference
    abs_diff = abs(val1 - val2)

    # Calculate average
    avg = (val1 + val2) / 2.0

    # Avoid division by zero
    if avg <= 0:
        return FEATURE_CONFIG['percent_diff_cap'] if val1 != val2 else 0.0

    # Calculate percent difference
    percent_diff = (abs_diff / avg) * 100.0

    # Cap at max value for extreme differences
    return min(percent_diff, FEATURE_CONFIG['percent_diff_cap'])