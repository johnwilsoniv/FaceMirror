"""
Feature extraction for ML-based lower face paralysis detection.
Processes data from combined_results.csv and expert labels for training.
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Configure logging to use logs subfolder
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

def prepare_data():
    """
    Prepare dataset for ML model training by merging detection results with expert labels.

    Returns:
        tuple: (features DataFrame, targets array)
    """
    # Load datasets
    logger.info("Loading datasets...")
    try:
        results_df = pd.read_csv("combined_results.csv")
        expert_df = pd.read_csv("FPRS FP Key.csv")

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

    # Check class distribution
    unique, counts = np.unique(targets, return_counts=True)
    class_dist = dict(zip(['None', 'Partial', 'Complete'], counts))
    logger.info(f"Class distribution before SMOTE: {class_dist}")

    # Apply moderate SMOTE to handle class imbalance
    # Focus on improving partial paralysis detection without being too aggressive
    logger.info("Original class counts before SMOTE: {0: None, 1: Partial, 2: Complete}")
    logger.info(f"Original counts: {counts}")
    
    # Use a more moderate sampling strategy for Partial class (1.5x instead of 2x)
    # Keep None and Complete at their original levels
    if len(counts) >= 3:  # Make sure we have all three classes
        smote_sampling_strategy = {0: counts[0], 1: int(counts[0] * 1.5), 2: counts[2]}
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1), sampling_strategy=smote_sampling_strategy)
        features_resampled, targets_resampled = smote.fit_resample(features, targets)
        
        # Log the new class distribution
        new_unique, new_counts = np.unique(targets_resampled, return_counts=True)
        logger.info(f"New class distribution after SMOTE: {dict(zip(new_unique, new_counts))}")
    else:
        # If we don't have all classes, just return the original data
        logger.warning("Not all classes present, skipping SMOTE")
        features_resampled, targets_resampled = features, targets
    
    # Check resampled class distribution
    unique, counts = np.unique(targets_resampled, return_counts=True)
    class_dist = dict(zip(['None', 'Partial', 'Complete'], counts))
    logger.info(f"Class distribution after SMOTE: {class_dist}")

    logger.info(f"Final dataset: {len(features_resampled)} samples with {features_resampled.shape[1]} features")

    return features_resampled, targets_resampled

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

    # Focus exclusively on BS action for lower face 
    action = 'BS'

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

    # 8. Additional features that might help with borderline cases
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
        features[f"{action}_AU12_ratio"] * 0.6 + features[f"{action}_AU25_ratio"] * 0.4
    )
    
    # Average percent difference
    features[f"{action}_avg_percent_diff"] = (
        features[f"{action}_AU12_percent_diff"] * 0.6 + features[f"{action}_AU25_percent_diff"] * 0.4
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
    val1_copy = val1_copy.replace(0, 0.0001)
    val2_copy = val2_copy.replace(0, 0.0001)

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
    avg = avg.replace(0, 0.0001)

    # Calculate percent difference
    percent_diff = (abs_diff / avg) * 100

    # Cap at 200% for extreme differences
    percent_diff[percent_diff > 200] = 200

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