"""
Feature extraction for mid face paralysis detection.
Standardized preprocessing and feature vector creation.
"""

import numpy as np
import pandas as pd
import logging
from mid_face_config import FEATURE_CONFIG, LOG_DIR
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
        'Paralysis - Left Mid Face': 'Expert_Left_Mid_Face',
        'Paralysis - Right Mid Face': 'Expert_Right_Mid_Face'
    })

    # Merge datasets on Patient ID
    merged_df = pd.merge(results_df, expert_df, on='Patient ID', how='inner')
    logger.info(f"Merged dataset contains {len(merged_df)} patients")

    # Create feature dataframes for left and right sides
    left_features = extract_features(merged_df, 'Left')
    right_features = extract_features(merged_df, 'Right')

    # Process targets for both sides
    left_targets = process_targets(merged_df['Expert_Left_Mid_Face'])
    right_targets = process_targets(merged_df['Expert_Right_Mid_Face'])

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
    logger.info(f"Class distribution: {class_dist}")

    logger.info(f"Final dataset: {len(features)} samples with {features.shape[1]} features")

    return features, targets


def extract_features(df, side):
    """
    Extract relevant features for mid face paralysis detection.
    Focuses on ES, ET, and RE actions for mid face analysis.

    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'

    Returns:
        pandas.DataFrame: Features for the specified side
    """
    features = pd.DataFrame()

    # Use configured actions
    actions = FEATURE_CONFIG['actions']

    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'

    # =====================
    # 1. EXTRACT BASIC AU VALUES
    # =====================
    for action in actions:
        # Build column names for AU values
        au45_col = f"{action}_{side} AU45_r"
        au07_col = f"{action}_{side} AU07_r"
        au45_norm_col = f"{action}_{side} AU45_r (Normalized)"
        au07_norm_col = f"{action}_{side} AU07_r (Normalized)"

        # Opposite side columns
        au45_opp_col = f"{action}_{opposite_side} AU45_r"
        au07_opp_col = f"{action}_{opposite_side} AU07_r"
        au45_opp_norm_col = f"{action}_{opposite_side} AU45_r (Normalized)"
        au07_opp_norm_col = f"{action}_{opposite_side} AU07_r (Normalized)"

        # 1.1 Raw AU values
        if au45_col in df.columns:
            features[f"{action}_AU45_raw"] = df[au45_col]
        else:
            features[f"{action}_AU45_raw"] = 0

        if au07_col in df.columns:
            features[f"{action}_AU07_raw"] = df[au07_col]
        else:
            features[f"{action}_AU07_raw"] = 0

        # 1.2 Normalized AU values (these are crucial for better generalization)
        if au45_norm_col in df.columns:
            features[f"{action}_AU45_norm"] = df[au45_norm_col]
        else:
            features[f"{action}_AU45_norm"] = features[f"{action}_AU45_raw"]

        if au07_norm_col in df.columns:
            features[f"{action}_AU07_norm"] = df[au07_norm_col]
        else:
            features[f"{action}_AU07_norm"] = features[f"{action}_AU07_raw"]

        # 1.3 Opposite side raw AU values
        if au45_opp_col in df.columns:
            features[f"{action}_opp_AU45_raw"] = df[au45_opp_col]
        else:
            features[f"{action}_opp_AU45_raw"] = 0

        if au07_opp_col in df.columns:
            features[f"{action}_opp_AU07_raw"] = df[au07_opp_col]
        else:
            features[f"{action}_opp_AU07_raw"] = 0

        # 1.4 Opposite side normalized AU values
        if au45_opp_norm_col in df.columns:
            features[f"{action}_opp_AU45_norm"] = df[au45_opp_norm_col]
        else:
            features[f"{action}_opp_AU45_norm"] = features[f"{action}_opp_AU45_raw"]

        if au07_opp_norm_col in df.columns:
            features[f"{action}_opp_AU07_norm"] = df[au07_opp_norm_col]
        else:
            features[f"{action}_opp_AU07_norm"] = features[f"{action}_opp_AU07_raw"]

        # =====================
        # 2. ASYMMETRY METRICS
        # =====================
        
        # 2.1 Ratio metrics (min/max)
        features[f"{action}_AU45_ratio"] = calculate_ratio(
            features[f"{action}_AU45_norm"],
            features[f"{action}_opp_AU45_norm"]
        )

        features[f"{action}_AU07_ratio"] = calculate_ratio(
            features[f"{action}_AU07_norm"],
            features[f"{action}_opp_AU07_norm"]
        )

        # 2.2 Percent difference metrics
        features[f"{action}_AU45_percent_diff"] = calculate_percent_diff(
            features[f"{action}_AU45_norm"],
            features[f"{action}_opp_AU45_norm"]
        )

        features[f"{action}_AU07_percent_diff"] = calculate_percent_diff(
            features[f"{action}_AU07_norm"],
            features[f"{action}_opp_AU07_norm"]
        )
        
        # 2.3 Directional asymmetry (which side is weaker)
        features[f"{action}_AU45_is_weaker"] = (
                features[f"{action}_AU45_norm"] < features[f"{action}_opp_AU45_norm"]
        ).astype(int)

        features[f"{action}_AU07_is_weaker"] = (
                features[f"{action}_AU07_norm"] < features[f"{action}_opp_AU07_norm"]
        ).astype(int)
        
        # =====================
        # 3. COMBINED METRICS
        # =====================
        
        # 3.1 AU45 and AU07 interaction
        features[f"{action}_AU45_AU07_interaction"] = (
                features[f"{action}_AU45_norm"] * features[f"{action}_AU07_norm"]
        )

        # 3.2 Asymmetry interaction (ratio product)
        features[f"{action}_ratio_product"] = (
                features[f"{action}_AU45_ratio"] * features[f"{action}_AU07_ratio"]
        )

        # 3.3 Asymmetry differential (difference between AU45 and AU07 asymmetry)
        features[f"{action}_asymmetry_differential"] = abs(
            features[f"{action}_AU45_percent_diff"] - features[f"{action}_AU07_percent_diff"]
        )
    
    # =====================
    # 4. CROSS-ACTION FEATURES
    # =====================
    
    # 4.1 Comparison features between ES and ET actions (critical for midface)
    if 'ES_AU45_norm' in features.columns and 'ET_AU45_norm' in features.columns:
        # ES/ET ratio (higher indicates potentially better eye movement)
        et_min_value = features['ET_AU45_norm'].replace(0, FEATURE_CONFIG['min_value'])
        es_min_value = features['ES_AU45_norm'].replace(0, FEATURE_CONFIG['min_value'])
        features['ES_ET_AU45_ratio'] = es_min_value / et_min_value
        features['ET_ES_AU45_diff'] = abs(features['ET_AU45_norm'] - features['ES_AU45_norm'])
    else:
        features['ES_ET_AU45_ratio'] = 1.0
        features['ET_ES_AU45_diff'] = 0.0

    if 'ES_AU07_norm' in features.columns and 'ET_AU07_norm' in features.columns:
        et_min_value = features['ET_AU07_norm'].replace(0, FEATURE_CONFIG['min_value'])
        es_min_value = features['ES_AU07_norm'].replace(0, FEATURE_CONFIG['min_value'])
        features['ES_ET_AU07_ratio'] = es_min_value / et_min_value
        features['ET_ES_AU07_diff'] = abs(features['ET_AU07_norm'] - features['ES_AU07_norm'])
    else:
        features['ES_ET_AU07_ratio'] = 1.0
        features['ET_ES_AU07_diff'] = 0.0
    
    # 4.2 Max/min values across actions
    for au in ['AU45', 'AU07']:
        # Calculate max, min and range across actions
        raw_columns = [f"{action}_{au}_raw" for action in actions if f"{action}_{au}_raw" in features.columns]
        norm_columns = [f"{action}_{au}_norm" for action in actions if f"{action}_{au}_norm" in features.columns]
        
        if raw_columns:
            features[f"max_{au}_raw"] = features[raw_columns].max(axis=1)
            features[f"min_{au}_raw"] = features[raw_columns].min(axis=1)
            features[f"range_{au}_raw"] = features[f"max_{au}_raw"] - features[f"min_{au}_raw"]
            
        if norm_columns:
            features[f"max_{au}_norm"] = features[norm_columns].max(axis=1)
            features[f"min_{au}_norm"] = features[norm_columns].min(axis=1)
            features[f"range_{au}_norm"] = features[f"max_{au}_norm"] - features[f"min_{au}_norm"]

        # Calculate max asymmetry across actions
        percent_diff_columns = [f"{action}_{au}_percent_diff" for action in actions 
                               if f"{action}_{au}_percent_diff" in features.columns]
        if percent_diff_columns:
            features[f"max_{au}_percent_diff"] = features[percent_diff_columns].max(axis=1)
            
        ratio_columns = [f"{action}_{au}_ratio" for action in actions 
                        if f"{action}_{au}_ratio" in features.columns]
        if ratio_columns:
            features[f"min_{au}_ratio"] = features[ratio_columns].min(axis=1)
    
    # =====================
    # 5. FUNCTIONAL FEATURES
    # =====================
    
    # 5.1 Functional blink metrics (for detecting subtle impairment)
    if 'ET_AU45_norm' in features.columns:
        # Calculate functional metrics for eye closure in ET action
        # Higher values (>1.0) indicate better eyelid closure
        features['ET_functional_AU45'] = features['ET_AU45_norm']
        
        # Functional ratio between sides (smaller indicates greater asymmetry)
        if 'ET_AU45_ratio' in features.columns:
            features['ET_functional_ratio'] = features['ET_AU45_ratio']
    
    # =====================
    # 6. ENHANCED DETECTION FEATURES
    # =====================
    
    # 6.1 Combined weighted asymmetry score
    au_weights = FEATURE_CONFIG['au_weights']
    for action in actions:
        if all(f"{action}_{au}_percent_diff" in features.columns for au in ['AU45', 'AU07']):
            features[f"{action}_weighted_asymmetry"] = (
                features[f"{action}_AU45_percent_diff"] * au_weights['AU45_r'] +
                features[f"{action}_AU07_percent_diff"] * au_weights['AU07_r']
            )
    
    # 6.2 ET/ES ratio weighted features
    if all(col in features.columns for col in ['ES_ET_AU45_ratio', 'ES_ET_AU07_ratio']):
        # Higher values indicate potentially more paralysis
        features['weighted_ET_ES_ratio'] = (
            features['ES_ET_AU45_ratio'] * au_weights['AU45_r'] +
            features['ES_ET_AU07_ratio'] * au_weights['AU07_r']
        )
    
    # 6.3 Add current detection result as a feature
    mid_face_col = f"{side} Mid Face Paralysis"
    if mid_face_col in df.columns:
        features['current_detection'] = df[mid_face_col].map(
            {'None': 0, 'Partial': 1, 'Complete': 2, 'none': 0, 'partial': 1, 'complete': 2})
    else:
        features['current_detection'] = 0

    # Ensure all binary features are properly encoded
    for col in features.columns:
        if col.endswith('_is_weaker'):
            features[col] = features[col].astype(int)

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
        zone (str): Zone being analyzed ('mid')
        aus (list): Action Units for this zone
        values (dict): AU values for this side
        other_values (dict): AU values for opposite side
        values_normalized (dict): Normalized AU values for this side
        other_values_normalized (dict): Normalized AU values for opposite side

    Returns:
        list: Feature vector for model input
    """
    # Relevant actions for midface
    actions = FEATURE_CONFIG['actions']

    # Initialize feature dictionary
    features = {}
    
    # Extract action-specific data
    action_data = {}
    for action in actions:
        if action in info:
            action_data[action] = {
                'left': info[action]['left'],
                'right': info[action]['right']
            }
    
    # If we have action data, use it for more complete feature extraction
    if action_data:
        for action in actions:
            if action in action_data:
                side_lower = side.lower()
                opposite_side = 'right' if side_lower == 'left' else 'left'
                
                # Extract AU values
                side_aus = action_data[action][side_lower].get('au_values', {})
                opposite_aus = action_data[action][opposite_side].get('au_values', {})
                
                # Extract normalized values if available
                side_normalized = action_data[action][side_lower].get('normalized_au_values', {})
                opposite_normalized = action_data[action][opposite_side].get('normalized_au_values', {})
                
                # Add raw values
                features[f"{action}_AU45_raw"] = side_aus.get('AU45_r', 0)
                features[f"{action}_AU07_raw"] = side_aus.get('AU07_r', 0)
                features[f"{action}_opp_AU45_raw"] = opposite_aus.get('AU45_r', 0)
                features[f"{action}_opp_AU07_raw"] = opposite_aus.get('AU07_r', 0)
                
                # Add normalized values
                features[f"{action}_AU45_norm"] = side_normalized.get('AU45_r', side_aus.get('AU45_r', 0))
                features[f"{action}_AU07_norm"] = side_normalized.get('AU07_r', side_aus.get('AU07_r', 0))
                features[f"{action}_opp_AU45_norm"] = opposite_normalized.get('AU45_r', opposite_aus.get('AU45_r', 0))
                features[f"{action}_opp_AU07_norm"] = opposite_normalized.get('AU07_r', opposite_aus.get('AU07_r', 0))
    else:
        # Fallback to the provided values (simpler feature set)
        for action in actions:
            # Start with zeros for all features
            features[f"{action}_AU45_raw"] = 0
            features[f"{action}_AU07_raw"] = 0
            features[f"{action}_opp_AU45_raw"] = 0
            features[f"{action}_opp_AU07_raw"] = 0
            features[f"{action}_AU45_norm"] = 0
            features[f"{action}_AU07_norm"] = 0
            features[f"{action}_opp_AU45_norm"] = 0
            features[f"{action}_opp_AU07_norm"] = 0
        
        # Use the values provided for the current action
        # Determine which action we're in
        current_action = "ET"  # Default to ET as it's most important for midface
        
        # Use the current values
        features[f"{current_action}_AU45_raw"] = values.get('AU45_r', 0)
        features[f"{current_action}_AU07_raw"] = values.get('AU07_r', 0)
        features[f"{current_action}_opp_AU45_raw"] = other_values.get('AU45_r', 0)
        features[f"{current_action}_opp_AU07_raw"] = other_values.get('AU07_r', 0)
        
        # Use normalized values if available
        if values_normalized and other_values_normalized:
            features[f"{current_action}_AU45_norm"] = values_normalized.get('AU45_r', values.get('AU45_r', 0))
            features[f"{current_action}_AU07_norm"] = values_normalized.get('AU07_r', values.get('AU07_r', 0))
            features[f"{current_action}_opp_AU45_norm"] = other_values_normalized.get('AU45_r', other_values.get('AU45_r', 0))
            features[f"{current_action}_opp_AU07_norm"] = other_values_normalized.get('AU07_r', other_values.get('AU07_r', 0))
        else:
            features[f"{current_action}_AU45_norm"] = features[f"{current_action}_AU45_raw"]
            features[f"{current_action}_AU07_norm"] = features[f"{current_action}_AU07_raw"]
            features[f"{current_action}_opp_AU45_norm"] = features[f"{current_action}_opp_AU45_raw"]
            features[f"{current_action}_opp_AU07_norm"] = features[f"{current_action}_opp_AU07_raw"]
    
    # Calculate derived features across all actions
    for action in actions:
        if all(f"{action}_{key}" in features for key in ["AU45_norm", "AU07_norm", "opp_AU45_norm", "opp_AU07_norm"]):
            # Calculate ratio metrics
            features[f"{action}_AU45_ratio"] = min(
                features[f"{action}_AU45_norm"], 
                features[f"{action}_opp_AU45_norm"]
            ) / max(
                features[f"{action}_AU45_norm"], 
                features[f"{action}_opp_AU45_norm"],
                FEATURE_CONFIG['min_value']
            )
            
            features[f"{action}_AU07_ratio"] = min(
                features[f"{action}_AU07_norm"], 
                features[f"{action}_opp_AU07_norm"]
            ) / max(
                features[f"{action}_AU07_norm"], 
                features[f"{action}_opp_AU07_norm"],
                FEATURE_CONFIG['min_value']
            )
            
            # Calculate percent difference
            features[f"{action}_AU45_percent_diff"] = calculate_single_percent_diff(
                features[f"{action}_AU45_norm"], 
                features[f"{action}_opp_AU45_norm"]
            )
            
            features[f"{action}_AU07_percent_diff"] = calculate_single_percent_diff(
                features[f"{action}_AU07_norm"], 
                features[f"{action}_opp_AU07_norm"]
            )
            
            # Directional asymmetry
            features[f"{action}_AU45_is_weaker"] = 1 if features[f"{action}_AU45_norm"] < features[f"{action}_opp_AU45_norm"] else 0
            features[f"{action}_AU07_is_weaker"] = 1 if features[f"{action}_AU07_norm"] < features[f"{action}_opp_AU07_norm"] else 0
            
            # Combined features
            features[f"{action}_AU45_AU07_interaction"] = features[f"{action}_AU45_norm"] * features[f"{action}_AU07_norm"]
            features[f"{action}_ratio_product"] = features[f"{action}_AU45_ratio"] * features[f"{action}_AU07_ratio"]
            features[f"{action}_asymmetry_differential"] = abs(
                features[f"{action}_AU45_percent_diff"] - features[f"{action}_AU07_percent_diff"]
            )
            
            # Weighted asymmetry
            features[f"{action}_weighted_asymmetry"] = (
                features[f"{action}_AU45_percent_diff"] * FEATURE_CONFIG['au_weights']['AU45_r'] +
                features[f"{action}_AU07_percent_diff"] * FEATURE_CONFIG['au_weights']['AU07_r']
            )
    
    # Add cross-action features if we have multiple actions
    if all(f"ES_{key}" in features and f"ET_{key}" in features 
           for key in ["AU45_norm", "AU07_norm"]):
        
        # Calculate ET/ES ratios
        et_au45 = max(features["ET_AU45_norm"], FEATURE_CONFIG['min_value'])
        es_au45 = max(features["ES_AU45_norm"], FEATURE_CONFIG['min_value'])
        et_au07 = max(features["ET_AU07_norm"], FEATURE_CONFIG['min_value'])
        es_au07 = max(features["ES_AU07_norm"], FEATURE_CONFIG['min_value'])
        
        features["ES_ET_AU45_ratio"] = es_au45 / et_au45
        features["ES_ET_AU07_ratio"] = es_au07 / et_au07
        features["ET_ES_AU45_diff"] = abs(et_au45 - es_au45)
        features["ET_ES_AU07_diff"] = abs(et_au07 - es_au07)
        
        # Weighted ratio
        features["weighted_ET_ES_ratio"] = (
            features["ES_ET_AU45_ratio"] * FEATURE_CONFIG['au_weights']['AU45_r'] +
            features["ES_ET_AU07_ratio"] * FEATURE_CONFIG['au_weights']['AU07_r']
        )
    
    # Add functional features for ET action
    if "ET_AU45_norm" in features:
        features["ET_functional_AU45"] = features["ET_AU45_norm"]
        if "ET_AU45_ratio" in features:
            features["ET_functional_ratio"] = features["ET_AU45_ratio"]
    
    # Add max/min features
    au45_norm_values = [features.get(f"{action}_AU45_norm", 0) for action in actions]
    au07_norm_values = [features.get(f"{action}_AU07_norm", 0) for action in actions]
    
    features["max_AU45_norm"] = max(au45_norm_values)
    features["min_AU45_norm"] = min(au45_norm_values)
    features["range_AU45_norm"] = features["max_AU45_norm"] - features["min_AU45_norm"]
    
    features["max_AU07_norm"] = max(au07_norm_values)
    features["min_AU07_norm"] = min(au07_norm_values)
    features["range_AU07_norm"] = features["max_AU07_norm"] - features["min_AU07_norm"]
    
    # Add max asymmetry
    au45_percent_diffs = [features.get(f"{action}_AU45_percent_diff", 0) for action in actions 
                          if f"{action}_AU45_percent_diff" in features]
    au07_percent_diffs = [features.get(f"{action}_AU07_percent_diff", 0) for action in actions 
                          if f"{action}_AU07_percent_diff" in features]
    
    if au45_percent_diffs:
        features["max_AU45_percent_diff"] = max(au45_percent_diffs)
    if au07_percent_diffs:
        features["max_AU07_percent_diff"] = max(au07_percent_diffs)
    
    # Side indicator
    features["side"] = 1 if side.lower() == 'right' else 0
    
    # Create feature vector (order must match the training data)
    # Return all features as a list - the ML model will use the ones it needs
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
