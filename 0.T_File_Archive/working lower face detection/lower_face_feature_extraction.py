"""
Feature extraction for ML-based lower face paralysis detection.
Processes data from combined_results.csv and expert labels for training.
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

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
    logger.info(f"Class distribution: {class_dist}")

    # Handle class imbalance if severe (optional - commented out by default)
    if False:  # Set to True to enable oversampling
        features, targets = balance_classes(features, targets)

    logger.info(f"Final dataset: {len(features)} samples with {features.shape[1]} features")

    return features, targets

def extract_features(df, side):
    """
    Extract relevant features for lower face paralysis detection.

    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'

    Returns:
        pandas.DataFrame: Features for the specified side
    """
    features = pd.DataFrame()

    # Actions to extract features from - focusing on actions that involve lower face
    actions = ['BS', 'SS', 'SO', 'SE']

    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'

    for action in actions:
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

        # 1. Basic features - raw AU values
        if au12_col in df.columns:
            features[f"{action}_AU12_raw"] = df[au12_col]
        else:
            features[f"{action}_AU12_raw"] = 0

        if au25_col in df.columns:
            features[f"{action}_AU25_raw"] = df[au25_col]
        else:
            features[f"{action}_AU25_raw"] = 0

        # 2. Normalized AU values
        if au12_norm_col in df.columns:
            features[f"{action}_AU12_norm"] = df[au12_norm_col]
        else:
            features[f"{action}_AU12_norm"] = features[f"{action}_AU12_raw"]

        if au25_norm_col in df.columns:
            features[f"{action}_AU25_norm"] = df[au25_norm_col]
        else:
            features[f"{action}_AU25_norm"] = features[f"{action}_AU25_raw"]

        # 3. Asymmetry metrics
        # Ratio metrics
        if au12_col in df.columns and au12_opp_col in df.columns:
            features[f"{action}_AU12_ratio"] = calculate_ratio(df[au12_col], df[au12_opp_col])
        else:
            features[f"{action}_AU12_ratio"] = 1.0

        if au25_col in df.columns and au25_opp_col in df.columns:
            features[f"{action}_AU25_ratio"] = calculate_ratio(df[au25_col], df[au25_opp_col])
        else:
            features[f"{action}_AU25_ratio"] = 1.0

        # Percent difference metrics
        if au12_col in df.columns and au12_opp_col in df.columns:
            features[f"{action}_AU12_percent_diff"] = calculate_percent_diff(df[au12_col], df[au12_opp_col])
        else:
            features[f"{action}_AU12_percent_diff"] = 0.0

        if au25_col in df.columns and au25_opp_col in df.columns:
            features[f"{action}_AU25_percent_diff"] = calculate_percent_diff(df[au25_col], df[au25_opp_col])
        else:
            features[f"{action}_AU25_percent_diff"] = 0.0

        # 4. Normalized asymmetry metrics
        if au12_norm_col in df.columns and au12_opp_norm_col in df.columns:
            features[f"{action}_AU12_norm_ratio"] = calculate_ratio(df[au12_norm_col], df[au12_opp_norm_col])
            features[f"{action}_AU12_norm_percent_diff"] = calculate_percent_diff(df[au12_norm_col], df[au12_opp_norm_col])
        else:
            features[f"{action}_AU12_norm_ratio"] = features[f"{action}_AU12_ratio"]
            features[f"{action}_AU12_norm_percent_diff"] = features[f"{action}_AU12_percent_diff"]

        if au25_norm_col in df.columns and au25_opp_norm_col in df.columns:
            features[f"{action}_AU25_norm_ratio"] = calculate_ratio(df[au25_norm_col], df[au25_opp_norm_col])
            features[f"{action}_AU25_norm_percent_diff"] = calculate_percent_diff(df[au25_norm_col], df[au25_opp_norm_col])
        else:
            features[f"{action}_AU25_norm_ratio"] = features[f"{action}_AU25_ratio"]
            features[f"{action}_AU25_norm_percent_diff"] = features[f"{action}_AU25_percent_diff"]

        # 5. Directional asymmetry (which side is weaker)
        if au12_col in df.columns and au12_opp_col in df.columns:
            features[f"{action}_AU12_is_weaker"] = (df[au12_col] < df[au12_opp_col]).astype(int)
        else:
            features[f"{action}_AU12_is_weaker"] = 0

        if au25_col in df.columns and au25_opp_col in df.columns:
            features[f"{action}_AU25_is_weaker"] = (df[au25_col] < df[au25_opp_col]).astype(int)
        else:
            features[f"{action}_AU25_is_weaker"] = 0

        # 6. Combined score feature
        if all(col in df.columns for col in [au12_norm_col, au25_norm_col, au12_opp_norm_col, au25_opp_norm_col]):
            features[f"{action}_combined_score"] = calculate_combined_score(
                df[au12_norm_col], df[au25_norm_col],
                df[au12_opp_norm_col], df[au25_opp_norm_col]
            )
        else:
            # If any column is missing, estimate combined score from available metrics
            available_ratios = []
            if f"{action}_AU12_norm_ratio" in features:
                available_ratios.append(features[f"{action}_AU12_norm_ratio"] * 0.6)  # Weight matches original
            if f"{action}_AU25_norm_ratio" in features:
                available_ratios.append(features[f"{action}_AU25_norm_ratio"] * 0.4)  # Weight matches original

            if available_ratios:
                # Sum available weighted ratios
                combined_ratio = sum(available_ratios)
                # Estimate combined min value as 0.5 if not available
                combined_min = 0.5
                features[f"{action}_combined_score"] = combined_ratio * (1 + combined_min / 5)
            else:
                # Default if no ratios available
                features[f"{action}_combined_score"] = 1.0

        # 7. Enhanced features - AU interactions
        if au12_col in df.columns and au25_col in df.columns:
            # Interaction between AU12 and AU25 values
            features[f"{action}_AU12_AU25_interaction"] = df[au12_col] * df[au25_col]
        else:
            features[f"{action}_AU12_AU25_interaction"] = 0.0

        # Asymmetry interaction (ratio product)
        features[f"{action}_ratio_product"] = features[f"{action}_AU12_ratio"] * features[f"{action}_AU25_ratio"]

        # Asymmetry differential (difference between AU12 and AU25 asymmetry)
        features[f"{action}_asymmetry_differential"] = abs(
            features[f"{action}_AU12_percent_diff"] - features[f"{action}_AU25_percent_diff"]
        )

    # 8. Add current detection result as a feature
    lower_face_col = f"{side} Lower Face Paralysis"
    if lower_face_col in df.columns:
        features['current_detection'] = df[lower_face_col].map({'None': 0, 'Partial': 1, 'Complete': 2, 'none': 0, 'partial': 1, 'complete': 2})
    else:
        features['current_detection'] = 0

    # 9. Cross-action features
    for au in ['AU12', 'AU25']:
        # Calculate max, min and range across actions
        max_columns = [f"{action}_{au}_raw" for action in actions if f"{action}_{au}_raw" in features.columns]
        if max_columns:
            features[f"max_{au}"] = features[max_columns].max(axis=1)
            features[f"min_{au}"] = features[max_columns].min(axis=1)
            features[f"range_{au}"] = features[f"max_{au}"] - features[f"min_{au}"]

        # Calculate max asymmetry across actions
        asymmetry_columns = [f"{action}_{au}_percent_diff" for action in actions if f"{action}_{au}_percent_diff" in features.columns]
        if asymmetry_columns:
            features[f"max_{au}_asymmetry"] = features[asymmetry_columns].max(axis=1)

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
    # Create copies to avoid warnings about setting values on a slice
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

def calculate_combined_score(au12_val, au25_val, au12_opp, au25_opp):
    """
    Calculate a combined score similar to the existing algorithm.

    Args:
        au12_val (pandas.Series): AU12 values for current side
        au25_val (pandas.Series): AU25 values for current side
        au12_opp (pandas.Series): AU12 values for opposite side
        au25_opp (pandas.Series): AU25 values for opposite side

    Returns:
        pandas.Series: Combined score values
    """
    # Weights for AUs - matching existing code
    au12_weight = 0.6
    au25_weight = 0.4

    # Calculate ratios
    au12_ratio = calculate_ratio(au12_val, au12_opp)
    au25_ratio = calculate_ratio(au25_val, au25_opp)

    # Calculate combined ratio
    combined_ratio = (au12_ratio * au12_weight) + (au25_ratio * au25_weight)

    # Get minimum values (same side)
    au12_min = pd.Series([min(a, b) for a, b in zip(au12_val, au12_opp)])
    au25_min = pd.Series([min(a, b) for a, b in zip(au25_val, au25_opp)])

    # Calculate combined minimum (use the value from the side we're calculating for)
    side_au12_min = pd.Series([a if a < b else a for a, b in zip(au12_val, au12_opp)])
    side_au25_min = pd.Series([a if a < b else a for a, b in zip(au25_val, au25_opp)])

    combined_min = (side_au12_min * au12_weight) + (side_au25_min * au25_weight)

    # Prevent division by zero or negative values
    combined_min = combined_min.clip(lower=0)

    # Final combined score
    return combined_ratio * (1 + combined_min / 5)

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

def balance_classes(features, targets):
    """
    Balance classes using resampling techniques to address class imbalance.

    Args:
        features (pandas.DataFrame): Feature data
        targets (numpy.ndarray): Target labels

    Returns:
        tuple: (balanced_features, balanced_targets)
    """
    # Combine features and targets for resampling
    data = features.copy()
    data['target'] = targets

    # Count samples in each class
    class_counts = data['target'].value_counts()
    logger.info(f"Original class counts: {class_counts.to_dict()}")

    # Get unique class labels
    classes = np.unique(targets)

    # Find the majority class count
    majority_class_count = class_counts.max()

    # Upsample minority classes
    upsampled_dfs = []
    for cls in classes:
        cls_df = data[data['target'] == cls]

        if len(cls_df) < majority_class_count:
            # Upsample to match majority class
            upsampled_cls = resample(
                cls_df,
                replace=True,
                n_samples=majority_class_count,
                random_state=42
            )
            upsampled_dfs.append(upsampled_cls)
        else:
            # Keep majority class as is
            upsampled_dfs.append(cls_df)

    # Combine upsampled dataframes
    balanced_data = pd.concat(upsampled_dfs)

    # Extract features and targets
    balanced_targets = balanced_data['target'].values
    balanced_features = balanced_data.drop('target', axis=1)

    # Check new class distribution
    unique, counts = np.unique(balanced_targets, return_counts=True)
    logger.info(f"Balanced class counts: {dict(zip(['None', 'Partial', 'Complete'], counts))}")

    return balanced_features, balanced_targets