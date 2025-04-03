"""
Feature extraction for ML-based synkinesis detection (optimized version).
Processes data from combined_results.csv and expert labels for training.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

logger = logging.getLogger(__name__)

def prepare_synkinesis_data():
    """
    Prepare dataset for ML model training by merging detection results with expert labels.

    Returns:
        tuple: Dictionary of (features DataFrame, targets array) for each synkinesis type
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
        'Ocular-Oral Synkinesis Left': 'Expert_Ocular_Oral_Left',
        'Ocular-Oral Synkinesis Right': 'Expert_Ocular_Oral_Right',
        'Oral-Ocular Synkinesis Left': 'Expert_Oral_Ocular_Left',
        'Oral-Ocular Synkinesis Right': 'Expert_Oral_Ocular_Right',
        'Snarl Smile Left': 'Expert_Snarl_Smile_Left',
        'Snarl Smile Right': 'Expert_Snarl_Smile_Right'
    })

    # Merge datasets on Patient ID
    merged_df = pd.merge(results_df, expert_df, on='Patient ID', how='inner')
    logger.info(f"Merged dataset contains {len(merged_df)} patients")

    # Prepare datasets for each synkinesis type
    synkinesis_datasets = {}
    
    # ===== Ocular-Oral Synkinesis =====
    # Extract features for left and right sides
    ocular_oral_left_features = extract_ocular_oral_features(merged_df, 'Left')
    ocular_oral_right_features = extract_ocular_oral_features(merged_df, 'Right')
    
    # Process targets
    ocular_oral_left_targets = process_targets(merged_df['Expert_Ocular_Oral_Left'])
    ocular_oral_right_targets = process_targets(merged_df['Expert_Ocular_Oral_Right'])
    
    # Create side indicator as separate Series (not adding to DataFrame)
    left_side = pd.Series(0, index=ocular_oral_left_features.index)  # 0 for left
    right_side = pd.Series(1, index=ocular_oral_right_features.index)  # 1 for right
    
    # Combine features and side indicator using concat
    ocular_oral_features = pd.concat([
        ocular_oral_left_features, 
        ocular_oral_right_features
    ], ignore_index=True)
    
    # Add side column after DataFrame is created
    side_combined = pd.concat([left_side, right_side], ignore_index=True)
    ocular_oral_features['side'] = side_combined
    
    # Combine targets
    ocular_oral_targets = np.concatenate([ocular_oral_left_targets, ocular_oral_right_targets])
    
    synkinesis_datasets['ocular_oral'] = (ocular_oral_features.fillna(0), ocular_oral_targets)
    
    # ===== Oral-Ocular Synkinesis =====
    # Extract features for left and right sides
    oral_ocular_left_features = extract_oral_ocular_features(merged_df, 'Left')
    oral_ocular_right_features = extract_oral_ocular_features(merged_df, 'Right')
    
    # Process targets
    oral_ocular_left_targets = process_targets(merged_df['Expert_Oral_Ocular_Left'])
    oral_ocular_right_targets = process_targets(merged_df['Expert_Oral_Ocular_Right'])
    
    # Create side indicator as separate Series
    left_side = pd.Series(0, index=oral_ocular_left_features.index)
    right_side = pd.Series(1, index=oral_ocular_right_features.index)
    
    # Combine features and side indicator using concat
    oral_ocular_features = pd.concat([
        oral_ocular_left_features, 
        oral_ocular_right_features
    ], ignore_index=True)
    
    # Add side column after DataFrame is created
    side_combined = pd.concat([left_side, right_side], ignore_index=True)
    oral_ocular_features['side'] = side_combined
    
    # Combine targets
    oral_ocular_targets = np.concatenate([oral_ocular_left_targets, oral_ocular_right_targets])
    
    synkinesis_datasets['oral_ocular'] = (oral_ocular_features.fillna(0), oral_ocular_targets)
    
    # ===== Snarl-Smile Synkinesis =====
    # Extract features for left and right sides
    snarl_smile_left_features = extract_snarl_smile_features(merged_df, 'Left')
    snarl_smile_right_features = extract_snarl_smile_features(merged_df, 'Right')
    
    # Process targets
    snarl_smile_left_targets = process_targets(merged_df['Expert_Snarl_Smile_Left'])
    snarl_smile_right_targets = process_targets(merged_df['Expert_Snarl_Smile_Right'])
    
    # Create side indicator as separate Series
    left_side = pd.Series(0, index=snarl_smile_left_features.index)
    right_side = pd.Series(1, index=snarl_smile_right_features.index)
    
    # Combine features and side indicator using concat
    snarl_smile_features = pd.concat([
        snarl_smile_left_features, 
        snarl_smile_right_features
    ], ignore_index=True)
    
    # Add side column after DataFrame is created
    side_combined = pd.concat([left_side, right_side], ignore_index=True)
    snarl_smile_features['side'] = side_combined
    
    # Combine targets
    snarl_smile_targets = np.concatenate([snarl_smile_left_targets, snarl_smile_right_targets])
    
    synkinesis_datasets['snarl_smile'] = (snarl_smile_features.fillna(0), snarl_smile_targets)

    return synkinesis_datasets

def extract_ocular_oral_features(df, side):
    """
    Extract features for Ocular-Oral synkinesis detection (optimized version).
    
    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'
        
    Returns:
        pandas.DataFrame: Features for Ocular-Oral synkinesis detection
    """
    # Define feature dictionary to collect all features before creating DataFrame
    feature_dict = {}
    
    # Relevant actions for ocular-oral synkinesis (eye movements)
    actions = ['ET', 'ES', 'RE', 'BL']
    
    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'
    
    # Define the AUs we're interested in
    trigger_aus = ['AU01_r', 'AU02_r', 'AU45_r']
    coupled_aus = ['AU12_r', 'AU25_r', 'AU14_r']
    
    # First, collect all the basic AU values from the dataframe
    basic_features = {}
    
    for action in actions:
        # Extract features for trigger AUs
        for au in trigger_aus:
            col_name = f"{action}_{side} {au}"
            norm_col_name = f"{action}_{side} {au} (Normalized)"
            
            if norm_col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[norm_col_name].values
            elif col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[col_name].values
            else:
                basic_features[f"{action}_{au}_norm"] = np.zeros(len(df))
        
        # Extract features for coupled AUs
        for au in coupled_aus:
            col_name = f"{action}_{side} {au}"
            norm_col_name = f"{action}_{side} {au} (Normalized)"
            
            if norm_col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[norm_col_name].values
            elif col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[col_name].values
            else:
                basic_features[f"{action}_{au}_norm"] = np.zeros(len(df))
    
    # Add all basic features to the feature dictionary
    feature_dict.update(basic_features)
    
    # Now calculate derived features
    derived_features = {}
    
    for action in actions:
        # Calculate interaction features
        for trigger_au in trigger_aus:
            for coupled_au in coupled_aus:
                trigger_val = basic_features.get(f"{action}_{trigger_au}_norm", np.zeros(len(df)))
                coupled_val = basic_features.get(f"{action}_{coupled_au}_norm", np.zeros(len(df)))
                
                # Calculate ratio and product in vectorized operations
                ratio = calculate_ratio_vectorized(trigger_val, coupled_val)
                product = trigger_val * coupled_val
                
                derived_features[f"{action}_{trigger_au}_{coupled_au}_ratio"] = ratio
                derived_features[f"{action}_{trigger_au}_{coupled_au}_product"] = product
        
        # Calculate summary features
        trigger_vals = np.column_stack([basic_features.get(f"{action}_{au}_norm", np.zeros(len(df))) 
                                        for au in trigger_aus])
        coupled_vals = np.column_stack([basic_features.get(f"{action}_{au}_norm", np.zeros(len(df))) 
                                       for au in coupled_aus])
        
        derived_features[f"{action}_trigger_avg"] = np.mean(trigger_vals, axis=1)
        derived_features[f"{action}_trigger_max"] = np.max(trigger_vals, axis=1)
        derived_features[f"{action}_coupled_avg"] = np.mean(coupled_vals, axis=1)
        derived_features[f"{action}_coupled_max"] = np.max(coupled_vals, axis=1)
        
        # Ratio of trigger to coupled activation
        trigger_avg = derived_features[f"{action}_trigger_avg"]
        coupled_avg = derived_features[f"{action}_coupled_avg"]
        derived_features[f"{action}_trigger_coupled_ratio"] = calculate_ratio_vectorized(
            trigger_avg, coupled_avg)
    
    # Add all derived features to the feature dictionary
    feature_dict.update(derived_features)
    
    # Add current detection result as feature
    result_col = f"Ocular-Oral {side}"
    if result_col in df.columns:
        mapping = {'None': 0, 'Partial': 1, 'Complete': 2, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
        feature_dict['current_detection'] = df[result_col].map(mapping, na_action='ignore').fillna(0).values
    else:
        feature_dict['current_detection'] = np.zeros(len(df))
    
    # Create DataFrame from the feature dictionary (all at once, not incrementally)
    features = pd.DataFrame(feature_dict)
    
    return features

def extract_oral_ocular_features(df, side):
    """
    Extract features for Oral-Ocular synkinesis detection (optimized version).
    
    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'
        
    Returns:
        pandas.DataFrame: Features for Oral-Ocular synkinesis detection
    """
    # Define feature dictionary to collect all features before creating DataFrame
    feature_dict = {}
    
    # Relevant actions for oral-ocular synkinesis (mouth movements)
    actions = ['BS', 'SS', 'SO', 'SE']
    
    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'
    
    # Define the AUs we're interested in
    trigger_aus = ['AU12_r', 'AU25_r']
    coupled_aus = ['AU45_r', 'AU06_r']
    
    # First, collect all the basic AU values from the dataframe
    basic_features = {}
    
    for action in actions:
        # Extract features for trigger AUs
        for au in trigger_aus:
            col_name = f"{action}_{side} {au}"
            norm_col_name = f"{action}_{side} {au} (Normalized)"
            
            if norm_col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[norm_col_name].values
            elif col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[col_name].values
            else:
                basic_features[f"{action}_{au}_norm"] = np.zeros(len(df))
        
        # Extract features for coupled AUs
        for au in coupled_aus:
            col_name = f"{action}_{side} {au}"
            norm_col_name = f"{action}_{side} {au} (Normalized)"
            
            if norm_col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[norm_col_name].values
            elif col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[col_name].values
            else:
                basic_features[f"{action}_{au}_norm"] = np.zeros(len(df))
    
    # Add all basic features to the feature dictionary
    feature_dict.update(basic_features)
    
    # Now calculate derived features
    derived_features = {}
    
    for action in actions:
        # Calculate interaction features
        for trigger_au in trigger_aus:
            for coupled_au in coupled_aus:
                trigger_val = basic_features.get(f"{action}_{trigger_au}_norm", np.zeros(len(df)))
                coupled_val = basic_features.get(f"{action}_{coupled_au}_norm", np.zeros(len(df)))
                
                # Calculate ratio and product in vectorized operations
                ratio = calculate_ratio_vectorized(trigger_val, coupled_val)
                product = trigger_val * coupled_val
                
                derived_features[f"{action}_{trigger_au}_{coupled_au}_ratio"] = ratio
                derived_features[f"{action}_{trigger_au}_{coupled_au}_product"] = product
        
        # Calculate summary features
        trigger_vals = np.column_stack([basic_features.get(f"{action}_{au}_norm", np.zeros(len(df))) 
                                        for au in trigger_aus])
        coupled_vals = np.column_stack([basic_features.get(f"{action}_{au}_norm", np.zeros(len(df))) 
                                       for au in coupled_aus])
        
        derived_features[f"{action}_trigger_avg"] = np.mean(trigger_vals, axis=1)
        derived_features[f"{action}_trigger_max"] = np.max(trigger_vals, axis=1)
        derived_features[f"{action}_coupled_avg"] = np.mean(coupled_vals, axis=1)
        derived_features[f"{action}_coupled_max"] = np.max(coupled_vals, axis=1)
        
        # Ratio of trigger to coupled activation
        trigger_avg = derived_features[f"{action}_trigger_avg"]
        coupled_avg = derived_features[f"{action}_coupled_avg"]
        derived_features[f"{action}_trigger_coupled_ratio"] = calculate_ratio_vectorized(
            trigger_avg, coupled_avg)
    
    # Add all derived features to the feature dictionary
    feature_dict.update(derived_features)
    
    # Add current detection result as feature
    result_col = f"Oral-Ocular {side}"
    if result_col in df.columns:
        mapping = {'None': 0, 'Partial': 1, 'Complete': 2, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
        feature_dict['current_detection'] = df[result_col].map(mapping, na_action='ignore').fillna(0).values
    else:
        feature_dict['current_detection'] = np.zeros(len(df))
    
    # Create DataFrame from the feature dictionary (all at once, not incrementally)
    features = pd.DataFrame(feature_dict)
    
    return features

def extract_snarl_smile_features(df, side):
    """
    Extract features for Snarl-Smile synkinesis detection (optimized version).
    
    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'
        
    Returns:
        pandas.DataFrame: Features for Snarl-Smile synkinesis detection
    """
    # Define feature dictionary to collect all features before creating DataFrame
    feature_dict = {}
    
    # Relevant actions for snarl-smile synkinesis (smile actions)
    actions = ['BS', 'SS']
    
    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'
    
    # Define the AUs we're interested in
    trigger_au = 'AU12_r'
    coupled_aus = ['AU09_r', 'AU10_r', 'AU14_r']
    
    # First, collect all the basic AU values from the dataframe
    basic_features = {}
    
    for action in actions:
        # Extract feature for trigger AU
        col_name = f"{action}_{side} {trigger_au}"
        norm_col_name = f"{action}_{side} {trigger_au} (Normalized)"
        
        if norm_col_name in df.columns:
            basic_features[f"{action}_{trigger_au}_norm"] = df[norm_col_name].values
        elif col_name in df.columns:
            basic_features[f"{action}_{trigger_au}_norm"] = df[col_name].values
        else:
            basic_features[f"{action}_{trigger_au}_norm"] = np.zeros(len(df))
    
        # Extract features for coupled AUs
        for au in coupled_aus:
            col_name = f"{action}_{side} {au}"
            norm_col_name = f"{action}_{side} {au} (Normalized)"
            
            if norm_col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[norm_col_name].values
            elif col_name in df.columns:
                basic_features[f"{action}_{au}_norm"] = df[col_name].values
            else:
                basic_features[f"{action}_{au}_norm"] = np.zeros(len(df))
    
    # Add all basic features to the feature dictionary
    feature_dict.update(basic_features)
    
    # Now calculate derived features
    derived_features = {}
    
    for action in actions:
        # Calculate interaction features
        for coupled_au in coupled_aus:
            trigger_val = basic_features.get(f"{action}_{trigger_au}_norm", np.zeros(len(df)))
            coupled_val = basic_features.get(f"{action}_{coupled_au}_norm", np.zeros(len(df)))
            
            # Calculate ratio and product in vectorized operations
            ratio = calculate_ratio_vectorized(trigger_val, coupled_val)
            product = trigger_val * coupled_val
            
            derived_features[f"{action}_{trigger_au}_{coupled_au}_ratio"] = ratio
            derived_features[f"{action}_{trigger_au}_{coupled_au}_product"] = product
        
        # Calculate summary features
        coupled_vals = np.column_stack([basic_features.get(f"{action}_{au}_norm", np.zeros(len(df))) 
                                       for au in coupled_aus])
        
        derived_features[f"{action}_coupled_avg"] = np.mean(coupled_vals, axis=1)
        derived_features[f"{action}_coupled_max"] = np.max(coupled_vals, axis=1)
        
        # Ratio of trigger to coupled activation
        trigger_val = basic_features.get(f"{action}_{trigger_au}_norm", np.zeros(len(df)))
        coupled_avg = derived_features[f"{action}_coupled_avg"]
        derived_features[f"{action}_trigger_coupled_ratio"] = calculate_ratio_vectorized(
            trigger_val, coupled_avg)
        
        # Calculate weighted score
        au09_weight = 0.4  # Nose wrinkle importance
        au10_weight = 0.3  # Upper lip raiser importance
        au14_weight = 0.3  # Dimpler importance
        
        au09_val = basic_features.get(f"{action}_AU09_r_norm", np.zeros(len(df)))
        au10_val = basic_features.get(f"{action}_AU10_r_norm", np.zeros(len(df)))
        au14_val = basic_features.get(f"{action}_AU14_r_norm", np.zeros(len(df)))
        
        weighted_score = (au09_val * au09_weight + 
                          au10_val * au10_weight + 
                          au14_val * au14_weight)
        
        derived_features[f"{action}_weighted_score"] = weighted_score
    
    # Add all derived features to the feature dictionary
    feature_dict.update(derived_features)
    
    # Count active components based on thresholds
    thresholds = {'AU09_r': 0.8, 'AU10_r': 0.8, 'AU14_r': 1.2}
    
    # Find action with maximum smile value
    max_smile_action = None
    max_smile_value = -1
    
    for action in actions:
        smile_vals = basic_features.get(f"{action}_AU12_r_norm", np.zeros(len(df)))
        curr_max = np.max(smile_vals)
        if curr_max > max_smile_value:
            max_smile_action = action
            max_smile_value = curr_max
    
    if max_smile_action:
        # Calculate active component count
        active_components = np.zeros(len(df))
        for au in coupled_aus:
            au_vals = basic_features.get(f"{max_smile_action}_{au}_norm", np.zeros(len(df)))
            threshold = thresholds.get(au, 0.8)
            active_components += (au_vals > threshold).astype(int)
        
        feature_dict['active_component_count'] = active_components
    else:
        feature_dict['active_component_count'] = np.zeros(len(df))
    
    # Add current detection result as feature
    result_col = f"Snarl-Smile {side}"
    if result_col in df.columns:
        mapping = {'None': 0, 'Partial': 1, 'Complete': 2, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
        feature_dict['current_detection'] = df[result_col].map(mapping, na_action='ignore').fillna(0).values
    else:
        feature_dict['current_detection'] = np.zeros(len(df))
    
    # Create DataFrame from the feature dictionary (all at once, not incrementally)
    features = pd.DataFrame(feature_dict)
    
    return features

def calculate_ratio_vectorized(val1, val2):
    """
    Calculate ratio between NumPy arrays using vectorized operations.
    
    Args:
        val1 (numpy.ndarray): First value array
        val2 (numpy.ndarray): Second value array
        
    Returns:
        numpy.ndarray: Ratio values
    """
    # Handle zero values
    val1_safe = np.copy(val1)
    val2_safe = np.copy(val2)
    
    # Replace zeros with small value to avoid division by zero
    val1_safe[val1_safe == 0] = 0.0001
    val2_safe[val2_safe == 0] = 0.0001
    
    # Calculate min and max values
    min_vals = np.minimum(val1_safe, val2_safe)
    max_vals = np.maximum(val1_safe, val2_safe)
    
    # Calculate ratio
    ratio = min_vals / max_vals
    
    # Cap at 1.0 for floating point issues
    ratio[ratio > 1.0] = 1.0
    
    return ratio

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
        'None': 0, 'no': 0, 'No': 0, 'N/A': 0, '': 0, 'normal': 0, 'Normal': 0, 'False': 0, 'false': 0,
        'Partial': 1, 'partial': 1, 'mild': 1, 'Mild': 1, 'moderate': 1, 'Moderate': 1, 'Yes': 1, 'yes': 1, 'True': 1, 'true': 1,
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
    class_counts = pd.Series(targets).value_counts()
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