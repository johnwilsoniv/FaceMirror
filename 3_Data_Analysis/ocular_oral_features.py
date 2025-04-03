# ocular_oral_features.py (Mirrors oral_ocular_features.py)
# - Capitalization fix for side/opposite_side
# - Corrected helper functions (calculate_ratio, calculate_percent_diff)
# - Added ET-specific summary features

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Import config for Ocular-Oral
try:
    from ocular_oral_config import (
        FEATURE_CONFIG, OCULAR_ORAL_ACTIONS, TRIGGER_AUS, COUPLED_AUS, LOG_DIR,
        FEATURE_SELECTION, MODEL_FILENAMES, CLASS_NAMES
    )
except ImportError:
    logging.warning("Could not import from ocular_oral_config. Using fallback definitions.")
    # Add minimal fallbacks for standalone execution if needed
    LOG_DIR = 'logs'; MODEL_DIR = 'models/synkinesis/ocular_oral' # Added MODEL_DIR
    MODEL_FILENAMES = {'feature_list': os.path.join(MODEL_DIR, 'features.list'),
                       'importance_file': os.path.join(MODEL_DIR, 'feature_importance.csv')}
    OCULAR_ORAL_ACTIONS = ['ET', 'ES', 'RE', 'BL']; TRIGGER_AUS = ['AU01_r', 'AU02_r', 'AU45_r']; COUPLED_AUS = ['AU12_r', 'AU25_r', 'AU14_r']
    FEATURE_CONFIG = {'actions': OCULAR_ORAL_ACTIONS, 'trigger_aus': TRIGGER_AUS, 'coupled_aus': COUPLED_AUS,
                      'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0} # Updated defaults
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 40, 'importance_file': MODEL_FILENAMES['importance_file']}
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
# Ensure logger is configured
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- prepare_data function ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data for Ocular-Oral synkinesis training, including feature selection. """
    logger.info("Loading datasets for Ocular-Oral Synkinesis...")
    try:
        results_df = pd.read_csv(results_file, low_memory=False)
        expert_df = pd.read_csv(expert_file)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise

    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        # Rename specific expert columns for Ocular-Oral
        'Ocular-Oral Synkinesis Left': 'Expert_Left_Ocular_Oral',
        'Ocular-Oral Synkinesis Right': 'Expert_Right_Ocular_Oral'})

    # Target Variable Processing (Use the generic binary mapper)
    if 'Expert_Left_Ocular_Oral' in expert_df.columns:
        expert_df['Target_Left_Ocular_Oral'] = process_targets(expert_df['Expert_Left_Ocular_Oral'])
    else:
        logger.error("Missing 'Expert_Left_Ocular_Oral' column")
        expert_df['Target_Left_Ocular_Oral'] = 0 # Default to 0 if missing
    if 'Expert_Right_Ocular_Oral' in expert_df.columns:
        expert_df['Target_Right_Ocular_Oral'] = process_targets(expert_df['Expert_Right_Ocular_Oral'])
    else:
        logger.error("Missing 'Expert_Right_Ocular_Oral' column")
        expert_df['Target_Right_Ocular_Oral'] = 0 # Default to 0 if missing

    logger.info(f"Counts in expert_df['Target_Left_Ocular_Oral'] AFTER mapping: \n{expert_df['Target_Left_Ocular_Oral'].value_counts(dropna=False)}")
    logger.info(f"Counts in expert_df['Target_Right_Ocular_Oral'] AFTER mapping: \n{expert_df['Target_Right_Ocular_Oral'].value_counts(dropna=False)}")

    # Prepare for Merge
    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
    expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()

    expert_cols_to_merge = ['Patient ID', 'Target_Left_Ocular_Oral', 'Target_Right_Ocular_Oral']
    try:
        merged_df = pd.merge(results_df, expert_df[expert_cols_to_merge], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data for Ocular-Oral: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    logger.info(f"Counts in merged_df['Target_Left_Ocular_Oral'] AFTER merge: \n{merged_df['Target_Left_Ocular_Oral'].value_counts(dropna=False)}")
    logger.info(f"Counts in merged_df['Target_Right_Ocular_Oral'] AFTER merge: \n{merged_df['Target_Right_Ocular_Oral'].value_counts(dropna=False)}")

    # Feature Extraction
    logger.info("Extracting Ocular-Oral features for Left side...")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info("Extracting Ocular-Oral features for Right side...")
    right_features_df = extract_features(merged_df, 'Right')

    # Combine features and targets
    if 'Target_Left_Ocular_Oral' not in merged_df.columns or 'Target_Right_Ocular_Oral' not in merged_df.columns:
         logger.error("Target columns missing in merged_df before creating targets array. Aborting.")
         return None, None
    left_targets = merged_df['Target_Left_Ocular_Oral'].values
    right_targets = merged_df['Target_Right_Ocular_Oral'].values
    targets = np.concatenate([left_targets, right_targets])

    # Log FINAL distribution
    unique_final, counts_final = np.unique(targets, return_counts=True)
    final_class_dist = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in unique_final], counts_final))
    logger.info(f"FINAL Ocular-Oral Class distribution input: {final_class_dist}")

    left_features_df['side_indicator'] = 0
    right_features_df['side_indicator'] = 1
    features = pd.concat([left_features_df, right_features_df], ignore_index=True)

    # Post-processing & Feature Selection (Identical logic to oral_ocular)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.fillna(0)
    initial_cols = features.columns.tolist()
    cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert column {col} to numeric: {e}. Marking for drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)

    logger.info(f"Generated initial {features.shape[1]} Ocular-Oral features.")

    # Apply Feature Selection (Checks config flag)
    fs_enabled = isinstance(FEATURE_SELECTION, dict) and FEATURE_SELECTION.get('enabled', False)
    if fs_enabled:
        logger.info("Applying Ocular-Oral feature selection...")
        n_top_features = FEATURE_SELECTION.get('top_n_features', 40) # Use config value
        importance_file = FEATURE_SELECTION.get('importance_file')
        if not importance_file or not os.path.exists(importance_file): logger.warning(f"FS enabled, but importance file not found: '{importance_file}'. Skipping.")
        else:
            try:
                importance_df = pd.read_csv(importance_file)
                if 'feature' not in importance_df.columns or importance_df.empty: logger.error("Importance file lacks 'feature' column or is empty. Skipping selection.")
                else:
                    top_feature_names = importance_df['feature'].head(n_top_features).tolist()
                    # Ensure side_indicator is always kept if it exists
                    if 'side_indicator' in features.columns and 'side_indicator' not in top_feature_names:
                        top_feature_names.append('side_indicator')

                    original_cols = features.columns.tolist()
                    cols_to_keep = [col for col in top_feature_names if col in original_cols]
                    missing_features = set(top_feature_names) - set(cols_to_keep)
                    if missing_features: logger.warning(f"Some important Ocular-Oral features missing from generated data: {missing_features}")
                    if not cols_to_keep: logger.error("No features left after filtering. Skipping selection.")
                    else:
                        logger.info(f"Selecting top {len(cols_to_keep)} Ocular-Oral features.")
                        features = features[cols_to_keep] # Select columns

            except Exception as e: logger.error(f"Error during Ocular-Oral feature selection: {e}. Skipping.", exc_info=True)
    else:
        logger.info("Ocular-Oral feature selection is disabled.") # Will be disabled on first run

    logger.info(f"Final Ocular-Oral dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.warning(f"NaNs found in FINAL Ocular-Oral features BEFORE saving list. Columns: {features.columns[features.isna().any()].tolist()}"); features = features.fillna(0)

    # Save final feature list
    final_feature_names = features.columns.tolist()
    try:
        feature_list_path = MODEL_FILENAMES.get('feature_list')
        if feature_list_path:
             os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
             joblib.dump(final_feature_names, feature_list_path)
             logger.info(f"Saved final {len(final_feature_names)} Ocular-Oral feature names list to {feature_list_path}")
        else: logger.error("Ocular-Oral feature list path not defined in config. Cannot save feature list.")
    except Exception as e: logger.error(f"Failed to save Ocular-Oral feature names list: {e}", exc_info=True)

    if 'targets' not in locals(): logger.error("Targets array was not created."); return None, None
    return features, targets


# --- Helper Functions (Copied from lower_face_features.py) ---
# --- CORRECTED calculate_ratio ---
def calculate_ratio(val1_series, val2_series):
    """
    Calculates the ratio min(val1, val2) / max(val1, val2) safely using Pandas/Numpy.
    Handles NaN and zero values, returning 1.0 if max value is near zero
    and 0.0 if min is zero but max is positive.
    """
    local_logger = logging.getLogger(__name__)
    try:
        min_val_config = FEATURE_CONFIG.get('min_value', 0.0001) if isinstance(FEATURE_CONFIG, dict) else 0.0001
    except NameError:
        min_val_config = 0.0001
        local_logger.warning("calculate_ratio: FEATURE_CONFIG not found, using default min_value.")

    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()

    min_vals_np = np.minimum(v1, v2)
    max_vals_np = np.maximum(v1, v2)

    ratio = np.ones_like(v1, dtype=float)
    mask_max_pos = max_vals_np > min_val_config
    mask_min_zero = min_vals_np <= min_val_config

    ratio[mask_max_pos & mask_min_zero] = 0.0
    valid_division_mask = mask_max_pos & ~mask_min_zero
    if np.any(valid_division_mask):
        ratio[valid_division_mask] = min_vals_np[valid_division_mask] / max_vals_np[valid_division_mask]

    if np.isnan(ratio).any() or np.isinf(ratio).any():
        local_logger.warning("NaN or Inf detected in calculate_ratio output. Handling.")
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

    return pd.Series(ratio, index=val1_series.index)
# --- END calculate_ratio ---

# --- CORRECTED calculate_percent_diff ---
def calculate_percent_diff(val1_series, val2_series):
    """
    Calculates the percentage difference: (abs(v1-v2) / avg(v1,v2)) * 100 using Pandas/Numpy.
    Handles zero average and caps the result.
    """
    local_logger = logging.getLogger(__name__)
    try:
        min_val_config = FEATURE_CONFIG.get('min_value', 0.0001) if isinstance(FEATURE_CONFIG, dict) else 0.0001
        percent_diff_cap = FEATURE_CONFIG.get('percent_diff_cap', 200.0) if isinstance(FEATURE_CONFIG, dict) else 200.0
    except NameError:
        min_val_config = 0.0001
        percent_diff_cap = 200.0
        local_logger.warning("calculate_percent_diff: FEATURE_CONFIG not found, using default values.")

    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()

    abs_diff = np.abs(v1 - v2)
    avg = (v1 + v2) / 2.0

    percent_diff = np.zeros_like(avg, dtype=float)

    mask_avg_pos = avg > min_val_config
    if np.any(mask_avg_pos):
        with np.errstate(divide='ignore', invalid='ignore'):
            division_result = abs_diff[mask_avg_pos] / avg[mask_avg_pos]
        percent_diff[mask_avg_pos] = division_result * 100.0

    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > min_val_config)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap

    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    percent_diff = np.nan_to_num(percent_diff, nan=0.0, posinf=percent_diff_cap, neginf=0.0)

    return pd.Series(percent_diff, index=val1_series.index)
# --- END CORRECTED calculate_percent_diff ---


# --- extract_features function (Training) ---
def extract_features(df, side):
    """ Extracts Ocular-Oral features for TRAINING using the dictionary method. """
    logger.debug(f"Extracting Ocular-Oral features for {side} side (Training)...")
    feature_data = {}
    # --- Capitalization Fix ---
    side_label = side.capitalize()
    opposite_side_label = 'Right' if side_label == 'Left' else 'Left'
    # --- End Capitalization Fix ---

    local_feature_config = FEATURE_CONFIG if isinstance(FEATURE_CONFIG, dict) else {}
    local_actions = OCULAR_ORAL_ACTIONS
    local_trigger_aus = TRIGGER_AUS
    local_coupled_aus = COUPLED_AUS
    use_normalized = local_feature_config.get('use_normalized', True)
    norm_suffix = " (Normalized)" if use_normalized else ""

    # 1. Basic AU & Interaction Features per Action
    all_action_features = {}
    et_coupled_vals_norm = [] # To store Series for ET coupled AUs
    et_trigger_vals_norm = [] # To store Series for ET trigger AUs

    for action in local_actions:
        action_features = {}
        current_action_coupled_vals = []
        current_action_trigger_vals = []

        # Get Trigger AU values
        for trig_au in local_trigger_aus:
            # --- Use Capitalized Labels ---
            col_raw = f"{action}_{side_label} {trig_au}"
            col_norm = f"{col_raw}{norm_suffix}"
            # --- End Use Capitalized Labels ---

            raw_val_series = df.get(col_raw, pd.Series(0.0, index=df.index))
            raw_val_side = pd.to_numeric(raw_val_series, errors='coerce').fillna(0.0)

            if use_normalized:
                norm_val_series = df.get(col_norm, raw_val_side) # Default to raw if norm missing
                norm_val_side = pd.to_numeric(norm_val_series, errors='coerce').fillna(raw_val_side)
            else: norm_val_side = raw_val_side

            action_features[f"{action}_{trig_au}_trig_norm"] = norm_val_side # Store the value used for calcs
            current_action_trigger_vals.append(norm_val_side) # Add to list for this action
            if action == 'ET': et_trigger_vals_norm.append(norm_val_side) # Add to ET specific list

        # Get Coupled AU values and calculate ratios
        for coup_au in local_coupled_aus:
             # --- Use Capitalized Labels ---
            col_raw = f"{action}_{side_label} {coup_au}"
            col_norm = f"{col_raw}{norm_suffix}"
            # --- End Use Capitalized Labels ---

            raw_val_series = df.get(col_raw, pd.Series(0.0, index=df.index))
            raw_val_side = pd.to_numeric(raw_val_series, errors='coerce').fillna(0.0)

            if use_normalized:
                 norm_val_series = df.get(col_norm, raw_val_side) # Default to raw if norm missing
                 norm_val_side = pd.to_numeric(norm_val_series, errors='coerce').fillna(raw_val_side)
            else: norm_val_side = raw_val_side

            coup_series = norm_val_side # Use the appropriate value
            action_features[f"{action}_{coup_au}_coup_norm"] = coup_series
            current_action_coupled_vals.append(coup_series) # Add to list for this action
            if action == 'ET': et_coupled_vals_norm.append(coup_series) # Add to ET specific list

            # Calculate ratios against trigger AUs for THIS action
            for trig_au in local_trigger_aus:
                # Retrieve the already calculated/stored trigger series for this action
                trig_series = action_features.get(f"{action}_{trig_au}_trig_norm", pd.Series(0.0, index=df.index))
                action_features[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = calculate_ratio(coup_series, trig_series)

        all_action_features.update(action_features)

    feature_data.update(all_action_features)

    # --- START: Add New ET-Specific Features (Training) ---
    if et_coupled_vals_norm:
        et_coupled_df = pd.concat(et_coupled_vals_norm, axis=1)
        feature_data['ET_Avg_Coupled_Norm'] = et_coupled_df.mean(axis=1)
        feature_data['ET_Max_Coupled_Norm'] = et_coupled_df.max(axis=1)
    else: # Default if no ET coupled AUs found
        feature_data['ET_Avg_Coupled_Norm'] = pd.Series(0.0, index=df.index)
        feature_data['ET_Max_Coupled_Norm'] = pd.Series(0.0, index=df.index)

    if et_trigger_vals_norm:
        et_trigger_df = pd.concat(et_trigger_vals_norm, axis=1)
        feature_data['ET_Avg_Trigger_Norm'] = et_trigger_df.mean(axis=1)
    else: # Default if no ET trigger AUs found
        feature_data['ET_Avg_Trigger_Norm'] = pd.Series(0.0, index=df.index)

    # Calculate ET-specific ratio using the newly calculated averages
    # Retrieve the average series we just computed
    et_avg_coup_series = feature_data.get('ET_Avg_Coupled_Norm', pd.Series(0.0, index=df.index))
    et_avg_trig_series = feature_data.get('ET_Avg_Trigger_Norm', pd.Series(0.0, index=df.index))
    feature_data['ET_Ratio_AvgCoup_vs_AvgTrig'] = calculate_ratio(et_avg_coup_series, et_avg_trig_series)
    # --- END: Add New ET-Specific Features (Training) ---

    # 2. Summary Features Across Actions (Identical logic, just uses the feature_data dict)
    summary_features = {}
    for coup_au in local_coupled_aus:
        coup_cols = [f"{action}_{coup_au}_coup_norm" for action in local_actions if f"{action}_{coup_au}_coup_norm" in feature_data]
        if coup_cols:
            coup_df = pd.concat([feature_data[col] for col in coup_cols], axis=1)
            summary_features[f"Avg_{coup_au}_AcrossActions"] = coup_df.mean(axis=1)
            summary_features[f"Max_{coup_au}_AcrossActions"] = coup_df.max(axis=1)
            summary_features[f"Std_{coup_au}_AcrossActions"] = coup_df.std(axis=1).fillna(0)

    for trig_au in local_trigger_aus:
         trig_cols = [f"{action}_{trig_au}_trig_norm" for action in local_actions if f"{action}_{trig_au}_trig_norm" in feature_data]
         if trig_cols:
             trig_df = pd.concat([feature_data[col] for col in trig_cols], axis=1)
             summary_features[f"Avg_{trig_au}_AcrossActions"] = trig_df.mean(axis=1)
             summary_features[f"Max_{trig_au}_AcrossActions"] = trig_df.max(axis=1)

    avg_coup_vals = []
    avg_trig_vals = []
    for coup_au in local_coupled_aus: avg_coup_vals.append(summary_features.get(f"Avg_{coup_au}_AcrossActions", pd.Series(0.0, index=df.index)))
    for trig_au in local_trigger_aus: avg_trig_vals.append(summary_features.get(f"Avg_{trig_au}_AcrossActions", pd.Series(0.0, index=df.index)))

    if avg_coup_vals and avg_trig_vals:
         overall_avg_coup = pd.concat(avg_coup_vals, axis=1).mean(axis=1)
         overall_avg_trig = pd.concat(avg_trig_vals, axis=1).mean(axis=1)
         summary_features["Ratio_AvgCoup_vs_AvgTrig"] = calculate_ratio(overall_avg_coup, overall_avg_trig)

    feature_data.update(summary_features)

    # Create DataFrame
    features_df = pd.DataFrame(feature_data, index=df.index)

    # Final check for non-numeric types
    non_numeric_cols = features_df.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric cols in Ocular-Oral extract_features: {non_numeric_cols.tolist()}. Coercing.")
        for col in non_numeric_cols: features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)

    logger.debug(f"Generated {features_df.shape[1]} Ocular-Oral features for {side} (Training).")
    return features_df


# --- extract_features_for_detection (Detection) ---
def extract_features_for_detection(row_data, side):
    """
    Extract Ocular-Oral features for detection from a row of data (pd.Series or dict).
    Relies on the saved feature list for final ordering and selection.

    Args:
        row_data (pd.Series or dict): A row from the merged dataframe containing all AU columns.
        side (str): Side ('Left' or 'Right').

    Returns:
        list: Feature vector for model input or None on error.
    """
    try:
        # Load necessary items from config dynamically inside the function
        from ocular_oral_config import FEATURE_CONFIG, OCULAR_ORAL_ACTIONS, TRIGGER_AUS, COUPLED_AUS, MODEL_FILENAMES
        local_logger = logging.getLogger(__name__)
    except ImportError:
        logging.error("Failed to import config within ocular_oral extract_features_for_detection.")
        # Provide minimal fallbacks if config is missing during detection
        FEATURE_CONFIG = {'use_normalized': True, 'min_value': 0.0001}
        OCULAR_ORAL_ACTIONS = ['ET', 'ES', 'RE', 'BL']
        TRIGGER_AUS = ['AU01_r', 'AU02_r', 'AU45_r']
        COUPLED_AUS = ['AU12_r', 'AU25_r', 'AU14_r']
        MODEL_FILENAMES = {'feature_list': 'models/synkinesis/ocular_oral/features.list'} # Essential fallback
        local_logger = logging # Use root logger if specific one fails

    if not isinstance(row_data, (pd.Series, dict)):
        local_logger.error(f"row_data must be a pd.Series or dict, got {type(row_data)}")
        return None

    # Convert dict to Series if necessary
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data

    # --- Capitalization Fix ---
    side_label = side.capitalize()
    # opposite_side_label = 'Right' if side_label == 'Left' else 'Left' # Not needed for scalar calcs here
    # --- End Capitalization Fix ---

    local_feature_config = FEATURE_CONFIG if isinstance(FEATURE_CONFIG, dict) else {}
    local_actions = OCULAR_ORAL_ACTIONS
    local_trigger_aus = TRIGGER_AUS
    local_coupled_aus = COUPLED_AUS
    use_normalized = local_feature_config.get('use_normalized', True)
    norm_suffix = " (Normalized)" if use_normalized else ""
    min_val_ratio = local_feature_config.get('min_value', 0.0001) # Use same min_value for consistency

    local_logger.debug(f"Extracting Ocular-Oral detection features for {side_label}...")
    feature_dict_final = {}

    # 1. Basic AU & Interaction Features per Action (Scalar version)
    all_action_features = {}
    et_coupled_vals_norm_scalar = [] # List to hold scalar values for ET
    et_trigger_vals_norm_scalar = [] # List to hold scalar values for ET

    for action in local_actions:
        action_features = {}
        current_action_coupled_vals_scalar = []
        current_action_trigger_vals_scalar = []

        # Get Trigger AU values
        for trig_au in local_trigger_aus:
            # --- Use Capitalized Labels ---
            col_raw = f"{action}_{side_label} {trig_au}"
            col_norm = f"{col_raw}{norm_suffix}"
            # --- End Use Capitalized Labels ---

            raw_val = 0.0 # Default
            try: raw_val = float(row_series.get(col_raw, 0.0)) # Get raw, default 0.0
            except (ValueError, TypeError): pass # Keep default 0.0 on conversion error

            norm_val = raw_val # Default to raw
            if use_normalized:
                 try: norm_val = float(row_series.get(col_norm, raw_val)) # Try getting norm, default to raw_val
                 except (ValueError, TypeError): pass # Keep raw_val if norm conversion fails

            trig_val_used = norm_val # This is the value used for features
            action_features[f"{action}_{trig_au}_trig_norm"] = trig_val_used
            current_action_trigger_vals_scalar.append(trig_val_used) # Add to list for this action
            if action == 'ET': et_trigger_vals_norm_scalar.append(trig_val_used) # Add to ET specific list

        # Get Coupled AU values and calculate ratios
        for coup_au in local_coupled_aus:
            # --- Use Capitalized Labels ---
            col_raw = f"{action}_{side_label} {coup_au}"
            col_norm = f"{col_raw}{norm_suffix}"
            # --- End Use Capitalized Labels ---

            raw_val = 0.0 # Default
            try: raw_val = float(row_series.get(col_raw, 0.0))
            except (ValueError, TypeError): pass

            norm_val = raw_val # Default to raw
            if use_normalized:
                 try: norm_val = float(row_series.get(col_norm, raw_val))
                 except (ValueError, TypeError): pass

            coup_val = norm_val # This is the value used for features
            action_features[f"{action}_{coup_au}_coup_norm"] = coup_val
            current_action_coupled_vals_scalar.append(coup_val) # Add to list for this action
            if action == 'ET': et_coupled_vals_norm_scalar.append(coup_val) # Add to ET specific list

            # Calculate ratios against trigger AUs for THIS action
            for trig_au in local_trigger_aus:
                trig_val = action_features.get(f"{action}_{trig_au}_trig_norm", 0.0) # Get stored trig value

                # --- Use Corrected Ratio Logic (Scalar) ---
                min_v = min(coup_val, trig_val)
                max_v = max(coup_val, trig_val)
                ratio = 1.0 # Default
                if max_v > min_val_ratio: # Check denominator
                    if min_v <= min_val_ratio: ratio = 0.0 # Min is zero, max is positive
                    else: ratio = min_v / max_v # Both positive
                # else: max_v is near zero, keep ratio = 1.0 (includes 0/0 case)
                # --- End Corrected Ratio Logic ---
                action_features[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = ratio

        all_action_features.update(action_features)

    feature_dict_final.update(all_action_features)

    # --- START: Add New ET-Specific Features (Detection - Scalar) ---
    et_avg_coup = np.mean(et_coupled_vals_norm_scalar) if et_coupled_vals_norm_scalar else 0.0
    et_max_coup = np.max(et_coupled_vals_norm_scalar) if et_coupled_vals_norm_scalar else 0.0
    et_avg_trig = np.mean(et_trigger_vals_norm_scalar) if et_trigger_vals_norm_scalar else 0.0

    feature_dict_final['ET_Avg_Coupled_Norm'] = et_avg_coup
    feature_dict_final['ET_Max_Coupled_Norm'] = et_max_coup
    feature_dict_final['ET_Avg_Trigger_Norm'] = et_avg_trig

    # Calculate ET-specific ratio using the averages
    et_min_v = min(et_avg_coup, et_avg_trig)
    et_max_v = max(et_avg_coup, et_avg_trig)
    et_ratio_avg = 1.0
    if et_max_v > min_val_ratio:
        if et_min_v <= min_val_ratio: et_ratio_avg = 0.0
        else: et_ratio_avg = et_min_v / et_max_v
    feature_dict_final['ET_Ratio_AvgCoup_vs_AvgTrig'] = et_ratio_avg
    # --- END: Add New ET-Specific Features (Detection - Scalar) ---


    # 2. Summary Features Across Actions (Scalar version)
    summary_features = {}
    for coup_au in local_coupled_aus:
        coup_vals = [feature_dict_final.get(f"{action}_{coup_au}_coup_norm", 0.0) for action in local_actions]
        summary_features[f"Avg_{coup_au}_AcrossActions"] = np.mean(coup_vals) if coup_vals else 0.0
        summary_features[f"Max_{coup_au}_AcrossActions"] = np.max(coup_vals) if coup_vals else 0.0
        summary_features[f"Std_{coup_au}_AcrossActions"] = np.std(coup_vals) if coup_vals else 0.0

    for trig_au in local_trigger_aus:
        trig_vals = [feature_dict_final.get(f"{action}_{trig_au}_trig_norm", 0.0) for action in local_actions]
        summary_features[f"Avg_{trig_au}_AcrossActions"] = np.mean(trig_vals) if trig_vals else 0.0
        summary_features[f"Max_{trig_au}_AcrossActions"] = np.max(trig_vals) if trig_vals else 0.0

    avg_coup_vals_list = [summary_features.get(f"Avg_{coup_au}_AcrossActions", 0.0) for coup_au in local_coupled_aus]
    avg_trig_vals_list = [summary_features.get(f"Avg_{trig_au}_AcrossActions", 0.0) for trig_au in local_trigger_aus]
    overall_avg_coup_val = np.mean(avg_coup_vals_list) if avg_coup_vals_list else 0.0
    overall_avg_trig_val = np.mean(avg_trig_vals_list) if avg_trig_vals_list else 0.0

    # Calculate ratio for overall averages
    overall_min_v = min(overall_avg_coup_val, overall_avg_trig_val)
    overall_max_v = max(overall_avg_coup_val, overall_avg_trig_val)
    summary_ratio = 1.0
    if overall_max_v > min_val_ratio:
        if overall_min_v <= min_val_ratio: summary_ratio = 0.0
        else: summary_ratio = overall_min_v / overall_max_v
    summary_features["Ratio_AvgCoup_vs_AvgTrig"] = summary_ratio
    feature_dict_final.update(summary_features)

    # Add side indicator
    feature_dict_final["side_indicator"] = 0 if side.lower() == 'left' else 1

    # --- Load the EXPECTED feature list ---
    feature_names_path = MODEL_FILENAMES.get('feature_list')
    if not feature_names_path or not os.path.exists(feature_names_path):
        local_logger.error(f"Ocular-Oral feature list not found: {feature_names_path}.")
        return None
    try:
        ordered_feature_names = joblib.load(feature_names_path)
        if not isinstance(ordered_feature_names, list):
             local_logger.error(f"Loaded Ocular-Oral feature names is not a list: {type(ordered_feature_names)}")
             return None
    except Exception as e:
        local_logger.error(f"Failed to load Ocular-Oral feature list: {e}", exc_info=True); return None

    # --- Build final feature list IN ORDER ---
    feature_list = []
    missing_in_dict = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name) # Get raw value or None
        final_val = 0.0 # Default

        if value is None:
            missing_in_dict.append(name)
        else:
            try:
                final_val = float(value)
                if np.isnan(final_val): final_val = 0.0 # Handle NaN after conversion
            except (ValueError, TypeError):
                # Log error if conversion fails for existing value
                local_logger.warning(f"Could not convert feature '{name}' value '{value}' to float. Defaulting to 0.0.")
                final_val = 0.0

        feature_list.append(final_val)

    if missing_in_dict: local_logger.warning(f"Ocular-Oral Detection: {len(missing_in_dict)} expected features missing from generated dict: {missing_in_dict[:5]}... Defaulting to 0.")
    if len(feature_list) != len(ordered_feature_names):
        local_logger.error(f"CRITICAL MISMATCH: Ocular-Oral detection list length ({len(feature_list)}) != expected ({len(ordered_feature_names)}).")
        return None

    local_logger.debug(f"Generated {len(feature_list)} Ocular-Oral detection features for {side_label}.")
    return feature_list


# --- process_targets function (Identical binary mapping) ---
def process_targets(target_series):
    """ Converts expert labels (Yes/No etc.) to binary 0/1 """
    if target_series is None: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    s_filled = target_series.fillna('no')
    s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected expert labels found (treated as 'No'): {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int).values