# lower_face_features.py (Ratio Fix + PercDiff Fix + Weaker Side Fix + Debug v7 + ...)

import numpy as np
import pandas as pd
import logging
import os
import joblib
import json # Added for potential future full dict logging if needed
# Import sys for debugging exit (can be commented out later)
# import sys

# Import config carefully, providing fallbacks
try:
    from lower_face_config import (
        FEATURE_CONFIG, LOWER_FACE_ACTIONS, LOWER_FACE_AUS, LOG_DIR,
        FEATURE_SELECTION, MODEL_DIR, CLASS_NAMES
    )
except ImportError:
    logging.warning("Could not import from lower_face_config. Using fallback definitions.")
    LOG_DIR = 'logs'; MODEL_DIR = 'models'
    FEATURE_CONFIG = {'actions': ['BS', 'SS', 'SO', 'SE'], 'aus': ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'], 'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
    LOWER_FACE_ACTIONS = FEATURE_CONFIG['actions']
    LOWER_FACE_AUS = FEATURE_CONFIG['aus']
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 50, 'importance_file': os.path.join(MODEL_DIR, 'lower_face_feature_importance.csv')}
    CLASS_NAMES = {0: 'None', 1: 'Partial', 2: 'Complete'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
# Make sure the logger is configured, either here or in main
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- prepare_data function (No changes needed here) ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data including feature selection. """
    logger.info("Loading datasets...")
    try:
        logger.info(f"Attempting to load results from: {results_file}")
        results_df = pd.read_csv(results_file, low_memory=False)
        logger.info(f"Attempting to load expert key from: {expert_file}")
        expert_df = pd.read_csv(expert_file, dtype=str) # Load expert key as string
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise

    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        'Paralysis - Left Lower Face': 'Expert_Left_Lower_Face',
        'Paralysis - Right Lower Face': 'Expert_Right_Lower_Face'})

    # Target Variable Processing
    target_mapping = {'none': 0, 'partial': 1, 'complete': 2}
    def standardize_and_map(series):
        s_filled = series.fillna('none_placeholder')
        s_clean = s_filled.astype(str).str.lower().str.strip()
        replacements = {'no': 'none', 'n/a': 'none', 'mild': 'partial', 'moderate': 'partial', 'severe': 'complete', 'normal': 'none', 'none_placeholder': 'none', 'nan': 'none'}
        s_replaced = s_clean.replace(replacements)
        mapped = s_replaced.map(target_mapping)
        final_mapped = mapped.fillna(0)
        return final_mapped.astype(int)

    if 'Expert_Left_Lower_Face' in expert_df.columns: expert_df['Target_Left_Lower'] = standardize_and_map(expert_df['Expert_Left_Lower_Face'])
    else: logger.error("Missing 'Expert_Left_Lower_Face' column"); expert_df['Target_Left_Lower'] = 0
    if 'Expert_Right_Lower_Face' in expert_df.columns: expert_df['Target_Right_Lower'] = standardize_and_map(expert_df['Expert_Right_Lower_Face'])
    else: logger.error("Missing 'Expert_Right_Lower_Face' column"); expert_df['Target_Right_Lower'] = 0

    logger.info(f"Counts in expert_df['Target_Left_Lower'] AFTER mapping: \n{expert_df['Target_Left_Lower'].value_counts(dropna=False)}")
    logger.info(f"Counts in expert_df['Target_Right_Lower'] AFTER mapping: \n{expert_df['Target_Right_Lower'].value_counts(dropna=False)}")

    # Prepare for Merge
    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
    expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()

    expert_cols_to_merge = ['Patient ID', 'Target_Left_Lower', 'Target_Right_Lower']
    try:
        if not all(col in expert_df.columns for col in expert_cols_to_merge): raise KeyError(f"Required target columns missing from expert_df: {expert_cols_to_merge}")
        merged_df = pd.merge(results_df, expert_df[expert_cols_to_merge], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    logger.info(f"Counts in merged_df['Target_Left_Lower'] AFTER merge: \n{merged_df['Target_Left_Lower'].value_counts(dropna=False)}")
    logger.info(f"Counts in merged_df['Target_Right_Lower'] AFTER merge: \n{merged_df['Target_Right_Lower'].value_counts(dropna=False)}")

    # Feature Extraction
    logger.info("Extracting features for Left side...")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info("Extracting features for Right side...")
    right_features_df = extract_features(merged_df, 'Right')

    # --- DEBUG LOGGING for first patient features ---
    if not left_features_df.empty and not merged_df.empty:
        if left_features_df.index.equals(merged_df.index):
             first_patient_id = merged_df.iloc[0]['Patient ID']
             if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"DEBUG (prepare_data): Features for Patient {first_patient_id} Left (BEFORE side_indicator/selection):\n{left_features_df.iloc[0].tolist()}")
             else:
                 logger.info(f"INFO_DEBUG (prepare_data): Features for Patient {first_patient_id} Left sample:\n{left_features_df.iloc[0].head().to_dict()}")
        else: logger.warning("DEBUG (prepare_data): Index mismatch. Cannot log first patient features.")
    # --- END DEBUG LOGGING ---

    # Combine features and targets
    if 'Target_Left_Lower' not in merged_df.columns or 'Target_Right_Lower' not in merged_df.columns:
         logger.error("Target columns missing in merged_df before creating targets array. Aborting.")
         return None, None
    left_targets = merged_df['Target_Left_Lower'].values
    right_targets = merged_df['Target_Right_Lower'].values
    targets = np.concatenate([left_targets, right_targets])

    unique_final, counts_final = np.unique(targets, return_counts=True)
    final_class_dist = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in unique_final], counts_final))
    logger.info(f"FINAL Class distribution input to SMOTE/Split: {final_class_dist}")

    left_features_df['side_indicator'] = 0
    right_features_df['side_indicator'] = 1
    features = pd.concat([left_features_df, right_features_df], ignore_index=True)

    # Post-processing & Feature Selection
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.fillna(0)
    initial_cols = features.columns.tolist()
    cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert column {col} to numeric: {e}. Marking for drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)

    logger.info(f"Generated initial {features.shape[1]} features.")

    fs_enabled = isinstance(FEATURE_SELECTION, dict) and FEATURE_SELECTION.get('enabled', False)
    if fs_enabled:
        logger.info("Applying feature selection...")
        n_top_features = FEATURE_SELECTION.get('top_n_features', 50)
        importance_file = FEATURE_SELECTION.get('importance_file')
        if not importance_file or not os.path.exists(importance_file): logger.warning(f"FS enabled, but importance file not found: '{importance_file}'. Skipping.")
        else:
            try:
                importance_df = pd.read_csv(importance_file)
                if 'feature' not in importance_df.columns or importance_df.empty: logger.error("Importance file lacks 'feature' column or is empty. Skipping selection.")
                else:
                    top_feature_names = importance_df['feature'].head(n_top_features).tolist()
                    if 'side_indicator' in features.columns and 'side_indicator' not in top_feature_names: top_feature_names.append('side_indicator')
                    original_cols = features.columns.tolist()
                    cols_to_keep = [col for col in top_feature_names if col in original_cols]
                    missing_features = set(top_feature_names) - set(cols_to_keep)
                    if missing_features: logger.warning(f"Some important features missing from generated data: {missing_features}")
                    if not cols_to_keep: logger.error("No features left after filtering. Skipping selection.")
                    else: logger.info(f"Selecting top {len(cols_to_keep)} features."); features = features[cols_to_keep]
            except Exception as e: logger.error(f"Error during feature selection: {e}. Skipping.", exc_info=True)
    else:
        logger.info("Lower face feature selection is disabled.") # Updated log message

    logger.info(f"Final dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.warning(f"NaNs found in FINAL features BEFORE saving list. Columns: {features.columns[features.isna().any()].tolist()}"); features = features.fillna(0)

    final_feature_names = features.columns.tolist()
    try:
        if MODEL_DIR:
             os.makedirs(MODEL_DIR, exist_ok=True)
             feature_list_path = os.path.join(MODEL_DIR, 'lower_face_features.list') # Use correct list name
             joblib.dump(final_feature_names, feature_list_path)
             logger.info(f"Saved final {len(final_feature_names)} lower face feature names list to {feature_list_path}")
        else: logger.error("MODEL_DIR not defined. Cannot save feature list.")
    except Exception as e: logger.error(f"Failed to save feature names list: {e}", exc_info=True)

    if 'targets' not in locals(): logger.error("Targets array was not created."); return None, None
    return features, targets


# --- Helper Functions (Pandas/Numpy based for Training) ---
# --- CORRECTED calculate_ratio (from previous step) ---
def calculate_ratio(val1_series, val2_series):
    """
    Calculates the ratio min(val1, val2) / max(val1, val2) safely.
    Handles NaN and zero values, returning 1.0 if max value is near zero
    and 0.0 if min is zero but max is positive.
    """
    local_logger = logging.getLogger(__name__) # Use logger defined at module level
    try: # Get config safely
        min_val_config = FEATURE_CONFIG.get('min_value', 0.0001) if isinstance(FEATURE_CONFIG, dict) else 0.0001
    except NameError: # Fallback if FEATURE_CONFIG not imported
        min_val_config = 0.0001
        local_logger.warning("calculate_ratio: FEATURE_CONFIG not found, using default min_value.")

    # Ensure numeric, fill NaNs with 0.0, convert to numpy
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()

    # Calculate min and max
    min_vals_np = np.minimum(v1, v2)
    max_vals_np = np.maximum(v1, v2)

    # Initialize ratio: Default to 1.0 (perfect symmetry, covers 0/0 case)
    ratio = np.ones_like(v1, dtype=float)

    # Identify where max value is significantly positive
    mask_max_pos = max_vals_np > min_val_config

    # Identify where min value is effectively zero
    mask_min_zero = min_vals_np <= min_val_config

    # Scenario 1: Max is positive, but Min is zero -> Ratio should be 0.0
    ratio[mask_max_pos & mask_min_zero] = 0.0

    # Scenario 2: Max is positive AND Min is positive -> Calculate Ratio
    valid_division_mask = mask_max_pos & ~mask_min_zero
    if np.any(valid_division_mask):
        # Perform division safely only for these elements
        ratio[valid_division_mask] = min_vals_np[valid_division_mask] / max_vals_np[valid_division_mask]

    # Handle potential NaNs/Infs just in case (e.g., if min_val_config was 0)
    if np.isnan(ratio).any() or np.isinf(ratio).any():
        local_logger.warning("NaN or Inf detected in calculate_ratio output. Check inputs/logic.")
        # Fallback: replace NaN/Inf with 1.0 (treat as symmetric)
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

    # Return as a Pandas Series
    return pd.Series(ratio, index=val1_series.index)
# --- END calculate_ratio ---

# --- CORRECTED calculate_percent_diff ---
def calculate_percent_diff(val1_series, val2_series):
    """
    Calculates the percentage difference: (abs(v1-v2) / avg(v1,v2)) * 100.
    Handles zero average and caps the result.
    """
    local_logger = logging.getLogger(__name__) # Use logger defined at module level
    try: # Get config safely
        min_val_config = FEATURE_CONFIG.get('min_value', 0.0001) if isinstance(FEATURE_CONFIG, dict) else 0.0001
        percent_diff_cap = FEATURE_CONFIG.get('percent_diff_cap', 200.0) if isinstance(FEATURE_CONFIG, dict) else 200.0
    except NameError:
        min_val_config = 0.0001
        percent_diff_cap = 200.0
        local_logger.warning("calculate_percent_diff: FEATURE_CONFIG not found, using default values.")

    # Ensure numeric, fill NaNs with 0.0, convert to numpy
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()

    abs_diff = np.abs(v1 - v2)
    avg = (v1 + v2) / 2.0

    # Initialize result array
    percent_diff = np.zeros_like(avg, dtype=float)

    # --- Scenario 1: Average is significantly positive ---
    mask_avg_pos = avg > min_val_config
    if np.any(mask_avg_pos):
        # Calculate division result only where avg is positive
        # Use np.errstate to handle potential 0/0 if min_val_config is 0, though unlikely
        with np.errstate(divide='ignore', invalid='ignore'):
            division_result = abs_diff[mask_avg_pos] / avg[mask_avg_pos]
        # Place the result * 100 into the final array
        percent_diff[mask_avg_pos] = division_result * 100.0

    # --- Scenario 2: Average is near zero, but difference is significant ---
    # Use min_val_config for difference check as well for consistency
    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > min_val_config)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap

    # --- Final Steps: Clipping and NaN/Inf handling ---
    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    # Replace any NaNs/Infs which might arise from 0/0 if min_val_config was 0
    percent_diff = np.nan_to_num(percent_diff, nan=0.0, posinf=percent_diff_cap, neginf=0.0)

    return pd.Series(percent_diff, index=val1_series.index)
# --- END CORRECTED calculate_percent_diff ---


# --- extract_features function (Training) ---
def extract_features(df, side):
    """ Extracts features for TRAINING using the dictionary method. """
    logger.debug(f"Extracting lower face features for {side} side (Training)...")
    feature_data = {}
    opposite_side = 'Right' if side == 'Left' else 'Left'
    try: # Get config safely
        local_feature_config = FEATURE_CONFIG if isinstance(FEATURE_CONFIG, dict) else {}
        local_actions = LOWER_FACE_ACTIONS if 'LOWER_FACE_ACTIONS' in globals() and LOWER_FACE_ACTIONS else ['BS', 'SS', 'SO', 'SE']
        local_aus = LOWER_FACE_AUS if 'LOWER_FACE_AUS' in globals() and LOWER_FACE_AUS else ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']
        use_normalized = local_feature_config.get('use_normalized', True)
    except NameError:
        logger.error("extract_features: Config variables not found, using defaults.")
        local_feature_config = {'use_normalized': True}
        local_actions = ['BS', 'SS', 'SO', 'SE']
        local_aus = ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']
        use_normalized = True


    # 1. Basic AU & Asymmetry Features
    for action in local_actions:
        for au in local_aus:
            base_col_name = f"{action}_{au}"
            au_col_side = f"{action}_{side} {au}"
            au_norm_col_side = f"{au_col_side} (Normalized)"
            au_col_opp = f"{action}_{opposite_side} {au}"
            au_norm_col_opp = f"{au_col_opp} (Normalized)"

            raw_val_side_series = df.get(au_col_side, pd.Series(0.0, index=df.index))
            raw_val_opp_series = df.get(au_col_opp, pd.Series(0.0, index=df.index))
            raw_val_side = pd.to_numeric(raw_val_side_series, errors='coerce').fillna(0.0)
            raw_val_opp = pd.to_numeric(raw_val_opp_series, errors='coerce').fillna(0.0)

            if use_normalized:
                # Use get with default=raw_val_side/opp if normalized key is missing
                norm_val_side_series = df.get(au_norm_col_side, raw_val_side)
                norm_val_opp_series = df.get(au_norm_col_opp, raw_val_opp)
                # Ensure numeric, fill NaN with corresponding raw value
                norm_val_side = pd.to_numeric(norm_val_side_series, errors='coerce').fillna(raw_val_side)
                norm_val_opp = pd.to_numeric(norm_val_opp_series, errors='coerce').fillna(raw_val_opp)
            else: norm_val_side = raw_val_side; norm_val_opp = raw_val_opp

            feature_data[f"{base_col_name}_raw_side"] = raw_val_side
            feature_data[f"{base_col_name}_raw_opp"] = raw_val_opp
            feature_data[f"{base_col_name}_norm_side"] = norm_val_side
            feature_data[f"{base_col_name}_norm_opp"] = norm_val_opp
            feature_data[f"{base_col_name}_Asym_Diff"] = norm_val_side - norm_val_opp
            feature_data[f"{base_col_name}_Asym_Ratio"] = calculate_ratio(norm_val_side, norm_val_opp)
            feature_data[f"{base_col_name}_Asym_PercDiff"] = calculate_percent_diff(norm_val_side, norm_val_opp)
            feature_data[f"{base_col_name}_Is_Weaker_Side"] = (norm_val_side < norm_val_opp).astype(int) # Use Series boolean directly

    # 2. Interaction/Summary Features
    avg_au12_ratio_vals = []
    max_au12_pd_vals_list = [] # Store series to find max later
    for act in local_actions:
        ratio_key = f"{act}_AU12_r_Asym_Ratio" # Use Asymmetry Ratio
        pd_key = f"{act}_AU12_r_Asym_PercDiff" # Use Asymmetry Percent Difference
        if ratio_key in feature_data and isinstance(feature_data[ratio_key], pd.Series):
            avg_au12_ratio_vals.append(feature_data[ratio_key])
        if pd_key in feature_data and isinstance(feature_data[pd_key], pd.Series):
            max_au12_pd_vals_list.append(feature_data[pd_key])

    if avg_au12_ratio_vals:
        feature_data['avg_AU12_Asym_Ratio'] = pd.concat(avg_au12_ratio_vals, axis=1).mean(axis=1)
    else: feature_data['avg_AU12_Asym_Ratio'] = pd.Series(1.0, index=df.index) # Default to 1.0 (symmetry)

    if max_au12_pd_vals_list:
        feature_data['max_AU12_Asym_PercDiff'] = pd.concat(max_au12_pd_vals_list, axis=1).max(axis=1)
    else: feature_data['max_AU12_Asym_PercDiff'] = pd.Series(0.0, index=df.index) # Default to 0.0 difference

    # Example: Ratio product for BS action (using asymmetry ratios)
    bs_au12_asym_ratio = feature_data.get('BS_AU12_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    bs_au25_asym_ratio = feature_data.get('BS_AU25_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    feature_data['BS_Asym_Ratio_Product_12_25'] = bs_au12_asym_ratio * bs_au25_asym_ratio

    # Create DataFrame
    features_df = pd.DataFrame(feature_data, index=df.index)

    # Final check for non-numeric types
    non_numeric_cols = features_df.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric cols in extract_features: {non_numeric_cols.tolist()}. Coercing.")
        for col in non_numeric_cols: features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)

    logger.debug(f"Generated {features_df.shape[1]} lower face features for {side} (Training).")
    return features_df


# --- extract_features_for_detection (Unchanged from previous, relies on corrected helpers) ---
def extract_features_for_detection(row_data, side, zone):
    """
    Extract lower face features for detection from a row of data (pd.Series or dict).
    Uses Pandas Series operations internally for consistency with training extraction.
    Relies on the saved feature list for final ordering and selection. Adds detailed logging.

    Args:
        row_data (pd.Series or dict): A row from the merged dataframe containing all AU columns.
        side (str): Side ('left' or 'right'). Note: Will be capitalized internally.
        zone (str): Zone ('lower').

    Returns:
        list: Feature vector for model input or None on error.
    """
    try:
        # Make sure config variables are accessible
        from lower_face_config import MODEL_DIR, FEATURE_SELECTION, FEATURE_CONFIG, LOWER_FACE_ACTIONS, LOWER_FACE_AUS
        local_logger = logging.getLogger(__name__)
    except ImportError:
        logging.error("Failed to import config within lower_face extract_features_for_detection.")
        local_logger = logging
        MODEL_DIR = 'models'; FEATURE_SELECTION = {'enabled': False}
        FEATURE_CONFIG = {'actions': ['BS', 'SS', 'SO', 'SE'], 'aus': ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'], 'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
        LOWER_FACE_ACTIONS = FEATURE_CONFIG['actions']; LOWER_FACE_AUS = FEATURE_CONFIG['aus']

    # --- Check input type ---
    if not isinstance(row_data, (pd.Series, dict)):
        local_logger.error(f"LOWER_FACE_DETECT_INPUT ({side}): Invalid row_data type: {type(row_data)}")
        return None

    # --- Get Patient ID for logging ---
    patient_id_from_row = 'UnknownPatient'
    if isinstance(row_data, pd.Series):
        patient_id_from_row = row_data.get('Patient ID', 'UnknownPatient')
    elif isinstance(row_data, dict):
        patient_id_from_row = row_data.get('Patient ID', 'UnknownPatient')

    # --- Set Test Patient ID ---
    test_patient_id = 'IMG_0422' # <<< CHANGE THIS if your test patient is different
    is_test_patient = (patient_id_from_row == test_patient_id)

    # --- Capitalize side and opposite_side for key construction ---
    side_label = side.capitalize() # 'left' -> 'Left', 'right' -> 'Right'
    opposite_side_label = 'Right' if side_label == 'Left' else 'Left'
    # --- End capitalization ---

    # Log basic input info (if test patient)
    if is_test_patient:
        local_logger.debug(f"LOWER_FACE_DETECT_INPUT ({patient_id_from_row}, {side_label}): Received {type(row_data)}.")
        key_vals_to_log = {k: row_data.get(k, 'MISSING') for k in ['Patient ID', 'BS_Left AU12_r', 'BS_Right AU12_r', 'BS_Left AU12_r (Normalized)', 'BS_Right AU12_r (Normalized)']}
        local_logger.debug(f"LOWER_FACE_DETECT_INPUT ({patient_id_from_row}, {side_label}) Sample Values: {key_vals_to_log}")

    # Convert input row_data to a single-row DataFrame
    try:
        df_single_row = pd.DataFrame([row_data]); df_single_row.index = [0]
        # --- ADDED LOGGING ---
        if is_test_patient:
            local_logger.debug(f"DF_CHECK ({patient_id_from_row}, {side_label}) Created df_single_row. Columns (first 10): {df_single_row.columns.tolist()[:10]}...") # Log only first few columns
            # Check if a specific expected key exists as a column
            test_key = f"BS_{side_label} AU12_r (Normalized)" # Use capitalized key
            local_logger.debug(f"DF_CHECK ({patient_id_from_row}, {side_label}) Does '{test_key}' exist as column? {test_key in df_single_row.columns}")
        # --- END ADDED LOGGING ---
    except Exception as e: local_logger.error(f"Could not convert row_data to DataFrame: {e}"); return None

    try: # Check config access again inside function
        local_feature_config = FEATURE_CONFIG if isinstance(FEATURE_CONFIG, dict) else {}
        local_actions = LOWER_FACE_ACTIONS if 'LOWER_FACE_ACTIONS' in globals() and LOWER_FACE_ACTIONS else ['BS', 'SS', 'SO', 'SE']
        local_aus = LOWER_FACE_AUS if 'LOWER_FACE_AUS' in globals() and LOWER_FACE_AUS else ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']
        use_normalized = local_feature_config.get('use_normalized', True)
        min_val_conf = local_feature_config.get('min_value', 0.0001)
    except NameError:
         local_logger.error("extract_features_for_detection: Config variables not found, using defaults.")
         local_feature_config = {'use_normalized': True, 'min_value': 0.0001}
         local_actions = ['BS', 'SS', 'SO', 'SE']
         local_aus = ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']
         use_normalized = True
         min_val_conf = 0.0001

    local_logger.debug(f"Extracting lower face detection features for {side_label} {zone} using DataFrame method...")
    feature_data = {} # Dictionary to hold Series results

    # --- 1. Basic AU & Asymmetry Features (Use Pandas helpers) ---
    for action in local_actions:
        for au in local_aus:
            base_col_name = f"{action}_{au}"
            # --- Use capitalized labels ---
            au_col_side = f"{action}_{side_label} {au}"
            au_norm_col_side = f"{au_col_side} (Normalized)"
            au_col_opp = f"{action}_{opposite_side_label} {au}"
            au_norm_col_opp = f"{au_col_opp} (Normalized)"
            # --- End Use capitalized labels ---

            # --- ADDED LOGGING ---
            if is_test_patient and action == 'BS' and au == 'AU12_r':
                local_logger.debug(f"DF_GET_CHECK ({patient_id_from_row}, {side_label}) Trying to get key: '{au_norm_col_side}'") # Log with capitalized key
                retrieved_series = df_single_row.get(au_norm_col_side) # Use capitalized key for get
                if retrieved_series is None:
                    local_logger.debug(f"DF_GET_CHECK ({patient_id_from_row}, {side_label}) Key '{au_norm_col_side}' NOT FOUND in df_single_row. Columns are: {df_single_row.columns.tolist()}")
                else:
                    local_logger.debug(f"DF_GET_CHECK ({patient_id_from_row}, {side_label}) Key '{au_norm_col_side}' FOUND. Value: {retrieved_series.iloc[0] if not retrieved_series.empty else 'EMPTY_SERIES'}")
            # --- END ADDED LOGGING ---


            # --- Revised Value Retrieval ---
            raw_val_side = 0.0 # Default
            raw_val_opp = 0.0
            norm_val_side = 0.0
            norm_val_opp = 0.0

            # Get Raw Side
            raw_val_side_obj = df_single_row.get(au_col_side) # Use capitalized key
            if raw_val_side_obj is not None and not pd.isna(raw_val_side_obj.iloc[0]): # Check if found and not NaN
                try:
                    raw_val_side = float(raw_val_side_obj.iloc[0])
                except (ValueError, TypeError):
                    local_logger.warning(f"Failed to convert raw_val_side '{raw_val_side_obj.iloc[0]}' to float for {au_col_side}. Defaulting to 0.0.")
                    raw_val_side = 0.0

            # Get Raw Opp
            raw_val_opp_obj = df_single_row.get(au_col_opp) # Use capitalized key
            if raw_val_opp_obj is not None and not pd.isna(raw_val_opp_obj.iloc[0]):
                try:
                    raw_val_opp = float(raw_val_opp_obj.iloc[0])
                except (ValueError, TypeError):
                    local_logger.warning(f"Failed to convert raw_val_opp '{raw_val_opp_obj.iloc[0]}' to float for {au_col_opp}. Defaulting to 0.0.")
                    raw_val_opp = 0.0

            if use_normalized:
                # Get Norm Side
                norm_val_side_obj = df_single_row.get(au_norm_col_side) # Use capitalized key
                if norm_val_side_obj is not None and not pd.isna(norm_val_side_obj.iloc[0]): # Check if found and not NaN
                    try:
                        norm_val_side = float(norm_val_side_obj.iloc[0]) # Assign to float variable
                    except (ValueError, TypeError):
                        local_logger.warning(f"Failed convert norm_val_side '{norm_val_side_obj.iloc[0]}' for {au_norm_col_side}. Default raw ({raw_val_side:.2f}).")
                        norm_val_side = raw_val_side # Fallback to raw float if conversion fails
                else:
                    # Use raw if normalized is missing or NaN (and log if it's our test case)
                    if is_test_patient and action == 'BS' and au == 'AU12_r':
                         local_logger.warning(f"DF_GET_CHECK ({patient_id_from_row}, {side_label}) Norm key '{au_norm_col_side}' not found or NaN, using raw value ({raw_val_side:.2f}) for norm_val_side.")
                    norm_val_side = raw_val_side # Use already processed raw float

                # Get Norm Opp
                norm_val_opp_obj = df_single_row.get(au_norm_col_opp) # Use capitalized key
                if norm_val_opp_obj is not None and not pd.isna(norm_val_opp_obj.iloc[0]):
                    try:
                        norm_val_opp = float(norm_val_opp_obj.iloc[0])
                    except (ValueError, TypeError):
                        local_logger.warning(f"Failed convert norm_val_opp '{norm_val_opp_obj.iloc[0]}' for {au_norm_col_opp}. Default raw ({raw_val_opp:.2f}).")
                        norm_val_opp = raw_val_opp # Fallback to raw float if conversion fails
                else:
                    norm_val_opp = raw_val_opp # Use already processed raw float
            else:
                norm_val_side = raw_val_side # Use already processed raw float
                norm_val_opp = raw_val_opp # Use already processed raw float
            # --- End Revised Value Retrieval ---

            # --- Store values as Series for helper functions ---
            # Make sure to use the processed float values here
            feature_data[f"{base_col_name}_raw_side"] = pd.Series([raw_val_side], index=df_single_row.index)
            feature_data[f"{base_col_name}_raw_opp"] = pd.Series([raw_val_opp], index=df_single_row.index)
            feature_data[f"{base_col_name}_norm_side"] = pd.Series([norm_val_side], index=df_single_row.index)
            feature_data[f"{base_col_name}_norm_opp"] = pd.Series([norm_val_opp], index=df_single_row.index)

            # --- Calculate Asymmetry Features ---
            feature_data[f"{base_col_name}_Asym_Diff"] = feature_data[f"{base_col_name}_norm_side"] - feature_data[f"{base_col_name}_norm_opp"]

            ratio_input_side_series = feature_data[f"{base_col_name}_norm_side"]
            ratio_input_opp_series = feature_data[f"{base_col_name}_norm_opp"]
            # --- ADDED LOGGING ---
            if is_test_patient and au in ['AU12_r', 'AU25_r']:
                local_logger.debug(f"HELPER_INPUT (RATIO) ({patient_id_from_row}, {side_label}, {action}, {au}): side_series[0]={ratio_input_side_series.iloc[0]:.4f}, opp_series[0]={ratio_input_opp_series.iloc[0]:.4f}")
            # --- END LOGGING ---
            calc_ratio_result = calculate_ratio(ratio_input_side_series, ratio_input_opp_series)
            if is_test_patient and au in ['AU12_r', 'AU25_r']: local_logger.debug(f"CALC_RATIO Output ({patient_id_from_row}, {side_label}, {action}, {au}): {calc_ratio_result.iloc[0]:.4f}") # Keep output log
            feature_data[f"{base_col_name}_Asym_Ratio"] = calc_ratio_result

            percdiff_input_side_series = feature_data[f"{base_col_name}_norm_side"]
            percdiff_input_opp_series = feature_data[f"{base_col_name}_norm_opp"]
             # --- ADDED LOGGING ---
            if is_test_patient and au in ['AU12_r', 'AU25_r']:
                local_logger.debug(f"HELPER_INPUT (PERCDIFF) ({patient_id_from_row}, {side_label}, {action}, {au}): side_series[0]={percdiff_input_side_series.iloc[0]:.4f}, opp_series[0]={percdiff_input_opp_series.iloc[0]:.4f}")
            # --- END LOGGING ---
            calc_percdiff_result = calculate_percent_diff(percdiff_input_side_series, percdiff_input_opp_series)
            if is_test_patient and au in ['AU12_r', 'AU25_r']: local_logger.debug(f"CALC_PERCDIFF Output ({patient_id_from_row}, {side_label}, {action}, {au}): {calc_percdiff_result.iloc[0]:.4f}") # Keep output log
            feature_data[f"{base_col_name}_Asym_PercDiff"] = calc_percdiff_result

            # --- FIXED: Wrap int in pd.Series ---
            feature_data[f"{base_col_name}_Is_Weaker_Side"] = pd.Series([int(norm_val_side < norm_val_opp)], index=df_single_row.index)
            # --- END FIX ---

            # --- ADDED CHECK for missing feature calculation ---
            # (Keep this logging check)
            if action == 'SS' and au == 'AU12_r':
                temp_key = f"{base_col_name}_Is_Weaker_Side"
                if temp_key in feature_data:
                     local_logger.debug(f"MISSING_FEAT_CHECK ({patient_id_from_row}, {side_label}): Calculated value for '{temp_key}': {feature_data[temp_key].iloc[0]}")
                else:
                     local_logger.error(f"MISSING_FEAT_CHECK ({patient_id_from_row}, {side_label}): Key '{temp_key}' NOT FOUND in feature_data after calculation!")
            # --- END ADDED CHECK ---


    # --- 2. Interaction/Summary Features (Use Pandas helpers) ---
    avg_au12_ratio_vals = []
    max_au12_pd_vals_list = []
    for act in local_actions:
        ratio_key = f"{act}_AU12_r_Asym_Ratio"
        pd_key = f"{act}_AU12_r_Asym_PercDiff"
        if ratio_key in feature_data and isinstance(feature_data[ratio_key], pd.Series):
            avg_au12_ratio_vals.append(feature_data[ratio_key])
        if pd_key in feature_data and isinstance(feature_data[pd_key], pd.Series):
            max_au12_pd_vals_list.append(feature_data[pd_key])

    # --- Detailed Logging for Aggregations ---
    if is_test_patient:
        avg_au12_ratio_raw_values = [s.iloc[0] for s in avg_au12_ratio_vals if not s.empty] # Handle potentially empty series
        local_logger.debug(f"AGG_AVG_RATIO Input ({patient_id_from_row}, {side_label}, AU12): {avg_au12_ratio_raw_values}")
    if avg_au12_ratio_vals:
        feature_data['avg_AU12_Asym_Ratio'] = pd.concat(avg_au12_ratio_vals, axis=1).mean(axis=1)
    else: feature_data['avg_AU12_Asym_Ratio'] = pd.Series(1.0, index=df_single_row.index)
    if is_test_patient:
        avg_ratio_output_val = feature_data['avg_AU12_Asym_Ratio'].iloc[0] if not feature_data['avg_AU12_Asym_Ratio'].empty else 'EMPTY_SERIES'
        local_logger.debug(f"AGG_AVG_RATIO Output ({patient_id_from_row}, {side_label}, AU12): {avg_ratio_output_val:.4f}") # Added formatting


    if is_test_patient:
        max_au12_pd_raw_values = [s.iloc[0] for s in max_au12_pd_vals_list if not s.empty] # Handle potentially empty series
        local_logger.debug(f"AGG_MAX_PERCDIFF Input ({patient_id_from_row}, {side_label}, AU12): {max_au12_pd_raw_values}")
    if max_au12_pd_vals_list:
        feature_data['max_AU12_Asym_PercDiff'] = pd.concat(max_au12_pd_vals_list, axis=1).max(axis=1)
    else: feature_data['max_AU12_Asym_PercDiff'] = pd.Series(0.0, index=df_single_row.index)
    if is_test_patient:
        max_pd_output_val = feature_data['max_AU12_Asym_PercDiff'].iloc[0] if not feature_data['max_AU12_Asym_PercDiff'].empty else 'EMPTY_SERIES'
        local_logger.debug(f"AGG_MAX_PERCDIFF Output ({patient_id_from_row}, {side_label}, AU12): {max_pd_output_val:.4f}") # Added formatting
    # --- End Detailed Logging ---

    bs_au12_asym_ratio = feature_data.get('BS_AU12_r_Asym_Ratio', pd.Series(1.0, index=df_single_row.index))
    bs_au25_asym_ratio = feature_data.get('BS_AU25_r_Asym_Ratio', pd.Series(1.0, index=df_single_row.index))
    feature_data['BS_Asym_Ratio_Product_12_25'] = bs_au12_asym_ratio * bs_au25_asym_ratio

    feature_data["side_indicator"] = pd.Series([0 if side_label.lower() == 'left' else 1], index=df_single_row.index) # Use side_label here too

    # --- Convert dictionary of Series back to dictionary of single values ---
    # Ensure ALL items from feature_data are included now, as _Is_Weaker_Side is a Series
    feature_dict_final = {k: v.iloc[0] for k, v in feature_data.items() if isinstance(v, pd.Series) and not v.empty}

    # Load feature list
    local_model_dir = MODEL_DIR if 'MODEL_DIR' in globals() and MODEL_DIR else 'models'
    feature_names_path = os.path.join(local_model_dir, 'lower_face_features.list') # Correct list name
    if not os.path.exists(feature_names_path): local_logger.error(f"Lower face feature list not found: {feature_names_path}."); return None
    try:
        ordered_feature_names = joblib.load(feature_names_path)
        if not isinstance(ordered_feature_names, list): local_logger.error(f"Loaded feature names is not a list: {type(ordered_feature_names)}"); return None
    except Exception as e: local_logger.error(f"Failed to load lower face feature list: {e}", exc_info=True); return None


    # --- ADDED LOGGING: Before Final Assembly Loop ---
    if is_test_patient: # is_test_patient defined earlier based on patient_id_from_row
         try:
             # Log a sample of the feature_dict_final before the loop
             sample_keys = [name for name in ordered_feature_names[:10] if name in feature_dict_final] # Get first 10 expected keys that exist
             sample_dict_final = {k: feature_dict_final.get(k, 'MISSING_IN_FINAL_DICT') for k in sample_keys} # Use get() for safety
             local_logger.debug(f"FINAL_ASSEMBLY ({patient_id_from_row}, {side_label}) feature_dict_final sample: {sample_dict_final}")
             # Specifically check the weak side key again
             weak_side_key = 'SS_AU12_r_Is_Weaker_Side'
             local_logger.debug(f"FINAL_ASSEMBLY ({patient_id_from_row}, {side_label}) Value for '{weak_side_key}': {feature_dict_final.get(weak_side_key, 'KEY_NOT_FOUND')}")
         except Exception as log_e:
             local_logger.error(f"Error during feature_dict_final logging: {log_e}")
    # --- END ADDED LOGGING ---


    # Build final list
    feature_list = []
    missing_in_dict = [] # Keep track of missing features
    for i, name in enumerate(ordered_feature_names): # Add index 'i'
        value = feature_dict_final.get(name) # Use get without default initially
        if value is None: # Check if key was truly missing
            missing_in_dict.append(name) # Add to missing list
            final_val = 0.0 # Default for missing keys
        else:
            # Key exists, attempt conversion
            try:
                final_val = float(value)
                if np.isnan(final_val): # Check for NaN after conversion
                    local_logger.warning(f"NaN value encountered for feature '{name}' after float conversion. Defaulting to 0.0.")
                    final_val = 0.0
            except (ValueError, TypeError):
                local_logger.warning(f"Failed to convert value '{value}' for feature '{name}' to float. Defaulting to 0.0.")
                final_val = 0.0

        # --- ADDED LOGGING: Inside Final Assembly Loop ---
        # Log the name, retrieved value, and final value being appended for the first few features
        if is_test_patient and i < 10: # Log first 10 features being assembled
             original_val_str = f"'{value}'" if value is not None else 'KEY_MISSING'
             local_logger.debug(f"FINAL_ASSEMBLY ({patient_id_from_row}, {side_label}) Index={i}, Name='{name}', Retrieved={original_val_str}, Appended={final_val:.4f}")
        # --- END ADDED LOGGING ---

        feature_list.append(final_val) # Append the processed float value

    # --- Log missing features AFTER the loop ---
    # Only log if there were actually missing features
    if missing_in_dict:
        local_logger.warning(f"Detection: {len(missing_in_dict)} expected features missing from final dict: {missing_in_dict[:5]}... Defaulting to 0.")
    # --- End logging missing features ---

    if len(feature_list) != len(ordered_feature_names):
         local_logger.error(f"CRITICAL MISMATCH: Lower face detection list length ({len(feature_list)}) != expected ({len(ordered_feature_names)}).")
         return None

    # --- Added logging of the extracted feature vector ---
    if is_test_patient:
        local_logger.debug(f"LOWER_FACE_DETECT_FEATURES ({patient_id_from_row}, {side_label}): Extracted {len(feature_list)} features.")
        local_logger.debug(f"  First 5: {[f'{v:.4f}' for v in feature_list[:5]]}") # Format output
        local_logger.debug(f"  Last 5: {[f'{v:.4f}' for v in feature_list[-5:]]}") # Format output
    # ---

    local_logger.debug(f"Generated {len(feature_list)} lower face detection features for {side_label} {zone}.")
    return feature_list

# --- process_targets function (copied for completeness if run standalone) ---
def process_targets(target_series):
    mapping = {'none': 0, 'no': 0, 'normal': 0, 'partial': 1, 'mild': 1, 'moderate': 1, 'complete': 2, 'severe': 2, 'nan': 0}
    processed = target_series.astype(str).str.lower().str.strip().map(mapping)
    processed = processed.fillna(0)
    return processed.astype(int).values