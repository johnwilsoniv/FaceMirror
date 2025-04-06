# snarl_smile_features.py
# - Extracts features ONLY for BS action (normalized) vs BL (for normalization).
# - REMOVED SS action processing and cross-action summaries.

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Import config for Snarl-Smile
try:
    from snarl_smile_config import (
        FEATURE_CONFIG, SNARL_SMILE_ACTIONS, TRIGGER_AUS, COUPLED_AUS, LOG_DIR,
        FEATURE_SELECTION, MODEL_FILENAMES, CLASS_NAMES
    )
    CONFIG_LOADED = True
except ImportError:
    logging.warning("Could not import from snarl_smile_config. Using fallback definitions.")
    CONFIG_LOADED = False
    # Fallbacks
    LOG_DIR = 'logs'; MODEL_DIR = 'models/synkinesis/snarl_smile'
    MODEL_FILENAMES = {'feature_list': os.path.join(MODEL_DIR,'features.list'),
                       'importance_file': os.path.join(MODEL_DIR,'feature_importance.csv')}
    SNARL_SMILE_ACTIONS = ['BS']; TRIGGER_AUS = ['AU12_r']; COUPLED_AUS = ['AU14_r', 'AU15_r'] # Updated fallback AUs
    FEATURE_CONFIG = {'actions': SNARL_SMILE_ACTIONS, 'trigger_aus': TRIGGER_AUS, 'coupled_aus': COUPLED_AUS,
                      'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 15, 'importance_file': MODEL_FILENAMES['importance_file']}
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- prepare_data function (Unchanged logic, calls modified extract_features) ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data for Snarl-Smile synkinesis training (BS-Only), including feature selection. """
    logger.info("Loading datasets for Snarl-Smile Synkinesis (BS-Only)...")
    try:
        results_df = pd.read_csv(results_file, low_memory=False)
        expert_df = pd.read_csv(expert_file)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise

    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        'Snarl Smile Left': 'Expert_Left_Snarl_Smile',
        'Snarl Smile Right': 'Expert_Right_Snarl_Smile'
        })

    def process_targets(target_series):
        """ Converts expert labels (Yes/No etc.) to binary 0/1 """
        if target_series is None: return np.array([], dtype=int)
        mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
        s_filled = target_series.fillna('no')
        s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
        mapped = s_clean.map(mapping)
        unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
        if not unexpected.empty: logger.warning(f"Unexpected Snarl-Smile expert labels found (treated as 'No'): {unexpected.unique().tolist()}")
        final_mapped = mapped.fillna(0)
        return final_mapped.astype(int).values

    if 'Expert_Left_Snarl_Smile' in expert_df.columns: expert_df['Target_Left_Snarl_Smile'] = process_targets(expert_df['Expert_Left_Snarl_Smile'])
    else: logger.error("Missing 'Expert_Left_Snarl_Smile' column"); expert_df['Target_Left_Snarl_Smile'] = 0
    if 'Expert_Right_Snarl_Smile' in expert_df.columns: expert_df['Target_Right_Snarl_Smile'] = process_targets(expert_df['Expert_Right_Snarl_Smile'])
    else: logger.error("Missing 'Expert_Right_Snarl_Smile' column"); expert_df['Target_Right_Snarl_Smile'] = 0

    logger.info(f"Counts in expert_df['Target_Left_Snarl_Smile'] AFTER mapping: \n{pd.Series(expert_df['Target_Left_Snarl_Smile']).value_counts(dropna=False)}")
    logger.info(f"Counts in expert_df['Target_Right_Snarl_Smile'] AFTER mapping: \n{pd.Series(expert_df['Target_Right_Snarl_Smile']).value_counts(dropna=False)}")

    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
    expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
    expert_cols_to_merge = ['Patient ID', 'Target_Left_Snarl_Smile', 'Target_Right_Snarl_Smile']

    try:
        merged_df = pd.merge(results_df, expert_df[expert_cols_to_merge], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data for Snarl-Smile: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    logger.info(f"Counts in merged_df['Target_Left_Snarl_Smile'] AFTER merge: \n{merged_df['Target_Left_Snarl_Smile'].value_counts(dropna=False)}")
    logger.info(f"Counts in merged_df['Target_Right_Snarl_Smile'] AFTER merge: \n{merged_df['Target_Right_Snarl_Smile'].value_counts(dropna=False)}")

    logger.info("Extracting Snarl-Smile features for Left side (BS-Only)...")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info("Extracting Snarl-Smile features for Right side (BS-Only)...")
    right_features_df = extract_features(merged_df, 'Right')

    if left_features_df is None or right_features_df is None:
         logger.error("Feature extraction failed for one or both sides.")
         return None, None

    if 'Target_Left_Snarl_Smile' not in merged_df.columns or 'Target_Right_Snarl_Smile' not in merged_df.columns:
         logger.error("Target columns missing in merged_df before creating targets array. Aborting.")
         return None, None
    left_targets = merged_df['Target_Left_Snarl_Smile'].values
    right_targets = merged_df['Target_Right_Snarl_Smile'].values
    targets = np.concatenate([left_targets, right_targets])

    unique_final, counts_final = np.unique(targets, return_counts=True)
    # Use CLASS_NAMES for logging the distribution
    final_class_dist = {CLASS_NAMES.get(i, f"Class_{i}"): int(c) for i, c in zip(unique_final, counts_final)}
    logger.info(f"FINAL Snarl-Smile Class distribution input: {final_class_dist}")


    left_features_df['side_indicator'] = 0
    right_features_df['side_indicator'] = 1
    features = pd.concat([left_features_df, right_features_df], ignore_index=True)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.fillna(0)
    initial_cols = features.columns.tolist()
    cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert column {col} to numeric: {e}. Marking for drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)

    logger.info(f"Generated initial {features.shape[1]} Snarl-Smile features (BS-Only).")

    fs_enabled = isinstance(FEATURE_SELECTION, dict) and FEATURE_SELECTION.get('enabled', False)
    if fs_enabled:
        logger.info("Applying Snarl-Smile feature selection...")
        n_top_features = FEATURE_SELECTION.get('top_n_features', 15) # Use config value
        importance_file = FEATURE_SELECTION.get('importance_file')
        if not importance_file or not os.path.exists(importance_file): logger.warning(f"FS enabled, but importance file not found: '{importance_file}'. Skipping selection.")
        else:
            try:
                importance_df = pd.read_csv(importance_file)
                if 'feature' not in importance_df.columns or importance_df.empty: logger.error("Importance file invalid. Skipping selection.")
                else:
                    top_feature_names = importance_df['feature'].head(n_top_features).tolist()
                    if 'side_indicator' in features.columns and 'side_indicator' not in top_feature_names: top_feature_names.append('side_indicator')
                    original_cols = features.columns.tolist()
                    cols_to_keep = [col for col in top_feature_names if col in original_cols]
                    missing_features = set(top_feature_names) - set(cols_to_keep)
                    if missing_features: logger.warning(f"Some important Snarl-Smile features missing: {missing_features}")
                    if not cols_to_keep: logger.error("No features left after filtering. Skipping selection.")
                    else: logger.info(f"Selecting top {len(cols_to_keep)} Snarl-Smile features."); features = features[cols_to_keep]
            except Exception as e: logger.error(f"Error during Snarl-Smile FS: {e}. Skipping selection.", exc_info=True)
    else:
        logger.info("Snarl-Smile feature selection is disabled.")

    logger.info(f"Final Snarl-Smile dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.warning(f"NaNs found in FINAL Snarl-Smile features BEFORE saving list. Columns: {features.columns[features.isna().any()].tolist()}. Filling with 0."); features = features.fillna(0)

    final_feature_names = features.columns.tolist()
    try:
        feature_list_path = MODEL_FILENAMES.get('feature_list')
        if feature_list_path:
             os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
             joblib.dump(final_feature_names, feature_list_path)
             logger.info(f"Saved final {len(final_feature_names)} Snarl-Smile feature names list to {feature_list_path}")
        else: logger.error("Snarl-Smile feature list path not defined in config.")
    except Exception as e: logger.error(f"Failed to save Snarl-Smile feature names list: {e}", exc_info=True)

    if 'targets' not in locals(): logger.error("Targets array creation failed."); return None, None
    return features, targets


# --- Helper Functions (Corrected versions - Unchanged) ---
def calculate_ratio(val1_series, val2_series):
    local_logger = logging.getLogger(__name__)
    try: min_val_config = FEATURE_CONFIG.get('min_value', 0.0001)
    except NameError: min_val_config = 0.0001; local_logger.warning("calculate_ratio: FEATURE_CONFIG not found.")
    epsilon=1e-9
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    min_vals_np = np.minimum(v1, v2); max_vals_np = np.maximum(v1, v2)
    ratio = np.ones_like(v1, dtype=float)
    mask_max_pos = max_vals_np > min_val_config; mask_min_zero = min_vals_np <= min_val_config
    ratio[mask_max_pos & mask_min_zero] = 0.0
    valid_division_mask = mask_max_pos & ~mask_min_zero
    if np.any(valid_division_mask): ratio[valid_division_mask] = min_vals_np[valid_division_mask] / (max_vals_np[valid_division_mask] + epsilon)
    if np.isnan(ratio).any() or np.isinf(ratio).any(): ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    return pd.Series(ratio, index=val1_series.index)

def calculate_percent_diff(val1_series, val2_series):
    local_logger = logging.getLogger(__name__)
    try:
        min_val_config = FEATURE_CONFIG.get('min_value', 0.0001)
        percent_diff_cap = FEATURE_CONFIG.get('percent_diff_cap', 200.0)
    except NameError: min_val_config = 0.0001; percent_diff_cap = 200.0; local_logger.warning("calculate_percent_diff: FEATURE_CONFIG not found.")
    epsilon=1e-9
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    abs_diff = np.abs(v1 - v2); avg = (v1 + v2) / 2.0
    percent_diff = np.zeros_like(avg, dtype=float)
    mask_avg_pos = avg > min_val_config
    if np.any(mask_avg_pos):
        # Use errstate to handle potential division by zero if avg is epsilon
        with np.errstate(divide='ignore', invalid='ignore'):
             division_result = abs_diff[mask_avg_pos] / (avg[mask_avg_pos] + epsilon)
        percent_diff[mask_avg_pos] = division_result * 100.0
    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > min_val_config)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap
    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    percent_diff = np.nan_to_num(percent_diff, nan=0.0, posinf=percent_diff_cap, neginf=0.0)
    return pd.Series(percent_diff, index=val1_series.index)


# --- extract_features function (Training - BS Only) ---
def extract_features(df, side):
    """ Extracts Snarl-Smile features for TRAINING using ONLY BS (normalized). """
    logger.debug(f"Extracting Snarl-Smile features for {side} side (Training - BS Only)...")
    feature_data = {}
    side_label = side.capitalize()

    # --- Use Config values ---
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED else {}
    local_actions = local_feature_config.get('actions', ['BS']) # Should be ['BS']
    local_trigger_aus = local_feature_config.get('trigger_aus', TRIGGER_AUS)
    local_coupled_aus = local_feature_config.get('coupled_aus', COUPLED_AUS)
    use_normalized = local_feature_config.get('use_normalized', True)
    # --- End Config ---

    # Ensure we only process BS, even if config was wrong
    if 'BS' not in local_actions:
        logger.error("BS action missing from config actions. Cannot generate BS-only features.")
        return None
    action = 'BS' # Hardcode to BS

    # Helper to get numeric series safely
    def get_numeric_series(col_name, default_val=0.0):
        series = df.get(col_name)
        if series is None:
            # Fallback for missing columns: Create a Series of default values
            logger.warning(f"Column '{col_name}' not found in DataFrame. Using default value {default_val}.")
            return pd.Series(default_val, index=df.index, dtype=float)
        # Attempt conversion, fill NaNs resulting from conversion *or* original NaNs
        numeric_series = pd.to_numeric(series, errors='coerce').fillna(default_val)
        return numeric_series.astype(float) # Ensure float type

    # --- Get Baseline Raw Values (Needed for Normalization) ---
    bl_raw_values = {}
    for au in local_trigger_aus + local_coupled_aus:
        bl_col_raw = f"BL_{side_label} {au}"
        bl_raw_values[au] = get_numeric_series(bl_col_raw)
    # --- End Baseline ---

    # --- Process BS Features ---
    if not local_trigger_aus:
        logger.error("Trigger AU list is empty in config.")
        return None
    trig_au = local_trigger_aus[0] # Assume only one trigger AU

    # Get Raw BS Values
    bs_raw_values = {}
    bs_raw_values[trig_au] = get_numeric_series(f"BS_{side_label} {trig_au}")
    for coup_au in local_coupled_aus:
        bs_raw_values[coup_au] = get_numeric_series(f"BS_{side_label} {coup_au}")

    # Calculate Normalized BS Values (and store features)
    bs_norm_values = {}
    trigger_norm_series = pd.Series(0.0, index=df.index) # Initialize
    if use_normalized:
        trigger_norm_series = (bs_raw_values[trig_au] - bl_raw_values.get(trig_au, 0.0)).clip(lower=0)
        feature_data[f"BS_{trig_au}_trig_norm"] = trigger_norm_series
        bs_norm_values[trig_au] = trigger_norm_series # Store for ratio calc

        bs_coupled_sum_norm = pd.Series(0.0, index=df.index)
        for coup_au in local_coupled_aus:
            coup_norm_series = (bs_raw_values[coup_au] - bl_raw_values.get(coup_au, 0.0)).clip(lower=0)
            feature_data[f"BS_{coup_au}_coup_norm"] = coup_norm_series
            bs_norm_values[coup_au] = coup_norm_series # Store for asymmetry & sum
            feature_data[f"BS_Ratio_{coup_au}_vs_{trig_au}"] = calculate_ratio(coup_norm_series, trigger_norm_series)
            bs_coupled_sum_norm = bs_coupled_sum_norm.add(coup_norm_series, fill_value=0)

        feature_data["BS_Coupled_Sum_Norm"] = bs_coupled_sum_norm
        feature_data["BS_Ratio_CoupledSum_vs_Trigger"] = calculate_ratio(bs_coupled_sum_norm, trigger_norm_series)

    else: # Fallback if use_normalized is False (though config should be True)
        logger.warning("use_normalized is False in config, generating raw BS features instead.")
        trigger_raw_series = bs_raw_values[trig_au]
        feature_data[f"BS_{trig_au}_trig_raw"] = trigger_raw_series # Store raw trigger
        bs_norm_values[trig_au] = trigger_raw_series # Use raw for ratios if norm not used

        bs_coupled_sum_raw = pd.Series(0.0, index=df.index)
        for coup_au in local_coupled_aus:
            coup_raw_series = bs_raw_values[coup_au]
            feature_data[f"BS_{coup_au}_coup_raw"] = coup_raw_series
            bs_norm_values[coup_au] = coup_raw_series # Use raw for asymmetry
            feature_data[f"BS_Ratio_{coup_au}_vs_{trig_au}"] = calculate_ratio(coup_raw_series, trigger_raw_series)
            bs_coupled_sum_raw = bs_coupled_sum_raw.add(coup_raw_series, fill_value=0)

        feature_data["BS_Coupled_Sum_Raw"] = bs_coupled_sum_raw
        feature_data["BS_Ratio_CoupledSum_vs_Trigger"] = calculate_ratio(bs_coupled_sum_raw, trigger_raw_series)
    # --- End Process BS Features ---

    # --- BS Asymmetry Features (using Normalized values if available) ---
    bs_asym_features = {}
    # Get Left/Right **normalized** BS values (calculate opposite side's norm value)
    bs_norm_values_left = {}
    bs_norm_values_right = {}

    # Calculate normalized values for both sides regardless of current 'side'
    for au in local_trigger_aus + local_coupled_aus:
        bs_left_raw = get_numeric_series(f"BS_Left {au}")
        bs_right_raw = get_numeric_series(f"BS_Right {au}")
        bl_left_raw = get_numeric_series(f"BL_Left {au}")
        bl_right_raw = get_numeric_series(f"BL_Right {au}")

        if use_normalized:
            bs_norm_values_left[au] = (bs_left_raw - bl_left_raw).clip(lower=0)
            bs_norm_values_right[au] = (bs_right_raw - bl_right_raw).clip(lower=0)
        else: # Use raw values if normalization is off
            bs_norm_values_left[au] = bs_left_raw
            bs_norm_values_right[au] = bs_right_raw

    # Trigger Asymmetry
    trig_au = local_trigger_aus[0]
    left_trig = bs_norm_values_left.get(trig_au, pd.Series(0.0, index=df.index))
    right_trig = bs_norm_values_right.get(trig_au, pd.Series(0.0, index=df.index))
    bs_asym_features[f"BS_Asym_Ratio_{trig_au}"] = calculate_ratio(left_trig, right_trig)
    bs_asym_features[f"BS_Asym_PercDiff_{trig_au}"] = calculate_percent_diff(left_trig, right_trig)

    # Coupled AU Asymmetry & Sum Asymmetry
    bs_coupled_sum_left = pd.Series(0.0, index=df.index)
    bs_coupled_sum_right = pd.Series(0.0, index=df.index)
    for coup_au in local_coupled_aus:
        left_coup = bs_norm_values_left.get(coup_au, pd.Series(0.0, index=df.index))
        right_coup = bs_norm_values_right.get(coup_au, pd.Series(0.0, index=df.index))

        bs_asym_features[f"BS_Asym_Ratio_{coup_au}"] = calculate_ratio(left_coup, right_coup)
        bs_asym_features[f"BS_Asym_PercDiff_{coup_au}"] = calculate_percent_diff(left_coup, right_coup)

        bs_coupled_sum_left = bs_coupled_sum_left.add(left_coup, fill_value=0)
        bs_coupled_sum_right = bs_coupled_sum_right.add(right_coup, fill_value=0)

    bs_asym_features["BS_Asym_Ratio_CoupledSum"] = calculate_ratio(bs_coupled_sum_left, bs_coupled_sum_right)
    bs_asym_features["BS_Asym_PercDiff_CoupledSum"] = calculate_percent_diff(bs_coupled_sum_left, bs_coupled_sum_right)

    feature_data.update(bs_asym_features)
    # --- End BS Asymmetry ---

    # --- REMOVED Summary Features Across Actions ---

    # Create DataFrame
    features_df = pd.DataFrame(feature_data, index=df.index)

    # Final check
    non_numeric_cols = features_df.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric cols in Snarl-Smile extract_features (BS-Only): {non_numeric_cols.tolist()}. Coercing.")
        for col in non_numeric_cols: features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df = features_df.fillna(0.0)

    logger.debug(f"Generated {features_df.shape[1]} Snarl-Smile features for {side_label} (Training - BS Only).")
    return features_df


# --- extract_features_for_detection (Detection - BS Only) ---
def extract_features_for_detection(row_data, side):
    """ Extracts Snarl-Smile features for detection using ONLY BS (normalized). """
    try:
        from snarl_smile_config import FEATURE_CONFIG, SNARL_SMILE_ACTIONS, TRIGGER_AUS, COUPLED_AUS, MODEL_FILENAMES
        local_logger = logging.getLogger(__name__)
        CONFIG_LOADED_DETECT = True
    except ImportError:
        logging.error("Failed config import within snarl_smile extract_features_for_detection (BS-Only)."); return None
        CONFIG_LOADED_DETECT = False
        # Minimal fallbacks
        FEATURE_CONFIG = {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
        SNARL_SMILE_ACTIONS = ['BS']; TRIGGER_AUS = ['AU12_r']; COUPLED_AUS = ['AU14_r', 'AU15_r']
        MODEL_FILENAMES = {'feature_list': 'models/synkinesis/snarl_smile/features.list'}
        local_logger = logging

    if not isinstance(row_data, (pd.Series, dict)): local_logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    pid_for_log = row_series.get('Patient ID', 'UnknownPID')

    if side not in ['Left', 'Right']:
        local_logger.error(f"Invalid 'side' argument '{side}'. Must be 'Left' or 'Right'.")
        side_label = side.capitalize()
        if side_label not in ['Left', 'Right']: return None
    else: side_label = side

    # --- Use Config values ---
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED_DETECT else {}
    local_actions = local_feature_config.get('actions', ['BS'])
    local_trigger_aus = local_feature_config.get('trigger_aus', TRIGGER_AUS)
    local_coupled_aus = local_feature_config.get('coupled_aus', COUPLED_AUS)
    use_normalized = local_feature_config.get('use_normalized', True)
    min_val_config = local_feature_config.get('min_value', 0.0001)
    percent_diff_cap = local_feature_config.get('percent_diff_cap', 200.0)
    epsilon = 1e-9
    # --- End Config ---

    if 'BS' not in local_actions:
        logger.error("BS action missing from config actions. Cannot generate BS-only detection features.")
        return None
    action = 'BS' # Hardcode

    local_logger.debug(f"Extracting Snarl-Smile detection features for {side_label} (BS-Only)...")
    feature_dict_final = {}

    # Helper to safely get float from row_series
    def get_float_value(key, default=0.0):
        val = row_series.get(key, default)
        try:
             f_val = float(val)
             if np.isnan(f_val) or np.isinf(f_val): return default
             return f_val
        except (ValueError, TypeError): return default

    # --- Get Baseline Raw Values ---
    bl_raw_values_scalar = {}
    for au in local_trigger_aus + local_coupled_aus:
        bl_col_raw = f"BL_{side_label} {au}"
        bl_raw_values_scalar[au] = get_float_value(bl_col_raw)
    # --- End Baseline ---

    # --- Process BS Features (Scalar) ---
    if not local_trigger_aus: logger.error("Trigger AU list empty."); return None
    trig_au = local_trigger_aus[0]

    # Get Raw BS Values
    bs_raw_values_scalar = {}
    bs_raw_values_scalar[trig_au] = get_float_value(f"BS_{side_label} {trig_au}")
    for coup_au in local_coupled_aus:
        bs_raw_values_scalar[coup_au] = get_float_value(f"BS_{side_label} {coup_au}")

    # Calculate Normalized BS Values (and store features)
    bs_norm_values_scalar = {}
    trigger_norm_val = 0.0
    if use_normalized:
        trigger_norm_val = max(0.0, bs_raw_values_scalar[trig_au] - bl_raw_values_scalar.get(trig_au, 0.0))
        feature_dict_final[f"BS_{trig_au}_trig_norm"] = trigger_norm_val
        bs_norm_values_scalar[trig_au] = trigger_norm_val

        bs_coupled_sum_norm_scalar = 0.0
        for coup_au in local_coupled_aus:
            coup_norm_val = max(0.0, bs_raw_values_scalar[coup_au] - bl_raw_values_scalar.get(coup_au, 0.0))
            feature_dict_final[f"BS_{coup_au}_coup_norm"] = coup_norm_val
            bs_norm_values_scalar[coup_au] = coup_norm_val

            # Ratio vs Trigger
            min_r = min(coup_norm_val, trigger_norm_val); max_r = max(coup_norm_val, trigger_norm_val)
            ratio = 1.0 if max_r <= min_val_config else (0.0 if min_r <= min_val_config else min_r / (max_r + epsilon))
            feature_dict_final[f"BS_Ratio_{coup_au}_vs_{trig_au}"] = np.clip(np.nan_to_num(ratio, nan=1.0), 0.0, 1.0)

            bs_coupled_sum_norm_scalar += coup_norm_val

        feature_dict_final["BS_Coupled_Sum_Norm"] = bs_coupled_sum_norm_scalar
        min_r_sum = min(bs_coupled_sum_norm_scalar, trigger_norm_val); max_r_sum = max(bs_coupled_sum_norm_scalar, trigger_norm_val)
        ratio_sum = 1.0 if max_r_sum <= min_val_config else (0.0 if min_r_sum <= min_val_config else min_r_sum / (max_r_sum + epsilon))
        feature_dict_final["BS_Ratio_CoupledSum_vs_Trigger"] = np.clip(np.nan_to_num(ratio_sum, nan=1.0), 0.0, 1.0)

    else: # Fallback if use_normalized is False
        logger.warning("use_normalized is False. Generating raw BS features for detection.")
        trigger_raw_val = bs_raw_values_scalar[trig_au]
        feature_dict_final[f"BS_{trig_au}_trig_raw"] = trigger_raw_val
        bs_norm_values_scalar[trig_au] = trigger_raw_val # Use raw for ratios

        bs_coupled_sum_raw_scalar = 0.0
        for coup_au in local_coupled_aus:
            coup_raw_val = bs_raw_values_scalar[coup_au]
            feature_dict_final[f"BS_{coup_au}_coup_raw"] = coup_raw_val
            bs_norm_values_scalar[coup_au] = coup_raw_val # Use raw for asymmetry

            min_r = min(coup_raw_val, trigger_raw_val); max_r = max(coup_raw_val, trigger_raw_val)
            ratio = 1.0 if max_r <= min_val_config else (0.0 if min_r <= min_val_config else min_r / (max_r + epsilon))
            feature_dict_final[f"BS_Ratio_{coup_au}_vs_{trig_au}"] = np.clip(np.nan_to_num(ratio, nan=1.0), 0.0, 1.0)
            bs_coupled_sum_raw_scalar += coup_raw_val

        feature_dict_final["BS_Coupled_Sum_Raw"] = bs_coupled_sum_raw_scalar
        min_r_sum = min(bs_coupled_sum_raw_scalar, trigger_raw_val); max_r_sum = max(bs_coupled_sum_raw_scalar, trigger_raw_val)
        ratio_sum = 1.0 if max_r_sum <= min_val_config else (0.0 if min_r_sum <= min_val_config else min_r_sum / (max_r_sum + epsilon))
        feature_dict_final["BS_Ratio_CoupledSum_vs_Trigger"] = np.clip(np.nan_to_num(ratio_sum, nan=1.0), 0.0, 1.0)
    # --- End Process BS Features ---

    # --- BS Asymmetry Features (Scalar Calculation) ---
    bs_norm_vals_left_scalar = {}
    bs_norm_vals_right_scalar = {}

    # Calculate normalized values for both sides
    for au in local_trigger_aus + local_coupled_aus:
        bs_l_raw = get_float_value(f"BS_Left {au}")
        bs_r_raw = get_float_value(f"BS_Right {au}")
        bl_l_raw = get_float_value(f"BL_Left {au}")
        bl_r_raw = get_float_value(f"BL_Right {au}")

        if use_normalized:
            bs_norm_vals_left_scalar[au] = max(0.0, bs_l_raw - bl_l_raw)
            bs_norm_vals_right_scalar[au] = max(0.0, bs_r_raw - bl_r_raw)
        else:
            bs_norm_vals_left_scalar[au] = bs_l_raw
            bs_norm_vals_right_scalar[au] = bs_r_raw

    # Trigger Asymmetry
    trig_au = local_trigger_aus[0]
    left_trig_val = bs_norm_vals_left_scalar.get(trig_au, 0.0)
    right_trig_val = bs_norm_vals_right_scalar.get(trig_au, 0.0)
    min_trig = min(left_trig_val, right_trig_val); max_trig = max(left_trig_val, right_trig_val)
    ratio_trig = 1.0 if max_trig <= min_val_config else (0.0 if min_trig <= min_val_config else min_trig / (max_trig + epsilon))
    feature_dict_final[f"BS_Asym_Ratio_{trig_au}"] = np.clip(np.nan_to_num(ratio_trig, nan=1.0), 0.0, 1.0)
    diff_trig = abs(left_trig_val - right_trig_val); avg_trig = (left_trig_val + right_trig_val) / 2.0
    pdiff_trig = 0.0
    if avg_trig > min_val_config: pdiff_trig = (diff_trig / (avg_trig + epsilon)) * 100.0
    elif diff_trig > min_val_config: pdiff_trig = percent_diff_cap
    feature_dict_final[f"BS_Asym_PercDiff_{trig_au}"] = np.clip(np.nan_to_num(pdiff_trig, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)

    # Coupled AU Asymmetry & Sum Asymmetry
    bs_coupled_sum_left_scalar = 0.0
    bs_coupled_sum_right_scalar = 0.0
    for coup_au in local_coupled_aus:
        left_coup_val = bs_norm_vals_left_scalar.get(coup_au, 0.0)
        right_coup_val = bs_norm_vals_right_scalar.get(coup_au, 0.0)

        min_coup = min(left_coup_val, right_coup_val); max_coup = max(left_coup_val, right_coup_val)
        ratio_coup = 1.0 if max_coup <= min_val_config else (0.0 if min_coup <= min_val_config else min_coup / (max_coup + epsilon))
        feature_dict_final[f"BS_Asym_Ratio_{coup_au}"] = np.clip(np.nan_to_num(ratio_coup, nan=1.0), 0.0, 1.0)

        diff_coup = abs(left_coup_val - right_coup_val); avg_coup = (left_coup_val + right_coup_val) / 2.0
        pdiff_coup = 0.0
        if avg_coup > min_val_config: pdiff_coup = (diff_coup / (avg_coup + epsilon)) * 100.0
        elif diff_coup > min_val_config: pdiff_coup = percent_diff_cap
        feature_dict_final[f"BS_Asym_PercDiff_{coup_au}"] = np.clip(np.nan_to_num(pdiff_coup, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)

        bs_coupled_sum_left_scalar += left_coup_val
        bs_coupled_sum_right_scalar += right_coup_val

    # Coupled Sum Asymmetry
    min_sum = min(bs_coupled_sum_left_scalar, bs_coupled_sum_right_scalar); max_sum = max(bs_coupled_sum_left_scalar, bs_coupled_sum_right_scalar)
    ratio_sum = 1.0 if max_sum <= min_val_config else (0.0 if min_sum <= min_val_config else min_sum / (max_sum + epsilon))
    feature_dict_final["BS_Asym_Ratio_CoupledSum"] = np.clip(np.nan_to_num(ratio_sum, nan=1.0), 0.0, 1.0)
    diff_sum = abs(bs_coupled_sum_left_scalar - bs_coupled_sum_right_scalar); avg_sum = (bs_coupled_sum_left_scalar + bs_coupled_sum_right_scalar) / 2.0
    pdiff_sum = 0.0
    if avg_sum > min_val_config: pdiff_sum = (diff_sum / (avg_sum + epsilon)) * 100.0
    elif diff_sum > min_val_config: pdiff_sum = percent_diff_cap
    feature_dict_final["BS_Asym_PercDiff_CoupledSum"] = np.clip(np.nan_to_num(pdiff_sum, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)
    # --- End BS Asymmetry ---

    # Add side indicator
    feature_dict_final["side_indicator"] = 0 if side_label.lower() == 'left' else 1

    # --- Load the EXPECTED feature list ---
    feature_names_path = MODEL_FILENAMES.get('feature_list')
    if not feature_names_path or not os.path.exists(feature_names_path):
        local_logger.error(f"Snarl-Smile feature list not found: {feature_names_path}."); return None
    try:
        ordered_feature_names = joblib.load(feature_names_path)
        if not isinstance(ordered_feature_names, list): local_logger.error("Loaded feature names not a list."); return None
    except Exception as e: local_logger.error(f"Failed load Snarl-Smile feature list: {e}", exc_info=True); return None

    # --- Build final feature list IN ORDER ---
    feature_list = []; missing_in_dict = []; type_errors = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name)
        final_val = 0.0
        if value is None: missing_in_dict.append(name)
        else:
            try:
                temp_val = float(value)
                if np.isnan(temp_val) or np.isinf(temp_val): final_val = 0.0
                else: final_val = temp_val
            except (ValueError, TypeError): type_errors.append(name); final_val = 0.0
        feature_list.append(final_val)

    if missing_in_dict: local_logger.warning(f"Snarl-Smile Detect (BS-Only, {side_label}): {len(missing_in_dict)} expected features missing: {missing_in_dict[:5]}... Using 0.0.")
    if type_errors: local_logger.warning(f"Snarl-Smile Detect (BS-Only, {side_label}): {len(type_errors)} features had type errors: {type_errors[:5]}... Using 0.0.")

    if len(feature_list) != len(ordered_feature_names):
        local_logger.error(f"CRITICAL MISMATCH SnSm (BS-Only): Feature list length ({len(feature_list)}) != expected ({len(ordered_feature_names)}).")
        return None

    local_logger.debug(f"Generated {len(feature_list)} Snarl-Smile detection features for {side_label} (BS-Only).")
    return feature_list


# --- process_targets function (Identical binary mapping - Unchanged) ---
def process_targets(target_series):
    if target_series is None: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int).values