# hypertonicity_features.py
# Extracts features for detecting hypertonicity/dysfunctional patterns using BL (RAW) and BS (NORMALIZED) values.
# REVERTED to BL+BS structure. Added side_indicator.

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Import config for Hypertonicity
try:
    from hypertonicity_config import (
        FEATURE_CONFIG, HYPERTONICITY_ACTIONS, INTEREST_AUS, LOG_DIR,
        FEATURE_SELECTION, MODEL_FILENAMES, CLASS_NAMES
    )
    CONFIG_LOADED = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import from hypertonicity_config: {e}", exc_info=True)
    CONFIG_LOADED = False
    # Fallbacks (less critical now but good practice)
    LOG_DIR = 'logs'; MODEL_DIR = 'models/synkinesis/hypertonicity'
    MODEL_FILENAMES = {'feature_list': os.path.join(MODEL_DIR,'features.list'),
                       'importance_file': os.path.join(MODEL_DIR,'feature_importance.csv')}
    HYPERTONICITY_ACTIONS = ['BL', 'BS']; INTEREST_AUS = ['AU12_r', 'AU14_r', 'AU06_r', 'AU07_r']
    FEATURE_CONFIG = {'actions': HYPERTONICITY_ACTIONS, 'interest_aus': INTEREST_AUS,
                      'use_normalized': True, # Intention is True for BS
                      'min_value': 0.0001, 'percent_diff_cap': 200.0}
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 20, 'importance_file': MODEL_FILENAMES['importance_file']}
    CLASS_NAMES = {0: 'None', 1: 'Hypertonicity'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- prepare_data function (Mostly standard, added side_indicator) ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data for Hypertonicity detection training (BL+BS), including feature selection. """
    logger.info("Loading datasets for Hypertonicity Detection (BL+BS)...")
    try:
        results_df = pd.read_csv(results_file, low_memory=False)
        expert_df = pd.read_csv(expert_file)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise

    expert_rename_map = {
        'Patient': 'Patient ID',
        'Hypertonicity Left': 'Expert_Left_Hypertonicity',
        'Hypertonicity Right': 'Expert_Right_Hypertonicity'
    }
    cols_to_rename = {k: v for k, v in expert_rename_map.items() if k in expert_df.columns}
    if 'Patient' in expert_df.columns and 'Patient ID' not in cols_to_rename: cols_to_rename['Patient'] = 'Patient ID'
    expert_df = expert_df.rename(columns=cols_to_rename)
    logger.info(f"Renamed expert columns: {cols_to_rename}")

    # --- process_targets (Unchanged) ---
    def process_targets(target_series):
        if target_series is None: return np.array([], dtype=int)
        mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
        numeric_yes_values = { val: 1 for val in target_series.unique() if isinstance(val, (int, float)) and val > 0 }
        mapping.update(numeric_yes_values); numeric_no_values = { val: 0 for val in target_series.unique() if isinstance(val, (int, float)) and val == 0 }
        mapping.update(numeric_no_values); s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
        mapped = target_series.map(mapping); mapped_str = s_clean.map(mapping); mapped = mapped.fillna(mapped_str)
        unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
        if not unexpected.empty: logger.warning(f"Unexpected Hypertonicity expert labels treated as 'No': {unexpected.unique().tolist()}")
        final_mapped = mapped.fillna(0); return final_mapped.astype(int).values
    # --- End process_targets ---

    target_left_col = 'Expert_Left_Hypertonicity'; target_right_col = 'Expert_Right_Hypertonicity'
    expert_df['Target_Left_Hypertonicity'] = process_targets(expert_df.get(target_left_col))
    expert_df['Target_Right_Hypertonicity'] = process_targets(expert_df.get(target_right_col))
    logger.info(f"Counts in expert_df['Target_Left_Hypertonicity'] AFTER mapping: \n{pd.Series(expert_df.get('Target_Left_Hypertonicity', [])).value_counts(dropna=False)}")
    logger.info(f"Counts in expert_df['Target_Right_Hypertonicity'] AFTER mapping: \n{pd.Series(expert_df.get('Target_Right_Hypertonicity', [])).value_counts(dropna=False)}")

    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
    expert_cols_to_merge = ['Patient ID', 'Target_Left_Hypertonicity', 'Target_Right_Hypertonicity']
    expert_cols_exist = [col for col in expert_cols_to_merge if col in expert_df.columns]
    missing_expert_merge_cols = set(expert_cols_to_merge) - set(expert_cols_exist)
    if missing_expert_merge_cols: logger.error(f"Expert columns missing for merge: {missing_expert_merge_cols}. Cannot proceed."); return None, None

    try: merged_df = pd.merge(results_df, expert_df[expert_cols_exist], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data for Hypertonicity: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    targets_available = 'Target_Left_Hypertonicity' in merged_df.columns and 'Target_Right_Hypertonicity' in merged_df.columns
    if not targets_available: logger.error("Target columns missing after merge."); return None, None
    logger.info(f"Counts in merged_df['Target_Left_Hypertonicity'] AFTER merge: \n{merged_df['Target_Left_Hypertonicity'].value_counts(dropna=False)}")
    logger.info(f"Counts in merged_df['Target_Right_Hypertonicity'] AFTER merge: \n{merged_df['Target_Right_Hypertonicity'].value_counts(dropna=False)}")

    logger.info("Extracting Hypertonicity features for Left side (BL+BS)...")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info("Extracting Hypertonicity features for Right side (BL+BS)...")
    right_features_df = extract_features(merged_df, 'Right')

    if left_features_df is None or right_features_df is None: logger.error("Feature extraction failed."); return None, None

    left_targets = merged_df['Target_Left_Hypertonicity'].values; right_targets = merged_df['Target_Right_Hypertonicity'].values
    targets = np.concatenate([left_targets, right_targets])
    unique_final, counts_final = np.unique(targets, return_counts=True)
    final_class_dist = {CLASS_NAMES.get(i, f"Class_{i}"): int(c) for i, c in zip(unique_final, counts_final)}
    logger.info(f"FINAL Hypertonicity Class distribution input: {final_class_dist}")

    # --- Add side indicator ---
    left_features_df['side_indicator'] = 0
    right_features_df['side_indicator'] = 1
    # --- End add side indicator ---

    features = pd.concat([left_features_df, right_features_df], ignore_index=True)

    # --- NaN/Inf/Numeric Conversion ---
    features.replace([np.inf, -np.inf], np.nan, inplace=True); features = features.fillna(0)
    initial_cols = features.columns.tolist(); cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert col {col}. Marking drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)
    logger.info(f"Generated initial {features.shape[1]} Hypertonicity features (BL+BS).")

    # --- Feature Selection (Keep disabled for now) ---
    fs_enabled = isinstance(FEATURE_SELECTION, dict) and FEATURE_SELECTION.get('enabled', False)
    if fs_enabled:
        logger.warning("Hypertonicity feature selection is ENABLED - ensure importance file is relevant to BL+BS features.")
        n_top_features = FEATURE_SELECTION.get('top_n_features', 20)
        importance_file = FEATURE_SELECTION.get('importance_file')
        if not importance_file or not os.path.exists(importance_file): logger.warning(f"FS enabled, but Importance file not found: '{importance_file}'. Skipping FS.")
        else:
            try:
                importance_df = pd.read_csv(importance_file)
                if 'feature' not in importance_df.columns or importance_df.empty: logger.error("Importance file invalid. Skipping FS.")
                else:
                    top_feature_names = importance_df['feature'].head(n_top_features).tolist()
                    # Ensure side_indicator is kept if present
                    if 'side_indicator' in features.columns and 'side_indicator' not in top_feature_names:
                        logger.debug("Adding 'side_indicator' to selected features.")
                        top_feature_names.append('side_indicator')
                    cols_to_keep = [col for col in top_feature_names if col in features.columns]
                    missing = set(top_feature_names) - set(cols_to_keep)
                    if missing: logger.warning(f"Important Hypertonicity features missing from generated set: {missing}")
                    if not cols_to_keep: logger.error("No features left after filtering based on importance file. Skipping FS.")
                    else: logger.info(f"Selecting top {len(cols_to_keep)} Hypertonicity features based on importance file."); features = features[cols_to_keep]
            except Exception as e: logger.error(f"Error during Hypertonicity FS: {e}. Skipping.", exc_info=True)
    else:
        logger.info("Hypertonicity feature selection is disabled.")
    # --- End Feature Selection ---

    logger.info(f"Final Hypertonicity dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.warning(f"NaNs in FINAL Hypertonicity features. Filling 0."); features = features.fillna(0)

    # --- Save Final Feature List ---
    final_feature_names = features.columns.tolist()
    try:
        feature_list_path = MODEL_FILENAMES.get('feature_list')
        if feature_list_path: os.makedirs(os.path.dirname(feature_list_path), exist_ok=True); joblib.dump(final_feature_names, feature_list_path); logger.info(f"Saved final {len(final_feature_names)} Hypertonicity features list: {feature_list_path}")
        else: logger.error("Hypertonicity feature list path not defined.")
    except Exception as e: logger.error(f"Failed to save Hypertonicity feature names list: {e}", exc_info=True)
    # --- End Save Feature List ---

    if 'targets' not in locals(): logger.error("Targets array creation failed."); return None, None
    return features, targets


# --- Helper Functions (Unchanged) ---
def calculate_ratio(val1_series, val2_series):
    local_logger = logging.getLogger(__name__)
    min_val_config = FEATURE_CONFIG.get('min_value', 0.0001)
    epsilon = 1e-9
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    min_vals_np = np.minimum(v1, v2); max_vals_np = np.maximum(v1, v2)
    ratio = np.ones_like(v1, dtype=float)
    mask_max_pos = max_vals_np > min_val_config; mask_min_zero = min_vals_np <= min_val_config
    ratio[mask_max_pos & mask_min_zero] = 0.0
    valid_division_mask = mask_max_pos & ~mask_min_zero
    if np.any(valid_division_mask):
        ratio[valid_division_mask] = min_vals_np[valid_division_mask] / (max_vals_np[valid_division_mask] + epsilon)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0) # Handle potential 0/0 -> NaN
    ratio = np.clip(ratio, 0.0, 1.0) # Ensure ratio is between 0 and 1
    return pd.Series(ratio, index=val1_series.index)

def calculate_percent_diff(val1_series, val2_series):
    local_logger = logging.getLogger(__name__)
    min_val_config = FEATURE_CONFIG.get('min_value', 0.0001)
    percent_diff_cap = FEATURE_CONFIG.get('percent_diff_cap', 200.0)
    epsilon = 1e-9
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    abs_diff = np.abs(v1 - v2); avg = (v1 + v2) / 2.0
    percent_diff = np.zeros_like(avg, dtype=float)
    mask_avg_pos = avg > min_val_config
    if np.any(mask_avg_pos):
        division_result = abs_diff[mask_avg_pos] / (avg[mask_avg_pos] + epsilon)
        percent_diff[mask_avg_pos] = division_result * 100.0
    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > min_val_config)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap
    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    percent_diff = np.nan_to_num(percent_diff, nan=0.0, posinf=percent_diff_cap, neginf=0.0)
    return pd.Series(percent_diff, index=val1_series.index)
# --- End Helper Functions ---


# --- extract_features function (Training - Rewritten for BL+BS) ---
def extract_features(df, side):
    """ Extracts Hypertonicity features for TRAINING using BL (raw) and BS (normalized). """
    logger.debug(f"Extracting Hypertonicity features for {side} side (Training - BL Raw + BS Norm)...")
    feature_data = {}
    side_label = side.capitalize()

    # --- Use Config values safely ---
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED else {}
    local_actions = local_feature_config.get('actions', ['BL', 'BS'])
    local_interest_aus = local_feature_config.get('interest_aus', INTEREST_AUS)
    # use_normalized flag now means "use normalized for non-BL actions"
    use_normalized_config = local_feature_config.get('use_normalized', True)
    # --- End Config ---

    # Helper to get numeric series safely
    def get_numeric_series(col_name, default_val=0.0):
        series = df.get(col_name)
        if series is None:
            return pd.Series(default_val, index=df.index, dtype=float)
        numeric_series = pd.to_numeric(series, errors='coerce').fillna(default_val)
        return numeric_series.astype(float) # Ensure float type

    # Store BL raw values for normalization
    bl_raw_values = {}
    for au in local_interest_aus:
        bl_col_raw = f"BL_{side_label} {au}"
        bl_raw_values[au] = get_numeric_series(bl_col_raw)

    # Process features for each action (BL and BS)
    bs_norm_values = {} # Store BS normalized values for asymmetry calc
    for action in local_actions:
        action_prefix = action
        is_baseline = (action == 'BL')

        for au in local_interest_aus:
            col_raw = f"{action_prefix}_{side_label} {au}"
            raw_series = get_numeric_series(col_raw)

            if is_baseline:
                # Feature 1: Raw BL value
                feature_data[f"BL_{au}_raw"] = raw_series
                # Store for asymmetry later
                # bl_raw_values[au] is already stored
            else: # Process BS action
                # Feature 2: Normalized BS value (BS_raw - BL_raw), clipped at 0
                bl_raw_series = bl_raw_values.get(au, pd.Series(0.0, index=df.index))
                norm_series = (raw_series - bl_raw_series).clip(lower=0)
                if use_normalized_config:
                    feature_data[f"BS_{au}_norm"] = norm_series
                    bs_norm_values[au] = norm_series # Store for asymmetry
                else: # If config is false, just use raw BS
                    feature_data[f"BS_{au}_raw"] = raw_series
                    # For asymmetry, we'd need raw BS values if not using norm
                    # This logic assumes use_normalized=True as per goal
                    # If needed, store raw_series in a bs_raw_values dict

                # Feature 3: Ratio of BL raw vs BS raw (how much resting tone relative to peak)
                feature_data[f"Ratio_BLraw_vs_BSraw_{au}"] = calculate_ratio(bl_raw_series, raw_series)

    # Feature 4: Asymmetry Features for BL (Raw)
    for au in local_interest_aus:
        bl_left_col = f"BL_Left {au}"
        bl_right_col = f"BL_Right {au}"
        bl_left_raw = get_numeric_series(bl_left_col)
        bl_right_raw = get_numeric_series(bl_right_col)
        feature_data[f"BL_Asym_Ratio_{au}_raw"] = calculate_ratio(bl_left_raw, bl_right_raw)
        feature_data[f"BL_Asym_PercDiff_{au}_raw"] = calculate_percent_diff(bl_left_raw, bl_right_raw)

    # Feature 5: Asymmetry Features for BS (Normalized, if config is True)
    if use_normalized_config:
        for au in local_interest_aus:
            # We need the normalized values for both Left and Right
            # Re-calculate normalized values for the *opposite* side needed for comparison
            opposite_side = 'Right' if side_label == 'Left' else 'Left'
            bs_left_raw = get_numeric_series(f"BS_Left {au}")
            bs_right_raw = get_numeric_series(f"BS_Right {au}")
            bl_left_raw_for_bs = get_numeric_series(f"BL_Left {au}")
            bl_right_raw_for_bs = get_numeric_series(f"BL_Right {au}")

            bs_left_norm = (bs_left_raw - bl_left_raw_for_bs).clip(lower=0)
            bs_right_norm = (bs_right_raw - bl_right_raw_for_bs).clip(lower=0)

            feature_data[f"BS_Asym_Ratio_{au}_norm"] = calculate_ratio(bs_left_norm, bs_right_norm)
            feature_data[f"BS_Asym_PercDiff_{au}_norm"] = calculate_percent_diff(bs_left_norm, bs_right_norm)
    # else: Could add BS raw asymmetry here if needed

    # Create DataFrame
    features_df = pd.DataFrame(feature_data, index=df.index)

    # Final check for numeric types and NaNs/Infs
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    inf_cap_high = 1e9 # A large number to replace inf
    inf_cap_low = -1e9
    features_df.replace([np.inf], inf_cap_high, inplace=True)
    features_df.replace([-np.inf], inf_cap_low, inplace=True)
    features_df = features_df.fillna(0.0)

    logger.debug(f"Generated {features_df.shape[1]} Hypertonicity features for {side_label} (Training BL+BS).")
    return features_df


# --- extract_features_for_detection (Detection - Rewritten for BL+BS) ---
def extract_features_for_detection(row_data, side):
    """ Extracts Hypertonicity features for detection using BL (raw) and BS (normalized). """
    try:
        # Need to re-import config within function scope if not guaranteed globally
        from hypertonicity_config import (FEATURE_CONFIG, HYPERTONICITY_ACTIONS,
                                          INTEREST_AUS, MODEL_FILENAMES)
        local_logger = logging.getLogger(__name__)
        CONFIG_LOADED_DETECT = True
    except ImportError:
        logging.error("Failed config import within hypertonicity extract_features_for_detection (BL+BS)."); return None
        # Provide minimal fallbacks if needed for standalone testing
        FEATURE_CONFIG = {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
        HYPERTONICITY_ACTIONS = ['BL', 'BS']; INTEREST_AUS = ['AU12_r', 'AU14_r', 'AU06_r', 'AU07_r']
        MODEL_FILENAMES = {'feature_list': 'models/synkinesis/hypertonicity/features.list'}
        local_logger = logging
        CONFIG_LOADED_DETECT = False

    if not isinstance(row_data, (pd.Series, dict)): local_logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    pid_for_log = row_series.get('Patient ID', 'UnknownPID')

    if side not in ['Left', 'Right']: local_logger.error(f"Invalid 'side': {side}"); return None
    else: side_label = side

    # --- Use Config values safely ---
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED_DETECT else {}
    local_actions = local_feature_config.get('actions', ['BL', 'BS'])
    local_interest_aus = local_feature_config.get('interest_aus', INTEREST_AUS)
    use_normalized_config = local_feature_config.get('use_normalized', True)
    min_val_config = local_feature_config.get('min_value', 0.0001)
    percent_diff_cap = local_feature_config.get('percent_diff_cap', 200.0)
    epsilon = 1e-9
    # --- End Config ---

    local_logger.debug(f"HyperT Detect BL+BS ({pid_for_log}, {side_label}) - ENTERING feature extraction.")
    feature_dict_final = {}

    # Helper to safely get float from row_series
    def get_float_value(key, default=0.0):
        val = row_series.get(key, default)
        try:
             f_val = float(val)
             if np.isnan(f_val) or np.isinf(f_val): return default
             return f_val
        except (ValueError, TypeError): return default

    # Store BL raw values for normalization
    bl_raw_values_scalar = {}
    for au in local_interest_aus:
        bl_col_raw = f"BL_{side_label} {au}"
        bl_raw_values_scalar[au] = get_float_value(bl_col_raw)

    # Process features for each action (BL and BS)
    bs_norm_values_scalar = {} # Store BS normalized values for asymmetry calc
    for action in local_actions:
        action_prefix = action
        is_baseline = (action == 'BL')

        for au in local_interest_aus:
            col_raw = f"{action_prefix}_{side_label} {au}"
            raw_val = get_float_value(col_raw)

            if is_baseline:
                # Feature 1: Raw BL value
                feature_dict_final[f"BL_{au}_raw"] = raw_val
            else: # Process BS action
                bl_raw_val = bl_raw_values_scalar.get(au, 0.0)
                norm_val = max(0.0, raw_val - bl_raw_val) # Clip at 0

                if use_normalized_config:
                    # Feature 2: Normalized BS value
                    feature_dict_final[f"BS_{au}_norm"] = norm_val
                    bs_norm_values_scalar[au] = norm_val # Store for asymmetry
                else:
                    feature_dict_final[f"BS_{au}_raw"] = raw_val
                    # bs_raw_values_scalar[au] = raw_val # Store if needed for raw asymmetry

                # Feature 3: Ratio of BL raw vs BS raw
                min_r = min(bl_raw_val, raw_val); max_r = max(bl_raw_val, raw_val)
                ratio_blbs = 1.0 if max_r <= min_val_config else (0.0 if min_r <= min_val_config else min_r / (max_r + epsilon))
                feature_dict_final[f"Ratio_BLraw_vs_BSraw_{au}"] = np.clip(np.nan_to_num(ratio_blbs, nan=1.0), 0.0, 1.0)

    # Feature 4: Asymmetry Features for BL (Raw)
    for au in local_interest_aus:
        bl_left_val = get_float_value(f"BL_Left {au}")
        bl_right_val = get_float_value(f"BL_Right {au}")

        min_bl = min(bl_left_val, bl_right_val); max_bl = max(bl_left_val, bl_right_val)
        ratio_bl = 1.0 if max_bl <= min_val_config else (0.0 if min_bl <= min_val_config else min_bl / (max_bl + epsilon))
        feature_dict_final[f"BL_Asym_Ratio_{au}_raw"] = np.clip(np.nan_to_num(ratio_bl, nan=1.0), 0.0, 1.0)

        diff_bl = abs(bl_left_val - bl_right_val); avg_bl = (bl_left_val + bl_right_val) / 2.0
        pdiff_bl = 0.0
        if avg_bl > min_val_config: pdiff_bl = (diff_bl / (avg_bl + epsilon)) * 100.0
        elif diff_bl > min_val_config: pdiff_bl = percent_diff_cap
        feature_dict_final[f"BL_Asym_PercDiff_{au}_raw"] = np.clip(np.nan_to_num(pdiff_bl, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)

    # Feature 5: Asymmetry Features for BS (Normalized, if config is True)
    if use_normalized_config:
        for au in local_interest_aus:
            # Need normalized values for both sides
            bs_left_raw_val = get_float_value(f"BS_Left {au}")
            bs_right_raw_val = get_float_value(f"BS_Right {au}")
            bl_left_raw_val_for_bs = get_float_value(f"BL_Left {au}")
            bl_right_raw_val_for_bs = get_float_value(f"BL_Right {au}")

            bs_left_norm_val = max(0.0, bs_left_raw_val - bl_left_raw_val_for_bs)
            bs_right_norm_val = max(0.0, bs_right_raw_val - bl_right_raw_val_for_bs)

            min_bs = min(bs_left_norm_val, bs_right_norm_val); max_bs = max(bs_left_norm_val, bs_right_norm_val)
            ratio_bs = 1.0 if max_bs <= min_val_config else (0.0 if min_bs <= min_val_config else min_bs / (max_bs + epsilon))
            feature_dict_final[f"BS_Asym_Ratio_{au}_norm"] = np.clip(np.nan_to_num(ratio_bs, nan=1.0), 0.0, 1.0)

            diff_bs = abs(bs_left_norm_val - bs_right_norm_val); avg_bs = (bs_left_norm_val + bs_right_norm_val) / 2.0
            pdiff_bs = 0.0
            if avg_bs > min_val_config: pdiff_bs = (diff_bs / (avg_bs + epsilon)) * 100.0
            elif diff_bs > min_val_config: pdiff_bs = percent_diff_cap
            feature_dict_final[f"BS_Asym_PercDiff_{au}_norm"] = np.clip(np.nan_to_num(pdiff_bs, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)

    # --- Add Side Indicator ---
    feature_dict_final["side_indicator"] = 0 if side_label.lower() == 'left' else 1
    # --- End Add Side Indicator ---

    # --- Load the EXPECTED feature list ---
    feature_names_path = MODEL_FILENAMES.get('feature_list')
    if not feature_names_path or not os.path.exists(feature_names_path): local_logger.error(f"Hypertonicity feature list not found: {feature_names_path}. Cannot order features for detection."); return None
    try:
        ordered_feature_names = joblib.load(feature_names_path);
        if not isinstance(ordered_feature_names, list): local_logger.error("Loaded feature names not a list."); return None
    except Exception as e: local_logger.error(f"Failed load Hypertonicity feature list: {e}", exc_info=True); return None

    # --- Build final feature list IN ORDER ---
    feature_list = []; missing = []; type_err = []; nan_inf_final = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name); final_val = 0.0
        if value is None:
            missing.append(name)
        else:
            try:
                temp_val = float(value);
                if np.isnan(temp_val) or np.isinf(temp_val):
                     final_val = 0.0; nan_inf_final.append(name)
                else: final_val = temp_val
            except (ValueError, TypeError):
                 type_err.append(name); final_val = 0.0
        feature_list.append(final_val)

    if missing: local_logger.warning(f"HyperT Detect BL+BS ({pid_for_log}, {side_label}): {len(missing)} missing features from expected list: {missing[:5]}... Used 0.0.")
    if type_err: local_logger.warning(f"HyperT Detect BL+BS ({pid_for_log}, {side_label}): {len(type_err)} features had type errors: {type_err[:5]}... Used 0.0.")

    if len(feature_list) != len(ordered_feature_names):
        local_logger.error(f"CRITICAL MISMATCH HyperT BL+BS ({pid_for_log}, {side_label}): Generated feature list length ({len(feature_list)}) != expected ({len(ordered_feature_names)}) based on {feature_names_path}. Returning None.")
        return None

    local_logger.debug(f"HyperT Detect BL+BS ({pid_for_log}, {side_label}) - Final generated feature_list (len {len(feature_list)}): {feature_list[:10]}...") # Log first 10
    return feature_list

# --- process_targets function (Copied - Unchanged) ---
def process_targets(target_series):
    if target_series is None: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    numeric_yes_values = { val: 1 for val in target_series.unique() if isinstance(val, (int, float)) and val > 0 }
    mapping.update(numeric_yes_values); numeric_no_values = { val: 0 for val in target_series.unique() if isinstance(val, (int, float)) and val == 0 }
    mapping.update(numeric_no_values); s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = target_series.map(mapping); mapped_str = s_clean.map(mapping); mapped = mapped.fillna(mapped_str)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0); return final_mapped.astype(int).values