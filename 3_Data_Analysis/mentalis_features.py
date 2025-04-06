# mentalis_features.py
# Extracts features for detecting Mentalis Synkinesis using NORMALIZED AU values.
# FINAL CONFIG (Pending Error Analysis): Full Features (BS+SE+Context), SMOTE Enabled.

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Import config for Mentalis Synkinesis
try:
    from mentalis_config import (
        FEATURE_CONFIG, MENTALIS_ACTIONS, COUPLED_AUS, CONTEXT_AUS, LOG_DIR, # CONTEXT_AUS is used
        FEATURE_SELECTION, MODEL_FILENAMES, CLASS_NAMES
    )
    CONFIG_LOADED = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import from mentalis_config: {e}", exc_info=True)
    CONFIG_LOADED = False
    # Fallbacks
    LOG_DIR = 'logs'; MODEL_DIR = 'models/synkinesis/mentalis'
    MODEL_FILENAMES = {'feature_list': os.path.join(MODEL_DIR,'features.list'),
                       'importance_file': os.path.join(MODEL_DIR,'feature_importance.csv')}
    MENTALIS_ACTIONS = ['BS', 'SE']; COUPLED_AUS = ['AU17_r']; CONTEXT_AUS = ['AU12_r', 'AU15_r', 'AU16_r'] # <<< CONTEXT_AUS included >>>
    FEATURE_CONFIG = {'actions': MENTALIS_ACTIONS, 'coupled_aus': COUPLED_AUS, 'context_aus': CONTEXT_AUS, # <<< CONTEXT_AUS included >>>
                      'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 15, 'importance_file': MODEL_FILENAMES['importance_file']}
    CLASS_NAMES = {0: 'None', 1: 'Mentalis Synkinesis'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- prepare_data function (Unchanged logic, calls full extract_features) ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data for Mentalis Synkinesis detection training (Full Features). """
    logger.info("Loading datasets for Mentalis Synkinesis Detection (Full Features)...")
    # ... (rest of loading, renaming, target processing is identical to previous version) ...
    try:
        results_df = pd.read_csv(results_file, low_memory=False)
        expert_df = pd.read_csv(expert_file)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise

    expert_rename_map = {
        'Patient': 'Patient ID',
        'Mentalis Synkinesis Left': 'Expert_Left_Mentalis',
        'Mentalis Synkinesis Right': 'Expert_Right_Mentalis'
    }
    cols_to_rename = {k: v for k, v in expert_rename_map.items() if k in expert_df.columns}
    if 'Patient' in expert_df.columns and 'Patient ID' not in cols_to_rename: cols_to_rename['Patient'] = 'Patient ID'
    expert_df = expert_df.rename(columns=cols_to_rename)
    logger.info(f"Renamed expert columns: {cols_to_rename}")

    def process_targets(target_series):
        if target_series is None: return np.array([], dtype=int)
        mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
        numeric_yes_values = { val: 1 for val in target_series.unique() if isinstance(val, (int, float)) and val > 0 }
        mapping.update(numeric_yes_values); numeric_no_values = { val: 0 for val in target_series.unique() if isinstance(val, (int, float)) and val == 0 }
        mapping.update(numeric_no_values); s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
        mapped = target_series.map(mapping); mapped_str = s_clean.map(mapping); mapped = mapped.fillna(mapped_str)
        unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
        if not unexpected.empty: logger.warning(f"Unexpected Mentalis expert labels treated as 'No': {unexpected.unique().tolist()}")
        final_mapped = mapped.fillna(0); return final_mapped.astype(int).values

    target_left_col = 'Expert_Left_Mentalis'; target_right_col = 'Expert_Right_Mentalis'
    expert_df['Target_Left_Mentalis'] = process_targets(expert_df.get(target_left_col))
    expert_df['Target_Right_Mentalis'] = process_targets(expert_df.get(target_right_col))

    logger.info(f"Counts in expert_df['Target_Left_Mentalis'] AFTER mapping: \n{pd.Series(expert_df.get('Target_Left_Mentalis', [])).value_counts(dropna=False)}")
    logger.info(f"Counts in expert_df['Target_Right_Mentalis'] AFTER mapping: \n{pd.Series(expert_df.get('Target_Right_Mentalis', [])).value_counts(dropna=False)}")

    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
    expert_cols_to_merge = ['Patient ID', 'Target_Left_Mentalis', 'Target_Right_Mentalis']
    expert_cols_exist = [col for col in expert_cols_to_merge if col in expert_df.columns]
    missing_expert_merge_cols = set(expert_cols_to_merge) - set(expert_cols_exist)
    if missing_expert_merge_cols: logger.error(f"Expert columns missing for merge: {missing_expert_merge_cols}. Cannot proceed."); return None, None

    try: merged_df = pd.merge(results_df, expert_df[expert_cols_exist], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data for Mentalis: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    targets_available = 'Target_Left_Mentalis' in merged_df.columns and 'Target_Right_Mentalis' in merged_df.columns
    if not targets_available: logger.error("Target columns missing after merge."); return None, None
    logger.info(f"Counts in merged_df['Target_Left_Mentalis'] AFTER merge: \n{merged_df['Target_Left_Mentalis'].value_counts(dropna=False)}")
    logger.info(f"Counts in merged_df['Target_Right_Mentalis'] AFTER merge: \n{merged_df['Target_Right_Mentalis'].value_counts(dropna=False)}")

    logger.info("Extracting Mentalis features for Left side (Full)...")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info("Extracting Mentalis features for Right side (Full)...")
    right_features_df = extract_features(merged_df, 'Right')

    if left_features_df is None or right_features_df is None: logger.error("Feature extraction failed."); return None, None

    left_targets = merged_df['Target_Left_Mentalis'].values; right_targets = merged_df['Target_Right_Mentalis'].values
    targets = np.concatenate([left_targets, right_targets])
    unique_final, counts_final = np.unique(targets, return_counts=True)
    final_class_dist = {CLASS_NAMES.get(i, f"Class_{i}"): int(c) for i, c in zip(unique_final, counts_final)}
    logger.info(f"FINAL Mentalis Class distribution input: {final_class_dist}")

    # Add side indicator
    left_features_df['side_indicator'] = 0
    right_features_df['side_indicator'] = 1

    features = pd.concat([left_features_df, right_features_df], ignore_index=True)

    features.replace([np.inf, -np.inf], np.nan, inplace=True); features = features.fillna(0)
    initial_cols = features.columns.tolist(); cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert col {col}. Marking drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)
    logger.info(f"Generated initial {features.shape[1]} Mentalis features (Full).") # Should be 21 + side_indicator = 22

    # Feature Selection (Keep disabled)
    fs_enabled = isinstance(FEATURE_SELECTION, dict) and FEATURE_SELECTION.get('enabled', False)
    if fs_enabled: logger.warning("Mentalis FS enabled but recommend keeping disabled initially.")
    else: logger.info("Mentalis feature selection is disabled.")

    logger.info(f"Final Mentalis dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.warning(f"NaNs in FINAL Mentalis features. Filling 0."); features = features.fillna(0)

    final_feature_names = features.columns.tolist()
    try:
        feature_list_path = MODEL_FILENAMES.get('feature_list')
        if feature_list_path: os.makedirs(os.path.dirname(feature_list_path), exist_ok=True); joblib.dump(final_feature_names, feature_list_path); logger.info(f"Saved final {len(final_feature_names)} Mentalis features list: {feature_list_path}")
        else: logger.error("Mentalis feature list path not defined.")
    except Exception as e: logger.error(f"Failed to save Mentalis feature names list: {e}", exc_info=True)

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
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    ratio = np.clip(ratio, 0.0, 1.0)
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
        with np.errstate(divide='ignore', invalid='ignore'):
             division_result = abs_diff[mask_avg_pos] / (avg[mask_avg_pos] + epsilon)
        percent_diff[mask_avg_pos] = division_result * 100.0
    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > min_val_config)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap
    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    percent_diff = np.nan_to_num(percent_diff, nan=0.0, posinf=percent_diff_cap, neginf=0.0)
    return pd.Series(percent_diff, index=val1_series.index)
# --- End Helper Functions ---


# --- extract_features function (Training - Full features including Context) ---
def extract_features(df, side):
    """ Extracts Full Mentalis Synkinesis features for TRAINING using NORMALIZED values """
    logger.debug(f"Extracting Full Mentalis Synkinesis features for {side} side (Training)...")
    feature_data = {}
    side_label = side.capitalize()

    # --- Use Config values safely ---
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED else {}
    local_actions = local_feature_config.get('actions', ['BS', 'SE'])
    local_coupled_aus = local_feature_config.get('coupled_aus', ['AU17_r'])
    local_context_aus = local_feature_config.get('context_aus', ['AU12_r', 'AU15_r', 'AU16_r']) # Context AUs included
    use_normalized = local_feature_config.get('use_normalized', True)
    # --- End Config ---

    # Helper to get numeric series safely
    def get_numeric_series(col_name, default_val=0.0):
        series = df.get(col_name)
        if series is None:
            return pd.Series(default_val, index=df.index, dtype=float)
        numeric_series = pd.to_numeric(series, errors='coerce').fillna(default_val)
        return numeric_series.astype(float)

    if not local_coupled_aus: logger.error("COUPLED_AUS empty."); return None
    if not local_context_aus: logger.warning("CONTEXT_AUS empty, ratio features will be skipped.") # Warn if context missing
    coup_au = local_coupled_aus[0] # AU17_r

    # --- Get Baseline Raw Values ---
    bl_raw_values = {}
    all_aus_for_bl = local_coupled_aus + local_context_aus
    for au in all_aus_for_bl:
        bl_col_raw = f"BL_{side_label} {au}"
        bl_raw_values[au] = get_numeric_series(bl_col_raw)
    # --- End Baseline ---

    coupled_series_list = [] # To collect AU17 series for summary features
    action_data_cache = {} # Cache series for ratio calculation

    # 1. Coupled AU (AU17) & Context AU Strength (Side-Specific, Per Action, Normalized)
    for action in local_actions: # BS, SE
        action_cache = {}
        bl_coup_series = bl_raw_values.get(coup_au, pd.Series(0.0, index=df.index))
        raw_coup_series = get_numeric_series(f"{action}_{side_label} {coup_au}")
        final_series_coup = raw_coup_series # Default to raw if not normalizing
        if use_normalized:
            final_series_coup = (raw_coup_series - bl_coup_series).clip(lower=0)
            feature_data[f"{action}_{coup_au}_norm"] = final_series_coup
        else:
            feature_data[f"{action}_{coup_au}_raw"] = final_series_coup

        coupled_series_list.append(final_series_coup)
        action_cache[coup_au] = final_series_coup # Store the potentially normalized series

        # Context AUs (Normalized)
        for context_au in local_context_aus:
            bl_ctx_series = bl_raw_values.get(context_au, pd.Series(0.0, index=df.index))
            raw_ctx_series = get_numeric_series(f"{action}_{side_label} {context_au}")
            final_series_ctx = raw_ctx_series
            if use_normalized:
                final_series_ctx = (raw_ctx_series - bl_ctx_series).clip(lower=0)
                feature_data[f"{action}_{context_au}_context_norm"] = final_series_ctx
            else:
                 feature_data[f"{action}_{context_au}_context_raw"] = final_series_ctx
            action_cache[context_au] = final_series_ctx # Store potentially normalized context AU

        action_data_cache[action] = action_cache

    # 2. Asymmetry Features for AU17 (Cross-Side Comparison, Per Action, Normalized)
    bs_norm_left = pd.Series(0.0, index=df.index); bs_norm_right = pd.Series(0.0, index=df.index)
    se_norm_left = pd.Series(0.0, index=df.index); se_norm_right = pd.Series(0.0, index=df.index)
    if use_normalized:
        bl_l_raw = get_numeric_series(f"BL_Left {coup_au}")
        bl_r_raw = get_numeric_series(f"BL_Right {coup_au}")
        bs_l_raw = get_numeric_series(f"BS_Left {coup_au}")
        bs_r_raw = get_numeric_series(f"BS_Right {coup_au}")
        se_l_raw = get_numeric_series(f"SE_Left {coup_au}")
        se_r_raw = get_numeric_series(f"SE_Right {coup_au}")
        bs_norm_left = (bs_l_raw - bl_l_raw).clip(lower=0)
        bs_norm_right = (bs_r_raw - bl_r_raw).clip(lower=0)
        se_norm_left = (se_l_raw - bl_l_raw).clip(lower=0)
        se_norm_right = (se_r_raw - bl_r_raw).clip(lower=0)
    # If not use_normalized, asymmetries will be calculated on zeros (effectively 1 and 0)

    feature_data[f"BS_Asym_Ratio_{coup_au}"] = calculate_ratio(bs_norm_left, bs_norm_right)
    feature_data[f"BS_Asym_PercDiff_{coup_au}"] = calculate_percent_diff(bs_norm_left, bs_norm_right)
    feature_data[f"SE_Asym_Ratio_{coup_au}"] = calculate_ratio(se_norm_left, se_norm_right)
    feature_data[f"SE_Asym_PercDiff_{coup_au}"] = calculate_percent_diff(se_norm_left, se_norm_right)

    # 3. Ratio Features (AU17 vs Context AUs, Side-Specific, Per Action, Normalized)
    if use_normalized and local_context_aus: # Only calculate if using normalized and context AUs exist
        for action in local_actions: # BS, SE
            if action in action_data_cache:
                action_cache = action_data_cache[action]
                au17_series = action_cache.get(coup_au) # Should be normalized AU17
                if au17_series is not None:
                    for context_au in local_context_aus:
                        context_series = action_cache.get(context_au) # Should be normalized context
                        if context_series is not None:
                             feature_name_ratio = f"{action}_Ratio_{coup_au}_vs_{context_au}"
                             feature_data[feature_name_ratio] = calculate_ratio(au17_series, context_series)

    # 4. Summary Features Across Actions (BS, SE) for AU17 (Normalized)
    if coupled_series_list and len(local_actions) > 1: # Check if list is populated and more than one action
        # Ensure series are numeric before concat
        valid_coupled_series = [pd.to_numeric(s, errors='coerce') for s in coupled_series_list if isinstance(s, pd.Series)]
        if valid_coupled_series:
             coup_df = pd.concat(valid_coupled_series, axis=1).fillna(0.0)
             feature_data[f"Avg_{coup_au}_AcrossActions"] = coup_df.mean(axis=1)
             feature_data[f"Max_{coup_au}_AcrossActions"] = coup_df.max(axis=1)
             feature_data[f"Std_{coup_au}_AcrossActions"] = coup_df.std(axis=1).fillna(0)
        else: # Fallback if conversion failed
            logger.warning(f"Could not calculate summary features for {coup_au}")
            feature_data[f"Avg_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
            feature_data[f"Max_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
            feature_data[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)

    elif len(local_actions) == 1 and coupled_series_list: # Handle single action case
         feature_data[f"Avg_{coup_au}_AcrossActions"] = coupled_series_list[0]
         feature_data[f"Max_{coup_au}_AcrossActions"] = coupled_series_list[0]
         feature_data[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
    else: # Fallback if no coupled series
        feature_data[f"Avg_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        feature_data[f"Max_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        feature_data[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)


    # Create DataFrame
    features_df = pd.DataFrame(feature_data, index=df.index)

    # Final check
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    features_df.replace([np.inf, -np.inf], 0.0, inplace=True)

    logger.debug(f"Generated {features_df.shape[1]} Full Mentalis features for {side_label} (Training).")
    return features_df


# --- extract_features_for_detection (Detection - Full Features) ---
def extract_features_for_detection(row_data, side):
    """ Extracts Full Mentalis Synkinesis features for detection using NORMALIZED values """
    try:
        # Re-import needed configs
        from mentalis_config import (FEATURE_CONFIG, MENTALIS_ACTIONS, COUPLED_AUS,
                                      CONTEXT_AUS, MODEL_FILENAMES)
        local_logger = logging.getLogger(__name__)
        CONFIG_LOADED_DETECT = True
    except ImportError:
        logging.error("Failed config import within mentalis extract_features_for_detection (Full)."); return None
        CONFIG_LOADED_DETECT = False
        # Fallbacks
        FEATURE_CONFIG = {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
        MENTALIS_ACTIONS = ['BS', 'SE']; COUPLED_AUS = ['AU17_r']; CONTEXT_AUS = ['AU12_r', 'AU15_r', 'AU16_r']
        MODEL_FILENAMES = {'feature_list': 'models/synkinesis/mentalis/features.list'}
        local_logger = logging

    if not isinstance(row_data, (pd.Series, dict)): local_logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    pid_for_log = row_series.get('Patient ID', 'UnknownPID')

    if side not in ['Left', 'Right']: local_logger.error(f"Invalid 'side': {side}"); return None
    else: side_label = side

    # --- Configs ---
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED_DETECT else {}
    local_actions = local_feature_config.get('actions', ['BS', 'SE'])
    local_coupled_aus = local_feature_config.get('coupled_aus', ['AU17_r'])
    local_context_aus = local_feature_config.get('context_aus', ['AU12_r', 'AU15_r', 'AU16_r'])
    use_normalized = local_feature_config.get('use_normalized', True)
    min_val_config = local_feature_config.get('min_value', 0.0001)
    percent_diff_cap = local_feature_config.get('percent_diff_cap', 200.0)
    epsilon = 1e-9
    # --- End Configs ---

    if not local_coupled_aus: logger.error("Coupled AU list empty."); return None
    coup_au = local_coupled_aus[0]

    local_logger.debug(f"Extracting Full Mentalis detection features for {side_label}...")
    feature_dict_final = {}

    # Helper to safely get float
    def get_float_value(key, default=0.0):
        val = row_series.get(key, default)
        try:
             f_val = float(val)
             if np.isnan(f_val) or np.isinf(f_val): return default
             return f_val
        except (ValueError, TypeError): return default

    # --- Get Baseline Raw Values ---
    bl_raw_values_scalar = {}
    all_aus_for_bl = local_coupled_aus + local_context_aus
    for au in all_aus_for_bl:
        bl_col_raw = f"BL_{side_label} {au}"
        bl_raw_values_scalar[au] = get_float_value(bl_col_raw)
    # --- End Baseline ---

    coupled_values_list = [] # For summary stats
    action_data_cache_scalar = {}

    # 1. Coupled & Context AU Features (Scalar)
    for action in local_actions:
        action_cache_scalar = {}
        bl_coup_val = bl_raw_values_scalar.get(coup_au, 0.0)
        raw_coup_val = get_float_value(f"{action}_{side_label} {coup_au}")
        final_val_coup = raw_coup_val
        if use_normalized:
            final_val_coup = max(0.0, raw_coup_val - bl_coup_val)
            feature_dict_final[f"{action}_{coup_au}_norm"] = final_val_coup
        else:
            feature_dict_final[f"{action}_{coup_au}_raw"] = final_val_coup

        coupled_values_list.append(final_val_coup)
        action_cache_scalar[coup_au] = final_val_coup

        # Context AUs
        for context_au in local_context_aus:
            bl_ctx_val = bl_raw_values_scalar.get(context_au, 0.0)
            raw_ctx_val = get_float_value(f"{action}_{side_label} {context_au}")
            final_val_ctx = raw_ctx_val
            if use_normalized:
                final_val_ctx = max(0.0, raw_ctx_val - bl_ctx_val)
                feature_dict_final[f"{action}_{context_au}_context_norm"] = final_val_ctx
            else:
                feature_dict_final[f"{action}_{context_au}_context_raw"] = final_val_ctx
            action_cache_scalar[context_au] = final_val_ctx
        action_data_cache_scalar[action] = action_cache_scalar

    # 2. Asymmetry Features for AU17 (Normalized if use_normalized=True)
    bs_norm_left = 0.0; bs_norm_right = 0.0
    se_norm_left = 0.0; se_norm_right = 0.0
    if use_normalized:
        bl_l_raw = get_float_value(f"BL_Left {coup_au}")
        bl_r_raw = get_float_value(f"BL_Right {coup_au}")
        bs_l_raw = get_float_value(f"BS_Left {coup_au}")
        bs_r_raw = get_float_value(f"BS_Right {coup_au}")
        se_l_raw = get_float_value(f"SE_Left {coup_au}")
        se_r_raw = get_float_value(f"SE_Right {coup_au}")
        bs_norm_left = max(0.0, bs_l_raw - bl_l_raw)
        bs_norm_right = max(0.0, bs_r_raw - bl_r_raw)
        se_norm_left = max(0.0, se_l_raw - bl_l_raw)
        se_norm_right = max(0.0, se_r_raw - bl_r_raw)

    # BS Asymmetry
    min_bs = min(bs_norm_left, bs_norm_right); max_bs = max(bs_norm_left, bs_norm_right)
    ratio_bs = 1.0 if max_bs <= min_val_config else (0.0 if min_bs <= min_val_config else min_bs / (max_bs + epsilon))
    feature_dict_final[f"BS_Asym_Ratio_{coup_au}"] = np.clip(np.nan_to_num(ratio_bs, nan=1.0), 0.0, 1.0)
    diff_bs = abs(bs_norm_left - bs_norm_right); avg_bs = (bs_norm_left + bs_norm_right) / 2.0
    pdiff_bs = 0.0
    if avg_bs > min_val_config: pdiff_bs = (diff_bs / (avg_bs + epsilon)) * 100.0
    elif diff_bs > min_val_config: pdiff_bs = percent_diff_cap
    feature_dict_final[f"BS_Asym_PercDiff_{coup_au}"] = np.clip(np.nan_to_num(pdiff_bs, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)

    # SE Asymmetry
    min_se = min(se_norm_left, se_norm_right); max_se = max(se_norm_left, se_norm_right)
    ratio_se = 1.0 if max_se <= min_val_config else (0.0 if min_se <= min_val_config else min_se / (max_se + epsilon))
    feature_dict_final[f"SE_Asym_Ratio_{coup_au}"] = np.clip(np.nan_to_num(ratio_se, nan=1.0), 0.0, 1.0)
    diff_se = abs(se_norm_left - se_norm_right); avg_se = (se_norm_left + se_norm_right) / 2.0
    pdiff_se = 0.0
    if avg_se > min_val_config: pdiff_se = (diff_se / (avg_se + epsilon)) * 100.0
    elif diff_se > min_val_config: pdiff_se = percent_diff_cap
    feature_dict_final[f"SE_Asym_PercDiff_{coup_au}"] = np.clip(np.nan_to_num(pdiff_se, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)

    # 3. Ratio Features (AU17 vs Context AUs, Scalar, Normalized)
    if use_normalized and local_context_aus:
        for action in local_actions: # BS, SE
             if action in action_data_cache_scalar:
                action_cache = action_data_cache_scalar[action]
                au17_val = action_cache.get(coup_au) # Should be normalized AU17
                if au17_val is not None:
                    for context_au in local_context_aus:
                        context_val = action_cache.get(context_au) # Should be normalized context
                        if context_val is not None:
                             feature_name_ratio = f"{action}_Ratio_{coup_au}_vs_{context_au}"
                             min_r = min(au17_val, context_val); max_r = max(au17_val, context_val);
                             ratio_r = 1.0 if max_r <= min_val_config else (0.0 if min_r <= min_val_config else min_r / (max_r + epsilon))
                             feature_dict_final[feature_name_ratio] = np.clip(np.nan_to_num(ratio_r, nan=1.0), 0.0, 1.0)

    # 4. Summary Features Across Actions for AU17 (Normalized)
    valid_coupled_values = [v for v in coupled_values_list if pd.notna(v) and np.isfinite(v)]
    if valid_coupled_values and len(local_actions) > 1:
        feature_dict_final[f"Avg_{coup_au}_AcrossActions"] = np.mean(valid_coupled_values)
        feature_dict_final[f"Max_{coup_au}_AcrossActions"] = np.max(valid_coupled_values)
        feature_dict_final[f"Std_{coup_au}_AcrossActions"] = np.std(valid_coupled_values) if len(valid_coupled_values) > 1 else 0.0
    elif len(local_actions) == 1 and valid_coupled_values: # Handle single action case
         feature_dict_final[f"Avg_{coup_au}_AcrossActions"] = valid_coupled_values[0]
         feature_dict_final[f"Max_{coup_au}_AcrossActions"] = valid_coupled_values[0]
         feature_dict_final[f"Std_{coup_au}_AcrossActions"] = 0.0
    else: # Fallback
        feature_dict_final[f"Avg_{coup_au}_AcrossActions"] = 0.0
        feature_dict_final[f"Max_{coup_au}_AcrossActions"] = 0.0
        feature_dict_final[f"Std_{coup_au}_AcrossActions"] = 0.0

    # Add side indicator
    feature_dict_final["side_indicator"] = 0 if side_label.lower() == 'left' else 1

    # --- Load the EXPECTED feature list ---
    feature_names_path = MODEL_FILENAMES.get('feature_list')
    if not feature_names_path or not os.path.exists(feature_names_path): local_logger.error(f"Mentalis feature list not found: {feature_names_path}."); return None
    try:
        ordered_feature_names = joblib.load(feature_names_path);
        if not isinstance(ordered_feature_names, list): local_logger.error("Loaded feature names not a list."); return None
    except Exception as e: local_logger.error(f"Failed load Mentalis feature list: {e}", exc_info=True); return None

    # --- Build final feature list IN ORDER ---
    feature_list = []; missing_in_dict = []; type_errors = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name); final_val = 0.0
        if value is None: missing_in_dict.append(name)
        else:
            try:
                temp_val = float(value);
                if np.isnan(temp_val) or np.isinf(temp_val): final_val = 0.0
                else: final_val = temp_val
            except (ValueError, TypeError): type_errors.append(name); final_val = 0.0
        feature_list.append(final_val)

    if missing_in_dict: local_logger.warning(f"Mentalis Detect Full ({pid_for_log}, {side_label}): {len(missing_in_dict)} expected features missing: {missing_in_dict[:5]}... Used 0.0.")
    if type_errors: local_logger.warning(f"Mentalis Detect Full ({pid_for_log}, {side_label}): {len(type_errors)} features had type errors: {type_errors[:5]}... Used 0.0.")

    if len(feature_list) != len(ordered_feature_names):
        local_logger.error(f"CRITICAL MISMATCH Mentalis Full: Feature list length ({len(feature_list)}) != expected ({len(ordered_feature_names)}).")
        return None

    local_logger.debug(f"Generated {len(feature_list)} Full Mentalis detection features for {side_label}.")
    return feature_list


# --- process_targets function (Copied - Unchanged) ---
def process_targets(target_series):
    # ... (Identical implementation as before) ...
    if target_series is None: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    numeric_yes_values = { val: 1 for val in target_series.unique() if isinstance(val, (int, float)) and val > 0 }
    mapping.update(numeric_yes_values); numeric_no_values = { val: 0 for val in target_series.unique() if isinstance(val, (int, float)) and val == 0 }
    mapping.update(numeric_no_values); s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = target_series.map(mapping); mapped_str = s_clean.map(mapping); mapped = mapped.fillna(mapped_str)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0); return final_mapped.astype(int).values