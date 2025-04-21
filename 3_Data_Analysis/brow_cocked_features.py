# brow_cocked_features.py
# Extracts features for detecting Brow Cocked phenomenon.
# V1: BL Raw values, BL Raw Asymmetry, ET Normalized Change (AU01/02 unclipped, AU07 clipped)
# Corrected expert key column names, process_targets, and get_float_value syntax.

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Import config for Brow Cocked
try:
    from brow_cocked_config import (
        FEATURE_CONFIG, BROW_COCKED_ACTIONS, INTEREST_AUS, CONTEXT_AUS, ALL_RELEVANT_AUS, LOG_DIR,
        FEATURE_SELECTION, MODEL_FILENAMES, CLASS_NAMES
    )
    CONFIG_LOADED = True
except ImportError:
    logging.warning("Could not import from brow_cocked_config. Using fallback definitions.")
    CONFIG_LOADED = False
    LOG_DIR = 'logs'; MODEL_DIR = 'models/synkinesis/brow_cocked'
    MODEL_FILENAMES = {'feature_list': os.path.join(MODEL_DIR,'features.list'),
                       'importance_file': os.path.join(MODEL_DIR,'feature_importance.csv')}
    BROW_COCKED_ACTIONS = ['BL', 'ET']; INTEREST_AUS = ['AU01_r', 'AU02_r']; CONTEXT_AUS = ['AU07_r']
    ALL_RELEVANT_AUS = INTEREST_AUS + CONTEXT_AUS
    FEATURE_CONFIG = {'actions': BROW_COCKED_ACTIONS, 'interest_aus': INTEREST_AUS, 'context_aus': CONTEXT_AUS,
                      'min_value': 0.0001, 'percent_diff_cap': 200.0}
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 10, 'importance_file': MODEL_FILENAMES['importance_file']}
    CLASS_NAMES = {0: 'None', 1: 'Brow Cocked'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper: Get Numeric Series ---
def _get_numeric_series(df, col_name, default_val=0.0):
    series = df.get(col_name)
    if series is None: logger.warning(f"Column '{col_name}' not found. Using {default_val}."); return pd.Series(default_val, index=df.index, dtype=float)
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(default_val); return numeric_series.astype(float)

# --- Helper Functions (Ratio + PercDiff) ---
def calculate_ratio(val1_series, val2_series):
    local_logger = logging.getLogger(__name__)
    try: min_val_config = FEATURE_CONFIG.get('min_value', 0.0001)
    except NameError: min_val_config = 0.0001; local_logger.warning("calculate_ratio: FEATURE_CONFIG not found.")
    epsilon=1e-9
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy(); v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    min_vals_np = np.minimum(v1, v2); max_vals_np = np.maximum(v1, v2)
    ratio = np.ones_like(v1, dtype=float)
    mask_max_pos = max_vals_np > min_val_config; mask_min_zero = min_vals_np <= min_val_config
    ratio[mask_max_pos & mask_min_zero] = 0.0
    valid_division_mask = mask_max_pos & ~mask_min_zero
    if np.any(valid_division_mask): ratio[valid_division_mask] = min_vals_np[valid_division_mask] / (max_vals_np[valid_division_mask] + epsilon)
    if np.isnan(ratio).any() or np.isinf(ratio).any(): ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    ratio = np.clip(ratio, 0.0, 1.0); return pd.Series(ratio, index=val1_series.index)

def calculate_percent_diff(val1_series, val2_series):
    local_logger = logging.getLogger(__name__)
    try:
        min_val_config = FEATURE_CONFIG.get('min_value', 0.0001); percent_diff_cap = FEATURE_CONFIG.get('percent_diff_cap', 200.0)
    except NameError: min_val_config = 0.0001; percent_diff_cap = 200.0; local_logger.warning("calculate_percent_diff: FEATURE_CONFIG not found.")
    epsilon=1e-9
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy(); v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    abs_diff = np.abs(v1 - v2); avg = (v1 + v2) / 2.0
    percent_diff = np.zeros_like(avg, dtype=float)
    mask_avg_pos = avg > min_val_config
    if np.any(mask_avg_pos):
        with np.errstate(divide='ignore', invalid='ignore'): division_result = abs_diff[mask_avg_pos] / (avg[mask_avg_pos] + epsilon)
        percent_diff[mask_avg_pos] = division_result * 100.0
    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > min_val_config)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap
    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    percent_diff = np.nan_to_num(percent_diff, nan=0.0, posinf=percent_diff_cap, neginf=0.0)
    return pd.Series(percent_diff, index=val1_series.index)

# --- Helper: Extract Basic Side Features (V1 Brow Cocked) ---
def _extract_basic_side_features(df, side):
    logger.debug(f"Extracting basic Brow Cocked features for {side} side...")
    side_label = side.capitalize(); basic_features = {}
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED else {}
    local_interest_aus = local_feature_config.get('interest_aus', ['AU01_r', 'AU02_r'])
    local_context_aus = local_feature_config.get('context_aus', ['AU07_r'])
    all_relevant_aus = local_interest_aus + local_context_aus
    bl_raw_values = {}; et_raw_values = {}
    for au in all_relevant_aus:
        bl_raw_values[au] = _get_numeric_series(df, f"BL_{side_label} {au}")
        et_raw_values[au] = _get_numeric_series(df, f"ET_{side_label} {au}")
    for au in local_interest_aus: basic_features[f"BL_{au}_raw"] = bl_raw_values[au]
    for au in local_interest_aus: norm_val = et_raw_values[au] - bl_raw_values.get(au, 0.0); basic_features[f"ET_{au}_norm"] = norm_val
    for au in local_context_aus: norm_val = (et_raw_values[au] - bl_raw_values.get(au, 0.0)).clip(lower=0); basic_features[f"ET_{au}_norm"] = norm_val
    return pd.DataFrame(basic_features, index=df.index)

# --- prepare_data function (Brow Cocked V1 - Corrected Feature Names & process_targets) ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data for Brow Cocked detection training (V1 - 10 Features). """
    logger.info("Loading datasets for Brow Cocked Detection (V1)...")
    # --- Data Loading ---
    try:
        results_df = pd.read_csv(results_file, low_memory=False); expert_df = pd.read_csv(expert_file)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise
    # --- Rename Expert Columns ---
    expert_rename_map = { 'Patient': 'Patient ID', 'Cocked Left': 'Expert_Left_BrowCocked', 'Cocked Right': 'Expert_Right_BrowCocked' }; cols_to_rename = {k: v for k, v in expert_rename_map.items() if k in expert_df.columns}
    if 'Patient' in expert_df.columns and 'Patient ID' not in cols_to_rename: cols_to_rename['Patient'] = 'Patient ID'; expert_df = expert_df.rename(columns=cols_to_rename)
    logger.info(f"Renamed expert columns: {cols_to_rename}")
    # --- Process Targets (with corrected definition) ---
    def process_targets(target_series):
        if target_series is None:
            return np.array([], dtype=int) # Return empty if input is None
        mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
        s_filled = target_series.fillna('no') # Now s_filled is defined
        s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
        mapped = s_clean.map(mapping); unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
        if not unexpected.empty: logger.warning(f"Unexpected Brow Cocked expert labels treated as 'No': {unexpected.unique().tolist()}")
        final_mapped = mapped.fillna(0); return final_mapped.astype(int).values

    target_left_col = 'Expert_Left_BrowCocked'; target_right_col = 'Expert_Right_BrowCocked'
    if target_left_col not in expert_df.columns: logger.error(f"Missing expert column '{target_left_col}'."); return None, None
    if target_right_col not in expert_df.columns: logger.error(f"Missing expert column '{target_right_col}'."); return None, None
    expert_df['Target_Left_BrowCocked'] = process_targets(expert_df.get(target_left_col)); expert_df['Target_Right_BrowCocked'] = process_targets(expert_df.get(target_right_col))
    logger.info(f"Counts Left BrowCocked AFTER mapping: \n{pd.Series(expert_df['Target_Left_BrowCocked']).value_counts(dropna=False)}"); logger.info(f"Counts Right BrowCocked AFTER mapping: \n{pd.Series(expert_df['Target_Right_BrowCocked']).value_counts(dropna=False)}")
    # --- Merge Data ---
    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip(); expert_cols_to_merge = ['Patient ID', 'Target_Left_BrowCocked', 'Target_Right_BrowCocked']
    expert_cols_exist = [col for col in expert_cols_to_merge if col in expert_df.columns]
    if len(expert_cols_exist) < len(expert_cols_to_merge): logger.error(f"Expert columns missing for merge: {set(expert_cols_to_merge) - set(expert_cols_exist)}."); return None, None
    try: merged_df = pd.merge(results_df, expert_df[expert_cols_exist], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data for Brow Cocked: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")
    targets_available = 'Target_Left_BrowCocked' in merged_df.columns and 'Target_Right_BrowCocked' in merged_df.columns
    if not targets_available: logger.error("Target columns missing after merge."); return None, None
    # --- Feature Extraction V1 ---
    logger.info("Extracting basic Left side features (V1)..."); left_basic_df = _extract_basic_side_features(merged_df, 'Left')
    logger.info("Extracting basic Right side features (V1)..."); right_basic_df = _extract_basic_side_features(merged_df, 'Right')
    if left_basic_df is None or right_basic_df is None: logger.error("Basic feature extraction failed."); return None, None
    logger.info("Calculating cross-side asymmetry features (V1)..."); cross_side_features = {}; local_interest_aus = FEATURE_CONFIG.get('interest_aus', ['AU01_r', 'AU02_r'])
    for au in local_interest_aus:
        left_val = left_basic_df.get(f'BL_{au}_raw', 0); right_val = right_basic_df.get(f'BL_{au}_raw', 0)
        cross_side_features[f'BL_Asym_Ratio_{au}_raw'] = calculate_ratio(left_val, right_val); cross_side_features[f'BL_Asym_PercDiff_{au}_raw'] = calculate_percent_diff(left_val, right_val)
    cross_side_features_df = pd.DataFrame(cross_side_features, index=merged_df.index)
    # Combine and Restructure
    all_features_list = []
    for index in range(len(merged_df)):
        left_row_features = {}; left_row_features.update(left_basic_df.iloc[index].to_dict()); left_row_features.update(cross_side_features_df.iloc[index].to_dict()); left_row_features['side_indicator'] = 0; all_features_list.append(left_row_features)
        right_row_features = {}; right_row_features.update(right_basic_df.iloc[index].to_dict()); right_row_features.update(cross_side_features_df.iloc[index].to_dict()); right_row_features['side_indicator'] = 1; all_features_list.append(right_row_features)
    features = pd.DataFrame(all_features_list)
    # Define and enforce final feature order (10 features)
    final_feature_names_ordered = [
        'BL_AU01_r_raw', 'BL_AU02_r_raw', 'ET_AU01_r_norm', 'ET_AU02_r_norm', 'ET_AU07_r_norm',
        'BL_Asym_Ratio_AU01_r_raw', 'BL_Asym_PercDiff_AU01_r_raw', 'BL_Asym_Ratio_AU02_r_raw', 'BL_Asym_PercDiff_AU02_r_raw',
        'side_indicator'
    ]
    try: features = features[final_feature_names_ordered]
    except KeyError as e: logger.error(f"KeyError reordering features: {e}."); missing = set(final_feature_names_ordered)-set(features.columns); extra = set(features.columns)-set(final_feature_names_ordered); logger.error(f"Missing: {missing}"); logger.error(f"Extra: {extra}"); return None, None
    # --- End Feature Extraction V1 ---
    # Final Class Distribution Log
    left_targets = merged_df['Target_Left_BrowCocked'].values; right_targets = merged_df['Target_Right_BrowCocked'].values; targets = np.concatenate([left_targets, right_targets])
    unique_final, counts_final = np.unique(targets, return_counts=True); final_class_dist = {CLASS_NAMES.get(i, f"Class_{i}"): int(c) for i, c in zip(unique_final, counts_final)}
    logger.info(f"FINAL Brow Cocked Class distribution input: {final_class_dist}")
    # Data Cleaning & Type Conversion
    features.replace([np.inf, -np.inf], np.nan, inplace=True); features = features.fillna(0); initial_cols = features.columns.tolist(); cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert col {col}. Marking drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)
    logger.info(f"Generated initial {features.shape[1]} Brow Cocked features (V1 - 10 Features).")
    logger.info("Brow Cocked feature selection is disabled.")
    logger.info(f"Final Brow Cocked dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.critical(f"NaNs found in FINAL Brow Cocked features."); features = features.fillna(0)
    # Save Final Feature List
    final_feature_names = features.columns.tolist()
    try:
        feature_list_path = MODEL_FILENAMES.get('feature_list');
        if feature_list_path: os.makedirs(os.path.dirname(feature_list_path), exist_ok=True); joblib.dump(final_feature_names, feature_list_path); logger.info(f"Saved final {len(final_feature_names)} Brow Cocked feature names list to {feature_list_path}")
        else: logger.error("Brow Cocked feature list path not defined.")
    except Exception as e: logger.error(f"Failed to save Brow Cocked feature names list: {e}", exc_info=True)
    if 'targets' not in locals() or targets is None: logger.error("Targets array creation failed."); return None, None
    return features, targets

# --- extract_features_for_detection (Detection - V1 - 10 Features) ---
def extract_features_for_detection(row_data, side):
    """ Extracts V1 (10) Brow Cocked features for detection. """
    try:
        from brow_cocked_config import FEATURE_CONFIG, BROW_COCKED_ACTIONS, INTEREST_AUS, CONTEXT_AUS, ALL_RELEVANT_AUS, MODEL_FILENAMES
        local_logger = logging.getLogger(__name__); CONFIG_LOADED_DETECT = True
    except ImportError:
        logging.error("Failed config import within brow_cocked extract_features_for_detection (V1)."); return None
        CONFIG_LOADED_DETECT = False; FEATURE_CONFIG = {'min_value': 0.0001, 'percent_diff_cap': 200.0}
        BROW_COCKED_ACTIONS = ['BL', 'ET']; INTEREST_AUS = ['AU01_r', 'AU02_r']; CONTEXT_AUS = ['AU07_r']
        ALL_RELEVANT_AUS = INTEREST_AUS + CONTEXT_AUS
        MODEL_FILENAMES = {'feature_list': 'models/synkinesis/brow_cocked/features.list'}; local_logger = logging

    if not isinstance(row_data, (pd.Series, dict)): local_logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    pid_for_log = row_series.get('Patient ID', 'UnknownPID')
    if side not in ['Left', 'Right']: local_logger.error(f"Invalid 'side': {side}"); return None
    else: side_label = side

    # Config values
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED_DETECT else {}; local_interest_aus = local_feature_config.get('interest_aus', INTEREST_AUS); local_context_aus = local_feature_config.get('context_aus', CONTEXT_AUS)
    min_val_config = local_feature_config.get('min_value', 0.0001); percent_diff_cap = local_feature_config.get('percent_diff_cap', 200.0); epsilon = 1e-9; all_relevant_aus = local_interest_aus + local_context_aus

    local_logger.debug(f"Extracting V1 (10) Brow Cocked detection features for side {side_label}...")
    feature_dict_final = {}

    # --- *** DEFINITIVELY CORRECTED get_float_value Helper Function *** ---
    def get_float_value(key, default=0.0):
        """ Safely gets a float value from the row_series, handling errors and NaN/Inf. """
        val = row_series.get(key, default)
        try:
            f_val = float(val)
            if np.isnan(f_val) or np.isinf(f_val):
                return default
            else:
                return f_val
        except (ValueError, TypeError):
            return default
    # --- *** End CORRECTED get_float_value *** ---

    # Get L/R BL/ET Raw Values
    bl_vals_L, bl_vals_R = {}, {}; et_vals_L, et_vals_R = {}, {}
    for au in all_relevant_aus:
        bl_vals_L[au] = get_float_value(f"BL_Left {au}")
        bl_vals_R[au] = get_float_value(f"BL_Right {au}")
        et_vals_L[au] = get_float_value(f"ET_Left {au}")
        et_vals_R[au] = get_float_value(f"ET_Right {au}")

    # Populate Feature Dictionary
    current_bl_vals = bl_vals_L if side_label == 'Left' else bl_vals_R; current_et_vals = et_vals_L if side_label == 'Left' else et_vals_R
    # A. BL Raw Features
    for au in local_interest_aus: feature_dict_final[f"BL_{au}_raw"] = current_bl_vals.get(au, 0.0)
    # C/D. ET Normalized Features
    for au in local_interest_aus: norm_val = current_et_vals.get(au, 0.0) - current_bl_vals.get(au, 0.0); feature_dict_final[f"ET_{au}_norm"] = norm_val
    for au in local_context_aus: norm_val = max(0.0, current_et_vals.get(au, 0.0) - current_bl_vals.get(au, 0.0)); feature_dict_final[f"ET_{au}_norm"] = norm_val
    # B. Resting Asymmetry Features
    for au in local_interest_aus:
        left_val_bl = bl_vals_L.get(au, 0.0); right_val_bl = bl_vals_R.get(au, 0.0)
        min_bl = min(left_val_bl, right_val_bl); max_bl = max(left_val_bl, right_val_bl); ratio_bl = 1.0 if max_bl <= min_val_config else (0.0 if min_bl <= min_val_config else min_bl / (max_bl + epsilon))
        feature_dict_final[f'BL_Asym_Ratio_{au}_raw'] = np.clip(np.nan_to_num(ratio_bl, nan=1.0), 0.0, 1.0)
        diff_bl = abs(left_val_bl - right_val_bl); avg_bl = (left_val_bl + right_val_bl) / 2.0; pdiff_bl = 0.0;
        if avg_bl > min_val_config: pdiff_bl = (diff_bl / (avg_bl + epsilon)) * 100.0
        elif diff_bl > min_val_config: pdiff_bl = percent_diff_cap
        feature_dict_final[f'BL_Asym_PercDiff_{au}_raw'] = np.clip(np.nan_to_num(pdiff_bl, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)
    # E. Side Indicator
    feature_dict_final["side_indicator"] = 0 if side_label.lower() == 'left' else 1

    # Load Expected Features
    feature_names_path = MODEL_FILENAMES.get('feature_list')
    if not feature_names_path or not os.path.exists(feature_names_path): local_logger.error(f"Brow Cocked feature list not found: {feature_names_path}."); return None
    try: ordered_feature_names = joblib.load(feature_names_path);
    except Exception as e: local_logger.error(f"Failed load Brow Cocked feature list: {e}", exc_info=True); return None
    if not isinstance(ordered_feature_names, list): local_logger.error("Loaded feature names not list."); return None

    # Build Ordered Feature List
    feature_list = []; missing_in_dict = []; type_errors = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name); final_val = 0.0
        if value is None: missing_in_dict.append(name)
        else:
            try: temp_val = float(value); final_val = 0.0 if np.isnan(temp_val) or np.isinf(temp_val) else temp_val
            except (ValueError, TypeError): type_errors.append(name); final_val = 0.0
        feature_list.append(final_val)
    if missing_in_dict: local_logger.warning(f"BrwCk Detect (V1, {pid_for_log}, {side_label}): {len(missing_in_dict)} missing: {missing_in_dict[:5]}... Used 0.0.")
    if type_errors: local_logger.warning(f"BrwCk Detect (V1, {pid_for_log}, {side_label}): {len(type_errors)} type errors: {type_errors[:5]}... Used 0.0.")
    if len(feature_list) != len(ordered_feature_names): local_logger.error(f"CRITICAL MISMATCH BrwCk (V1): List len ({len(feature_list)}) != expected ({len(ordered_feature_names)})."); return None

    local_logger.debug(f"Generated {len(feature_list)} V1 Brow Cocked features for {side_label}.")
    return feature_list