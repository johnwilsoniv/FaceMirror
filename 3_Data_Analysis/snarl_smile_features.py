# snarl_smile_features.py
# - Extracts features ONLY for BS action (normalized) vs BL (for normalization).
# - FINAL V7 Feature Set (15 features):
#   A. Single-Side Norm AUs: Norm(AU12), Norm(AU10), Norm(AU14), Norm(AU15) (4)
#   B. Single-Side Ratios: Ratio(AU10/12), Ratio(AU14/12), Ratio(AU15/12), Ratio(AU15/10) (4)
#   C. Max-Across-Sides: MaxNorm(AU10), MaxNorm(AU15), MaxRatio(AU15/10) (3)
#   D. Targeted Asymmetry: AsymRatio(NormAU12), AsymPercDiff(NormAU12), AsymRatio(NormAU15), AsymPercDiff(NormAU15) (4)
#   E. REMOVED Side Indicator

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Import config for Snarl-Smile
try:
    # Ensure config matches V7 expectation (15 features, specific AUs)
    from snarl_smile_config import (
        FEATURE_CONFIG, SNARL_SMILE_ACTIONS, TRIGGER_AUS, COUPLED_AUS, LOG_DIR,
        FEATURE_SELECTION, MODEL_FILENAMES, CLASS_NAMES
    )
    CONFIG_LOADED = True
    # Verify COUPLED_AUS for V7 consistency if possible
    if COUPLED_AUS != ['AU10_r', 'AU14_r', 'AU15_r']:
        logging.warning(f"Config COUPLED_AUS {COUPLED_AUS} differs from expected V7 ['AU10_r', 'AU14_r', 'AU15_r']")
except ImportError:
    logging.warning("Could not import from snarl_smile_config. Using fallback definitions for V7.")
    CONFIG_LOADED = False
    LOG_DIR = 'logs'; MODEL_DIR = 'models/synkinesis/snarl_smile'
    MODEL_FILENAMES = {'feature_list': os.path.join(MODEL_DIR,'features.list'),
                       'importance_file': os.path.join(MODEL_DIR,'feature_importance.csv')}
    SNARL_SMILE_ACTIONS = ['BS']; TRIGGER_AUS = ['AU12_r']; COUPLED_AUS = ['AU10_r', 'AU14_r', 'AU15_r'] # V7 AUs
    FEATURE_CONFIG = {'actions': SNARL_SMILE_ACTIONS, 'trigger_aus': TRIGGER_AUS, 'coupled_aus': COUPLED_AUS,
                      'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 15, 'importance_file': MODEL_FILENAMES['importance_file']}
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}

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

# --- Helper: Extract Basic Single-Side Features (V7 - No AU17) ---
def _extract_basic_side_features(df, side):
    """ Extracts single-side Norm AUs and Ratios for the given side (V7). """
    logger.debug(f"Extracting basic V7 features for {side} side...")
    side_label = side.capitalize(); basic_features = {}
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED else {}
    local_trigger_aus = local_feature_config.get('trigger_aus', ['AU12_r'])
    local_coupled_aus = local_feature_config.get('coupled_aus', ['AU10_r', 'AU14_r', 'AU15_r']) # V7 AUs
    use_normalized = local_feature_config.get('use_normalized', True)
    all_aus_involved = local_trigger_aus + local_coupled_aus

    bl_raw_values = {}; bs_raw_values_side = {}
    for au in all_aus_involved:
        bl_raw_values[au] = _get_numeric_series(df, f"BL_{side_label} {au}")
        bs_raw_values_side[au] = _get_numeric_series(df, f"BS_{side_label} {au}")

    trig_au = local_trigger_aus[0]; norm_values = {}
    if use_normalized:
        # A. Calculate all Norm values first
        for au in all_aus_involved:
            norm_values[au] = (bs_raw_values_side[au] - bl_raw_values.get(au, 0.0)).clip(lower=0)
            feature_name = f"BS_{au}_trig_norm" if au == trig_au else f"BS_{au}_coup_norm"
            basic_features[feature_name] = norm_values[au] # (4 features here)

        # Pre-calculate needed Norm series for ratios/sums
        trigger_norm_series = norm_values[trig_au]
        norm_au10 = norm_values.get('AU10_r', pd.Series(0.0, index=df.index))
        norm_au14 = norm_values.get('AU14_r', pd.Series(0.0, index=df.index))
        norm_au15 = norm_values.get('AU15_r', pd.Series(0.0, index=df.index))

        # B. Calculate Ratios
        basic_features[f"BS_Ratio_AU10_vs_AU12"] = calculate_ratio(norm_au10, trigger_norm_series)
        basic_features[f"BS_Ratio_AU14_vs_AU12"] = calculate_ratio(norm_au14, trigger_norm_series)
        basic_features[f"BS_Ratio_AU15_vs_AU12"] = calculate_ratio(norm_au15, trigger_norm_series)
        basic_features[f"BS_Ratio_AU15_vs_AU10"] = calculate_ratio(norm_au15, norm_au10) # (4 features here)

    else: # Fallback
        logger.warning("use_normalized=False. Using raw values.")
        # Assign raw values to norm_values dictionary
        norm_values[trig_au] = bs_raw_values_side[trig_au]
        basic_features[f"BS_{trig_au}_trig_raw"] = norm_values[trig_au]
        for coup_au in local_coupled_aus:
            norm_values[coup_au] = bs_raw_values_side[coup_au]
            basic_features[f"BS_{coup_au}_coup_raw"] = norm_values[coup_au]

        # Get needed raw values directly from the norm_values dict
        trigger_raw_series = norm_values[trig_au]
        norm_au10 = norm_values.get('AU10_r', pd.Series(0.0, index=df.index))
        norm_au14 = norm_values.get('AU14_r', pd.Series(0.0, index=df.index))
        norm_au15 = norm_values.get('AU15_r', pd.Series(0.0, index=df.index))

        # Calculate ratios using these retrieved raw values
        basic_features[f"BS_Ratio_AU10_vs_AU12"] = calculate_ratio(norm_au10, trigger_raw_series)
        basic_features[f"BS_Ratio_AU14_vs_AU12"] = calculate_ratio(norm_au14, trigger_raw_series)
        basic_features[f"BS_Ratio_AU15_vs_AU12"] = calculate_ratio(norm_au15, trigger_raw_series)
        basic_features[f"BS_Ratio_AU15_vs_AU10"] = calculate_ratio(norm_au15, norm_au10)

    return pd.DataFrame(basic_features, index=df.index) # Returns 4 + 4 = 8 features

# --- prepare_data function (V7 - Removed side_indicator addition) ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data for Snarl-Smile synkinesis training (BS-Only, V7 - 15 Features). """
    logger.info("Loading datasets for Snarl-Smile Synkinesis (BS-Only, V7 - 15 Features)...")
    # --- Data Loading and Initial Processing ---
    try:
        results_df = pd.read_csv(results_file, low_memory=False); expert_df = pd.read_csv(expert_file)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise
    expert_df = expert_df.rename(columns={ 'Patient': 'Patient ID', 'Snarl Smile Left': 'Expert_Left_Snarl_Smile', 'Snarl Smile Right': 'Expert_Right_Snarl_Smile' })
    def process_targets(target_series):
        if target_series is None: return np.array([], dtype=int)
        mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }; s_filled = target_series.fillna('no')
        s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
        mapped = s_clean.map(mapping); unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
        if not unexpected.empty: logger.warning(f"Unexpected Snarl-Smile expert labels treated as 'No': {unexpected.unique().tolist()}")
        final_mapped = mapped.fillna(0); return final_mapped.astype(int).values
    if 'Expert_Left_Snarl_Smile' in expert_df.columns: expert_df['Target_Left_Snarl_Smile'] = process_targets(expert_df['Expert_Left_Snarl_Smile'])
    else: logger.error("Missing 'Expert_Left_Snarl_Smile' column"); expert_df['Target_Left_Snarl_Smile'] = 0
    if 'Expert_Right_Snarl_Smile' in expert_df.columns: expert_df['Target_Right_Snarl_Smile'] = process_targets(expert_df['Expert_Right_Snarl_Smile'])
    else: logger.error("Missing 'Expert_Right_Snarl_Smile' column"); expert_df['Target_Right_Snarl_Smile'] = 0
    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
    expert_cols_to_merge = ['Patient ID', 'Target_Left_Snarl_Smile', 'Target_Right_Snarl_Smile']
    expert_cols_exist = [col for col in expert_cols_to_merge if col in expert_df.columns]
    if len(expert_cols_exist) < len(expert_cols_to_merge): logger.error(f"Expert columns missing for merge: {set(expert_cols_to_merge) - set(expert_cols_exist)}."); return None, None
    try: merged_df = pd.merge(results_df, expert_df[expert_cols_exist], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data for Snarl-Smile: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")
    targets_available = 'Target_Left_Snarl_Smile' in merged_df.columns and 'Target_Right_Snarl_Smile' in merged_df.columns
    if not targets_available: logger.error("Target columns missing after merge."); return None, None
    # --- End Data Loading ---

    # --- Feature Extraction V7 ---
    logger.info("Extracting basic Left side features (V7)...")
    left_basic_df = _extract_basic_side_features(merged_df, 'Left')
    logger.info("Extracting basic Right side features (V7)...")
    right_basic_df = _extract_basic_side_features(merged_df, 'Right')
    if left_basic_df is None or right_basic_df is None: logger.error("Basic feature extraction failed."); return None, None

    logger.info("Calculating cross-side features (Max, Asymmetry V7)...")
    cross_side_features = {}; local_trigger_aus = FEATURE_CONFIG.get('trigger_aus', ['AU12_r']); trig_au = local_trigger_aus[0]
    # C. Max-Across-Sides Features (3 features)
    cross_side_features['Max_BS_AU10_Norm'] = np.maximum(left_basic_df.get('BS_AU10_r_coup_norm', 0), right_basic_df.get('BS_AU10_r_coup_norm', 0))
    cross_side_features['Max_BS_AU15_Norm'] = np.maximum(left_basic_df.get('BS_AU15_r_coup_norm', 0), right_basic_df.get('BS_AU15_r_coup_norm', 0))
    cross_side_features['Max_BS_Ratio_AU15_vs_AU10'] = np.maximum(left_basic_df.get('BS_Ratio_AU15_vs_AU10', 0), right_basic_df.get('BS_Ratio_AU15_vs_AU10', 0))
    # D. Targeted Asymmetry Features (4 features)
    left_norm_au12 = left_basic_df.get(f'BS_{trig_au}_trig_norm', 0); right_norm_au12 = right_basic_df.get(f'BS_{trig_au}_trig_norm', 0)
    left_norm_au15 = left_basic_df.get('BS_AU15_r_coup_norm', 0); right_norm_au15 = right_basic_df.get('BS_AU15_r_coup_norm', 0)
    cross_side_features['Asym_Ratio_BS_AU12_Norm'] = calculate_ratio(left_norm_au12, right_norm_au12)
    cross_side_features['Asym_PercDiff_BS_AU12_Norm'] = calculate_percent_diff(left_norm_au12, right_norm_au12)
    cross_side_features['Asym_Ratio_BS_AU15_Norm'] = calculate_ratio(left_norm_au15, right_norm_au15)
    cross_side_features['Asym_PercDiff_BS_AU15_Norm'] = calculate_percent_diff(left_norm_au15, right_norm_au15)
    cross_side_features_df = pd.DataFrame(cross_side_features, index=merged_df.index) # Contains 3 + 4 = 7 features

    # Combine basic side features with cross-side features
    left_full_df = pd.concat([left_basic_df, cross_side_features_df], axis=1) # 8 + 7 = 15
    right_full_df = pd.concat([right_basic_df, cross_side_features_df], axis=1) # 8 + 7 = 15

    # <<< REMOVED side_indicator addition >>>

    left_targets = merged_df['Target_Left_Snarl_Smile'].values; right_targets = merged_df['Target_Right_Snarl_Smile'].values

    # Concatenate final features and targets
    features = pd.concat([left_full_df, right_full_df], ignore_index=True)
    targets = np.concatenate([left_targets, right_targets])
    # --- End Feature Extraction V7 ---

    # Final Class Distribution Log
    unique_final, counts_final = np.unique(targets, return_counts=True)
    final_class_dist = {CLASS_NAMES.get(i, f"Class_{i}"): int(c) for i, c in zip(unique_final, counts_final)}
    logger.info(f"FINAL Snarl-Smile Class distribution input: {final_class_dist}")

    # Data Cleaning & Type Conversion
    features.replace([np.inf, -np.inf], np.nan, inplace=True); features = features.fillna(0)
    initial_cols = features.columns.tolist(); cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert col {col}. Marking drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)

    logger.info(f"Generated initial {features.shape[1]} Snarl-Smile features (BS-Only, V7 - 15 Features).") # Should be 15

    logger.info("Snarl-Smile feature selection is disabled.")

    logger.info(f"Final Snarl-Smile dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.critical(f"NaNs found in FINAL Snarl-Smile features AFTER cleaning."); features = features.fillna(0)

    # Save Final Feature List
    final_feature_names = features.columns.tolist()
    try:
        feature_list_path = MODEL_FILENAMES.get('feature_list')
        if feature_list_path:
             os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
             joblib.dump(final_feature_names, feature_list_path)
             logger.info(f"Saved final {len(final_feature_names)} Snarl-Smile feature names list to {feature_list_path}")
        else: logger.error("Snarl-Smile feature list path not defined in config.")
    except Exception as e: logger.error(f"Failed to save Snarl-Smile feature names list: {e}", exc_info=True)

    if 'targets' not in locals() or targets is None: logger.error("Targets array creation failed."); return None, None
    return features, targets


# --- extract_features_for_detection (Detection - BS Only, V7 - 15 Features) ---
def extract_features_for_detection(row_data, side):
    """ Extracts V7 (15) Snarl-Smile features for detection using ONLY BS (normalized). """
    try:
        # Ensure config COUPLED_AUS matches V7
        from snarl_smile_config import FEATURE_CONFIG, SNARL_SMILE_ACTIONS, TRIGGER_AUS, COUPLED_AUS, MODEL_FILENAMES
        local_logger = logging.getLogger(__name__); CONFIG_LOADED_DETECT = True
    except ImportError:
        logging.error("Failed config import within snarl_smile extract_features_for_detection (V7)."); return None
        CONFIG_LOADED_DETECT = False; FEATURE_CONFIG = {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
        SNARL_SMILE_ACTIONS = ['BS']; TRIGGER_AUS = ['AU12_r']; COUPLED_AUS = ['AU10_r', 'AU14_r', 'AU15_r'] # V7 AUs
        MODEL_FILENAMES = {'feature_list': 'models/synkinesis/snarl_smile/features.list'}; local_logger = logging

    if not isinstance(row_data, (pd.Series, dict)): local_logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    pid_for_log = row_series.get('Patient ID', 'UnknownPID')

    if side not in ['Left', 'Right']: local_logger.error(f"Invalid 'side': {side}"); return None
    else: side_label = side

    # --- Config values ---
    local_feature_config = FEATURE_CONFIG if CONFIG_LOADED_DETECT else {}; local_actions = local_feature_config.get('actions', ['BS'])
    local_trigger_aus = local_feature_config.get('trigger_aus', TRIGGER_AUS); local_coupled_aus = local_feature_config.get('coupled_aus', COUPLED_AUS) # V7 AUs
    use_normalized = local_feature_config.get('use_normalized', True); min_val_config = local_feature_config.get('min_value', 0.0001)
    percent_diff_cap = local_feature_config.get('percent_diff_cap', 200.0); epsilon = 1e-9
    all_aus_involved = local_trigger_aus + local_coupled_aus; trig_au = local_trigger_aus[0]
    # --- End Config ---

    if 'BS' not in local_actions: logger.error("BS action missing."); return None

    local_logger.debug(f"Extracting V7 (15) Snarl-Smile detection features for side {side_label}...")
    feature_dict_final = {}

    # --- Helper: get_float_value (Corrected Syntax) ---
    def get_float_value(key, default=0.0):
        val = row_series.get(key, default)
        try:
            f_val = float(val)
            if np.isnan(f_val) or np.isinf(f_val):
                return default
            else:
                return f_val
        except (ValueError, TypeError):
            return default
    # --- End Helper ---

    # --- Need ALL Left & Right BS/BL values ---
    bs_vals_L, bs_vals_R = {}, {}; bl_vals_L, bl_vals_R = {}, {}
    for au in all_aus_involved:
        bs_vals_L[au] = get_float_value(f"BS_Left {au}"); bs_vals_R[au] = get_float_value(f"BS_Right {au}")
        bl_vals_L[au] = get_float_value(f"BL_Left {au}"); bl_vals_R[au] = get_float_value(f"BL_Right {au}")

    # --- Calculate L/R Normalized Values ---
    norm_vals_L, norm_vals_R = {}, {}
    if use_normalized:
        for au in all_aus_involved: norm_vals_L[au] = max(0.0, bs_vals_L[au] - bl_vals_L.get(au, 0.0)); norm_vals_R[au] = max(0.0, bs_vals_R[au] - bl_vals_R.get(au, 0.0))
    else: norm_vals_L = bs_vals_L; norm_vals_R = bs_vals_R

    # --- Calculate L/R Ratios ---
    ratio_vals_L, ratio_vals_R = {}, {}
    norm_L_trig = norm_vals_L.get(trig_au, 0.0); norm_R_trig = norm_vals_R.get(trig_au, 0.0); norm_L_au10 = norm_vals_L.get('AU10_r', 0.0); norm_R_au10 = norm_vals_R.get('AU10_r', 0.0)
    norm_L_au14 = norm_vals_L.get('AU14_r', 0.0); norm_R_au14 = norm_vals_R.get('AU14_r', 0.0); norm_L_au15 = norm_vals_L.get('AU15_r', 0.0); norm_R_au15 = norm_vals_R.get('AU15_r', 0.0)
    min_L=min(norm_L_au10,norm_L_trig); max_L=max(norm_L_au10,norm_L_trig); ratio_L=1.0 if max_L<=min_val_config else (0.0 if min_L<=min_val_config else min_L/(max_L+epsilon)); ratio_vals_L['AU10_vs_AU12']=np.clip(np.nan_to_num(ratio_L,nan=1.0),0.0,1.0)
    min_R=min(norm_R_au10,norm_R_trig); max_R=max(norm_R_au10,norm_R_trig); ratio_R=1.0 if max_R<=min_val_config else (0.0 if min_R<=min_val_config else min_R/(max_R+epsilon)); ratio_vals_R['AU10_vs_AU12']=np.clip(np.nan_to_num(ratio_R,nan=1.0),0.0,1.0)
    min_L=min(norm_L_au14,norm_L_trig); max_L=max(norm_L_au14,norm_L_trig); ratio_L=1.0 if max_L<=min_val_config else (0.0 if min_L<=min_val_config else min_L/(max_L+epsilon)); ratio_vals_L['AU14_vs_AU12']=np.clip(np.nan_to_num(ratio_L,nan=1.0),0.0,1.0)
    min_R=min(norm_R_au14,norm_R_trig); max_R=max(norm_R_au14,norm_R_trig); ratio_R=1.0 if max_R<=min_val_config else (0.0 if min_R<=min_val_config else min_R/(max_R+epsilon)); ratio_vals_R['AU14_vs_AU12']=np.clip(np.nan_to_num(ratio_R,nan=1.0),0.0,1.0)
    min_L=min(norm_L_au15,norm_L_trig); max_L=max(norm_L_au15,norm_L_trig); ratio_L=1.0 if max_L<=min_val_config else (0.0 if min_L<=min_val_config else min_L/(max_L+epsilon)); ratio_vals_L['AU15_vs_AU12']=np.clip(np.nan_to_num(ratio_L,nan=1.0),0.0,1.0)
    min_R=min(norm_R_au15,norm_R_trig); max_R=max(norm_R_au15,norm_R_trig); ratio_R=1.0 if max_R<=min_val_config else (0.0 if min_R<=min_val_config else min_R/(max_R+epsilon)); ratio_vals_R['AU15_vs_AU12']=np.clip(np.nan_to_num(ratio_R,nan=1.0),0.0,1.0)
    min_L=min(norm_L_au15,norm_L_au10); max_L=max(norm_L_au15,norm_L_au10); ratio_L=1.0 if max_L<=min_val_config else (0.0 if min_L<=min_val_config else min_L/(max_L+epsilon)); ratio_vals_L['AU15_vs_AU10']=np.clip(np.nan_to_num(ratio_L,nan=1.0),0.0,1.0)
    min_R=min(norm_R_au15,norm_R_au10); max_R=max(norm_R_au15,norm_R_au10); ratio_R=1.0 if max_R<=min_val_config else (0.0 if min_R<=min_val_config else min_R/(max_R+epsilon)); ratio_vals_R['AU15_vs_AU10']=np.clip(np.nan_to_num(ratio_R,nan=1.0),0.0,1.0)

    # --- Populate Feature Dictionary ---
    # A. Single-Side Norm AUs (Use the value for the *target* side) - 4 features
    current_side_norm_vals = norm_vals_L if side_label == 'Left' else norm_vals_R
    feature_dict_final[f"BS_{trig_au}_trig_norm"] = current_side_norm_vals.get(trig_au, 0.0)
    feature_dict_final[f"BS_AU10_r_coup_norm"] = current_side_norm_vals.get('AU10_r', 0.0)
    feature_dict_final[f"BS_AU14_r_coup_norm"] = current_side_norm_vals.get('AU14_r', 0.0)
    feature_dict_final[f"BS_AU15_r_coup_norm"] = current_side_norm_vals.get('AU15_r', 0.0)
    # B. Single-Side Ratios (Use the value for the *target* side) - 4 features
    current_side_ratio_vals = ratio_vals_L if side_label == 'Left' else ratio_vals_R
    feature_dict_final[f"BS_Ratio_AU10_vs_AU12"] = current_side_ratio_vals.get('AU10_vs_AU12', 0.0)
    feature_dict_final[f"BS_Ratio_AU14_vs_AU12"] = current_side_ratio_vals.get('AU14_vs_AU12', 0.0)
    feature_dict_final[f"BS_Ratio_AU15_vs_AU12"] = current_side_ratio_vals.get('AU15_vs_AU12', 0.0)
    feature_dict_final[f"BS_Ratio_AU15_vs_AU10"] = current_side_ratio_vals.get('AU15_vs_AU10', 0.0)
    # C. Max-Across-Sides Features - 3 features
    feature_dict_final['Max_BS_AU10_Norm'] = max(norm_L_au10, norm_R_au10)
    feature_dict_final['Max_BS_AU15_Norm'] = max(norm_L_au15, norm_R_au15)
    feature_dict_final['Max_BS_Ratio_AU15_vs_AU10'] = max(ratio_vals_L.get('AU15_vs_AU10',0.0), ratio_vals_R.get('AU15_vs_AU10',0.0))

    # --- D. Targeted Asymmetry Features - *** CORRECTED READABLE VERSION *** ---
    # Asym AU12 Norm Ratio
    min_asym_au12 = min(norm_L_trig, norm_R_trig); max_asym_au12 = max(norm_L_trig, norm_R_trig); ratio_asym_au12 = 1.0
    if max_asym_au12 > min_val_config:
        if min_asym_au12 <= min_val_config: ratio_asym_au12 = 0.0
        else: ratio_asym_au12 = min_asym_au12 / (max_asym_au12 + epsilon)
    feature_dict_final['Asym_Ratio_BS_AU12_Norm'] = np.clip(np.nan_to_num(ratio_asym_au12, nan=1.0), 0.0, 1.0)
    # Asym AU12 Norm PercDiff
    diff_asym_au12 = abs(norm_L_trig - norm_R_trig); avg_asym_au12 = (norm_L_trig + norm_R_trig) / 2.0; pdiff_asym_au12 = 0.0;
    if avg_asym_au12 > min_val_config: pdiff_asym_au12 = (diff_asym_au12 / (avg_asym_au12 + epsilon)) * 100.0
    elif diff_asym_au12 > min_val_config: pdiff_asym_au12 = percent_diff_cap
    feature_dict_final['Asym_PercDiff_BS_AU12_Norm'] = np.clip(np.nan_to_num(pdiff_asym_au12, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)
    # Asym AU15 Norm Ratio
    min_asym_au15 = min(norm_L_au15, norm_R_au15); max_asym_au15 = max(norm_L_au15, norm_R_au15); ratio_asym_au15 = 1.0
    if max_asym_au15 > min_val_config:
        if min_asym_au15 <= min_val_config: ratio_asym_au15 = 0.0
        else: ratio_asym_au15 = min_asym_au15 / (max_asym_au15 + epsilon)
    feature_dict_final['Asym_Ratio_BS_AU15_Norm'] = np.clip(np.nan_to_num(ratio_asym_au15, nan=1.0), 0.0, 1.0)
    # Asym AU15 Norm PercDiff
    diff_asym_au15 = abs(norm_L_au15 - norm_R_au15); avg_asym_au15 = (norm_L_au15 + norm_R_au15) / 2.0; pdiff_asym_au15 = 0.0;
    if avg_asym_au15 > min_val_config: pdiff_asym_au15 = (diff_asym_au15 / (avg_asym_au15 + epsilon)) * 100.0
    elif diff_asym_au15 > min_val_config: pdiff_asym_au15 = percent_diff_cap
    feature_dict_final['Asym_PercDiff_BS_AU15_Norm'] = np.clip(np.nan_to_num(pdiff_asym_au15, nan=0.0, posinf=percent_diff_cap, neginf=0.0), 0, percent_diff_cap)
    # --- End Asymmetry Block ---

    # <<< REMOVED side_indicator feature >>>

    # --- Load the EXPECTED feature list (should contain the 15 feature names) ---
    feature_names_path = MODEL_FILENAMES.get('feature_list')
    if not feature_names_path or not os.path.exists(feature_names_path): local_logger.error(f"Snarl-Smile feature list not found: {feature_names_path}."); return None
    try:
        ordered_feature_names = joblib.load(feature_names_path);
        if not isinstance(ordered_feature_names, list): local_logger.error("Loaded feature names not a list."); return None
    except Exception as e: local_logger.error(f"Failed load Snarl-Smile feature list: {e}", exc_info=True); return None

    # --- Build final feature list IN ORDER ---
    feature_list = []; missing_in_dict = []; type_errors = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name); final_val = 0.0
        if value is None: missing_in_dict.append(name)
        else:
            try: temp_val = float(value); final_val = 0.0 if np.isnan(temp_val) or np.isinf(temp_val) else temp_val
            except (ValueError, TypeError): type_errors.append(name); final_val = 0.0
        feature_list.append(final_val)

    if missing_in_dict: local_logger.warning(f"Snarl-Smile Detect (V7, {pid_for_log}, {side_label}): {len(missing_in_dict)} expected features missing: {missing_in_dict[:5]}... Using 0.0.")
    if type_errors: local_logger.warning(f"Snarl-Smile Detect (V7, {pid_for_log}, {side_label}): {len(type_errors)} features had type errors: {type_errors[:5]}... Using 0.0.")

    if len(feature_list) != len(ordered_feature_names):
        local_logger.error(f"CRITICAL MISMATCH SnSm (V7): Feature list length ({len(feature_list)}) != expected ({len(ordered_feature_names)} from {feature_names_path}).")
        local_logger.error(f"Generated keys ({len(feature_dict_final)}): {sorted(list(feature_dict_final.keys()))}")
        local_logger.error(f"Expected keys ({len(ordered_feature_names)}): {sorted(ordered_feature_names)}")
        return None

    local_logger.debug(f"Generated {len(feature_list)} V7 Snarl-Smile detection features for {side_label}.")
    return feature_list


# --- process_targets function (Identical) ---
def process_targets(target_series):
    if target_series is None: return np.array([], dtype=int); mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping); unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0); return final_mapped.astype(int).values