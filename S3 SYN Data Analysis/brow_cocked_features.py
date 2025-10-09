# --- START OF FILE brow_cocked_features.py ---

# brow_cocked_features.py (Refactored V3 - ML Approach, BL high + RE paretic definition)
# - Focuses on BL (baseline asymmetry, higher rest)
# - Focuses on RE (normalized movement asymmetry, paretic movement on cocked side)
# - ET features removed as per refined definition.
# - Added directed asymmetry features.

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Use central config and utils
from synkinesis_config import SYNKINESIS_CONFIG, CLASS_NAMES, INPUT_FILES
from paralysis_utils import calculate_ratio, calculate_percent_diff, process_binary_target, standardize_synkinesis_labels

logger = logging.getLogger(__name__)

# --- Define type for this file ---
SYNK_TYPE = 'brow_cocked'

# Get type-specific config
try:
    config = SYNKINESIS_CONFIG[SYNK_TYPE]
    feature_cfg = config.get('feature_extraction', {})
    interest_aus = config.get('interest_aus', ['AU01_r', 'AU02_r']) # AUs indicating brow movement
    # context_aus for ET is no longer primary for feature generation with V3 definition
    # context_aus = config.get('context_aus', ['AU07_r'])
    feature_sel_cfg = config.get('feature_selection', {})
    filenames = config.get('filenames', {})
    expert_cols = config.get('expert_columns', {})
    target_cols = config.get('target_columns', {})
    CONFIG_LOADED = True
except KeyError:
    logger.critical(f"CRITICAL: Synkinesis type '{SYNK_TYPE}' not found in SYNKINESIS_CONFIG.")
    CONFIG_LOADED = False; config = {}; feature_cfg = {}; interest_aus = ['AU01_r', 'AU02_r']; feature_sel_cfg = {}; filenames = {}; expert_cols = {}; target_cols = {}


# --- prepare_data function (Standard Refactored Template - Largely Unchanged) ---
def prepare_data(results_file=None, expert_file=None):
    """
    Prepares data for the specific SYNK_TYPE, handling NA/Not Assessed labels,
    feature selection, and returning metadata alongside features and targets.
    Maps standardized labels correctly for binary classification.
    """
    # --- Configuration Setup ---
    try:
        if 'SYNK_TYPE' not in globals(): raise NameError("SYNK_TYPE variable is not defined globally.")
        config = SYNKINESIS_CONFIG[SYNK_TYPE]
        name = config.get('name', SYNK_TYPE.replace('_', ' ').title())
        feature_cfg = config.get('feature_extraction', {})
        feature_sel_cfg = config.get('feature_selection', {})
        filenames = config.get('filenames', {})
        expert_cols_cfg = config.get('expert_columns', {})
        target_cols_cfg = config.get('target_columns', {})
        CONFIG_LOADED = True # Redundant check but keeps structure
    except NameError as ne:
        logger.critical(f"CRITICAL: {ne} - Feature extraction cannot proceed.")
        return None, None, None
    except KeyError:
        logger.critical(f"CRITICAL: Synkinesis type '{SYNK_TYPE}' not found in SYNKINESIS_CONFIG.")
        return None, None, None

    logger.info(f"[{name}] Loading datasets for training data preparation...")
    results_file = results_file or INPUT_FILES.get('results_csv')
    expert_file = expert_file or INPUT_FILES.get('expert_key_csv')
    if not results_file or not expert_file or not os.path.exists(results_file) or not os.path.exists(expert_file):
        logger.error(f"[{name}] Input files missing. Abort."); return None, None, None

    try: # Load data
        logger.info(f"[{name}] Loading results: {results_file}"); results_df = pd.read_csv(results_file, low_memory=False)
        logger.info(f"[{name}] Loading expert key: {expert_file}")
        expert_df = pd.read_csv(
            expert_file,
            dtype=str,
            keep_default_na=False,
            na_values=['']
        )
        logger.info(f"[{name}] Loaded {len(results_df)} results rows, {len(expert_df)} expert key rows")
    except Exception as e: logger.error(f"[{name}] Error loading data: {e}.", exc_info=True); return None, None, None

    exp_left_orig = expert_cols_cfg.get('left'); exp_right_orig = expert_cols_cfg.get('right')
    target_left_col = target_cols_cfg.get('left'); target_right_col = target_cols_cfg.get('right')
    if not exp_left_orig or not exp_right_orig or not target_left_col or not target_right_col:
         logger.error(f"[{name}] Config missing expert/target column names. Abort."); return None, None, None
    if exp_left_orig not in expert_df.columns or exp_right_orig not in expert_df.columns:
         logger.error(f"[{name}] Raw expert columns ('{exp_left_orig}', '{exp_right_orig}') not found in expert key. Abort."); return None, None, None
    rename_map = {'Patient': 'Patient ID', exp_left_orig: exp_left_orig, exp_right_orig: exp_right_orig}
    expert_df = expert_df.rename(columns={k: v for k,v in rename_map.items() if k in expert_df.columns})

    expert_std_left_col = f'Expert_Std_Left_{SYNK_TYPE}'
    expert_std_right_col = f'Expert_Std_Right_{SYNK_TYPE}'
    expert_df[expert_std_left_col] = expert_df[exp_left_orig].apply(standardize_synkinesis_labels)
    expert_df[expert_std_right_col] = expert_df[exp_right_orig].apply(standardize_synkinesis_labels)
    logger.info(f"[{name}] Standardized expert labels created.")
    # logger.info(f"Value counts Std Left:\n{expert_df[expert_std_left_col].value_counts(dropna=False).to_string()}")
    # logger.info(f"Value counts Std Right:\n{expert_df[expert_std_right_col].value_counts(dropna=False).to_string()}")

    expert_cols_to_merge = ['Patient ID', expert_std_left_col, expert_std_right_col]
    expert_df_subset = expert_df[expert_cols_to_merge].copy()
    if expert_df_subset['Patient ID'].duplicated().any():
        logger.warning(f"[{name}] Duplicate Patient IDs found in expert key subset. Keeping first.")
        expert_df_subset.drop_duplicates(subset=['Patient ID'], keep='first', inplace=True)

    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df_subset['Patient ID'] = expert_df_subset['Patient ID'].astype(str).str.strip()
    try: merged_df = pd.merge(results_df, expert_df_subset, on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"[{name}] Merge failed: {e}", exc_info=True); return None, None, None
    logger.info(f"[{name}] Merged data: {len(merged_df)} patients")
    if merged_df.empty or expert_std_left_col not in merged_df.columns or expert_std_right_col not in merged_df.columns:
         logger.error(f"[{name}] Merge empty or standardized expert columns missing. Abort."); return None, None, None

    valid_left_mask = merged_df[expert_std_left_col] != 'NA'
    valid_right_mask = merged_df[expert_std_right_col] != 'NA'
    logger.info(f"[{name}] Filtering based on Standardized 'NA': Left: {valid_left_mask.sum()} valid, Right: {valid_right_mask.sum()} valid.")

    logger.info(f"[{name}] Extracting features for Left side (all rows)...")
    if 'extract_features' not in globals(): raise NameError(f"extract_features function not found in {__name__}")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info(f"[{name}] Extracting features for Right side (all rows)...")
    right_features_df = extract_features(merged_df, 'Right')
    if left_features_df is None or right_features_df is None:
        logger.error(f"[{name}] Feature extraction returned None. Aborting."); return None, None, None

    filtered_left_features = left_features_df[valid_left_mask].copy()
    filtered_right_features = right_features_df[valid_right_mask].copy()
    # logger.info(f"[{name}] Filtered features shapes: Left {filtered_left_features.shape}, Right {filtered_right_features.shape}.")

    if 'Patient ID' not in merged_df.columns: logger.error(f"[{name}] 'Patient ID' missing post-merge."); return None,None,None
    metadata_left = merged_df.loc[valid_left_mask, ['Patient ID']].copy(); metadata_right = merged_df.loc[valid_right_mask, ['Patient ID']].copy()
    metadata_left['Side'] = 'Left'; metadata_right['Side'] = 'Right'

    valid_left_expert_labels_std = merged_df.loc[valid_left_mask, expert_std_left_col]
    valid_right_expert_labels_std = merged_df.loc[valid_right_mask, expert_std_right_col]

    positive_class_name = CLASS_NAMES.get(1, 'Synkinesis')
    left_targets = valid_left_expert_labels_std.apply(lambda x: 1 if x == positive_class_name else 0).values
    right_targets = valid_right_expert_labels_std.apply(lambda x: 1 if x == positive_class_name else 0).values

    # unique_left, counts_left = np.unique(left_targets, return_counts=True); left_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_left], counts_left))
    # unique_right, counts_right = np.unique(right_targets, return_counts=True); right_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_right], counts_right))
    # logger.info(f"[{name}] Mapped Left Target distribution (post-filter): {left_dist}")
    # logger.info(f"[{name}] Mapped Right Target distribution (post-filter): {right_dist}")

    if filtered_left_features.empty and filtered_right_features.empty: logger.error(f"[{name}] No valid data left after filtering 'NA'."); return None, None, None

    feature_list_path_check = filenames.get('feature_list')
    add_side_indicator = False
    if feature_list_path_check and os.path.exists(feature_list_path_check):
        try:
            final_feature_names_loaded = joblib.load(feature_list_path_check)
            if 'side_indicator' in final_feature_names_loaded:
                add_side_indicator = True
        except Exception as e:
            logger.warning(f"[{name}] Could not load feature list for side indicator check: {e}")

    if add_side_indicator:
         # logger.info(f"[{name}] Adding 'side_indicator' based on loaded feature list.")
         filtered_left_features['side_indicator'] = 0
         filtered_right_features['side_indicator'] = 1
    # else:
        # logger.info(f"[{name}] 'side_indicator' not added (or feature list check failed/skipped).")


    features = pd.concat([filtered_left_features, filtered_right_features], ignore_index=True)
    targets = np.concatenate([left_targets, right_targets])
    metadata = pd.concat([metadata_left, metadata_right], ignore_index=True)
    if not (len(features) == len(targets) == len(metadata)):
        logger.error(f"[{name}] Mismatch lengths after concat! Feat:{len(features)}, Tgt:{len(targets)}, Meta:{len(metadata)}."); return None, None, None
    unique_final, counts_final = np.unique(targets, return_counts=True); final_class_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_final], counts_final))
    logger.info(f"[{name}] FINAL Combined Class distribution for training/split: {final_class_dist} (Total: {len(targets)})")

    features.replace([np.inf, -np.inf], np.nan, inplace=True); features = features.fillna(0)
    initial_cols = features.columns.tolist(); cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"[{name}] Num convert fail {col}. Drop."); cols_to_drop.append(col)
    if cols_to_drop: features = features.drop(columns=cols_to_drop)
    features = features.fillna(0); # logger.info(f"[{name}] Initial features processed: {features.shape[1]}.")

    fs_enabled = feature_sel_cfg.get('enabled', False)
    if fs_enabled:
        # logger.info(f"[{name}] Applying feature selection...")
        n_top = feature_sel_cfg.get('top_n_features'); imp_file = feature_sel_cfg.get('importance_file')
        if not imp_file or not os.path.exists(imp_file) or n_top is None: logger.warning(f"[{name}] FS config invalid. Skip.");
        else:
            try:
                imp_df = pd.read_csv(imp_file); top_names = imp_df['feature'].head(n_top).tolist()
                side_indicator_exists = 'side_indicator' in features.columns
                if side_indicator_exists and 'side_indicator' not in top_names: top_names.append('side_indicator')

                cols_to_keep = [col for col in top_names if col in features.columns]; missing = set(top_names)-set(cols_to_keep)
                if side_indicator_exists: missing -= {'side_indicator'}

                if missing: logger.warning(f"[{name}] FS missing important: {missing}")
                if not cols_to_keep: logger.error(f"[{name}] No features left after FS. Skip.")
                else: logger.info(f"[{name}] Selecting top {len(cols_to_keep)} features."); features = features[cols_to_keep]
            except Exception as e: logger.error(f"[{name}] FS error: {e}. Skip.", exc_info=True)
    # else: logger.info(f"[{name}] Feature selection disabled.")

    logger.info(f"[{name}] Final dataset shape for training: {features.shape}. Targets: {len(targets)}")
    if features.isnull().values.any(): logger.warning(f"NaNs in FINAL feats. Fill 0."); features = features.fillna(0)

    final_feature_names = features.columns.tolist()
    feature_list_path = filenames.get('feature_list')
    if feature_list_path:
        try:
            os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
            joblib.dump(final_feature_names, feature_list_path)
            # logger.info(f"Saved {len(final_feature_names)} feature names to {feature_list_path}")
        except Exception as e:
            logger.error(f"[{name}] Failed save feature list: {e}", exc_info=True)
    else:
        logger.error(f"[{name}] Feature list path not defined.")

    return features, targets, metadata


# --- extract_features (Brow Cocked - Training V3: BL high + RE paretic) ---
def extract_features(df, side):
    """ Extracts V3 Brow Cocked features for TRAINING (BL high rest, RE paretic movement). ET features removed. """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Brow Cocked')
    logger.debug(f"[{name}] Extracting V3 features (BL high, RE paretic) for {side} side (Training)...")
    feature_data = {}
    side_label = side.capitalize()
    # opposite_side_label = 'Right' if side_label == 'Left' else 'Left' # Not strictly needed for all V3 features

    min_val = feature_cfg.get('min_value', 0.0001)
    perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0)

    # Define AUs involved - only brow AUs are relevant for V3
    brow_aus = interest_aus # AU01_r, AU02_r
    all_aus_needed = list(set(brow_aus)) # Simplified

    # Helper to get numeric series
    def get_numeric_series(col_name, default_val=0.0):
        series = df.get(col_name)
        if series is None:
             # logger.warning(f"Column '{col_name}' not found in DataFrame. Using default {default_val}.")
             return pd.Series(default_val, index=df.index, dtype=float)
        return pd.to_numeric(series, errors='coerce').fillna(default_val).astype(float)

    # --- Get Raw Values for BL, RE for BOTH Sides ---
    bl_vals_L, bl_vals_R = {}, {}
    re_vals_L, re_vals_R = {}, {}

    for au in all_aus_needed: # Only brow_aus
        bl_vals_L[au] = get_numeric_series(f"BL_Left {au}")
        bl_vals_R[au] = get_numeric_series(f"BL_Right {au}")
        re_vals_L[au] = get_numeric_series(f"RE_Left {au}")
        re_vals_R[au] = get_numeric_series(f"RE_Right {au}")

    # --- Calculate Intermediate Values (Normalized RE) ---
    norm_re_vals_L, norm_re_vals_R = {}, {}

    for au in brow_aus: # AU01_r, AU02_r
        norm_re_vals_L[au] = (re_vals_L[au] - bl_vals_L.get(au, 0.0)).clip(lower=0)
        norm_re_vals_R[au] = (re_vals_R[au] - bl_vals_R.get(au, 0.0)).clip(lower=0)

    # --- Select Values for the Current Side ---
    current_bl_vals = bl_vals_L if side_label == 'Left' else bl_vals_R
    # current_re_vals = re_vals_L if side_label == 'Left' else re_vals_R # Raw RE not primary for this definition
    current_norm_re_vals = norm_re_vals_L if side_label == 'Left' else norm_re_vals_R

    # --- Feature Calculation ---

    # Group A: Baseline State & Asymmetry (Higher resting on cocked side)
    for au in brow_aus:
        feature_data[f"BL_{au}_raw_target_side"] = current_bl_vals.get(au, pd.Series(0.0, index=df.index))
        feature_data[f"BL_Asym_Ratio_{au}"] = calculate_ratio(bl_vals_L[au], bl_vals_R[au], min_value=min_val)
        feature_data[f"BL_Asym_PercDiff_{au}"] = calculate_percent_diff(bl_vals_L[au], bl_vals_R[au], min_value=min_val, cap=perc_diff_cap)
        feature_data[f"BL_Asym_Diff_LminusR_{au}"] = bl_vals_L[au] - bl_vals_R[au] # Left higher = positive

    # Group B: RE Voluntary Movement & Asymmetry (Paretic movement on cocked side)
    for au in brow_aus:
        # feature_data[f"RE_{au}_raw_target_side"] = current_re_vals.get(au, pd.Series(0.0, index=df.index)) # Less relevant
        feature_data[f"RE_{au}_norm_target_side"] = current_norm_re_vals.get(au, pd.Series(0.0, index=df.index))
        feature_data[f"RE_Norm_Asym_Ratio_{au}"] = calculate_ratio(norm_re_vals_L[au], norm_re_vals_R[au], min_value=min_val)
        feature_data[f"RE_Norm_Asym_PercDiff_{au}"] = calculate_percent_diff(norm_re_vals_L[au], norm_re_vals_R[au], min_value=min_val, cap=perc_diff_cap)
        feature_data[f"RE_Norm_Asym_Diff_LminusR_{au}"] = norm_re_vals_L[au] - norm_re_vals_R[au] # Left moves more = positive

    # ET FEATURES REMOVED FOR V3 DEFINITION

    # --- Combine and Clean ---
    features_df = pd.DataFrame(feature_data, index=df.index)
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0.0)
    logger.debug(f"[{name}] Generated {features_df.shape[1]} V3 features (BL high, RE paretic) for {side} (Training).")
    return features_df


# --- extract_features_for_detection (Brow Cocked - Detection V3) ---
def extract_features_for_detection(row_data, side):
    """ Extracts V3 Brow Cocked features for detection (BL high, RE paretic) based on the saved feature list. """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Brow Cocked')
    # logger.debug(f"[{name}] Extracting V3 detection features (BL high, RE paretic) for {side}...")

    filenames = config.get('filenames', {})
    feature_list_path = filenames.get('feature_list')
    if not feature_list_path or not os.path.exists(feature_list_path):
        logger.error(f"[{name}] Feature list missing: {feature_list_path}. Abort."); return None
    try:
        ordered_feature_names = joblib.load(feature_list_path)
        if not isinstance(ordered_feature_names, list):
             raise TypeError("Loaded feature names is not a list.")
        # logger.debug(f"[{name}] Loaded {len(ordered_feature_names)} features from {feature_list_path}")
    except Exception as e:
        logger.error(f"[{name}] Load feature list failed: {e}"); return None

    if not isinstance(row_data, (pd.Series, dict)):
        logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    if side not in ['Left', 'Right']:
        logger.error(f"Invalid 'side': {side}"); return None
    else:
        side_label = side

    feature_cfg = config.get('feature_extraction', {})
    min_val_cfg = feature_cfg.get('min_value', 0.0001) # Renamed to avoid conflict
    perc_diff_cap_cfg = feature_cfg.get('percent_diff_cap', 200.0) # Renamed
    epsilon = 1e-9

    brow_aus = config.get('interest_aus', ['AU01_r', 'AU02_r'])
    all_aus_needed = list(set(brow_aus))

    feature_dict_generated = {}

    def get_float_value(key, default=0.0):
        val = row_series.get(key, default)
        try:
            f_val = float(val)
            return 0.0 if np.isnan(f_val) or np.isinf(f_val) else f_val
        except (ValueError, TypeError):
            return default

    def scalar_ratio(v1, v2, min_v_func=0.0001): # Use min_v_func to avoid scope issue
        v1_f, v2_f = float(v1), float(v2)
        min_val_local, max_val_local = min(v1_f, v2_f), max(v1_f, v2_f)
        if max_val_local <= min_v_func: return 1.0
        if min_val_local <= min_v_func: return 0.0
        return np.clip(np.nan_to_num(min_val_local / (max_val_local + epsilon), nan=1.0), 0.0, 1.0)

    def scalar_perc_diff(v1, v2, min_v_func=0.0001, cap_func=200.0): # Use min_v_func, cap_func
        v1_f, v2_f = float(v1), float(v2)
        diff = abs(v1_f - v2_f); avg = (v1_f + v2_f) / 2.0; pdiff = 0.0
        if avg > min_v_func: pdiff = (diff / (avg + epsilon)) * 100.0
        elif diff > min_v_func: pdiff = cap_func
        return np.clip(np.nan_to_num(pdiff, nan=0.0, posinf=cap_func, neginf=0.0), 0, cap_func)

    bl_vals_L, bl_vals_R = {}, {}; re_vals_L, re_vals_R = {}, {}
    for au in all_aus_needed:
        bl_vals_L[au] = get_float_value(f"BL_Left {au}")
        bl_vals_R[au] = get_float_value(f"BL_Right {au}")
        re_vals_L[au] = get_float_value(f"RE_Left {au}")
        re_vals_R[au] = get_float_value(f"RE_Right {au}")

    norm_re_vals_L, norm_re_vals_R = {}, {}
    for au in brow_aus:
        norm_re_vals_L[au] = max(0.0, re_vals_L[au] - bl_vals_L.get(au, 0.0))
        norm_re_vals_R[au] = max(0.0, re_vals_R[au] - bl_vals_R.get(au, 0.0))

    current_bl_vals = bl_vals_L if side_label == 'Left' else bl_vals_R
    current_norm_re_vals = norm_re_vals_L if side_label == 'Left' else norm_re_vals_R

    # Group A: Baseline
    for au in brow_aus:
        feature_dict_generated[f"BL_{au}_raw_target_side"] = current_bl_vals.get(au, 0.0)
        feature_dict_generated[f"BL_Asym_Ratio_{au}"] = scalar_ratio(bl_vals_L[au], bl_vals_R[au], min_v_func=min_val_cfg)
        feature_dict_generated[f"BL_Asym_PercDiff_{au}"] = scalar_perc_diff(bl_vals_L[au], bl_vals_R[au], min_v_func=min_val_cfg, cap_func=perc_diff_cap_cfg)
        feature_dict_generated[f"BL_Asym_Diff_LminusR_{au}"] = bl_vals_L[au] - bl_vals_R[au]

    # Group B: RE Movement
    for au in brow_aus:
        feature_dict_generated[f"RE_{au}_norm_target_side"] = current_norm_re_vals.get(au, 0.0)
        feature_dict_generated[f"RE_Norm_Asym_Ratio_{au}"] = scalar_ratio(norm_re_vals_L[au], norm_re_vals_R[au], min_v_func=min_val_cfg)
        feature_dict_generated[f"RE_Norm_Asym_PercDiff_{au}"] = scalar_perc_diff(norm_re_vals_L[au], norm_re_vals_R[au], min_v_func=min_val_cfg, cap_func=perc_diff_cap_cfg)
        feature_dict_generated[f"RE_Norm_Asym_Diff_LminusR_{au}"] = norm_re_vals_L[au] - norm_re_vals_R[au]

    # ET FEATURES REMOVED

    if "side_indicator" in ordered_feature_names:
        feature_dict_generated["side_indicator"] = 0 if side.lower() == 'left' else 1
        # logger.debug(f"[{name}] Added 'side_indicator' as required by feature list.")

    feature_list = []
    missing = []
    type_err = []
    for feat_name in ordered_feature_names:
        value = feature_dict_generated.get(feat_name)
        final_val = 0.0
        if value is None: missing.append(feat_name)
        else:
            try:
                temp_val = float(value)
                if not (np.isnan(temp_val) or np.isinf(temp_val)): final_val = temp_val
            except (ValueError, TypeError): type_err.append(feat_name)
        feature_list.append(final_val)

    if missing:
        logger.error(f"[{name}] Detect V3 ({side}): CRITICAL - Missing features required by list: {missing}. Calc logic error?")
        return None
    if type_err:
        logger.warning(f"[{name}] Detect V3 ({side}): Type errors for features {type_err}. Using 0.")

    if len(feature_list) != len(ordered_feature_names):
        logger.error(f"[{name}] Detect V3 ({side}): FINAL Feature list length mismatch. Aborting.")
        return None

    # logger.debug(f"[{name}] Generated {len(feature_list)} V3 detection features for {side} matching loaded list.")
    return feature_list

# --- END OF FILE brow_cocked_features.py ---