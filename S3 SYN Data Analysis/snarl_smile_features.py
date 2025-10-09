# snarl_smile_features.py (Refactored to use Utils and Config - V7 Features)

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
SYNK_TYPE = 'snarl_smile'

# Get type-specific config
try:
    config = SYNKINESIS_CONFIG[SYNK_TYPE]
    feature_cfg = config.get('feature_extraction', {})
    actions = config.get('relevant_actions', ['BS']) # Should be just BS for V7
    trigger_aus = config.get('trigger_aus', ['AU12_r']) # Should be just AU12 for V7
    coupled_aus = config.get('coupled_aus', ['AU10_r', 'AU14_r', 'AU15_r']) # V7 Coupled
    feature_sel_cfg = config.get('feature_selection', {})
    filenames = config.get('filenames', {})
    expert_cols = config.get('expert_columns', {})
    target_cols = config.get('target_columns', {})
    CONFIG_LOADED = True
    # V7 Sanity Checks
    if actions != ['BS']: logger.warning(f"Snarl-Smile config actions '{actions}' differ from expected V7 ['BS']")
    if trigger_aus != ['AU12_r']: logger.warning(f"Snarl-Smile config trigger_aus '{trigger_aus}' differ from expected V7 ['AU12_r']")
    if coupled_aus != ['AU10_r', 'AU14_r', 'AU15_r']: logger.warning(f"Snarl-Smile config coupled_aus '{coupled_aus}' differ from expected V7 ['AU10_r', 'AU14_r', 'AU15_r']")
except KeyError:
    logger.critical(f"CRITICAL: Synkinesis type '{SYNK_TYPE}' not found in SYNKINESIS_CONFIG.")
    CONFIG_LOADED = False; config = {}; feature_cfg = {}; actions = ['BS']; trigger_aus = ['AU12_r']; coupled_aus = ['AU10_r', 'AU14_r', 'AU15_r']; feature_sel_cfg = {}; filenames = {}; expert_cols = {}; target_cols = {}


# --- prepare_data function (Refactored for V7 - Expects 15 Features) ---
def prepare_data(results_file=None, expert_file=None):
    """
    Prepares data for the specific SYNK_TYPE, handling NA/Not Assessed labels,
    feature selection, and returning metadata alongside features and targets.
    Maps standardized labels correctly for binary classification.
    """
    # --- Configuration Setup ---
    try:
        # SYNK_TYPE needs to be defined globally at the top of the specific feature file
        if 'SYNK_TYPE' not in globals(): raise NameError("SYNK_TYPE variable is not defined globally.")
        config = SYNKINESIS_CONFIG[SYNK_TYPE]
        name = config.get('name', SYNK_TYPE.replace('_', ' ').title())
        feature_cfg = config.get('feature_extraction', {})
        feature_sel_cfg = config.get('feature_selection', {})
        filenames = config.get('filenames', {})
        expert_cols_cfg = config.get('expert_columns', {})
        target_cols_cfg = config.get('target_columns', {})
        CONFIG_LOADED = True
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
            dtype=str,             # Keep reading as string
            keep_default_na=False, # *** Do NOT interpret default NA values as NaN ***
            na_values=['']         # *** Treat ONLY empty strings as true NA/NaN ***
        )
        logger.info(f"[{name}] Loaded {len(results_df)} results rows, {len(expert_df)} expert key rows")
    except Exception as e: logger.error(f"[{name}] Error loading data: {e}.", exc_info=True); return None, None, None

    # Rename/Process Expert Columns using config
    exp_left_orig = expert_cols_cfg.get('left'); exp_right_orig = expert_cols_cfg.get('right')
    target_left_col = target_cols_cfg.get('left'); target_right_col = target_cols_cfg.get('right')
    if not exp_left_orig or not exp_right_orig or not target_left_col or not target_right_col:
         logger.error(f"[{name}] Config missing expert/target column names. Abort."); return None, None, None
    if exp_left_orig not in expert_df.columns or exp_right_orig not in expert_df.columns:
         logger.error(f"[{name}] Raw expert columns ('{exp_left_orig}', '{exp_right_orig}') not found in expert key. Abort."); return None, None, None
    rename_map = {'Patient': 'Patient ID', exp_left_orig: exp_left_orig, exp_right_orig: exp_right_orig}
    expert_df = expert_df.rename(columns={k: v for k,v in rename_map.items() if k in expert_df.columns})

    # --- Standardize Expert Labels ---
    expert_std_left_col = f'Expert_Std_Left_{SYNK_TYPE}'
    expert_std_right_col = f'Expert_Std_Right_{SYNK_TYPE}'
    expert_df[expert_std_left_col] = expert_df[exp_left_orig].apply(standardize_synkinesis_labels)
    expert_df[expert_std_right_col] = expert_df[exp_right_orig].apply(standardize_synkinesis_labels)
    logger.info(f"[{name}] Standardized expert labels created.")
    logger.info(f"Value counts Std Left:\n{expert_df[expert_std_left_col].value_counts(dropna=False).to_string()}")
    logger.info(f"Value counts Std Right:\n{expert_df[expert_std_right_col].value_counts(dropna=False).to_string()}")

    # Prepare subset for merge
    expert_cols_to_merge = ['Patient ID', expert_std_left_col, expert_std_right_col]
    expert_df_subset = expert_df[expert_cols_to_merge].copy()
    if expert_df_subset['Patient ID'].duplicated().any():
        logger.warning(f"[{name}] Duplicate Patient IDs found in expert key subset. Keeping first.")
        expert_df_subset.drop_duplicates(subset=['Patient ID'], keep='first', inplace=True)

    # Merge
    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df_subset['Patient ID'] = expert_df_subset['Patient ID'].astype(str).str.strip()
    try: merged_df = pd.merge(results_df, expert_df_subset, on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"[{name}] Merge failed: {e}", exc_info=True); return None, None, None
    logger.info(f"[{name}] Merged data: {len(merged_df)} patients")
    if merged_df.empty or expert_std_left_col not in merged_df.columns or expert_std_right_col not in merged_df.columns:
         logger.error(f"[{name}] Merge empty or standardized expert columns missing. Abort."); return None, None, None

    # --- Identify Valid Rows based on Standardized Labels ---
    valid_left_mask = merged_df[expert_std_left_col] != 'NA'
    valid_right_mask = merged_df[expert_std_right_col] != 'NA'
    logger.info(f"[{name}] Filtering based on Standardized 'NA': Left: {valid_left_mask.sum()} valid, Right: {valid_right_mask.sum()} valid.")

    # --- Feature Extraction (using the specific function from the file) ---
    logger.info(f"[{name}] Extracting features for Left side (all rows)...")
    if 'extract_features' not in globals(): raise NameError(f"extract_features function not found in {__name__}")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info(f"[{name}] Extracting features for Right side (all rows)...")
    right_features_df = extract_features(merged_df, 'Right')
    if left_features_df is None or right_features_df is None:
        logger.error(f"[{name}] Feature extraction returned None. Aborting."); return None, None, None

    # --- Filter Features based on VALID masks ---
    filtered_left_features = left_features_df[valid_left_mask].copy()
    filtered_right_features = right_features_df[valid_right_mask].copy()
    logger.info(f"[{name}] Filtered features shapes: Left {filtered_left_features.shape}, Right {filtered_right_features.shape}.")

    # --- Filter Metadata ---
    if 'Patient ID' not in merged_df.columns: logger.error(f"[{name}] 'Patient ID' missing post-merge."); return None,None,None
    metadata_left = merged_df.loc[valid_left_mask, ['Patient ID']].copy(); metadata_right = merged_df.loc[valid_right_mask, ['Patient ID']].copy()
    metadata_left['Side'] = 'Left'; metadata_right['Side'] = 'Right'

    # --- Filter and Process Targets based on VALID masks ---
    valid_left_expert_labels_std = merged_df.loc[valid_left_mask, expert_std_left_col]
    valid_right_expert_labels_std = merged_df.loc[valid_right_mask, expert_std_right_col]

    # --- Map standardized labels directly to 0/1 for binary ---
    # Anything not 'None' after standardization is treated as the positive class (1)
    # This now correctly handles 'Synkinesis' (from 'Yes') and ignores 'Partial'/'Complete' if they weren't standardized to 'NA'
    left_targets = valid_left_expert_labels_std.apply(lambda x: 1 if x != 'None' else 0).values
    right_targets = valid_right_expert_labels_std.apply(lambda x: 1 if x != 'None' else 0).values
    # --- End Target Mapping Change ---

    unique_left, counts_left = np.unique(left_targets, return_counts=True); left_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_left], counts_left))
    unique_right, counts_right = np.unique(right_targets, return_counts=True); right_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_right], counts_right))
    logger.info(f"[{name}] Mapped Left Target distribution (post-filter): {left_dist}")
    logger.info(f"[{name}] Mapped Right Target distribution (post-filter): {right_dist}")

    # --- Combine Filtered Data ---
    if filtered_left_features.empty and filtered_right_features.empty: logger.error(f"[{name}] No valid data left after filtering 'NA'."); return None, None, None

    # --- Side Indicator Handling (Conditional - Check if needed by specific model/features) ---
    # Most of the synkinesis models (like V7 Snarl, Brow Cocked) calculate asymmetry
    # features internally or don't use side indicator. Add it back ONLY if the feature
    # list saved during training for a specific type *includes* 'side_indicator'.
    # By default, we won't add it here. It should be added in extract_features if required.
    # Example check (adjust as needed):
    # if 'side_indicator' in final_feature_names_loaded_from_list: # Pseudo-code
    #      filtered_left_features['side_indicator'] = 0
    #      filtered_right_features['side_indicator'] = 1
    #      logger.info(f"[{name}] Added 'side_indicator' based on feature list.")

    features = pd.concat([filtered_left_features, filtered_right_features], ignore_index=True)
    targets = np.concatenate([left_targets, right_targets])
    metadata = pd.concat([metadata_left, metadata_right], ignore_index=True)
    if not (len(features) == len(targets) == len(metadata)):
        logger.error(f"[{name}] Mismatch lengths after concat! Feat:{len(features)}, Tgt:{len(targets)}, Meta:{len(metadata)}."); return None, None, None
    unique_final, counts_final = np.unique(targets, return_counts=True); final_class_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_final], counts_final))
    logger.info(f"[{name}] FINAL Combined Class distribution for training/split: {final_class_dist} (Total: {len(targets)})")

    # --- Post-processing & Feature Selection ---
    features.replace([np.inf, -np.inf], np.nan, inplace=True); features = features.fillna(0)
    initial_cols = features.columns.tolist(); cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"[{name}] Num convert fail {col}. Drop."); cols_to_drop.append(col)
    if cols_to_drop: features = features.drop(columns=cols_to_drop)
    features = features.fillna(0); logger.info(f"[{name}] Initial features processed: {features.shape[1]}.")

    # Feature Selection
    fs_enabled = feature_sel_cfg.get('enabled', False)
    if fs_enabled:
        logger.info(f"[{name}] Applying feature selection...")
        n_top = feature_sel_cfg.get('top_n_features'); imp_file = feature_sel_cfg.get('importance_file')
        if not imp_file or not os.path.exists(imp_file) or n_top is None: logger.warning(f"[{name}] FS config invalid. Skip.");
        else:
            try:
                imp_df = pd.read_csv(imp_file); top_names = imp_df['feature'].head(n_top).tolist()
                # Handle side indicator if it exists and FS is on
                side_indicator_exists = 'side_indicator' in features.columns
                if side_indicator_exists and 'side_indicator' not in top_names: top_names.append('side_indicator')

                cols_to_keep = [col for col in top_names if col in features.columns]; missing = set(top_names)-set(cols_to_keep)
                if side_indicator_exists: missing -= {'side_indicator'} # Don't warn about side indicator

                if missing: logger.warning(f"[{name}] FS missing important: {missing}")
                if not cols_to_keep: logger.error(f"[{name}] No features left after FS. Skip.")
                else: logger.info(f"[{name}] Selecting top {len(cols_to_keep)} features."); features = features[cols_to_keep]
            except Exception as e: logger.error(f"[{name}] FS error: {e}. Skip.", exc_info=True)
    else: logger.info(f"[{name}] Feature selection disabled.")

    logger.info(f"[{name}] Final dataset shape for training: {features.shape}. Targets: {len(targets)}")
    if features.isnull().values.any(): logger.warning(f"NaNs in FINAL feats. Fill 0."); features = features.fillna(0)

    # --- Save final feature list ---
    final_feature_names = features.columns.tolist()
    feature_list_path = filenames.get('feature_list')
    if feature_list_path:
        try: os.makedirs(os.path.dirname(feature_list_path), exist_ok=True); joblib.dump(final_feature_names, feature_list_path); logger.info(f"Saved {len(final_feature_names)} feature names to {feature_list_path}")
        except Exception as e: logger.error(f"[{name}] Failed save feature list: {e}", exc_info=True)
    else: logger.error(f"[{name}] Feature list path not defined.")

    # Return features, targets, and metadata
    return features, targets, metadata

# --- extract_features function (Snarl-Smile - Training V7 - 15 Features) ---
# This function generates the specific 15 features for V7
def extract_features(df, side):
    """ Extracts V7 (15) Snarl-Smile features for TRAINING. """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Snarl-Smile')
    logger.debug(f"[{name}] Extracting V7 features for {side} side (Training)...")
    feature_data = {}
    side_label = side.capitalize()
    opposite_side_label = 'Right' if side_label == 'Left' else 'Left'

    use_normalized = feature_cfg.get('use_normalized', True)
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0)
    all_aus_involved = trigger_aus + coupled_aus # V7 AUs

    # Helper to get numeric series
    def get_numeric_series(col_name, default_val=0.0):
        series = df.get(col_name)
        return pd.to_numeric(series, errors='coerce').fillna(default_val).astype(float) if series is not None else pd.Series(default_val, index=df.index, dtype=float)

    # Get BL and BS raw values for BOTH sides
    bl_vals_L, bl_vals_R = {}, {}
    bs_vals_L, bs_vals_R = {}, {}
    for au in all_aus_involved:
        bl_vals_L[au] = get_numeric_series(f"BL_Left {au}")
        bl_vals_R[au] = get_numeric_series(f"BL_Right {au}")
        bs_vals_L[au] = get_numeric_series(f"BS_Left {au}")
        bs_vals_R[au] = get_numeric_series(f"BS_Right {au}")

    # Calculate Normalized Values for BOTH sides
    norm_vals_L, norm_vals_R = {}, {}
    if use_normalized:
        for au in all_aus_involved:
            norm_vals_L[au] = (bs_vals_L[au] - bl_vals_L.get(au, 0.0)).clip(lower=0)
            norm_vals_R[au] = (bs_vals_R[au] - bl_vals_R.get(au, 0.0)).clip(lower=0)
    else:
        norm_vals_L = bs_vals_L
        norm_vals_R = bs_vals_R # Use raw if norm disabled

    # Get values for the TARGET side
    current_side_norm_vals = norm_vals_L if side_label == 'Left' else norm_vals_R
    trig_au = trigger_aus[0] # AU12

    # A. Single-Side Norm AUs (4 features)
    feature_data[f"BS_{trig_au}_trig_norm"] = current_side_norm_vals.get(trig_au, pd.Series(0.0, index=df.index))
    feature_data[f"BS_AU10_r_coup_norm"] = current_side_norm_vals.get('AU10_r', pd.Series(0.0, index=df.index))
    feature_data[f"BS_AU14_r_coup_norm"] = current_side_norm_vals.get('AU14_r', pd.Series(0.0, index=df.index))
    feature_data[f"BS_AU15_r_coup_norm"] = current_side_norm_vals.get('AU15_r', pd.Series(0.0, index=df.index))

    # B. Single-Side Ratios (4 features)
    trigger_norm_series = feature_data[f"BS_{trig_au}_trig_norm"]
    norm_au10 = feature_data[f"BS_AU10_r_coup_norm"]
    norm_au14 = feature_data[f"BS_AU14_r_coup_norm"]
    norm_au15 = feature_data[f"BS_AU15_r_coup_norm"]
    feature_data[f"BS_Ratio_AU10_vs_AU12"] = calculate_ratio(norm_au10, trigger_norm_series, min_value=min_val)
    feature_data[f"BS_Ratio_AU14_vs_AU12"] = calculate_ratio(norm_au14, trigger_norm_series, min_value=min_val)
    feature_data[f"BS_Ratio_AU15_vs_AU12"] = calculate_ratio(norm_au15, trigger_norm_series, min_value=min_val)
    feature_data[f"BS_Ratio_AU15_vs_AU10"] = calculate_ratio(norm_au15, norm_au10, min_value=min_val)

    # C. Max-Across-Sides Features (3 features)
    feature_data['Max_BS_AU10_Norm'] = np.maximum(norm_vals_L.get('AU10_r', 0.0), norm_vals_R.get('AU10_r', 0.0))
    feature_data['Max_BS_AU15_Norm'] = np.maximum(norm_vals_L.get('AU15_r', 0.0), norm_vals_R.get('AU15_r', 0.0))
    # Need to calculate ratios for both sides first
    ratio_L_15_10 = calculate_ratio(norm_vals_L.get('AU15_r',0.0), norm_vals_L.get('AU10_r',0.0), min_value=min_val)
    ratio_R_15_10 = calculate_ratio(norm_vals_R.get('AU15_r',0.0), norm_vals_R.get('AU10_r',0.0), min_value=min_val)
    feature_data['Max_BS_Ratio_AU15_vs_AU10'] = np.maximum(ratio_L_15_10, ratio_R_15_10)

    # D. Targeted Asymmetry Features (4 features)
    left_norm_au12 = norm_vals_L.get(trig_au, 0.0)
    right_norm_au12 = norm_vals_R.get(trig_au, 0.0)
    left_norm_au15 = norm_vals_L.get('AU15_r', 0.0)
    right_norm_au15 = norm_vals_R.get('AU15_r', 0.0)
    feature_data['Asym_Ratio_BS_AU12_Norm'] = calculate_ratio(pd.Series(left_norm_au12), pd.Series(right_norm_au12), min_value=min_val)
    feature_data['Asym_PercDiff_BS_AU12_Norm'] = calculate_percent_diff(pd.Series(left_norm_au12), pd.Series(right_norm_au12), min_value=min_val, cap=perc_diff_cap)
    feature_data['Asym_Ratio_BS_AU15_Norm'] = calculate_ratio(pd.Series(left_norm_au15), pd.Series(right_norm_au15), min_value=min_val)
    feature_data['Asym_PercDiff_BS_AU15_Norm'] = calculate_percent_diff(pd.Series(left_norm_au15), pd.Series(right_norm_au15), min_value=min_val, cap=perc_diff_cap)

    # E. REMOVE side_indicator for V7 feature set

    features_df = pd.DataFrame(feature_data, index=df.index)
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0.0)
    logger.debug(f"[{name}] Generated {features_df.shape[1]} V7 features for {side} (Training).")
    # Assert correct number of features before returning
    expected_v7_count = 15
    if features_df.shape[1] != expected_v7_count:
        logger.error(f"[{name}] Incorrect number of V7 features generated ({features_df.shape[1]} vs {expected_v7_count}). Check logic.")
        logger.error(f"Generated columns: {features_df.columns.tolist()}")
        return None # Indicate failure
    return features_df

# --- extract_features_for_detection (Detection - V7 - 15 Features) ---
def extract_features_for_detection(row_data, side):
    """ Extracts V7 (15) Snarl-Smile features for detection. """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Snarl-Smile')
    logger.debug(f"[{name}] Extracting V7 detection features for {side}...")

    feature_list_path = filenames.get('feature_list')
    if not feature_list_path or not os.path.exists(feature_list_path):
        logger.error(f"[{name}] Feature list missing. Abort."); return None
    try:
        ordered_feature_names = joblib.load(feature_list_path)
        assert isinstance(ordered_feature_names, list)
    except Exception as e:
        logger.error(f"[{name}] Load feature list failed: {e}"); return None
    # V7 Check: Ensure the loaded list has 15 features
    expected_v7_count = 15
    if len(ordered_feature_names) != expected_v7_count:
        logger.warning(f"[{name}] Loaded feature list '{feature_list_path}' has {len(ordered_feature_names)} features, expected {expected_v7_count} for V7.")
        # Continue for now, but this indicates a mismatch

    if not isinstance(row_data, (pd.Series, dict)):
        logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    pid_for_log = row_series.get('Patient ID', 'UnknownPID')
    if side not in ['Left', 'Right']:
        logger.error(f"Invalid 'side': {side}"); return None
    else:
        side_label = side

    use_normalized = feature_cfg.get('use_normalized', True)
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0)
    epsilon = 1e-9
    all_aus_involved = trigger_aus + coupled_aus
    trig_au = trigger_aus[0]
    feature_dict_final = {}

    def get_float_value(key, default=0.0):
        val = row_series.get(key, default)
        try:
            f_val = float(val)
            return 0.0 if np.isnan(f_val) or np.isinf(f_val) else f_val
        except (ValueError, TypeError):
            return default

    # --- CORRECTED scalar_ratio ---
    def scalar_ratio(v1, v2, min_v=0.0001):
        v1_f = float(v1)
        v2_f = float(v2)
        min_val_local = min(v1_f, v2_f) # Use different name
        max_val_local = max(v1_f, v2_f) # Use different name
        epsilon = 1e-9
        if max_val_local <= min_v:
             return 1.0
        if min_val_local <= min_v: # Use min_val_local
             return 0.0
        # Use min_val_local and max_val_local
        return np.clip(np.nan_to_num(min_val_local / (max_val_local + epsilon), nan=1.0), 0.0, 1.0)
    # --- END CORRECTION ---

    def scalar_perc_diff(v1,v2,min_v=0.0001,cap=200.0):
        v1_f,v2_f = float(v1),float(v2)
        diff=abs(v1_f-v2_f)
        avg=(v1_f+v2_f)/2.0
        pdiff=0.0
        if avg>min_v: pdiff=(diff/(avg+epsilon))*100.0
        elif diff>min_v: pdiff=cap
        return np.clip(np.nan_to_num(pdiff,nan=0.0,posinf=cap,neginf=0.0),0,cap)

    bs_vals_L, bs_vals_R = {}, {}
    bl_vals_L, bl_vals_R = {}, {}
    for au in all_aus_involved:
        bs_vals_L[au] = get_float_value(f"BS_Left {au}")
        bs_vals_R[au] = get_float_value(f"BS_Right {au}")
        bl_vals_L[au] = get_float_value(f"BL_Left {au}")
        bl_vals_R[au] = get_float_value(f"BL_Right {au}")
    norm_vals_L, norm_vals_R = {}, {}
    if use_normalized:
        for au in all_aus_involved:
            norm_vals_L[au] = max(0.0, bs_vals_L[au] - bl_vals_L.get(au, 0.0))
            norm_vals_R[au] = max(0.0, bs_vals_R[au] - bl_vals_R.get(au, 0.0))
    else:
        norm_vals_L = bs_vals_L
        norm_vals_R = bs_vals_R
    ratio_vals_L, ratio_vals_R = {}, {} # Calculate ratios for both sides
    norm_L_trig = norm_vals_L.get(trig_au, 0.0)
    norm_R_trig = norm_vals_R.get(trig_au, 0.0)
    norm_L_au10 = norm_vals_L.get('AU10_r', 0.0)
    norm_R_au10 = norm_vals_R.get('AU10_r', 0.0)
    norm_L_au14 = norm_vals_L.get('AU14_r', 0.0)
    norm_R_au14 = norm_vals_R.get('AU14_r', 0.0)
    norm_L_au15 = norm_vals_L.get('AU15_r', 0.0)
    norm_R_au15 = norm_vals_R.get('AU15_r', 0.0)
    ratio_vals_L['AU10_vs_AU12']=scalar_ratio(norm_L_au10, norm_L_trig, min_v=min_val) # Uses corrected scalar_ratio
    ratio_vals_R['AU10_vs_AU12']=scalar_ratio(norm_R_au10, norm_R_trig, min_v=min_val) # Uses corrected scalar_ratio
    ratio_vals_L['AU14_vs_AU12']=scalar_ratio(norm_L_au14, norm_L_trig, min_v=min_val) # Uses corrected scalar_ratio
    ratio_vals_R['AU14_vs_AU12']=scalar_ratio(norm_R_au14, norm_R_trig, min_v=min_val) # Uses corrected scalar_ratio
    ratio_vals_L['AU15_vs_AU12']=scalar_ratio(norm_L_au15, norm_L_trig, min_v=min_val) # Uses corrected scalar_ratio
    ratio_vals_R['AU15_vs_AU12']=scalar_ratio(norm_R_au15, norm_R_trig, min_v=min_val) # Uses corrected scalar_ratio
    ratio_vals_L['AU15_vs_AU10']=scalar_ratio(norm_L_au15, norm_L_au10, min_v=min_val) # Uses corrected scalar_ratio
    ratio_vals_R['AU15_vs_AU10']=scalar_ratio(norm_R_au15, norm_R_au10, min_v=min_val) # Uses corrected scalar_ratio

    current_side_norm_vals = norm_vals_L if side_label == 'Left' else norm_vals_R
    current_side_ratio_vals = ratio_vals_L if side_label == 'Left' else ratio_vals_R
    # A. Single-Side Norm AUs
    feature_dict_final[f"BS_{trig_au}_trig_norm"] = current_side_norm_vals.get(trig_au, 0.0)
    feature_dict_final[f"BS_AU10_r_coup_norm"] = current_side_norm_vals.get('AU10_r', 0.0)
    feature_dict_final[f"BS_AU14_r_coup_norm"] = current_side_norm_vals.get('AU14_r', 0.0)
    feature_dict_final[f"BS_AU15_r_coup_norm"] = current_side_norm_vals.get('AU15_r', 0.0)
    # B. Single-Side Ratios
    feature_dict_final[f"BS_Ratio_AU10_vs_AU12"] = current_side_ratio_vals.get('AU10_vs_AU12', 0.0)
    feature_dict_final[f"BS_Ratio_AU14_vs_AU12"] = current_side_ratio_vals.get('AU14_vs_AU12', 0.0)
    feature_dict_final[f"BS_Ratio_AU15_vs_AU12"] = current_side_ratio_vals.get('AU15_vs_AU12', 0.0)
    feature_dict_final[f"BS_Ratio_AU15_vs_AU10"] = current_side_ratio_vals.get('AU15_vs_AU10', 0.0)
    # C. Max-Across-Sides
    feature_dict_final['Max_BS_AU10_Norm'] = max(norm_vals_L.get('AU10_r', 0.0), norm_vals_R.get('AU10_r', 0.0))
    feature_dict_final['Max_BS_AU15_Norm'] = max(norm_vals_L.get('AU15_r', 0.0), norm_vals_R.get('AU15_r', 0.0))
    feature_dict_final['Max_BS_Ratio_AU15_vs_AU10'] = max(ratio_vals_L.get('AU15_vs_AU10',0.0), ratio_vals_R.get('AU15_vs_AU10',0.0))
    # D. Targeted Asymmetry
    feature_dict_final['Asym_Ratio_BS_AU12_Norm'] = scalar_ratio(norm_L_trig, norm_R_trig, min_v=min_val) # Uses corrected scalar_ratio
    feature_dict_final['Asym_PercDiff_BS_AU12_Norm'] = scalar_perc_diff(norm_L_trig, norm_R_trig, min_v=min_val, cap=perc_diff_cap)
    feature_dict_final['Asym_Ratio_BS_AU15_Norm'] = scalar_ratio(norm_L_au15, norm_R_au15, min_v=min_val) # Uses corrected scalar_ratio
    feature_dict_final['Asym_PercDiff_BS_AU15_Norm'] = scalar_perc_diff(norm_L_au15, norm_R_au15, min_v=min_val, cap=perc_diff_cap)
    # <<< REMOVED side_indicator >>>

    # --- Build final ordered list ---
    feature_list = []
    missing = []
    type_err = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name)
        final_val = 0.0 # Default value

        if value is None:
            missing.append(name)
            # final_val remains 0.0
        else:
            try:
                temp_val = float(value)
                # Check for NaN/Inf *after* successful conversion
                if np.isnan(temp_val) or np.isinf(temp_val):
                    pass # final_val is already 0.0
                else:
                    final_val = temp_val # Assign the valid float
            except (ValueError, TypeError):
                type_err.append(name)
                # final_val remains 0.0
        feature_list.append(final_val) # Append the determined value

    if missing:
        logger.warning(f"[{name}] Detect V7 ({side}): Missing features {missing}. Using 0.")
    if type_err:
        logger.warning(f"[{name}] Detect V7 ({side}): Type errors {type_err}. Using 0.")
    # Check final length against expected V7 count
    if len(feature_list) != expected_v7_count:
        logger.error(f"CRITICAL MISMATCH SnSm V7: List len ({len(feature_list)}) != expected ({expected_v7_count}). Expected: {ordered_feature_names}. Got: {sorted(list(feature_dict_final.keys()))}")
        return None
    logger.debug(f"[{name}] Generated {len(feature_list)} V7 detection features for {side}.")
    return feature_list


# --- process_targets (Copied) ---
def process_targets(target_series):
    # ... (Identical implementation) ...
    if target_series is None: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    s_filled = target_series.fillna('no')
    s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty:
        logger.warning(f"Unexpected labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int).values