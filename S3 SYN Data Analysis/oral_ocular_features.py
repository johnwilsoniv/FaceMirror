# oral_ocular_features.py (Refactored to use Utils and Config)

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
SYNK_TYPE = 'oral_ocular'

# Get type-specific config
try:
    config = SYNKINESIS_CONFIG[SYNK_TYPE]
    feature_cfg = config.get('feature_extraction', {})
    actions = config.get('relevant_actions', []) # Use relevant_actions
    trigger_aus = config.get('trigger_aus', [])
    coupled_aus = config.get('coupled_aus', [])
    feature_sel_cfg = config.get('feature_selection', {})
    filenames = config.get('filenames', {})
    expert_cols = config.get('expert_columns', {})
    target_cols = config.get('target_columns', {})
    CONFIG_LOADED = True
except KeyError:
    logger.critical(f"CRITICAL: Synkinesis type '{SYNK_TYPE}' not found in SYNKINESIS_CONFIG.")
    CONFIG_LOADED = False; config = {}; feature_cfg = {}; actions = []; trigger_aus = []; coupled_aus = []; feature_sel_cfg = {}; filenames = {}; expert_cols = {}; target_cols = {}

# --- prepare_data function (Refactored) ---
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

# --- extract_features function (Oral-Ocular - Training) ---
# This uses the corrected logic provided before
def extract_features(df, side):
    """ Extracts Oral-Ocular features for TRAINING (50 features version). """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Oral-Ocular'); logger.debug(f"[{name}] Extracting features for {side} side (Training)...")
    feature_data = {}; side_label = side.capitalize(); opposite_side_label = 'Right' if side_label == 'Left' else 'Left'
    use_normalized = feature_cfg.get('use_normalized', True); min_val = feature_cfg.get('min_value', 0.0001); perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0); norm_suffix = " (Normalized)" if use_normalized else ""

    def get_numeric_series(col_name, default_val=0.0):
        series = df.get(col_name)
        return pd.to_numeric(series, errors='coerce').fillna(default_val).astype(float) if series is not None else pd.Series(default_val, index=df.index, dtype=float)

    all_action_features = {}
    for action in actions:
        action_features = {}
        for trig_au in trigger_aus:
            col_raw = f"{action}_{side_label} {trig_au}"; col_norm = f"{col_raw}{norm_suffix}"; raw_val_side = get_numeric_series(col_raw); val_side_to_use = raw_val_side
            if use_normalized: val_side_to_use = get_numeric_series(col_norm, default_val=raw_val_side)
            action_features[f"{action}_{trig_au}_trig_norm"] = val_side_to_use
        for coup_au in coupled_aus:
            col_raw = f"{action}_{side_label} {coup_au}"; col_norm = f"{col_raw}{norm_suffix}"; raw_val_side = get_numeric_series(col_raw); val_side_to_use = raw_val_side
            if use_normalized: val_side_to_use = get_numeric_series(col_norm, default_val=raw_val_side)
            coup_series = val_side_to_use; action_features[f"{action}_{coup_au}_coup_norm"] = coup_series
            for trig_au in trigger_aus:
                trig_series = action_features.get(f"{action}_{trig_au}_trig_norm", pd.Series(0.0, index=df.index))
                action_features[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = calculate_ratio(coup_series, trig_series, min_value=min_val)
        all_action_features.update(action_features)
    feature_data.update(all_action_features)

    bs_asym_ratio_list = []; bs_asym_percdiff_list = []
    for coup_au in coupled_aus:
        bs_coup_col_left_raw = f"BS_Left {coup_au}"; bs_coup_col_right_raw = f"BS_Right {coup_au}"; bs_coup_col_left_norm = f"BS_Left {coup_au}{norm_suffix}"; bs_coup_col_right_norm = f"BS_Right {coup_au}{norm_suffix}"; bs_coup_raw_left = get_numeric_series(bs_coup_col_left_raw); bs_coup_raw_right = get_numeric_series(bs_coup_col_right_raw); bs_coup_left = bs_coup_raw_left; bs_coup_right = bs_coup_raw_right
        if use_normalized: bs_coup_left = get_numeric_series(bs_coup_col_left_norm, default_val=bs_coup_raw_left); bs_coup_right = get_numeric_series(bs_coup_col_right_norm, default_val=bs_coup_raw_right)
        asym_ratio = calculate_ratio(bs_coup_left, bs_coup_right, min_value=min_val); asym_percdiff = calculate_percent_diff(bs_coup_left, bs_coup_right, min_value=min_val, cap=perc_diff_cap)
        feature_data[f"BS_Asym_Ratio_{coup_au}"] = asym_ratio; feature_data[f"BS_Asym_PercDiff_{coup_au}"] = asym_percdiff; bs_asym_ratio_list.append(asym_ratio); bs_asym_percdiff_list.append(asym_percdiff)
    feature_data['BS_Avg_Coupled_Asym_Ratio'] = pd.concat(bs_asym_ratio_list, axis=1).mean(axis=1) if bs_asym_ratio_list else pd.Series(1.0, index=df.index)
    feature_data['BS_Max_Coupled_Asym_PercDiff'] = pd.concat(bs_asym_percdiff_list, axis=1).max(axis=1) if bs_asym_percdiff_list else pd.Series(0.0, index=df.index)

    # --- Calculate Summary Features Across ALL Actions (FIXED) ---
    summary_features = {}
    for coup_au in coupled_aus:
        coup_cols = [f"{action}_{coup_au}_coup_norm" for action in actions if f"{action}_{coup_au}_coup_norm" in feature_data]
        if coup_cols:
            coup_series_list = [feature_data[col] for col in coup_cols if isinstance(feature_data[col], pd.Series)]
            if coup_series_list:
                coup_df = pd.concat(coup_series_list, axis=1)
                summary_features[f"Avg_{coup_au}_AcrossActions"] = coup_df.mean(axis=1)
                summary_features[f"Max_{coup_au}_AcrossActions"] = coup_df.max(axis=1)
                summary_features[f"Std_{coup_au}_AcrossActions"] = coup_df.std(axis=1).fillna(0)
            else:
                summary_features[f"Avg_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
                summary_features[f"Max_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
                summary_features[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        else:
            summary_features[f"Avg_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
            summary_features[f"Max_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
            summary_features[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)

    for trig_au in trigger_aus:
        trig_cols = [f"{action}_{trig_au}_trig_norm" for action in actions if f"{action}_{trig_au}_trig_norm" in feature_data]
        if trig_cols:
            trig_series_list = [feature_data[col] for col in trig_cols if isinstance(feature_data[col], pd.Series)]
            if trig_series_list:
                trig_df = pd.concat(trig_series_list, axis=1)
                summary_features[f"Avg_{trig_au}_AcrossActions"] = trig_df.mean(axis=1)
                summary_features[f"Max_{trig_au}_AcrossActions"] = trig_df.max(axis=1)
            else:
                summary_features[f"Avg_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
                summary_features[f"Max_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        else:
            summary_features[f"Avg_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
            summary_features[f"Max_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index)

    # Ensure we retrieve Series before the final concat/mean
    avg_coup_series = [summary_features.get(f"Avg_{c}_AcrossActions") for c in coupled_aus]
    avg_trig_series = [summary_features.get(f"Avg_{t}_AcrossActions") for t in trigger_aus]

    # Filter out non-Series items
    avg_coup_series = [s for s in avg_coup_series if isinstance(s, pd.Series)]
    avg_trig_series = [s for s in avg_trig_series if isinstance(s, pd.Series)]

    if avg_coup_series and avg_trig_series:
        # This concat should now work
        overall_avg_coup = pd.concat(avg_coup_series, axis=1).mean(axis=1)
        overall_avg_trig = pd.concat(avg_trig_series, axis=1).mean(axis=1)
        summary_features["Ratio_AvgCoup_vs_AvgTrig"] = calculate_ratio(overall_avg_coup, overall_avg_trig, min_value=min_val)
    else:
        # Ensure default is a Series
        summary_features["Ratio_AvgCoup_vs_AvgTrig"] = pd.Series(1.0, index=df.index)

    feature_data.update(summary_features)
    # --- END OF FIXES ---

    features_df = pd.DataFrame(feature_data, index=df.index); features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0.0)
    logger.debug(f"[{name}] Generated {features_df.shape[1]} features for {side} (Training)."); return features_df

# --- extract_features_for_detection (Detection) ---
# This uses the corrected logic provided before
def extract_features_for_detection(row_data, side):
    """ Extracts Oral-Ocular features for detection (50 features version). """
    if not CONFIG_LOADED:
        return None
    name = config.get('name', 'Oral-Ocular')
    logger.debug(f"[{name}] Extracting detection features for {side}...")
    feature_list_path = filenames.get('feature_list')
    if not feature_list_path or not os.path.exists(feature_list_path):
        logger.error(f"[{name}] Feature list missing. Abort.")
        return None
    try:
        ordered_feature_names = joblib.load(feature_list_path)
        assert isinstance(ordered_feature_names, list)
    except Exception as e:
        logger.error(f"[{name}] Load feature list failed: {e}")
        return None
    if not isinstance(row_data, (pd.Series, dict)):
        logger.error(f"Invalid row_data type: {type(row_data)}")
        return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    if side not in ['Left', 'Right']:
        logger.error(f"Invalid side: {side}")
        return None
    else:
        side_label = side
    use_normalized = feature_cfg.get('use_normalized', True)
    min_val = feature_cfg.get('min_value', 0.0001)
    norm_suffix = " (Normalized)" if use_normalized else ""
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

    all_action_features_scalar = {}
    for action in actions:
        action_features_scalar = {}
        for trig_au in trigger_aus:
            col_raw = f"{action}_{side_label} {trig_au}"
            col_norm = f"{col_raw}{norm_suffix}"
            raw_val = get_float_value(col_raw)
            val_to_use = raw_val
            if use_normalized:
                val_to_use = get_float_value(col_norm, default=raw_val)
            action_features_scalar[f"{action}_{trig_au}_trig_norm"] = val_to_use
        for coup_au in coupled_aus:
            col_raw = f"{action}_{side_label} {coup_au}"
            col_norm = f"{col_raw}{norm_suffix}"
            raw_val = get_float_value(col_raw)
            val_to_use = raw_val
            if use_normalized:
                val_to_use = get_float_value(col_norm, default=raw_val)
            coup_val = val_to_use
            action_features_scalar[f"{action}_{coup_au}_coup_norm"] = coup_val
            for trig_au in trigger_aus:
                trig_val = action_features_scalar.get(f"{action}_{trig_au}_trig_norm", 0.0)
                action_features_scalar[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = scalar_ratio(coup_val, trig_val, min_v=min_val) # Uses corrected scalar_ratio
        all_action_features_scalar.update(action_features_scalar)
    feature_dict_final.update(all_action_features_scalar)

    bs_asym_ratio_scalar_list = []
    bs_asym_percdiff_scalar_list = []
    percent_diff_cap = feature_cfg.get('percent_diff_cap', 200.0)
    for coup_au in coupled_aus:
        bs_coup_raw_left = get_float_value(f"BS_Left {coup_au}")
        bs_coup_raw_right = get_float_value(f"BS_Right {coup_au}")
        bs_coup_left = bs_coup_raw_left
        bs_coup_right = bs_coup_raw_right
        if use_normalized:
            bs_coup_left = get_float_value(f"BS_Left {coup_au}{norm_suffix}", default=bs_coup_raw_left)
            bs_coup_right = get_float_value(f"BS_Right {coup_au}{norm_suffix}", default=bs_coup_raw_right)
        asym_ratio = scalar_ratio(bs_coup_left, bs_coup_right, min_v=min_val) # Uses corrected scalar_ratio
        feature_dict_final[f"BS_Asym_Ratio_{coup_au}"] = asym_ratio
        bs_asym_ratio_scalar_list.append(asym_ratio)
        abs_diff_bs = abs(bs_coup_left - bs_coup_right)
        avg_bs = (bs_coup_left + bs_coup_right) / 2.0
        asym_percdiff = 0.0
        epsilon = 1e-9
        if avg_bs > min_val:
            asym_percdiff = (abs_diff_bs / (avg_bs + epsilon)) * 100.0
        elif abs_diff_bs > min_val:
            asym_percdiff = percent_diff_cap
        asym_percdiff = min(asym_percdiff, percent_diff_cap)
        feature_dict_final[f"BS_Asym_PercDiff_{coup_au}"] = asym_percdiff
        bs_asym_percdiff_scalar_list.append(asym_percdiff)
    feature_dict_final['BS_Avg_Coupled_Asym_Ratio'] = np.mean(bs_asym_ratio_scalar_list) if bs_asym_ratio_scalar_list else 1.0
    feature_dict_final['BS_Max_Coupled_Asym_PercDiff'] = np.max(bs_asym_percdiff_scalar_list) if bs_asym_percdiff_scalar_list else 0.0

    summary_features = {}
    for coup_au in coupled_aus:
        vals = [feature_dict_final.get(f"{action}_{coup_au}_coup_norm", 0.0) for action in actions]
        summary_features[f"Avg_{coup_au}_AcrossActions"] = np.mean(vals)
        summary_features[f"Max_{coup_au}_AcrossActions"] = np.max(vals)
        summary_features[f"Std_{coup_au}_AcrossActions"] = np.std(vals)
    for trig_au in trigger_aus:
        vals = [feature_dict_final.get(f"{action}_{trig_au}_trig_norm", 0.0) for action in actions]
        summary_features[f"Avg_{trig_au}_AcrossActions"] = np.mean(vals)
        summary_features[f"Max_{trig_au}_AcrossActions"] = np.max(vals)
    avg_coup_overall = np.mean([summary_features.get(f"Avg_{c}_AcrossActions", 0.0) for c in coupled_aus])
    avg_trig_overall = np.mean([summary_features.get(f"Avg_{t}_AcrossActions", 0.0) for t in trigger_aus])
    summary_features["Ratio_AvgCoup_vs_AvgTrig"] = scalar_ratio(avg_coup_overall, avg_trig_overall, min_v=min_val) # Uses corrected scalar_ratio
    feature_dict_final.update(summary_features)
    feature_dict_final["side_indicator"] = 0 if side.lower() == 'left' else 1

    feature_list = []
    missing = []
    type_err = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name)
        final_val = 0.0

        if value is None:
            missing.append(name)
        else:
            try:
                temp_val = float(value)
                if np.isnan(temp_val) or np.isinf(temp_val):
                    pass
                else:
                    final_val = temp_val
            except (ValueError, TypeError):
                type_err.append(name)
        feature_list.append(final_val)

    if missing:
        logger.warning(f"[{name}] Detect ({side}): Missing features {missing}. Using 0.")
    if type_err:
        logger.warning(f"[{name}] Detect ({side}): Type errors {type_err}. Using 0.")
    if len(feature_list) != len(ordered_feature_names):
        logger.error(f"[{name}] Detect ({side}): Feature list length mismatch.")
        return None
    logger.debug(f"[{name}] Generated {len(feature_list)} detection features for {side}.")
    return feature_list