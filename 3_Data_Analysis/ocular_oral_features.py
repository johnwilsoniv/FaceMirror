# --- START OF FILE ocular_oral_features.py ---

# ocular_oral_features.py (Refactored V2 - Added Asymmetry Features)

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
SYNK_TYPE = 'ocular_oral'

# Get type-specific config
try:
    config = SYNKINESIS_CONFIG[SYNK_TYPE]
    feature_cfg = config.get('feature_extraction', {})
    actions = config.get('relevant_actions', []) # Use relevant_actions
    trigger_aus = config.get('trigger_aus', []) # Eyes
    coupled_aus = config.get('coupled_aus', []) # Mouth
    feature_sel_cfg = config.get('feature_selection', {})
    filenames = config.get('filenames', {})
    expert_cols = config.get('expert_columns', {})
    target_cols = config.get('target_columns', {})
    CONFIG_LOADED = True
except KeyError:
    logger.critical(f"CRITICAL: Synkinesis type '{SYNK_TYPE}' not found in SYNKINESIS_CONFIG. Feature extraction cannot proceed.")
    CONFIG_LOADED = False; config = {}; feature_cfg = {}; actions = []; trigger_aus = []; coupled_aus = []; feature_sel_cfg = {}; filenames = {}; expert_cols = {}; target_cols = {}


# --- prepare_data function (Standard Refactored Template) ---
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
    positive_class_name = CLASS_NAMES.get(1, 'Synkinesis')
    left_targets = valid_left_expert_labels_std.apply(lambda x: 1 if x == positive_class_name else 0).values
    right_targets = valid_right_expert_labels_std.apply(lambda x: 1 if x == positive_class_name else 0).values

    unique_left, counts_left = np.unique(left_targets, return_counts=True); left_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_left], counts_left))
    unique_right, counts_right = np.unique(right_targets, return_counts=True); right_dist = dict(zip([CLASS_NAMES.get(i, i) for i in unique_right], counts_right))
    logger.info(f"[{name}] Mapped Left Target distribution (post-filter): {left_dist}")
    logger.info(f"[{name}] Mapped Right Target distribution (post-filter): {right_dist}")

    # --- Combine Filtered Data ---
    if filtered_left_features.empty and filtered_right_features.empty: logger.error(f"[{name}] No valid data left after filtering 'NA'."); return None, None, None

    # --- Side Indicator Handling (Conditional) ---
    feature_list_path_check = filenames.get('feature_list')
    add_side_indicator = False
    if feature_list_path_check and os.path.exists(feature_list_path_check):
        try:
            final_feature_names_loaded = joblib.load(feature_list_path_check)
            if 'side_indicator' in final_feature_names_loaded:
                add_side_indicator = True
        except Exception as e: logger.warning(f"[{name}] Could not load feature list for side indicator check: {e}")

    if add_side_indicator:
         logger.info(f"[{name}] Adding 'side_indicator' based on loaded feature list.")
         filtered_left_features['side_indicator'] = 0
         filtered_right_features['side_indicator'] = 1
    else: logger.info(f"[{name}] 'side_indicator' not added (or feature list check failed/skipped).")

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
                side_indicator_exists = 'side_indicator' in features.columns
                if side_indicator_exists and 'side_indicator' not in top_names: top_names.append('side_indicator')
                cols_to_keep = [col for col in top_names if col in features.columns]; missing = set(top_names)-set(cols_to_keep)
                if side_indicator_exists: missing -= {'side_indicator'}
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
        try:
            os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
            joblib.dump(final_feature_names, feature_list_path)
            logger.info(f"Saved {len(final_feature_names)} feature names to {feature_list_path}")
        except Exception as e: logger.error(f"[{name}] Failed save feature list: {e}", exc_info=True)
    else: logger.error(f"[{name}] Feature list path not defined.")

    # Return features, targets, and metadata
    return features, targets, metadata

# --- extract_features function (Ocular-Oral - Training V2 - Includes Asymmetry) ---
def extract_features(df, side):
    """ Extracts Ocular-Oral features for TRAINING including asymmetry of coupled activation. """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Ocular-Oral')
    logger.debug(f"[{name}] Extracting V2 features for {side} side (Training)...")
    feature_data = {}
    side_label = side.capitalize(); opposite_side_label = 'Right' if side_label == 'Left' else 'Left'

    use_normalized = feature_cfg.get('use_normalized', True) # Should be True for this type
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0)
    norm_suffix = " (Normalized)" if use_normalized else "" # Keep suffix logic if needed elsewhere

    # Helper to get numeric series
    def get_numeric_series(col_name, default_val=0.0):
        """Helper to safely get numeric series from DataFrame."""
        series = df.get(col_name)
        if series is None:
             logger.warning(f"Column '{col_name}' not found in DataFrame. Using default {default_val}.")
             return pd.Series(default_val, index=df.index, dtype=float)
        return pd.to_numeric(series, errors='coerce').fillna(default_val).astype(float)

    # --- Calculate intermediate values for BOTH sides ---
    # We need baseline values to calculate normalized action values
    bl_vals_L, bl_vals_R = {}, {}
    all_involved_aus = list(set(trigger_aus + coupled_aus))
    for au in all_involved_aus:
        bl_vals_L[au] = get_numeric_series(f"BL_Left {au}")
        bl_vals_R[au] = get_numeric_series(f"BL_Right {au}")

    action_norm_vals_L = {}; action_norm_vals_R = {} # Store {action: {au: series}}
    action_data_cache = {} # Cache for ratio calculation later

    for action in actions:
        action_features = {} # Features specific to this action/side combo
        norm_vals_L_current_action = {}; norm_vals_R_current_action = {}
        trigger_vals_side = [] # Trigger values for the target side in this action
        coupled_vals_side = [] # Coupled values for the target side in this action

        # Calculate normalized trigger values for BOTH sides
        for trig_au in trigger_aus:
            # Left side
            raw_L = get_numeric_series(f"{action}_Left {trig_au}")
            bl_L = bl_vals_L.get(trig_au, 0.0)
            norm_L = (raw_L - bl_L).clip(lower=0) if use_normalized else raw_L
            norm_vals_L_current_action[trig_au] = norm_L
            # Right side
            raw_R = get_numeric_series(f"{action}_Right {trig_au}")
            bl_R = bl_vals_R.get(trig_au, 0.0)
            norm_R = (raw_R - bl_R).clip(lower=0) if use_normalized else raw_R
            norm_vals_R_current_action[trig_au] = norm_R

            # Store the trigger value for the target side
            if side_label == 'Left':
                action_features[f"{action}_{trig_au}_trig_norm"] = norm_L
                trigger_vals_side.append(norm_L)
            else:
                action_features[f"{action}_{trig_au}_trig_norm"] = norm_R
                trigger_vals_side.append(norm_R)

        # Calculate normalized coupled values for BOTH sides
        for coup_au in coupled_aus:
             # Left side
            raw_L = get_numeric_series(f"{action}_Left {coup_au}")
            bl_L = bl_vals_L.get(coup_au, 0.0)
            norm_L = (raw_L - bl_L).clip(lower=0) if use_normalized else raw_L
            norm_vals_L_current_action[coup_au] = norm_L
             # Right side
            raw_R = get_numeric_series(f"{action}_Right {coup_au}")
            bl_R = bl_vals_R.get(coup_au, 0.0)
            norm_R = (raw_R - bl_R).clip(lower=0) if use_normalized else raw_R
            norm_vals_R_current_action[coup_au] = norm_R

             # Store the coupled value for the target side and calculate ratios
            if side_label == 'Left':
                coupled_series_target = norm_L
                action_features[f"{action}_{coup_au}_coup_norm"] = coupled_series_target
                coupled_vals_side.append(coupled_series_target)
            else:
                coupled_series_target = norm_R
                action_features[f"{action}_{coup_au}_coup_norm"] = coupled_series_target
                coupled_vals_side.append(coupled_series_target)

            # Calculate ratios (Coupled vs Trigger) for the target side
            for trig_au in trigger_aus:
                trigger_series_target = action_features.get(f"{action}_{trig_au}_trig_norm", pd.Series(0.0, index=df.index))
                action_features[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = calculate_ratio(coupled_series_target, trigger_series_target, min_value=min_val)

        # --- NEW: Calculate Asymmetry of Coupled AUs for this action ---
        for coup_au in coupled_aus:
            norm_L = norm_vals_L_current_action.get(coup_au, pd.Series(0.0, index=df.index))
            norm_R = norm_vals_R_current_action.get(coup_au, pd.Series(0.0, index=df.index))
            action_features[f"{action}_Asym_Ratio_{coup_au}_coup_norm"] = calculate_ratio(norm_L, norm_R, min_value=min_val)
            action_features[f"{action}_Asym_PercDiff_{coup_au}_coup_norm"] = calculate_percent_diff(norm_L, norm_R, min_value=min_val, cap=perc_diff_cap)

        # Store normalized values for potential summary features later
        action_norm_vals_L[action] = norm_vals_L_current_action
        action_norm_vals_R[action] = norm_vals_R_current_action

        # Add all features calculated for this action to the main dictionary
        feature_data.update(action_features)

    # --- Summary Features (Calculated AFTER processing all actions) ---
    # Summary features based on target side's coupled/trigger activations
    # (Example: ET summary - modify if needed for average across all actions)
    et_coupled_vals_norm = []; et_trigger_vals_norm = []
    if 'ET' in actions:
        et_features_target = {k: v for k, v in feature_data.items() if k.startswith('ET_') and ('_coup_norm' in k or '_trig_norm' in k) and 'Asym' not in k}
        for feat_name, series in et_features_target.items():
            if '_coup_norm' in feat_name: et_coupled_vals_norm.append(series)
            if '_trig_norm' in feat_name: et_trigger_vals_norm.append(series)

    # Calculate ET summary features (Avg, Max, Ratio)
    et_coupled_vals_norm = [s for s in et_coupled_vals_norm if isinstance(s, pd.Series)]
    et_trigger_vals_norm = [s for s in et_trigger_vals_norm if isinstance(s, pd.Series)]

    if et_coupled_vals_norm:
        et_coupled_df = pd.concat(et_coupled_vals_norm, axis=1).fillna(0.0)
        feature_data['ET_Avg_Coupled_Norm'] = et_coupled_df.mean(axis=1)
        feature_data['ET_Max_Coupled_Norm'] = et_coupled_df.max(axis=1)
    else: feature_data['ET_Avg_Coupled_Norm'] = pd.Series(0.0, index=df.index); feature_data['ET_Max_Coupled_Norm'] = pd.Series(0.0, index=df.index)

    if et_trigger_vals_norm:
        et_trigger_df = pd.concat(et_trigger_vals_norm, axis=1).fillna(0.0)
        feature_data['ET_Avg_Trigger_Norm'] = et_trigger_df.mean(axis=1)
    else: feature_data['ET_Avg_Trigger_Norm'] = pd.Series(0.0, index=df.index)

    et_avg_coup = feature_data.get('ET_Avg_Coupled_Norm'); et_avg_trig = feature_data.get('ET_Avg_Trigger_Norm')
    if isinstance(et_avg_coup, pd.Series) and isinstance(et_avg_trig, pd.Series): feature_data['ET_Ratio_AvgCoup_vs_AvgTrig'] = calculate_ratio(et_avg_coup, et_avg_trig, min_value=min_val)
    else: feature_data['ET_Ratio_AvgCoup_vs_AvgTrig'] = pd.Series(1.0, index=df.index)

    # Summary features across ALL actions (Average/Max/Std activation for each AU type)
    # This part remains complex and might need careful review based on importance results later
    summary_features = {}
    for coup_au in coupled_aus:
        coup_cols = [f"{action}_{coup_au}_coup_norm" for action in actions if f"{action}_{coup_au}_coup_norm" in feature_data]
        if coup_cols:
            coup_series_list = [feature_data[col] for col in coup_cols if isinstance(feature_data[col], pd.Series)]
            if coup_series_list:
                coup_df = pd.concat(coup_series_list, axis=1).fillna(0.0)
                summary_features[f"Avg_{coup_au}_AcrossActions"] = coup_df.mean(axis=1)
                summary_features[f"Max_{coup_au}_AcrossActions"] = coup_df.max(axis=1)
                summary_features[f"Std_{coup_au}_AcrossActions"] = coup_df.std(axis=1).fillna(0)
            else: # Fallback
                summary_features[f"Avg_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index); summary_features[f"Max_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index); summary_features[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        else: # Fallback
            summary_features[f"Avg_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index); summary_features[f"Max_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index); summary_features[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)

    for trig_au in trigger_aus:
        trig_cols = [f"{action}_{trig_au}_trig_norm" for action in actions if f"{action}_{trig_au}_trig_norm" in feature_data]
        if trig_cols:
            trig_series_list = [feature_data[col] for col in trig_cols if isinstance(feature_data[col], pd.Series)]
            if trig_series_list:
                trig_df = pd.concat(trig_series_list, axis=1).fillna(0.0)
                summary_features[f"Avg_{trig_au}_AcrossActions"] = trig_df.mean(axis=1)
                summary_features[f"Max_{trig_au}_AcrossActions"] = trig_df.max(axis=1)
            else: # Fallback
                 summary_features[f"Avg_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index); summary_features[f"Max_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        else: # Fallback
             summary_features[f"Avg_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index); summary_features[f"Max_{trig_au}_AcrossActions"] = pd.Series(0.0, index=df.index)

    # Overall ratio of average coupled vs average trigger
    avg_coup_series = [summary_features.get(f"Avg_{c}_AcrossActions") for c in coupled_aus]; avg_trig_series = [summary_features.get(f"Avg_{t}_AcrossActions") for t in trigger_aus]
    avg_coup_series = [s for s in avg_coup_series if isinstance(s, pd.Series)]; avg_trig_series = [s for s in avg_trig_series if isinstance(s, pd.Series)]
    if avg_coup_series and avg_trig_series:
         overall_avg_coup = pd.concat(avg_coup_series, axis=1).mean(axis=1); overall_avg_trig = pd.concat(avg_trig_series, axis=1).mean(axis=1)
         summary_features["Ratio_AvgCoup_vs_AvgTrig"] = calculate_ratio(overall_avg_coup, overall_avg_trig, min_value=min_val)
    else: summary_features["Ratio_AvgCoup_vs_AvgTrig"] = pd.Series(1.0, index=df.index)

    feature_data.update(summary_features)

    # --- Final Cleanup ---
    features_df = pd.DataFrame(feature_data, index=df.index)
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0.0)
    logger.debug(f"[{name}] Generated {features_df.shape[1]} V2 features for {side} (Training).")
    return features_df


# --- extract_features_for_detection (Ocular-Oral - Detection V2 - Includes Asymmetry) ---
def extract_features_for_detection(row_data, side):
    """ Extracts Ocular-Oral V2 features for detection including asymmetry. """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Ocular-Oral'); logger.debug(f"[{name}] Extracting V2 detection features for {side}...")

    filenames = config.get('filenames', {})
    feature_list_path = filenames.get('feature_list');
    if not feature_list_path or not os.path.exists(feature_list_path): logger.error(f"[{name}] Feature list missing. Abort."); return None
    try:
        ordered_feature_names = joblib.load(feature_list_path);
        if not isinstance(ordered_feature_names, list): raise TypeError("Features list is not a list.")
    except Exception as e: logger.error(f"[{name}] Load feature list failed: {e}"); return None

    if not isinstance(row_data, (pd.Series, dict)): logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    if side not in ['Left', 'Right']: logger.error(f"Invalid side: {side}"); return None
    else: side_label = side

    feature_cfg = config.get('feature_extraction', {})
    actions = config.get('relevant_actions', [])
    trigger_aus = config.get('trigger_aus', [])
    coupled_aus = config.get('coupled_aus', [])
    use_normalized = feature_cfg.get('use_normalized', True)
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0)
    norm_suffix = " (Normalized)" if use_normalized else ""
    epsilon = 1e-9
    feature_dict_generated = {} # Store calculated features

    def get_float_value(key, default=0.0):
        """Safely get a float value from the row series."""
        val = row_series.get(key, default);
        try:
            f_val = float(val);
            return default if np.isnan(f_val) or np.isinf(f_val) else f_val
        except (ValueError, TypeError): return default

    def scalar_ratio(v1, v2, min_v=0.0001):
        """Scalar ratio calculation."""
        v1_f, v2_f = float(v1), float(v2)
        min_val_local, max_val_local = min(v1_f, v2_f), max(v1_f, v2_f)
        if max_val_local <= min_v: return 1.0
        if min_val_local <= min_v: return 0.0
        return np.clip(np.nan_to_num(min_val_local / (max_val_local + epsilon), nan=1.0), 0.0, 1.0)

    def scalar_perc_diff(v1, v2, min_v=0.0001, cap=200.0):
        """Scalar percent difference calculation."""
        v1_f, v2_f = float(v1), float(v2)
        diff = abs(v1_f - v2_f); avg = (v1_f + v2_f) / 2.0; pdiff = 0.0
        if avg > min_v: pdiff = (diff / (avg + epsilon)) * 100.0
        elif diff > min_v: pdiff = cap
        return np.clip(np.nan_to_num(pdiff, nan=0.0, posinf=cap, neginf=0.0), 0, cap)

    # --- Calculate intermediate values for BOTH sides ---
    bl_vals_L, bl_vals_R = {}, {}
    all_involved_aus = list(set(trigger_aus + coupled_aus))
    for au in all_involved_aus:
        bl_vals_L[au] = get_float_value(f"BL_Left {au}")
        bl_vals_R[au] = get_float_value(f"BL_Right {au}")

    action_norm_vals_L = {}; action_norm_vals_R = {}
    all_action_features_scalar = {} # Store features calculated per action

    for action in actions:
        action_features_scalar = {}
        norm_vals_L_current = {}; norm_vals_R_current = {}

        # Trigger AUs Norm Vals (Both Sides) & Target Side Feature
        for trig_au in trigger_aus:
            raw_L = get_float_value(f"{action}_Left {trig_au}")
            norm_L = max(0.0, raw_L - bl_vals_L.get(trig_au, 0.0)) if use_normalized else raw_L
            norm_vals_L_current[trig_au] = norm_L
            raw_R = get_float_value(f"{action}_Right {trig_au}")
            norm_R = max(0.0, raw_R - bl_vals_R.get(trig_au, 0.0)) if use_normalized else raw_R
            norm_vals_R_current[trig_au] = norm_R
            action_features_scalar[f"{action}_{trig_au}_trig_norm"] = norm_L if side_label == 'Left' else norm_R

        # Coupled AUs Norm Vals (Both Sides), Target Side Feature, Ratios, Asymmetry
        for coup_au in coupled_aus:
            raw_L = get_float_value(f"{action}_Left {coup_au}")
            norm_L = max(0.0, raw_L - bl_vals_L.get(coup_au, 0.0)) if use_normalized else raw_L
            norm_vals_L_current[coup_au] = norm_L
            raw_R = get_float_value(f"{action}_Right {coup_au}")
            norm_R = max(0.0, raw_R - bl_vals_R.get(coup_au, 0.0)) if use_normalized else raw_R
            norm_vals_R_current[coup_au] = norm_R

            coupled_val_target = norm_L if side_label == 'Left' else norm_R
            action_features_scalar[f"{action}_{coup_au}_coup_norm"] = coupled_val_target

            # Ratios (Target side)
            for trig_au in trigger_aus:
                trigger_val_target = action_features_scalar.get(f"{action}_{trig_au}_trig_norm", 0.0)
                action_features_scalar[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = scalar_ratio(coupled_val_target, trigger_val_target, min_v=min_val)

            # Asymmetry of Coupled Norm Activation
            action_features_scalar[f"{action}_Asym_Ratio_{coup_au}_coup_norm"] = scalar_ratio(norm_L, norm_R, min_v=min_val)
            action_features_scalar[f"{action}_Asym_PercDiff_{coup_au}_coup_norm"] = scalar_perc_diff(norm_L, norm_R, min_v=min_val, cap=perc_diff_cap)

        action_norm_vals_L[action] = norm_vals_L_current
        action_norm_vals_R[action] = norm_vals_R_current
        all_action_features_scalar.update(action_features_scalar)

    feature_dict_generated.update(all_action_features_scalar) # Add all action-specific features

    # --- Summary Features ---
    # ET Specific Summaries (if ET is relevant)
    et_coupled_vals_scalar = []; et_trigger_vals_scalar = []
    if 'ET' in actions:
        for coup_au in coupled_aus: et_coupled_vals_scalar.append(feature_dict_generated.get(f"ET_{coup_au}_coup_norm", 0.0))
        for trig_au in trigger_aus: et_trigger_vals_scalar.append(feature_dict_generated.get(f"ET_{trig_au}_trig_norm", 0.0))

    feature_dict_generated['ET_Avg_Coupled_Norm'] = np.mean(et_coupled_vals_scalar) if et_coupled_vals_scalar else 0.0
    feature_dict_generated['ET_Max_Coupled_Norm'] = np.max(et_coupled_vals_scalar) if et_coupled_vals_scalar else 0.0
    feature_dict_generated['ET_Avg_Trigger_Norm'] = np.mean(et_trigger_vals_scalar) if et_trigger_vals_scalar else 0.0
    feature_dict_generated['ET_Ratio_AvgCoup_vs_AvgTrig'] = scalar_ratio(feature_dict_generated['ET_Avg_Coupled_Norm'], feature_dict_generated['ET_Avg_Trigger_Norm'], min_v=min_val)

    # Summaries Across ALL actions
    summary_features_scalar = {}
    for coup_au in coupled_aus:
        vals = [feature_dict_generated.get(f"{action}_{coup_au}_coup_norm", 0.0) for action in actions]
        summary_features_scalar[f"Avg_{coup_au}_AcrossActions"] = np.mean(vals) if vals else 0.0
        summary_features_scalar[f"Max_{coup_au}_AcrossActions"] = np.max(vals) if vals else 0.0
        summary_features_scalar[f"Std_{coup_au}_AcrossActions"] = np.std(vals) if len(vals)>1 else 0.0
    for trig_au in trigger_aus:
        vals = [feature_dict_generated.get(f"{action}_{trig_au}_trig_norm", 0.0) for action in actions]
        summary_features_scalar[f"Avg_{trig_au}_AcrossActions"] = np.mean(vals) if vals else 0.0
        summary_features_scalar[f"Max_{trig_au}_AcrossActions"] = np.max(vals) if vals else 0.0

    avg_coup_overall = np.mean([summary_features_scalar.get(f"Avg_{c}_AcrossActions", 0.0) for c in coupled_aus])
    avg_trig_overall = np.mean([summary_features_scalar.get(f"Avg_{t}_AcrossActions", 0.0) for t in trigger_aus])
    summary_features_scalar["Ratio_AvgCoup_vs_AvgTrig"] = scalar_ratio(avg_coup_overall, avg_trig_overall, min_v=min_val)
    feature_dict_generated.update(summary_features_scalar)

    # Side indicator (Conditional)
    if "side_indicator" in ordered_feature_names:
        feature_dict_generated["side_indicator"] = 0 if side.lower() == 'left' else 1

    # --- Build final ordered list ---
    feature_list = []
    missing = []
    type_err = []
    for feat_name in ordered_feature_names:
        value = feature_dict_generated.get(feat_name)
        final_val = 0.0
        if value is None:
            missing.append(feat_name)
        else:
            try:
                temp_val = float(value)
                if not (np.isnan(temp_val) or np.isinf(temp_val)): final_val = temp_val
            except (ValueError, TypeError): type_err.append(feat_name)
        feature_list.append(final_val)

    if missing: logger.error(f"[{name}] Detect V2 ({side}): CRITICAL - Missing features required by list: {missing}."); return None
    if type_err: logger.warning(f"[{name}] Detect V2 ({side}): Type errors {type_err}. Using 0.")
    if len(feature_list) != len(ordered_feature_names):
        logger.error(f"[{name}] Detect V2 ({side}): FINAL Feature list length mismatch ({len(feature_list)} vs {len(ordered_feature_names)})."); return None

    logger.debug(f"[{name}] Generated {len(feature_list)} V2 detection features for {side}.")
    return feature_list

# --- END OF FILE ocular_oral_features.py ---