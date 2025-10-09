# mentalis_features.py (Refactored to use Utils and Config)

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
SYNK_TYPE = 'mentalis'

# Get type-specific config
try:
    config = SYNKINESIS_CONFIG[SYNK_TYPE]
    feature_cfg = config.get('feature_extraction', {})
    # --- Use relevant_actions for consistency ---
    actions = config.get('relevant_actions', [])
    # --- END FIX ---
    coupled_aus = config.get('coupled_aus', [])
    context_aus = config.get('context_aus', []) # Context AUs included
    feature_sel_cfg = config.get('feature_selection', {})
    filenames = config.get('filenames', {})
    expert_cols = config.get('expert_columns', {})
    target_cols = config.get('target_columns', {})
    CONFIG_LOADED = True
except KeyError:
    logger.critical(f"CRITICAL: Synkinesis type '{SYNK_TYPE}' not found in SYNKINESIS_CONFIG.")
    CONFIG_LOADED = False; config = {}; feature_cfg = {}; actions = []; coupled_aus = []; context_aus = []; feature_sel_cfg = {}; filenames = {}; expert_cols = {}; target_cols = {}

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

# --- extract_features function (Mentalis - Training) ---
# This uses the corrected logic provided before
def extract_features(df, side):
    """ Extracts Full Mentalis Synkinesis features for TRAINING using NORMALIZED values """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Mentalis')
    logger.debug(f"[{name}] Extracting features for {side} side (Training)...")
    feature_data = {}
    side_label = side.capitalize()
    use_normalized = feature_cfg.get('use_normalized', True)
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0) # Added for consistency
    norm_suffix = " (Normalized)" if use_normalized else "" # Added for consistency

    def get_numeric_series(col_name, default_val=0.0):
        series = df.get(col_name)
        return pd.to_numeric(series, errors='coerce').fillna(default_val).astype(float) if series is not None else pd.Series(default_val, index=df.index, dtype=float)

    if not coupled_aus:
        logger.error(f"[{name}] COUPLED_AUS empty."); return None
    coup_au = coupled_aus[0] # AU17_r

    # Get baseline raw values for all involved AUs
    bl_raw_values = {}
    all_aus_for_bl = list(set(coupled_aus + context_aus)) # Unique AUs
    for au in all_aus_for_bl:
        bl_raw_values[au] = get_numeric_series(f"BL_{side_label} {au}")

    # --- Calculate features per action ---
    coupled_series_list = []
    action_data_cache = {}
    for action in actions:
        action_cache = {}
        # Coupled AU (AU17)
        bl_coup_series = bl_raw_values.get(coup_au, pd.Series(0.0, index=df.index))
        raw_coup_series = get_numeric_series(f"{action}_{side_label} {coup_au}")
        final_series_coup = raw_coup_series
        if use_normalized:
            final_series_coup = (raw_coup_series - bl_coup_series).clip(lower=0)
            feature_data[f"{action}_{coup_au}_norm"] = final_series_coup
        else:
            feature_data[f"{action}_{coup_au}_raw"] = final_series_coup
        coupled_series_list.append(final_series_coup)
        action_cache[coup_au] = final_series_coup

        # Context AUs
        for context_au in context_aus:
            bl_ctx_series = bl_raw_values.get(context_au, pd.Series(0.0, index=df.index))
            raw_ctx_series = get_numeric_series(f"{action}_{side_label} {context_au}")
            final_series_ctx = raw_ctx_series
            if use_normalized:
                final_series_ctx = (raw_ctx_series - bl_ctx_series).clip(lower=0)
                feature_data[f"{action}_{context_au}_context_norm"] = final_series_ctx
            else:
                feature_data[f"{action}_{context_au}_context_raw"] = final_series_ctx
            action_cache[context_au] = final_series_ctx
        action_data_cache[action] = action_cache

    # --- Calculate Asymmetry Features ---
    bs_norm_left = pd.Series(0.0, index=df.index); bs_norm_right = pd.Series(0.0, index=df.index)
    se_norm_left = pd.Series(0.0, index=df.index); se_norm_right = pd.Series(0.0, index=df.index)

    if use_normalized:
        # Use helper, default to 0 if column missing
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
    else:
        # If not normalized, use raw values for asymmetry calc
        bs_norm_left = get_numeric_series(f"BS_Left {coup_au}")
        bs_norm_right = get_numeric_series(f"BS_Right {coup_au}")
        se_norm_left = get_numeric_series(f"SE_Left {coup_au}")
        se_norm_right = get_numeric_series(f"SE_Right {coup_au}")

    feature_data[f"BS_Asym_Ratio_{coup_au}"] = calculate_ratio(bs_norm_left, bs_norm_right, min_value=min_val)
    feature_data[f"BS_Asym_PercDiff_{coup_au}"] = calculate_percent_diff(bs_norm_left, bs_norm_right, min_value=min_val, cap=perc_diff_cap)
    feature_data[f"SE_Asym_Ratio_{coup_au}"] = calculate_ratio(se_norm_left, se_norm_right, min_value=min_val)
    feature_data[f"SE_Asym_PercDiff_{coup_au}"] = calculate_percent_diff(se_norm_left, se_norm_right, min_value=min_val, cap=perc_diff_cap)

    # --- Calculate Ratios within actions ---
    ratio_prefix = "norm" if use_normalized else "raw"
    if context_aus: # Only calculate if context AUs are defined
        for action in actions:
            if action in action_data_cache:
                action_cache = action_data_cache[action]
                au17_series = action_cache.get(coup_au) # Get normalized/raw AU17 for this action
                if au17_series is not None and isinstance(au17_series, pd.Series):
                    for context_au in context_aus:
                        context_series = action_cache.get(context_au) # Get normalized/raw context AU
                        if context_series is not None and isinstance(context_series, pd.Series):
                             feature_data[f"{action}_Ratio_{coup_au}_vs_{context_au}_{ratio_prefix}"] = calculate_ratio(au17_series, context_series, min_value=min_val)
                        else:
                             # Default ratio to 1 if context AU is missing or not a Series
                             feature_data[f"{action}_Ratio_{coup_au}_vs_{context_au}_{ratio_prefix}"] = pd.Series(1.0, index=df.index)
                else:
                    # Default all ratios to 1 if AU17 is missing for the action
                    for context_au in context_aus:
                        feature_data[f"{action}_Ratio_{coup_au}_vs_{context_au}_{ratio_prefix}"] = pd.Series(1.0, index=df.index)


    # --- Calculate Summary Features Across Actions (FIXED) ---
    summary_features = {} # Re-initialize for this section
    # Ensure only Series objects are in the list before concat
    valid_coupled_series = [s for s in coupled_series_list if isinstance(s, pd.Series)]

    if valid_coupled_series:
        coup_df = pd.concat(valid_coupled_series, axis=1).fillna(0.0) # Concat valid series
        summary_features[f"Avg_{coup_au}_AcrossActions"] = coup_df.mean(axis=1)
        summary_features[f"Max_{coup_au}_AcrossActions"] = coup_df.max(axis=1)
        summary_features[f"Std_{coup_au}_AcrossActions"] = coup_df.std(axis=1).fillna(0)
    else:
        # FIX: Assign Series instead of float
        summary_features[f"Avg_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        summary_features[f"Max_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        summary_features[f"Std_{coup_au}_AcrossActions"] = pd.Series(0.0, index=df.index)
        # END FIX

    feature_data.update(summary_features) # Add summary stats to the main dict
    # --- END OF FIXES ---

    features_df = pd.DataFrame(feature_data, index=df.index)
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0.0)
    logger.debug(f"[{name}] Generated {features_df.shape[1]} features for {side} (Training).")
    return features_df

# --- extract_features_for_detection (Detection) ---
# This uses the corrected logic provided before
def extract_features_for_detection(row_data, side):
    """ Extracts Full Mentalis Synkinesis features for detection using NORMALIZED values """
    if not CONFIG_LOADED: return None
    name = config.get('name', 'Mentalis')
    logger.debug(f"[{name}] Extracting detection features for {side}...")
    feature_list_path = filenames.get('feature_list')
    if not feature_list_path or not os.path.exists(feature_list_path):
        logger.error(f"[{name}] Feature list missing. Abort."); return None
    try:
        ordered_feature_names = joblib.load(feature_list_path)
        assert isinstance(ordered_feature_names, list)
    except Exception as e:
        logger.error(f"[{name}] Load feature list failed: {e}"); return None
    if not isinstance(row_data, (pd.Series, dict)):
        logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data
    if side not in ['Left', 'Right']:
        logger.error(f"Invalid side: {side}"); return None
    else:
        side_label = side
    use_normalized = feature_cfg.get('use_normalized', True)
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_diff_cap = feature_cfg.get('percent_diff_cap', 200.0)
    norm_suffix = " (Normalized)" if use_normalized else "" # Added for consistency
    epsilon = 1e-9
    if not coupled_aus:
        logger.error("Coupled AU list empty."); return None
    coup_au = coupled_aus[0]
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

    # Get baseline raw values
    bl_raw_values_scalar = {}
    all_aus_for_bl = list(set(coupled_aus + context_aus))
    for au in all_aus_for_bl:
        bl_raw_values_scalar[au] = get_float_value(f"BL_{side_label} {au}")

    # Calculate features per action
    coupled_values_list = []
    action_data_cache_scalar = {}
    for action in actions:
        action_cache_scalar = {}
        # Coupled AU
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
        for context_au in context_aus:
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

    # Calculate Asymmetry Features
    bs_norm_left=0.0; bs_norm_right=0.0
    se_norm_left=0.0; se_norm_right=0.0
    if use_normalized:
        bl_l_raw=get_float_value(f"BL_Left {coup_au}")
        bl_r_raw=get_float_value(f"BL_Right {coup_au}")
        bs_l_raw=get_float_value(f"BS_Left {coup_au}")
        bs_r_raw=get_float_value(f"BS_Right {coup_au}")
        se_l_raw=get_float_value(f"SE_Left {coup_au}")
        se_r_raw=get_float_value(f"SE_Right {coup_au}")
        bs_norm_left=max(0.0,bs_l_raw-bl_l_raw)
        bs_norm_right=max(0.0,bs_r_raw-bl_r_raw)
        se_norm_left=max(0.0,se_l_raw-bl_l_raw)
        se_norm_right=max(0.0,se_r_raw-bl_r_raw)
    else: # Use raw values if not normalizing
        bs_norm_left=get_float_value(f"BS_Left {coup_au}")
        bs_norm_right=get_float_value(f"BS_Right {coup_au}")
        se_norm_left=get_float_value(f"SE_Left {coup_au}")
        se_norm_right=get_float_value(f"SE_Right {coup_au}")

    feature_dict_final[f"BS_Asym_Ratio_{coup_au}"]=scalar_ratio(bs_norm_left,bs_norm_right,min_v=min_val) # Uses corrected scalar_ratio
    feature_dict_final[f"BS_Asym_PercDiff_{coup_au}"]=scalar_perc_diff(bs_norm_left,bs_norm_right,min_v=min_val,cap=perc_diff_cap)
    feature_dict_final[f"SE_Asym_Ratio_{coup_au}"]=scalar_ratio(se_norm_left,se_norm_right,min_v=min_val) # Uses corrected scalar_ratio
    feature_dict_final[f"SE_Asym_PercDiff_{coup_au}"]=scalar_perc_diff(se_norm_left,se_norm_right,min_v=min_val,cap=perc_diff_cap)

    # Calculate Ratios within actions
    ratio_prefix = "norm" if use_normalized else "raw"
    if context_aus:
        for action in actions:
             if action in action_data_cache_scalar:
                action_cache=action_data_cache_scalar[action]
                au17_val=action_cache.get(coup_au) # Get norm/raw AU17
                if au17_val is not None:
                    for context_au in context_aus:
                        context_val=action_cache.get(context_au) # Get norm/raw context AU
                        feature_dict_final[f"{action}_Ratio_{coup_au}_vs_{context_au}_{ratio_prefix}"] = scalar_ratio(au17_val, context_val, min_v=min_val) if context_val is not None else 1.0 # Uses corrected scalar_ratio
                else: # Default ratio to 1 if AU17 missing
                     for context_au in context_aus:
                         feature_dict_final[f"{action}_Ratio_{coup_au}_vs_{context_au}_{ratio_prefix}"] = 1.0


    # Calculate Summary Features Across Actions
    valid_coupled_values = [v for v in coupled_values_list if pd.notna(v) and np.isfinite(v)]
    feature_dict_final[f"Avg_{coup_au}_AcrossActions"]=np.mean(valid_coupled_values) if valid_coupled_values else 0.0
    feature_dict_final[f"Max_{coup_au}_AcrossActions"]=np.max(valid_coupled_values) if valid_coupled_values else 0.0
    feature_dict_final[f"Std_{coup_au}_AcrossActions"]=np.std(valid_coupled_values) if len(valid_coupled_values)>1 else 0.0

    # Side indicator
    feature_dict_final["side_indicator"]=0 if side.lower()=='left' else 1

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
        logger.warning(f"[{name}] Detect ({side}): Missing features {missing}. Using 0.")
    if type_err:
        logger.warning(f"[{name}] Detect ({side}): Type errors {type_err}. Using 0.")
    if len(feature_list) != len(ordered_feature_names):
        logger.error(f"[{name}] Detect ({side}): Feature list length mismatch.")
        return None
    logger.debug(f"[{name}] Generated {len(feature_list)} detection features for {side}.")
    return feature_list