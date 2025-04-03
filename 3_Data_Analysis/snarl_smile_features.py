# snarl_smile_features.py (Mirrors oral_ocular_features.py)

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
except ImportError:
    logging.warning("Could not import from snarl_smile_config. Using fallback definitions.")
    # Fallbacks
    LOG_DIR = 'logs'; MODEL_FILENAMES = {'feature_list': 'models/synkinesis/snarl_smile/features.list', 'importance_file': 'models/synkinesis/snarl_smile/feature_importance.csv'}
    SNARL_SMILE_ACTIONS = ['BS', 'SS']; TRIGGER_AUS = ['AU12_r']; COUPLED_AUS = ['AU09_r', 'AU10_r', 'AU14_r']
    FEATURE_CONFIG = {'actions': SNARL_SMILE_ACTIONS, 'trigger_aus': TRIGGER_AUS, 'coupled_aus': COUPLED_AUS, 'use_normalized': True, 'min_value_for_ratio': 0.05}
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 40, 'importance_file': MODEL_FILENAMES['importance_file']}
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- prepare_data function ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepares data for Snarl-Smile synkinesis training, including feature selection. """
    logger.info("Loading datasets for Snarl-Smile Synkinesis...")
    try:
        results_df = pd.read_csv(results_file, low_memory=False)
        expert_df = pd.read_csv(expert_file)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise

    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        # Rename specific expert columns for Snarl-Smile
        'Snarl Smile Left': 'Expert_Left_Snarl_Smile',   # Adjust if actual column name differs
        'Snarl Smile Right': 'Expert_Right_Snarl_Smile' # Adjust if actual column name differs
        })

    # Target Variable Processing
    if 'Expert_Left_Snarl_Smile' in expert_df.columns:
        expert_df['Target_Left_Snarl_Smile'] = process_targets(expert_df['Expert_Left_Snarl_Smile'])
    else:
        logger.error("Missing 'Expert_Left_Snarl_Smile' column")
        expert_df['Target_Left_Snarl_Smile'] = 0
    if 'Expert_Right_Snarl_Smile' in expert_df.columns:
        expert_df['Target_Right_Snarl_Smile'] = process_targets(expert_df['Expert_Right_Snarl_Smile'])
    else:
        logger.error("Missing 'Expert_Right_Snarl_Smile' column")
        expert_df['Target_Right_Snarl_Smile'] = 0

    logger.info(f"Counts in expert_df['Target_Left_Snarl_Smile'] AFTER mapping: \n{expert_df['Target_Left_Snarl_Smile'].value_counts(dropna=False)}")
    logger.info(f"Counts in expert_df['Target_Right_Snarl_Smile'] AFTER mapping: \n{expert_df['Target_Right_Snarl_Smile'].value_counts(dropna=False)}")

    # Prepare for Merge
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

    # Feature Extraction
    logger.info("Extracting Snarl-Smile features for Left side...")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info("Extracting Snarl-Smile features for Right side...")
    right_features_df = extract_features(merged_df, 'Right')

    # Combine features and targets
    if 'Target_Left_Snarl_Smile' not in merged_df.columns or 'Target_Right_Snarl_Smile' not in merged_df.columns:
         logger.error("Target columns missing in merged_df before creating targets array. Aborting.")
         return None, None
    left_targets = merged_df['Target_Left_Snarl_Smile'].values
    right_targets = merged_df['Target_Right_Snarl_Smile'].values
    targets = np.concatenate([left_targets, right_targets])

    unique_final, counts_final = np.unique(targets, return_counts=True)
    final_class_dist = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in unique_final], counts_final))
    logger.info(f"FINAL Snarl-Smile Class distribution input: {final_class_dist}")

    left_features_df['side_indicator'] = 0
    right_features_df['side_indicator'] = 1
    features = pd.concat([left_features_df, right_features_df], ignore_index=True)

    # Post-processing & Feature Selection
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.fillna(0)
    initial_cols = features.columns.tolist(); cols_to_drop = []
    for col in initial_cols:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Could not convert column {col} to numeric: {e}."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)
    logger.info(f"Generated initial {features.shape[1]} Snarl-Smile features.")

    fs_enabled = isinstance(FEATURE_SELECTION, dict) and FEATURE_SELECTION.get('enabled', False)
    if fs_enabled:
        logger.info("Applying Snarl-Smile feature selection...")
        n_top_features = FEATURE_SELECTION.get('top_n_features', 40)
        importance_file = FEATURE_SELECTION.get('importance_file')
        if not importance_file or not os.path.exists(importance_file): logger.warning(f"FS enabled, but importance file not found: '{importance_file}'. Skipping.")
        else:
            try:
                importance_df = pd.read_csv(importance_file)
                if 'feature' not in importance_df.columns or importance_df.empty: logger.error("Importance file invalid. Skipping.")
                else:
                    top_feature_names = importance_df['feature'].head(n_top_features).tolist()
                    if 'side_indicator' in features.columns and 'side_indicator' not in top_feature_names: top_feature_names.append('side_indicator')
                    original_cols = features.columns.tolist()
                    cols_to_keep = [col for col in top_feature_names if col in original_cols]
                    missing_features = set(top_feature_names) - set(cols_to_keep)
                    if missing_features: logger.warning(f"Important Snarl-Smile features missing: {missing_features}")
                    if not cols_to_keep: logger.error("No features left after filtering. Skipping.")
                    else: logger.info(f"Selecting top {len(cols_to_keep)} Snarl-Smile features."); features = features[cols_to_keep]
            except Exception as e: logger.error(f"Error during Snarl-Smile FS: {e}. Skipping.", exc_info=True)
    else:
        logger.info("Snarl-Smile feature selection is disabled.")

    logger.info(f"Final Snarl-Smile dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.warning(f"NaNs found in FINAL Snarl-Smile features. Filling with 0."); features = features.fillna(0)

    # Save final feature list
    final_feature_names = features.columns.tolist()
    try:
        feature_list_path = MODEL_FILENAMES.get('feature_list')
        if feature_list_path:
             os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
             joblib.dump(final_feature_names, feature_list_path)
             logger.info(f"Saved final {len(final_feature_names)} Snarl-Smile feature names list to {feature_list_path}")
        else: logger.error("Snarl-Smile feature list path not defined.")
    except Exception as e: logger.error(f"Failed to save Snarl-Smile feature names list: {e}", exc_info=True)

    if 'targets' not in locals(): logger.error("Targets array creation failed."); return None, None
    return features, targets


# --- Helper Functions (Identical) ---
def calculate_ratio(val1_series, val2_series):
    min_val_config = FEATURE_CONFIG.get('min_value_for_ratio', 0.05)
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    max_vals_safe = np.maximum(np.maximum(v1, v2), min_val_config)
    min_vals_safe = np.maximum(np.minimum(v1, v2), 0.0)
    ratio = np.divide(min_vals_safe, max_vals_safe, out=np.ones_like(min_vals_safe, dtype=float), where=max_vals_safe!=0)
    return pd.Series(np.nan_to_num(ratio, nan=1.0), index=val1_series.index)

def calculate_percent_diff(val1_series, val2_series):
    min_val_config = FEATURE_CONFIG.get('min_value_for_ratio', 0.05)
    percent_diff_cap = 200.0
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    abs_diff = np.abs(v1 - v2); avg = (v1 + v2) / 2.0
    percent_diff = np.zeros_like(avg, dtype=float)
    mask_avg_valid = avg > min_val_config
    if np.any(mask_avg_valid): percent_diff[mask_avg_valid] = (abs_diff[mask_avg_valid] / avg[mask_avg_valid]) * 100.0
    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > 0)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap
    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    return pd.Series(np.nan_to_num(percent_diff, nan=0.0), index=val1_series.index)


# --- extract_features function (Training) ---
def extract_features(df, side):
    """ Extracts Snarl-Smile features for TRAINING. """
    logger.debug(f"Extracting Snarl-Smile features for {side} side (Training)...")
    feature_data = {}
    opposite_side = 'Right' if side == 'Left' else 'Left'
    local_feature_config = FEATURE_CONFIG
    local_actions = SNARL_SMILE_ACTIONS
    local_trigger_aus = TRIGGER_AUS # Likely just AU12_r
    local_coupled_aus = COUPLED_AUS
    use_normalized = local_feature_config.get('use_normalized', True)
    norm_suffix = " (Normalized)" if use_normalized else ""

    # 1. Basic AU & Interaction Features per Action
    all_action_features = {}
    for action in local_actions:
        action_features = {}
        # Get Trigger AU values (e.g., AU12_r)
        for trig_au in local_trigger_aus:
            col = f"{action}_{side} {trig_au}{norm_suffix}"
            action_features[f"{action}_{trig_au}_trig_norm"] = df.get(col, pd.Series(0.0, index=df.index))

        # Get Coupled AU values (e.g., AU09, AU10, AU14) and ratios
        for coup_au in local_coupled_aus:
            col = f"{action}_{side} {coup_au}{norm_suffix}"
            coup_series = df.get(col, pd.Series(0.0, index=df.index))
            action_features[f"{action}_{coup_au}_coup_norm"] = coup_series

            for trig_au in local_trigger_aus: # Ratio vs each trigger AU
                trig_series = action_features.get(f"{action}_{trig_au}_trig_norm", pd.Series(0.0, index=df.index))
                action_features[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = calculate_ratio(coup_series, trig_series)

        all_action_features.update(action_features)
    feature_data.update(all_action_features)

    # 2. Summary Features Across Actions (BS, SS)
    summary_features = {}
    for coup_au in local_coupled_aus:
        coup_cols = [f"{action}_{coup_au}_coup_norm" for action in local_actions if f"{action}_{coup_au}_coup_norm" in feature_data]
        if coup_cols:
            coup_df = pd.concat([feature_data[col] for col in coup_cols], axis=1)
            summary_features[f"Avg_{coup_au}_AcrossActions"] = coup_df.mean(axis=1)
            summary_features[f"Max_{coup_au}_AcrossActions"] = coup_df.max(axis=1)
            summary_features[f"Std_{coup_au}_AcrossActions"] = coup_df.std(axis=1).fillna(0)

    for trig_au in local_trigger_aus:
         trig_cols = [f"{action}_{trig_au}_trig_norm" for action in local_actions if f"{action}_{trig_au}_trig_norm" in feature_data]
         if trig_cols:
             trig_df = pd.concat([feature_data[col] for col in trig_cols], axis=1)
             summary_features[f"Avg_{trig_au}_AcrossActions"] = trig_df.mean(axis=1)
             summary_features[f"Max_{trig_au}_AcrossActions"] = trig_df.max(axis=1)

    # Summary Ratio: Average coupled vs Average trigger
    avg_coup_vals = [summary_features.get(f"Avg_{coup_au}_AcrossActions", pd.Series(0.0, index=df.index)) for coup_au in local_coupled_aus]
    avg_trig_vals = [summary_features.get(f"Avg_{trig_au}_AcrossActions", pd.Series(0.0, index=df.index)) for trig_au in local_trigger_aus]
    if avg_coup_vals and avg_trig_vals:
         overall_avg_coup = pd.concat(avg_coup_vals, axis=1).mean(axis=1) # Avg of AU9,10,14 avgs
         overall_avg_trig = pd.concat(avg_trig_vals, axis=1).mean(axis=1) # Avg of AU12 avg
         summary_features["Ratio_AvgCoup_vs_AvgTrig"] = calculate_ratio(overall_avg_coup, overall_avg_trig)

    # Weighted Score (Optional - can add complexity)
    # weights = local_feature_config.get('weights', {'AU09_r': 0.4, 'AU10_r': 0.3, 'AU14_r': 0.3})
    # summary_features['WeightedScore_AvgAcrossActions'] = (
    #     summary_features.get('Avg_AU09_r_AcrossActions', 0) * weights.get('AU09_r', 0) +
    #     summary_features.get('Avg_AU10_r_AcrossActions', 0) * weights.get('AU10_r', 0) +
    #     summary_features.get('Avg_AU14_r_AcrossActions', 0) * weights.get('AU14_r', 0)
    # )

    feature_data.update(summary_features)

    # Create DataFrame
    features_df = pd.DataFrame(feature_data, index=df.index)

    # Final check
    non_numeric_cols = features_df.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric cols in Snarl-Smile extract_features: {non_numeric_cols.tolist()}. Coercing.")
        for col in non_numeric_cols: features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)

    logger.debug(f"Generated {features_df.shape[1]} Snarl-Smile features for {side} (Training).")
    return features_df


# --- extract_features_for_detection (Detection) ---
def extract_features_for_detection(row_data, side):
    """ Extracts Snarl-Smile features for detection from a row of data. """
    try:
        from snarl_smile_config import FEATURE_CONFIG, SNARL_SMILE_ACTIONS, TRIGGER_AUS, COUPLED_AUS, MODEL_FILENAMES
        local_logger = logging.getLogger(__name__)
    except ImportError:
        logging.error("Failed config import within snarl_smile extract_features_for_detection."); return None

    if not isinstance(row_data, (pd.Series, dict)): local_logger.error(f"Invalid row_data type: {type(row_data)}"); return None
    row_series = pd.Series(row_data) if isinstance(row_data, dict) else row_data

    local_feature_config = FEATURE_CONFIG; local_actions = SNARL_SMILE_ACTIONS
    local_trigger_aus = TRIGGER_AUS; local_coupled_aus = COUPLED_AUS
    use_normalized = local_feature_config.get('use_normalized', True)
    norm_suffix = " (Normalized)" if use_normalized else ""
    min_val_ratio = local_feature_config.get('min_value_for_ratio', 0.05)

    local_logger.debug(f"Extracting Snarl-Smile detection features for {side}...")
    feature_dict_final = {}

    # 1. Basic AU & Interaction Features per Action (Scalar)
    all_action_features = {}
    for action in local_actions:
        action_features = {}
        for trig_au in local_trigger_aus:
            col = f"{action}_{side} {trig_au}{norm_suffix}"
            action_features[f"{action}_{trig_au}_trig_norm"] = row_series.get(col, 0.0)

        for coup_au in local_coupled_aus:
            col = f"{action}_{side} {coup_au}{norm_suffix}"
            coup_val = row_series.get(col, 0.0)
            action_features[f"{action}_{coup_au}_coup_norm"] = coup_val
            for trig_au in local_trigger_aus:
                trig_val = action_features.get(f"{action}_{trig_au}_trig_norm", 0.0)
                max_val_safe = max(max(coup_val, trig_val), min_val_ratio)
                min_val_safe = max(min(coup_val, trig_val), 0.0)
                ratio = min_val_safe / max_val_safe if max_val_safe != 0 else 1.0
                action_features[f"{action}_Ratio_{coup_au}_vs_{trig_au}"] = ratio
        all_action_features.update(action_features)
    feature_dict_final.update(all_action_features)

    # 2. Summary Features Across Actions (Scalar)
    summary_features = {}
    for coup_au in local_coupled_aus:
        coup_vals = [feature_dict_final.get(f"{action}_{coup_au}_coup_norm", 0.0) for action in local_actions]
        summary_features[f"Avg_{coup_au}_AcrossActions"] = np.mean(coup_vals) if coup_vals else 0.0
        summary_features[f"Max_{coup_au}_AcrossActions"] = np.max(coup_vals) if coup_vals else 0.0
        summary_features[f"Std_{coup_au}_AcrossActions"] = np.std(coup_vals) if coup_vals else 0.0

    for trig_au in local_trigger_aus:
        trig_vals = [feature_dict_final.get(f"{action}_{trig_au}_trig_norm", 0.0) for action in local_actions]
        summary_features[f"Avg_{trig_au}_AcrossActions"] = np.mean(trig_vals) if trig_vals else 0.0
        summary_features[f"Max_{trig_au}_AcrossActions"] = np.max(trig_vals) if trig_vals else 0.0

    avg_coup_vals_list = [summary_features.get(f"Avg_{coup_au}_AcrossActions", 0.0) for coup_au in local_coupled_aus]
    avg_trig_vals_list = [summary_features.get(f"Avg_{trig_au}_AcrossActions", 0.0) for trig_au in local_trigger_aus]
    overall_avg_coup_val = np.mean(avg_coup_vals_list) if avg_coup_vals_list else 0.0
    overall_avg_trig_val = np.mean(avg_trig_vals_list) if avg_trig_vals_list else 0.0
    max_val_safe_summary = max(max(overall_avg_coup_val, overall_avg_trig_val), min_val_ratio)
    min_val_safe_summary = max(min(overall_avg_coup_val, overall_avg_trig_val), 0.0)
    summary_features["Ratio_AvgCoup_vs_AvgTrig"] = min_val_safe_summary / max_val_safe_summary if max_val_safe_summary != 0 else 1.0
    feature_dict_final.update(summary_features)

    # Add side indicator
    feature_dict_final["side_indicator"] = 0 if side.lower() == 'left' else 1

    # --- Load the EXPECTED feature list ---
    feature_names_path = MODEL_FILENAMES.get('feature_list')
    if not feature_names_path or not os.path.exists(feature_names_path):
        local_logger.error(f"Snarl-Smile feature list not found: {feature_names_path}."); return None
    try:
        ordered_feature_names = joblib.load(feature_names_path)
        if not isinstance(ordered_feature_names, list): local_logger.error("Loaded feature names not a list."); return None
    except Exception as e: local_logger.error(f"Failed load Snarl-Smile feature list: {e}", exc_info=True); return None

    # --- Build final feature list IN ORDER ---
    feature_list = []; missing_in_dict = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name, 0.0)
        try: feature_list.append(float(value))
        except (ValueError, TypeError): feature_list.append(0.0)
        if name not in feature_dict_final: missing_in_dict.append(name)

    if missing_in_dict: local_logger.warning(f"Snarl-Smile Detect: {len(missing_in_dict)} features missing: {missing_in_dict[:5]}...")
    if len(feature_list) != len(ordered_feature_names):
        local_logger.error(f"Snarl-Smile feature mismatch: Expected {len(ordered_feature_names)}, got {len(feature_list)}.")
        return None

    local_logger.debug(f"Generated {len(feature_list)} Snarl-Smile detection features for {side}.")
    return feature_list


# --- process_targets function (Identical binary mapping) ---
def process_targets(target_series):
    if target_series is None: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int).values