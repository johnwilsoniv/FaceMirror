# paralysis_utils.py (v8.6 - FS logging, unique fix)

import importlib
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt  # Kept for visualize_confusion_matrix
import seaborn as sns  # Kept for visualize_confusion_matrix
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_fscore_support, f1_score,
    roc_auc_score, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold  # KFold kept for potential use
from sklearn.preprocessing import \
    StandardScaler  # StandardScaler kept for find_similar_patients example (though features are usually scaled before this)
from imblearn.over_sampling import SMOTE  # SMOTE kept for analyze_training_influence example
import joblib
import xgboost as xgb  # xgb kept for analyze_training_influence example
import scipy.stats  # For calculate_entropy
from copy import deepcopy
from sklearn.calibration import CalibratedClassifierCV  # Kept as it's part of the broader ecosystem

logger = logging.getLogger(__name__)

# --- Explicitly Import Class Maps from Configs with Fallbacks ---
try:
    # Attempt to import from paralysis_config first
    from paralysis_config import CLASS_NAMES as PARALYSIS_CLASS_NAMES_CONFIG, \
        INPUT_FILES as PARALYSIS_INPUT_FILES_CONFIG, \
        ZONE_CONFIG as PARALYSIS_ZONE_CONFIG_CONFIG

    PARALYSIS_MAP = PARALYSIS_CLASS_NAMES_CONFIG
except ImportError:
    # Fallback if paralysis_config is not found or CLASS_NAMES is not defined
    PARALYSIS_MAP = {0: 'None', 1: 'Partial', 2: 'Complete'}
    PARALYSIS_INPUT_FILES_CONFIG = {}  # Define as empty dict if import fails
    PARALYSIS_ZONE_CONFIG_CONFIG = {}  # Define as empty dict if import fails
    logger.info("paralysis_config.py not found or CLASS_NAMES not defined. Using default PARALYSIS_MAP.")

try:
    from synkinesis_config import CLASS_NAMES as SYNKINESIS_CLASS_NAMES_CONFIG

    SYNKINESIS_MAP = SYNKINESIS_CLASS_NAMES_CONFIG
except ImportError:
    SYNKINESIS_MAP = {0: 'None', 1: 'Synkinesis'}
    logger.info("synkinesis_config.py not found or CLASS_NAMES not defined. Using default SYNKINESIS_MAP.")


# --- Feature Calculation Helpers ---
def calculate_ratio(val1_series, val2_series, min_value=0.0001):
    """
    Calculates the ratio min(s1, s2) / max(s1, s2).
    Handles NaN, coercion to numeric, and division by zero or near-zero.
    """
    s1 = pd.to_numeric(pd.Series(val1_series), errors='coerce').fillna(0.0)
    s2 = pd.to_numeric(pd.Series(val2_series), errors='coerce').fillna(0.0)

    min_vals = pd.Series(np.minimum(s1, s2), index=s1.index)
    max_vals = pd.Series(np.maximum(s1, s2), index=s1.index)

    ratio = pd.Series(1.0, index=s1.index)  # Default ratio is 1.0 (no asymmetry or values are too small)

    # Mask for cases where max_vals is positive and significant
    mask_max_pos = max_vals > min_value
    # Mask for cases where min_vals is zero or insignificant (but max_vals is not)
    mask_min_zero_but_max_pos = (min_vals <= min_value) & mask_max_pos

    epsilon = 1e-9  # To prevent division by absolute zero if max_vals is exactly 0 after checks

    # Valid division: max_vals is significant and not zero
    valid_division_mask = mask_max_pos & (max_vals != 0)
    ratio.loc[valid_division_mask] = min_vals.loc[valid_division_mask] / (max_vals.loc[valid_division_mask] + epsilon)

    # If min is zero/small and max is significant, ratio is 0
    ratio.loc[mask_min_zero_but_max_pos] = 0.0

    # Fill any remaining NaNs (e.g., if both s1 and s2 were NaN initially) with 1.0 and clip
    ratio = ratio.fillna(1.0).clip(0.0, 1.0)
    return ratio


def calculate_percent_diff(val1_series, val2_series, min_value=0.0001, cap=200.0):
    """
    Calculates the percentage difference: |s1 - s2| / ((s1 + s2) / 2) * 100.
    Handles NaN, coercion, division by zero/near-zero, and caps the result.
    """
    s1 = pd.to_numeric(pd.Series(val1_series), errors='coerce').fillna(0.0)
    s2 = pd.to_numeric(pd.Series(val2_series), errors='coerce').fillna(0.0)

    abs_diff = (s1 - s2).abs()
    avg = (s1 + s2) / 2.0

    percent_diff = pd.Series(0.0, index=s1.index, dtype=float)  # Default to 0

    # Mask for cases where average is positive and significant
    mask_avg_pos = avg.abs() > min_value  # Use abs for average if it can be negative
    # Mask for cases where absolute difference is significant (to avoid capping small true differences)
    mask_diff_pos = abs_diff > min_value

    epsilon = 1e-9  # To prevent division by absolute zero

    # Valid division: average is significant and not zero
    valid_division_mask = mask_avg_pos & (avg != 0)
    percent_diff.loc[valid_division_mask] = (abs_diff.loc[valid_division_mask] / (
                avg.loc[valid_division_mask].abs() + epsilon)) * 100.0

    # If average is zero/small but difference is significant, set to cap (indicates large relative difference)
    percent_diff.loc[~mask_avg_pos & mask_diff_pos] = cap

    # Fill any remaining NaNs with 0.0 and clip
    percent_diff = percent_diff.fillna(0.0).clip(0, cap)
    return percent_diff


# --- Label Standardization Functions ---
def standardize_paralysis_labels(val):
    """Standardizes various paralysis label inputs to consistent string representations."""
    label_map = PARALYSIS_MAP  # Use the globally defined map
    if val is None or pd.isna(val): return 'NA'
    val_str = str(val).strip().lower()
    if val_str == '' or val_str == 'not assessed': return 'NA'

    # Check numeric keys first if label_map uses them
    if 0 in label_map and val_str in ['none', 'no', '0', '0.0', 'normal']: return label_map[0]
    if 1 in label_map and val_str in ['partial', 'mild', 'moderate', 'incomplete', 'i', 'p', '1', '1.0']: return \
    label_map[1]
    if 2 in label_map and val_str in ['complete', 'severe', 'c', '2', '2.0']: return label_map[2]

    # Fallback for string keys or if numeric keys didn't match (less ideal)
    if val_str in ['none', 'no', '0', '0.0', 'normal']: return label_map.get(0, 'None')
    if val_str in ['partial', 'mild', 'moderate', 'incomplete', 'i', 'p', '1', '1.0']: return label_map.get(1,
                                                                                                            'Partial')
    if val_str in ['complete', 'severe', 'c', '2', '2.0']: return label_map.get(2, 'Complete')

    if val_str == 'error': return 'Error'  # Specific 'Error' state
    return 'NA'  # Default for unmapped values


def standardize_synkinesis_labels(val):
    """Standardizes various synkinesis label inputs to consistent string representations."""
    label_map = SYNKINESIS_MAP  # Use the globally defined map
    positive_label_name = label_map.get(1, 'Synkinesis')  # Default if key 1 not in map
    negative_label_name = label_map.get(0, 'None')  # Default if key 0 not in map

    if val is None or pd.isna(val): return 'NA'
    val_str = str(val).strip().lower()
    if val_str == '' or val_str == 'not assessed': return 'NA'

    # Check numeric keys first
    if 1 in label_map and val_str in ['yes', 'true', '1', '1.0', 'y', 't', 'partial', 'mild', 'moderate', 'incomplete',
                                      'i', 'p', 'complete', 'severe', 'c', '2', '2.0']:
        return label_map[1]
    if 0 in label_map and val_str in ['no', 'false', '0', '0.0', 'none', 'normal', 'n', 'f']:
        return label_map[0]

    # Fallback for string keys or if numeric keys didn't match
    if val_str in ['yes', 'true', '1', '1.0', 'y', 't', 'partial', 'mild', 'moderate', 'incomplete', 'i', 'p',
                   'complete', 'severe', 'c', '2', '2.0']:
        return positive_label_name
    if val_str in ['no', 'false', '0', '0.0', 'none', 'normal', 'n', 'f']:
        return negative_label_name

    if val_str == 'error': return 'Error'
    return 'NA'


# --- Binary Target Processing ---
def process_binary_target(target_series):
    """Processes a target series into binary (0 or 1) format."""
    if target_series is None: return np.array([], dtype=int)
    if not isinstance(target_series, pd.Series): target_series = pd.Series(target_series)
    if target_series.empty: return np.array([], dtype=int)

    # Comprehensive mapping to 'yes' or 'no'
    s_clean = target_series.astype(str).str.lower().str.strip().replace({
        'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no', 'not assessed': 'no',
        'normal': 'no', 'f': 'false', 'n': 'no', 'false': 'no', '0': 'no', '0.0': 'no',
        'yes': 'yes', 'true': 'yes', 'y': 'yes', 't': 'yes', '1': 'yes', '1.0': 'yes',
        # Include paralysis-like terms as 'yes' for a generic binary case
        'partial': 'yes', 'mild': 'yes', 'moderate': 'yes', 'incomplete': 'yes', 'i': 'yes', 'p': 'yes',
        'complete': 'yes', 'severe': 'yes', 'c': 'yes', '2': 'yes', '2.0': 'yes'
    })
    mapping = {'yes': 1, 'no': 0}
    mapped = s_clean.map(mapping)

    # Default unmapped values (after comprehensive cleaning) to 0.
    # This assumes anything not explicitly 'yes' is 'no'.
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int).values


# --- Base AU Feature Extraction Helper ---
def _extract_base_au_features(df_input, side, actions_list, aus_list, feature_extraction_config,
                              zone_display_name="Zone"):
    """
    Helper function to extract common Action Unit (AU) based features for a given side.
    Compares AUs on the specified 'side' with the opposite side.
    """
    feature_data_dict = {}
    opposite_side_str = 'Right' if side == 'Left' else 'Left'

    use_normalized_val = feature_extraction_config.get('use_normalized', True)
    min_value_param = feature_extraction_config.get('min_value', 0.0001)
    percent_diff_cap_val = feature_extraction_config.get('percent_diff_cap', 200.0)

    for action_str in actions_list:
        for au_str in aus_list:
            base_col_name_str = f"{action_str}_{au_str}"  # e.g., BS_AU12_r

            # Construct column names for current and opposite side, raw and normalized
            au_col_current_side = f"{action_str}_{side} {au_str}"  # e.g., BS_Left AU12_r
            au_norm_col_current_side = f"{au_col_current_side} (Normalized)"
            au_col_opposite_side = f"{action_str}_{opposite_side_str} {au_str}"  # e.g., BS_Right AU12_r
            au_norm_col_opposite_side = f"{au_col_opposite_side} (Normalized)"

            # Get raw values, default to Series of 0.0 if column missing
            raw_val_current_side_series = df_input.get(au_col_current_side, pd.Series(0.0, index=df_input.index))
            raw_val_opposite_side_series = df_input.get(au_col_opposite_side, pd.Series(0.0, index=df_input.index))

            # Coerce to numeric and fill NaNs with 0.0
            raw_val_current_side = pd.to_numeric(raw_val_current_side_series, errors='coerce').fillna(0.0)
            raw_val_opposite_side = pd.to_numeric(raw_val_opposite_side_series, errors='coerce').fillna(0.0)

            val_current_side_to_use = raw_val_current_side
            val_opposite_side_to_use = raw_val_opposite_side

            if use_normalized_val:
                # Get normalized values, fallback to raw values if normalized are missing or NaN
                norm_val_current_side_series = df_input.get(au_norm_col_current_side,
                                                            raw_val_current_side_series)  # Fallback to raw series
                norm_val_opposite_side_series = df_input.get(au_norm_col_opposite_side,
                                                             raw_val_opposite_side_series)  # Fallback to raw series

                # Coerce normalized to numeric, if that fails (e.g. all NaNs), use already coerced raw value
                val_current_side_to_use_temp = pd.to_numeric(norm_val_current_side_series, errors='coerce')
                val_current_side_to_use = val_current_side_to_use_temp.fillna(raw_val_current_side)

                val_opposite_side_to_use_temp = pd.to_numeric(norm_val_opposite_side_series, errors='coerce')
                val_opposite_side_to_use = val_opposite_side_to_use_temp.fillna(raw_val_opposite_side)

            # Store features
            feature_data_dict[f"{base_col_name_str}_val_side"] = val_current_side_to_use
            feature_data_dict[f"{base_col_name_str}_val_opp"] = val_opposite_side_to_use
            feature_data_dict[f"{base_col_name_str}_Asym_Diff"] = val_current_side_to_use - val_opposite_side_to_use
            feature_data_dict[f"{base_col_name_str}_Asym_Ratio"] = calculate_ratio(
                val_current_side_to_use, val_opposite_side_to_use, min_value=min_value_param
            )
            feature_data_dict[f"{base_col_name_str}_Asym_PercDiff"] = calculate_percent_diff(
                val_current_side_to_use, val_opposite_side_to_use, min_value=min_value_param, cap=percent_diff_cap_val
            )
            feature_data_dict[f"{base_col_name_str}_Is_Weaker_Side"] = (
                        val_current_side_to_use < val_opposite_side_to_use).astype(int)

    return pd.DataFrame(feature_data_dict, index=df_input.index)


# --- Generalized prepare_data function ---
def prepare_data_generalized(zone_key, results_file_path=None, expert_file_path=None, base_config_dict=None,
                             input_files_global_dict=None, class_names_global_dict=None):
    if base_config_dict is None:
        from paralysis_config import ZONE_CONFIG as base_config_imported  # Dynamic import if not passed
        base_config_dict = base_config_imported
    if input_files_global_dict is None:
        from paralysis_config import INPUT_FILES as input_files_imported  # Dynamic import
        input_files_global_dict = input_files_imported
    if class_names_global_dict is None:
        class_names_global_dict = PARALYSIS_MAP  # Use the globally defined map

    try:
        config_zone_specific = base_config_dict[zone_key]
        zone_name_display = config_zone_specific.get('name', zone_key.capitalize() + ' Face')
        feature_sel_cfg_zone = config_zone_specific.get('feature_selection', {})
        filenames_zone = config_zone_specific.get('filenames', {})
        expert_cols_zone = config_zone_specific.get('expert_columns', {})
    except KeyError:
        logger.critical(f"CRITICAL: Zone '{zone_key}' config missing in base_config_dict. Cannot prepare data.")
        return None, None, None

    current_results_file = results_file_path or input_files_global_dict.get('results_csv')
    current_expert_file = expert_file_path or input_files_global_dict.get('expert_key_csv')

    if not all([current_results_file, current_expert_file, os.path.exists(current_results_file),
                os.path.exists(current_expert_file)]):
        logger.error(
            f"[{zone_name_display}] Input files not found. Results: '{current_results_file}', Expert: '{current_expert_file}'")
        return None, None, None
    try:
        results_df = pd.read_csv(current_results_file, low_memory=False)
        expert_df = pd.read_csv(current_expert_file, dtype=str, keep_default_na=False,
                                na_values=['', 'NA', 'N/A'])  # Treat various NAs as NaN
    except Exception as e:
        logger.error(f"[{zone_name_display}] Error loading data: {e}.", exc_info=True)
        return None, None, None

    # Standardize 'Patient ID' column name
    expert_rename_map = {'Patient': 'Patient ID'}  # Default rename
    # Check if 'Patient ID' already exists, if so, don't rename 'Patient'
    if 'Patient ID' in expert_df.columns and 'Patient' in expert_df.columns and 'Patient ID' != 'Patient':
        logger.debug(f"[{zone_name_display}] Both 'Patient' and 'Patient ID' found in expert file. Using 'Patient ID'.")
        if 'Patient' in expert_rename_map: del expert_rename_map[
            'Patient']  # Avoid renaming 'Patient' if 'Patient ID' is primary

    expert_left_orig_name = expert_cols_zone.get('left')
    expert_right_orig_name = expert_cols_zone.get('right')

    if not all([expert_left_orig_name, expert_right_orig_name]):
        logger.error(
            f"[{zone_name_display}] Missing 'left' or 'right' expert column names in config for zone '{zone_key}'.")
        return None, None, None

    for col_name in [expert_left_orig_name, expert_right_orig_name]:
        if col_name not in expert_df.columns:
            logger.error(
                f"[{zone_name_display}] Expert column '{col_name}' (from config) not found in expert file columns: {expert_df.columns.tolist()}.")
            return None, None, None
        expert_rename_map[col_name] = col_name  # Ensure these columns are part of rename map (effectively keeps them)

    expert_df_renamed = expert_df.rename(columns=expert_rename_map)

    # Define required columns for merging after potential rename
    # Patient ID might have been 'Patient' initially
    patient_id_col_final = 'Patient ID' if 'Patient ID' in expert_df_renamed.columns else None
    if not patient_id_col_final:  # Should not happen if rename logic is correct
        logger.error(f"[{zone_name_display}] 'Patient ID' column not found in expert_df_renamed.")
        return None, None, None

    expert_cols_to_merge_list = [patient_id_col_final, expert_left_orig_name, expert_right_orig_name]

    # Check for duplicates based on the final Patient ID column and keep first
    expert_df_subset_final = expert_df_renamed[expert_cols_to_merge_list].copy()
    if expert_df_subset_final[patient_id_col_final].duplicated().any():
        logger.warning(
            f"[{zone_name_display}] Duplicate Patient IDs found in expert file. Keeping first instance for each ID.")
        expert_df_subset_final.drop_duplicates(subset=[patient_id_col_final], keep='first', inplace=True)

    # Standardize Patient ID types for merge
    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
    expert_df_subset_final[patient_id_col_final] = expert_df_subset_final[patient_id_col_final].astype(str).str.strip()

    try:
        merged_df = pd.merge(results_df, expert_df_subset_final, on=patient_id_col_final, how='inner',
                             validate="many_to_one")
    except Exception as e:
        logger.error(f"[{zone_name_display}] Merge failed: {e}", exc_info=True)
        return None, None, None
    if merged_df.empty:
        logger.error(
            f"[{zone_name_display}] Merge resulted in an empty DataFrame. Check Patient IDs and file contents.")
        return None, None, None

    # Standardize expert labels and create valid masks
    merged_df['Expert_Std_Left'] = merged_df[expert_left_orig_name].apply(standardize_paralysis_labels)
    merged_df['Expert_Std_Right'] = merged_df[expert_right_orig_name].apply(standardize_paralysis_labels)
    valid_left_mask_final = merged_df['Expert_Std_Left'] != 'NA'
    valid_right_mask_final = merged_df['Expert_Std_Right'] != 'NA'

    # Dynamically import the feature extraction module for the zone
    feature_module_name = f"{zone_key}_face_features"
    try:
        feature_module = importlib.import_module(feature_module_name)
        logger.info(f"[{zone_name_display}] Extracting features for Left side using {feature_module_name}...")
        left_features_df = feature_module.extract_features(merged_df, 'Left', config_zone_specific)
        logger.info(f"[{zone_name_display}] Extracting features for Right side using {feature_module_name}...")
        right_features_df = feature_module.extract_features(merged_df, 'Right', config_zone_specific)
    except ModuleNotFoundError:
        logger.error(f"[{zone_name_display}] Feature module '{feature_module_name}.py' not found.", exc_info=True)
        return None, None, None
    except AttributeError:
        logger.error(
            f"[{zone_name_display}] 'extract_features' function not found in module '{feature_module_name}.py'.",
            exc_info=True)
        return None, None, None
    except Exception as e_extract:
        logger.error(f"[{zone_name_display}] Error during feature extraction via {feature_module_name}: {e_extract}",
                     exc_info=True)
        return None, None, None

    if left_features_df is None or right_features_df is None:
        logger.error(f"[{zone_name_display}] Feature extraction returned None for one or both sides.")
        return None, None, None

    # Filter features and metadata based on valid labels
    filtered_left_features = left_features_df[valid_left_mask_final].copy()
    filtered_right_features = right_features_df[valid_right_mask_final].copy()

    metadata_left = merged_df.loc[valid_left_mask_final, [patient_id_col_final]].copy();
    metadata_left.rename(columns={patient_id_col_final: 'Patient ID'}, inplace=True);
    metadata_left['Side'] = 'Left'
    metadata_right = merged_df.loc[valid_right_mask_final, [patient_id_col_final]].copy();
    metadata_right.rename(columns={patient_id_col_final: 'Patient ID'}, inplace=True);
    metadata_right['Side'] = 'Right'

    valid_left_expert_labels_raw = merged_df.loc[valid_left_mask_final, expert_left_orig_name]
    valid_right_expert_labels_raw = merged_df.loc[valid_right_mask_final, expert_right_orig_name]

    target_mapping_final = {name: num_val for num_val, name in class_names_global_dict.items()}

    def standardize_and_map_valid(series_to_map, current_zone_name_log):
        if series_to_map.empty: return np.array([], dtype=int)
        s_standardized = series_to_map.apply(standardize_paralysis_labels)
        mapped = s_standardized.map(target_mapping_final)
        if mapped.isna().any():
            unmapped_series = s_standardized[mapped.isna()]
            if not unmapped_series.empty:
                unmapped_labels = pd.unique(unmapped_series)
                logger.warning(
                    f"[{current_zone_name_log}] Standardized labels {list(unmapped_labels)} not found in target_mapping_final ({target_mapping_final}). Defaulting these to 0 (None).")
            else:
                logger.debug(
                    f"[{current_zone_name_log}] mapped.isna().any() was true, but no specific unmapped labels found to list.")
            mapped = mapped.fillna(0)
        return mapped.astype(int)

    left_targets = standardize_and_map_valid(valid_left_expert_labels_raw, zone_name_display)
    right_targets = standardize_and_map_valid(valid_right_expert_labels_raw, zone_name_display)

    # Add side indicator
    if not filtered_left_features.empty: filtered_left_features['side_indicator'] = 0
    if not filtered_right_features.empty: filtered_right_features['side_indicator'] = 1

    features_combined = pd.concat([df for df in [filtered_left_features, filtered_right_features] if not df.empty],
                                  ignore_index=True)
    targets_combined = np.concatenate([left_targets, right_targets]) if len(left_targets) > 0 or len(
        right_targets) > 0 else np.array([])
    metadata_combined = pd.concat([df for df in [metadata_left, metadata_right] if not df.empty], ignore_index=True)

    if not (len(features_combined) == len(targets_combined) == len(metadata_combined)):
        logger.error(
            f"[{zone_name_display}] Mismatch in lengths after final concat! Features: {len(features_combined)}, Targets: {len(targets_combined)}, Metadata: {len(metadata_combined)}.")
        return None, None, None

    features_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_combined = features_combined.fillna(0)

    # --- Feature Selection Logic ---
    fs_enabled_final = feature_sel_cfg_zone.get('enabled', False)
    if fs_enabled_final:
        n_top_features_cfg_str = feature_sel_cfg_zone.get('top_n_features')
        importance_file_path_cfg = feature_sel_cfg_zone.get('importance_file')

        if not importance_file_path_cfg or not os.path.exists(importance_file_path_cfg):
            logger.warning(
                f"[{zone_name_display}] FS: Importance file not found: '{importance_file_path_cfg}'. Using all {features_combined.shape[1]} features.")
        elif n_top_features_cfg_str is None:
            logger.warning(
                f"[{zone_name_display}] FS: 'top_n_features' not set in config. Using all {features_combined.shape[1]} features.")
        else:
            try:
                n_top_features_cfg = int(n_top_features_cfg_str)
                importance_df_fs = pd.read_csv(importance_file_path_cfg)
                logger.debug(
                    f"[{zone_name_display}] FS: Loaded importance file '{importance_file_path_cfg}' with {len(importance_df_fs)} features listed.")

                if 'feature' not in importance_df_fs.columns or importance_df_fs.empty:
                    logger.error(
                        f"[{zone_name_display}] FS: Importance file '{importance_file_path_cfg}' is invalid. Using all {features_combined.shape[1]} features.")
                else:
                    all_features_from_file = importance_df_fs['feature'].tolist()
                    logger.debug(
                        f"[{zone_name_display}] FS: Total features in importance file: {len(all_features_from_file)}. Requested top_n: {n_top_features_cfg}.")

                    top_feature_names_from_file = importance_df_fs['feature'].head(n_top_features_cfg).tolist()
                    logger.debug(
                        f"[{zone_name_display}] FS: Features taken from file (up to top_n): {len(top_feature_names_from_file)}. Sample: {top_feature_names_from_file[:5]}")

                    current_features_cols_list = features_combined.columns.tolist()
                    logger.debug(
                        f"[{zone_name_display}] FS: Current features in DataFrame before selection: {len(current_features_cols_list)}. Sample: {current_features_cols_list[:5]}")

                    cols_to_keep_final_fs = [col for col in top_feature_names_from_file if
                                             col in current_features_cols_list]
                    logger.debug(
                        f"[{zone_name_display}] FS: Features kept after matching with current DataFrame: {len(cols_to_keep_final_fs)}. Sample: {cols_to_keep_final_fs[:5]}")

                    if not cols_to_keep_final_fs:
                        logger.error(
                            f"[{zone_name_display}] FS: No features from importance file list found in current features. Using all {features_combined.shape[1]} features.")
                    else:
                        if 'side_indicator' in current_features_cols_list and 'side_indicator' not in cols_to_keep_final_fs:
                            cols_to_keep_final_fs.append('side_indicator')
                            logger.debug(
                                f"[{zone_name_display}] FS: 'side_indicator' added. Total to keep now: {len(cols_to_keep_final_fs)}")

                        cols_to_keep_final_fs = list(pd.Series(cols_to_keep_final_fs).drop_duplicates().tolist())
                        features_combined = features_combined[cols_to_keep_final_fs]
                        logger.info(
                            f"[{zone_name_display}] Selected {features_combined.shape[1]} features via importance file (requested top {n_top_features_cfg}).")
            except ValueError:
                logger.error(
                    f"[{zone_name_display}] FS: 'top_n_features' ({n_top_features_cfg_str}) is not a valid integer. Using all {features_combined.shape[1]} features.")
            except Exception as e_fs_final:
                logger.error(
                    f"[{zone_name_display}] FS error: {e_fs_final}. Using all {features_combined.shape[1]} features.",
                    exc_info=True)
    else:
        logger.info(
            f"[{zone_name_display}] Feature selection disabled by config. Using all {features_combined.shape[1]} features.")

    if features_combined.isnull().values.any():
        logger.warning(f"[{zone_name_display}] NaNs found in FINAL features AFTER selection. Filling with 0 again.")
        features_combined = features_combined.fillna(0)

    final_feature_names_list_save = features_combined.columns.tolist()
    feature_list_path_to_save = filenames_zone.get('feature_list')
    if feature_list_path_to_save:
        try:
            os.makedirs(os.path.dirname(feature_list_path_to_save), exist_ok=True)
            with open(feature_list_path_to_save, 'w') as f:
                for feature_name in final_feature_names_list_save:
                    f.write(f"{feature_name}\n")
            logger.info(
                f"[{zone_name_display}] Final feature list ({len(final_feature_names_list_save)}) saved to {feature_list_path_to_save}.")
        except Exception as e_save_fl_final:
            logger.error(f"[{zone_name_display}] Failed to save final feature list: {e_save_fl_final}", exc_info=True)
    else:
        logger.error(
            f"[{zone_name_display}] Feature list path ('feature_list') not defined in config filenames for zone '{zone_key}'. Cannot save.")

    return features_combined, targets_combined, metadata_combined


# --- Performance Analysis Helpers (from v7) ---
def visualize_confusion_matrix(cm, categories, title, output_dir):
    """Visualizes a confusion matrix and saves it to a file."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories, yticklabels=categories, cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f"{title} Confusion Matrix")
        # Sanitize title for filename
        safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{safe_title}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close plot to free memory
        logger.debug(f"Confusion matrix '{title}' saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save Confusion Matrix '{title}': {e}", exc_info=True)
        plt.close()  # Ensure plot is closed on error too


def perform_error_analysis(data, output_dir, filename_base, finding_name, class_names_map):
    """Performs error analysis and saves details to a text file."""
    try:
        if 'Expert' not in data.columns or 'Prediction' not in data.columns:
            logger.warning(f"Error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' column missing.")
            return
        # Ensure columns are numeric before comparison
        if not pd.api.types.is_numeric_dtype(data['Expert']) or not pd.api.types.is_numeric_dtype(data['Prediction']):
            logger.warning(f"Error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' not numeric.")
            return

        errors_df = data[data['Prediction'] != data['Expert']].copy()
        error_patterns = {}
        for _, row in errors_df.iterrows():
            expert_label = class_names_map.get(int(row['Expert']), f"UnknownExpert({int(row['Expert'])})")
            pred_label = class_names_map.get(int(row['Prediction']), f"UnknownPred({int(row['Prediction'])})")
            pattern = f"Expert_{expert_label}_Predicted_{pred_label}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

        filename_base_safe = os.path.basename(filename_base)  # Use only filename for report name
        total = len(data)
        n_err = len(errors_df)
        err_rate = (n_err / total * 100) if total > 0 else 0

        details_dir = output_dir  # Use provided output_dir directly
        os.makedirs(details_dir, exist_ok=True)
        path = os.path.join(details_dir, f"{filename_base_safe}_error_details.txt")  # More specific filename

        with open(path, 'w') as f:
            f.write(f"Error Analysis Details: {finding_name}\nFile: {os.path.basename(path)}\n{'=' * 40}\n")
            f.write(
                f"Total Valid Predictions Analyzed: {total}\nNumber of Errors: {n_err} ({err_rate:.2f}%)\n\nError Patterns Summary:\n")
            if error_patterns:
                sorted_patterns = sorted(error_patterns.items(), key=lambda item: item[1], reverse=True)
                for pattern, count in sorted_patterns:
                    f.write(f"- {pattern}: {count} ({(count / n_err * 100) if n_err > 0 else 0:.2f}% of errors)\n")
            else:
                f.write("- No errors.\n")

            f.write("\nIndividual Error Cases:\n")
            if n_err > 0:
                id_col = 'Patient ID' if 'Patient ID' in errors_df.columns else None
                side_col = 'Side' if 'Side' in errors_df.columns else None

                errors_df_reset = errors_df.reset_index()  # Keep original index if needed as 'index' column
                sort_cols = [col for col in [id_col, side_col] if col and col in errors_df_reset.columns]

                # Use original index for sorting if Patient ID/Side are not available or cause issues
                if not sort_cols or not all(col in errors_df_reset.columns for col in sort_cols):
                    sort_cols = ['index'] if 'index' in errors_df_reset.columns else []

                try:
                    errors_sorted = errors_df_reset.sort_values(
                        by=sort_cols) if sort_cols else errors_df_reset.sort_index()
                except Exception as e_sort:
                    logger.warning(f"Sorting errors failed: {e_sort}. Using unsorted errors.")
                    errors_sorted = errors_df_reset  # Fallback to unsorted if sort fails

                id_col_write = id_col if id_col and id_col in errors_sorted.columns else (
                    'index' if 'index' in errors_sorted.columns else None)

                for i, row in errors_sorted.iterrows():  # i is the new index after reset/sort
                    row_idx_val = row.get('index', i)  # Prefer original index if available

                    patient_info_parts = []
                    if id_col_write and id_col_write != 'index' and id_col_write in row and pd.notna(row[id_col_write]):
                        patient_info_parts.append(f"Pt {row[id_col_write]}")
                    else:
                        patient_info_parts.append(f"CaseIdx {row_idx_val}")

                    if side_col and side_col in row and pd.notna(row[side_col]):
                        patient_info_parts.append(f"Side: {row[side_col]}")

                    patient_info_str = ", ".join(patient_info_parts)

                    expert_str = class_names_map.get(int(row['Expert']), f"?({int(row['Expert'])})")
                    pred_str = class_names_map.get(int(row['Prediction']), f"?({int(row['Prediction'])})")
                    f.write(f"  {patient_info_str} - Exp: {expert_str}, Pred: {pred_str}\n")
            else:
                f.write("- No errors to list.\n")
        logger.debug(f"Error analysis for '{finding_name}' saved to {path}")
    except KeyError as ke:
        logger.error(f"KeyError in perform_error_analysis for '{finding_name}': {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"Error in perform_error_analysis for '{finding_name}': {e}", exc_info=True)


def analyze_critical_errors(data, output_dir, filename_base, finding_name, class_names_map):
    """Analyzes critical errors (None <-> Complete) and saves details."""
    none_label_val = 0
    complete_label_val = 2
    if not (none_label_val in class_names_map and complete_label_val in class_names_map):
        logger.warning(
            f"Critical error analysis for '{finding_name}' skipped: Class names for 0 or 2 missing from map {class_names_map}.")
        return

    none_label_str = class_names_map[none_label_val]
    complete_label_str = class_names_map[complete_label_val]

    try:
        if 'Expert' not in data.columns or 'Prediction' not in data.columns:
            logger.warning(f"Critical error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' missing.")
            return
        if not pd.api.types.is_numeric_dtype(data['Expert']) or not pd.api.types.is_numeric_dtype(data['Prediction']):
            logger.warning(
                f"Critical error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' not numeric.")
            return

        critical_errors = data[
            ((data['Expert'] == none_label_val) & (data['Prediction'] == complete_label_val)) |
            ((data['Expert'] == complete_label_val) & (data['Prediction'] == none_label_val))
            ].copy()

        n2c = len(critical_errors[(critical_errors['Expert'] == none_label_val) & (
                    critical_errors['Prediction'] == complete_label_val)])
        c2n = len(critical_errors[(critical_errors['Expert'] == complete_label_val) & (
                    critical_errors['Prediction'] == none_label_val)])

        filename_base_safe = os.path.basename(filename_base)
        path = os.path.join(output_dir, f"{filename_base_safe}_critical_errors.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            f.write(
                f"Critical Errors Analysis - {finding_name} ({none_label_str} <-> {complete_label_str})\n{'=' * 40}\n\n")
            f.write(f"Total Critical Errors: {len(critical_errors)}\n")
            f.write(f"  - Misclassified '{none_label_str}' as '{complete_label_str}': {n2c}\n")
            f.write(f"  - Misclassified '{complete_label_str}' as '{none_label_str}': {c2n}\n\n")

            if not critical_errors.empty:
                f.write("Detailed Cases:\n")
                id_col = 'Patient ID' if 'Patient ID' in critical_errors.columns else None
                side_col = 'Side' if 'Side' in critical_errors.columns else None
                critical_errors_reset = critical_errors.reset_index()
                id_col_write = id_col if id_col and id_col in critical_errors_reset.columns else (
                    'index' if 'index' in critical_errors_reset.columns else None)

                for idx, row in critical_errors_reset.iterrows():
                    row_idx_val = row.get('index', idx)
                    patient_info_parts = []
                    if id_col_write and id_col_write != 'index' and id_col_write in row and pd.notna(row[id_col_write]):
                        patient_info_parts.append(f"Pt {row[id_col_write]}")
                    else:
                        patient_info_parts.append(f"CaseIdx {row_idx_val}")
                    if side_col and side_col in row and pd.notna(row[side_col]):
                        patient_info_parts.append(f"Side: {row[side_col]}")
                    patient_info_str = ", ".join(patient_info_parts)

                    exp_str = class_names_map.get(int(row['Expert']), f"?({int(row['Expert'])})")
                    pred_str = class_names_map.get(int(row['Prediction']), f"?({int(row['Prediction'])})")
                    f.write(f"  {patient_info_str} - Exp: {exp_str}, Pred: {pred_str}\n")
            else:
                f.write("No critical errors found.\n")
        logger.debug(f"Critical error analysis for '{finding_name}' saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save critical error file {path if 'path' in locals() else 'unknown'}: {e}",
                     exc_info=True)


def analyze_partial_errors(data, output_dir, filename_base, finding_name, class_names_map):
    """Analyzes errors involving the 'Partial' class and saves details."""
    partial_label_val = 1
    none_label_val = 0
    complete_label_val = 2

    required_labels = [none_label_val, partial_label_val, complete_label_val]
    if not all(label in class_names_map for label in required_labels):
        logger.warning(
            f"Partial error analysis for '{finding_name}' skipped: Class names for 0, 1, or 2 missing from map {class_names_map}.")
        return

    partial_label_str = class_names_map[partial_label_val]
    none_label_str = class_names_map[none_label_val]
    complete_label_str = class_names_map[complete_label_val]

    if 'Expert' not in data.columns or 'Prediction' not in data.columns:
        logger.warning(f"Partial error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' missing.")
        return

    filename_base_safe = os.path.basename(filename_base)
    path = os.path.join(output_dir, f"{filename_base_safe}_partial_errors.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Check if 'Partial' class is even present in the data
    if partial_label_val not in data['Expert'].unique() and partial_label_val not in data['Prediction'].unique():
        try:
            with open(path, "w") as f:
                f.write(
                    f"Partial Errors Analysis - {finding_name}\n{'=' * 40}\n\nNo '{partial_label_str}' labels found in Expert or Prediction columns.\n")
            logger.info(f"Partial error analysis for '{finding_name}' (no partial labels found) saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save empty partial error file {path}: {e}", exc_info=True)
        return

    try:
        if not pd.api.types.is_numeric_dtype(data['Expert']) or not pd.api.types.is_numeric_dtype(data['Prediction']):
            logger.warning(
                f"Partial error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' not numeric.")
            return

        partial_errors = data[
            ((data['Expert'] == partial_label_val) & (data['Prediction'] != partial_label_val)) |
            ((data['Expert'] != partial_label_val) & (data['Prediction'] == partial_label_val))
            ].copy()

        p2n = len(partial_errors[(partial_errors['Expert'] == partial_label_val) & (
                    partial_errors['Prediction'] == none_label_val)])
        p2c = len(partial_errors[(partial_errors['Expert'] == partial_label_val) & (
                    partial_errors['Prediction'] == complete_label_val)])
        n2p = len(partial_errors[(partial_errors['Expert'] == none_label_val) & (
                    partial_errors['Prediction'] == partial_label_val)])
        c2p = len(partial_errors[(partial_errors['Expert'] == complete_label_val) & (
                    partial_errors['Prediction'] == partial_label_val)])

        with open(path, "w") as f:
            f.write(
                f"Partial Errors Analysis - {finding_name}\n{'=' * 40}\n\nTotal Errors involving '{partial_label_str}': {len(partial_errors)}\n")
            f.write(f"  - Expert '{partial_label_str}' Predicted '{none_label_str}': {p2n}\n")
            f.write(f"  - Expert '{partial_label_str}' Predicted '{complete_label_str}': {p2c}\n")
            f.write(f"  - Expert '{none_label_str}' Predicted '{partial_label_str}': {n2p}\n")
            f.write(f"  - Expert '{complete_label_str}' Predicted '{partial_label_str}': {c2p}\n\n")

            if not partial_errors.empty:
                f.write("Detailed Cases:\n")
                id_col = 'Patient ID' if 'Patient ID' in partial_errors.columns else None
                side_col = 'Side' if 'Side' in partial_errors.columns else None
                partial_errors_reset = partial_errors.reset_index()
                id_col_write = id_col if id_col and id_col in partial_errors_reset.columns else (
                    'index' if 'index' in partial_errors_reset.columns else None)

                for idx, row in partial_errors_reset.iterrows():
                    row_idx_val = row.get('index', idx)
                    patient_info_parts = []
                    if id_col_write and id_col_write != 'index' and id_col_write in row and pd.notna(row[id_col_write]):
                        patient_info_parts.append(f"Pt {row[id_col_write]}")
                    else:
                        patient_info_parts.append(f"CaseIdx {row_idx_val}")
                    if side_col and side_col in row and pd.notna(row[side_col]):
                        patient_info_parts.append(f"Side: {row[side_col]}")
                    patient_info_str = ", ".join(patient_info_parts)

                    exp_str = class_names_map.get(int(row['Expert']), f"?({int(row['Expert'])})")
                    pred_str = class_names_map.get(int(row['Prediction']), f"?({int(row['Prediction'])})")
                    f.write(f"  {patient_info_str} - Exp: {exp_str}, Pred: {pred_str}\n")
            else:
                f.write("No partial errors found.\n")
        logger.debug(f"Partial error analysis for '{finding_name}' saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save partial error file {path}: {e}", exc_info=True)


# --- evaluate_thresholds: Kept for binary classification scenarios, might be adapted or removed if not used ---
def evaluate_thresholds(data, proba_left_col, expert_std_left_col, proba_right_col, expert_std_right_col,
                        output_dir, finding_name, results_csv_path, pr_curve_path, class_names_map):
    """Evaluates different probability thresholds for binary classification."""
    required_cols = [proba_left_col, expert_std_left_col, proba_right_col, expert_std_right_col]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        logger.error(f"evaluate_thresholds for '{finding_name}' missing required columns: {missing}. Aborting.")
        return

    # Expects class_names_map to have 0 and 1 for binary mapping
    if not (0 in class_names_map and 1 in class_names_map):
        logger.error(
            f"evaluate_thresholds for '{finding_name}': class_names_map ({class_names_map}) must contain 0 and 1 for binary labels. Aborting.")
        return

    negative_class_name = class_names_map[0]
    positive_class_name = class_names_map[1]

    left_df = data[[proba_left_col, expert_std_left_col]].copy()
    left_df.columns = ['Probability', 'Expert_Std']  # Standardize column names
    right_df = data[[proba_right_col, expert_std_right_col]].copy()
    right_df.columns = ['Probability', 'Expert_Std']

    combined_df = pd.concat([left_df, right_df], ignore_index=True)
    # Filter out rows with 'NA' expert labels or missing probabilities
    combined_valid = combined_df[(combined_df['Expert_Std'] != 'NA') & combined_df['Probability'].notna()].copy()

    if combined_valid.empty:
        logger.warning(f"No valid data for '{finding_name}' threshold evaluation after filtering NA/NaNs.")
        return

    target_mapping = {negative_class_name: 0, positive_class_name: 1}
    combined_valid['Expert'] = combined_valid['Expert_Std'].map(target_mapping)
    combined_valid.dropna(subset=['Expert'], inplace=True)  # Drop if mapping resulted in NaN

    if combined_valid.empty:
        logger.warning(f"No valid data after mapping Expert_Std for '{finding_name}'. Check class names and data.")
        return

    combined_valid['Expert'] = combined_valid['Expert'].astype(int)

    if len(combined_valid['Expert'].unique()) < 2:
        logger.warning(
            f"Only one class present for '{finding_name}' after processing. Skipping threshold evaluation. Unique labels: {combined_valid['Expert'].unique()}")
        return

    thresholds = np.arange(0.05, 1.0, 0.05)  # 0.05 to 0.95
    results = []
    for threshold in thresholds:
        y_pred = (combined_valid['Probability'] >= threshold).astype(int)
        y_true = combined_valid['Expert']

        # pos_label=1 for binary metrics (assuming 1 is the positive class)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1,
                                                                   zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Ensure labels are [0, 1] for consistent TN, FP, FN, TP

        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.size == 4:  # Standard 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:  # Only one class predicted and true (all TN or all TP)
            if y_true.unique()[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
            # Other cells (fp, fn) remain 0

        results.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F1-Score': f1,
                        'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn})

    results_df = pd.DataFrame(results)
    try:
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
        results_df.to_csv(results_csv_path, index=False, float_format='%.4f')
        logger.debug(f"Threshold evaluation results for '{finding_name}' saved to {results_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save threshold results for '{finding_name}': {e}")

    try:
        plt.figure(figsize=(8, 6))
        plt.plot(results_df['Recall'], results_df['Precision'], marker='o', linestyle='-')
        plt.xlabel(f"Recall ({finding_name})")
        plt.ylabel(f"Precision ({finding_name})")
        plt.title(f"{finding_name} Precision-Recall Curve vs. Threshold")
        plt.grid(True)
        plt.xlim([0.0, 1.05]);
        plt.ylim([0.0, 1.05])  # Standard PR curve limits
        for i, row in results_df.iterrows():
            if i % 2 == 0:  # Annotate every other point to avoid clutter
                plt.text(row['Recall'] + 0.01, row['Precision'] + 0.02, f"{row['Threshold']:.2f}")
        os.makedirs(os.path.dirname(pr_curve_path), exist_ok=True)
        plt.savefig(pr_curve_path)
        plt.close()
        logger.debug(f"PR curve for '{finding_name}' saved to {pr_curve_path}")
    except Exception as e:
        logger.error(f"Failed to generate/save PR curve for '{finding_name}': {e}")
        plt.close()


# --- Uncertainty & Influence Helpers ---
def calculate_entropy(probabilities):
    """Calculates Shannon entropy for a probability distribution."""
    probabilities = np.array(probabilities)
    # Normalize if not summing to 1, but only if sum is positive
    if not np.isclose(np.sum(probabilities), 1.0) and np.sum(probabilities) > 0:
        probabilities = probabilities / np.sum(probabilities)
    elif np.sum(probabilities) == 0:  # All probabilities are zero
        return np.log2(len(probabilities)) if len(
            probabilities) > 0 else 0.0  # Max entropy for uniform or 0 if no classes

    # Clip probabilities to avoid log(0)
    prob = np.clip(probabilities, 1e-9, 1.0)
    return scipy.stats.entropy(prob, base=2)


def calculate_margin(probabilities):
    """Calculates the margin of confidence (diff between top two probabilities)."""
    if probabilities is None: return 1.0  # Max uncertainty if no probabilities
    probs_array = np.asarray(probabilities)
    if len(probs_array) < 2: return 1.0  # Max uncertainty if only one class prob

    sorted_probs = np.sort(probs_array)[::-1]  # Sort descending
    return sorted_probs[0] - sorted_probs[1]


def analyze_training_influence(zone, model, scaler, feature_names,
                               X_train_orig, y_train_orig,
                               X_test_orig, y_test_orig,  # Test set for evaluating impact
                               training_analysis_df,  # Not directly used here but part of signature
                               candidate_indices,  # Indices from X_train_orig to analyze
                               config):  # Zone-specific config
    """
    Analyzes the influence of specific training samples by retraining the model without them.
    (Simplified version - full LOO can be very expensive).
    """
    zone_name = config.get('name', zone.capitalize().replace('_', ' ') + ' Face')
    training_params = config.get('training', {})
    smote_config = training_params.get('smote', {})
    smote_enabled = smote_config.get('enabled', False)
    random_state = training_params.get('random_state', 42)
    # Use base XGBoost model_params from config, not the tuned ones from a full run for consistency here
    model_params_config = training_params.get('model_params', {})

    influence_scores = {}  # Store {index: delta_f1}

    if candidate_indices.empty:
        logger.info(f"[{zone_name}] No candidate indices provided for influence analysis.")
        return influence_scores

    try:
        # Ensure X_test_orig is DataFrame for scaler, if not already
        if not isinstance(X_test_orig, pd.DataFrame) and feature_names:
            X_test_orig_df = pd.DataFrame(X_test_orig, columns=feature_names)
        elif isinstance(X_test_orig, pd.DataFrame):
            X_test_orig_df = X_test_orig
        else:
            logger.error(
                f"[{zone_name}] X_test_orig is not DataFrame and no feature_names provided for influence analysis.")
            return {}  # Return empty if cannot proceed

        X_test_scaled = scaler.transform(X_test_orig_df)
        y_pred_baseline = model.predict(X_test_scaled)  # Baseline model is the fully trained one
        baseline_f1 = f1_score(y_test_orig, y_pred_baseline, average='weighted', zero_division=0)
        logger.info(f"[{zone_name}] Baseline F1 for influence analysis: {baseline_f1:.4f}")

        # Prepare parameters for retraining XGBoost model
        # These should be the *non-tuned* base parameters for consistency
        retrain_params = {k: v for k, v in model_params_config.items() if
                          k in ['objective', 'eval_metric', 'learning_rate', 'max_depth', 'min_child_weight',
                                'subsample', 'colsample_bytree', 'gamma', 'n_estimators',
                                'scale_pos_weight', 'reg_alpha', 'reg_lambda']}  # Removed random_state, use global
        retrain_params['random_state'] = random_state  # Consistent random state

        # Ensure objective and num_class are set based on PARALYSIS_MAP (or the relevant class map)
        class_map_for_influence = PARALYSIS_MAP  # Assuming paralysis for now
        if class_map_for_influence and len(class_map_for_influence) > 2:
            retrain_params['objective'] = model_params_config.get('objective', 'multi:softprob')
            retrain_params['num_class'] = len(class_map_for_influence)
            if 'eval_metric' not in retrain_params: retrain_params['eval_metric'] = 'mlogloss'
        else:  # Binary case
            retrain_params['objective'] = model_params_config.get('objective', 'binary:logistic')
            if 'num_class' in retrain_params: del retrain_params['num_class']  # Not needed for binary
            if 'eval_metric' not in retrain_params: retrain_params['eval_metric'] = 'logloss'

        retrain_params['use_label_encoder'] = False  # Modern XGBoost

        analyzed_count = 0
        for idx in candidate_indices:
            if idx not in X_train_orig.index:
                logger.warning(f"[{zone_name}] Index {idx} for influence analysis not in X_train_orig.index. Skipping.")
                influence_scores[idx] = np.nan
                continue

            try:
                X_train_temp_df = X_train_orig.drop(index=idx)
                # Get integer position of index label to delete from numpy array y_train_orig
                idx_pos = X_train_orig.index.get_loc(idx)
                if not np.isscalar(idx_pos):  # Should be scalar if index is unique
                    logger.warning(f"[{zone_name}] Index {idx} resulted in non-scalar position. Skipping.")
                    influence_scores[idx] = np.nan
                    continue
                y_train_temp_arr = np.delete(y_train_orig, idx_pos)
            except Exception as e_drop:
                logger.warning(f"[{zone_name}] Error dropping index {idx} for influence analysis: {e_drop}. Skipping.")
                influence_scores[idx] = np.nan
                continue

            X_train_processed_df = X_train_temp_df
            y_train_processed_arr = y_train_temp_arr

            if smote_enabled:  # Re-apply SMOTE if it was used in original training
                unique_labels_temp, counts_temp = np.unique(y_train_temp_arr, return_counts=True)
                if len(unique_labels_temp) <= 1 or any(c == 0 for c in counts_temp):  # SMOTE needs >1 class
                    logger.debug(f"[{zone_name}] Skipping SMOTE for influence trial {idx}: not enough class diversity.")
                    influence_scores[idx] = np.nan
                    continue
                try:
                    min_class_size_temp = np.min(counts_temp[counts_temp > 0])  # Smallest class with samples
                    k_neighbors_config = smote_config.get('k_neighbors', 5)
                    actual_k_neighbors = max(1, min(k_neighbors_config,
                                                    min_class_size_temp - 1)) if min_class_size_temp > 1 else 0

                    if actual_k_neighbors >= 1:
                        # Use simple 'auto' SMOTE for influence to reduce complexity, or mirror main config
                        smote_influence = SMOTE(random_state=random_state, k_neighbors=actual_k_neighbors,
                                                sampling_strategy='auto')
                        # SMOTE expects DataFrame features
                        X_smoted_temp, y_smoted_temp = smote_influence.fit_resample(X_train_temp_df, y_train_temp_arr)
                        X_train_processed_df = pd.DataFrame(X_smoted_temp,
                                                            columns=feature_names)  # Convert back if SMOTE returns array
                        y_train_processed_arr = y_smoted_temp
                except Exception as e_smote_inf:
                    logger.warning(
                        f"[{zone_name}] SMOTE failed during influence analysis for index {idx}: {e_smote_inf}. Using un-SMOTEd temp data.")
                    # X_train_processed_df, y_train_processed_arr remain as X_train_temp_df, y_train_temp_arr

            if len(np.unique(y_train_processed_arr)) <= 1:  # Check again after SMOTE
                logger.debug(
                    f"[{zone_name}] Not enough class diversity after processing for influence trial {idx}. Skipping.")
                influence_scores[idx] = np.nan
                continue

            try:
                # Scale the potentially SMOTEd temporary training data
                # Important: fit_transform on the new X_train_processed_df, then transform X_test_orig_df
                temp_scaler = StandardScaler()
                X_train_scaled_temp = temp_scaler.fit_transform(X_train_processed_df)
                X_test_scaled_temp = temp_scaler.transform(X_test_orig_df)  # Use the same test set

                temp_model = xgb.XGBClassifier(**retrain_params)
                temp_model.fit(X_train_scaled_temp, y_train_processed_arr,
                               verbose=False)  # No early stopping here for simplicity

                y_pred_temp = temp_model.predict(X_test_scaled_temp)
                temp_f1 = f1_score(y_test_orig, y_pred_temp, average='weighted', zero_division=0)
                influence_scores[idx] = temp_f1 - baseline_f1
            except Exception as e_retrain:
                logger.warning(f"[{zone_name}] Retraining/evaluation failed for influence index {idx}: {e_retrain}.")
                influence_scores[idx] = np.nan  # Mark as NaN if retraining fails

            analyzed_count += 1
            if analyzed_count % 10 == 0:  # Log progress
                logger.info(
                    f"[{zone_name}] Influence analysis: Processed {analyzed_count}/{len(candidate_indices)} candidates.")

        logger.info(f"[{zone_name}] Influence analysis complete for {analyzed_count} candidates.")
        return influence_scores
    except Exception as e:
        logger.error(f"Error in overall training influence analysis for {zone_name}: {e}", exc_info=True)
        return {}  # Return empty on major error


# --- Review Candidate Generation Helpers ---
def generate_review_recommendations(training_analysis_df, error_analysis_df, metadata_df, class_names_map):
    """
    Generates review recommendations by combining training analysis, error analysis, and metadata.
    (This is a high-level combination, specific scoring is in calculate_review_priority_score)
    """
    review_df = training_analysis_df.copy() if training_analysis_df is not None and not training_analysis_df.empty else pd.DataFrame()

    if metadata_df is not None and not metadata_df.empty:
        # Ensure indices can align for merge/join. If they are different, might need reset_index then merge on common columns.
        # Assuming training_analysis_df and metadata_df share the same index from original data split.
        if review_df.empty:  # If training_analysis_df was empty, start with metadata
            review_df = metadata_df.copy()
        elif review_df.index.equals(metadata_df.index):
            review_df = review_df.join(metadata_df, how='left')  # Use join if indices are aligned
        else:  # Fallback if indices are not directly alignable (less ideal)
            logger.warning(
                "Indices of training_analysis_df and metadata_df do not match for review recommendations. Attempting merge on Patient ID and Side if available.")
            merge_cols_meta = [col for col in ['Patient ID', 'Side'] if
                               col in metadata_df.columns and col in review_df.columns]
            if merge_cols_meta:
                review_df = pd.merge(review_df.reset_index(), metadata_df, on=merge_cols_meta, how='left').set_index(
                    review_df.index.name or 'index')
            else:
                logger.error(
                    "Cannot merge metadata_df into review_df due to differing indices and no common 'Patient ID'/'Side' columns.")

    if error_analysis_df is not None and not error_analysis_df.empty:
        # error_analysis_df usually comes from test set, training_analysis_df from train set.
        # This merge might not be meaningful unless error_analysis_df is also from training data or structured similarly.
        # Assuming 'Error_Type', 'Model_Confidence' are specific to error_analysis_df.
        merge_cols = ['Patient ID', 'Side']  # Common identifiers
        if all(col in error_analysis_df.columns for col in merge_cols) and all(
                col in review_df.columns for col in merge_cols):
            error_cols_to_merge = [col for col in (merge_cols + ['Error_Type', 'Model_Confidence']) if
                                   col in error_analysis_df.columns]
            # Perform a left merge to keep all review_df rows and add error info where it matches
            review_df = pd.merge(review_df.reset_index(), error_analysis_df[error_cols_to_merge], on=merge_cols,
                                 how='left', suffixes=('', '_err')).set_index(review_df.index.name or 'index')
        else:
            logger.warning(
                "Cannot merge error_analysis_df: 'Patient ID' or 'Side' missing in review_df or error_analysis_df.")

    # Apply review priority scoring
    if not review_df.empty:
        review_df['Review_Priority'] = review_df.apply(
            lambda row: calculate_review_priority_score(row, class_names_map), axis=1)
        review_df = review_df.sort_values('Review_Priority', ascending=False)
    else:
        logger.warning("Review DataFrame is empty before priority scoring. No recommendations generated.")

    return review_df


def calculate_review_priority_score(row, class_names_map):
    """Calculates a priority score for reviewing a sample based on its properties."""
    score = 0.0

    # Misclassification (Primary factor)
    # 'Is_Correct' should be boolean (True/False). If it's 0/1, adjust check.
    # Assuming 'Is_Correct' is False for misclassified.
    if 'Is_Correct' in row and pd.notna(row['Is_Correct']) and not row['Is_Correct']:
        score += 30.0

        # Critical Error (e.g., None <-> Complete)
        # Needs 'Expert_Label' and 'Predicted_Label' (numeric) and class_names_map
        if 'Expert_Label' in row and 'Predicted_Label' in row and \
                pd.notna(row['Expert_Label']) and pd.notna(row['Predicted_Label']):
            try:
                expert_num = int(row['Expert_Label'])
                pred_num = int(row['Predicted_Label'])
                none_val = [k for k, v in class_names_map.items() if v.lower() == 'none']
                complete_val = [k for k, v in class_names_map.items() if v.lower() == 'complete']

                if none_val and complete_val:
                    none_num = int(none_val[0])
                    complete_num = int(complete_val[0])
                    if (expert_num == none_num and pred_num == complete_num) or \
                            (expert_num == complete_num and pred_num == none_num):
                        score += 50.0  # High penalty for critical errors
            except ValueError:
                pass  # Cannot determine critical error if labels are not numeric

    # High Uncertainty indicators
    if 'Entropy' in row and pd.notna(row['Entropy']):
        score += row['Entropy'] * 10.0  # Higher entropy = more uncertainty

    if 'Margin' in row and pd.notna(row['Margin']):
        score += (1.0 - row['Margin']) * 10.0  # Smaller margin = more uncertainty (margin is 0 to 1)

    # Low confidence in the true label (if known and probabilities are available)
    if 'Prob_True_Label' in row and pd.notna(row['Prob_True_Label']) and row['Prob_True_Label'] < 0.5:
        score += (0.5 - row['Prob_True_Label']) * 40.0  # Penalize more if prob of true label is very low

    # Influence Score (Delta_F1 or other metric)
    # A negative influence score means removing the point *improved* the model, so it's highly influential (bad way)
    # A large positive influence score means removing it *hurt* the model, so it was a helpful point.
    # We typically want to review points whose removal improves the model (negative delta_F1).
    if 'Influence_Score (Delta_F1)' in row and pd.notna(row['Influence_Score (Delta_F1)']):
        influence = row['Influence_Score (Delta_F1)']
        if influence < 0:  # Point was detrimental
            score += abs(influence) * 100.0  # Strongly prioritize detrimental points
        # Optional: Slightly prioritize helpful but borderline points if influence is positive but small
        # elif influence > 0 and influence < 0.01: # Small positive influence
        # score += influence * 10.0

    # Check for 'Flag_Reason' from generate_review_candidates (if merged)
    if 'Flag_Reason' in row and pd.notna(row['Flag_Reason']) and isinstance(row['Flag_Reason'], str):
        if 'CRITICAL' in row['Flag_Reason'].upper():  # If a critical flag was set explicitly
            score += 40.0  # Add to already high score if critical
        if 'LOW_MARGIN' in row['Flag_Reason'].upper() or 'HIGH_ENTROPY' in row['Flag_Reason'].upper():
            score += 5.0

    return score


def find_similar_patients(patient_id, side, features_df, metadata_df, n_similar=10):
    """Finds patients with similar feature vectors to a target patient."""
    from sklearn.metrics.pairwise import cosine_similarity  # Local import

    if not all(col in metadata_df.columns for col in ['Patient ID', 'Side']):
        logger.warning("find_similar_patients: Metadata missing 'Patient ID' or 'Side'. Cannot find target.")
        return pd.DataFrame()

    # Find the index of the target patient in the metadata
    target_row_mask = (metadata_df['Patient ID'] == str(patient_id)) & (metadata_df['Side'] == str(side))
    if not target_row_mask.any():
        logger.warning(f"find_similar_patients: Target patient ID '{patient_id}' Side '{side}' not found in metadata.")
        return pd.DataFrame()

    target_idx = metadata_df[target_row_mask].index[0]  # Get the actual index value

    # Ensure target_idx is valid for features_df (which should share the same index if prepared together)
    if target_idx not in features_df.index:
        logger.warning(
            f"find_similar_patients: Target index {target_idx} (from metadata) not found in features_df.index.")
        return pd.DataFrame()

    target_features = features_df.loc[[target_idx]]  # Use .loc for label-based indexing

    # Calculate cosine similarity between the target and all other samples
    # Ensure features_df does not contain the target itself for similarity comparison if desired,
    # or handle it by taking [1:n_similar+1] later.
    similarities = cosine_similarity(target_features, features_df)[0]  # Get the array of similarities

    # Get indices of top N similar patients, excluding the target itself (which has similarity 1.0)
    # Argsort returns indices that would sort the array. [::-1] reverses for descending.
    sorted_indices_by_similarity = np.argsort(similarities)[::-1]

    # Remove target_idx from sorted_indices if present (it should be the first one)
    similar_indices = [idx for idx in sorted_indices_by_similarity if features_df.index[idx] != target_idx][:n_similar]

    similar_patients_data = []
    for sim_original_idx_pos in similar_indices:
        sim_actual_idx_label = features_df.index[sim_original_idx_pos]  # Get the original index label
        if sim_actual_idx_label in metadata_df.index:  # Check if this index is also in metadata
            similar_patients_data.append({
                'Patient ID': metadata_df.loc[sim_actual_idx_label, 'Patient ID'],
                'Side': metadata_df.loc[sim_actual_idx_label, 'Side'],
                'Similarity': similarities[sim_original_idx_pos]  # Similarity score
            })
        else:
            logger.warning(
                f"find_similar_patients: Index {sim_actual_idx_label} from features_df not in metadata_df for similar patient.")

    return pd.DataFrame(similar_patients_data)


def create_review_context(patient_id, side, all_data_dict):
    """
    Creates a context dictionary for a patient, summarizing their data across different zones.
    'all_data_dict' is expected to be a dictionary like: {'lower': df_lower, 'mid': df_mid, ...}
    where each df contains 'Patient ID', 'Side', 'Expert_Label_Name', 'Predicted_Label_Name', etc.
    """
    context = {'patient_id': patient_id, 'side': side, 'zones': {}}

    for zone_key, zone_data_df in all_data_dict.items():
        if isinstance(zone_data_df, pd.DataFrame):
            if 'Patient ID' in zone_data_df.columns and 'Side' in zone_data_df.columns:
                mask = (zone_data_df['Patient ID'] == str(patient_id)) & (zone_data_df['Side'] == str(side))
                if mask.any():
                    patient_zone_data = zone_data_df[mask].iloc[0]  # Get the first match
                    context['zones'][zone_key] = {
                        'expert_label': patient_zone_data.get('Expert_Label_Name', 'N/A'),
                        'model_prediction': patient_zone_data.get('Predicted_Label_Name', 'N/A'),
                        'confidence': patient_zone_data.get('Model_Confidence', np.nan),  # Or a specific prob column
                        'is_correct': patient_zone_data.get('Is_Correct', None)  # Boolean or None
                        # Add other relevant fields from patient_zone_data
                    }
                else:
                    context['zones'][zone_key] = {'status': 'Data not found for this zone'}
            else:
                context['zones'][zone_key] = {'status': 'Patient ID or Side column missing in zone data'}
        else:
            context['zones'][zone_key] = {'status': 'Zone data not a DataFrame'}

    return context


def calculate_patient_similarity_matrix(features_df, batch_size=1000):
    """Calculates a full patient-patient similarity matrix, batched for memory efficiency."""
    from sklearn.metrics.pairwise import cosine_similarity  # Local import

    n_samples = features_df.shape[0]
    if n_samples == 0: return np.array([[]])  # Empty matrix for empty input

    similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)  # Use float32 to save memory

    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        batch_i_data = features_df.iloc[i:end_i]

        for j in range(0, n_samples, batch_size):  # Can optimize by only doing j >= i for symmetric matrix
            end_j = min(j + batch_size, n_samples)
            batch_j_data = features_df.iloc[j:end_j]

            sim_block = cosine_similarity(batch_i_data, batch_j_data)
            similarity_matrix[i:end_i, j:end_j] = sim_block
            # If optimizing for symmetry: if i != j: similarity_matrix[j:end_j, i:end_i] = sim_block.T

    return similarity_matrix


def export_error_analysis_details(errors_df, output_path, class_names_map):
    """Exports detailed error analysis to a CSV file, adding severity."""
    if errors_df is None or errors_df.empty:
        logger.info("export_error_analysis_details: No errors to export.")
        return

    # Ensure 'Expert_Label_Name' and 'Predicted_Label_Name' exist or create them
    if 'Expert_Label' in errors_df.columns and 'Expert_Label_Name' not in errors_df.columns:
        errors_df['Expert_Label_Name'] = errors_df['Expert_Label'].map(class_names_map).fillna('Unknown')
    if 'Predicted_Label' in errors_df.columns and 'Predicted_Label_Name' not in errors_df.columns:
        errors_df['Predicted_Label_Name'] = errors_df['Predicted_Label'].map(class_names_map).fillna('Unknown')

    # Calculate Model_Confidence if not present (e.g., max probability)
    if 'Model_Confidence' not in errors_df.columns:
        prob_cols = [col for col in errors_df.columns if col.startswith('Prob_') and col != 'Prob_True_Label']
        if prob_cols:
            errors_df['Model_Confidence'] = errors_df[prob_cols].max(axis=1)
        else:  # Fallback if no individual probability columns
            errors_df['Model_Confidence'] = np.nan

            # Determine error severity

    def get_error_severity(row):
        expert_name = row.get('Expert_Label_Name', '').lower()
        pred_name = row.get('Predicted_Label_Name', '').lower()
        if (expert_name == 'none' and pred_name == 'complete') or \
                (expert_name == 'complete' and pred_name == 'none'):
            return 'Critical'
        elif (expert_name == 'partial' and pred_name != 'partial') or \
                (pred_name == 'partial' and expert_name != 'partial'):
            return 'Partial-Related'
        return 'Standard'

    errors_df['Error_Severity'] = errors_df.apply(get_error_severity, axis=1)

    # Sort by severity (Critical first), then by model confidence (higher confidence errors might be more concerning)
    errors_df = errors_df.sort_values(by=['Error_Severity', 'Model_Confidence'], ascending=[True, False])

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        errors_df.to_csv(output_path, index=False, float_format='%.4f')
        logger.info(f"Detailed error analysis exported to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export detailed error analysis to {output_path}: {e}", exc_info=True)


def generate_review_candidates(zone, model, scaler, feature_names, config,
                               X_train_orig, y_train_orig,  # Original training data
                               X_test_orig, y_test_orig,
                               # Original test data (for context, not usually for influence on train)
                               metadata_train,  # Metadata corresponding to X_train_orig
                               training_analysis_df,  # DataFrame from train_enhanced_model (uncertainty, etc.)
                               class_names_map,
                               decision_threshold=0.5,  # For binary classification if applicable
                               influence_scores_manual=None,  # Pre-calculated influence scores {index: score}
                               top_k_influence=50,  # For flagging based on influence rank (not directly used here)
                               entropy_quantile=0.9,
                               margin_quantile=0.1,
                               true_label_prob_threshold=0.4):
    """
    Generates a DataFrame of training samples recommended for review based on various criteria.
    """
    is_binary = len(class_names_map) <= 2
    zone_name_rev = config.get('name', zone.capitalize().replace('_', ' ') + ' Face')

    if training_analysis_df is None or training_analysis_df.empty:
        logger.error(f"[{zone_name_rev}] Review Candidates: training_analysis_df is missing or empty. Cannot generate.")
        return pd.DataFrame()  # Return empty DataFrame

    # Ensure metadata_train is a DataFrame and aligns with training_analysis_df
    if metadata_train is None or not isinstance(metadata_train, pd.DataFrame):
        logger.warning(
            f"[{zone_name_rev}] Review Candidates: metadata_train is None or not a DataFrame. Creating dummy metadata based on training_analysis_df index.")
        metadata_train = pd.DataFrame(index=training_analysis_df.index)
    if 'Patient ID' not in metadata_train.columns:
        metadata_train['Patient ID'] = "UnknownID_Rev_" + metadata_train.index.astype(str)
    if 'Side' not in metadata_train.columns:
        metadata_train['Side'] = "UnknownSide_Rev"

    # Align indices carefully. training_analysis_df is the primary source of model-derived metrics.
    # X_train_orig and metadata_train should align with training_analysis_df's index.
    current_analysis_df_rev = training_analysis_df.copy()

    # If X_train_orig's index doesn't match, it implies training_analysis_df might be from a SMOTEd set
    # or there's a mismatch. For review candidates based on original samples, indices must match.
    if not X_train_orig.index.equals(current_analysis_df_rev.index):
        logger.warning(
            f"[{zone_name_rev}] Review Candidates: Index mismatch between X_train_orig and training_analysis_df. Reindexing training_analysis_df to match X_train_orig. This assumes training_analysis_df was generated on X_train_orig.")
        current_analysis_df_rev = current_analysis_df_rev.reindex(X_train_orig.index)
        # If reindexing introduces all NaNs for a key column, it's problematic
        if 'Expert_Label' in current_analysis_df_rev and current_analysis_df_rev['Expert_Label'].isnull().all():
            logger.error(
                f"[{zone_name_rev}] Review Candidates: Reindexing training_analysis_df resulted in all NaNs for 'Expert_Label'. Aborting candidate generation.")
            return pd.DataFrame()
        # Drop rows where essential data became NaN after reindex (if training_analysis_df was smaller)
        current_analysis_df_rev.dropna(subset=['Expert_Label'], inplace=True)  # Or other key columns like probabilities

    # Align metadata_train to the (potentially reindexed and cleaned) current_analysis_df_rev
    if not current_analysis_df_rev.index.equals(metadata_train.index):
        logger.warning(
            f"[{zone_name_rev}] Review Candidates: Index mismatch between (reindexed)analysis_df and metadata_train. Reindexing metadata_train.")
        metadata_train = metadata_train.reindex(current_analysis_df_rev.index)
        if 'Patient ID' in metadata_train.columns and metadata_train['Patient ID'].isnull().any():
            metadata_train['Patient ID'].fillna(f"UnknownID_Reidx_{zone_name_rev}", inplace=True)
        if 'Side' in metadata_train.columns and metadata_train['Side'].isnull().any():
            metadata_train['Side'].fillna("UnknownSide_Reidx", inplace=True)

    # If current_analysis_df_rev became empty afterdropna, return empty
    if current_analysis_df_rev.empty:
        logger.warning(
            f"[{zone_name_rev}] Review Candidates: training_analysis_df is empty after reindexing/cleaning. No candidates.")
        return pd.DataFrame()

    # If binary, ensure 'Predicted_Label' and 'Is_Correct' are correctly derived if not already present
    if is_binary:
        positive_class_label_name = class_names_map.get(1)  # Assuming 1 is positive
        if positive_class_label_name:
            prob_col_positive_class = f"Prob_{positive_class_label_name.replace(' ', '_')}"
            if prob_col_positive_class in current_analysis_df_rev.columns:
                if 'Predicted_Label' not in current_analysis_df_rev.columns:  # Create if missing
                    current_analysis_df_rev['Predicted_Label'] = (
                                current_analysis_df_rev[prob_col_positive_class] >= decision_threshold).astype(int)
                if 'Is_Correct' not in current_analysis_df_rev.columns and 'Expert_Label' in current_analysis_df_rev.columns:  # Create if missing
                    current_analysis_df_rev['Is_Correct'] = (
                                current_analysis_df_rev['Expert_Label'] == current_analysis_df_rev['Predicted_Label'])
            else:
                logger.warning(
                    f"[{zone_name_rev}] Binary case: Prob column '{prob_col_positive_class}' not found for deriving Predicted_Label/Is_Correct.")

    try:
        # Join metadata with analysis data. Use 'inner' to ensure only samples present in both are kept.
        # This assumes metadata_train has been successfully reindexed to current_analysis_df_rev.index
        candidates_df_rev = metadata_train.join(current_analysis_df_rev, how='inner')
        if candidates_df_rev.empty:
            logger.warning(
                f"[{zone_name_rev}] Review Candidates: No candidates after joining metadata with analysis data. Check index alignment and content.")
            return pd.DataFrame()

        candidates_df_rev['Flag_Reason'] = ''  # Initialize empty flag reason string

        # Flag Misclassified
        if 'Is_Correct' in candidates_df_rev.columns:
            misclassified_mask = ~candidates_df_rev['Is_Correct'].fillna(
                True)  # Treat NaN Is_Correct as True (not misclassified)
            candidates_df_rev.loc[misclassified_mask, 'Flag_Reason'] += 'Misclassified; '

            # Flag Critical Errors (only if misclassified and multi-class)
            if not is_binary and 0 in class_names_map and 2 in class_names_map:  # Check for None and Complete classes
                if 'Expert_Label' in candidates_df_rev.columns and 'Predicted_Label' in candidates_df_rev.columns:
                    # Ensure labels are numeric for comparison
                    expert_label_num = pd.to_numeric(candidates_df_rev['Expert_Label'], errors='coerce')
                    predicted_label_num = pd.to_numeric(candidates_df_rev['Predicted_Label'], errors='coerce')

                    critical_mask = ((expert_label_num == 0) & (predicted_label_num == 2)) | \
                                    ((expert_label_num == 2) & (predicted_label_num == 0))

                    # Apply critical flag only to rows that are already misclassified
                    # Ensure misclassified_mask is aligned with critical_mask (should be if derived from same df)
                    if misclassified_mask.index.equals(critical_mask.index):
                        candidates_df_rev.loc[
                            misclassified_mask & critical_mask.fillna(False), 'Flag_Reason'] += 'CRITICAL_Error; '
                    else:  # Should not happen if masks are from same df
                        logger.warning(
                            f"[{zone_name_rev}] Index mismatch for critical error flagging. This is unexpected.")

        # Flag High Entropy
        entropy_threshold_val = np.inf  # Default to not flagging
        if 'Entropy' in candidates_df_rev.columns and pd.api.types.is_numeric_dtype(candidates_df_rev['Entropy']) and \
                candidates_df_rev['Entropy'].notna().any():
            try:
                entropy_threshold_val = candidates_df_rev['Entropy'].quantile(entropy_quantile)
                candidates_df_rev.loc[candidates_df_rev[
                                          'Entropy'] >= entropy_threshold_val, 'Flag_Reason'] += f'High_Entropy(>{entropy_threshold_val:.2f}); '
            except Exception as e_entropy:
                logger.warning(f"[{zone_name_rev}] Could not calculate entropy threshold: {e_entropy}")

        # Flag Low Margin
        margin_threshold_val = -np.inf  # Default to not flagging
        if 'Margin' in candidates_df_rev.columns and pd.api.types.is_numeric_dtype(candidates_df_rev['Margin']) and \
                candidates_df_rev['Margin'].notna().any():
            try:
                margin_threshold_val = candidates_df_rev['Margin'].quantile(margin_quantile)
                candidates_df_rev.loc[candidates_df_rev[
                                          'Margin'] <= margin_threshold_val, 'Flag_Reason'] += f'Low_Margin(<{margin_threshold_val:.2f}); '
            except Exception as e_margin:
                logger.warning(f"[{zone_name_rev}] Could not calculate margin threshold: {e_margin}")

        # Flag Low Probability of True Label
        if 'Prob_True_Label' in candidates_df_rev.columns and pd.api.types.is_numeric_dtype(
                candidates_df_rev['Prob_True_Label']):
            candidates_df_rev.loc[candidates_df_rev[
                                      'Prob_True_Label'] < true_label_prob_threshold, 'Flag_Reason'] += f'Low_Prob_True(<{true_label_prob_threshold:.1f}); '

        # Add Influence Scores (if provided)
        candidates_df_rev['Influence_Score (Delta_F1)'] = np.nan  # Initialize column
        if influence_scores_manual is not None and isinstance(influence_scores_manual, dict):
            # Map scores using index. Ensure candidates_df_rev.index matches keys in influence_scores_manual
            candidates_df_rev['Influence_Score (Delta_F1)'] = candidates_df_rev.index.map(
                influence_scores_manual).fillna(np.nan)
            influence_calculated_mask = candidates_df_rev['Influence_Score (Delta_F1)'].notna()
            if influence_calculated_mask.any():
                # Append influence reason only if score is calculated
                # Apply to add '; Influence(...)' to existing reasons or set it if reason is empty
                def format_influence_reason(row):
                    reason = row['Flag_Reason']
                    influence_val = row['Influence_Score (Delta_F1)']
                    influence_str = f"Influence({influence_val:.3f})"
                    return (reason + '; ' + influence_str) if reason else influence_str

                candidates_df_rev.loc[influence_calculated_mask, 'Flag_Reason'] = candidates_df_rev[
                    influence_calculated_mask].apply(format_influence_reason, axis=1)

        # Clean up Flag_Reason string
        candidates_df_rev['Flag_Reason'] = candidates_df_rev['Flag_Reason'].str.strip().str.rstrip(';')

        # Filter to only include candidates that have at least one flag
        flagged_candidates_interim = candidates_df_rev[candidates_df_rev['Flag_Reason'] != ''].copy()

        if flagged_candidates_interim.empty:
            logger.info(f"[{zone_name_rev}] No candidates flagged for review based on criteria.")
            return pd.DataFrame()

        # Calculate Priority Score (using the separate helper for clarity)
        flagged_candidates_interim['PriorityScore'] = flagged_candidates_interim.apply(
            lambda row: calculate_review_priority_score(row, class_names_map), axis=1
        )

        # Add readable label names
        if 'Expert_Label' in flagged_candidates_interim:
            flagged_candidates_interim['Expert_Label_Name'] = flagged_candidates_interim['Expert_Label'].map(
                class_names_map).fillna('Unknown')
        if 'Predicted_Label' in flagged_candidates_interim:
            flagged_candidates_interim['Predicted_Label_Name'] = flagged_candidates_interim['Predicted_Label'].map(
                class_names_map).fillna('Unknown')

        # Select and order output columns
        prob_cols_to_keep = [col for col in flagged_candidates_interim.columns if col.startswith("Prob_")]

        output_columns = ['Patient ID', 'Side']
        # Add common analysis columns if they exist
        for col in ['Expert_Label_Name', 'Predicted_Label_Name', 'Is_Correct', 'Expert_Label', 'Predicted_Label']:
            if col in flagged_candidates_interim.columns: output_columns.append(col)
        output_columns.extend(sorted(prob_cols_to_keep))  # Add all probability columns, sorted
        # Add uncertainty and influence metrics
        for col in ['Entropy', 'Margin', 'Influence_Score (Delta_F1)', 'Flag_Reason', 'PriorityScore']:
            if col in flagged_candidates_interim.columns: output_columns.append(col)

        # Ensure only existing columns are selected
        final_columns_present = [col for col in output_columns if col in flagged_candidates_interim.columns]
        final_candidates_df_rev = flagged_candidates_interim[final_columns_present].copy()

        # Convert key identifiers to string for consistent output
        for col in ['Patient ID', 'Side']:
            if col in final_candidates_df_rev: final_candidates_df_rev[col] = final_candidates_df_rev[col].astype(str)

        # Sort by PriorityScore (descending), then by Patient ID and Side for consistency
        sort_by_cols = []
        ascending_order = []
        if 'PriorityScore' in final_candidates_df_rev:
            sort_by_cols.append('PriorityScore');
            ascending_order.append(False)  # Descending
        if 'Patient ID' in final_candidates_df_rev:
            sort_by_cols.append('Patient ID');
            ascending_order.append(True)
        if 'Side' in final_candidates_df_rev:
            sort_by_cols.append('Side');
            ascending_order.append(True)

        if sort_by_cols:
            final_candidates_df_rev = final_candidates_df_rev.sort_values(by=sort_by_cols, ascending=ascending_order)

        return final_candidates_df_rev

    except Exception as e_rev:
        logger.error(f"Error generating review candidates for {zone_name_rev}: {e_rev}", exc_info=True)
        return pd.DataFrame()  # Return empty on error