# paralysis_utils.py

import importlib
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_fscore_support, f1_score,
    roc_auc_score, balanced_accuracy_score, make_scorer
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold  # KFold kept for potential use
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # SMOTE kept for analyze_training_influence example
import joblib  # Kept for potential use in future util functions
import xgboost as xgb  # xgb kept for analyze_training_influence example
import scipy.stats  # For calculate_entropy
from copy import deepcopy
from sklearn.calibration import CalibratedClassifierCV  # Kept as it's part of the broader ecosystem

logger = logging.getLogger(__name__)

try:
    # Attempt to import from paralysis_config first
    from paralysis_config import CLASS_NAMES as PARALYSIS_CLASS_NAMES_CONFIG, \
        INPUT_FILES as PARALYSIS_INPUT_FILES_CONFIG, \
        ZONE_CONFIG as PARALYSIS_ZONE_CONFIG_CONFIG, \
        REVIEW_CONFIG  # Added REVIEW_CONFIG import for generate_review_candidates

    PARALYSIS_MAP = PARALYSIS_CLASS_NAMES_CONFIG
except ImportError:
    # Fallback if paralysis_config is not found or CLASS_NAMES is not defined
    PARALYSIS_MAP = {0: 'None', 1: 'Partial', 2: 'Complete'}
    PARALYSIS_INPUT_FILES_CONFIG = {}
    PARALYSIS_ZONE_CONFIG_CONFIG = {}
    REVIEW_CONFIG = {}  # Default empty REVIEW_CONFIG
    logger.info(
        "paralysis_config.py not found or required configs missing. Using default PARALYSIS_MAP and empty configs.")

# Synkinesis detection code removed - paralysis detection only

from paralysis_training_helpers import calculate_class_weights_for_model

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

    mask_max_pos = max_vals > min_value
    mask_min_zero_but_max_pos = (min_vals <= min_value) & mask_max_pos
    epsilon = 1e-9

    valid_division_mask = mask_max_pos & (np.abs(max_vals) > epsilon)  # Check absolute value of max_vals for division
    ratio.loc[valid_division_mask] = min_vals.loc[valid_division_mask] / (
    max_vals.loc[valid_division_mask])  # No need to add epsilon if check is > epsilon

    # If min is zero/small and max is significant, ratio is 0
    ratio.loc[mask_min_zero_but_max_pos] = 0.0

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

    percent_diff = pd.Series(0.0, index=s1.index, dtype=float)

    mask_avg_pos = avg.abs() > min_value
    mask_diff_pos = abs_diff > min_value  # Check if difference itself is significant
    epsilon = 1e-9

    valid_division_mask = mask_avg_pos & (np.abs(avg) > epsilon)
    percent_diff.loc[valid_division_mask] = (abs_diff.loc[valid_division_mask] / (
        avg.loc[valid_division_mask].abs())) * 100.0

    percent_diff.loc[~mask_avg_pos & mask_diff_pos] = cap  # If avg is near zero but diff is not, max it out

    percent_diff = percent_diff.fillna(0.0).clip(0, cap)
    return percent_diff

def standardize_paralysis_labels(val):
    label_map = PARALYSIS_MAP
    if val is None or pd.isna(val): return 'NA'
    val_str = str(val).strip().lower()
    if val_str == '' or val_str == 'not assessed': return 'NA'

    # Attempt direct mapping first using string versions of keys for flexibility
    for key_num, label_name in label_map.items():
        if str(key_num) == val_str: return label_name  # e.g., if val_str is "0", "1", "2"

    # Check common string synonyms against mapped values
    if val_str in ['none', 'no', '0', '0.0', 'normal']: return label_map.get(0, 'None')  # Fallback if 0 not in map
    if val_str in ['partial', 'mild', 'moderate', 'incomplete', 'i', 'p', '1', '1.0']: return label_map.get(1,
                                                                                                            'Partial')
    if val_str in ['complete', 'severe', 'c', '2', '2.0']: return label_map.get(2, 'Complete')

    if val_str == 'error': return 'Error'
    return 'NA'


# standardize_synkinesis_labels function removed - paralysis detection only

def process_binary_target(target_series):
    if target_series is None: return np.array([], dtype=int)
    if not isinstance(target_series, pd.Series): target_series = pd.Series(target_series)
    if target_series.empty: return np.array([], dtype=int)

    s_clean = target_series.astype(str).str.lower().str.strip().replace({
        'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no', 'not assessed': 'no',
        'normal': 'no', 'f': 'false', 'n': 'no', 'false': 'no', '0': 'no', '0.0': 'no',
        'yes': 'yes', 'true': 'yes', 'y': 'yes', 't': 'yes', '1': 'yes', '1.0': 'yes',
        'partial': 'yes', 'mild': 'yes', 'moderate': 'yes', 'incomplete': 'yes', 'i': 'yes', 'p': 'yes',
        'complete': 'yes', 'severe': 'yes', 'c': 'yes', '2': 'yes', '2.0': 'yes'
    })
    mapping = {'yes': 1, 'no': 0}
    mapped = s_clean.map(mapping)
    final_mapped = mapped.fillna(0)  # Default unmapped to 0 (no)
    return final_mapped.astype(int).values

def hybrid_feature_selection(features_df, targets, config_fs, n_features_to_select):
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier  # For importance based selection

    logger.info(
        f"Starting hybrid feature selection. Initial features: {features_df.shape[1]}, Target N: {n_features_to_select}")

    # Step 1: Variance threshold (remove zero or low variance features)
    var_thresh_val = config_fs.get('variance_threshold', 0.01)
    selector_var = VarianceThreshold(threshold=var_thresh_val)
    try:
        features_var = selector_var.fit_transform(features_df)
        features_var_df = pd.DataFrame(features_var, columns=features_df.columns[selector_var.get_support()],
                                       index=features_df.index)
        logger.info(f"After VarianceThreshold ({var_thresh_val}): {features_var_df.shape[1]} features remaining.")
    except ValueError as e_var:  # If all features are removed or other issues
        logger.warning(
            f"VarianceThreshold failed ({e_var}). Skipping this step, using all features from previous step.")
        features_var_df = features_df.copy()

    # Step 2: Univariate feature selection (e.g., ANOVA F-value for classification)
    # Select more features than finally needed, to give RF a good pool
    k_best = min(n_features_to_select * 3, features_var_df.shape[1])
    if k_best > 0 and features_var_df.shape[1] > 0:  # Check if there are features to select from
        selector_f = SelectKBest(f_classif, k=k_best)
        try:
            features_f_transformed = selector_f.fit_transform(features_var_df, targets)
            features_f_df = pd.DataFrame(features_f_transformed,
                                         columns=features_var_df.columns[selector_f.get_support()],
                                         index=features_var_df.index)
            logger.info(f"After SelectKBest (k={k_best}): {features_f_df.shape[1]} features remaining.")
        except Exception as e_kbest:  # Catch errors if k_best is invalid for current data
            logger.warning(
                f"SelectKBest failed ({e_kbest}). Skipping this step, using features from VarianceThreshold.")
            features_f_df = features_var_df.copy()
    else:
        logger.info("Skipping SelectKBest as no features or k_best is 0.")
        features_f_df = features_var_df.copy()

    # Step 3: Tree-based importance (Random Forest)
    if features_f_df.shape[1] > 0 and features_f_df.shape[1] > n_features_to_select:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(features_f_df, targets)
        importances = pd.Series(rf.feature_importances_, index=features_f_df.columns).sort_values(ascending=False)
        top_features_rf = importances.head(n_features_to_select).index.tolist()
        logger.info(f"After RF importance: Selected {len(top_features_rf)} features.")
        final_selected_features_df = features_df[top_features_rf]  # Select from original DF to preserve original values
    elif features_f_df.shape[1] > 0:  # If fewer features than n_features_to_select remain, take all of them
        logger.info(
            f"Fewer features ({features_f_df.shape[1]}) than target ({n_features_to_select}) after KBest. Taking all remaining.")
        final_selected_features_df = features_df[features_f_df.columns.tolist()]
    else:  # No features left
        logger.warning("No features remaining after filtering steps. Returning empty DataFrame.")
        return pd.DataFrame(index=features_df.index)

    return final_selected_features_df

def _extract_base_au_features(df_input, side, actions_list, aus_list, feature_extraction_config,
                              zone_display_name="Zone"):
    feature_data_dict = {}
    opposite_side_str = 'Right' if side == 'Left' else 'Left'

    use_normalized_val = feature_extraction_config.get('use_normalized', True)
    min_value_param = feature_extraction_config.get('min_value', 0.0001)
    percent_diff_cap_val = feature_extraction_config.get('percent_diff_cap', 200.0)

    for action_str in actions_list:
        for au_str in aus_list:
            base_col_name_str = f"{action_str}_{au_str}"

            au_col_current_side = f"{action_str}_{side} {au_str}"
            au_norm_col_current_side = f"{au_col_current_side} (Normalized)"
            au_col_opposite_side = f"{action_str}_{opposite_side_str} {au_str}"
            au_norm_col_opposite_side = f"{au_col_opposite_side} (Normalized)"

            raw_val_current_side_series = df_input.get(au_col_current_side, pd.Series(0.0, index=df_input.index))
            raw_val_opposite_side_series = df_input.get(au_col_opposite_side, pd.Series(0.0, index=df_input.index))

            raw_val_current_side = pd.to_numeric(raw_val_current_side_series, errors='coerce').fillna(0.0)
            raw_val_opposite_side = pd.to_numeric(raw_val_opposite_side_series, errors='coerce').fillna(0.0)

            val_current_side_to_use = raw_val_current_side
            val_opposite_side_to_use = raw_val_opposite_side

            if use_normalized_val:
                norm_val_current_side_series = df_input.get(au_norm_col_current_side, raw_val_current_side_series)
                norm_val_opposite_side_series = df_input.get(au_norm_col_opposite_side, raw_val_opposite_side_series)

                val_current_side_to_use_temp = pd.to_numeric(norm_val_current_side_series, errors='coerce')
                val_current_side_to_use = val_current_side_to_use_temp.fillna(raw_val_current_side)

                val_opposite_side_to_use_temp = pd.to_numeric(norm_val_opposite_side_series, errors='coerce')
                val_opposite_side_to_use = val_opposite_side_to_use_temp.fillna(raw_val_opposite_side)

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

def prepare_data_generalized(zone_key, results_file_path=None, expert_file_path=None, base_config_dict=None,
                             input_files_global_dict=None, class_names_global_dict=None):
    if base_config_dict is None:
        from paralysis_config import ZONE_CONFIG as base_config_imported
        base_config_dict = base_config_imported
    if input_files_global_dict is None:
        from paralysis_config import INPUT_FILES as input_files_imported
        input_files_global_dict = input_files_imported
    if class_names_global_dict is None:
        class_names_global_dict = PARALYSIS_MAP

    try:
        config_zone_specific = base_config_dict[zone_key]
        zone_name_display = config_zone_specific.get('name', zone_key.capitalize() + ' Face')
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
                                na_values=['', 'NA', 'N/A', 'Not Assessed', 'not assessed'])
    except Exception as e:
        logger.error(f"[{zone_name_display}] Error loading data: {e}.", exc_info=True)
        return None, None, None

    expert_rename_map = {'Patient': 'Patient ID'}
    if 'Patient ID' in expert_df.columns and 'Patient' in expert_df.columns and 'Patient ID' != 'Patient':
        if 'Patient' in expert_rename_map: del expert_rename_map['Patient']

    expert_left_orig_name = expert_cols_zone.get('left')
    expert_right_orig_name = expert_cols_zone.get('right')

    if not all([expert_left_orig_name, expert_right_orig_name]):
        logger.error(
            f"[{zone_name_display}] Missing 'left' or 'right' expert column names in config for zone '{zone_key}'.")
        return None, None, None
    for col_name in [expert_left_orig_name, expert_right_orig_name]:
        if col_name not in expert_df.columns:
            logger.error(
                f"[{zone_name_display}] Expert column '{col_name}' (from config) not found in expert file. Columns: {expert_df.columns.tolist()}.")
            return None, None, None
        expert_rename_map[col_name] = col_name  # Ensure these columns are part of rename map

    expert_df_renamed = expert_df.rename(columns=expert_rename_map)
    patient_id_col_final = 'Patient ID' if 'Patient ID' in expert_df_renamed.columns else None
    if not patient_id_col_final:
        logger.error(
            f"[{zone_name_display}] 'Patient ID' column not found in expert_df_renamed after attempting rename.")
        return None, None, None

    expert_cols_to_merge_list = [patient_id_col_final, expert_left_orig_name, expert_right_orig_name]
    expert_df_subset_final = expert_df_renamed[expert_cols_to_merge_list].copy()

    if expert_df_subset_final[patient_id_col_final].duplicated().any():
        logger.warning(
            f"[{zone_name_display}] Duplicate Patient IDs found in expert file. Keeping first instance for each ID.")
        expert_df_subset_final.drop_duplicates(subset=[patient_id_col_final], keep='first', inplace=True)

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

    merged_df['Expert_Std_Left'] = merged_df[expert_left_orig_name].apply(standardize_paralysis_labels)
    merged_df['Expert_Std_Right'] = merged_df[expert_right_orig_name].apply(standardize_paralysis_labels)
    valid_left_mask_final = merged_df['Expert_Std_Left'] != 'NA'
    valid_right_mask_final = merged_df['Expert_Std_Right'] != 'NA'

    feature_module_name = f"{zone_key}_face_features"  # Assumes files like 'lower_face_features.py'
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

    filtered_left_features = left_features_df[valid_left_mask_final].copy()
    filtered_right_features = right_features_df[valid_right_mask_final].copy()

    metadata_left = merged_df.loc[valid_left_mask_final, [patient_id_col_final]].copy()
    metadata_left.rename(columns={patient_id_col_final: 'Patient ID'}, inplace=True)
    metadata_left['Side'] = 'Left'
    metadata_right = merged_df.loc[valid_right_mask_final, [patient_id_col_final]].copy()
    metadata_right.rename(columns={patient_id_col_final: 'Patient ID'}, inplace=True)
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
                unmapped_labels = pd.unique(unmapped_series)  # Get unique unmapped original values
                logger.warning(
                    f"[{current_zone_name_log}] Standardized labels {list(unmapped_labels)} (from original: {list(series_to_map[mapped.isna()].unique())}) not found in target_mapping_final ({target_mapping_final}). Defaulting these to 0 (None).")
            mapped = mapped.fillna(0)  # Default NA after mapping to 0 (None class)
        return mapped.astype(int)

    left_targets = standardize_and_map_valid(valid_left_expert_labels_raw, zone_name_display)
    right_targets = standardize_and_map_valid(valid_right_expert_labels_raw, zone_name_display)

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
    features_combined = features_combined.fillna(0)  # Fill any NaNs created by inf replacement or earlier steps

    # Feature selection based on importance file is NOT done here anymore.
    # It's handled by train_model_workflow after this function returns.
    # This function now returns all extracted features.
    logger.info(
        f"[{zone_name_display}] prepare_data_generalized returning {features_combined.shape[1]} initial features.")

    # Saving the initial full feature list (optional, for debugging)
    # The main workflow will save the *selected* feature list.
    initial_feature_names_list_save = features_combined.columns.tolist()
    feature_list_path_from_config = filenames_zone.get('feature_list')
    if feature_list_path_from_config:
        try:
            # Create a different name for the initial full list
            initial_list_filename = os.path.basename(feature_list_path_from_config).replace(".list",
                                                                                            "_initial_full.list")
            initial_list_dir = os.path.dirname(feature_list_path_from_config)
            initial_list_path = os.path.join(initial_list_dir, initial_list_filename)

            os.makedirs(initial_list_dir, exist_ok=True)
            with open(initial_list_path, 'w') as f:
                for feature_name in initial_feature_names_list_save:
                    f.write(f"{feature_name}\n")
            logger.info(
                f"[{zone_name_display}] Initial full feature list ({len(initial_feature_names_list_save)}) saved to {initial_list_path}.")
        except Exception as e_save_fl_initial:
            logger.error(f"[{zone_name_display}] Failed to save initial full feature list: {e_save_fl_initial}")

    if features_combined.isnull().values.any():  # Final check
        logger.warning(
            f"[{zone_name_display}] NaNs found in FINAL features from prepare_data just before returning. Filling with 0 again.")
        features_combined = features_combined.fillna(0)

    return features_combined, targets_combined, metadata_combined

def visualize_confusion_matrix(cm, categories, title, output_dir):
    try:
        plt.figure(figsize=(max(6, len(categories) * 1.5), max(4, len(categories) * 1.2)))  # Dynamic size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories, yticklabels=categories, cbar=False, annot_kws={"size": 10})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f"{title} Confusion Matrix", fontsize=14)
        safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in title).replace(' ',
                                                                                                       '_')  # Sanitize
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{safe_title}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Confusion matrix '{title}' saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save Confusion Matrix '{title}': {e}", exc_info=True)
        if 'plt' in locals() and plt.gcf().get_axes(): plt.close()  # Ensure plot is closed on error too


def perform_error_analysis(data, output_dir, filename_base, finding_name, class_names_map):
    try:
        if 'Expert' not in data.columns or 'Prediction' not in data.columns:
            logger.warning(f"Error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' column missing.")
            return
        if not pd.api.types.is_numeric_dtype(data['Expert']) or not pd.api.types.is_numeric_dtype(data['Prediction']):
            logger.warning(
                f"Error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' not numeric. Types: Exp={data['Expert'].dtype}, Pred={data['Prediction'].dtype}")
            return

        errors_df = data[data['Prediction'] != data['Expert']].copy()
        error_patterns = {}
        for _, row in errors_df.iterrows():
            expert_label = class_names_map.get(int(row['Expert']), f"Unknown({int(row['Expert'])})")
            pred_label = class_names_map.get(int(row['Prediction']), f"Unknown({int(row['Prediction'])})")
            pattern = f"Expert_{expert_label}_Predicted_{pred_label}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

        filename_base_safe = "".join(
            c if c.isalnum() or c in ('_', '-') else '_' for c in os.path.basename(filename_base))
        total = len(data)
        n_err = len(errors_df)
        err_rate = (n_err / total * 100) if total > 0 else 0

        details_dir = output_dir
        os.makedirs(details_dir, exist_ok=True)
        path = os.path.join(details_dir, f"{filename_base_safe}_error_details.txt")

        with open(path, 'w') as f:
            f.write(
                f"Error Analysis Details: {finding_name}\nFile: {os.path.basename(filename_base)}\n{'=' * 40}\n")  # Use original filename_base for report
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
                errors_df_reset = errors_df.reset_index()
                sort_cols = [col for col in [id_col, side_col] if col and col in errors_df_reset.columns]
                if not sort_cols or not all(col in errors_df_reset.columns for col in sort_cols):
                    sort_cols = ['index'] if 'index' in errors_df_reset.columns else []
                try:
                    errors_sorted = errors_df_reset.sort_values(
                        by=sort_cols) if sort_cols else errors_df_reset.sort_index()
                except Exception as e_sort:
                    logger.warning(f"Sorting errors failed: {e_sort}. Using unsorted errors.")
                    errors_sorted = errors_df_reset
                id_col_write = id_col if id_col and id_col in errors_sorted.columns else (
                    'index' if 'index' in errors_sorted.columns else None)
                for i, row in errors_sorted.iterrows():
                    row_idx_val = row.get('index', i)
                    patient_info_parts = []
                    if id_col_write and id_col_write != 'index' and id_col_write in row and pd.notna(row[id_col_write]):
                        patient_info_parts.append(f"Pt {row[id_col_write]}")
                    else:
                        patient_info_parts.append(f"CaseIdx {row_idx_val}")
                    if side_col and side_col in row and pd.notna(row[side_col]): patient_info_parts.append(
                        f"Side: {row[side_col]}")
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
    none_label_val, complete_label_val = -1, -1
    for k, v in class_names_map.items():
        if str(v).lower() == 'none': none_label_val = int(k)
        if str(v).lower() == 'complete': complete_label_val = int(k)

    if none_label_val == -1 or complete_label_val == -1:
        logger.warning(
            f"Critical error analysis for '{finding_name}' skipped: Class names for 'None' or 'Complete' missing from map {class_names_map}.")
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

        filename_base_safe = "".join(
            c if c.isalnum() or c in ('_', '-') else '_' for c in os.path.basename(filename_base))
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
                    row_idx_val = row.get('index', idx);
                    patient_info_parts = []
                    if id_col_write and id_col_write != 'index' and id_col_write in row and pd.notna(row[id_col_write]):
                        patient_info_parts.append(f"Pt {row[id_col_write]}")
                    else:
                        patient_info_parts.append(f"CaseIdx {row_idx_val}")
                    if side_col and side_col in row and pd.notna(row[side_col]): patient_info_parts.append(
                        f"Side: {row[side_col]}")
                    patient_info_str = ", ".join(patient_info_parts)
                    exp_str = class_names_map.get(int(row['Expert']), f"?({int(row['Expert'])})")
                    pred_str = class_names_map.get(int(row['Prediction']), f"?({int(row['Prediction'])})")
                    f.write(f"  {patient_info_str} - Exp: {exp_str}, Pred: {pred_str}\n")
            else:
                f.write("No critical errors found.\n")
        logger.debug(f"Critical error analysis for '{finding_name}' saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save critical error file {path if 'path' in locals() else 'unknown_path'}: {e}",
                     exc_info=True)


def analyze_partial_errors(data, output_dir, filename_base, finding_name, class_names_map):
    partial_label_val, none_label_val, complete_label_val = -1, -1, -1
    for k, v in class_names_map.items():
        if str(v).lower() == 'partial': partial_label_val = int(k)
        if str(v).lower() == 'none': none_label_val = int(k)
        if str(v).lower() == 'complete': complete_label_val = int(k)

    if partial_label_val == -1 or none_label_val == -1 or complete_label_val == -1:
        logger.warning(
            f"Partial error analysis for '{finding_name}' skipped: Class names for 'Partial', 'None', or 'Complete' missing from map {class_names_map}.")
        return

    partial_label_str = class_names_map[partial_label_val]
    none_label_str = class_names_map[none_label_val]
    complete_label_str = class_names_map[complete_label_val]

    if 'Expert' not in data.columns or 'Prediction' not in data.columns:
        logger.warning(f"Partial error analysis for '{finding_name}' skipped: 'Expert' or 'Prediction' missing.")
        return

    filename_base_safe = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in os.path.basename(filename_base))
    path = os.path.join(output_dir, f"{filename_base_safe}_partial_errors.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

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
                    row_idx_val = row.get('index', idx);
                    patient_info_parts = []
                    if id_col_write and id_col_write != 'index' and id_col_write in row and pd.notna(row[id_col_write]):
                        patient_info_parts.append(f"Pt {row[id_col_write]}")
                    else:
                        patient_info_parts.append(f"CaseIdx {row_idx_val}")
                    if side_col and side_col in row and pd.notna(row[side_col]): patient_info_parts.append(
                        f"Side: {row[side_col]}")
                    patient_info_str = ", ".join(patient_info_parts)
                    exp_str = class_names_map.get(int(row['Expert']), f"?({int(row['Expert'])})")
                    pred_str = class_names_map.get(int(row['Prediction']), f"?({int(row['Prediction'])})")
                    f.write(f"  {patient_info_str} - Exp: {exp_str}, Pred: {pred_str}\n")
            else:
                f.write("No partial errors found.\n")
        logger.debug(f"Partial error analysis for '{finding_name}' saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save partial error file {path}: {e}", exc_info=True)

def calculate_entropy(probabilities):
    probabilities = np.array(probabilities)
    if not np.isclose(np.sum(probabilities), 1.0) and np.sum(probabilities) > 1e-9:  # check sum > 0
        probabilities = probabilities / np.sum(probabilities)
    elif np.sum(probabilities) < 1e-9:
        return np.log2(len(probabilities)) if len(probabilities) > 0 else 0.0
    prob = np.clip(probabilities, 1e-9, 1.0)
    return scipy.stats.entropy(prob, base=2)


def calculate_margin(probabilities):
    if probabilities is None: return 1.0
    probs_array = np.asarray(probabilities)
    if len(probs_array) < 2: return 1.0
    sorted_probs = np.sort(probs_array)[::-1]
    return sorted_probs[0] - sorted_probs[1]


def analyze_training_influence(zone, model, scaler, feature_names,
                               X_train_orig_df, y_train_orig_arr,  # Changed to df and arr for clarity
                               X_test_orig_df, y_test_orig_arr,
                               training_analysis_df,  # For context, not direct use here
                               candidate_indices,  # Indices from X_train_orig_df to analyze
                               config):
    zone_name = config.get('name', zone.capitalize().replace('_', ' ') + ' Face')
    training_params = config.get('training', {})
    smote_config = training_params.get('smote', {})
    smote_enabled = smote_config.get('enabled', False)
    random_state = training_params.get('random_state', 42)
    model_params_config = training_params.get('model_params', {})  # Base XGB params

    influence_scores = {}
    if candidate_indices.empty:
        logger.info(f"[{zone_name}] No candidate indices provided for influence analysis.")
        return influence_scores

    try:
        # Scale the full test set once using the main scaler
        X_test_scaled_main = scaler.transform(X_test_orig_df[feature_names])  # Ensure using correct feature order/set

        # Get baseline performance using the main model (passed in)
        y_pred_baseline = model.predict(X_test_scaled_main)
        baseline_f1 = f1_score(y_test_orig_arr, y_pred_baseline, average='weighted', zero_division=0)
        logger.info(f"[{zone_name}] Baseline F1 for influence analysis: {baseline_f1:.4f} using main model.")

        retrain_params = {k: v for k, v in model_params_config.items() if k in [
            'objective', 'eval_metric', 'learning_rate', 'max_depth', 'min_child_weight',
            'subsample', 'colsample_bytree', 'gamma', 'n_estimators',
            'scale_pos_weight', 'reg_alpha', 'reg_lambda']}
        retrain_params['random_state'] = random_state

        class_map_for_influence = PARALYSIS_MAP  # Assuming paralysis context
        num_classes_influence = len(class_map_for_influence)
        if num_classes_influence > 1: retrain_params['num_class'] = num_classes_influence
        if 'objective' not in retrain_params: retrain_params[
            'objective'] = 'multi:softprob' if num_classes_influence > 2 else 'binary:logistic'
        if 'eval_metric' not in retrain_params: retrain_params[
            'eval_metric'] = 'mlogloss' if num_classes_influence > 2 else 'logloss'
        retrain_params['use_label_encoder'] = False

        analyzed_count = 0
        for idx_label in candidate_indices:  # idx_label is the actual index label from X_train_orig_df
            if idx_label not in X_train_orig_df.index:
                logger.warning(
                    f"[{zone_name}] Index {idx_label} for influence analysis not in X_train_orig_df.index. Skipping.")
                influence_scores[idx_label] = np.nan
                continue

            try:
                X_train_temp_df = X_train_orig_df.drop(index=idx_label)
                y_train_temp_arr = np.delete(y_train_orig_arr, X_train_orig_df.index.get_loc(idx_label))
            except Exception as e_drop:
                logger.warning(
                    f"[{zone_name}] Error dropping index {idx_label} for influence analysis: {e_drop}. Skipping.")
                influence_scores[idx_label] = np.nan;
                continue

            X_train_processed_df, y_train_processed_arr = X_train_temp_df.copy(), y_train_temp_arr.copy()

            if smote_enabled:  # Re-apply SMOTE (simplified for influence analysis)
                unique_labels_temp, counts_temp = np.unique(y_train_temp_arr, return_counts=True)
                if len(unique_labels_temp) <= 1 or any(c == 0 for c in counts_temp):
                    logger.debug(
                        f"[{zone_name}] Skipping SMOTE for influence trial {idx_label}: not enough class diversity.");
                    influence_scores[idx_label] = np.nan;
                    continue
                try:
                    min_class_size_temp = np.min(counts_temp[counts_temp > 0])
                    k_neighbors_smote = max(1, min(smote_config.get('k_neighbors', 5),
                                                   min_class_size_temp - 1)) if min_class_size_temp > 1 else 0
                    if k_neighbors_smote >= 1:
                        smote_influence = SMOTE(random_state=random_state, k_neighbors=k_neighbors_smote,
                                                sampling_strategy='auto')
                        # Ensure X_train_temp_df has feature_names for SMOTE if it expects a DataFrame
                        X_smoted_temp, y_smoted_temp = smote_influence.fit_resample(X_train_temp_df[feature_names],
                                                                                    y_train_temp_arr)  # Use current feature_names
                        X_train_processed_df = pd.DataFrame(X_smoted_temp, columns=feature_names)
                        y_train_processed_arr = y_smoted_temp
                except Exception as e_smote_inf:
                    logger.warning(
                        f"[{zone_name}] SMOTE failed during influence for index {idx_label}: {e_smote_inf}. Using un-SMOTEd.")

            if len(np.unique(y_train_processed_arr)) <= 1:
                logger.debug(
                    f"[{zone_name}] Not enough class diversity after processing for influence trial {idx_label}. Skipping.");
                influence_scores[idx_label] = np.nan;
                continue

            try:
                # Important: Fit a new scaler on this temporary training data
                temp_scaler_influence = StandardScaler()
                X_train_scaled_temp = temp_scaler_influence.fit_transform(
                    X_train_processed_df[feature_names])  # Use current feature_names
                # Transform the original test set using this new scaler
                X_test_scaled_temp = temp_scaler_influence.transform(X_test_orig_df[feature_names])

                temp_model = xgb.XGBClassifier(**retrain_params)
                # Sample weights for this temporary model
                class_weights_temp_map = calculate_class_weights_for_model(y_train_processed_arr,
                                                                           training_params.get('class_weights', {}))
                sample_weights_temp = np.array(
                    [class_weights_temp_map.get(int(lbl), 1.0) for lbl in y_train_processed_arr])

                temp_model.fit(X_train_scaled_temp, y_train_processed_arr, sample_weight=sample_weights_temp,
                               verbose=False)

                y_pred_temp = temp_model.predict(X_test_scaled_temp)
                temp_f1 = f1_score(y_test_orig_arr, y_pred_temp, average='weighted', zero_division=0)
                influence_scores[idx_label] = temp_f1 - baseline_f1
            except Exception as e_retrain:
                logger.warning(
                    f"[{zone_name}] Retraining/evaluation failed for influence index {idx_label}: {e_retrain}.");
                influence_scores[idx_label] = np.nan

            analyzed_count += 1
            if analyzed_count % 10 == 0: logger.info(
                f"[{zone_name}] Influence analysis: Processed {analyzed_count}/{len(candidate_indices)} candidates.")

        logger.info(f"[{zone_name}] Influence analysis complete for {analyzed_count} candidates.")
        return influence_scores
    except Exception as e:
        logger.error(f"Error in overall training influence analysis for {zone_name}: {e}", exc_info=True)
        return {}

def calculate_review_priority_score(row, class_names_map, review_weights_config=None):
    if review_weights_config is None: review_weights_config = REVIEW_CONFIG.get('priority_weights', {})
    score = 0.0

    if 'Is_Correct' in row and pd.notna(row['Is_Correct']) and not row['Is_Correct']:
        score += 30.0 * review_weights_config.get('error_severity', 1.0)
        if 'Expert_Label' in row and 'Predicted_Label' in row and \
                pd.notna(row['Expert_Label']) and pd.notna(row['Predicted_Label']):
            try:
                expert_num = int(row['Expert_Label']);
                pred_num = int(row['Predicted_Label'])
                none_val, complete_val = -1, -1
                for k, v in class_names_map.items():
                    if str(v).lower() == 'none': none_val = int(k)
                    if str(v).lower() == 'complete': complete_val = int(k)
                if none_val != -1 and complete_val != -1:
                    if (expert_num == none_val and pred_num == complete_val) or \
                            (expert_num == complete_val and pred_num == none_val):
                        score += 50.0 * review_weights_config.get('error_severity', 1.0)
            except ValueError:
                pass

    if 'Entropy' in row and pd.notna(row['Entropy']): score += row['Entropy'] * 10.0 * review_weights_config.get(
        'confidence', 1.0)
    if 'Margin' in row and pd.notna(row['Margin']): score += (1.0 - row['Margin']) * 10.0 * review_weights_config.get(
        'confidence', 1.0)
    if 'Prob_True_Label' in row and pd.notna(row['Prob_True_Label']) and row['Prob_True_Label'] < 0.5:
        score += (0.5 - row['Prob_True_Label']) * 40.0 * review_weights_config.get('confidence', 1.0)

    if 'Influence_Score (Delta_F1)' in row and pd.notna(row['Influence_Score (Delta_F1)']):
        influence = row['Influence_Score (Delta_F1)']
        if influence < 0: score += abs(influence) * 100.0 * review_weights_config.get('influence', 1.0)

    if 'Flag_Reason' in row and pd.notna(row['Flag_Reason']) and isinstance(row['Flag_Reason'], str):
        if 'CRITICAL' in row['Flag_Reason'].upper(): score += 40.0 * review_weights_config.get('error_severity', 1.0)
        if 'INCONSISTENT' in row['Flag_Reason'].upper(): score += 20.0 * review_weights_config.get('inconsistency', 1.0)
    return score


def generate_review_candidates(zone, model, scaler, feature_names, config,  # feature_names are selected ones
                               X_train_orig, y_train_orig,
                               # Original training data (X_train_orig is DataFrame with selected features)
                               X_test_orig, y_test_orig,
                               # Original test data (X_test_orig is DataFrame with selected features)
                               metadata_train,  # Metadata corresponding to X_train_orig
                               training_analysis_df,  # DataFrame from train_model_workflow (uncertainty, etc.)
                               class_names_map,
                               influence_scores_manual=None):
    zone_name_rev = config.get('name', zone.capitalize().replace('_', ' ') + ' Face')
    review_cfg = config.get('training', {}).get('review_analysis', {})
    review_weights_cfg = REVIEW_CONFIG.get('priority_weights', {})  # Global review weights

    if training_analysis_df is None or training_analysis_df.empty:
        logger.error(f"[{zone_name_rev}] Review Candidates: training_analysis_df is missing or empty. Cannot generate.")
        return pd.DataFrame()

    if metadata_train is None or not isinstance(metadata_train, pd.DataFrame):
        logger.warning(f"[{zone_name_rev}] Review Candidates: metadata_train is None or not DataFrame. Creating dummy.")
        metadata_train = pd.DataFrame(index=training_analysis_df.index)
    if 'Patient ID' not in metadata_train.columns: metadata_train[
        'Patient ID'] = "UnknownID_Rev_" + metadata_train.index.astype(str)
    if 'Side' not in metadata_train.columns: metadata_train['Side'] = "UnknownSide_Rev"

    current_analysis_df_rev = training_analysis_df.copy()
    if not X_train_orig.index.equals(current_analysis_df_rev.index):
        logger.warning(
            f"[{zone_name_rev}] Review Candidates: Index mismatch X_train_orig & training_analysis_df. Reindexing analysis_df.")
        current_analysis_df_rev = current_analysis_df_rev.reindex(X_train_orig.index)
        current_analysis_df_rev.dropna(subset=['Expert_Label'], inplace=True)  # Drop if essential data became NaN

    if not current_analysis_df_rev.index.equals(metadata_train.index):
        logger.warning(
            f"[{zone_name_rev}] Review Candidates: Index mismatch analysis_df & metadata_train. Reindexing metadata.")
        metadata_train = metadata_train.reindex(current_analysis_df_rev.index)
        if 'Patient ID' in metadata_train: metadata_train['Patient ID'].fillna(f"UnknownID_Reidx_{zone_name_rev}",
                                                                               inplace=True)
        if 'Side' in metadata_train: metadata_train['Side'].fillna("UnknownSide_Reidx", inplace=True)

    if current_analysis_df_rev.empty:
        logger.warning(f"[{zone_name_rev}] Review Candidates: training_analysis_df empty after reindexing/cleaning.");
        return pd.DataFrame()

    try:
        candidates_df_rev = metadata_train.join(current_analysis_df_rev, how='inner')
        if candidates_df_rev.empty:
            logger.warning(
                f"[{zone_name_rev}] Review Candidates: No candidates after joining metadata with analysis data.");
            return pd.DataFrame()

        candidates_df_rev['Flag_Reason'] = ''
        if 'Is_Correct' in candidates_df_rev.columns:
            misclassified_mask = ~candidates_df_rev['Is_Correct'].fillna(True)
            candidates_df_rev.loc[misclassified_mask, 'Flag_Reason'] += 'Misclassified; '
            if 0 in class_names_map and 2 in class_names_map:  # Check for None and Complete
                if 'Expert_Label' in candidates_df_rev.columns and 'Predicted_Label' in candidates_df_rev.columns:
                    expert_label_num = pd.to_numeric(candidates_df_rev['Expert_Label'], errors='coerce')
                    predicted_label_num = pd.to_numeric(candidates_df_rev['Predicted_Label'], errors='coerce')
                    critical_mask = ((expert_label_num == 0) & (predicted_label_num == 2)) | \
                                    ((expert_label_num == 2) & (predicted_label_num == 0))
                    if misclassified_mask.index.equals(critical_mask.index):  # Ensure alignment
                        candidates_df_rev.loc[
                            misclassified_mask & critical_mask.fillna(False), 'Flag_Reason'] += 'CRITICAL_Error; '

        entropy_quantile_cfg = review_cfg.get('entropy_quantile', 0.9)
        if 'Entropy' in candidates_df_rev.columns and pd.api.types.is_numeric_dtype(candidates_df_rev['Entropy']) and \
                candidates_df_rev['Entropy'].notna().any():
            try:
                entropy_threshold_val = candidates_df_rev['Entropy'].quantile(entropy_quantile_cfg)
                candidates_df_rev.loc[candidates_df_rev[
                                          'Entropy'] >= entropy_threshold_val, 'Flag_Reason'] += f'High_Entropy(>{entropy_threshold_val:.2f}); '
            except Exception as e_entropy:
                logger.warning(f"[{zone_name_rev}] Could not calculate entropy threshold for review: {e_entropy}")

        margin_quantile_cfg = review_cfg.get('margin_quantile', 0.1)
        if 'Margin' in candidates_df_rev.columns and pd.api.types.is_numeric_dtype(candidates_df_rev['Margin']) and \
                candidates_df_rev['Margin'].notna().any():
            try:
                margin_threshold_val = candidates_df_rev['Margin'].quantile(margin_quantile_cfg)
                candidates_df_rev.loc[candidates_df_rev[
                                          'Margin'] <= margin_threshold_val, 'Flag_Reason'] += f'Low_Margin(<{margin_threshold_val:.2f}); '
            except Exception as e_margin:
                logger.warning(f"[{zone_name_rev}] Could not calculate margin threshold for review: {e_margin}")

        true_label_prob_thresh_cfg = review_cfg.get('true_label_prob_threshold', 0.4)
        if 'Prob_True_Label' in candidates_df_rev.columns and pd.api.types.is_numeric_dtype(
                candidates_df_rev['Prob_True_Label']):
            candidates_df_rev.loc[candidates_df_rev[
                                      'Prob_True_Label'] < true_label_prob_thresh_cfg, 'Flag_Reason'] += f'Low_Prob_True(<{true_label_prob_thresh_cfg:.1f}); '

        candidates_df_rev['Influence_Score (Delta_F1)'] = np.nan
        if influence_scores_manual is not None and isinstance(influence_scores_manual, dict):
            candidates_df_rev['Influence_Score (Delta_F1)'] = candidates_df_rev.index.map(
                influence_scores_manual).fillna(np.nan)
            influence_calculated_mask = candidates_df_rev['Influence_Score (Delta_F1)'].notna()
            if influence_calculated_mask.any():
                def format_influence_reason(row_inf):
                    reason = row_inf['Flag_Reason'];
                    influence_val = row_inf['Influence_Score (Delta_F1)']
                    influence_str = f"Influence({influence_val:.3f})"
                    return (reason + '; ' + influence_str) if reason else influence_str

                candidates_df_rev.loc[influence_calculated_mask, 'Flag_Reason'] = candidates_df_rev[
                    influence_calculated_mask].apply(format_influence_reason, axis=1)

        candidates_df_rev['Flag_Reason'] = candidates_df_rev['Flag_Reason'].str.strip().str.rstrip(';')
        flagged_candidates_interim = candidates_df_rev[candidates_df_rev['Flag_Reason'] != ''].copy()

        if flagged_candidates_interim.empty:
            logger.info(f"[{zone_name_rev}] No candidates flagged for review based on criteria.");
            return pd.DataFrame()

        flagged_candidates_interim['PriorityScore'] = flagged_candidates_interim.apply(
            lambda row: calculate_review_priority_score(row, class_names_map, review_weights_cfg), axis=1
        )

        if 'Expert_Label' in flagged_candidates_interim: flagged_candidates_interim['Expert_Label_Name'] = \
        flagged_candidates_interim['Expert_Label'].map(class_names_map).fillna('Unknown')
        if 'Predicted_Label' in flagged_candidates_interim: flagged_candidates_interim['Predicted_Label_Name'] = \
        flagged_candidates_interim['Predicted_Label'].map(class_names_map).fillna('Unknown')

        prob_cols_to_keep = sorted([col for col in flagged_candidates_interim.columns if col.startswith("Prob_")])
        output_columns = ['Patient ID', 'Side']
        for col in ['Expert_Label_Name', 'Predicted_Label_Name', 'Is_Correct', 'Expert_Label', 'Predicted_Label']:
            if col in flagged_candidates_interim.columns: output_columns.append(col)
        output_columns.extend(prob_cols_to_keep)
        for col in ['Entropy', 'Margin', 'Influence_Score (Delta_F1)', 'Flag_Reason', 'PriorityScore']:
            if col in flagged_candidates_interim.columns: output_columns.append(col)

        final_columns_present = [col for col in output_columns if col in flagged_candidates_interim.columns]
        final_candidates_df_rev = flagged_candidates_interim[final_columns_present].copy()

        for col in ['Patient ID', 'Side']:
            if col in final_candidates_df_rev: final_candidates_df_rev[col] = final_candidates_df_rev[col].astype(str)

        sort_by_cols, ascending_order = [], []
        if 'PriorityScore' in final_candidates_df_rev: sort_by_cols.append('PriorityScore'); ascending_order.append(
            False)
        if 'Patient ID' in final_candidates_df_rev: sort_by_cols.append('Patient ID'); ascending_order.append(True)
        if 'Side' in final_candidates_df_rev: sort_by_cols.append('Side'); ascending_order.append(True)
        if sort_by_cols: final_candidates_df_rev = final_candidates_df_rev.sort_values(by=sort_by_cols,
                                                                                       ascending=ascending_order)

        return final_candidates_df_rev
    except Exception as e_rev:
        logger.error(f"Error generating review candidates for {zone_name_rev}: {e_rev}", exc_info=True)
        return pd.DataFrame()

# Other utility functions from original (evaluate_thresholds, find_similar_patients, etc.)
# can be added here if they are still needed and adapted to the new data structures/flow.
# For example, evaluate_thresholds would need to be called with probabilities from the *final model*.
# find_similar_patients would use the *selected features*.

# def evaluate_thresholds(data_df, proba_cols_map, expert_label_col, output_dir, finding_name, class_names_map):
# ... (logic for this would need careful adaptation based on how probabilities are stored and if it's binary/multiclass)