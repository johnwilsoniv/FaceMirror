# lower_face_features.py (Refactored - Uses _extract_base_au_features from utils)

import numpy as np
import pandas as pd
import logging
import os  # Not strictly needed here anymore if prepare_data is removed
import joblib  # Not strictly needed here anymore

# Use central config and utils
from paralysis_config import ZONE_CONFIG  # For zone-specific details
from paralysis_utils import _extract_base_au_features  # Import the new helper

logger = logging.getLogger(__name__)
ZONE = 'lower'  # Define zone for this file


# --- extract_features function (LOWER FACE - Training) ---
def extract_features(df, side, zone_specific_config):  # df is merged_df from prepare_data_generalized
    """ Extracts features for LOWER FACE training using the helper and adding custom ones. """
    zone_name = zone_specific_config.get('name', 'Lower Face')
    actions = zone_specific_config.get('actions', [])
    aus = zone_specific_config.get('aus', [])
    feature_cfg = zone_specific_config.get('feature_extraction', {})

    logger.debug(f"[{zone_name}] Extracting features for {side} side (Training) via zone-specific logic...")

    # 1. Extract base AU features using the helper
    base_features_df = _extract_base_au_features(df, side, actions, aus, feature_cfg, zone_name)

    # Convert to dict of Series for easy addition of custom features
    # Ensure correct index from the input 'df' is maintained
    feature_data = {col: pd.Series(base_features_df[col].values, index=df.index) for col in base_features_df.columns}

    # 2. Interaction/Summary Features (LOWER FACE SPECIFIC)
    # Retrieve already calculated base features from feature_data to build summary features
    avg_au12_ratio_vals = []
    max_au12_pd_vals_list = []
    for act in actions:  # Iterate through actions defined for this zone
        ratio_key = f"{act}_AU12_r_Asym_Ratio"
        pd_key = f"{act}_AU12_r_Asym_PercDiff"
        if ratio_key in feature_data and isinstance(feature_data[ratio_key], pd.Series):
            avg_au12_ratio_vals.append(feature_data[ratio_key])
        if pd_key in feature_data and isinstance(feature_data[pd_key], pd.Series):
            max_au12_pd_vals_list.append(feature_data[pd_key])

    if avg_au12_ratio_vals:
        feature_data['avg_AU12_Asym_Ratio'] = pd.concat(avg_au12_ratio_vals, axis=1).mean(axis=1)
    else:
        feature_data['avg_AU12_Asym_Ratio'] = pd.Series(1.0, index=df.index)

    if max_au12_pd_vals_list:
        feature_data['max_AU12_Asym_PercDiff'] = pd.concat(max_au12_pd_vals_list, axis=1).max(axis=1)
    else:
        feature_data['max_AU12_Asym_PercDiff'] = pd.Series(0.0, index=df.index)

    bs_au12_asym_ratio = feature_data.get('BS_AU12_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    bs_au25_asym_ratio = feature_data.get('BS_AU25_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    feature_data['BS_Asym_Ratio_Product_12_25'] = bs_au12_asym_ratio * bs_au25_asym_ratio

    features_df_final = pd.DataFrame(feature_data, index=df.index)  # Ensure index is consistent

    # Final check for non-numeric types, just in case custom features introduced them
    non_numeric_cols = features_df_final.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"[{zone_name}] Non-numeric cols in final features: {non_numeric_cols.tolist()}. Coercing.")
        for col in non_numeric_cols: features_df_final[col] = pd.to_numeric(features_df_final[col],
                                                                            errors='coerce').fillna(0.0)

    logger.debug(f"[{zone_name}] Generated {features_df_final.shape[1]} features for {side} (Training).")
    return features_df_final


# --- extract_features_for_detection (LOWER FACE - Detection) ---
def extract_features_for_detection(row_data, side, zone_key_for_detection):
    """ Extracts features for LOWER FACE detection from a row of data. """
    # This function is called by the main detection pipeline, not directly by training.
    # It needs to load its specific configuration for the 'zone_key_for_detection'.
    try:
        from paralysis_config import ZONE_CONFIG as global_zone_config  # Ensure it's accessible
        det_config = global_zone_config[zone_key_for_detection]
        det_feature_cfg = det_config.get('feature_extraction', {})
        det_actions = det_config.get('actions', [])  # Actions for the specific zone
        det_aus = det_config.get('aus', [])  # AUs for the specific zone
        det_filenames = det_config.get('filenames', {})
        det_zone_name = det_config.get('name', zone_key_for_detection.capitalize() + ' Face')
    except KeyError:
        logger.error(f"Config for requested zone '{zone_key_for_detection}' not found in detection. Cannot proceed.")
        return None

    feature_list_path = det_filenames.get('feature_list')
    if not feature_list_path or not os.path.exists(feature_list_path):
        logger.error(
            f"[{det_zone_name}] Feature list not found at {feature_list_path}. Cannot extract detection features.")
        return None
    try:
        # Handle both .list (text) and .pkl (pickle) files
        if feature_list_path.endswith('.list'):
            with open(feature_list_path, 'r') as f:
                ordered_feature_names = [line.strip() for line in f if line.strip()]
        else:
            ordered_feature_names = joblib.load(feature_list_path)
        if not isinstance(ordered_feature_names, list):
            logger.error(f"[{det_zone_name}] Loaded feature names is not a list.");
            return None
    except Exception as e:
        logger.error(f"[{det_zone_name}] Failed load feature list: {e}"); return None

    if isinstance(row_data, dict):
        df_single_row = pd.DataFrame([row_data], index=[0])
    elif isinstance(row_data, pd.Series):
        df_single_row = pd.DataFrame([row_data.to_dict()], index=[0])
    else:
        logger.error(f"[{det_zone_name}] Invalid row_data type: {type(row_data)}"); return None

    # 1. Extract base AU features using the helper
    # Pass the actions and aus specific to this detection zone
    base_features_df_det = _extract_base_au_features(df_single_row, side, det_actions, det_aus, det_feature_cfg,
                                                     det_zone_name)
    feature_data_det = base_features_df_det.to_dict('series')

    # 2. Interaction/Summary Features (LOWER FACE SPECIFIC for detection)
    avg_au12_ratio_vals_det = []
    max_au12_pd_vals_list_det = []
    for act in det_actions:  # Use det_actions for the current zone
        ratio_key = f"{act}_AU12_r_Asym_Ratio"
        pd_key = f"{act}_AU12_r_Asym_PercDiff"
        if ratio_key in feature_data_det and isinstance(feature_data_det[ratio_key], pd.Series) and not \
        feature_data_det[ratio_key].empty:
            avg_au12_ratio_vals_det.append(feature_data_det[ratio_key].iloc[0])
        if pd_key in feature_data_det and isinstance(feature_data_det[pd_key], pd.Series) and not feature_data_det[
            pd_key].empty:
            max_au12_pd_vals_list_det.append(feature_data_det[pd_key].iloc[0])

    feature_data_det['avg_AU12_Asym_Ratio'] = pd.Series(
        [np.mean(avg_au12_ratio_vals_det) if avg_au12_ratio_vals_det else 1.0])
    feature_data_det['max_AU12_Asym_PercDiff'] = pd.Series(
        [np.max(max_au12_pd_vals_list_det) if max_au12_pd_vals_list_det else 0.0])

    bs_au12_asym_ratio_series = feature_data_det.get('BS_AU12_r_Asym_Ratio', pd.Series([1.0]))
    bs_au25_asym_ratio_series = feature_data_det.get('BS_AU25_r_Asym_Ratio', pd.Series([1.0]))

    val_bs_au12_ratio = bs_au12_asym_ratio_series.iloc[0] if not bs_au12_asym_ratio_series.empty else 1.0
    val_bs_au25_ratio = bs_au25_asym_ratio_series.iloc[0] if not bs_au25_asym_ratio_series.empty else 1.0

    feature_data_det['BS_Asym_Ratio_Product_12_25'] = pd.Series([val_bs_au12_ratio * val_bs_au25_ratio])

    feature_data_det["side_indicator"] = pd.Series([0 if side.lower() == 'left' else 1])

    # Final Assembly to ordered list
    feature_dict_final_det = {k: v.iloc[0] for k, v in feature_data_det.items() if
                              isinstance(v, pd.Series) and not v.empty}
    feature_vector = []
    missing_count = 0
    for name in ordered_feature_names:
        value = feature_dict_final_det.get(name)
        if value is None:
            missing_count += 1
            feature_vector.append(0.0)  # Default for missing features
        else:
            try:
                val_float = float(value)
                feature_vector.append(0.0 if np.isnan(val_float) else val_float)
            except (ValueError, TypeError):
                feature_vector.append(0.0)  # Default if conversion fails

    if missing_count > 0: logger.warning(
        f"[{det_zone_name}] Detection: {missing_count} expected features were missing for {side} and defaulted to 0.0.")
    if len(feature_vector) != len(ordered_feature_names):
        logger.error(
            f"[{det_zone_name}] CRITICAL MISMATCH: Final feature vector length {len(feature_vector)} != expected {len(ordered_feature_names)} based on loaded list.")
        return None  # Or handle error appropriately

    logger.debug(f"[{det_zone_name}] Generated {len(feature_vector)} detection features for {side}.")
    return feature_vector