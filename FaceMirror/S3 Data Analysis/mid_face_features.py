# mid_face_features.py (Refactored to use _extract_base_au_features)

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Use central config and utils
from paralysis_config import ZONE_CONFIG  # For zone-specific details
from paralysis_utils import _extract_base_au_features, calculate_ratio  # Import helpers

logger = logging.getLogger(__name__)
ZONE = 'mid'  # Define zone for this file


# extract_features function (MID FACE - Training)
def extract_features(df, side, zone_specific_config):
    """ Extracts features for MID FACE training using the helper and adding custom ones. """
    zone_name = zone_specific_config.get('name', 'Mid Face')
    actions = zone_specific_config.get('actions', [])
    aus = zone_specific_config.get('aus', [])
    feature_cfg = zone_specific_config.get('feature_extraction', {})
    min_val_cfg = feature_cfg.get('min_value', 0.0001)

    logger.debug(f"[{zone_name}] Extracting features for {side} side (Training)...")

    base_features_df = _extract_base_au_features(df, side, actions, aus, feature_cfg, zone_name)
    feature_data = base_features_df.to_dict('series')

    # 2. ET/ES Ratio Features (MID FACE SPECIFIC)
    for au_str_loop in aus:  # Use aus from config
        action_et = 'ET';
        action_es = 'ES'
        et_val_side = feature_data.get(f"{action_et}_{au_str_loop}_val_side")
        es_val_side = feature_data.get(f"{action_es}_{au_str_loop}_val_side")
        et_val_opp = feature_data.get(f"{action_et}_{au_str_loop}_val_opp")
        es_val_opp = feature_data.get(f"{action_es}_{au_str_loop}_val_opp")

        if all(isinstance(s, pd.Series) for s in [et_val_side, es_val_side, et_val_opp, es_val_opp]):
            ratio_side = calculate_ratio(et_val_side, es_val_side, min_value=min_val_cfg)
            ratio_opp = calculate_ratio(et_val_opp, es_val_opp, min_value=min_val_cfg)
            feature_data[f"{au_str_loop}_ETES_Ratio_Side"] = ratio_side
            feature_data[f"{au_str_loop}_ETES_Ratio_Opp"] = ratio_opp
            # ... (other ETES asymmetry features as in your original)
        else:  # Add defaults if any component is missing
            logger.debug(
                f"[{zone_name}] Missing ES/ET values for AU {au_str_loop} on side {side}. Using default ET/ES ratio features.")
            feature_data[f"{au_str_loop}_ETES_Ratio_Side"] = pd.Series(1.0, index=df.index)
            feature_data[f"{au_str_loop}_ETES_Ratio_Opp"] = pd.Series(1.0, index=df.index)
            # ... (add defaults for other ETES asymmetry features)

    # 3. Interaction/Summary Features (MID FACE SPECIFIC)
    es_au45_val_side = feature_data.get('ES_AU45_r_val_side', pd.Series(0.0, index=df.index))
    et_au45_val_side = feature_data.get('ET_AU45_r_val_side', pd.Series(0.0, index=df.index))
    es_au07_val_side = feature_data.get('ES_AU07_r_val_side', pd.Series(0.0, index=df.index))
    et_au07_val_side = feature_data.get('ET_AU07_r_val_side', pd.Series(0.0, index=df.index))

    feature_data['ES_ET_AU45_ratio_side'] = calculate_ratio(es_au45_val_side, et_au45_val_side, min_value=min_val_cfg)
    feature_data['ET_ES_AU45_diff_side'] = (et_au45_val_side - es_au45_val_side).abs()
    feature_data['ES_ET_AU07_ratio_side'] = calculate_ratio(es_au07_val_side, et_au07_val_side, min_value=min_val_cfg)
    feature_data['ET_ES_AU07_diff_side'] = (et_au07_val_side - es_au07_val_side).abs()

    for au_base_str in aus:  # Use aus from config
        val_side_cols = [f"{act}_{au_base_str}_val_side" for act in actions if
                         f"{act}_{au_base_str}_val_side" in feature_data]
        if val_side_cols:
            all_series = [pd.to_numeric(feature_data[col], errors='coerce').fillna(0) for col in val_side_cols]
            if all_series:  # Check if list is not empty
                try:
                    stacked_vals = np.stack([s.to_numpy() for s in all_series], axis=1)
                    max_vals = np.max(stacked_vals, axis=1);
                    min_vals_agg = np.min(stacked_vals, axis=1)  # Renamed min_vals
                    feature_data[f"max_{au_base_str}_val_side"] = pd.Series(max_vals, index=df.index)
                    feature_data[f"min_{au_base_str}_val_side"] = pd.Series(min_vals_agg, index=df.index)
                    feature_data[f"range_{au_base_str}_val_side"] = pd.Series(max_vals - min_vals_agg, index=df.index)
                except ValueError as e:  # Catch errors if stacking empty arrays
                    logger.debug(
                        f"[{zone_name}] Stacking error for {au_base_str} value (side: {side}): {e}. Defaulting features.");
                    feature_data[f"max_{au_base_str}_val_side"] = pd.Series(0.0, index=df.index)
                    feature_data[f"min_{au_base_str}_val_side"] = pd.Series(0.0, index=df.index)
                    feature_data[f"range_{au_base_str}_val_side"] = pd.Series(0.0, index=df.index)
            else:  # Add default series if no valid columns found
                feature_data[f"max_{au_base_str}_val_side"] = pd.Series(0.0, index=df.index)
                feature_data[f"min_{au_base_str}_val_side"] = pd.Series(0.0, index=df.index)
                feature_data[f"range_{au_base_str}_val_side"] = pd.Series(0.0, index=df.index)
        # ... (similar logic for _Asym_PercDiff and _Asym_Ratio summaries) ...

    features_df_final = pd.DataFrame(feature_data, index=df.index)
    logger.debug(f"[{zone_name}] Generated {features_df_final.shape[1]} features for {side} (Training).")
    return features_df_final


# extract_features_for_detection (MID FACE - Detection) - Needs similar refactoring
def extract_features_for_detection(row_data, side, zone_key_for_detection):
    # This function will also call _extract_base_au_features
    # and then add its specific summary/interaction features for a single row.
    # (Implementation similar to lower_face_features.py's detection function, but with mid-face specific summaries)
    try:
        from paralysis_config import ZONE_CONFIG as global_zone_config  # Ensure it's accessible
        det_config = global_zone_config[zone_key_for_detection]
        det_feature_cfg = det_config.get('feature_extraction', {})
        det_actions = det_config.get('actions', [])
        det_aus = det_config.get('aus', [])
        det_filenames = det_config.get('filenames', {})
        det_zone_name = det_config.get('name', zone_key_for_detection.capitalize() + ' Face')
        min_val_cfg_det = det_feature_cfg.get('min_value', 0.0001)

    except KeyError:
        logger.error(f"Config for requested zone '{zone_key_for_detection}' not found in detection. Cannot proceed.")
        return None

    feature_list_path = det_filenames.get('feature_list')
    if not feature_list_path or not os.path.exists(feature_list_path):
        logger.error(f"[{det_zone_name}] Feature list not found. Cannot extract detection features.")
        return None
    try:
        ordered_feature_names = joblib.load(feature_list_path)
    except Exception as e:
        logger.error(f"[{det_zone_name}] Failed load feature list: {e}"); return None

    if isinstance(row_data, dict):
        df_single_row = pd.DataFrame([row_data], index=[0])
    elif isinstance(row_data, pd.Series):
        df_single_row = pd.DataFrame([row_data.to_dict()], index=[0])
    else:
        logger.error(f"[{det_zone_name}] Invalid row_data type: {type(row_data)}"); return None

    base_features_df_det = _extract_base_au_features(df_single_row, side, det_actions, det_aus, det_feature_cfg,
                                                     det_zone_name)
    feature_data_det = base_features_df_det.to_dict('series')

    # ET/ES Ratio Features for detection
    for au_str_loop_det in det_aus:
        action_et = 'ET';
        action_es = 'ES'
        et_val_side_det = feature_data_det.get(f"{action_et}_{au_str_loop_det}_val_side")
        es_val_side_det = feature_data_det.get(f"{action_es}_{au_str_loop_det}_val_side")
        # ... (similar for opp side, and calculation as in training extract_features) ...
        # Ensure to use .iloc[0] for single row Series
        if all(isinstance(s, pd.Series) and not s.empty for s in
               [et_val_side_det, es_val_side_det]):  # Simplified check
            feature_data_det[f"{au_str_loop_det}_ETES_Ratio_Side"] = calculate_ratio(et_val_side_det, es_val_side_det,
                                                                                     min_value=min_val_cfg_det)
        else:
            feature_data_det[f"{au_str_loop_det}_ETES_Ratio_Side"] = pd.Series([1.0])
        # ... (add other ETES features with .iloc[0] and defaults)

    # Interaction/Summary Features for detection
    es_au45_s = feature_data_det.get('ES_AU45_r_val_side', pd.Series([0.0]))
    et_au45_s = feature_data_det.get('ET_AU45_r_val_side', pd.Series([0.0]))
    feature_data_det['ES_ET_AU45_ratio_side'] = calculate_ratio(es_au45_s, et_au45_s, min_value=min_val_cfg_det)
    # ... (other mid-face specific summary features, ensuring .iloc[0] is used for Series values) ...

    feature_data_det["side_indicator"] = pd.Series([0 if side.lower() == 'left' else 1])

    feature_dict_final_det = {k: v.iloc[0] for k, v in feature_data_det.items() if
                              isinstance(v, pd.Series) and not v.empty}
    # ... (final assembly into feature_vector as in lower_face_features.py) ...
    feature_vector = []
    for name in ordered_feature_names:
        value = feature_dict_final_det.get(name, 0.0)  # Default to 0.0 if missing
        try:
            val_float = float(value)
            feature_vector.append(0.0 if np.isnan(val_float) else val_float)
        except (ValueError, TypeError):
            feature_vector.append(0.0)
    return feature_vector