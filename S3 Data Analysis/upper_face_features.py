# upper_face_features.py (Refactored to use _extract_base_au_features)

import numpy as np
import pandas as pd
import logging
import os
import joblib

# Use central config and utils
from paralysis_config import ZONE_CONFIG  # For zone-specific details
from paralysis_utils import _extract_base_au_features  # Import the new helper

logger = logging.getLogger(__name__)
ZONE = 'upper'  # Define zone for this file


# extract_features function (UPPER FACE - Training)
def extract_features(df, side, zone_specific_config):
    """ Extracts features for UPPER FACE training using the helper and adding custom ones. """
    zone_name = zone_specific_config.get('name', 'Upper Face')
    actions = zone_specific_config.get('actions', [])
    aus = zone_specific_config.get('aus', [])
    feature_cfg = zone_specific_config.get('feature_extraction', {})

    logger.debug(f"[{zone_name}] Extracting features for {side} side (Training)...")

    base_features_df = _extract_base_au_features(df, side, actions, aus, feature_cfg, zone_name)
    feature_data = base_features_df.to_dict('series')

    # Interaction/Summary Features (UPPER FACE SPECIFIC)
    au1_ratio = feature_data.get('RE_AU01_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    au2_ratio = feature_data.get('RE_AU02_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    feature_data['RE_avg_Asym_Ratio'] = (au1_ratio + au2_ratio) / 2.0

    au1_pd = feature_data.get('RE_AU01_r_Asym_PercDiff', pd.Series(0.0, index=df.index))
    au2_pd = feature_data.get('RE_AU02_r_Asym_PercDiff', pd.Series(0.0, index=df.index))
    feature_data['RE_avg_Asym_PercDiff'] = (au1_pd + au2_pd) / 2.0

    pd_series_list = [s for s in [au1_pd, au2_pd] if isinstance(s, pd.Series)]
    if pd_series_list:
        feature_data['RE_max_Asym_PercDiff'] = pd.concat(pd_series_list, axis=1).max(axis=1)
    else:
        feature_data['RE_max_Asym_PercDiff'] = pd.Series(0.0, index=df.index)

    au1_val_side = feature_data.get('RE_AU01_r_val_side', pd.Series(0.0, index=df.index))
    au2_val_side = feature_data.get('RE_AU02_r_val_side', pd.Series(0.0, index=df.index))
    feature_data['RE_AU01_AU02_product_side'] = au1_val_side * au2_val_side
    feature_data['RE_AU01_AU02_sum_side'] = au1_val_side + au2_val_side

    features_df_final = pd.DataFrame(feature_data, index=df.index)
    logger.debug(f"[{zone_name}] Generated {features_df_final.shape[1]} features for {side} (Training).")
    return features_df_final


# extract_features_for_detection (UPPER FACE - Detection) - Needs similar refactoring
def extract_features_for_detection(row_data, side, zone_key_for_detection):
    # This function will also call _extract_base_au_features
    # and then add its specific summary/interaction features for a single row.
    # (Implementation similar to lower_face_features.py's detection function, but with upper-face specific summaries)
    try:
        from paralysis_config import ZONE_CONFIG as global_zone_config
        det_config = global_zone_config[zone_key_for_detection]
        det_feature_cfg = det_config.get('feature_extraction', {})
        det_actions = det_config.get('actions', [])
        det_aus = det_config.get('aus', [])
        det_filenames = det_config.get('filenames', {})
        det_zone_name = det_config.get('name', zone_key_for_detection.capitalize() + ' Face')
    except KeyError:
        logger.error(f"Config for requested zone '{zone_key_for_detection}' not found in detection. Cannot proceed.")
        return None

    feature_list_path = det_filenames.get('feature_list')
    if not feature_list_path or not os.path.exists(feature_list_path):
        logger.error(f"[{det_zone_name}] Feature list not found. Cannot extract detection features.")
        return None
    try:
        # Handle both .pkl and .list text files
        if feature_list_path.endswith('.list'):
            with open(feature_list_path, 'r') as f:
                ordered_feature_names = [line.strip() for line in f if line.strip()]
        else:
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

    # Interaction/Summary Features for detection (Upper Face)
    au1_ratio_s = feature_data_det.get('RE_AU01_r_Asym_Ratio', pd.Series([1.0]))
    au2_ratio_s = feature_data_det.get('RE_AU02_r_Asym_Ratio', pd.Series([1.0]))
    au1_ratio_val = au1_ratio_s.iloc[0] if isinstance(au1_ratio_s, pd.Series) and not au1_ratio_s.empty else 1.0
    au2_ratio_val = au2_ratio_s.iloc[0] if isinstance(au2_ratio_s, pd.Series) and not au2_ratio_s.empty else 1.0
    feature_data_det['RE_avg_Asym_Ratio'] = pd.Series([(au1_ratio_val + au2_ratio_val) / 2.0])

    au1_pd_s = feature_data_det.get('RE_AU01_r_Asym_PercDiff', pd.Series([0.0]))
    au2_pd_s = feature_data_det.get('RE_AU02_r_Asym_PercDiff', pd.Series([0.0]))
    au1_pd_val = au1_pd_s.iloc[0] if isinstance(au1_pd_s, pd.Series) and not au1_pd_s.empty else 0.0
    au2_pd_val = au2_pd_s.iloc[0] if isinstance(au2_pd_s, pd.Series) and not au2_pd_s.empty else 0.0
    feature_data_det['RE_avg_Asym_PercDiff'] = pd.Series([(au1_pd_val + au2_pd_val) / 2.0])
    feature_data_det['RE_max_Asym_PercDiff'] = pd.Series([max(au1_pd_val, au2_pd_val)])

    au1_val_side_s = feature_data_det.get('RE_AU01_r_val_side', pd.Series([0.0]))
    au2_val_side_s = feature_data_det.get('RE_AU02_r_val_side', pd.Series([0.0]))
    au1_val_side_val = au1_val_side_s.iloc[0] if isinstance(au1_val_side_s, pd.Series) and not au1_val_side_s.empty else 0.0
    au2_val_side_val = au2_val_side_s.iloc[0] if isinstance(au2_val_side_s, pd.Series) and not au2_val_side_s.empty else 0.0
    feature_data_det['RE_AU01_AU02_product_side'] = pd.Series([au1_val_side_val * au2_val_side_val])
    feature_data_det['RE_AU01_AU02_sum_side'] = pd.Series([au1_val_side_val + au2_val_side_val])

    feature_data_det["side_indicator"] = pd.Series([0 if side.lower() == 'left' else 1])

    feature_dict_final_det = {k: v.iloc[0] for k, v in feature_data_det.items() if
                              isinstance(v, pd.Series) and not v.empty}
    # ... (final assembly into feature_vector as in lower_face_features.py) ...
    feature_vector = []
    for name in ordered_feature_names:
        value = feature_dict_final_det.get(name, 0.0)
        try:
            val_float = float(value)
            feature_vector.append(0.0 if np.isnan(val_float) else val_float)
        except (ValueError, TypeError):
            feature_vector.append(0.0)
    return feature_vector