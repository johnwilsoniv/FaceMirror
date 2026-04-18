# brow_cocked_features.py
#
# Brow Cocked: detects persistent unilateral elevation of the eyebrow that
# distinguishes itself from hypertonicity by tracking both baseline state
# (BL) and the difference between sides during voluntary brow raise (RE).
# Interest AUs are AU01 (Inner Brow Raiser) and AU02 (Outer Brow Raiser).
# Features include signed L-minus-R differences in addition to ratios and
# percent differences, since the directionality of the cock matters.

import logging

import pandas as pd

from paralysis_utils import (
    _get_au_value_series,
    calculate_percent_diff,
    calculate_ratio,
)
from synkinesis_features_base import detection_wrapper, get_type_components

SYNK_TYPE = 'brow_cocked'
logger = logging.getLogger(__name__)


def extract_features(df, side):
    components = get_type_components(SYNK_TYPE)
    cfg = components['cfg']
    feature_cfg = components['feature_cfg']
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_cap = feature_cfg.get('percent_diff_cap', 200.0)

    side_cap = side.capitalize() if isinstance(side, str) else 'Left'
    interest_aus = cfg.get('interest_aus', ['AU01_r', 'AU02_r'])

    feature_data = {}
    for au in interest_aus:
        bl_left = _get_au_value_series(df, 'BL', 'Left', au, use_normalized=False)
        bl_right = _get_au_value_series(df, 'BL', 'Right', au, use_normalized=False)
        bl_target = bl_left if side_cap == 'Left' else bl_right

        feature_data[f"BL_{au}_raw_target_side"] = bl_target
        feature_data[f"BL_Asym_Ratio_{au}"] = calculate_ratio(bl_left, bl_right, min_value=min_val)
        feature_data[f"BL_Asym_PercDiff_{au}"] = calculate_percent_diff(
            bl_left, bl_right, min_value=min_val, cap=perc_cap
        )
        feature_data[f"BL_Asym_Diff_LminusR_{au}"] = bl_left - bl_right

        re_left_norm = _get_au_value_series(df, 'RE', 'Left', au, use_normalized=True)
        re_right_norm = _get_au_value_series(df, 'RE', 'Right', au, use_normalized=True)
        re_target_norm = re_left_norm if side_cap == 'Left' else re_right_norm

        feature_data[f"RE_{au}_norm_target_side"] = re_target_norm
        feature_data[f"RE_Norm_Asym_Ratio_{au}"] = calculate_ratio(re_left_norm, re_right_norm, min_value=min_val)
        feature_data[f"RE_Norm_Asym_PercDiff_{au}"] = calculate_percent_diff(
            re_left_norm, re_right_norm, min_value=min_val, cap=perc_cap
        )
        feature_data[f"RE_Norm_Asym_Diff_LminusR_{au}"] = re_left_norm - re_right_norm

    return pd.DataFrame(feature_data, index=df.index)


def extract_features_for_detection(row_data, side):
    components = get_type_components(SYNK_TYPE)
    return detection_wrapper(
        row_data, side, extract_features,
        components['filenames'].get('feature_list'),
    )
