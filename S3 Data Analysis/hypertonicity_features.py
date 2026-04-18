# hypertonicity_features.py
#
# Hypertonicity: detects elevated baseline (resting) muscle tone on the
# affected side. Computed entirely from the BL action since "resting" is the
# defining state, augmented with BL-vs-BS reference ratios that distinguish
# true elevated tone from generally weak movement amplitude. Interest AUs
# are AU12 (smile pull) and AU14 (dimpler) — the two AUs most consistently
# elevated at rest in post-paralytic synkinesis.

import logging

import pandas as pd

from paralysis_utils import (
    _get_au_value_series,
    calculate_percent_diff,
    calculate_ratio,
)
from synkinesis_features_base import detection_wrapper, get_type_components

SYNK_TYPE = 'hypertonicity'
logger = logging.getLogger(__name__)


def extract_features(df, side):
    components = get_type_components(SYNK_TYPE)
    cfg = components['cfg']
    feature_cfg = components['feature_cfg']
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_cap = feature_cfg.get('percent_diff_cap', 200.0)
    use_norm = feature_cfg.get('use_normalized', True)

    side_cap = side.capitalize() if isinstance(side, str) else 'Left'
    opposite = 'Right' if side_cap == 'Left' else 'Left'

    interest_aus = cfg.get('interest_aus', ['AU12_r', 'AU14_r'])
    reference_action = cfg.get('reference_action', 'BS')

    feature_data = {}
    for au in interest_aus:
        bl_side = _get_au_value_series(df, 'BL', side_cap, au, use_normalized=False)
        bl_opp = _get_au_value_series(df, 'BL', opposite, au, use_normalized=False)
        bs_side_norm = _get_au_value_series(df, reference_action, side_cap, au, use_normalized=use_norm)
        bs_side_raw = _get_au_value_series(df, reference_action, side_cap, au, use_normalized=False)

        feature_data[f"BL_{au}_raw"] = bl_side
        feature_data[f"BL_Asym_Ratio_{au}_raw"] = calculate_ratio(bl_side, bl_opp, min_value=min_val)
        feature_data[f"BL_Asym_PercDiff_{au}_raw"] = calculate_percent_diff(
            bl_side, bl_opp, min_value=min_val, cap=perc_cap
        )
        feature_data[f"{reference_action}_Norm_{au}"] = bs_side_norm
        feature_data[f"Ratio_BLraw_vs_{reference_action}raw_{au}"] = calculate_ratio(
            bl_side, bs_side_raw, min_value=min_val
        )

    return pd.DataFrame(feature_data, index=df.index)


def extract_features_for_detection(row_data, side):
    components = get_type_components(SYNK_TYPE)
    return detection_wrapper(
        row_data, side, extract_features,
        components['filenames'].get('feature_list'),
    )
