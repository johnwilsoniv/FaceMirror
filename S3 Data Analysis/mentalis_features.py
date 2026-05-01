# mentalis_features.py
#
# Mentalis: detects parasitic chin raise (AU17) co-activated with intended
# smile (AU12) or lip-corner depression (AU15) across many volitional
# actions. Modeled as trigger=[AU12, AU15] / coupled=[AU17] so the standard
# helper produces per-action coupled value + cross-AU ratios + aggregates.
# Bilateral asymmetry on AU17 during BS and SE is added explicitly since
# unilateral hyperactivity of the chin is a hallmark.

import logging

import pandas as pd

from paralysis_utils import calculate_percent_diff, calculate_ratio
from synkinesis_features_base import (
    detection_wrapper,
    extract_via_helpers,
    get_type_components,
)

SYNK_TYPE = 'mentalis'
logger = logging.getLogger(__name__)


def extract_features(df, side):
    components = get_type_components(SYNK_TYPE)
    base_df = extract_via_helpers(df, side, components)

    cfg = components['cfg']
    feature_cfg = components['feature_cfg']
    min_val = feature_cfg.get('min_value', 0.0001)
    perc_cap = feature_cfg.get('percent_diff_cap', 200.0)
    side_cap = side.capitalize() if isinstance(side, str) else 'Left'
    opposite = 'Right' if side_cap == 'Left' else 'Left'

    augment = {}
    asymmetry_actions = cfg.get('asymmetry_actions', ['BS', 'SE'])
    for action in asymmetry_actions:
        for au in components['coupled_aus']:
            from paralysis_utils import _get_au_value_series
            v_side = _get_au_value_series(df, action, side_cap, au, use_normalized=feature_cfg.get('use_normalized', True))
            v_opp = _get_au_value_series(df, action, opposite, au, use_normalized=feature_cfg.get('use_normalized', True))
            augment[f"{action}_Asym_Ratio_{au}_explicit"] = calculate_ratio(v_side, v_opp, min_value=min_val)
            augment[f"{action}_Asym_PercDiff_{au}_explicit"] = calculate_percent_diff(
                v_side, v_opp, min_value=min_val, cap=perc_cap
            )

    if augment:
        return pd.concat([base_df, pd.DataFrame(augment, index=base_df.index)], axis=1)
    return base_df


def extract_features_for_detection(row_data, side):
    components = get_type_components(SYNK_TYPE)
    return detection_wrapper(
        row_data, side, extract_features,
        components['filenames'].get('feature_list'),
    )
