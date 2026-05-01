# snarl_smile_features.py
#
# Snarl-Smile: detects upper lip retraction (AU10) and lip corner depression
# (AU14, AU15) parasitically activated during volitional smiling (AU12).
# BS-only single-action pattern — the historical V7 used 15 features. This
# wrapper relies on the standard helpers to produce the equivalent feature
# set, plus a small set of summary maxima used historically.

import logging

import pandas as pd

from synkinesis_features_base import (
    detection_wrapper,
    extract_via_helpers,
    get_type_components,
)

SYNK_TYPE = 'snarl_smile'
logger = logging.getLogger(__name__)


def extract_features(df, side):
    components = get_type_components(SYNK_TYPE)
    base_df = extract_via_helpers(df, side, components)

    augment = {}
    for au in components['coupled_aus']:
        coup_col = f"BS_{au}_coup_norm"
        if coup_col in base_df.columns:
            augment[f"Max_BS_{au}_Norm"] = base_df[coup_col]  # single-action max == value

    if 'BS_Ratio_AU15_r_vs_AU10_r' not in base_df.columns:
        from paralysis_utils import calculate_ratio
        min_val = components['feature_cfg'].get('min_value', 0.0001)
        au10 = base_df.get('BS_AU10_r_coup_norm')
        au15 = base_df.get('BS_AU15_r_coup_norm')
        if au10 is not None and au15 is not None:
            augment['BS_Ratio_AU15_r_vs_AU10_r'] = calculate_ratio(au15, au10, min_value=min_val)
            augment['Max_BS_Ratio_AU15_vs_AU10'] = augment['BS_Ratio_AU15_r_vs_AU10_r']

    if augment:
        return pd.concat([base_df, pd.DataFrame(augment, index=base_df.index)], axis=1)
    return base_df


def extract_features_for_detection(row_data, side):
    components = get_type_components(SYNK_TYPE)
    return detection_wrapper(
        row_data, side, extract_features,
        components['filenames'].get('feature_list'),
    )
