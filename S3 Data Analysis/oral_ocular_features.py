# oral_ocular_features.py
#
# Oral-Ocular coupling: detects eye narrowing parasitically driven by mouth
# actions (smiling triggering eye squint). Trigger AUs are mouth (AU12, AU25);
# coupled AUs are cheek/blink (AU06, AU45) observed across BS, SS, SO, SE,
# PL, LT actions. BS gets extra bilateral asymmetry summary features since
# it is the most reliably elicited oral action.

import logging

import pandas as pd

from synkinesis_features_base import (
    detection_wrapper,
    extract_via_helpers,
    get_type_components,
)

SYNK_TYPE = 'oral_ocular'
logger = logging.getLogger(__name__)


def extract_features(df, side):
    components = get_type_components(SYNK_TYPE)
    base_df = extract_via_helpers(df, side, components)

    coupled = components['coupled_aus']
    augment = {}

    bs_asym_ratio_cols = [f"BS_Asym_Ratio_{au}_coup" for au in coupled if f"BS_Asym_Ratio_{au}_coup" in base_df.columns]
    bs_asym_pd_cols = [f"BS_Asym_PercDiff_{au}_coup" for au in coupled if f"BS_Asym_PercDiff_{au}_coup" in base_df.columns]
    if bs_asym_ratio_cols:
        augment['BS_Avg_Coupled_Asym_Ratio'] = base_df[bs_asym_ratio_cols].mean(axis=1)
    if bs_asym_pd_cols:
        augment['BS_Max_Coupled_Asym_PercDiff'] = base_df[bs_asym_pd_cols].max(axis=1)

    if augment:
        return pd.concat([base_df, pd.DataFrame(augment, index=base_df.index)], axis=1)
    return base_df


def extract_features_for_detection(row_data, side):
    components = get_type_components(SYNK_TYPE)
    return detection_wrapper(
        row_data, side, extract_features,
        components['filenames'].get('feature_list'),
    )
