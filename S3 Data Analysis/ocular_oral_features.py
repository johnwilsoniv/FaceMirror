# ocular_oral_features.py
#
# Ocular-Oral coupling: detects mouth movements parasitically driven by eye
# actions (closing eyes triggering smile/lip activation). Trigger AUs are
# brow/blink (AU01, AU02, AU45); coupled AUs are mouth (AU12, AU25, AU14)
# observed across ET, ES, RE, BK actions.

import logging

import pandas as pd

from synkinesis_features_base import (
    detection_wrapper,
    extract_via_helpers,
    get_type_components,
)

SYNK_TYPE = 'ocular_oral'
logger = logging.getLogger(__name__)


def extract_features(df, side):
    components = get_type_components(SYNK_TYPE)
    base_df = extract_via_helpers(df, side, components)

    actions = components['actions']
    coupled = components['coupled_aus']
    trigger = components['trigger_aus']

    augment = {}
    for action in actions:
        coup_cols = [f"{action}_{au}_coup_norm" for au in coupled if f"{action}_{au}_coup_norm" in base_df.columns]
        trig_cols = [f"{action}_{au}_trig_norm" for au in trigger if f"{action}_{au}_trig_norm" in base_df.columns]
        if coup_cols:
            stacked_coup = base_df[coup_cols]
            augment[f"{action}_Avg_Coupled_Norm"] = stacked_coup.mean(axis=1)
            augment[f"{action}_Max_Coupled_Norm"] = stacked_coup.max(axis=1)
        if trig_cols:
            stacked_trig = base_df[trig_cols]
            augment[f"{action}_Avg_Trigger_Norm"] = stacked_trig.mean(axis=1)
        if coup_cols and trig_cols:
            from paralysis_utils import calculate_ratio
            min_val = components['feature_cfg'].get('min_value', 0.0001)
            augment[f"{action}_Ratio_AvgCoup_vs_AvgTrig"] = calculate_ratio(
                augment[f"{action}_Avg_Coupled_Norm"],
                augment[f"{action}_Avg_Trigger_Norm"],
                min_value=min_val,
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
