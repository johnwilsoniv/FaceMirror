# synkinesis_features_base.py
#
# Shared scaffolding for per-type feature extractors. Provides the common
# pattern: training extractor returns a DataFrame; detection extractor wraps
# the same code on a single row and returns values ordered to match the
# saved features.list file produced at training time.

import logging
import os

import joblib
import numpy as np
import pandas as pd

from paralysis_utils import (
    _extract_base_au_features,
    _extract_coupling_features,
    _extract_paralysis_conditioned_features,
    _get_au_value_series,
    calculate_percent_diff,
    calculate_ratio,
)
from synkinesis_config import SYNKINESIS_CONFIG

logger = logging.getLogger(__name__)


def get_type_components(synk_type):
    cfg = SYNKINESIS_CONFIG[synk_type]
    return {
        'cfg': cfg,
        'name': cfg.get('name', synk_type),
        'actions': cfg.get('actions', []),
        'trigger_aus': cfg.get('trigger_aus', []),
        'coupled_aus': cfg.get('coupled_aus', []),
        'context_aus': cfg.get('context_aus', []),
        'interest_aus': cfg.get('interest_aus', []),
        'feature_cfg': cfg.get('feature_extraction', {}),
        'filenames': cfg.get('filenames', {}),
    }


def load_feature_list(features_path):
    if not features_path or not os.path.exists(features_path):
        return None
    if features_path.endswith('.list'):
        try:
            with open(features_path, 'r') as f:
                first_line = f.readline().strip()
                f.seek(0)
                if first_line and not first_line.startswith('\x80'):
                    return [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            pass
    return joblib.load(features_path)


def features_to_ordered_list(features_df, features_path):
    ordered = load_feature_list(features_path)
    if ordered is None:
        logger.error(f"Feature list not found at {features_path}; returning columns in extraction order.")
        return features_df.iloc[0].tolist()
    out = []
    for col in ordered:
        if col in features_df.columns:
            value = features_df[col].iloc[0]
        else:
            logger.debug(f"Feature '{col}' missing from extraction; defaulting to 0.")
            value = 0.0
        if pd.isna(value) or np.isinf(value):
            value = 0.0
        out.append(float(value))
    return out


def detection_wrapper(row_data, side, extract_func, features_path):
    if isinstance(row_data, pd.Series):
        df = pd.DataFrame([row_data.to_dict()])
    elif isinstance(row_data, dict):
        df = pd.DataFrame([row_data])
    else:
        df = row_data.copy()
    features_df = extract_func(df, side)
    if features_df is None:
        return None
    return features_to_ordered_list(features_df, features_path)


def extract_via_helpers(df, side, components):
    """Default extractor for coupling-pattern types: combines base AU features
    on the union of trigger+coupled AUs with the trigger/coupled overlay. When
    `feature_cfg['include_paralysis_conditioned']` is True, also adds
    paralysis-conditioned interaction features that distinguish true synkinesis
    from contralateral compensation."""
    actions = components['actions']
    trigger = components['trigger_aus']
    coupled = components['coupled_aus']
    context = components['context_aus']
    feature_cfg = components['feature_cfg']

    base_aus = sorted(set(trigger) | set(coupled))
    base_df = _extract_base_au_features(
        df, side, actions, base_aus, feature_cfg,
        zone_display_name=components['name'],
    )
    coupling_df = _extract_coupling_features(
        df, side, actions, trigger, coupled, feature_cfg,
        type_display_name=components['name'],
        context_aus=context,
        include_aggregates=feature_cfg.get('include_aggregates', True),
    )
    pieces = [base_df, coupling_df]
    if feature_cfg.get('include_paralysis_conditioned', False) and trigger and coupled:
        cond_df = _extract_paralysis_conditioned_features(
            df, side, actions, trigger, coupled, feature_cfg,
            type_display_name=components['name'],
        )
        if not cond_df.empty:
            pieces.append(cond_df)
    return pd.concat(pieces, axis=1)
