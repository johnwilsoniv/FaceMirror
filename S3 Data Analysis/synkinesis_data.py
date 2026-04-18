# synkinesis_data.py
#
# Data loading + binary label standardization for synkinesis training.
# Handles the Yes / None / Not Assessed columns in FPRS FP Key.csv and
# combines Left/Right sides into one stacked dataset.

import importlib
import logging
import os

import numpy as np
import pandas as pd

from synkinesis_config import EXCLUDED_PATIENTS, INPUT_FILES, SYNKINESIS_CONFIG

logger = logging.getLogger(__name__)


def standardize_label(val):
    """Yes → 1, None → 0, Not Assessed / blank → NaN (excluded from training)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().lower()
    if s in ('', 'nan', 'na', 'n/a', 'not assessed'):
        return np.nan
    if s in ('yes', 'y', 'true', '1', '1.0'):
        return 1
    if s in ('none', 'no', 'n', 'false', '0', '0.0'):
        return 0
    logger.debug(f"Unrecognized label '{val}' → NaN")
    return np.nan


def load_expert_key(expert_csv=None):
    expert_csv = expert_csv or INPUT_FILES['expert_key_csv']
    if not os.path.exists(expert_csv):
        raise FileNotFoundError(f"Expert key not found: {expert_csv}")
    df = pd.read_csv(expert_csv, dtype=str, keep_default_na=False, na_values=[''])
    if 'Patient' in df.columns:
        df = df.rename(columns={'Patient': 'Patient ID'})
    df['Patient ID'] = df['Patient ID'].astype(str).str.strip()
    if df['Patient ID'].duplicated().any():
        logger.warning("Duplicate Patient IDs in expert key — keeping first.")
        df = df.drop_duplicates(subset=['Patient ID'], keep='first')
    return df


def load_results(results_csv=None):
    results_csv = results_csv or INPUT_FILES['results_csv']
    if not os.path.exists(results_csv):
        raise FileNotFoundError(
            f"Results CSV not found: {results_csv}. Run the PyFaceAU analyzer first."
        )
    df = pd.read_csv(results_csv, low_memory=False)
    df['Patient ID'] = df['Patient ID'].astype(str).str.strip()
    return df


def prepare_type_data(synk_type, results_csv=None, expert_csv=None):
    """Returns (features_df, y, metadata_df) for binary training of `synk_type`.

    Pulls the per-type feature module, extracts Left and Right features for every
    patient with a non-Not-Assessed label on that side, stacks them into a single
    table with `Side` metadata, and returns aligned numeric features + binary y.
    """
    if synk_type not in SYNKINESIS_CONFIG:
        raise KeyError(f"Unknown synkinesis type: {synk_type}")
    cfg = SYNKINESIS_CONFIG[synk_type]
    name = cfg['name']

    expert_df = load_expert_key(expert_csv)
    results_df = load_results(results_csv)

    expert_left_col = cfg['expert_columns']['left']
    expert_right_col = cfg['expert_columns']['right']
    for col in (expert_left_col, expert_right_col):
        if col not in expert_df.columns:
            raise KeyError(f"[{name}] Expert column '{col}' missing from key.")

    expert_df = expert_df.assign(
        _y_left=expert_df[expert_left_col].apply(standardize_label),
        _y_right=expert_df[expert_right_col].apply(standardize_label),
    )

    keep_cols = ['Patient ID', '_y_left', '_y_right']
    merged = results_df.merge(expert_df[keep_cols], on='Patient ID', how='inner')

    if EXCLUDED_PATIENTS:
        excluded_present = sorted(set(merged['Patient ID']) & set(EXCLUDED_PATIENTS))
        if excluded_present:
            merged = merged[~merged['Patient ID'].isin(EXCLUDED_PATIENTS)].reset_index(drop=True)
            logger.info(f"[{name}] Excluded {len(excluded_present)} patient(s) per EXCLUDED_PATIENTS: {excluded_present}")

    logger.info(f"[{name}] Merged {len(merged)} rows from {merged['Patient ID'].nunique()} patients.")

    feature_module = importlib.import_module(f"{synk_type}_features")
    extract = feature_module.extract_features

    pieces = []
    for side, label_col in (('Left', '_y_left'), ('Right', '_y_right')):
        valid_mask = merged[label_col].notna()
        if not valid_mask.any():
            logger.warning(f"[{name}] No valid {side} labels.")
            continue
        sub = merged.loc[valid_mask].reset_index(drop=True)
        feats = extract(sub, side)
        if feats is None or feats.empty:
            logger.error(f"[{name}] {side} feature extraction returned empty.")
            continue
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feats['_y'] = sub[label_col].astype(int).values
        feats['_side'] = side
        feats['_patient_id'] = sub['Patient ID'].values
        pieces.append(feats)

    if not pieces:
        raise ValueError(f"[{name}] No valid samples on either side.")

    combined = pd.concat(pieces, ignore_index=True)
    metadata = combined[['_patient_id', '_side']].rename(
        columns={'_patient_id': 'Patient ID', '_side': 'Side'}
    )
    y = combined['_y'].astype(int).values
    features_df = combined.drop(columns=['_y', '_side', '_patient_id'])

    counts = pd.Series(y).value_counts().to_dict()
    logger.info(
        f"[{name}] Final dataset: features={features_df.shape}, "
        f"label_distribution={counts}"
    )
    return features_df, y, metadata
