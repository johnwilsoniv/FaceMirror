# upper_face_features.py (Consistent Helpers + DataFrame Detection Logic v7)

import numpy as np
import pandas as pd
import logging
import os
import joblib
# import sys # Can comment out if not needed for debugging now

# Import config carefully, providing fallbacks
try:
    from upper_face_config import (
        FEATURE_CONFIG, UPPER_FACE_ACTIONS, UPPER_FACE_AUS, LOG_DIR,
        FEATURE_SELECTION, MODEL_DIR,
        CLASS_NAMES
    )
except ImportError:
    logging.warning("Could not import from upper_face_config. Using fallback definitions.")
    LOG_DIR = 'logs'; MODEL_DIR = 'models'
    FEATURE_CONFIG = {'actions': ['RE'], 'aus': ['AU01_r', 'AU02_r'], 'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
    UPPER_FACE_ACTIONS = FEATURE_CONFIG['actions']; UPPER_FACE_AUS = FEATURE_CONFIG['aus']
    FEATURE_SELECTION = {'enabled': False, 'top_n_features': 15, 'importance_file': os.path.join(MODEL_DIR, 'upper_face_feature_importance.csv')}
    CLASS_NAMES = {0: 'None', 1: 'Partial', 2: 'Complete'}

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- prepare_data function ---
def prepare_data(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """ Prepare dataset for ML model training for UPPER FACE. """
    logger.info("Loading datasets for Upper Face...")
    try:
        logger.info(f"Attempting to load results from: {results_file}")
        results_df = pd.read_csv(results_file, low_memory=False)
        logger.info(f"Attempting to load expert key from: {expert_file}")
        expert_df = pd.read_csv(expert_file, dtype=str)
        logger.info(f"Loaded {len(results_df)} results, {len(expert_df)} grades")
    except FileNotFoundError as e: logger.error(f"Error loading data: {e}."); raise
    except Exception as e: logger.error(f"Unexpected error loading data: {e}"); raise

    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        'Paralysis - Left Upper Face': 'Expert_Left_Upper_Face', # Correct column
        'Paralysis - Right Upper Face': 'Expert_Right_Upper_Face' # Correct column
        })

    # --- Target Variable Processing ---
    target_mapping = {'none': 0, 'partial': 1, 'complete': 2}
    def standardize_and_map(series):
        s_filled = series.fillna('none_placeholder')
        s_clean = s_filled.astype(str).str.lower().str.strip()
        replacements = {'no': 'none', 'n/a': 'none', 'mild': 'partial', 'moderate': 'partial', 'severe': 'complete', 'normal': 'none', 'none_placeholder': 'none', 'nan': 'none'}
        s_replaced = s_clean.replace(replacements)
        mapped = s_replaced.map(target_mapping)
        final_mapped = mapped.fillna(0)
        return final_mapped.astype(int)

    if 'Expert_Left_Upper_Face' in expert_df.columns:
        expert_df['Target_Left_Upper'] = standardize_and_map(expert_df['Expert_Left_Upper_Face'])
    else: logger.error("Missing 'Expert_Left_Upper_Face' column"); expert_df['Target_Left_Upper'] = 0
    if 'Expert_Right_Upper_Face' in expert_df.columns:
        expert_df['Target_Right_Upper'] = standardize_and_map(expert_df['Expert_Right_Upper_Face'])
    else: logger.error("Missing 'Expert_Right_Upper_Face' column"); expert_df['Target_Right_Upper'] = 0

    logger.info(f"Counts in expert_df['Target_Left_Upper'] AFTER mapping: \n{expert_df['Target_Left_Upper'].value_counts(dropna=False)}")
    logger.info(f"Counts in expert_df['Target_Right_Upper'] AFTER mapping: \n{expert_df['Target_Right_Upper'].value_counts(dropna=False)}")

    # Prepare for Merge
    results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
    expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()

    expert_cols_to_merge = ['Patient ID', 'Target_Left_Upper', 'Target_Right_Upper'] # Use correct target names
    try:
        if not all(col in expert_df.columns for col in expert_cols_to_merge): raise KeyError(f"Required columns missing from expert_df: {expert_cols_to_merge}")
        merged_df = pd.merge(results_df, expert_df[expert_cols_to_merge], on='Patient ID', how='inner', validate="many_to_one")
    except Exception as e: logger.error(f"Merge failed: {e}"); raise
    logger.info(f"Merged data: {len(merged_df)} patients")
    if merged_df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    logger.info(f"Counts in merged_df['Target_Left_Upper'] AFTER merge: \n{merged_df['Target_Left_Upper'].value_counts(dropna=False)}")
    logger.info(f"Counts in merged_df['Target_Right_Upper'] AFTER merge: \n{merged_df['Target_Right_Upper'].value_counts(dropna=False)}")

    # Feature Extraction
    logger.info("Extracting Upper Face features for Left side...")
    left_features_df = extract_features(merged_df, 'Left')
    logger.info("Extracting Upper Face features for Right side...")
    right_features_df = extract_features(merged_df, 'Right')

    # --- DEBUG LOGGING for first patient features ---
    if not left_features_df.empty and not merged_df.empty:
        if left_features_df.index.equals(merged_df.index):
             first_patient_id = merged_df.iloc[0]['Patient ID']
             if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"DEBUG (prepare_data): Upper Face Features for Patient {first_patient_id} Left:\n{left_features_df.iloc[0].tolist()}")
             else:
                 logger.info(f"INFO_DEBUG (prepare_data): Upper Face Features for Patient {first_patient_id} Left sample:\n{left_features_df.iloc[0].head().to_dict()}")
        else: logger.warning("DEBUG (prepare_data): Index mismatch.")
    # --- END DEBUG LOGGING ---

    # Combine features and targets
    if 'Target_Left_Upper' not in merged_df.columns or 'Target_Right_Upper' not in merged_df.columns:
         logger.error("Target columns missing in merged_df. Aborting."); return None, None
    left_targets = merged_df['Target_Left_Upper'].values
    right_targets = merged_df['Target_Right_Upper'].values
    targets = np.concatenate([left_targets, right_targets])

    unique_final, counts_final = np.unique(targets, return_counts=True)
    final_class_dist = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in unique_final], counts_final))
    logger.info(f"FINAL Upper Face Class distribution: {final_class_dist}")

    left_features_df['side_indicator'] = 0
    right_features_df['side_indicator'] = 1
    features = pd.concat([left_features_df, right_features_df], ignore_index=True)

    # Post-processing & Feature Selection
    features.replace([np.inf, -np.inf], np.nan, inplace=True); features = features.fillna(0)
    cols_to_drop = []
    for col in features.columns:
        try: features[col] = pd.to_numeric(features[col], errors='coerce')
        except Exception as e: logger.warning(f"Num convert fail {col}: {e}. Drop."); cols_to_drop.append(col)
    if cols_to_drop: logger.warning(f"Dropping non-numeric columns: {cols_to_drop}"); features = features.drop(columns=cols_to_drop)
    features = features.fillna(0)

    logger.info(f"Generated initial {features.shape[1]} features for upper face.")

    fs_enabled = isinstance(FEATURE_SELECTION, dict) and FEATURE_SELECTION.get('enabled', False)
    if fs_enabled:
        logger.info("Applying feature selection for upper face...")
        n_top_features = FEATURE_SELECTION.get('top_n_features', 15) # Default to 15 for upper face
        importance_file = FEATURE_SELECTION.get('importance_file')
        if not importance_file or not os.path.exists(importance_file): logger.warning(f"FS enabled, but file '{importance_file}' not found. Skipping.")
        else:
            try:
                importance_df = pd.read_csv(importance_file)
                if 'feature' not in importance_df.columns or importance_df.empty: logger.error("Importance file invalid. Skipping.");
                else:
                    top_feature_names = importance_df['feature'].head(n_top_features).tolist()
                    if 'side_indicator' in features.columns and 'side_indicator' not in top_feature_names: top_feature_names.append('side_indicator')
                    original_cols = features.columns.tolist(); cols_to_keep = [col for col in top_feature_names if col in original_cols]
                    missing_features = set(top_feature_names) - set(cols_to_keep)
                    if missing_features: logger.warning(f"Important features missing: {missing_features}")
                    if not cols_to_keep: logger.error("No features left. Skipping selection.")
                    else: logger.info(f"Selecting top {len(cols_to_keep)} features."); features = features[cols_to_keep]
            except Exception as e: logger.error(f"FS error: {e}. Skipping.", exc_info=True)
    else: logger.info("Upper face feature selection is disabled.")

    logger.info(f"Final upper face dataset: {len(features)} samples with {features.shape[1]} features.")
    if features.isnull().values.any(): logger.warning(f"NaNs in FINAL features. Columns: {features.columns[features.isna().any()].tolist()}"); features = features.fillna(0)

    final_feature_names = features.columns.tolist()
    try:
        if MODEL_DIR:
             os.makedirs(MODEL_DIR, exist_ok=True)
             feature_list_path = os.path.join(MODEL_DIR, 'upper_face_features.list') # upper_face list name
             joblib.dump(final_feature_names, feature_list_path)
             logger.info(f"Saved final {len(final_feature_names)} upper face feature names list to {feature_list_path}")
        else: logger.error("MODEL_DIR undefined. Cannot save feature list.")
    except Exception as e: logger.error(f"Failed save feature names list: {e}", exc_info=True)

    if 'targets' not in locals(): logger.error("Targets array missing."); return None, None
    return features, targets


# --- CORRECTED Helper Functions (Copied from lower_face_features.py) ---
def calculate_ratio(val1_series, val2_series):
    """ Calculates ratio of min to max, handles zeros. For pd.Series """
    try: min_val_config = FEATURE_CONFIG.get('min_value', 0.0001)
    except NameError: min_val_config = 0.0001
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0)
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0)
    min_vals_np = np.minimum(v1.to_numpy(), v2.to_numpy())
    max_vals_np = np.maximum(v1.to_numpy(), v2.to_numpy())
    ratio = np.ones_like(v1, dtype=float)
    mask_max_pos = max_vals_np > min_val_config
    mask_min_zero = min_vals_np <= min_val_config
    ratio[mask_max_pos & mask_min_zero] = 0.0
    valid_division_mask = mask_max_pos & ~mask_min_zero
    if np.any(valid_division_mask):
        ratio[valid_division_mask] = min_vals_np[valid_division_mask] / max_vals_np[valid_division_mask]
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(ratio, index=val1_series.index)

def calculate_percent_diff(val1_series, val2_series):
    """ Calculates capped percent difference. For pd.Series """
    try:
        min_val_config = FEATURE_CONFIG.get('min_value', 0.0001)
        percent_diff_cap = FEATURE_CONFIG.get('percent_diff_cap', 200.0)
    except NameError:
        min_val_config = 0.0001; percent_diff_cap = 200.0
    v1 = pd.to_numeric(val1_series, errors='coerce').fillna(0.0).to_numpy()
    v2 = pd.to_numeric(val2_series, errors='coerce').fillna(0.0).to_numpy()
    abs_diff = np.abs(v1 - v2); avg = (v1 + v2) / 2.0
    percent_diff = np.zeros_like(avg, dtype=float)
    mask_avg_pos = avg > min_val_config
    if np.any(mask_avg_pos):
        with np.errstate(divide='ignore', invalid='ignore'):
            division_result = abs_diff[mask_avg_pos] / avg[mask_avg_pos]
        percent_diff[mask_avg_pos] = division_result * 100.0
    mask_avg_zero_diff_pos = (avg <= min_val_config) & (abs_diff > min_val_config)
    percent_diff[mask_avg_zero_diff_pos] = percent_diff_cap
    percent_diff = np.clip(percent_diff, 0, percent_diff_cap)
    percent_diff = np.nan_to_num(percent_diff, nan=0.0, posinf=percent_diff_cap, neginf=0.0)
    return pd.Series(percent_diff, index=val1_series.index)
# --- END CORRECTED Helper Functions ---


# --- extract_features function (Training - Upper Face) ---
def extract_features(df, side):
    """ Extracts features for UPPER FACE training using dictionary method. """
    logger.debug(f"Extracting upper face features for {side} side (Training)...")
    feature_data = {}
    opposite_side = 'Right' if side == 'Left' else 'Left'
    local_feature_config = FEATURE_CONFIG if isinstance(FEATURE_CONFIG, dict) else {}
    local_actions = UPPER_FACE_ACTIONS if 'UPPER_FACE_ACTIONS' in globals() and UPPER_FACE_ACTIONS else ['RE']
    local_aus = UPPER_FACE_AUS if 'UPPER_FACE_AUS' in globals() and UPPER_FACE_AUS else ['AU01_r', 'AU02_r']
    use_normalized = local_feature_config.get('use_normalized', True)

    # 1. Basic AU & Asymmetry Features
    for action in local_actions: # Typically only 'RE'
        for au in local_aus:
            base_col_name = f"{action}_{au}"
            # --- Use consistent capitalization ---
            au_col_side = f"{action}_{side} {au}"
            au_norm_col_side = f"{au_col_side} (Normalized)"
            au_col_opp = f"{action}_{opposite_side} {au}"
            au_norm_col_opp = f"{au_col_opp} (Normalized)"
            # --- End capitalization ---

            raw_val_side_series = df.get(au_col_side, pd.Series(0.0, index=df.index))
            raw_val_opp_series = df.get(au_col_opp, pd.Series(0.0, index=df.index))
            raw_val_side = pd.to_numeric(raw_val_side_series, errors='coerce').fillna(0.0)
            raw_val_opp = pd.to_numeric(raw_val_opp_series, errors='coerce').fillna(0.0)

            if use_normalized:
                norm_val_side_series = df.get(au_norm_col_side, raw_val_side)
                norm_val_opp_series = df.get(au_norm_col_opp, raw_val_opp)
                norm_val_side = pd.to_numeric(norm_val_side_series, errors='coerce').fillna(raw_val_side)
                norm_val_opp = pd.to_numeric(norm_val_opp_series, errors='coerce').fillna(raw_val_opp)
            else: norm_val_side = raw_val_side; norm_val_opp = raw_val_opp

            feature_data[f"{base_col_name}_raw_side"] = raw_val_side
            feature_data[f"{base_col_name}_raw_opp"] = raw_val_opp
            feature_data[f"{base_col_name}_norm_side"] = norm_val_side
            feature_data[f"{base_col_name}_norm_opp"] = norm_val_opp
            feature_data[f"{base_col_name}_Asym_Diff"] = norm_val_side - norm_val_opp
            feature_data[f"{base_col_name}_Asym_Ratio"] = calculate_ratio(norm_val_side, norm_val_opp)
            feature_data[f"{base_col_name}_Asym_PercDiff"] = calculate_percent_diff(norm_val_side, norm_val_opp)
            feature_data[f"{base_col_name}_Is_Weaker_Side"] = (norm_val_side < norm_val_opp).astype(int)

    # 2. Interaction/Summary Features (Upper Face Specific)
    au1_ratio = feature_data.get('RE_AU01_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    au2_ratio = feature_data.get('RE_AU02_r_Asym_Ratio', pd.Series(1.0, index=df.index))
    feature_data['RE_avg_Asym_Ratio'] = (au1_ratio + au2_ratio) / 2.0

    au1_pd = feature_data.get('RE_AU01_r_Asym_PercDiff', pd.Series(0.0, index=df.index))
    au2_pd = feature_data.get('RE_AU02_r_Asym_PercDiff', pd.Series(0.0, index=df.index))
    feature_data['RE_avg_Asym_PercDiff'] = (au1_pd + au2_pd) / 2.0
    # Check if Series exist before concat/max
    pd_series_list = [s for s in [au1_pd, au2_pd] if isinstance(s, pd.Series)]
    if pd_series_list:
        feature_data['RE_max_Asym_PercDiff'] = pd.concat(pd_series_list, axis=1).max(axis=1)
    else:
        feature_data['RE_max_Asym_PercDiff'] = pd.Series(0.0, index=df.index)


    au1_norm = feature_data.get('RE_AU01_r_norm_side', pd.Series(0.0, index=df.index))
    au2_norm = feature_data.get('RE_AU02_r_norm_side', pd.Series(0.0, index=df.index))
    feature_data['RE_AU01_AU02_product_side'] = au1_norm * au2_norm
    feature_data['RE_AU01_AU02_sum_side'] = au1_norm + au2_norm

    # Create DataFrame
    features_df = pd.DataFrame(feature_data, index=df.index)
    # Final check for non-numeric types
    non_numeric_cols = features_df.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric cols in upper extract_features: {non_numeric_cols.tolist()}. Coercing.")
        for col in non_numeric_cols: features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
    logger.debug(f"Generated {features_df.shape[1]} upper face features for {side} (Training).")
    return features_df


# --- extract_features_for_detection (Upper Face - Using Pandas Internally) ---
def extract_features_for_detection(row_data, side, zone):
    """ Extract upper face features for detection using Pandas helpers. """
    try:
        from upper_face_config import MODEL_DIR, FEATURE_SELECTION, FEATURE_CONFIG, UPPER_FACE_ACTIONS, UPPER_FACE_AUS
        local_logger = logging.getLogger(__name__)
    except ImportError:
        logging.error("Config import failed in upper_face extract_features_for_detection.")
        local_logger = logging
        MODEL_DIR = 'models'; FEATURE_SELECTION = {'enabled': False}
        FEATURE_CONFIG = {'actions': ['RE'], 'aus': ['AU01_r', 'AU02_r'], 'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0}
        UPPER_FACE_ACTIONS = FEATURE_CONFIG['actions']; UPPER_FACE_AUS = FEATURE_CONFIG['aus']

    if not isinstance(row_data, (pd.Series, dict)): local_logger.error(f"row_data type {type(row_data)} invalid"); return None
    try: df_single_row = pd.DataFrame([row_data]); df_single_row.index = [0]
    except Exception as e: local_logger.error(f"DataFrame conversion failed: {e}"); return None

    local_feature_config = FEATURE_CONFIG if isinstance(FEATURE_CONFIG, dict) else {}
    local_actions = UPPER_FACE_ACTIONS if 'UPPER_FACE_ACTIONS' in globals() and UPPER_FACE_ACTIONS else ['RE']
    local_aus = UPPER_FACE_AUS if 'UPPER_FACE_AUS' in globals() and UPPER_FACE_AUS else ['AU01_r', 'AU02_r']
    if not local_feature_config or not local_actions or not local_aus: local_logger.error("Upper face config error."); return None

    local_logger.debug(f"Extracting upper face detection features for {side} {zone} using DataFrame method...")
    feature_data = {}
    # --- Use consistent capitalization ---
    side_label = side.capitalize()
    opposite_side_label = 'Right' if side_label == 'Left' else 'Left'
    # --- End capitalization ---
    use_normalized = local_feature_config.get('use_normalized', True)

    # 1. Basic AU & Asymmetry Features
    for action in local_actions:
        for au in local_aus:
            base_col_name = f"{action}_{au}"
            au_col_side = f"{action}_{side_label} {au}"; au_norm_col_side = f"{au_col_side} (Normalized)"
            au_col_opp = f"{action}_{opposite_side_label} {au}"; au_norm_col_opp = f"{au_col_opp} (Normalized)"

            raw_val_side_series = df_single_row.get(au_col_side, pd.Series(0.0, index=df_single_row.index))
            raw_val_opp_series = df_single_row.get(au_col_opp, pd.Series(0.0, index=df_single_row.index))
            raw_val_side = pd.to_numeric(raw_val_side_series, errors='coerce').fillna(0.0)
            raw_val_opp = pd.to_numeric(raw_val_opp_series, errors='coerce').fillna(0.0)

            if use_normalized:
                norm_val_side_series = df_single_row.get(au_norm_col_side, raw_val_side)
                norm_val_opp_series = df_single_row.get(au_norm_col_opp, raw_val_opp)
                norm_val_side = pd.to_numeric(norm_val_side_series, errors='coerce').fillna(raw_val_side)
                norm_val_opp = pd.to_numeric(norm_val_opp_series, errors='coerce').fillna(raw_val_opp)
            else: norm_val_side = raw_val_side; norm_val_opp = raw_val_opp

            feature_data[f"{base_col_name}_raw_side"] = raw_val_side; feature_data[f"{base_col_name}_raw_opp"] = raw_val_opp
            feature_data[f"{base_col_name}_norm_side"] = norm_val_side; feature_data[f"{base_col_name}_norm_opp"] = norm_val_opp
            feature_data[f"{base_col_name}_Asym_Diff"] = norm_val_side - norm_val_opp
            feature_data[f"{base_col_name}_Asym_Ratio"] = calculate_ratio(norm_val_side, norm_val_opp)
            feature_data[f"{base_col_name}_Asym_PercDiff"] = calculate_percent_diff(norm_val_side, norm_val_opp)
            feature_data[f"{base_col_name}_Is_Weaker_Side"] = (norm_val_side < norm_val_opp).astype(int)

    # 2. Interaction/Summary Features (Upper Face Specific)
    au1_ratio = feature_data.get('RE_AU01_r_Asym_Ratio', pd.Series(1.0, index=df_single_row.index))
    au2_ratio = feature_data.get('RE_AU02_r_Asym_Ratio', pd.Series(1.0, index=df_single_row.index))
    feature_data['RE_avg_Asym_Ratio'] = (au1_ratio + au2_ratio) / 2.0

    au1_pd = feature_data.get('RE_AU01_r_Asym_PercDiff', pd.Series(0.0, index=df_single_row.index))
    au2_pd = feature_data.get('RE_AU02_r_Asym_PercDiff', pd.Series(0.0, index=df_single_row.index))
    feature_data['RE_avg_Asym_PercDiff'] = (au1_pd + au2_pd) / 2.0
    pd_series_list = [s for s in [au1_pd, au2_pd] if isinstance(s, pd.Series)]
    if pd_series_list: feature_data['RE_max_Asym_PercDiff'] = pd.concat(pd_series_list, axis=1).max(axis=1)
    else: feature_data['RE_max_Asym_PercDiff'] = pd.Series(0.0, index=df_single_row.index)

    au1_norm = feature_data.get('RE_AU01_r_norm_side', pd.Series(0.0, index=df_single_row.index))
    au2_norm = feature_data.get('RE_AU02_r_norm_side', pd.Series(0.0, index=df_single_row.index))
    feature_data['RE_AU01_AU02_product_side'] = au1_norm * au2_norm
    feature_data['RE_AU01_AU02_sum_side'] = au1_norm + au2_norm

    # --- Use side_label (capitalized) ---
    feature_data["side_indicator"] = pd.Series([0 if side_label.lower() == 'left' else 1], index=df_single_row.index)

    # --- Convert dict of Series to dict of single values ---
    feature_dict_final = {k: v.iloc[0] for k, v in feature_data.items() if isinstance(v, pd.Series) and not v.empty}

    # Load feature list
    local_model_dir = MODEL_DIR if 'MODEL_DIR' in globals() and MODEL_DIR else 'models'
    feature_names_path = os.path.join(local_model_dir, 'upper_face_features.list') # upper face list
    if not os.path.exists(feature_names_path): local_logger.error(f"Upper face feature list not found: {feature_names_path}."); return None
    try: ordered_feature_names = joblib.load(feature_names_path)
    except Exception as e: local_logger.error(f"Failed load upper face feature list: {e}"); return None

    # Build final list
    feature_list = []; missing_in_dict = []
    for name in ordered_feature_names:
        value = feature_dict_final.get(name, 0.0)
        try: feature_list.append(float(value))
        except (ValueError, TypeError): feature_list.append(0.0)
        if name not in feature_dict_final: missing_in_dict.append(name)

    if missing_in_dict: local_logger.warning(f"Detection: {len(missing_in_dict)} expected upper face features missing: {missing_in_dict[:5]}... Defaulting to 0.")
    if len(feature_list) != len(ordered_feature_names): local_logger.error(f"CRITICAL MISMATCH: Upper face list length {len(feature_list)} != expected {len(ordered_feature_names)}."); return None

    local_logger.debug(f"Generated {len(feature_list)} upper face detection features for {side_label}.")
    return feature_list


# --- process_targets function (included for completeness) ---
def process_targets(target_series):
    """ Maps expert labels to numerical targets. """
    target_mapping = {'none': 0, 'partial': 1, 'complete': 2}
    s_filled_none = target_series.fillna('none_placeholder')
    s_clean = s_filled_none.astype(str).str.lower().str.strip()
    replacements = {'no': 'none', 'n/a': 'none', 'mild': 'partial', 'moderate': 'partial', 'severe': 'complete', 'normal': 'none', 'none_placeholder': 'none', 'nan': 'none'}
    s_replaced = s_clean.replace(replacements)
    mapped = s_replaced.map(target_mapping)
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int).values