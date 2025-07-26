# paralysis_performance.py (v7.1 - Correct extraction of probabilities from detector details and added detailed logging)

import pandas as pd
import numpy as np
import logging
import os
import sys  # Added for sys.stdout in logging
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

try:
    from paralysis_config import ZONE_CONFIG, LOGGING_CONFIG, INPUT_FILES, ANALYSIS_DIR

    # Force DEBUG for this run by modifying the loaded config
    LOGGING_CONFIG['level'] = 'INFO'
except ImportError as e:
    print(f"CRITICAL: Failed to import from paralysis_config.py - {e}")
    ZONE_CONFIG = {}
    LOGGING_CONFIG = {'level': 'DEBUG', 'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'}
    INPUT_FILES = {}
    ANALYSIS_DIR = 'analysis_results'

try:
    from paralysis_detector import ParalysisDetector
except ImportError as e:
    print(f"CRITICAL: Failed to import ParalysisDetector - {e}")
    ParalysisDetector = None

try:
    from paralysis_utils import (
        standardize_paralysis_labels,
        PARALYSIS_MAP,
        visualize_confusion_matrix,
        perform_error_analysis,
        analyze_critical_errors,
        analyze_partial_errors
    )

    UTILS_LOADED = True
except ImportError as e:
    print(f"CRITICAL: Failed to import from paralysis_utils.py - {e}")
    UTILS_LOADED = False


    def standardize_paralysis_labels(val):
        return str(val)


    PARALYSIS_MAP = {0: 'None', 1: 'Partial', 2: 'Complete'}


    def visualize_confusion_matrix(*args, **kwargs):
        pass


    def perform_error_analysis(*args, **kwargs):
        pass


    def analyze_critical_errors(*args, **kwargs):
        pass


    def analyze_partial_errors(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)
FITZPATRICK_COL_NAME = "Fitzpatrick"


def _setup_logging(log_file, level_str):
    log_level_val = logging.DEBUG  # Force DEBUG for file handler for this troubleshooting

    log_format_str = LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    log_formatter = logging.Formatter(log_format_str)

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level_val)
    file_handler.setFormatter(log_formatter)

    console_handler_level_config = LOGGING_CONFIG.get('level', 'DEBUG').upper()  # Use config level for console
    console_handler_level = getattr(logging, console_handler_level_config, logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_handler_level)
    console_handler.setFormatter(log_formatter)

    root_logger.setLevel(min(log_level_val, console_handler_level))  # Root logger at the most verbose of its handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info(
        f"Logging setup complete. Root Level: {logging.getLevelName(root_logger.getEffectiveLevel())}. File Handler Level: {logging.getLevelName(file_handler.level)}. Console Handler Level: {logging.getLevelName(console_handler.level)}. Log File: {log_file}")


def analyze_accuracy_by_fitzpatrick_group(df, expert_col_num, pred_col_num, fitzpatrick_raw_col=FITZPATRICK_COL_NAME,
                                          finding_name="Finding", logger_obj=None):
    if logger_obj is None: logger_obj = logging.getLogger(__name__)
    if fitzpatrick_raw_col not in df.columns:
        logger_obj.warning(f"[{finding_name}] Fitz col '{fitzpatrick_raw_col}' not found. Skipping Fitz analysis.")
        return pd.DataFrame()
    if not all(col in df.columns for col in [expert_col_num, pred_col_num]):
        logger_obj.error(f"[{finding_name}] Missing expert/pred cols for Fitz analysis. Skipping.")
        return pd.DataFrame()
    analysis_df = df.copy()
    logger_obj.info(f"--- [{finding_name}] Fitzpatrick Sub-Analysis ---")
    analysis_df['Fitzpatrick_Numeric'] = pd.to_numeric(analysis_df[fitzpatrick_raw_col], errors='coerce')
    analysis_df['Fitzpatrick_Group'] = pd.Series(index=analysis_df.index, dtype='object')
    analysis_df.loc[analysis_df['Fitzpatrick_Numeric'].isin([1, 2, 3, 4]), 'Fitzpatrick_Group'] = 'Fitzpatrick 1-4'
    analysis_df.loc[analysis_df['Fitzpatrick_Numeric'].isin([5, 6]), 'Fitzpatrick_Group'] = 'Fitzpatrick 5-6'
    analysis_df.dropna(subset=['Fitzpatrick_Group', expert_col_num, pred_col_num], inplace=True)
    if analysis_df.empty:
        logger_obj.warning(f"[{finding_name}] No data for Fitz groups. Skipping accuracy.")
        return pd.DataFrame()
    analysis_df['is_correct'] = (analysis_df[expert_col_num].astype(int) == analysis_df[pred_col_num].astype(int))
    accuracy_by_group = analysis_df.groupby('Fitzpatrick_Group')['is_correct'].agg(['mean', 'count', 'sum'])
    accuracy_by_group.rename(columns={'mean': 'accuracy', 'count': 'total_samples', 'sum': 'correct_predictions'},
                             inplace=True)
    if not accuracy_by_group.empty:
        for group_name_iter, row_data in accuracy_by_group.iterrows():
            logger_obj.info(
                f"  {group_name_iter:<18}: Acc={row_data['accuracy']:.3f} ({int(row_data['correct_predictions'])}/{int(row_data['total_samples'])})")
    group1_name, group2_name = 'Fitzpatrick 1-4', 'Fitzpatrick 5-6'
    if group1_name in accuracy_by_group.index and group2_name in accuracy_by_group.index:
        g1_correct = int(accuracy_by_group.loc[group1_name, 'correct_predictions']);
        g1_total = int(accuracy_by_group.loc[group1_name, 'total_samples'])
        g2_correct = int(accuracy_by_group.loc[group2_name, 'correct_predictions']);
        g2_total = int(accuracy_by_group.loc[group2_name, 'total_samples'])
        if g1_total > 0 and g2_total > 0:
            table = np.array([[g1_correct, g1_total - g1_correct], [g2_correct, g2_total - g2_correct]])
            try:
                _, p_value = stats.fisher_exact(table)  # type: ignore
                logger_obj.info(f"  Fisher's Exact p-value for Fitz group acc diff: {p_value:.4f}")
            except ValueError as e:
                logger_obj.error(f"  Fisher's Exact Test error: {e}")
    logger_obj.info(f"--- End Fitzpatrick Sub-Analysis for {finding_name} ---")
    return accuracy_by_group


def analyze_zone(data, zone_name_display, pred_left_col, expert_left_col, pred_right_col, expert_right_col, output_dir,
                 config):
    logger.info(f"Analyzing {zone_name_display} performance using ML Predictions")
    logger.info(f"Using class map: {PARALYSIS_MAP}")
    filenames = config.get('filenames', {})
    safe_zone_name_part = zone_name_display.lower().replace(' ', '_').replace('-', '_')
    critical_error_file_cfg = filenames.get('critical_errors_report')
    partial_error_file_cfg = filenames.get('partial_errors_report')
    critical_error_path = os.path.join(output_dir, os.path.basename(
        critical_error_file_cfg)) if critical_error_file_cfg else os.path.join(output_dir,
                                                                               f"{safe_zone_name_part}_critical_errors.txt")
    partial_error_path = os.path.join(output_dir, os.path.basename(
        partial_error_file_cfg)) if partial_error_file_cfg else os.path.join(output_dir,
                                                                             f"{safe_zone_name_part}_partial_errors.txt")
    errors_filename_base_for_perform = os.path.join(output_dir, f"{safe_zone_name_part}_errors_ML_Preds")
    os.makedirs(output_dir, exist_ok=True)

    analysis_df_cols_select = ['Patient ID', pred_left_col, expert_left_col, pred_right_col, expert_right_col]
    if FITZPATRICK_COL_NAME in data.columns: analysis_df_cols_select.append(FITZPATRICK_COL_NAME)
    for side_suffix in ['Left', 'Right']:
        for class_idx in sorted(PARALYSIS_MAP.keys()):
            class_name_str = PARALYSIS_MAP[class_idx]
            prob_col_name = f'Prob_{class_name_str.replace(" ", "_")}_{side_suffix}'
            if prob_col_name in data.columns:
                if prob_col_name not in analysis_df_cols_select: analysis_df_cols_select.append(prob_col_name)
    analysis_df = data[[col for col in analysis_df_cols_select if col in data.columns]].copy()
    analysis_df['Expert_Std_Left'] = analysis_df[expert_left_col].apply(standardize_paralysis_labels)
    analysis_df['Expert_Std_Right'] = analysis_df[expert_right_col].apply(standardize_paralysis_labels)
    analysis_df['Pred_Std_Left'] = analysis_df[pred_left_col]
    analysis_df['Pred_Std_Right'] = analysis_df[pred_right_col]

    left_reshape_cols = ['Patient ID', 'Pred_Std_Left', 'Expert_Std_Left']
    rename_map_left = {'Pred_Std_Left': 'Prediction', 'Expert_Std_Left': 'Expert'}
    for class_idx in sorted(PARALYSIS_MAP.keys()):
        class_name_str = PARALYSIS_MAP[class_idx];
        safe_name = class_name_str.replace(" ", "_")
        side_specific_prob_col = f'Prob_{safe_name}_Left'
        if side_specific_prob_col in analysis_df.columns:
            left_reshape_cols.append(side_specific_prob_col)
            rename_map_left[side_specific_prob_col] = f'Prob_{safe_name}'
    if FITZPATRICK_COL_NAME in analysis_df.columns: left_reshape_cols.append(FITZPATRICK_COL_NAME)
    left_analysis = analysis_df[[col for col in left_reshape_cols if col in analysis_df.columns]].copy()
    left_analysis.rename(columns=rename_map_left, inplace=True);
    left_analysis['Side'] = 'Left'

    right_reshape_cols = ['Patient ID', 'Pred_Std_Right', 'Expert_Std_Right']
    rename_map_right = {'Pred_Std_Right': 'Prediction', 'Expert_Std_Right': 'Expert'}
    for class_idx in sorted(PARALYSIS_MAP.keys()):
        class_name_str = PARALYSIS_MAP[class_idx];
        safe_name = class_name_str.replace(" ", "_")
        side_specific_prob_col = f'Prob_{safe_name}_Right'
        if side_specific_prob_col in analysis_df.columns:
            right_reshape_cols.append(side_specific_prob_col)
            rename_map_right[side_specific_prob_col] = f'Prob_{safe_name}'
    if FITZPATRICK_COL_NAME in analysis_df.columns: right_reshape_cols.append(FITZPATRICK_COL_NAME)
    right_analysis = analysis_df[[col for col in right_reshape_cols if col in analysis_df.columns]].copy()
    right_analysis.rename(columns=rename_map_right, inplace=True);
    right_analysis['Side'] = 'Right'

    zone_df = pd.concat([left_analysis, right_analysis], ignore_index=True)
    logger.debug(f"[{zone_name_display}] Reshaped zone_df columns: {zone_df.columns.tolist()}")
    prob_none_col_check = f'Prob_{PARALYSIS_MAP[0].replace(" ", "_")}'  # e.g. Prob_None
    if prob_none_col_check in zone_df.columns:  # Check if the first generic prob col exists
        cols_to_show = ['Patient ID', 'Side'] + [f'Prob_{PARALYSIS_MAP[ci].replace(" ", "_")}' for ci in
                                                 sorted(PARALYSIS_MAP.keys()) if
                                                 f'Prob_{PARALYSIS_MAP[ci].replace(" ", "_")}' in zone_df.columns]
        logger.debug(
            f"[{zone_name_display}] zone_df head with generic probas:\n{zone_df[cols_to_show].head().to_string()}")
    else:
        logger.warning(
            f"[{zone_name_display}] Generic probability columns (e.g., '{prob_none_col_check}') not found in reshaped zone_df. Columns present: {zone_df.columns.tolist()}")

    valid_rows_mask = (zone_df['Expert'] != 'NA') & (zone_df['Prediction'] != 'Error') & (zone_df['Prediction'].notna())
    zone_df_valid = zone_df[valid_rows_mask].copy()
    logger.info(
        f"[{zone_name_display}] Filtered out {len(zone_df) - len(zone_df_valid)} rows with 'NA' expert labels or 'Error'/'NaN' predictions.")
    logger.info(f"[{zone_name_display}] Shape AFTER filtering NA/Error for metrics: {zone_df_valid.shape}")
    if zone_df_valid.empty: logger.error(f"No valid data for {zone_name_display}. Skipping metrics."); return

    label_to_num_map = {v: k for k, v in PARALYSIS_MAP.items()}
    zone_df_valid['Expert_Num'] = zone_df_valid['Expert'].map(label_to_num_map)
    zone_df_valid['Prediction_Num'] = zone_df_valid['Prediction'].map(label_to_num_map)
    if zone_df_valid['Expert_Num'].isnull().any() or zone_df_valid['Prediction_Num'].isnull().any():
        initial_len = len(zone_df_valid)
        zone_df_valid.dropna(subset=['Expert_Num', 'Prediction_Num'], inplace=True)
        logger.warning(
            f"[{zone_name_display}] Dropped {initial_len - len(zone_df_valid)} rows due to label mapping failure to numeric.")
        if zone_df_valid.empty: logger.error(f"No valid mapped labels for {zone_name_display}."); return
    zone_df_valid['Expert_Num'] = zone_df_valid['Expert_Num'].astype(int)
    zone_df_valid['Prediction_Num'] = zone_df_valid['Prediction_Num'].astype(int)

    # Classification Reports and Confusion Matrices
    all_possible_labels_num = sorted(PARALYSIS_MAP.keys())
    all_possible_labels_str = [PARALYSIS_MAP.get(l) for l in all_possible_labels_num]
    y_true_combined = zone_df_valid['Expert_Num']
    y_pred_combined = zone_df_valid['Prediction_Num']
    if not y_true_combined.empty:
        report_labels_combined_num = sorted(np.unique(np.concatenate([y_true_combined, y_pred_combined])))
        target_names_combined = [PARALYSIS_MAP.get(l, f'Class_{l}') for l in report_labels_combined_num]
        if target_names_combined:
            combined_report_str = classification_report(y_true_combined, y_pred_combined,
                                                        labels=report_labels_combined_num,
                                                        target_names=target_names_combined, zero_division=0)
            logger.info(f"Combined {zone_name_display} Classification Report:\n{combined_report_str}")
            combined_cm = confusion_matrix(y_true_combined, y_pred_combined, labels=all_possible_labels_num)
            visualize_confusion_matrix(combined_cm, all_possible_labels_str, f"Combined {zone_name_display} (ML Preds)",
                                       output_dir)
        else:
            logger.warning(f"[{zone_name_display}] No target names for classification report.")

    # Generate error_details_for_export
    if not zone_df_valid.empty:
        error_details_for_export = []
        # Initialize a flag for logging missing prob columns only once per analyze_zone call per column type
        logged_missing_prob_cols_in_row_valid = set()

        for idx, row_valid in zone_df_valid.iterrows():
            expert_num_val = int(row_valid.get('Expert_Num', -1));
            pred_num_val = int(row_valid.get('Prediction_Num', -1))
            detail_row = {'Patient ID': row_valid.get('Patient ID', 'Unknown'),
                          'Side': row_valid.get('Side', 'Unknown'),
                          'Zone': zone_name_display, 'Expert_Label': expert_num_val, 'Model_Prediction': pred_num_val,
                          'Expert_Label_Name': PARALYSIS_MAP.get(expert_num_val, 'Unknown'),
                          'Predicted_Label_Name': PARALYSIS_MAP.get(pred_num_val, 'Unknown'),
                          'Is_Correct': expert_num_val == pred_num_val,
                          'Error_Type': 'Correct' if expert_num_val == pred_num_val else 'Misclassified'}
            current_model_confidence = np.nan
            for class_idx_map_val, class_name_map_str_val in PARALYSIS_MAP.items():
                prob_col_generic = f'Prob_{class_name_map_str_val.replace(" ", "_")}'
                if prob_col_generic in row_valid.index and pd.notna(row_valid[prob_col_generic]):
                    prob_val = row_valid[prob_col_generic]
                    detail_row[prob_col_generic] = prob_val
                    if pred_num_val == class_idx_map_val: current_model_confidence = prob_val
                else:
                    detail_row[prob_col_generic] = np.nan
                    if prob_col_generic not in logged_missing_prob_cols_in_row_valid:
                        logger.warning(
                            f"Prob col '{prob_col_generic}' NOT in row_valid or is NaN for Pt {row_valid.get('Patient ID', 'N/A')}, Side {row_valid.get('Side', 'N/A')} during error_details generation. Row keys: {row_valid.index.tolist()}")
                        logged_missing_prob_cols_in_row_valid.add(prob_col_generic)
            detail_row['Model_Confidence'] = current_model_confidence
            if ((expert_num_val == 0 and pred_num_val == 2) or (expert_num_val == 2 and pred_num_val == 0)):
                detail_row['Error_Type'] = 'Critical_Error'
            error_details_for_export.append(detail_row)
        error_details_export_df = pd.DataFrame(error_details_for_export)
        error_details_csv_path = os.path.join(output_dir, f"{safe_zone_name_part}_error_details.csv")
        error_details_export_df.to_csv(error_details_csv_path, index=False, float_format='%.4f')
        logger.info(f"Error details for review package saved to {error_details_csv_path}")

    cols_for_error_analysis_util = ['Expert_Num', 'Prediction_Num', 'Patient ID', 'Side']
    if FITZPATRICK_COL_NAME in zone_df_valid.columns: cols_for_error_analysis_util.append(FITZPATRICK_COL_NAME)
    for class_name_str_map_val in PARALYSIS_MAP.values():
        prob_col_gen = f'Prob_{class_name_str_map_val.replace(" ", "_")}'
        if prob_col_gen in zone_df_valid.columns: cols_for_error_analysis_util.append(prob_col_gen)
    error_analysis_df_for_utils = zone_df_valid[
        [col for col in cols_for_error_analysis_util if col in zone_df_valid.columns]].copy()
    error_analysis_df_for_utils.rename(columns={'Expert_Num': 'Expert', 'Prediction_Num': 'Prediction'}, inplace=True)
    perform_error_analysis(error_analysis_df_for_utils, output_dir, os.path.basename(errors_filename_base_for_perform),
                           zone_name_display, PARALYSIS_MAP)
    analyze_critical_errors(error_analysis_df_for_utils, output_dir, critical_error_path, zone_name_display,
                            PARALYSIS_MAP)
    analyze_partial_errors(error_analysis_df_for_utils, output_dir, partial_error_path, zone_name_display,
                           PARALYSIS_MAP)
    analyze_accuracy_by_fitzpatrick_group(zone_df_valid, 'Expert_Num', 'Prediction_Num',
                                          finding_name=f"{zone_name_display} Paralysis", logger_obj=logger)


def analyze_zone_performance(zone_key):
    if not UTILS_LOADED or ParalysisDetector is None:
        print(f"CRITICAL: Prerequisites not met for zone {zone_key}. Aborting.")
        if 'logger' in globals() and logger.handlers:
            logger.critical(f"Utils or ParalysisDetector not loaded for zone {zone_key}.")
        else:
            print(f"Logger not available for critical error in analyze_zone_performance for zone {zone_key}")
        return
    if zone_key not in ZONE_CONFIG:
        print(f"ERROR: Zone '{zone_key}' not found in paralysis_config.py")
        if 'logger' in globals() and logger.handlers:
            logger.error(f"Zone key '{zone_key}' not found in ZONE_CONFIG.")
        else:
            print(f"Logger not available for error in analyze_zone_performance for zone {zone_key}")
        return

    config = ZONE_CONFIG[zone_key]
    zone_name_display = config.get('name', zone_key.capitalize() + ' Face')
    filenames = config.get('filenames', {})
    zone_analysis_dir = os.path.join(ANALYSIS_DIR, zone_key)

    log_file_name_from_cfg = filenames.get('analysis_log')
    default_log_filename = f'{zone_key}_analysis.log'
    actual_log_filename = os.path.basename(log_file_name_from_cfg) if log_file_name_from_cfg else default_log_filename
    analysis_log_path = os.path.join(zone_analysis_dir, actual_log_filename)

    try:
        os.makedirs(os.path.dirname(analysis_log_path), exist_ok=True)
        _setup_logging(analysis_log_path,
                       LOGGING_CONFIG.get('level', 'DEBUG'))  # Use config level, _setup_logging forces DEBUG for file
    except OSError as e:
        logging.basicConfig(level=LOGGING_CONFIG.get('level', 'DEBUG').upper(),
                            format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'),
                            handlers=[logging.StreamHandler(sys.stdout)], force=True)
        logger.error(
            f"Could not create log directory {os.path.dirname(analysis_log_path)}: {e}. Logging to console only for this run.")

    logger.info(f"===== Starting Performance Analysis for Zone: {zone_name_display} (Key: {zone_key}) =====")
    logger.info(f"Analysis results output directory: {zone_analysis_dir}")
    logger.info(f"Log file for this run: {analysis_log_path}")

    try:
        detector = ParalysisDetector(zone=zone_key)
        if not all([detector.model, detector.scaler, detector.feature_names]):
            logger.error(f"[{zone_name_display}] Detector init failed (missing artifacts). Aborting analysis.")
            return
    except Exception as e:
        logger.error(f"[{zone_name_display}] Failed init ParalysisDetector: {e}", exc_info=True);
        return

    try:
        results_csv = INPUT_FILES.get('results_csv');
        expert_csv = INPUT_FILES.get('expert_key_csv')
        if not results_csv or not os.path.exists(results_csv): logger.error(
            f"Results CSV missing: {results_csv}"); return
        if not expert_csv or not os.path.exists(expert_csv): logger.error(
            f"Expert key CSV missing: {expert_csv}"); return
        results_df = pd.read_csv(results_csv, low_memory=False)
        if 'Patient ID' in results_df.columns:
            results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
        else:
            logger.error(f"'Patient ID' missing in {results_csv}"); return
        expert_df = pd.read_csv(expert_csv, dtype=str, keep_default_na=False, na_values=[''])
        patient_id_col_in_expert = 'Patient' if 'Patient' in expert_df.columns else 'Patient ID'
        if patient_id_col_in_expert not in expert_df.columns: logger.error(
            f"Patient ID column missing in {expert_csv}"); return
        expert_df.rename(columns={patient_id_col_in_expert: 'Patient ID'}, inplace=True)
        expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
        expert_cols_config = config.get('expert_columns', {})
        expert_left_raw_col = expert_cols_config.get('left');
        expert_right_raw_col = expert_cols_config.get('right')
        if not expert_left_raw_col or expert_left_raw_col not in expert_df.columns: logger.error(
            f"Expert left col '{expert_left_raw_col}' missing in {expert_csv}."); return
        if not expert_right_raw_col or expert_right_raw_col not in expert_df.columns: logger.error(
            f"Expert right col '{expert_right_raw_col}' missing in {expert_csv}."); return
        expert_df_subset_cols = ['Patient ID', expert_left_raw_col, expert_right_raw_col]
        if FITZPATRICK_COL_NAME in expert_df.columns:
            if FITZPATRICK_COL_NAME not in expert_df_subset_cols: expert_df_subset_cols.append(FITZPATRICK_COL_NAME)
        final_expert_cols_to_select = [col for col in expert_df_subset_cols if col in expert_df.columns]
        expert_df_subset = expert_df[final_expert_cols_to_select].copy()
        if expert_df_subset['Patient ID'].duplicated().any(): expert_df_subset.drop_duplicates(subset=['Patient ID'],
                                                                                               keep='first',
                                                                                               inplace=True)
        merged_df = pd.merge(results_df, expert_df_subset, on='Patient ID', how='inner', validate="many_to_one")
        if merged_df.empty: logger.error(f"[{zone_name_display}] Merge resulted in empty DataFrame."); return

        logger.info(f"[{zone_name_display}] Re-running detection for predictions and probabilities...")
        predictions_left, predictions_right = [], []
        num_classes = len(PARALYSIS_MAP)
        prob_data_left = {f'Prob_{PARALYSIS_MAP[ci].replace(" ", "_")}_Left': [] for ci in sorted(PARALYSIS_MAP.keys())}
        prob_data_right = {f'Prob_{PARALYSIS_MAP[ci].replace(" ", "_")}_Right': [] for ci in
                           sorted(PARALYSIS_MAP.keys())}
        pred_left_col_name = f'ML_Pred_Left_{zone_key.capitalize()}'
        pred_right_col_name = f'ML_Pred_Right_{zone_key.capitalize()}'

        for index, row in merged_df.iterrows():
            current_patient_id = str(row.get('Patient ID', 'Unknown_ID'))
            row_data_dict = row.to_dict()
            actual_probas_l, actual_probas_r = [np.nan] * num_classes, [np.nan] * num_classes
            details_l_from_detector, details_r_from_detector = None, None

            try:
                result_l, _, details_l = detector.detect(row_data_dict, 'left')
                details_l_from_detector = details_l
                predictions_left.append(result_l)
                if isinstance(details_l, dict) and 'probabilities' in details_l and \
                        isinstance(details_l['probabilities'], (list, np.ndarray)) and \
                        len(details_l['probabilities']) == num_classes:
                    actual_probas_l = list(details_l['probabilities'])
                else:
                    logger.debug(
                        f"Probs L for Pt {current_patient_id}, Zone {zone_name_display}, malformed/missing. Details: {details_l}. actual_probas_l will be NaNs.")
            except Exception as e_l:
                logger.error(f"Detector error L for Pt {current_patient_id}, Zone {zone_name_display}: {e_l}",
                             exc_info=False)
                predictions_left.append('Error')
            logger.debug(
                f"Pt {current_patient_id}, Zone {zone_name_display}, Side L: result='{predictions_left[-1] if predictions_left else 'N/A'}', details_dict_from_detector={details_l_from_detector}, actual_probas_l_for_append={actual_probas_l}")

            try:
                result_r, _, details_r = detector.detect(row_data_dict, 'right')
                details_r_from_detector = details_r
                predictions_right.append(result_r)
                if isinstance(details_r, dict) and 'probabilities' in details_r and \
                        isinstance(details_r['probabilities'], (list, np.ndarray)) and \
                        len(details_r['probabilities']) == num_classes:
                    actual_probas_r = list(details_r['probabilities'])
                else:
                    logger.debug(
                        f"Probs R for Pt {current_patient_id}, Zone {zone_name_display}, malformed/missing. Details: {details_r}. actual_probas_r will be NaNs.")
            except Exception as e_r:
                logger.error(f"Detector error R for Pt {current_patient_id}, Zone {zone_name_display}: {e_r}",
                             exc_info=False)
                predictions_right.append('Error')
            logger.debug(
                f"Pt {current_patient_id}, Zone {zone_name_display}, Side R: result='{predictions_right[-1] if predictions_right else 'N/A'}', details_dict_from_detector={details_r_from_detector}, actual_probas_r_for_append={actual_probas_r}")

            for class_idx in sorted(PARALYSIS_MAP.keys()):
                class_name_str = PARALYSIS_MAP[class_idx];
                safe_class_name = class_name_str.replace(" ", "_")
                val_l = actual_probas_l[class_idx] if len(actual_probas_l) == num_classes else np.nan
                val_r = actual_probas_r[class_idx] if len(actual_probas_r) == num_classes else np.nan
                prob_data_left[f'Prob_{safe_class_name}_Left'].append(float(val_l) if pd.notna(val_l) else np.nan)
                prob_data_right[f'Prob_{safe_class_name}_Right'].append(float(val_r) if pd.notna(val_r) else np.nan)
                if current_patient_id == merged_df['Patient ID'].iloc[0] and index == merged_df.index[
                    0]:  # Only for the very first patient
                    logger.debug(f"  Appending to prob_data_left['Prob_{safe_class_name}_Left']: {val_l}")
                    logger.debug(f"  Appending to prob_data_right['Prob_{safe_class_name}_Right']: {val_r}")

        merged_df[pred_left_col_name] = predictions_left
        merged_df[pred_right_col_name] = predictions_right
        for col, data_list in prob_data_left.items(): merged_df[col] = data_list
        for col, data_list in prob_data_right.items(): merged_df[col] = data_list

        log_cols_example = ['Patient ID'] + [f'Prob_{PARALYSIS_MAP[ci].replace(" ", "_")}_Left' for ci in
                                             sorted(PARALYSIS_MAP.keys()) if
                                             f'Prob_{PARALYSIS_MAP[ci].replace(" ", "_")}_Left' in merged_df.columns]
        if len(log_cols_example) > 1:
            logger.info(
                f"[{zone_name_display}] Added prediction and probability columns to merged_df. Columns example: {merged_df[log_cols_example].columns.tolist()}")
            logger.debug(
                f"[{zone_name_display}] merged_df with probas head (showing select prob_left cols):\n{merged_df[log_cols_example].head().to_string()}")
        else:
            logger.warning(
                f"[{zone_name_display}] No Prob_X_Left columns found in merged_df to show in log head example.")

        analyze_zone(merged_df, zone_name_display, pred_left_col_name, expert_left_raw_col, pred_right_col_name,
                     expert_right_raw_col, zone_analysis_dir, config)
        logger.info(f"===== Performance Analysis for Zone: {zone_name_display} Complete =====")

    except FileNotFoundError as e:
        logger.error(f"[{zone_name_display}] Input file not found: {e}.")
    except pd.errors.EmptyDataError as ede:
        logger.error(f"[{zone_name_display}] Input file empty: {ede}.")
    except Exception as e:
        logger.error(f"[{zone_name_display}] UNHANDLED EXCEPTION in analyze_zone_performance: {str(e)}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level='DEBUG', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)
    if not UTILS_LOADED:
        print("CRITICAL: paralysis_utils could not be loaded.")
    elif ParalysisDetector is None:
        print("CRITICAL: ParalysisDetector could not be loaded.")
    else:
        zones_to_analyze = ['lower', 'mid', 'upper']
        # zones_to_analyze = ['lower']
        for zone_key_iter in zones_to_analyze:
            if zone_key_iter in ZONE_CONFIG:
                print(
                    f"\n--- Starting Performance Analysis for {ZONE_CONFIG[zone_key_iter].get('name', zone_key_iter.capitalize() + ' Face')} (Key: {zone_key_iter}) ---")
                analyze_zone_performance(zone_key_iter)
                print(
                    f"--- Finished Performance Analysis for {ZONE_CONFIG[zone_key_iter].get('name', zone_key_iter.capitalize() + ' Face')} ---")
            else:
                print(f"Warning: Zone '{zone_key_iter}' not found in config. Skipping.")