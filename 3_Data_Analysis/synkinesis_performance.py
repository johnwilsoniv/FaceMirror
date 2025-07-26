# synkinesis_performance.py (v6 - Adds Fisher's Exact Test for Fitzpatrick analysis)

import pandas as pd
import numpy as np
import logging
import os
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats  # Added for Fisher's Exact Test

# Import central config, generic detector, and UPDATED utils
try:
    from synkinesis_config import SYNKINESIS_CONFIG, LOGGING_CONFIG, INPUT_FILES, ANALYSIS_DIR
except ImportError as e:
    print(f"CRITICAL: Failed to import from synkinesis_config.py - {e}")
    SYNKINESIS_CONFIG = {}
    LOGGING_CONFIG = {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'}
    INPUT_FILES = {}
    ANALYSIS_DIR = 'analysis_results'

try:
    from synkinesis_detector import SynkinesisDetector
except ImportError as e:
    print(f"CRITICAL: Failed to import SynkinesisDetector - {e}")
    SynkinesisDetector = None  # Allow script structure analysis

try:
    from paralysis_utils import (
        standardize_synkinesis_labels,  # Import SPECIFIC standardizer
        SYNKINESIS_MAP,  # Import the map to pass to analysis functions
        visualize_confusion_matrix,
        perform_error_analysis,
        evaluate_thresholds
    )

    UTILS_LOADED = True
except ImportError as e:
    print(f"CRITICAL: Failed to import from paralysis_utils.py - {e}")
    UTILS_LOADED = False


    # Dummy functions and map if utils not found
    def standardize_synkinesis_labels(val):
        return str(val)


    SYNKINESIS_MAP = {0: 'None', 1: 'Synkinesis'}  # Fallback


    def visualize_confusion_matrix(*args, **kwargs):
        pass


    def perform_error_analysis(*args, **kwargs):
        pass


    def evaluate_thresholds(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)  # Configured in analyze_synkinesis_performance

FITZPATRICK_COL_NAME = "Fitzpatrick"  # Standard name for the column from expert key


def _setup_logging(log_file, level):
    """Sets up logging for a specific analysis run."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    log_dir = os.path.dirname(log_file)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    root_logger = logging.getLogger();
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
                        force=True)


# --- UPDATED HELPER: Analyze Accuracy by Fitzpatrick Group (with Fisher's Exact Test) ---
def analyze_accuracy_by_fitzpatrick_group(df, expert_col_num, pred_col_num, fitzpatrick_raw_col=FITZPATRICK_COL_NAME,
                                          finding_name="Finding", logger_obj=None):
    """
    Analyzes and logs prediction accuracy based on Fitzpatrick scale groups.
    Performs Fisher's Exact Test to check for significant differences.
    Assumes df contains numeric expert and prediction columns.
    """
    if logger_obj is None:
        logger_obj = logging.getLogger(__name__)

    if fitzpatrick_raw_col not in df.columns:
        logger_obj.warning(
            f"[{finding_name}] Fitzpatrick column '{fitzpatrick_raw_col}' not found in DataFrame. Skipping Fitzpatrick analysis.")
        return

    if not all(col in df.columns for col in [expert_col_num, pred_col_num]):
        logger_obj.error(
            f"[{finding_name}] Missing expert ('{expert_col_num}') or prediction ('{pred_col_num}') column for Fitzpatrick analysis. Skipping.")
        return

    analysis_df = df.copy()

    logger_obj.info(f"--- [{finding_name}] Fitzpatrick Sub-Analysis ---")
    raw_counts = analysis_df[fitzpatrick_raw_col].value_counts(dropna=False).sort_index()
    logger_obj.info(f"Raw '{fitzpatrick_raw_col}' value distribution (Total: {len(analysis_df)}):")
    for val, count in raw_counts.items():
        logger_obj.info(f"  Value '{val}': {count}")

    analysis_df['Fitzpatrick_Numeric'] = pd.to_numeric(analysis_df[fitzpatrick_raw_col], errors='coerce')
    nan_fitz_count = analysis_df['Fitzpatrick_Numeric'].isnull().sum()
    if nan_fitz_count > 0:
        logger_obj.info(
            f"{nan_fitz_count} rows had non-numeric or missing '{fitzpatrick_raw_col}' values and will be excluded from Fitzpatrick grouping.")

    analysis_df['Fitzpatrick_Group'] = pd.Series(index=analysis_df.index, dtype='object')
    analysis_df.loc[analysis_df['Fitzpatrick_Numeric'].isin([1, 2, 3, 4]), 'Fitzpatrick_Group'] = 'Fitzpatrick 1-4'
    analysis_df.loc[analysis_df['Fitzpatrick_Numeric'].isin([5, 6]), 'Fitzpatrick_Group'] = 'Fitzpatrick 5-6'

    analysis_df.dropna(subset=['Fitzpatrick_Group', expert_col_num, pred_col_num], inplace=True)

    if analysis_df.empty:
        logger_obj.warning(
            f"[{finding_name}] No data remaining after filtering for valid Fitzpatrick groups. Skipping accuracy calculation and statistical test.")
        return

    logger_obj.info(
        f"Analyzing {len(analysis_df)} samples with valid Fitzpatrick groups (1-4 or 5-6) and valid expert/prediction numbers.")

    analysis_df['is_correct'] = (analysis_df[expert_col_num].astype(int) == analysis_df[pred_col_num].astype(int))

    accuracy_by_group = analysis_df.groupby('Fitzpatrick_Group')['is_correct'].agg(['mean', 'count', 'sum'])
    accuracy_by_group.rename(columns={'mean': 'accuracy', 'count': 'total_samples', 'sum': 'correct_predictions'},
                             inplace=True)

    logger_obj.info(f"Accuracy by Fitzpatrick Group for {finding_name}:")
    if not accuracy_by_group.empty:
        for group_name_iter, row_data in accuracy_by_group.iterrows():
            logger_obj.info(f"  {group_name_iter:<18}: Accuracy = {row_data['accuracy']:.3f} "
                            f"({int(row_data['correct_predictions'])}/{int(row_data['total_samples'])} correct)")
    else:
        logger_obj.info("  No data available for Fitzpatrick group accuracy breakdown. Skipping statistical test.")
        logger_obj.info(f"--- End of Fitzpatrick Sub-Analysis for {finding_name} ---")
        return accuracy_by_group

    # --- Statistical Test for Difference (Fisher's Exact Test) ---
    group1_name = 'Fitzpatrick 1-4'
    group2_name = 'Fitzpatrick 5-6'

    if group1_name in accuracy_by_group.index and group2_name in accuracy_by_group.index:
        correct_g1 = int(accuracy_by_group.loc[group1_name, 'correct_predictions'])
        total_g1 = int(accuracy_by_group.loc[group1_name, 'total_samples'])
        incorrect_g1 = total_g1 - correct_g1

        correct_g2 = int(accuracy_by_group.loc[group2_name, 'correct_predictions'])
        total_g2 = int(accuracy_by_group.loc[group2_name, 'total_samples'])
        incorrect_g2 = total_g2 - correct_g2

        if total_g1 == 0 or total_g2 == 0:
            logger_obj.warning(
                f"  One or both Fitzpatrick groups ({group1_name}: N={total_g1}, {group2_name}: N={total_g2}) have zero total samples. Skipping statistical test.")
        else:
            table = np.array([
                [correct_g1, incorrect_g1],
                [correct_g2, incorrect_g2]
            ])

            try:
                odds_ratio, p_value = stats.fisher_exact(table)
                logger_obj.info(
                    f"  Statistical Test (Fisher's Exact) for difference in accuracy between Fitzpatrick groups:")
                logger_obj.info(f"    Contingency Table (Correct, Incorrect):")
                logger_obj.info(f"      {group1_name}: [{table[0, 0]}, {table[0, 1]}]")
                logger_obj.info(f"      {group2_name}: [{table[1, 0]}, {table[1, 1]}]")
                logger_obj.info(f"    Odds Ratio = {odds_ratio:.3f}, p-value = {p_value:.4f}")

                alpha = 0.05
                if p_value < alpha:
                    logger_obj.info(
                        f"    The difference in accuracy between Fitzpatrick groups IS statistically significant (p < {alpha}).")
                else:
                    logger_obj.info(
                        f"    The difference in accuracy between Fitzpatrick groups IS NOT statistically significant (p >= {alpha}).")
            except ValueError as e:
                logger_obj.error(f"    Error during Fisher's Exact Test: {e}. Table was: {table.tolist()}")
    else:
        logger_obj.info(
            f"  Could not find both '{group1_name}' and '{group2_name}' in accuracy results to perform statistical test.")
        logger_obj.info(f"  Available groups for statistical test: {list(accuracy_by_group.index)}")

    logger_obj.info(f"--- End of Fitzpatrick Sub-Analysis for {finding_name} ---")
    return accuracy_by_group


# --- UPDATED: analyze_results_at_threshold (uses specific standardizer & passed map) ---
def analyze_results_at_threshold(data, finding_name, pred_left_col, expert_std_left_col, pred_right_col,
                                 expert_std_right_col, output_dir, config):
    """
    Analyzes performance using thresholded predictions and STANDARDIZED expert labels.
    Filters 'NA' labels internally before calculating metrics. Uses SYNKINESIS_MAP.
    Also performs Fitzpatrick sub-analysis if data is available.

    Args:
        data (pd.DataFrame): DataFrame containing thresholded predictions (0/1),
                             STANDARDIZED expert labels ('None'/'Synkinesis'/'NA'),
                             and potentially FITZPATRICK_COL_NAME.
        finding_name (str): User-friendly name (e.g., "Ocular-Oral").
        pred_left_col, expert_std_left_col, pred_right_col, expert_std_right_col: Column names.
        output_dir (str): Base directory for analysis results (ANALYSIS_DIR).
        config (dict): The specific configuration slice for this synkinesis type.
    """
    class_names_map = SYNKINESIS_MAP  # Use the imported synkinesis map
    threshold = config.get('DETECTION_THRESHOLD', 'N/A')  # Get threshold used for predictions
    logger.info(f"--- Analyzing {finding_name} Performance (at Detector Threshold: {threshold}) ---")
    logger.info(f"Using class map: {class_names_map}")

    base_required_cols = ['Patient ID', pred_left_col, expert_std_left_col, pred_right_col, expert_std_right_col]
    # Check for Fitzpatrick column, add if present in data
    all_cols_for_selection = base_required_cols[:]
    if FITZPATRICK_COL_NAME in data.columns:
        all_cols_for_selection.append(FITZPATRICK_COL_NAME)

    if not all(col in data.columns for col in base_required_cols):  # Still check base required
        missing = [c for c in base_required_cols if c not in data.columns]
        logger.error(f"Missing base cols for threshold analysis: {missing}. Abort.");
        return

    # Get class names from the specific map
    negative_class_name = class_names_map.get(0, 'None')
    positive_class_name = class_names_map.get(1, 'Synkinesis')

    # Prepare left/right dfs using STANDARDIZED expert columns
    # Include Fitzpatrick if available
    left_cols_to_select = ['Patient ID', pred_left_col, expert_std_left_col]
    right_cols_to_select = ['Patient ID', pred_right_col, expert_std_right_col]
    if FITZPATRICK_COL_NAME in data.columns:
        left_cols_to_select.append(FITZPATRICK_COL_NAME)
        right_cols_to_select.append(FITZPATRICK_COL_NAME)

    left = data[left_cols_to_select].copy();
    left['Side'] = 'Left'
    right = data[right_cols_to_select].copy();
    right['Side'] = 'Right'

    # Rename columns for consistency
    rename_cols_base = ['Patient ID', 'Prediction_Thr', 'Expert_Std']
    if FITZPATRICK_COL_NAME in data.columns:
        rename_cols_fitz = rename_cols_base + [FITZPATRICK_COL_NAME, 'Side']
        left.columns = rename_cols_fitz
        right.columns = rename_cols_fitz
    else:
        rename_cols_no_fitz = rename_cols_base + ['Side']
        left.columns = rename_cols_no_fitz
        right.columns = rename_cols_no_fitz

    # Filter 'NA' Standardized Labels
    left_valid = left[left['Expert_Std'] != 'NA'].copy()
    right_valid = right[right['Expert_Std'] != 'NA'].copy()
    logger.info(f"Filtered NA expert labels. Left valid: {len(left_valid)}, Right valid: {len(right_valid)}")

    if left_valid.empty and right_valid.empty:
        logger.error(f"[{finding_name}] No valid data left after filtering 'NA' labels. Cannot analyze.")
        return

    # Map Valid Standardized Labels ('None', positive_class_name) to 0/1
    target_mapping = {negative_class_name: 0, positive_class_name: 1}
    left_valid['Expert_Num'] = left_valid['Expert_Std'].map(target_mapping)
    right_valid['Expert_Num'] = right_valid['Expert_Std'].map(target_mapping)

    # Handle unexpected labels after mapping
    unexpected_left = left_valid.loc[left_valid['Expert_Num'].isnull(), 'Expert_Std'].unique()
    unexpected_right = right_valid.loc[right_valid['Expert_Num'].isnull(), 'Expert_Std'].unique()
    if len(unexpected_left) > 0: logger.warning(f"Unexpected left labels after mapping: {unexpected_left}. Dropping.")
    if len(unexpected_right) > 0: logger.warning(
        f"Unexpected right labels after mapping: {unexpected_right}. Dropping.")
    left_valid.dropna(subset=['Expert_Num'], inplace=True)
    right_valid.dropna(subset=['Expert_Num'], inplace=True)

    # Ensure Prediction_Thr column is numeric 0/1
    left_valid['Prediction_Num'] = pd.to_numeric(left_valid['Prediction_Thr'], errors='coerce').fillna(0).astype(int)
    right_valid['Prediction_Num'] = pd.to_numeric(right_valid['Prediction_Thr'], errors='coerce').fillna(0).astype(int)

    # Combine valid, mapped data
    combined_valid = pd.concat([left_valid, right_valid], ignore_index=True)
    if combined_valid.empty:
        logger.error(f"[{finding_name}] combined_valid dataframe is empty after NA filtering/mapping.")
        return

    combined_valid['Expert_Num'] = combined_valid['Expert_Num'].astype(int)

    # Proceed with analysis using combined_valid, left_valid, right_valid
    labels = sorted(combined_valid['Expert_Num'].unique())  # Should be [0, 1] or subset
    target_names = [class_names_map.get(l, str(l)) for l in labels]
    if len(labels) < 2: logger.warning(
        f"[{finding_name}] Only one class ({labels}) present in valid data. Metrics/plots may be limited.")
    logger.info(
        f"Analyzing {len(combined_valid)} valid predictions. Expert Dist: {combined_valid['Expert_Num'].map(class_names_map).value_counts().to_dict()}. Pred Dist: {combined_valid['Prediction_Num'].map(class_names_map).value_counts().to_dict()}.")

    # Create synkinesis-type-specific output directory within ANALYSIS_DIR
    synk_type_output_dir = os.path.join(output_dir, config.get('type', finding_name.lower().replace(' ', '_')))
    os.makedirs(synk_type_output_dir, exist_ok=True)

    # Reporting and Visualization
    try:
        # Define labels for reports/CMs (use all possible binary labels)
        all_labels_num = sorted(class_names_map.keys())  # [0, 1]
        all_target_names = [class_names_map[l] for l in all_labels_num]  # ['None', 'Synkinesis']

        # Combined
        if not combined_valid.empty and combined_valid['Expert_Num'].notna().all() and combined_valid[
            'Prediction_Num'].notna().all():
            report_labels_num = sorted(
                np.unique(np.concatenate([combined_valid['Expert_Num'], combined_valid['Prediction_Num']])))
            report_target_names = [class_names_map.get(l, str(l)) for l in report_labels_num]
            logger.info(f"\n--- Combined {finding_name} ---")
            report_c_str = classification_report(combined_valid['Expert_Num'], combined_valid['Prediction_Num'],
                                                 labels=report_labels_num, target_names=report_target_names,
                                                 zero_division=0)
            logger.info("Report:\n" + report_c_str)
            cm_c = confusion_matrix(combined_valid['Expert_Num'], combined_valid['Prediction_Num'],
                                    labels=all_labels_num)  # Use all labels for consistent size
            logger.info("CM:\n" + str(cm_c))
            visualize_confusion_matrix(cm_c, all_target_names, f"Combined {finding_name}", synk_type_output_dir)
        else:
            logger.info(f"\n--- Combined {finding_name}: Not enough valid data for report/CM ---")

        # Left
        report_l_left_valid = left_valid[left_valid['Expert_Num'].notna() & left_valid['Prediction_Num'].notna()]
        if not report_l_left_valid.empty:
            logger.info(f"\n--- Left {finding_name} ---")
            report_l_labels = sorted(
                np.unique(np.concatenate([report_l_left_valid['Expert_Num'], report_l_left_valid['Prediction_Num']])))
            report_l_names = [class_names_map.get(l, str(l)) for l in report_l_labels]
            report_l_str = classification_report(report_l_left_valid['Expert_Num'],
                                                 report_l_left_valid['Prediction_Num'], labels=report_l_labels,
                                                 target_names=report_l_names, zero_division=0)
            logger.info("Report:\n" + report_l_str)
            cm_l = confusion_matrix(report_l_left_valid['Expert_Num'], report_l_left_valid['Prediction_Num'],
                                    labels=all_labels_num)
            logger.info("CM:\n" + str(cm_l))
            visualize_confusion_matrix(cm_l, all_target_names, f"Left {finding_name}", synk_type_output_dir)
        else:
            logger.info(f"\n--- Left {finding_name}: No valid numeric data for report ---")

        # Right
        report_r_right_valid = right_valid[right_valid['Expert_Num'].notna() & right_valid['Prediction_Num'].notna()]
        if not report_r_right_valid.empty:
            logger.info(f"\n--- Right {finding_name} ---")
            report_r_labels = sorted(
                np.unique(np.concatenate([report_r_right_valid['Expert_Num'], report_r_right_valid['Prediction_Num']])))
            report_r_names = [class_names_map.get(l, str(l)) for l in report_r_labels]
            report_r_str = classification_report(report_r_right_valid['Expert_Num'],
                                                 report_r_right_valid['Prediction_Num'], labels=report_r_labels,
                                                 target_names=report_r_names, zero_division=0)
            logger.info("Report:\n" + report_r_str)
            cm_r = confusion_matrix(report_r_right_valid['Expert_Num'], report_r_right_valid['Prediction_Num'],
                                    labels=all_labels_num)
            logger.info("CM:\n" + str(cm_r))
            visualize_confusion_matrix(cm_r, all_target_names, f"Right {finding_name}", synk_type_output_dir)
        else:
            logger.info(f"\n--- Right {finding_name}: No valid numeric data for report ---")

        # Error Analysis (using numerical data and PASSING the map)
        error_analysis_df_cols = ['Patient ID', 'Side', 'Expert_Num', 'Prediction_Num']
        if FITZPATRICK_COL_NAME in combined_valid.columns:  # Carry Fitzpatrick to error analysis if present
            error_analysis_df_cols.append(FITZPATRICK_COL_NAME)

        error_analysis_df = combined_valid[error_analysis_df_cols].copy()
        error_analysis_df.rename(columns={'Expert_Num': 'Expert', 'Prediction_Num': 'Prediction'}, inplace=True)
        thresh_str = str(threshold).replace('.', '_')
        # Use basename logic for error file
        error_filename_config = config.get('filenames', {}).get('error_report',
                                                                f"{config.get('type', 'synkinesis')}_errors_thresh_{thresh_str}.txt").replace(
            '{thresh}', thresh_str)
        error_filename_base = os.path.basename(error_filename_config).replace('.txt',
                                                                              '')  # Get base name without extension

        perform_error_analysis(error_analysis_df, synk_type_output_dir, error_filename_base, finding_name,
                               class_names_map)  # Pass SYNKINESIS_MAP

        # --- Fitzpatrick Sub-Analysis Call ---
        analyze_accuracy_by_fitzpatrick_group(
            combined_valid,  # DataFrame with Expert_Num, Prediction_Num, and potentially FITZPATRICK_COL_NAME
            expert_col_num='Expert_Num',
            pred_col_num='Prediction_Num',
            fitzpatrick_raw_col=FITZPATRICK_COL_NAME,
            finding_name=f"{finding_name} (Thresholded)",
            logger_obj=logger
        )

    except Exception as e:
        logger.error(f"[{finding_name}] Error during thresholded metrics/vis: {str(e)}", exc_info=True)


# --- Main Performance Analysis Function ---
def analyze_synkinesis_performance(synkinesis_type):
    """ Main performance analysis pipeline for a specific synkinesis type. """
    if not UTILS_LOADED: print(f"ERROR: Cannot analyze {synkinesis_type}, paralysis_utils failed to load."); return
    if SynkinesisDetector is None: print(
        f"ERROR: Cannot analyze {synkinesis_type}, SynkinesisDetector failed to load."); return
    if synkinesis_type not in SYNKINESIS_CONFIG: print(f"ERROR: Type '{synkinesis_type}' not in config."); return

    config = SYNKINESIS_CONFIG[synkinesis_type]
    config['type'] = synkinesis_type  # Ensure type key exists for subdir naming
    name = config.get('name', synkinesis_type.replace('_', ' ').title())
    filenames = config.get('filenames', {});
    expert_cols_cfg = config.get('expert_columns', {})

    # Setup logging within the type-specific analysis directory
    synk_type_analysis_dir = os.path.join(ANALYSIS_DIR, synkinesis_type)
    log_file = filenames.get('analysis_log')
    if not log_file:  # Fallback path construction
        log_file = os.path.join(synk_type_analysis_dir, f'{synkinesis_type}_analysis.log')
    else:  # Ensure log file path uses the analysis subdir
        log_file = os.path.join(synk_type_analysis_dir, os.path.basename(log_file))

    _setup_logging(log_file, LOGGING_CONFIG.get('level', 'INFO'))
    logger.info(f"===== Starting Performance Analysis for Synkinesis: {name} =====")
    logger.info(f"Analysis results root directory for this type: {synk_type_analysis_dir}")
    logger.info(f"Log file for this run: {log_file}")

    # Initialize Detector
    detector = None
    try:
        detector = SynkinesisDetector(synkinesis_type=synkinesis_type)
    except ValueError as ve:
        logger.error(f"[{name}] {ve}"); return
    except Exception as e_init:
        logger.error(f"[{name}] Failed init SynkinesisDetector: {e_init}", exc_info=True); return
    if not all([detector.model, detector.scaler, detector.feature_names]): logger.error(
        f"[{name}] Detector missing artifacts. Abort."); return

    try:  # Load data
        results_csv = INPUT_FILES.get('results_csv');
        expert_csv = INPUT_FILES.get('expert_key_csv')
        if not results_csv or not expert_csv: logger.error(f"[{name}] Input CSV paths missing. Abort."); return
        if not os.path.exists(results_csv) or not os.path.exists(expert_csv): logger.error(
            f"[{name}] Input files not found ({results_csv}, {expert_csv}). Abort."); return
        logger.info(f"[{name}] Loading data...");
        results_df = pd.read_csv(results_csv, low_memory=False)

        temp_expert_df_check = pd.read_csv(expert_csv, nrows=1)
        if FITZPATRICK_COL_NAME in temp_expert_df_check.columns:
            logger.info(f"Found '{FITZPATRICK_COL_NAME}' column in expert key.")
        else:
            logger.info(
                f"'{FITZPATRICK_COL_NAME}' column NOT found in expert key. Fitzpatrick analysis will be skipped if column is missing in data.")
        del temp_expert_df_check
        expert_df = pd.read_csv(expert_csv, keep_default_na=False, na_values=[''], dtype=str)

        # Rename/Process Expert Columns
        exp_left_orig = expert_cols_cfg.get('left');
        exp_right_orig = expert_cols_cfg.get('right')
        if not exp_left_orig or not exp_right_orig: logger.error(
            f"[{name}] Config missing expert column names. Abort."); return
        if exp_left_orig not in expert_df.columns or exp_right_orig not in expert_df.columns:
            logger.error(f"Missing raw expert columns '{exp_left_orig}' or '{exp_right_orig}' in expert DF. Abort.")
            return

        rename_map = {'Patient': 'Patient ID', exp_left_orig: exp_left_orig, exp_right_orig: exp_right_orig}
        expert_df = expert_df.rename(columns={k: v for k, v in rename_map.items() if k in expert_df.columns})

        # Standardize Labels using the SPECIFIC utility function
        logger.info(f"Standardizing expert labels using standardize_synkinesis_labels...")
        expert_std_left_col = f'Expert_Std_Left_{synkinesis_type}'
        expert_std_right_col = f'Expert_Std_Right_{synkinesis_type}'
        expert_df[expert_std_left_col] = expert_df[exp_left_orig].apply(standardize_synkinesis_labels)
        expert_df[expert_std_right_col] = expert_df[exp_right_orig].apply(standardize_synkinesis_labels)
        logger.info(f"[{name}] Expert labels standardized into '{expert_std_left_col}' and '{expert_std_right_col}'.")
        logger.info(
            f"Value counts in Standardized expert columns:\nLeft:\n{expert_df[expert_std_left_col].value_counts(dropna=False).to_string()}\nRight:\n{expert_df[expert_std_right_col].value_counts(dropna=False).to_string()}")

        # Merge
        results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip();
        expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()

        merge_cols_for_expert_df = ['Patient ID', expert_std_left_col, expert_std_right_col]
        if FITZPATRICK_COL_NAME in expert_df.columns:
            if FITZPATRICK_COL_NAME not in merge_cols_for_expert_df:
                merge_cols_for_expert_df.append(FITZPATRICK_COL_NAME)

        expert_df_merge_subset_cols = [col for col in merge_cols_for_expert_df if col in expert_df.columns]

        if expert_df['Patient ID'].duplicated().any():
            logger.warning("Duplicate Patient IDs found in expert data. Keeping first entry.")
            expert_df.drop_duplicates(subset=['Patient ID'], keep='first', inplace=True)

        merged_df = pd.merge(results_df, expert_df[expert_df_merge_subset_cols], on='Patient ID', how='inner',
                             validate="many_to_one")
        logger.info(f"[{name}] Merged data: {len(merged_df)} rows.");
        if merged_df.empty or expert_std_left_col not in merged_df.columns or expert_std_right_col not in merged_df.columns:
            logger.error(f"[{name}] Merge failed or standardized target cols missing. Abort.");
            return

        if FITZPATRICK_COL_NAME in merged_df.columns:
            logger.info(f"'{FITZPATRICK_COL_NAME}' column successfully merged into analysis data.")
        else:
            logger.warning(
                f"'{FITZPATRICK_COL_NAME}' column not present in merged_df. Fitzpatrick analysis may be skipped.")

        # Re-run Detection
        logger.info(f"[{name}] Re-running detection (getting predictions and probabilities)...");
        preds_l, preds_r = [], [];
        confs_l, confs_r = [], [];
        probas_l, probas_r = [], []
        num_rows = len(merged_df);
        log_interval = max(1, num_rows // 10)
        detection_errors = 0
        for index, row in merged_df.iterrows():
            if (index + 1) % log_interval == 0: logger.info(f"Processing row {index + 1}/{num_rows}...")
            pid = row.get('Patient ID', 'Unk');
            row_dict = row.to_dict()
            det_l, conf_l, proba_l, details_l = detector.detect(row_dict, 'Left')
            det_r, conf_r, proba_r, details_r = detector.detect(row_dict, 'Right')
            if details_l.get('error'): detection_errors += 1; logger.error(
                f"PID {pid} Left Detect Error: {details_l['error']}")
            if details_r.get('error'): detection_errors += 1; logger.error(
                f"PID {pid} Right Detect Error: {details_r['error']}")
            preds_l.append(1 if det_l else 0);
            confs_l.append(conf_l);
            probas_l.append(proba_l)
            preds_r.append(1 if det_r else 0);
            confs_r.append(conf_r);
            probas_r.append(proba_r)

        if detection_errors > 0: logger.warning(f"[{name}] Encountered {detection_errors} errors during detection.")

        pred_l_col = f'ML_Pred_Left_{synkinesis_type}';
        conf_l_col = f'ML_Conf_Left_{synkinesis_type}';
        proba_l_col = f'ML_Proba_Left_{synkinesis_type}'
        pred_r_col = f'ML_Pred_Right_{synkinesis_type}';
        conf_r_col = f'ML_Conf_Right_{synkinesis_type}';
        proba_r_col = f'ML_Proba_Right_{synkinesis_type}'
        merged_df[pred_l_col] = preds_l;
        merged_df[conf_l_col] = confs_l;
        merged_df[proba_l_col] = probas_l
        merged_df[pred_r_col] = preds_r;
        merged_df[conf_r_col] = confs_r;
        merged_df[proba_r_col] = probas_r

        analysis_df = merged_df.copy()

        analyze_results_at_threshold(
            analysis_df, name,
            pred_l_col, expert_std_left_col,
            pred_r_col, expert_std_right_col,
            ANALYSIS_DIR,
            config
        )

        thresh_csv_path = filenames.get('threshold_eval_csv')
        pr_curve_path = filenames.get('pr_curve_png')
        if not thresh_csv_path or not pr_curve_path:
            logger.warning(f"[{name}] Missing threshold eval/PR curve paths. Skipping.")
        else:
            thresh_csv_path_zoned = os.path.join(synk_type_analysis_dir, os.path.basename(thresh_csv_path))
            pr_curve_path_zoned = os.path.join(synk_type_analysis_dir, os.path.basename(pr_curve_path))
            evaluate_thresholds(
                analysis_df,
                proba_l_col, expert_std_left_col,
                proba_r_col, expert_std_right_col,
                synk_type_analysis_dir,
                name,
                thresh_csv_path_zoned,
                pr_curve_path_zoned,
                SYNKINESIS_MAP
            )
        logger.info(f"===== Performance Analysis for Synkinesis: {name} Complete =====")
    except FileNotFoundError as e:
        logger.error(f"[{name}] Input file not found: {e}. Abort."); return
    except Exception as e:
        logger.error(f"[{name}] UNHANDLED EXCEPTION in performance analysis: {e}", exc_info=True)


if __name__ == "__main__":
    if not UTILS_LOADED:
        print("CRITICAL: paralysis_utils could not be loaded. Performance analysis cannot run.")
    elif SynkinesisDetector is None:
        print("CRITICAL: SynkinesisDetector could not be loaded. Performance analysis cannot run.")
    elif not SYNKINESIS_CONFIG:
        print("CRITICAL: SYNKINESIS_CONFIG is empty. Cannot run performance analysis.")
    else:
        all_types = list(SYNKINESIS_CONFIG.keys())
        print(f"Found synkinesis types in config: {all_types}")
        analyze_types = all_types
        for type_key in analyze_types:
            if type_key in SYNKINESIS_CONFIG:
                print(f"\n--- Starting Performance Analysis for Synkinesis Type: {type_key.upper()} ---")
                analyze_synkinesis_performance(type_key)
                print(f"--- Finished Performance Analysis for Synkinesis Type: {type_key.upper()} ---")
            else:
                print(f"Warning: Synkinesis type '{type_key}' not found in config despite being listed? Skipping.")