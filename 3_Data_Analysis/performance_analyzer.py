# performance_analyzer.py
# Analyzes ML predictions against an expert key.
# - Updated to analyze Mentalis Synkinesis and Hypertonicity columns
# - Added robust duplicate column check and fix for 'Patient ID'
# - Added debug logging for specific case comparison
# - Hardcoded DEBUG logging level
# <<< CHANGE: Added logic for combined synkinesis/hypertonicity stats >>>

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import json # For pretty-printing dicts in debug logs

# --- Configuration ---
RESULTS_FILE = '../3.5_Results/combined_results.csv' # Relative path often used
EXPERT_FILE = 'FPRS FP Key.csv'
OUTPUT_DIR = 'performance_analysis_results'
LOG_FILE = os.path.join(OUTPUT_DIR, 'performance_analysis.log')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging Setup ---
# Clear existing handlers to avoid duplicate logs if run multiple times
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
# Configure logging - Hardcoded DEBUG level
log_level = logging.INFO # Set to INFO for cleaner output, DEBUG for tracing
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
                              logging.StreamHandler()],
                    force=True) # force=True ensures reconfiguration works
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # Silence matplotlib debug logs
logger.info(f"Hardcoded analysis log level to {logging.getLevelName(log_level)}") # Log the set level

# --- Column Mapping (Expert Key -> Standard Name) ---
COLUMN_MAPPING = {
    'Patient': 'Patient ID',
    # Paralysis
    'Paralysis - Left Upper Face': 'Left Upper Face Paralysis',
    'Paralysis - Left Mid Face': 'Left Mid Face Paralysis',
    'Paralysis - Left Lower Face': 'Left Lower Face Paralysis',
    'Paralysis - Right Upper Face': 'Right Upper Face Paralysis',
    'Paralysis - Right Mid Face': 'Right Mid Face Paralysis',
    'Paralysis - Right Lower Face': 'Right Lower Face Paralysis',
    # Synkinesis
    'Oral-Ocular Synkinesis Left': 'Oral-Ocular Left',
    'Oral-Ocular Synkinesis Right': 'Oral-Ocular Right',
    'Ocular-Oral Synkinesis Left': 'Ocular-Oral Left',
    'Ocular-Oral Synkinesis Right': 'Ocular-Oral Right',
    'Snarl Smile Left': 'Snarl-Smile Left',
    'Snarl Smile Right': 'Snarl-Smile Right',
    'Mentalis Synkinesis Left': 'Mentalis Left',
    'Mentalis Synkinesis Right': 'Mentalis Right',
    # Hypertonicity
    'Hypertonicity Left': 'Hypertonicity Left',
    'Hypertonicity Right': 'Hypertonicity Right',
}

# --- Define column groups based on the *standardized* names ---
PARALYSIS_COLUMNS = [val for key, val in COLUMN_MAPPING.items() if 'Paralysis' in key and 'Patient' not in key]
SYNKINESIS_COLUMNS = [val for key, val in COLUMN_MAPPING.items() if ('Synkinesis' in key or 'Snarl Smile' in key or 'Oral' in key) and 'Patient' not in key]
HYPERTONICITY_COLUMNS = [val for key, val in COLUMN_MAPPING.items() if 'Hypertonicity' in key and 'Patient' not in key]
ALL_ANALYSIS_COLUMNS = PARALYSIS_COLUMNS + SYNKINESIS_COLUMNS + HYPERTONICITY_COLUMNS

# <<< CHANGE: Define base types for combined analysis >>>
COMBINED_ANALYSIS_TYPES = list(set(c.replace(' Left','').replace(' Right','') for c in SYNKINESIS_COLUMNS + HYPERTONICITY_COLUMNS))

# --- Label Standardization Functions ---
def standardize_paralysis_label(val):
    """Standardizes paralysis labels to 'None', 'Partial', 'Complete'."""
    if val is None or pd.isna(val): return 'None'
    val_str = str(val).strip().lower()
    if val_str in ['none', 'no', 'n/a', '0', '0.0', 'normal', '', 'nan']: return 'None'
    if val_str in ['partial', 'mild', 'moderate', '1', '1.0', 'p']: return 'Partial'
    if val_str in ['complete', 'severe', '2', '2.0', 'c']: return 'Complete'
    logger.debug(f"Unexpected paralysis label: '{val}'. Defaulting to 'None'.")
    return 'None'

def standardize_binary_label(val):
    """Standardizes binary labels (synkinesis, hypertonicity) to 'Yes', 'No'."""
    if val is None or pd.isna(val): return 'No'
    val_str = str(val).strip().lower()
    # Explicitly check for 'yes'/'no' first
    if val_str == 'yes': return 'Yes'
    if val_str == 'no': return 'No'
    # Then check for other common true/false equivalents
    if val_str in ['y', '1', '1.0', 'true']: return 'Yes'
    if val_str in ['n', '0', '0.0', 'false', '', 'nan']: return 'No'
    # If none of the above match, log it and default to 'No'
    logger.debug(f"Unexpected binary label: '{val}'. Defaulting to 'No'.")
    return 'No'

# --- Visualization Helper ---
def visualize_confusion_matrix(cm, categories, title, output_dir):
    """Generates and saves a heatmap visualization of a confusion matrix."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories, yticklabels=categories, cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f"{title} Confusion Matrix")
        # Create a filesystem-safe filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()[:100].replace(' ','_')
        save_path = os.path.join(output_dir, f"{safe_title}_confusion_matrix.png")
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved: {save_path}")
        plt.close() # Close the figure to free memory
    except Exception as e:
        logger.error(f"Failed to save confusion matrix '{title}': {e}", exc_info=True)
        plt.close() # Ensure plot is closed even on error

# --- Main Analysis Function ---
def analyze_performance(results_csv=RESULTS_FILE, expert_csv=EXPERT_FILE, output_dir=OUTPUT_DIR):
    """
    Loads results and expert data, merges them, standardizes labels,
    calculates performance metrics (side-specific and combined),
    and saves summaries and error details.
    """
    logger.info(f"Starting performance analysis.")
    logger.info(f"Generated results file (script default): {results_csv}")
    logger.info(f"Expert key file (script default): {expert_csv}")
    logger.info(f"Output directory: {output_dir}")

    # --- Load Data ---
    # (Loading logic remains the same)
    current_working_directory = os.getcwd()
    absolute_results_path = os.path.abspath(results_csv)
    absolute_expert_path = os.path.abspath(expert_csv)
    logger.info(f"Current Working Directory: {current_working_directory}")
    logger.info(f"Attempting to load Results file from absolute path: {absolute_results_path}")
    logger.info(f"Attempting to load Expert file from absolute path: {absolute_expert_path}")
    try:
        results_df = pd.read_csv(results_csv)
        logger.info(f"Loaded {len(results_df)} rows from {results_csv}")
    except FileNotFoundError: logger.error(f"Results file not found: {absolute_results_path}"); return
    except Exception as e: logger.error(f"Error loading results: {e}", exc_info=True); return
    try:
        expert_df = pd.read_csv(expert_csv)
        logger.info(f"Loaded {len(expert_df)} rows from {expert_csv}")
    except FileNotFoundError: logger.error(f"Expert key file not found: {absolute_expert_path}"); return
    except Exception as e: logger.error(f"Error loading expert key: {e}", exc_info=True); return

    # --- Prepare Expert Data ---
    # (Preparation logic remains the same, including duplicate Patient ID fix)
    cols_to_select = ['Patient'] + list(COLUMN_MAPPING.keys())
    cols_present_in_expert = [col for col in cols_to_select if col in expert_df.columns]
    missing_expert_cols = [col for col in cols_to_select if col not in expert_df.columns]
    if missing_expert_cols: logger.warning(f"Columns missing from expert key '{expert_csv}': {missing_expert_cols}")
    expert_df_filtered = expert_df[cols_present_in_expert].copy()
    expert_df_renamed = expert_df_filtered.rename(columns=COLUMN_MAPPING)
    logger.debug(f"Expert DF columns after initial renaming: {expert_df_renamed.columns.tolist()}")
    if 'Patient ID' in expert_df_renamed.columns:
        patient_id_cols = expert_df_renamed.columns[expert_df_renamed.columns == 'Patient ID']
        if len(patient_id_cols) > 1:
            logger.warning(f"Found {len(patient_id_cols)} 'Patient ID' cols. Keeping first.")
            expert_df_renamed = expert_df_renamed.loc[:, ~expert_df_renamed.columns.duplicated(keep='first')]
            logger.info(f"Cols after duplicate 'Patient ID' removal: {expert_df_renamed.columns.tolist()}")
            if 'Patient ID' not in expert_df_renamed.columns: logger.error("Failed to resolve duplicate 'Patient ID'."); return
    else: logger.error("'Patient ID' (from 'Patient') not found in expert data."); return

    # --- Standardize Patient IDs ---
    # (Standardization logic remains the same)
    try:
        if 'Patient ID' in results_df.columns: results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
        else: logger.error("'Patient ID' missing in results_df."); return
        if 'Patient ID' in expert_df_renamed.columns: expert_df_renamed['Patient ID'] = expert_df_renamed['Patient ID'].astype(str).str.strip()
        else: logger.error("'Patient ID' missing in expert_df_renamed."); return
    except Exception as e_std: logger.error(f"Error standardizing Patient IDs: {e_std}", exc_info=True); return

    # --- Merge Data ---
    # (Merge logic remains the same)
    try:
        merged_df = pd.merge(results_df, expert_df_renamed, on='Patient ID', how='inner', suffixes=('_pred', '_expert'))
        num_analyzed = len(merged_df)
        logger.info(f"Analyzing {num_analyzed} patients found in both results and expert key files.")
        if merged_df.empty: logger.error("No matching patients after merge."); return
        logger.debug(f"Columns in merged_df after merge: {merged_df.columns.tolist()}")
    except Exception as merge_e: logger.error(f"Merge failed: {merge_e}", exc_info=True); return

    # --- Standardize Labels ---
    # (Standardization logic remains the same, including debug for specific case)
    test_patient_id = 'IMG_0490'
    test_col_base = 'Snarl-Smile Right'
    for col_base in ALL_ANALYSIS_COLUMNS:
        pred_col = f"{col_base}_pred" if f"{col_base}_pred" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in results_df.columns else None)
        expert_col = f"{col_base}_expert" if f"{col_base}_expert" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in expert_df_renamed.columns else None)
        if pred_col and expert_col and pred_col in merged_df.columns and expert_col in merged_df.columns:
            logger.debug(f"Standardizing labels for: {col_base} (Pred: '{pred_col}', Expert: '{expert_col}')")
            # Debug before
            if col_base == test_col_base:
                test_case_row = merged_df[merged_df['Patient ID'] == test_patient_id]
                if not test_case_row.empty:
                    raw_pred_val = test_case_row[pred_col].iloc[0] if pred_col in test_case_row else 'COL_MISS'
                    raw_expert_val = test_case_row[expert_col].iloc[0] if expert_col in test_case_row else 'COL_MISS'
                    logger.debug(f"--- DEBUG {test_patient_id}/{col_base} PRE-STD --- P:'{raw_pred_val}'({type(raw_pred_val)}), E:'{raw_expert_val}'({type(raw_expert_val)})")
            # Apply func
            if col_base in PARALYSIS_COLUMNS: standardize_func = standardize_paralysis_label
            elif col_base in SYNKINESIS_COLUMNS or col_base in HYPERTONICITY_COLUMNS: standardize_func = standardize_binary_label
            else: logger.warning(f"No std rule for '{col_base}'."); continue
            merged_df[pred_col] = merged_df[pred_col].apply(standardize_func)
            merged_df[expert_col] = merged_df[expert_col].apply(standardize_func)
            # Debug after
            if col_base == test_col_base:
                test_case_row_after = merged_df[merged_df['Patient ID'] == test_patient_id]
                if not test_case_row_after.empty:
                    std_pred_val = test_case_row_after[pred_col].iloc[0] if pred_col in test_case_row_after else 'COL_MISS'
                    std_expert_val = test_case_row_after[expert_col].iloc[0] if expert_col in test_case_row_after else 'COL_MISS'
                    logger.debug(f"--- DEBUG {test_patient_id}/{col_base} POST-STD --- P:'{std_pred_val}', E:'{std_expert_val}'")
        elif col_base in ALL_ANALYSIS_COLUMNS: logger.warning(f"Skipping std for '{col_base}': Col missing post-merge.")


    # --- Analyze Performance ---
    performance_summary = {}
    error_details = {}
    logger.info("\n--- Performance Analysis (Side-Specific) ---")

    # --- SIDE-SPECIFIC ANALYSIS LOOP ---
    for col_base in ALL_ANALYSIS_COLUMNS:
        pred_col = f"{col_base}_pred" if f"{col_base}_pred" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in results_df.columns else None)
        expert_col = f"{col_base}_expert" if f"{col_base}_expert" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in expert_df_renamed.columns else None)
        if not pred_col or not expert_col or pred_col not in merged_df.columns or expert_col not in merged_df.columns: continue # Skip if columns missing

        logger.info(f"\n--- Analyzing: {col_base} ---")
        y_true = merged_df[expert_col]
        y_pred = merged_df[pred_col]

        # Determine labels
        if col_base in PARALYSIS_COLUMNS: labels = ['None', 'Partial', 'Complete']
        elif col_base in SYNKINESIS_COLUMNS + HYPERTONICITY_COLUMNS: labels = ['No', 'Yes']
        else: logger.warning(f"Unknown type for '{col_base}'."); continue
        zero_division_setting = 0

        # Calculate and Log Metrics
        try:
            accuracy = accuracy_score(y_true, y_pred)
            report_dict = classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=zero_division_setting)
            precision_w = report_dict.get('weighted avg', {}).get('precision', np.nan)
            recall_w = report_dict.get('weighted avg', {}).get('recall', np.nan)
            f1_w = report_dict.get('weighted avg', {}).get('f1-score', np.nan)
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Weighted Precision: {precision_w:.4f}")
            logger.info(f"Weighted Recall: {recall_w:.4f}")
            logger.info(f"Weighted F1-Score: {f1_w:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=zero_division_setting)}")
            logger.info(f"Confusion Matrix (Rows=True, Cols=Pred):\nLabels: {labels}\n{cm}")

            performance_summary[col_base] = {
                'accuracy': accuracy, 'precision_weighted': precision_w, 'recall_weighted': recall_w, 'f1_weighted': f1_w,
                'class_report': report_dict, 'confusion_matrix': cm.tolist()
            }
            visualize_confusion_matrix(cm, labels, col_base, output_dir)

            # Identify and Store Errors
            id_col = 'Patient ID'
            if id_col in merged_df.columns:
                errors_mask = y_true != y_pred
                errors_df = merged_df.loc[errors_mask, [id_col, expert_col, pred_col]].copy()
                errors_df.rename(columns={expert_col: 'Expert', pred_col: 'Prediction'}, inplace=True)
                if not errors_df.empty: error_details[col_base] = errors_df
                logger.info(f"Found {len(errors_df)} errors for {col_base}.")
            else: logger.warning(f"Cannot perform error analysis for {col_base}: '{id_col}' missing.")

        except ValueError as ve: logger.error(f"Metrics ValueError for {col_base}: {ve}."); performance_summary[col_base] = {'error': f"ValueError: {ve}"}
        except Exception as e: logger.error(f"Metrics/Error ID error for {col_base}: {e}", exc_info=True); performance_summary[col_base] = {'error': str(e)}


    # <<< CHANGE: --- COMBINED SYNKINESIS/HYPERTONICITY ANALYSIS --- >>>
    logger.info("\n\n--- Performance Analysis (Combined Synkinesis/Hypertonicity Types) ---")
    for type_base in COMBINED_ANALYSIS_TYPES:
        # Construct column names for this type
        pred_col_left = f"{type_base} Left_pred"
        expert_col_left = f"{type_base} Left_expert"
        pred_col_right = f"{type_base} Right_pred"
        expert_col_right = f"{type_base} Right_expert"

        # Check if all necessary columns exist after merge and standardization
        required_cols = [pred_col_left, expert_col_left, pred_col_right, expert_col_right]
        if not all(col in merged_df.columns for col in required_cols):
            logger.warning(f"Skipping combined analysis for '{type_base}': One or more required columns missing in merged_df ({required_cols}).")
            continue

        logger.info(f"\n--- Analyzing Combined: {type_base} ---")

        # Concatenate left and right side data
        y_true_combined = pd.concat([merged_df[expert_col_left], merged_df[expert_col_right]], ignore_index=True)
        y_pred_combined = pd.concat([merged_df[pred_col_left], merged_df[pred_col_right]], ignore_index=True)

        # Define labels for binary classification
        labels = ['No', 'Yes']
        zero_division_setting = 0

        # Calculate and Log Combined Metrics
        try:
            accuracy = accuracy_score(y_true_combined, y_pred_combined)
            report_dict = classification_report(y_true_combined, y_pred_combined, labels=labels, target_names=labels, output_dict=True, zero_division=zero_division_setting)
            # Weighted averages are most representative for potentially imbalanced data
            precision_w = report_dict.get('weighted avg', {}).get('precision', np.nan)
            recall_w = report_dict.get('weighted avg', {}).get('recall', np.nan)
            f1_w = report_dict.get('weighted avg', {}).get('f1-score', np.nan)
            cm = confusion_matrix(y_true_combined, y_pred_combined, labels=labels)

            # Log combined metrics
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Weighted Precision: {precision_w:.4f}")
            logger.info(f"Weighted Recall: {recall_w:.4f}")
            logger.info(f"Weighted F1-Score: {f1_w:.4f}")
            logger.info(f"Classification Report (Combined):\n{classification_report(y_true_combined, y_pred_combined, labels=labels, target_names=labels, zero_division=zero_division_setting)}")
            logger.info(f"Confusion Matrix (Combined - Rows=True, Cols=Pred):\nLabels: {labels}\n{cm}")

            # Store combined metrics in performance_summary
            summary_key = f"{type_base} Combined"
            performance_summary[summary_key] = {
                'accuracy': accuracy, 'precision_weighted': precision_w, 'recall_weighted': recall_w, 'f1_weighted': f1_w,
                'class_report': report_dict, 'confusion_matrix': cm.tolist()
            }
            # Generate CM visualization for combined results
            visualize_confusion_matrix(cm, labels, summary_key, output_dir)

        except ValueError as ve:
            logger.error(f"Metrics ValueError for combined {type_base}: {ve}.")
            performance_summary[f"{type_base} Combined"] = {'error': f"ValueError: {ve}"}
        except Exception as e:
            logger.error(f"Metrics error for combined {type_base}: {e}", exc_info=True)
            performance_summary[f"{type_base} Combined"] = {'error': str(e)}
    # <<< END CHANGE >>>


    # --- Save Summary and Errors ---
    # (Saving logic needs update to include combined results)
    try:
        # Save Performance Summary Text File
        summary_file = os.path.join(output_dir, 'performance_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Performance Analysis Summary ({pd.Timestamp.now()})\n")
            f.write(f"Analyzed {num_analyzed} patients.\n")
            f.write("========================================\n\n")

            # <<< CHANGE: Iterate through summary keys, including combined ones >>>
            for col, metrics in performance_summary.items():
                f.write(f"--- METRICS: {col} ---\n")
                if 'error' in metrics: f.write(f"  Error: {metrics['error']}\n")
                else:
                    f.write(f"  Accuracy: {metrics.get('accuracy', np.nan):.4f}\n")
                    f.write(f"  Weighted Precision: {metrics.get('precision_weighted', np.nan):.4f}\n")
                    f.write(f"  Weighted Recall: {metrics.get('recall_weighted', np.nan):.4f}\n")
                    f.write(f"  Weighted F1-Score: {metrics.get('f1_weighted', np.nan):.4f}\n")
                    f.write("  Class Metrics (P/R/F1/Support):\n")
                    report_data = metrics.get('class_report', {})

                    # <<< CHANGE: Determine labels dynamically based on key >>>
                    if 'Paralysis' in col: report_labels = ['None', 'Partial', 'Complete']
                    elif 'Combined' in col or any(ctype in col for ctype in COMBINED_ANALYSIS_TYPES): report_labels = ['No', 'Yes']
                    elif col in ALL_ANALYSIS_COLUMNS : # Side-specific binary
                        report_labels = ['No', 'Yes']
                    else: report_labels = list(report_data.keys()) # Fallback

                    present_labels = [label for label in report_labels if label in report_data and isinstance(report_data[label], dict)]
                    for label in present_labels:
                        scores = report_data.get(label, {})
                        f.write(f"    {label:<10}: P={scores.get('precision', 0):.3f}, R={scores.get('recall', 0):.3f}, F1={scores.get('f1-score', 0):.3f}, Sup={scores.get('support', 0)}\n")
                    for avg_key in ['macro avg', 'weighted avg']:
                         avg_scores = report_data.get(avg_key)
                         if avg_scores and isinstance(avg_scores, dict):
                             f.write(f"    {avg_key:<10}: P={avg_scores.get('precision', 0):.3f}, R={avg_scores.get('recall', 0):.3f}, F1={avg_scores.get('f1-score', 0):.3f}, Sup={avg_scores.get('support', 0)}\n")
                    cm_labels_str = ", ".join(present_labels)
                    f.write(f"  Confusion Matrix (Rows=True, Cols=Pred):\n     Labels: {cm_labels_str}\n{np.array(metrics.get('confusion_matrix', 'N/A'))}\n")
                f.write("-" * 40 + "\n\n")
        logger.info(f"Performance summary saved to {summary_file}")

        # Save Detailed Errors to Excel (Error details remain side-specific)
        errors_file = os.path.join(output_dir, 'error_details.xlsx')
        try:
            with pd.ExcelWriter(errors_file, engine='openpyxl') as writer:
                if not error_details: pd.DataFrame([{'Status': 'No Misclassifications Found'}]).to_excel(writer, sheet_name='Summary', index=False)
                else:
                    for col, df_errors in error_details.items():
                        safe_sheet_name = "".join(c for c in col if c.isalnum() or c in (' ', '_')).rstrip()[:30].replace(' ','_')
                        if not df_errors.empty: df_errors.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                        else: pd.DataFrame([{'Status': f'No Errors for {col}'}]).to_excel(writer, sheet_name=safe_sheet_name, index=False)
            if error_details: logger.info(f"Detailed error cases saved to {errors_file}")
            else: logger.info("No errors found to save in detail.")
        except Exception as e_excel: logger.error(f"Failed to save errors Excel '{errors_file}': {e_excel}", exc_info=True)

    except Exception as e_save: logger.error(f"Error saving summary/error files: {e_save}", exc_info=True)

    logger.info("Performance analysis finished.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Facial AU Pipeline Performance")
    parser.add_argument('--results', default=RESULTS_FILE, help=f"Path to combined_results.csv (default: {RESULTS_FILE})")
    parser.add_argument('--expert', default=EXPERT_FILE, help=f"Path to expert key CSV (default: {EXPERT_FILE})")
    parser.add_argument('--output', default=OUTPUT_DIR, help=f"Directory to save results (default: {OUTPUT_DIR})")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    analyze_performance(results_csv=args.results, expert_csv=args.expert, output_dir=args.output)