# performance_analyzer.py
# Analyzes ML predictions against an expert key.
# - Updated to analyze Mentalis Synkinesis and Hypertonicity columns
# - Added robust duplicate column check and fix for 'Patient ID'
# - Added debug logging for specific case comparison
# - Hardcoded DEBUG logging level

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
log_level = logging.INFO # <<< SET TO DEBUG >>>
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
    calculates performance metrics, and saves summaries and error details.
    """
    logger.info(f"Starting performance analysis.")
    logger.info(f"Generated results file (script default): {results_csv}")
    logger.info(f"Expert key file (script default): {expert_csv}")
    logger.info(f"Output directory: {output_dir}")

    # --- ADDED: Log CWD and exact path before loading ---
    current_working_directory = os.getcwd()
    absolute_results_path = os.path.abspath(results_csv)
    absolute_expert_path = os.path.abspath(expert_csv)
    logger.info(f"Current Working Directory: {current_working_directory}")
    logger.info(f"Attempting to load Results file from absolute path: {absolute_results_path}")
    logger.info(f"Attempting to load Expert file from absolute path: {absolute_expert_path}")
    # --- END ADDED LOGGING ---

    # --- Load Data ---
    try:
        results_df = pd.read_csv(results_csv)
        logger.info(f"Loaded {len(results_df)} rows from {results_csv}")
    except FileNotFoundError:
        logger.error(f"Results file not found using path: {results_csv} (Absolute: {absolute_results_path})")
        return
    except Exception as e: logger.error(f"Error loading results file {results_csv}: {e}", exc_info=True); return

    try:
        expert_df = pd.read_csv(expert_csv)
        logger.info(f"Loaded {len(expert_df)} rows from {expert_csv}")
    except FileNotFoundError:
        logger.error(f"Expert key file not found using path: {expert_csv} (Absolute: {absolute_expert_path})")
        return
    except Exception as e: logger.error(f"Error loading expert key file {expert_csv}: {e}", exc_info=True); return

    # --- Prepare Expert Data ---
    # Select only relevant columns based on COLUMN_MAPPING keys present in the expert file
    cols_to_select = ['Patient'] + list(COLUMN_MAPPING.keys())
    cols_present_in_expert = [col for col in cols_to_select if col in expert_df.columns]
    missing_expert_cols = [col for col in cols_to_select if col not in expert_df.columns]
    if missing_expert_cols:
        logger.warning(f"Columns missing from expert key file '{expert_csv}': {missing_expert_cols}")

    # Filter and rename expert columns
    expert_df_filtered = expert_df[cols_present_in_expert].copy()
    expert_df_renamed = expert_df_filtered.rename(columns=COLUMN_MAPPING)
    logger.debug(f"Expert DF columns after initial renaming: {expert_df_renamed.columns.tolist()}")

    # --- Robust Check & Fix for Duplicate 'Patient ID' columns AFTER renaming ---
    if 'Patient ID' in expert_df_renamed.columns:
        patient_id_cols = expert_df_renamed.columns[expert_df_renamed.columns == 'Patient ID']
        if len(patient_id_cols) > 1:
            logger.warning(f"Found {len(patient_id_cols)} columns named 'Patient ID' after renaming expert data. Attempting to keep only the first instance.")
            # Keep the first 'Patient ID' column, remove subsequent duplicates
            expert_df_renamed = expert_df_renamed.loc[:, ~expert_df_renamed.columns.duplicated(keep='first')]
            logger.info(f"Columns after duplicate 'Patient ID' removal attempt: {expert_df_renamed.columns.tolist()}")
            if 'Patient ID' not in expert_df_renamed.columns: # Check if it was somehow removed entirely
                 logger.error("Failed to resolve duplicate 'Patient ID' columns. Aborting.")
                 return
    else:
         # This case happens if 'Patient' was not in the original expert_df columns
         logger.error("Column 'Patient ID' (expected from renaming 'Patient') not found in expert data. Cannot proceed.")
         return
    # --- End Duplicate Check ---

    # --- Standardize Patient IDs (Crucial for Merge) ---
    try:
        if 'Patient ID' in results_df.columns:
            logger.debug(f"Standardizing results_df['Patient ID']")
            results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
        else:
            logger.error("'Patient ID' column missing in results_df. Cannot merge."); return

        if 'Patient ID' in expert_df_renamed.columns:
             logger.debug(f"Standardizing expert_df_renamed['Patient ID']")
             # Now we are more confident 'Patient ID' is unique and present
             expert_df_renamed['Patient ID'] = expert_df_renamed['Patient ID'].astype(str).str.strip()
             logger.debug("Successfully standardized expert 'Patient ID'.")
        else:
            # Should not happen given the checks above, but added for safety
            logger.error("'Patient ID' column missing in expert_df_renamed after checks. Cannot merge."); return
    except Exception as e_std:
        logger.error(f"Error during Patient ID standardization: {e_std}", exc_info=True)
        return

    # --- Merge Data ---
    try:
        # Use suffixes to distinguish columns if names clash (e.g., if 'Hypertonicity Left' exists in both)
        merged_df = pd.merge(results_df, expert_df_renamed, on='Patient ID', how='inner', suffixes=('_pred', '_expert'))
        num_analyzed = len(merged_df)
        logger.info(f"Analyzing {num_analyzed} patients found in both results and expert key files.")
        if merged_df.empty:
            logger.error("No matching patients found between results and expert key after merge. Check 'Patient ID' values.")
            return

        # --- ADD LOGGING: Columns after merge ---
        logger.debug(f"Columns in merged_df after merge: {merged_df.columns.tolist()}")
        # --- END LOGGING ---

    except Exception as merge_e:
        logger.error(f"Merge operation failed: {merge_e}", exc_info=True)
        return

    # --- Standardize Labels & ADD DEBUG FOR SPECIFIC CASE ---
    # Define the case you want to debug specifically
    test_patient_id = 'IMG_0490'        # <<< CHANGE AS NEEDED >>> Patient known to have the issue
    test_col_base = 'Snarl-Smile Right' # <<< CHANGE AS NEEDED >>> Column known to have the issue

    for col_base in ALL_ANALYSIS_COLUMNS:
        # Determine the final prediction and expert column names after potential merge suffixing
        # Check for suffixed column first, then original name (if it didn't clash during merge)
        pred_col = f"{col_base}_pred" if f"{col_base}_pred" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in results_df.columns else None)
        expert_col = f"{col_base}_expert" if f"{col_base}_expert" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in expert_df_renamed.columns else None)

        # Proceed only if both prediction and expert columns were found in the merged data
        if pred_col and expert_col and pred_col in merged_df.columns and expert_col in merged_df.columns:
            logger.debug(f"Standardizing labels for: {col_base} (Pred Col: '{pred_col}', Expert Col: '{expert_col}')")

            # --- ADD LOGGING: Before Standardization (for test case) ---
            # Force logging for the specific test case regardless of log level setting above
            if col_base == test_col_base:
                test_case_row = merged_df[merged_df['Patient ID'] == test_patient_id]
                if not test_case_row.empty:
                    # Safely get values using .get() in case columns somehow still missing for this specific row
                    raw_pred_val = test_case_row[pred_col].iloc[0] if pred_col in test_case_row else 'COLUMN_MISSING'
                    raw_expert_val = test_case_row[expert_col].iloc[0] if expert_col in test_case_row else 'COLUMN_MISSING'
                    logger.debug(f"--- DEBUG {test_patient_id} / {col_base} ---")
                    logger.debug(f"BEFORE Standardization: Raw Pred ('{pred_col}') = '{raw_pred_val}' (Type: {type(raw_pred_val)}), Raw Expert ('{expert_col}') = '{raw_expert_val}' (Type: {type(raw_expert_val)})")
                else:
                    logger.debug(f"--- DEBUG {test_patient_id} / {col_base} --- Patient ID not found in merged_df for debug logging.")
            # --- END LOGGING ---

            # Apply the appropriate standardization function based on column type
            if col_base in PARALYSIS_COLUMNS:
                standardize_func = standardize_paralysis_label
            elif col_base in SYNKINESIS_COLUMNS or col_base in HYPERTONICITY_COLUMNS:
                standardize_func = standardize_binary_label
            else:
                logger.warning(f"No standardization rule defined for '{col_base}'. Skipping standardization.")
                continue # Skip if type unknown

            # Apply standardization IN PLACE on the merged dataframe
            merged_df[pred_col] = merged_df[pred_col].apply(standardize_func)
            merged_df[expert_col] = merged_df[expert_col].apply(standardize_func)

            # --- ADD LOGGING: After Standardization (for test case) ---
            if col_base == test_col_base:
                test_case_row_after = merged_df[merged_df['Patient ID'] == test_patient_id]
                if not test_case_row_after.empty:
                    std_pred_val = test_case_row_after[pred_col].iloc[0] if pred_col in test_case_row_after else 'COLUMN_MISSING'
                    std_expert_val = test_case_row_after[expert_col].iloc[0] if expert_col in test_case_row_after else 'COLUMN_MISSING'
                    logger.debug(f"AFTER Standardization: Std Pred ('{pred_col}') = '{std_pred_val}', Std Expert ('{expert_col}') = '{std_expert_val}'")
                    logger.debug(f"--- END DEBUG {test_patient_id} / {col_base} ---")
                # No need for else, handled above
            # --- END LOGGING ---

        else:
             # Log if we intended to analyze this column but couldn't find its parts after merge
             if col_base in ALL_ANALYSIS_COLUMNS:
                 logger.warning(f"Skipping standardization and analysis for '{col_base}': Column missing after merge (Tried Pred='{pred_col}', Expert='{expert_col}'). Check results/expert files and mapping.")

    # --- Analyze Performance ---
    performance_summary = {}
    error_details = {} # Dictionary to hold DataFrames of errors for each analysis column
    logger.info("\n--- Performance Analysis ---")

    for col_base in ALL_ANALYSIS_COLUMNS:
        # Re-determine the final column names after merge and potential suffixing
        pred_col = f"{col_base}_pred" if f"{col_base}_pred" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in results_df.columns else None)
        expert_col = f"{col_base}_expert" if f"{col_base}_expert" in merged_df.columns else (col_base if col_base in merged_df.columns and col_base in expert_df_renamed.columns else None)

        # Skip if columns aren't present in the merged data
        if not pred_col or not expert_col or pred_col not in merged_df.columns or expert_col not in merged_df.columns:
            # This warning is now logged during standardization loop, no need to repeat unless debugging specifics
            # logger.warning(f"Skipping metric calculation for '{col_base}': Standardized column(s) missing.")
            continue

        logger.info(f"\n--- Analyzing: {col_base} ---")
        y_true = merged_df[expert_col] # Should contain standardized labels
        y_pred = merged_df[pred_col] # Should contain standardized labels

        # --- ADD LOGGING: Comparison Result (for test case) ---
        if col_base == test_col_base:
             test_case_row_final = merged_df[merged_df['Patient ID'] == test_patient_id]
             if not test_case_row_final.empty:
                 true_val = test_case_row_final[expert_col].iloc[0] if expert_col in test_case_row_final else 'COLUMN_MISSING'
                 pred_val = test_case_row_final[pred_col].iloc[0] if pred_col in test_case_row_final else 'COLUMN_MISSING'
                 is_error = true_val != pred_val if true_val != 'COLUMN_MISSING' and pred_val != 'COLUMN_MISSING' else 'Cannot Compare'
                 logger.debug(f"--- COMPARISON {test_patient_id} / {col_base} ---")
                 logger.debug(f"Comparing Final Standardized: y_true ('{true_val}') != y_pred ('{pred_val}') -> Is Error? {is_error}")
                 logger.debug(f"--- END COMPARISON {test_patient_id} / {col_base} ---")
             # No else needed
        # --- END LOGGING ---

        # Determine labels for metrics based on column type
        if col_base in PARALYSIS_COLUMNS:
            labels = ['None', 'Partial', 'Complete']
        elif col_base in SYNKINESIS_COLUMNS or col_base in HYPERTONICITY_COLUMNS:
            labels = ['No', 'Yes']
        else:
            logger.warning(f"Unknown analysis type for '{col_base}'. Skipping metrics.")
            continue
        zero_division_setting = 0 # Set to 0 to avoid warnings when a class has no predictions/support

        # Calculate and Log Metrics
        try:
            accuracy = accuracy_score(y_true, y_pred)
            # Use output_dict=True for easier parsing later
            report_dict = classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=zero_division_setting)
            # Calculate weighted averages manually from dict if needed, or use dict directly
            precision_w = report_dict.get('weighted avg', {}).get('precision', np.nan)
            recall_w = report_dict.get('weighted avg', {}).get('recall', np.nan)
            f1_w = report_dict.get('weighted avg', {}).get('f1-score', np.nan)
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            # Log overall metrics
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Weighted Precision: {precision_w:.4f}")
            logger.info(f"Weighted Recall: {recall_w:.4f}")
            logger.info(f"Weighted F1-Score: {f1_w:.4f}")
            # Log text report for readability
            logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=zero_division_setting)}")
            # Log confusion matrix
            logger.info(f"Confusion Matrix (Rows=True, Cols=Pred):\nLabels: {labels}\n{cm}")

            # Store metrics for summary file
            performance_summary[col_base] = {
                'accuracy': accuracy, 'precision_weighted': precision_w, 'recall_weighted': recall_w, 'f1_weighted': f1_w,
                'class_report': report_dict, 'confusion_matrix': cm.tolist()
            }
            # Generate CM visualization
            visualize_confusion_matrix(cm, labels, col_base, output_dir)

            # --- Identify and Store Errors ---
            id_col = 'Patient ID' # Ensure this matches the actual column name
            if id_col in merged_df.columns:
                # Find rows where standardized true and pred labels differ
                errors_mask = y_true != y_pred
                # Select relevant columns for error report
                errors_df = merged_df.loc[errors_mask, [id_col, expert_col, pred_col]].copy()
                # Rename for clarity in the output file
                errors_df.rename(columns={expert_col: 'Expert', pred_col: 'Prediction'}, inplace=True)
                if not errors_df.empty:
                    error_details[col_base] = errors_df # Store the DataFrame of errors
                    logger.info(f"Found {len(errors_df)} errors for {col_base}.")
            else:
                logger.warning(f"Cannot perform error analysis for {col_base}: '{id_col}' column missing in merged_df.")
            # --- End Error Identification ---

        except ValueError as ve:
            logger.error(f"ValueError calculating metrics for {col_base}: {ve}. Check labels and data types.")
            performance_summary[col_base] = {'error': f"ValueError: {ve}"}
        except Exception as e:
            logger.error(f"Error calculating metrics or identifying errors for {col_base}: {e}", exc_info=True)
            performance_summary[col_base] = {'error': str(e)}

    # --- Save Summary and Errors ---
    try:
        # Save Performance Summary Text File
        summary_file = os.path.join(output_dir, 'performance_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Performance Analysis Summary ({pd.Timestamp.now()})\n")
            f.write(f"Analyzed {num_analyzed} patients.\n")
            f.write("========================================\n\n")
            for col, metrics in performance_summary.items():
                f.write(f"--- METRICS: {col} ---\n")
                if 'error' in metrics:
                    f.write(f"  Error: {metrics['error']}\n")
                else:
                    # Write overall metrics
                    f.write(f"  Accuracy: {metrics.get('accuracy', np.nan):.4f}\n")
                    f.write(f"  Weighted Precision: {metrics.get('precision_weighted', np.nan):.4f}\n")
                    f.write(f"  Weighted Recall: {metrics.get('recall_weighted', np.nan):.4f}\n")
                    f.write(f"  Weighted F1-Score: {metrics.get('f1_weighted', np.nan):.4f}\n")
                    # Write class-specific metrics
                    f.write("  Class Metrics (P/R/F1/Support):\n")
                    report_data = metrics.get('class_report', {})
                    # Determine appropriate labels based on analysis type
                    if col in PARALYSIS_COLUMNS: report_labels = ['None', 'Partial', 'Complete']
                    elif col in SYNKINESIS_COLUMNS or col in HYPERTONICITY_COLUMNS: report_labels = ['No', 'Yes']
                    else: report_labels = list(report_data.keys()) # Fallback
                    # Only report labels that actually exist in the report dictionary
                    present_labels = [label for label in report_labels if label in report_data and isinstance(report_data[label], dict)]
                    for label in present_labels:
                        scores = report_data.get(label, {})
                        f.write(f"    {label:<10}: P={scores.get('precision', 0):.3f}, R={scores.get('recall', 0):.3f}, F1={scores.get('f1-score', 0):.3f}, Sup={scores.get('support', 0)}\n")
                    # Report averages if they exist
                    for avg_key in ['macro avg', 'weighted avg']:
                         avg_scores = report_data.get(avg_key)
                         if avg_scores and isinstance(avg_scores, dict):
                             f.write(f"    {avg_key:<10}: P={avg_scores.get('precision', 0):.3f}, R={avg_scores.get('recall', 0):.3f}, F1={avg_scores.get('f1-score', 0):.3f}, Sup={avg_scores.get('support', 0)}\n")
                    # Write confusion matrix
                    cm_labels_str = ", ".join(present_labels)
                    f.write(f"  Confusion Matrix (Rows=True, Cols=Pred):\n     Labels: {cm_labels_str}\n{np.array(metrics.get('confusion_matrix', 'N/A'))}\n")
                f.write("-" * 40 + "\n\n")
        logger.info(f"Performance summary saved to {summary_file}")

        # Save Detailed Errors to Excel
        errors_file = os.path.join(output_dir, 'error_details.xlsx')
        try:
            with pd.ExcelWriter(errors_file, engine='openpyxl') as writer:
                if not error_details:
                    # Write a placeholder if no errors found at all
                    pd.DataFrame([{'Status': 'No Misclassifications Found Across All Analyses'}]).to_excel(writer, sheet_name='Summary', index=False)
                else:
                    # Write each error DataFrame to a separate sheet
                    for col, df_errors in error_details.items():
                        # Sanitize sheet name (Excel limits sheet name length and characters)
                        safe_sheet_name = "".join(c for c in col if c.isalnum() or c in (' ', '_')).rstrip()[:30].replace(' ','_')
                        if not df_errors.empty:
                            df_errors.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                        else:
                            # Optionally write a sheet indicating no errors for this specific category
                            pd.DataFrame([{'Status': f'No Errors for {col}'}]).to_excel(writer, sheet_name=safe_sheet_name, index=False)
            if error_details:
                logger.info(f"Detailed error cases saved to {errors_file}")
            else:
                logger.info("No errors found to save in detail.")
        except Exception as e_excel:
            logger.error(f"Failed to save error details Excel file '{errors_file}': {e_excel}", exc_info=True)

    except Exception as e_save:
        logger.error(f"Error saving summary/error files: {e_save}", exc_info=True)

    logger.info("Performance analysis finished.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Arguments are now only for file paths, debug is hardcoded
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Facial AU Pipeline Performance against Expert Key")
    parser.add_argument('--results', default=RESULTS_FILE,
                        help=f"Path to the combined_results.csv file generated by the pipeline (default: {RESULTS_FILE})")
    parser.add_argument('--expert', default=EXPERT_FILE,
                        help=f"Path to the expert key CSV file (default: {EXPERT_FILE})")
    parser.add_argument('--output', default=OUTPUT_DIR,
                        help=f"Directory to save analysis results (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Run the analysis (logging level is already set to DEBUG above)
    analyze_performance(results_csv=args.results, expert_csv=args.expert, output_dir=args.output)