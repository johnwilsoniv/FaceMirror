# lower_face_performance_analysis.py (Include Patient ID in Errors + Full Dict Debug Logging)

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import json # <--- Import json for logging dictionaries
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# import sys # Keep commented out unless debugging needed

# --- Import necessary components ---
from lower_face_detector import LowerFaceParalysisDetector
try:
    from lower_face_features import extract_features_for_detection
except ImportError:
    logging.error("Could not import extract_features_for_detection for debugging.")
    extract_features_for_detection = None

try:
    from lower_face_config import LOG_DIR, CLASS_NAMES
except ImportError:
    logging.warning("Could not import from lower_face_config."); LOG_DIR = 'logs'; CLASS_NAMES = {0: 'None', 1: 'Partial', 2: 'Complete'}
# --- End Imports ---

# Configure logging
os.makedirs('analysis_results', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join('analysis_results', 'lowerface_analysis.log'), mode='w'), logging.StreamHandler()], force=True)
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# --- Initialize the Detector ---
try:
    lower_face_detector = LowerFaceParalysisDetector(); logger.info("Lower Face Detector initialized successfully.")
except Exception as e: logger.error(f"Failed to initialize LowerFaceParalysisDetector: {e}", exc_info=True); lower_face_detector = None
# --- End Detector Init ---


def analyze_performance(results_file='combined_results.csv', expert_file='FPRS FP Key.csv',
                      output_dir='analysis_results'):
    """ Analyze LOWER FACE model performance by re-running detection. """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting LOWER FACE performance analysis by re-running detection")

    if lower_face_detector is None: logger.error("Lower face detector not available."); return
    # if extract_features_for_detection is None: logger.error("extract_features_for_detection not imported.")

    try:
        logger.info(f"Loading results from {results_file}")
        results_df = pd.read_csv(results_file, low_memory=False)
        logger.info(f"Loading expert labels from {expert_file}")
        expert_df = pd.read_csv(expert_file)

        expert_df = expert_df.rename(columns={
            'Patient': 'Patient ID',
            'Paralysis - Left Lower Face': 'Expert_Left_Lower_Face',
            'Paralysis - Right Lower Face': 'Expert_Right_Lower_Face'})

        expert_cols_to_standardize = ['Expert_Left_Lower_Face', 'Expert_Right_Lower_Face']
        for col in expert_cols_to_standardize:
             if col in expert_df.columns: expert_df[col] = expert_df[col].apply(standardize_labels)
             else: logger.warning(f"Expert column {col} not found.")

        results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip()
        expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
        try: merged_df = pd.merge(results_df, expert_df, on='Patient ID', how='inner', validate="many_to_one")
        except Exception as me: logger.error(f"Merge failed: {me}"); return
        logger.info(f"Merged data contains {len(merged_df)} patients")
        if merged_df.empty: logger.error("Merge failed."); return

        logger.info("Re-running lower face detection using loaded models...")
        predictions_left = []; predictions_right = []

        for index, row in merged_df.iterrows():
            current_patient_id = row['Patient ID']
            row_data_dict = row.to_dict()

            # --- START ADDED DEBUG LOGGING ---
            test_patient_id = 'IMG_0422' # <<< CHANGE THIS if your test patient is different
            if current_patient_id == test_patient_id:
                try:
                    # Log before calling detect for Left
                    logger.debug(f"ANALYSIS SCRIPT - Full input row_data_dict for {current_patient_id} (Before Left Detect):")
                    # Use default=str to handle potential NaNs directly from CSV
                    logger.debug(json.dumps(row_data_dict, indent=2, default=str))
                except Exception as log_e:
                    logger.error(f"Error logging full row_data_dict in analysis script (Left): {log_e}")
            # --- END ADDED DEBUG LOGGING ---

            try:
                result_l, _, _ = lower_face_detector.detect(row_data_dict, 'Left', 'lower')
                predictions_left.append(result_l)
            except Exception as e_l:
                logger.error(f"Error predict left lower {current_patient_id}: {e_l}")
                predictions_left.append('Error')

            # --- START ADDED DEBUG LOGGING ---
            if current_patient_id == test_patient_id:
                try:
                    # Log before calling detect for Right
                    logger.debug(f"ANALYSIS SCRIPT - Full input row_data_dict for {current_patient_id} (Before Right Detect):")
                    # Use default=str to handle potential NaNs directly from CSV
                    logger.debug(json.dumps(row_data_dict, indent=2, default=str))
                except Exception as log_e:
                    logger.error(f"Error logging full row_data_dict in analysis script (Right): {log_e}")
            # --- END ADDED DEBUG LOGGING ---

            try:
                result_r, _, _ = lower_face_detector.detect(row_data_dict, 'Right', 'lower')
                predictions_right.append(result_r)
            except Exception as e_r:
                logger.error(f"Error predict right lower {current_patient_id}: {e_r}")
                predictions_right.append('Error')

        merged_df['ML_Pred_Left_Lower'] = predictions_left
        merged_df['ML_Pred_Right_Lower'] = predictions_right

        analyze_zone(merged_df, 'Lower Face', 'ML_Pred_Left_Lower', 'Expert_Left_Lower_Face', 'ML_Pred_Right_Lower', 'Expert_Right_Lower_Face', output_dir)
        logger.info("Performance analysis complete")

    except Exception as e: logger.error(f"Error in performance analysis: {str(e)}", exc_info=True)

# --- standardize_labels function ---
def standardize_labels(val):
    # (Keep this function as before)
    if val is None or pd.isna(val): return 'None'
    val_str = str(val).strip().lower()
    if val_str in ['none', 'no', 'n/a', '0', '0.0', 'normal', '', 'nan']: return 'None'
    if val_str in ['partial', 'mild', 'moderate', '1', '1.0']: return 'Partial'
    if val_str in ['complete', 'severe', '2', '2.0']: return 'Complete'
    logger.warning(f"Unexpected label: '{val}'. Defaulting to 'None'.")
    return 'None'

# --- analyze_zone function (MODIFIED to keep Patient ID) ---
def analyze_zone(data, zone_name, pred_left_col, expert_left_col, pred_right_col, expert_right_col, output_dir):
    """ Analyze model performance, keeping Patient ID. """
    logger.info(f"Analyzing {zone_name} performance using ML Predictions")
    required_cols = ['Patient ID', pred_left_col, expert_left_col, pred_right_col, expert_right_col]
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]; logger.error(f"Missing columns: {missing}. Abort."); return

    left_analysis = data[['Patient ID', pred_left_col, expert_left_col]].copy()
    right_analysis = data[['Patient ID', pred_right_col, expert_right_col]].copy()
    left_analysis['Side'] = 'Left'; right_analysis['Side'] = 'Right'
    left_analysis.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']
    right_analysis.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']

    left_analysis['Prediction'] = left_analysis['Prediction'].apply(standardize_labels); left_analysis['Expert'] = left_analysis['Expert'].apply(standardize_labels)
    right_analysis['Prediction'] = right_analysis['Prediction'].apply(standardize_labels); right_analysis['Expert'] = right_analysis['Expert'].apply(standardize_labels)

    zone_df = pd.concat([left_analysis, right_analysis], ignore_index=True)
    categories = ['None', 'Partial', 'Complete']
    logger.info(f"Sample data for {zone_name} analysis:\n{zone_df.head().to_string()}")

    try:
        zone_df_valid = zone_df[zone_df['Prediction'] != 'Error'].copy()
        if zone_df_valid.empty: logger.error(f"No valid predictions for {zone_name}."); return
        left_analysis_valid = zone_df_valid[zone_df_valid['Side'] == 'Left']
        right_analysis_valid = zone_df_valid[zone_df_valid['Side'] == 'Right']

        if not left_analysis_valid.empty: left_report = classification_report(left_analysis_valid['Expert'], left_analysis_valid['Prediction'], labels=categories, target_names=categories, output_dict=True, zero_division=0); logger.info(f"Left {zone_name} accuracy: {left_report['accuracy']:.4f}")
        else: left_report = {'accuracy': np.nan, 'Partial': {}, 'Complete': {}}
        if not right_analysis_valid.empty: right_report = classification_report(right_analysis_valid['Expert'], right_analysis_valid['Prediction'], labels=categories, target_names=categories, output_dict=True, zero_division=0); logger.info(f"Right {zone_name} accuracy: {right_report['accuracy']:.4f}")
        else: right_report = {'accuracy': np.nan, 'Partial': {}, 'Complete': {}}
        combined_report = classification_report(zone_df_valid['Expert'], zone_df_valid['Prediction'], labels=categories, target_names=categories, output_dict=True, zero_division=0); logger.info(f"Combined {zone_name} accuracy: {combined_report['accuracy']:.4f}")

        partial_metrics = {'Left': left_report.get('Partial', {}), 'Right': right_report.get('Partial', {}), 'Combined': combined_report.get('Partial', {})}
        logger.info(f"Partial paralysis metrics:"); [logger.info(f"  {s} - P: {m.get('precision', 0):.4f}, R: {m.get('recall', 0):.4f}, F1: {m.get('f1-score', 0):.4f}") for s, m in partial_metrics.items()]
        complete_metrics = {'Left': left_report.get('Complete', {}), 'Right': right_report.get('Complete', {}), 'Combined': combined_report.get('Complete', {})}
        logger.info(f"Complete paralysis metrics:"); [logger.info(f"  {s} - P: {m.get('precision', 0):.4f}, R: {m.get('recall', 0):.4f}, F1: {m.get('f1-score', 0):.4f}") for s, m in complete_metrics.items()]

        if not left_analysis_valid.empty: left_cm = confusion_matrix(left_analysis_valid['Expert'], left_analysis_valid['Prediction'], labels=categories); visualize_confusion_matrix(left_cm, categories, f"Left {zone_name} (ML Preds)", output_dir)
        if not right_analysis_valid.empty: right_cm = confusion_matrix(right_analysis_valid['Expert'], right_analysis_valid['Prediction'], labels=categories); visualize_confusion_matrix(right_cm, categories, f"Right {zone_name} (ML Preds)", output_dir)
        combined_cm = confusion_matrix(zone_df_valid['Expert'], zone_df_valid['Prediction'], labels=categories); visualize_confusion_matrix(combined_cm, categories, f"Combined {zone_name} (ML Preds)", output_dir)

        perform_error_analysis(zone_df_valid, output_dir, f"{zone_name}_errors_ML_Preds")
        analyze_critical_errors(zone_df_valid, output_dir, f"lowerface_critical_errors") # Keep consistent filename
        analyze_partial_errors(zone_df_valid, output_dir, f"lowerface_partial_errors") # Keep consistent filename

    except Exception as e: logger.error(f"Error during metrics/vis for {zone_name}: {str(e)}", exc_info=True)

# --- visualize_confusion_matrix (No changes needed) ---
def visualize_confusion_matrix(cm, categories, title, output_dir):
    # (Keep as before)
    try: plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar=False); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f"{title} Confusion Matrix"); save_path = os.path.join(output_dir, f"{title.replace(' ', '_').replace('(','').replace(')','')}_confusion_matrix.png"); plt.tight_layout(); plt.savefig(save_path); logger.info(f"CM saved: {save_path}"); plt.close()
    except Exception as e: logger.error(f"Failed save CM {title}: {e}"); plt.close()

# --- perform_error_analysis (MODIFIED to include Patient ID) ---
def perform_error_analysis(data, output_dir, filename):
    """ Perform detailed error analysis including Patient ID. """
    error_cases = data[data['Prediction'] != data['Expert']].copy(); error_patterns = {}
    for _, row in error_cases.iterrows(): pattern = f"{row['Expert']}_to_{row['Prediction']}"; error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
    logger.info(f"Error patterns for {filename}:"); [logger.info(f"- {p}: {c} cases") for p, c in sorted(error_patterns.items())] if error_patterns else logger.info("- No errors found.")
    path = os.path.join(output_dir, f"{filename}.txt"); total_samples = len(data); total_errors = len(error_cases)
    try:
        with open(path, 'w') as f:
            f.write(f"Error Analysis: {filename}\n============={'='*len(filename)}\n\n")
            f.write(f"Total errors: {total_errors}/{total_samples} ({total_errors/total_samples*100:.2f}%)\n\n" if total_samples > 0 else f"Total errors: {total_errors}/0 (N/A %)\n\n")
            f.write("Error patterns:\n")
            if total_errors > 0: [f.write(f"- {p}: {c} cases ({c/total_errors*100:.2f}% of errors)\n") for p, c in sorted(error_patterns.items())]
            else: f.write("- No errors found.\n")
            f.write("\nDetailed error cases:\n"); error_cases_reset = error_cases.reset_index()
            for idx, row in error_cases_reset.iterrows():
                # --- Include Patient ID ---
                f.write(f"Index {idx}, Patient {row['Patient ID']}, Side: {row['Side']} - Exp: {row['Expert']}, Pred: {row['Prediction']}\n")
            # --- End Include Patient ID ---
        logger.info(f"Error analysis saved to {path}")
    except Exception as e: logger.error(f"Failed save error file {path}: {e}", exc_info=True)

# --- analyze_critical_errors (MODIFIED to include Patient ID) ---
def analyze_critical_errors(data, output_dir, filename):
    """ Analyze critical error cases including Patient ID. """
    logger.info("Analyzing critical error cases (None <-> Complete)...")
    critical_errors = data[((data['Expert'] == 'None') & (data['Prediction'] == 'Complete')) | ((data['Expert'] == 'Complete') & (data['Prediction'] == 'None'))].copy()
    n2c = len(critical_errors[(critical_errors['Expert'] == 'None') & (critical_errors['Prediction'] == 'Complete')])
    c2n = len(critical_errors[(critical_errors['Expert'] == 'Complete') & (critical_errors['Prediction'] == 'None')])
    logger.info(f"Found {len(critical_errors)} critical errors (None->Complete: {n2c}, Complete->None: {c2n})")
    path = os.path.join(output_dir, f"{filename}.txt")
    try:
        with open(path, "w") as f:
            f.write(f"Critical Errors Analysis - {filename.split('_')[0]} (None <-> Complete)\n"); f.write(f"{'='*40}\n\n")
            f.write(f"Total Critical Errors: {len(critical_errors)}\n"); f.write(f"  None -> Complete: {n2c}\n"); f.write(f"  Complete -> None: {c2n}\n\n")
            if not critical_errors.empty:
                f.write("Detailed Cases:\n"); critical_errors_reset = critical_errors.reset_index()
                # --- Include Patient ID ---
                for idx, row in critical_errors_reset.iterrows():
                    f.write(f"  Index {idx}, Patient {row['Patient ID']}, Side: {row['Side']} - Exp: {row['Expert']}, Pred: {row['Prediction']}\n")
                # --- End Include Patient ID ---
            else: f.write("No critical errors found.\n")
        logger.info(f"Critical errors analysis saved: {path}")
    except Exception as e: logger.error(f"Failed save critical error file: {e}", exc_info=True)

# --- analyze_partial_errors (MODIFIED to include Patient ID) ---
def analyze_partial_errors(data, output_dir, filename):
    """ Analyze partial misclassification cases including Patient ID. """
    logger.info("Analyzing partial misclassification cases...")
    partial_errors = data[((data['Expert'] == 'Partial') & (data['Prediction'] != 'Partial')) | ((data['Prediction'] == 'Partial') & (data['Expert'] != 'Partial'))].copy()
    p2c = partial_errors[(partial_errors['Expert'] == 'Partial') & (partial_errors['Prediction'] == 'Complete')]; p2n = partial_errors[(partial_errors['Expert'] == 'Partial') & (partial_errors['Prediction'] == 'None')]
    n2p = partial_errors[(partial_errors['Expert'] == 'None') & (partial_errors['Prediction'] == 'Partial')]; c2p = partial_errors[(partial_errors['Expert'] == 'Complete') & (partial_errors['Prediction'] == 'Partial')]
    logger.info(f"Found {len(partial_errors)} partial misclassification cases:"); logger.info(f"  P->C: {len(p2c)}, P->N: {len(p2n)}, N->P: {len(n2p)}, C->P: {len(c2p)}")
    path = os.path.join(output_dir, f"{filename}.txt")
    try:
        with open(path, "w") as f:
            f.write(f"Partial Misclassification Analysis - {filename.split('_')[0]}\n"); f.write(f"{'='*40}\n\n")
            f.write(f"Total Partial Errors: {len(partial_errors)}\n"); f.write(f"  P->C: {len(p2c)}\n"); f.write(f"  P->N: {len(p2n)}\n"); f.write(f"  N->P: {len(n2p)}\n"); f.write(f"  C->P: {len(c2p)}\n\n")
            for name, df_err in [("P->C", p2c), ("P->N", p2n), ("N->P", n2p), ("C->P", c2p)]:
                 if not df_err.empty:
                     f.write(f"{name} Errors ({len(df_err)} cases)\n{'-'*(len(name)+14)}\n"); df_err_reset = df_err.reset_index()
                     # --- Include Patient ID ---
                     for idx, row in df_err_reset.iterrows():
                         f.write(f"  Index {idx}, Patient {row['Patient ID']}, Side: {row['Side']}\n")
                     # --- End Include Patient ID ---
                     f.write("\n")
                 else: f.write(f"{name} Errors: 0 cases\n\n")
        logger.info(f"Partial errors analysis saved: {path}")
    except Exception as e: logger.error(f"Failed save partial error file: {e}", exc_info=True)


if __name__ == "__main__":
    analyze_performance()