# snarl_smile_performance_analysis.py
# - FINAL VERSION FOR V7 MODEL
# - Uses SnarlSmileDetector (which internally uses threshold from config)
# - Reports performance based on the detector's output

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
import seaborn as sns
import json # For pretty printing dicts

# --- Import necessary components ---
try:
    from snarl_smile_detector import SnarlSmileDetector
    from snarl_smile_config import LOG_DIR, CLASS_NAMES, MODEL_FILENAMES, DETECTION_THRESHOLD # Import threshold for logging
except ImportError:
    logging.error("Could not import SnarlSmileDetector or config. Analysis cannot run.")
    DETECTION_THRESHOLD = "N/A (Config Import Failed)"
    raise SystemExit("Missing necessary imports for performance analysis.")

# Configure logging
ANALYSIS_OUTPUT_DIR = 'analysis_results'
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(ANALYSIS_OUTPUT_DIR, 'snarl_smile_analysis_final.log') # Final log name
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()], force=True)
logger = logging.getLogger(__name__)

# --- Initialize the Detector ---
snarl_smile_detector = None
try:
    model_path = MODEL_FILENAMES.get('model'); scaler_path = MODEL_FILENAMES.get('scaler'); features_path = MODEL_FILENAMES.get('feature_list')
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path] if p):
        logger.error(f"Snarl-Smile model artifacts missing. Cannot run analysis.")
    else:
        snarl_smile_detector = SnarlSmileDetector()
        if snarl_smile_detector.model is None or snarl_smile_detector.scaler is None or snarl_smile_detector.feature_names is None:
            logger.error("Detector failed to load artifacts."); snarl_smile_detector = None
        else: logger.info("SnarlSmileDetector initialized successfully for analysis.")
except Exception as e: logger.error(f"Failed to initialize SnarlSmileDetector: {e}", exc_info=True); snarl_smile_detector = None


# --- Helper: process_targets ---
def process_targets(target_series):
    # (No changes needed)
    if target_series is None or target_series.empty: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }; s_filled = target_series.fillna('no')
    s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping); unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected expert labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0); return final_mapped.astype(int).values


# --- Generic Analysis Function ---
def analyze_synkinesis_results(data, synk_type_name, pred_left_col, expert_left_col, pred_right_col, expert_right_col, output_dir):
    # (No changes needed - reports based on input prediction columns)
    logger.info(f"--- Analyzing {synk_type_name} Performance (at Detector Threshold: {snarl_smile_detector.threshold if snarl_smile_detector else 'N/A'}) ---") # Log threshold used
    required_cols = ['Patient ID', pred_left_col, expert_left_col, pred_right_col, expert_right_col]
    missing_cols = [col for col in required_cols if col not in data.columns];
    if missing_cols: logger.error(f"Missing columns: {missing_cols}. Aborting."); return
    left_df = data[['Patient ID', pred_left_col, expert_left_col]].copy(); left_df['Side'] = 'Left'; left_df.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']
    right_df = data[['Patient ID', pred_right_col, expert_right_col]].copy(); right_df['Side'] = 'Right'; right_df.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']
    combined_df = pd.concat([left_df, right_df], ignore_index=True)
    try:
        combined_df[['Prediction', 'Expert']] = combined_df[['Prediction', 'Expert']].astype(int); left_df[['Prediction', 'Expert']] = left_df[['Prediction', 'Expert']].astype(int); right_df[['Prediction', 'Expert']] = right_df[['Prediction', 'Expert']].astype(int)
    except Exception as e: logger.error(f"Type conversion error: {e}."); return
    labels = sorted(combined_df['Expert'].unique()); target_names = [CLASS_NAMES.get(l, f"Class_{l}") for l in labels]
    if len(labels) < 2 : logger.warning(f"Only one class ({labels}) present. Metrics limited.")
    logger.info(f"Analyzing {len(combined_df)} predictions."); logger.info(f"Expert label distribution: {combined_df['Expert'].value_counts().to_dict()}"); logger.info(f"Prediction distribution: {combined_df['Prediction'].value_counts().to_dict()}")
    try:
        logger.info(f"\n--- Combined {synk_type_name} Results ---")
        report_combined = classification_report(combined_df['Expert'], combined_df['Prediction'], labels=labels, target_names=target_names, zero_division=0)
        logger.info("Classification Report:\n" + report_combined); cm_combined = confusion_matrix(combined_df['Expert'], combined_df['Prediction'], labels=labels)
        logger.info("Confusion Matrix:\n" + str(cm_combined)); visualize_confusion_matrix(cm_combined, target_names, f"Combined {synk_type_name}", output_dir)
        logger.info(f"\n--- Left Side {synk_type_name} Results ---")
        report_left = classification_report(left_df['Expert'], left_df['Prediction'], labels=labels, target_names=target_names, zero_division=0)
        logger.info("Classification Report:\n" + report_left); cm_left = confusion_matrix(left_df['Expert'], left_df['Prediction'], labels=labels)
        logger.info("Confusion Matrix:\n" + str(cm_left)); visualize_confusion_matrix(cm_left, target_names, f"Left {synk_type_name}", output_dir)
        logger.info(f"\n--- Right Side {synk_type_name} Results ---")
        report_right = classification_report(right_df['Expert'], right_df['Prediction'], labels=labels, target_names=target_names, zero_division=0)
        logger.info("Classification Report:\n" + report_right); cm_right = confusion_matrix(right_df['Expert'], right_df['Prediction'], labels=labels)
        logger.info("Confusion Matrix:\n" + str(cm_right)); visualize_confusion_matrix(cm_right, target_names, f"Right {synk_type_name}", output_dir)
        # Add thresh to filename for clarity, even if it's 0.5
        thresh_str = str(snarl_smile_detector.threshold).replace('.', '_') if snarl_smile_detector else 'NA'
        perform_error_analysis(combined_df, output_dir, f"{synk_type_name}_errors_thresh_{thresh_str}")
    except Exception as e: logger.error(f"Metrics/vis error: {str(e)}", exc_info=True)


# --- visualize_confusion_matrix ---
def visualize_confusion_matrix(cm, categories, title, output_dir):
    # (No changes needed)
    try: plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar=False); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f"{title} Confusion Matrix"); safe_title = title.replace(' ', '_').replace('-', '_').lower(); save_path = os.path.join(output_dir, f"{safe_title}_confusion_matrix.png"); plt.tight_layout(); plt.savefig(save_path); logger.info(f"CM saved: {save_path}"); plt.close()
    except Exception as e: logger.error(f"Failed to save CM '{title}': {e}"); plt.close()


# --- perform_error_analysis ---
def perform_error_analysis(data, output_dir, filename_base):
    # (No changes needed)
    errors_df = data[data['Prediction'] != data['Expert']].copy(); error_patterns = {}; label_map = {label: CLASS_NAMES.get(label, str(label)) for label in data['Expert'].unique()}
    for _, row in errors_df.iterrows(): expert_label = label_map.get(row['Expert'], str(row['Expert'])); pred_label = label_map.get(row['Prediction'], str(row['Prediction'])); pattern = f"Expert_{expert_label}_Predicted_{pred_label}"; error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
    logger.info(f"\n--- Error Analysis: {filename_base} ---"); total_predictions = len(data); num_errors = len(errors_df); error_rate = (num_errors / total_predictions * 100) if total_predictions > 0 else 0
    logger.info(f"Total Predictions: {total_predictions}, Number of Errors: {num_errors} ({error_rate:.2f}%)")
    if error_patterns: logger.info("Error Patterns (Counts):"); sorted_patterns = sorted(error_patterns.items(), key=lambda item: item[1], reverse=True); [logger.info(f"- {pattern}: {count}") for pattern, count in sorted_patterns]
    else: logger.info("No prediction errors found.")
    error_file_path = os.path.join(output_dir, f"{filename_base}_details.txt")
    try:
        with open(error_file_path, 'w') as f:
            f.write(f"Error Analysis Details: {filename_base}\n=========================================\n"); f.write(f"Total Predictions: {total_predictions}\nNumber of Errors: {num_errors} ({error_rate:.2f}%)\n\nError Patterns Summary:\n")
            if error_patterns: [f.write(f"- {pattern}: {count} ({(count/num_errors*100) if num_errors > 0 else 0:.2f}% of errors)\n") for pattern, count in sorted_patterns]
            else: f.write("- No errors.\n")
            f.write("\nIndividual Error Cases:\n")
            if num_errors > 0: errors_sorted = errors_df.sort_values(by=['Patient ID', 'Side']).reset_index(); [f.write(f"#{i+1}: Pt {row['Patient ID']}, Side: {row['Side']} - Exp: {label_map.get(row['Expert'], '?')}, Pred: {label_map.get(row['Prediction'], '?')}\n") for i, row in errors_sorted.iterrows()]
            else: f.write("- No errors to list.\n")
        logger.info(f"Error details saved: {error_file_path}")
    except Exception as e: logger.error(f"Failed to write error details: {e}", exc_info=True)

# --- Main Analysis Function ---
def analyze_performance(results_file='combined_results.csv', expert_file='FPRS FP Key.csv',
                      output_dir=ANALYSIS_OUTPUT_DIR):
    """ Analyze SNARL-SMILE model performance by re-running detection with configured threshold. """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"--- Starting SNARL-SMILE Performance Analysis ---")
    # Log the threshold from the detector, which reads from config
    current_threshold = snarl_smile_detector.threshold if snarl_smile_detector else DETECTION_THRESHOLD # Safely access threshold
    logger.info(f"Using Detection Threshold: {current_threshold}")
    logger.info(f"Using Results: {results_file}, Expert Key: {expert_file}, Output: {output_dir}")
    if snarl_smile_detector is None: logger.error("Detector unavailable."); return

    try:
        logger.info(f"Loading data...");
        try: results_df = pd.read_csv(results_file, low_memory=False); expert_df = pd.read_csv(expert_file)
        except FileNotFoundError as e: logger.error(f"Load error: {e}."); return
        except Exception as e: logger.error(f"Load error: {e}", exc_info=True); return

        expert_df = expert_df.rename(columns={'Patient': 'Patient ID', 'Snarl Smile Left': 'Expert_Left_Snarl_Smile', 'Snarl Smile Right': 'Expert_Right_Snarl_Smile'})
        if 'Expert_Left_Snarl_Smile' not in expert_df.columns: logger.warning("'Snarl Smile Left' missing.")
        if 'Expert_Right_Snarl_Smile' not in expert_df.columns: logger.warning("'Snarl Smile Right' missing.")
        expert_df['Expert_Left_Snarl_Smile'] = process_targets(expert_df.get('Expert_Left_Snarl_Smile', pd.Series(dtype=str)))
        expert_df['Expert_Right_Snarl_Smile'] = process_targets(expert_df.get('Expert_Right_Snarl_Smile', pd.Series(dtype=str)))
        logger.info("Expert labels standardized.")

        results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
        expert_cols_to_merge = ['Patient ID', 'Expert_Left_Snarl_Smile', 'Expert_Right_Snarl_Smile']
        expert_cols_present = [col for col in expert_cols_to_merge if col in expert_df.columns]
        if len(expert_cols_present) < len(expert_cols_to_merge): logger.warning(f"Missing expert cols for merge: {set(expert_cols_to_merge) - set(expert_cols_present)}")
        try: merged_df = pd.merge(results_df, expert_df[expert_cols_present], on='Patient ID', how='inner', validate="many_to_one")
        except KeyError as ke: logger.error(f"Merge failed (KeyError): {ke}."); return
        except Exception as merge_e: logger.error(f"Merge failed: {merge_e}", exc_info=True); return
        logger.info(f"Merged data: {len(merged_df)} rows.");
        if merged_df.empty: logger.error("Merged DataFrame empty."); return

        logger.info("Re-running Snarl-Smile detection on merged data...")
        predictions_left, predictions_right = [], []; confidences_left, confidences_right = [], []
        # No need to store probabilities separately if only running standard analysis
        num_rows = len(merged_df); log_interval = max(1, num_rows // 10)
        for index, row in merged_df.iterrows():
            if (index + 1) % log_interval == 0: logger.info(f"Processing row {index + 1} / {num_rows}...")
            pid = row.get('Patient ID', 'Unknown'); row_dict = row.to_dict()
            try: det_l, conf_l = snarl_smile_detector.detect_snarl_smile_synkinesis(row_dict, 'Left'); predictions_left.append(1 if det_l else 0); confidences_left.append(conf_l)
            except Exception as e_l: logger.error(f"Error L Pt {pid} (idx {index}): {e_l}", exc_info=False); predictions_left.append(-1); confidences_left.append(np.nan)
            try: det_r, conf_r = snarl_smile_detector.detect_snarl_smile_synkinesis(row_dict, 'Right'); predictions_right.append(1 if det_r else 0); confidences_right.append(conf_r)
            except Exception as e_r: logger.error(f"Error R Pt {pid} (idx {index}): {e_r}", exc_info=False); predictions_right.append(-1); confidences_right.append(np.nan)

        merged_df['ML_Pred_Left_Snarl_Smile'] = predictions_left; merged_df['ML_Conf_Left_Snarl_Smile'] = confidences_left
        merged_df['ML_Pred_Right_Snarl_Smile'] = predictions_right; merged_df['ML_Conf_Right_Snarl_Smile'] = confidences_right
        analysis_df = merged_df[(merged_df['ML_Pred_Left_Snarl_Smile'] != -1) & (merged_df['ML_Pred_Right_Snarl_Smile'] != -1)].copy()
        num_failed = len(merged_df) - len(analysis_df)
        if num_failed > 0: logger.warning(f"Excluded {num_failed} rows due to errors.")
        if analysis_df.empty: logger.error("No valid predictions. Cannot analyze."); return
        logger.info(f"{len(analysis_df) * 2} valid predictions for analysis.")

        # Analyze Results using the detector's configured threshold
        analyze_synkinesis_results(
            data=analysis_df, synk_type_name='Snarl-Smile',
            pred_left_col='ML_Pred_Left_Snarl_Smile', expert_left_col='Expert_Left_Snarl_Smile',
            pred_right_col='ML_Pred_Right_Snarl_Smile', expert_right_col='Expert_Right_Snarl_Smile',
            output_dir=output_dir
        )
        logger.info("--- SNARL-SMILE Performance Analysis Complete ---")
    except Exception as e: logger.error(f"Unexpected error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    logger.info("Running Snarl-Smile performance analysis script directly.")
    # Assumes model artifacts from V7 training run exist.
    analyze_performance()
    logger.info("Snarl-Smile analysis script finished.")