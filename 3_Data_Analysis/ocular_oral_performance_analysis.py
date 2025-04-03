# ocular_oral_performance_analysis.py (Updated Import/Instantiation & Capitalization Fix)

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
import seaborn as sns

# --- Import necessary components ---
try:
    # --- UPDATED Import ---
    from ocular_oral_detector import OcularOralDetector
    from ocular_oral_config import LOG_DIR, CLASS_NAMES
except ImportError:
    logging.error("Could not import OcularOralDetector or config. Analysis cannot run.")
    raise SystemExit("Missing necessary imports for performance analysis.")

# Configure logging
ANALYSIS_OUTPUT_DIR = 'analysis_results'
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(ANALYSIS_OUTPUT_DIR, 'ocular_oral_analysis.log')
# Ensure logger is configured
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
                    force=True) # Force reconfigures if already set
logger = logging.getLogger(__name__)

# --- Initialize the Detector ---
try:
    # --- UPDATED Instantiation ---
    ocular_oral_detector = OcularOralDetector()
    if ocular_oral_detector.model is None:
        raise RuntimeError("OcularOralDetector initialized but failed to load model/scaler.")
    logger.info("OcularOralDetector initialized successfully for analysis.")
except Exception as e:
    logger.error(f"Failed to initialize OcularOralDetector: {e}", exc_info=True)
    ocular_oral_detector = None

# --- Main Analysis Function ---
def analyze_performance(results_file='combined_results.csv', expert_file='FPRS FP Key.csv',
                      output_dir=ANALYSIS_OUTPUT_DIR):
    """ Analyze OCULAR-ORAL model performance by re-running detection. """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"--- Starting OCULAR-ORAL Performance Analysis ---")
    logger.info(f"Using results: {results_file}, expert key: {expert_file}, output: {output_dir}")
    if ocular_oral_detector is None: logger.error("Detector not available. Aborting."); return
    try:
        logger.info(f"Loading data..."); results_df = pd.read_csv(results_file, low_memory=False); expert_df = pd.read_csv(expert_file)
        expert_df = expert_df.rename(columns={'Patient': 'Patient ID', 'Ocular-Oral Synkinesis Left': 'Expert_Left_Ocular_Oral', 'Ocular-Oral Synkinesis Right': 'Expert_Right_Ocular_Oral'})
        expert_df['Expert_Left_Ocular_Oral'] = process_targets(expert_df.get('Expert_Left_Ocular_Oral', pd.Series(dtype=str)))
        expert_df['Expert_Right_Ocular_Oral'] = process_targets(expert_df.get('Expert_Right_Ocular_Oral', pd.Series(dtype=str)))
        logger.info("Expert labels standardized.")
        results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
        expert_cols = ['Patient ID', 'Expert_Left_Ocular_Oral', 'Expert_Right_Ocular_Oral']
        merged_df = pd.merge(results_df, expert_df[expert_cols], on='Patient ID', how='inner', validate="many_to_one")
        logger.info(f"Merged data: {len(merged_df)} patients with Ocular-Oral labels.")
        if merged_df.empty: logger.error("Merge failed."); return
        logger.info("Re-running Ocular-Oral detection..."); predictions_left, predictions_right = [], []; confidences_left, confidences_right = [], []

        # --- Re-run detection ---
        for index, row in merged_df.iterrows():
            pid = row['Patient ID']; row_dict = row.to_dict()
            try:
                # --- Pass Capitalized Side ---
                det_l, conf_l = ocular_oral_detector.detect_ocular_oral_synkinesis(row_dict, 'Left')
                predictions_left.append(1 if det_l else 0); confidences_left.append(conf_l)
            except Exception as e_l: logger.error(f"Error L {pid}: {e_l}"); predictions_left.append(-1); confidences_left.append(np.nan)
            try:
                 # --- Pass Capitalized Side ---
                det_r, conf_r = ocular_oral_detector.detect_ocular_oral_synkinesis(row_dict, 'Right')
                predictions_right.append(1 if det_r else 0); confidences_right.append(conf_r)
            except Exception as e_r: logger.error(f"Error R {pid}: {e_r}"); predictions_right.append(-1); confidences_right.append(np.nan)
        # --- End Re-run ---

        merged_df['ML_Pred_Left_Ocular_Oral'] = predictions_left; merged_df['ML_Conf_Left_Ocular_Oral'] = confidences_left
        merged_df['ML_Pred_Right_Ocular_Oral'] = predictions_right; merged_df['ML_Conf_Right_Ocular_Oral'] = confidences_right
        analysis_df = merged_df[(merged_df['ML_Pred_Left_Ocular_Oral'] != -1) & (merged_df['ML_Pred_Right_Ocular_Oral'] != -1)].copy()
        if analysis_df.empty: logger.error("No valid predictions."); return
        logger.info(f"{len(analysis_df) * 2} valid predictions generated.")
        analyze_synkinesis_results(analysis_df, 'Ocular-Oral', 'ML_Pred_Left_Ocular_Oral', 'Expert_Left_Ocular_Oral', 'ML_Pred_Right_Ocular_Oral', 'Expert_Right_Ocular_Oral', output_dir)
        logger.info("--- OCULAR-ORAL Performance Analysis Complete ---")
    except Exception as e: logger.error(f"Unexpected error: {str(e)}", exc_info=True)

# --- Helper: process_targets (Identical binary mapping) ---
def process_targets(target_series):
    if target_series is None or target_series.empty: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    s_filled = target_series.fillna('no') # Default missing expert labels to 'no'
    s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')] # Identify values not in mapping and not already 'no'
    if not unexpected.empty: logger.warning(f"Unexpected Ocular-Oral expert labels found (treated as 'No'): {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0) # Treat unmapped values (including the original NaNs now filled with 'no') as 0
    return final_mapped.astype(int).values

# --- Generic Analysis Function (Identical structure) ---
def analyze_synkinesis_results(data, synk_type_name, pred_left_col, expert_left_col, pred_right_col, expert_right_col, output_dir):
    # ... (Identical logic) ...
    logger.info(f"--- Analyzing {synk_type_name} Performance ---")
    required = ['Patient ID', pred_left_col, expert_left_col, pred_right_col, expert_right_col]
    if not all(col in data.columns for col in required): logger.error(f"Missing cols: {[c for c in required if c not in data.columns]}. Abort."); return
    left = data[['Patient ID', pred_left_col, expert_left_col]].copy(); left['Side'] = 'Left'
    right = data[['Patient ID', pred_right_col, expert_right_col]].copy(); right['Side'] = 'Right'
    left.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']; right.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']
    left[['Prediction', 'Expert']] = left[['Prediction', 'Expert']].astype(int); right[['Prediction', 'Expert']] = right[['Prediction', 'Expert']].astype(int)
    combined = pd.concat([left, right], ignore_index=True)
    labels = sorted(combined['Expert'].unique()) # Use actual labels present
    if not labels: logger.error(f"No expert labels found for {synk_type_name}. Aborting analysis."); return
    target_names = [CLASS_NAMES.get(l, str(l)) for l in labels]
    logger.info(f"Analyzing {len(combined)} predictions. Expert dist: {combined['Expert'].value_counts().to_dict()}. Pred dist: {combined['Prediction'].value_counts().to_dict()}.")
    try:
        logger.info(f"\n--- Combined {synk_type_name} ---"); report_c = classification_report(combined['Expert'], combined['Prediction'], labels=labels, target_names=target_names, zero_division=0)
        logger.info("Report:\n" + report_c); cm_c = confusion_matrix(combined['Expert'], combined['Prediction'], labels=labels); logger.info("CM:\n" + str(cm_c)); visualize_confusion_matrix(cm_c, target_names, f"Combined {synk_type_name}", output_dir)
        logger.info(f"\n--- Left {synk_type_name} ---"); report_l = classification_report(left['Expert'], left['Prediction'], labels=labels, target_names=target_names, zero_division=0)
        logger.info("Report:\n" + report_l); cm_l = confusion_matrix(left['Expert'], left['Prediction'], labels=labels); logger.info("CM:\n" + str(cm_l)); visualize_confusion_matrix(cm_l, target_names, f"Left {synk_type_name}", output_dir)
        logger.info(f"\n--- Right {synk_type_name} ---"); report_r = classification_report(right['Expert'], right['Prediction'], labels=labels, target_names=target_names, zero_division=0)
        logger.info("Report:\n" + report_r); cm_r = confusion_matrix(right['Expert'], right['Prediction'], labels=labels); logger.info("CM:\n" + str(cm_r)); visualize_confusion_matrix(cm_r, target_names, f"Right {synk_type_name}", output_dir)
        perform_error_analysis(combined, output_dir, f"{synk_type_name}_errors")
    except Exception as e: logger.error(f"Metrics/vis error: {str(e)}", exc_info=True)


# --- visualize_confusion_matrix (Identical) ---
def visualize_confusion_matrix(cm, categories, title, output_dir):
    # ... (Identical logic) ...
    try: plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar=False); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f"{title} CM"); safe_title = title.replace(' ', '_').replace('-', '_'); save_path = os.path.join(output_dir, f"{safe_title}_confusion_matrix.png"); plt.tight_layout(); plt.savefig(save_path); logger.info(f"CM saved: {save_path}"); plt.close()
    except Exception as e: logger.error(f"Save CM {title} failed: {e}"); plt.close()

# --- perform_error_analysis (Identical) ---
def perform_error_analysis(data, output_dir, filename_base):
    # ... (Identical logic) ...
    errors = data[data['Prediction'] != data['Expert']].copy(); patterns = {}; label_map = {0: CLASS_NAMES.get(0,'0'), 1: CLASS_NAMES.get(1,'1')}
    for _, row in errors.iterrows(): pattern = f"Exp_{label_map.get(row['Expert'])}_Pred_{label_map.get(row['Prediction'])}"; patterns[pattern] = patterns.get(pattern, 0) + 1
    logger.info(f"\n--- Errors: {filename_base} ---");
    if patterns: logger.info("Patterns:"); [logger.info(f"- {p}: {c}") for p,c in sorted(patterns.items())]
    else: logger.info("No errors.")
    path = os.path.join(output_dir, f"{filename_base}.txt"); total = len(data); n_err = len(errors)
    try:
        with open(path, 'w') as f:
            f.write(f"Errors: {filename_base}\n==\nTotal: {total}, Errors: {n_err} ({n_err/total*100:.2f}%)\n\nPatterns:\n" if total > 0 else f"Errors: {n_err}\n\nPatterns:\n")
            if n_err > 0: [f.write(f"- {p}: {c} ({c/n_err*100:.2f}%)\n") for p,c in sorted(patterns.items())]
            else: f.write("- None\n")
            f.write("\nCases:\n"); errors_sorted = errors.sort_values(by=['Patient ID', 'Side']).reset_index()
            for i, row in errors_sorted.iterrows(): f.write(f"#{i+1}: Pt {row['Patient ID']}, {row['Side']} - Exp: {label_map.get(row['Expert'])}, Pred: {label_map.get(row['Prediction'])}\n")
        logger.info(f"Errors saved: {path}")
    except Exception as e: logger.error(f"Save errors {path} failed: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Running Ocular-Oral performance analysis script directly.")
    analyze_performance()
    logger.info("Script done.")