# mentalis_performance_analysis.py
# Analyzes performance of the Mentalis Synkinesis detector.

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

try:
    from mentalis_detector import MentalisDetector
    from mentalis_config import LOG_DIR, CLASS_NAMES, MODEL_FILENAMES
except ImportError:
    logging.error("Could not import MentalisDetector or config.")
    raise SystemExit("Missing imports for performance analysis.")

# Configure logging
ANALYSIS_OUTPUT_DIR = 'analysis_results'
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(ANALYSIS_OUTPUT_DIR, 'mentalis_analysis.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()], force=True)
logger = logging.getLogger(__name__)

# Initialize the Detector
mentalis_detector = None
try:
    if not all(os.path.exists(MODEL_FILENAMES.get(k)) for k in ['model', 'scaler', 'feature_list']): logger.error(f"Mentalis model artifacts missing. Cannot run analysis.")
    else:
        mentalis_detector = MentalisDetector()
        if mentalis_detector.model is None: raise RuntimeError("Detector failed to load artifacts.")
        logger.info("MentalisDetector initialized for analysis.")
except Exception as e: logger.error(f"Failed to initialize MentalisDetector: {e}", exc_info=True)

# --- Helper: process_targets (Copied) ---
def process_targets(target_series):
    # (Identical logic as in features file)
    if target_series is None or target_series.empty: return np.array([], dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    numeric_yes_values = { val: 1 for val in target_series.unique() if isinstance(val, (int, float)) and val > 0 }
    mapping.update(numeric_yes_values); numeric_no_values = { val: 0 for val in target_series.unique() if isinstance(val, (int, float)) and val == 0 }
    mapping.update(numeric_no_values); s_filled = target_series.fillna('no'); s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = target_series.map(mapping); mapped_str = s_clean.map(mapping); mapped = mapped.fillna(mapped_str)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty: logger.warning(f"Unexpected expert labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0); return final_mapped.astype(int).values

# --- Generic Analysis Function ---
def analyze_synkinesis_results(data, synk_type_name, pred_left_col, expert_left_col, pred_right_col, expert_right_col, output_dir):
    # (Identical logic as other analysis files)
    logger.info(f"--- Analyzing {synk_type_name} Performance ---"); required = ['Patient ID', pred_left_col, expert_left_col, pred_right_col, expert_right_col]
    if not all(col in data.columns for col in required): logger.error(f"Missing cols: {[c for c in required if c not in data.columns]}. Abort."); return
    left = data[['Patient ID', pred_left_col, expert_left_col]].copy(); left['Side'] = 'Left'; right = data[['Patient ID', pred_right_col, expert_right_col]].copy(); right['Side'] = 'Right'
    left.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']; right.columns = ['Patient ID', 'Prediction', 'Expert', 'Side']
    try: left[['Prediction', 'Expert']] = left[['Prediction', 'Expert']].astype(int); right[['Prediction', 'Expert']] = right[['Prediction', 'Expert']].astype(int)
    except ValueError as ve: logger.error(f"Error converting pred/expert to int: {ve}. Check data."); return
    combined = pd.concat([left, right], ignore_index=True); labels = sorted(combined['Expert'].unique()); target_names = [CLASS_NAMES.get(l, str(l)) for l in labels]
    if len(labels) < 2 : logger.warning(f"Only one class ({labels}) present. Metrics limited.")
    logger.info(f"Analyzing {len(combined)} preds. Expert: {combined['Expert'].value_counts().to_dict()}. Pred: {combined['Prediction'].value_counts().to_dict()}.")
    try:
        logger.info(f"\n--- Combined {synk_type_name} ---"); report_c = classification_report(combined['Expert'], combined['Prediction'], labels=labels, target_names=target_names, zero_division=0); logger.info("Report:\n" + report_c); cm_c = confusion_matrix(combined['Expert'], combined['Prediction'], labels=labels); logger.info("CM:\n" + str(cm_c)); visualize_confusion_matrix(cm_c, target_names, f"Combined {synk_type_name}", output_dir)
        logger.info(f"\n--- Left {synk_type_name} ---"); report_l = classification_report(left['Expert'], left['Prediction'], labels=labels, target_names=target_names, zero_division=0); logger.info("Report:\n" + report_l); cm_l = confusion_matrix(left['Expert'], left['Prediction'], labels=labels); logger.info("CM:\n" + str(cm_l)); visualize_confusion_matrix(cm_l, target_names, f"Left {synk_type_name}", output_dir)
        logger.info(f"\n--- Right {synk_type_name} ---"); report_r = classification_report(right['Expert'], right['Prediction'], labels=labels, target_names=target_names, zero_division=0); logger.info("Report:\n" + report_r); cm_r = confusion_matrix(right['Expert'], right['Prediction'], labels=labels); logger.info("CM:\n" + str(cm_r)); visualize_confusion_matrix(cm_r, target_names, f"Right {synk_type_name}", output_dir)
        perform_error_analysis(combined, output_dir, f"{synk_type_name}_errors")
    except Exception as e: logger.error(f"Metrics/vis error: {str(e)}", exc_info=True)

# --- visualize_confusion_matrix ---
def visualize_confusion_matrix(cm, categories, title, output_dir):
    # (Identical logic)
    try: plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar=False); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f"{title} CM"); safe_title = title.replace(' ', '_').replace('-', '_').lower(); save_path = os.path.join(output_dir, f"{safe_title}_confusion_matrix.png"); plt.tight_layout(); plt.savefig(save_path); logger.info(f"CM saved: {save_path}"); plt.close()
    except Exception as e: logger.error(f"Save CM {title} failed: {e}"); plt.close()

# --- perform_error_analysis ---
def perform_error_analysis(data, output_dir, filename_base):
    # (Identical logic)
    errors = data[data['Prediction'] != data['Expert']].copy(); patterns = {}; label_map = {label: CLASS_NAMES.get(label, str(label)) for label in data['Expert'].unique()}
    for _, row in errors.iterrows(): pattern = f"Expert_{label_map.get(row['Expert'], '?')}_Predicted_{label_map.get(row['Prediction'], '?')}"; patterns[pattern] = patterns.get(pattern, 0) + 1
    logger.info(f"\n--- Errors: {filename_base} ---"); total = len(data); n_err = len(errors); err_rate = (n_err / total * 100) if total > 0 else 0
    logger.info(f"Total: {total}, Errors: {n_err} ({err_rate:.2f}%)")
    if patterns: logger.info("Patterns:"); [logger.info(f"- {p}: {c}") for p,c in sorted(patterns.items())]
    else: logger.info("No errors.")
    path = os.path.join(output_dir, f"{filename_base}_details.txt")
    try:
        with open(path, 'w') as f:
            f.write(f"Errors: {filename_base}\n==\nTotal: {total}, Errors: {n_err} ({err_rate:.2f}%)\n\nPatterns:\n")
            if n_err > 0: [f.write(f"- {p}: {c} ({c/n_err*100:.2f}% of errors)\n") for p,c in sorted(patterns.items())]
            else: f.write("- None\n")
            f.write("\nCases:\n"); errors_sorted = errors.sort_values(by=['Patient ID', 'Side']).reset_index()
            for i, row in errors_sorted.iterrows(): f.write(f"#{i+1}: Pt {row['Patient ID']}, {row['Side']} - Exp: {label_map.get(row['Expert'], '?')}, Pred: {label_map.get(row['Prediction'], '?')}\n")
        logger.info(f"Errors saved: {path}")
    except Exception as e: logger.error(f"Save errors {path} failed: {e}", exc_info=True)

# --- Main Analysis Function (Specific to Mentalis) ---
def analyze_performance(results_file='combined_results.csv', expert_file='FPRS FP Key.csv',
                      output_dir=ANALYSIS_OUTPUT_DIR):
    """ Analyze MENTALIS SYNKINESIS model performance by re-running detection. """
    os.makedirs(output_dir, exist_ok=True); logger.info(f"--- Starting MENTALIS SYNKINESIS Performance Analysis ---")
    logger.info(f"Files: {results_file}, {expert_file}. Output: {output_dir}")
    if mentalis_detector is None: logger.error("Detector unavailable."); return

    try:
        logger.info(f"Loading data..."); results_df = pd.read_csv(results_file, low_memory=False); expert_df = pd.read_csv(expert_file)
        # --- Rename based on expected columns ---
        expert_rename_map = { 'Patient': 'Patient ID', 'Mentalis Synkinesis Left': 'Expert_Left_Mentalis', 'Mentalis Synkinesis Right': 'Expert_Right_Mentalis'}
        cols_to_rename = {k: v for k, v in expert_rename_map.items() if k in expert_df.columns}
        if 'Patient' in expert_df.columns and 'Patient ID' not in cols_to_rename: cols_to_rename['Patient'] = 'Patient ID'
        expert_df = expert_df.rename(columns=cols_to_rename)
        # --- End Rename ---
        target_left_col = 'Expert_Left_Mentalis'; target_right_col = 'Expert_Right_Mentalis'
        if target_left_col not in expert_df.columns: logger.error(f"Expert column '{target_left_col}' not found.")
        if target_right_col not in expert_df.columns: logger.error(f"Expert column '{target_right_col}' not found.")
        expert_df['Expert_Left_Mentalis'] = process_targets(expert_df.get(target_left_col, pd.Series(dtype=str)))
        expert_df['Expert_Right_Mentalis'] = process_targets(expert_df.get(target_right_col, pd.Series(dtype=str)))
        logger.info("Expert labels standardized.")

        results_df['Patient ID'] = results_df['Patient ID'].astype(str).str.strip(); expert_df['Patient ID'] = expert_df['Patient ID'].astype(str).str.strip()
        expert_cols = ['Patient ID', 'Expert_Left_Mentalis', 'Expert_Right_Mentalis']
        expert_cols_present = [c for c in expert_cols if c in expert_df.columns]

        merged_df = pd.merge(results_df, expert_df[expert_cols_present], on='Patient ID', how='inner', validate="many_to_one")
        logger.info(f"Merged data: {len(merged_df)} patients with Mentalis labels.")
        if merged_df.empty: logger.error("Merge failed."); return
        if target_left_col not in merged_df.columns or target_right_col not in merged_df.columns: logger.error(f"Target columns missing after merge."); return

        logger.info("Re-running Mentalis Synkinesis detection..."); predictions_left, predictions_right = [], []; confidences_left, confidences_right = [], []
        num_rows = len(merged_df); log_interval = max(1, num_rows // 10)
        for index, row in merged_df.iterrows():
            if (index + 1) % log_interval == 0: logger.info(f"Processing row {index + 1} / {num_rows}...")
            pid = row.get('Patient ID', 'Unknown'); row_dict = row.to_dict()
            try: det_l, conf_l = mentalis_detector.detect_mentalis_synkinesis(row_dict, 'Left'); predictions_left.append(1 if det_l else 0); confidences_left.append(conf_l)
            except Exception as e_l: logger.error(f"Error L {pid} (idx {index}): {e_l}", exc_info=False); predictions_left.append(-1); confidences_left.append(np.nan)
            try: det_r, conf_r = mentalis_detector.detect_mentalis_synkinesis(row_dict, 'Right'); predictions_right.append(1 if det_r else 0); confidences_right.append(conf_r)
            except Exception as e_r: logger.error(f"Error R {pid} (idx {index}): {e_r}", exc_info=False); predictions_right.append(-1); confidences_right.append(np.nan)

        merged_df['ML_Pred_Left_Mentalis'] = predictions_left; merged_df['ML_Conf_Left_Mentalis'] = confidences_left
        merged_df['ML_Pred_Right_Mentalis'] = predictions_right; merged_df['ML_Conf_Right_Mentalis'] = confidences_right
        analysis_df = merged_df[(merged_df['ML_Pred_Left_Mentalis'] != -1) & (merged_df['ML_Pred_Right_Mentalis'] != -1)].copy()
        num_failed = len(merged_df) - len(analysis_df)
        if num_failed > 0: logger.warning(f"Excluded {num_failed} rows due to detection errors.")
        if analysis_df.empty: logger.error("No valid predictions."); return
        logger.info(f"{len(analysis_df) * 2} valid predictions generated.")

        analyze_synkinesis_results(analysis_df, 'Mentalis Synkinesis', 'ML_Pred_Left_Mentalis', 'Expert_Left_Mentalis', 'ML_Pred_Right_Mentalis', 'Expert_Right_Mentalis', output_dir)
        logger.info("--- MENTALIS SYNKINESIS Performance Analysis Complete ---")
    except Exception as e: logger.error(f"Unexpected error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    logger.info("Running Mentalis Synkinesis performance analysis script directly.")
    analyze_performance()
    logger.info("Script done.")