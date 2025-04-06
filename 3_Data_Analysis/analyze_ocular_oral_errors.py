import pandas as pd
import numpy as np
import joblib
import logging
import os
import argparse
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific scikit-learn warnings about feature names during transform
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# --- Import necessary components for Ocular-Oral ---
try:
    from ocular_oral_features import extract_features_for_detection
    OCULAR_ORAL_FEATURE_EXTRACTION_AVAILABLE = True
except ImportError:
    logging.error("CRITICAL: Could not import extract_features_for_detection from ocular_oral_features.py.")
    OCULAR_ORAL_FEATURE_EXTRACTION_AVAILABLE = False

try:
    from ocular_oral_config import CLASS_NAMES, MODEL_FILENAMES
except ImportError:
    logging.warning("Could not import CLASS_NAMES/MODEL_FILENAMES from ocular_oral_config. Using default.")
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}
    MODEL_FILENAMES = { # Add minimal fallback for load_artifacts
        'model': 'models/synkinesis/ocular_oral/model.pkl',
        'scaler': 'models/synkinesis/ocular_oral/scaler.pkl',
        'feature_list': 'models/synkinesis/ocular_oral/features.list'
    }
# --- End Imports ---

# Configure logging
ANALYSIS_OUTPUT_DIR = 'analysis_results'
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(ANALYSIS_OUTPUT_DIR, 'ocular_oral_error_analysis.log') # Specific log file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
                    force=True)
logger = logging.getLogger(__name__)


def load_artifacts(model_dir):
    """Loads model, scaler, and feature list."""
    logger.info(f"Loading Ocular-Oral artifacts from: {model_dir}")
    # Use paths directly from MODEL_FILENAMES potentially loaded from config
    model_path = MODEL_FILENAMES.get('model')
    scaler_path = MODEL_FILENAMES.get('scaler')
    features_path = MODEL_FILENAMES.get('feature_list')

    # Construct full paths if model_dir is provided and paths are relative
    if model_dir:
        model_path = os.path.join(model_dir, os.path.basename(model_path)) if model_path else None
        scaler_path = os.path.join(model_dir, os.path.basename(scaler_path)) if scaler_path else None
        features_path = os.path.join(model_dir, os.path.basename(features_path)) if features_path else None

    artifacts = {'model': None, 'scaler': None, 'feature_names': None}
    try:
        if not model_path or not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path or not os.path.exists(scaler_path): raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not features_path or not os.path.exists(features_path): raise FileNotFoundError(f"Feature list file not found: {features_path}")

        artifacts['model'] = joblib.load(model_path)
        artifacts['scaler'] = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        if not isinstance(feature_names, list):
            raise TypeError("Loaded feature names is not a list.")
        artifacts['feature_names'] = feature_names

        logger.info(f"Successfully loaded artifacts. Model expects {len(artifacts['feature_names'])} features.")
        return artifacts

    except Exception as e:
        logger.error(f"Error loading artifacts: {e}", exc_info=True)
        return None

def process_targets(target_series):
    """Converts expert labels (Yes/No etc.) to binary 0/1."""
    if target_series is None: return pd.Series(dtype=int)
    mapping = { 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 1:1, 0:0 }
    s_filled = target_series.fillna('no')
    s_clean = s_filled.astype(str).str.lower().str.strip().replace({'none': 'no', 'n/a': 'no', '': 'no', 'nan': 'no'})
    mapped = s_clean.map(mapping)
    unexpected = s_clean[mapped.isna() & (s_clean != 'no')]
    if not unexpected.empty:
         logger.warning(f"Unexpected Ocular-Oral expert labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int)

def analyze_errors(results_file, expert_file, model_dir, output_dir):
    """Main function to perform error analysis for Ocular-Oral."""
    logger.info("--- Starting Ocular-Oral Error Analysis ---")

    if not OCULAR_ORAL_FEATURE_EXTRACTION_AVAILABLE:
        logger.error("Cannot proceed without Ocular-Oral feature extraction function.")
        return

    # Load Model Artifacts
    artifacts = load_artifacts(model_dir)
    if artifacts is None or artifacts.get('model') is None:
        logger.error("Failed to load necessary model artifacts. Aborting.")
        return
    model = artifacts['model']
    scaler = artifacts['scaler']
    model_features = artifacts['feature_names'] # List of features model expects

    # Load and Prepare Data
    try:
        logger.info(f"Loading data from {results_file} and {expert_file}")
        df_results = pd.read_csv(results_file, low_memory=False)
        df_expert = pd.read_csv(expert_file)

        # Prepare expert labels
        df_expert = df_expert.rename(columns={
            'Patient': 'Patient ID',
            'Ocular-Oral Synkinesis Left': 'Expert_Left_Ocular_Oral',
            'Ocular-Oral Synkinesis Right': 'Expert_Right_Ocular_Oral'
        })
        for col in ['Expert_Left_Ocular_Oral', 'Expert_Right_Ocular_Oral']:
            target_col = col.replace('Expert', 'Target')
            if col in df_expert.columns:
                df_expert[target_col] = process_targets(df_expert[col])
            else:
                logger.warning(f"Missing expert column: {col}. Creating default target column.")
                df_expert[target_col] = 0

        # Merge
        df_results['Patient ID'] = df_results['Patient ID'].astype(str).str.strip()
        df_expert['Patient ID'] = df_expert['Patient ID'].astype(str).str.strip()
        expert_cols_to_merge = ['Patient ID', 'Target_Left_Ocular_Oral', 'Target_Right_Ocular_Oral']
        df = pd.merge(df_results, df_expert[expert_cols_to_merge], on='Patient ID', how='inner')
        logger.info(f"Loaded and merged data for {len(df)} patients.")
        if df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    except Exception as e:
        logger.error(f"Failed to load or prepare data: {e}", exc_info=True)
        return

    # Re-run Predictions
    predictions = {}
    logger.info("Re-running Ocular-Oral predictions using loaded model...")
    error_count = 0
    for index, row in df.iterrows():
        patient_id = row['Patient ID']
        row_data = row.to_dict()
        for side in ['Left', 'Right']:
            prediction_result = -1 # Default to error
            try:
                features_list = extract_features_for_detection(row_data, side) # Expects 'Left' or 'Right'
                if features_list is None:
                    logger.warning(f"Feature extraction failed for {patient_id} - {side}. Skipping prediction.")
                    error_count +=1
                elif len(features_list) != len(model_features):
                    logger.error(f"Feature mismatch for {patient_id} - {side}. Expected {len(model_features)}, got {len(features_list)}. Skipping prediction.")
                    error_count +=1
                else:
                    features_df = pd.DataFrame([features_list], columns=model_features)
                    if features_df.isnull().values.any():
                        nan_cols = features_df.columns[features_df.isna().any()].tolist()
                        logger.warning(f"NaNs found in OcOr features before scaling for {patient_id} - {side}: {nan_cols}. Filling with 0.")
                        features_df = features_df.fillna(0)

                    scaled_features = scaler.transform(features_df)
                    prediction = model.predict(scaled_features)[0]
                    prediction_result = int(prediction)

            except Exception as pred_e:
                logger.error(f"Prediction failed for {patient_id} - {side}: {pred_e}", exc_info=True)
                error_count+=1
            finally:
                predictions[(patient_id, side)] = prediction_result

    logger.info(f"Predictions complete. Encountered {error_count} errors during prediction/feature extraction.")

    # Add predictions to DataFrame
    df['Prediction_Left'] = df['Patient ID'].apply(lambda pid: predictions.get((pid, 'Left'), -1))
    df['Prediction_Right'] = df['Patient ID'].apply(lambda pid: predictions.get((pid, 'Right'), -1))

    # --- Identify ALL Error Groups (Left and Right) ---
    logger.info("Identifying analysis groups (FN, FP, TP, TN)...")
    df_analysis_list = []

    for side in ['Left', 'Right']:
        target_col = f'Target_{side}_Ocular_Oral'
        pred_col = f'Prediction_{side}'

        # False Negatives
        fn_df = df[(df[target_col] == 1) & (df[pred_col] == 0)].copy()
        fn_df['Group'] = f'FN_{side}'
        if not fn_df.empty: df_analysis_list.append(fn_df)
        logger.info(f"Found {len(fn_df)} FN_{side} cases.")

        # False Positives
        fp_df = df[(df[target_col] == 0) & (df[pred_col] == 1)].copy()
        fp_df['Group'] = f'FP_{side}'
        if not fp_df.empty: df_analysis_list.append(fp_df)
        logger.info(f"Found {len(fp_df)} FP_{side} cases.")

        # True Positives
        tp_df = df[(df[target_col] == 1) & (df[pred_col] == 1)].copy()
        tp_df['Group'] = f'TP_{side}'
        if not tp_df.empty: df_analysis_list.append(tp_df)
        logger.info(f"Found {len(tp_df)} TP_{side} cases.")

        # True Negatives (Sample)
        tn_candidates = df[(df[target_col] == 0) & (df[pred_col] == 0)]
        sample_size_tn = min(5, len(tn_candidates)) # Sample up to 5 TNs per side
        tn_df_sample = tn_candidates.sample(n=sample_size_tn, random_state=42).copy()
        tn_df_sample['Group'] = f'TN_{side}_Sample'
        if not tn_df_sample.empty: df_analysis_list.append(tn_df_sample)
        logger.info(f"Sampled {len(tn_df_sample)} TN_{side} cases.")

    if not df_analysis_list:
        logger.warning("No cases found for analysis groups (FN, FP, TP, TN). Cannot generate comparison.")
        return

    df_analysis = pd.concat(df_analysis_list, ignore_index=True)

    # --- Extract RAW/NORMALIZED AUs for Comparison ---
    # Focus on Trigger and Coupled AUs during the main trigger action ET
    # Define AUs based on config if possible, else use defaults
    try:
        from ocular_oral_config import TRIGGER_AUS, COUPLED_AUS
    except ImportError:
        TRIGGER_AUS = ['AU01_r', 'AU02_r', 'AU45_r']
        COUPLED_AUS = ['AU12_r', 'AU25_r', 'AU14_r']

    core_aus_to_compare = []
    # Add ET Trigger AUs (Left and Right, Raw and Norm)
    for side in ['Left', 'Right']:
        for au in TRIGGER_AUS:
            core_aus_to_compare.append(f"ET_{side} {au}")
            core_aus_to_compare.append(f"ET_{side} {au} (Normalized)")
    # Add ET Coupled AUs (Left and Right, Raw and Norm)
    for side in ['Left', 'Right']:
        for au in COUPLED_AUS:
            core_aus_to_compare.append(f"ET_{side} {au}")
            core_aus_to_compare.append(f"ET_{side} {au} (Normalized)")
    # Add baseline coupled AUs for comparison (if available)
    for side in ['Left', 'Right']:
         for au in COUPLED_AUS:
             core_aus_to_compare.append(f"BL_{side} {au} (Normalized)") # Usually compare norm baseline

    # Select only columns that actually exist in the DataFrame
    output_cols = ['Patient ID', 'Group', 'Target_Left_Ocular_Oral', 'Target_Right_Ocular_Oral', 'Prediction_Left', 'Prediction_Right'] + \
                  [col for col in core_aus_to_compare if col in df_analysis.columns]

    if len(output_cols) <= 6: # Only the ID/Group/Target/Pred columns, no features found
         logger.error("None of the core AU columns for comparison were found in the data. Cannot generate output.")
         return

    df_output = df_analysis[output_cols].copy()
    logger.info(f"Extracting {len(output_cols)-6} core AU columns for comparison...")

    # Save Output
    output_filename = "ocular_oral_error_AU_comparison.csv" # Specific filename
    output_path = os.path.join(output_dir, output_filename)
    try:
        # Save transposed version for easier comparison
        df_output.set_index(['Group', 'Patient ID', 'Target_Left_Ocular_Oral', 'Target_Right_Ocular_Oral', 'Prediction_Left', 'Prediction_Right']).sort_index().T.round(4).to_csv(output_path)
        logger.info(f"Error analysis CORE AU comparison saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save error analysis output: {e}", exc_info=True)

    logger.info("--- Ocular-Oral Error Analysis Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Ocular-Oral Synkinesis Model Errors.")
    parser.add_argument("--results_file", type=str, default="combined_results.csv", help="Path to the combined results CSV file.")
    parser.add_argument("--expert_file", type=str, default="FPRS FP Key.csv", help="Path to the expert key CSV file.")
    parser.add_argument("--model_dir", type=str, default="models/synkinesis/ocular_oral", help="Directory containing model.pkl, scaler.pkl, features.list.") # Corrected default
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save the analysis output.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    analyze_errors(args.results_file, args.expert_file, args.model_dir, args.output_dir)