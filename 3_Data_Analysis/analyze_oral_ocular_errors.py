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

# --- Import necessary components for Oral-Ocular ---
try:
    # We still need the feature extraction function to re-run predictions
    from oral_ocular_features import extract_features_for_detection
    ORAL_OCULAR_FEATURE_EXTRACTION_AVAILABLE = True
except ImportError:
    logging.error("CRITICAL: Could not import extract_features_for_detection from oral_ocular_features.py.")
    ORAL_OCULAR_FEATURE_EXTRACTION_AVAILABLE = False

try:
    # Assuming oral_ocular_config.py is accessible
    from oral_ocular_config import CLASS_NAMES, MODEL_FILENAMES # Need MODEL_FILENAMES for artifacts
except ImportError:
    logging.warning("Could not import from oral_ocular_config. Using default.")
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}
    MODEL_FILENAMES = { # Add minimal fallback for load_artifacts
        'model': 'models/synkinesis/oral_ocular/model.pkl',
        'scaler': 'models/synkinesis/oral_ocular/scaler.pkl',
        'feature_list': 'models/synkinesis/oral_ocular/features.list'
    }
# --- End Imports ---

# Configure logging
ANALYSIS_OUTPUT_DIR = 'analysis_results'
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(ANALYSIS_OUTPUT_DIR, 'oral_ocular_error_analysis.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
                    force=True)
logger = logging.getLogger(__name__)


def load_artifacts(model_dir):
    """Loads model, scaler, and feature list."""
    logger.info(f"Loading Oral-Ocular artifacts from: {model_dir}")
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
         logger.warning(f"Unexpected Oral-Ocular expert labels treated as 'No': {unexpected.unique().tolist()}")
    final_mapped = mapped.fillna(0)
    return final_mapped.astype(int)

def analyze_errors(results_file, expert_file, model_dir, output_dir):
    """Main function to perform error analysis."""
    logger.info("--- Starting Oral-Ocular Error Analysis ---")

    if not ORAL_OCULAR_FEATURE_EXTRACTION_AVAILABLE:
        logger.error("Cannot proceed without feature extraction function.")
        return

    # Load Model Artifacts first to get expected features
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
            'Oral-Ocular Synkinesis Left': 'Expert_Left_Oral_Ocular',
            'Oral-Ocular Synkinesis Right': 'Expert_Right_Oral_Ocular'
        })
        for col in ['Expert_Left_Oral_Ocular', 'Expert_Right_Oral_Ocular']:
            target_col = col.replace('Expert', 'Target')
            if col in df_expert.columns:
                df_expert[target_col] = process_targets(df_expert[col])
            else:
                logger.warning(f"Missing expert column: {col}. Creating default target column.")
                df_expert[target_col] = 0

        # Merge
        df_results['Patient ID'] = df_results['Patient ID'].astype(str).str.strip()
        df_expert['Patient ID'] = df_expert['Patient ID'].astype(str).str.strip()
        expert_cols_to_merge = ['Patient ID', 'Target_Left_Oral_Ocular', 'Target_Right_Oral_Ocular']
        df = pd.merge(df_results, df_expert[expert_cols_to_merge], on='Patient ID', how='inner')
        logger.info(f"Loaded and merged data for {len(df)} patients.")
        if df.empty: raise ValueError("Merge resulted in empty DataFrame.")

    except Exception as e:
        logger.error(f"Failed to load or prepare data: {e}", exc_info=True)
        return

    # Re-run Predictions
    predictions = {}
    logger.info("Re-running predictions using loaded model...")
    error_count = 0
    # --- Store extracted features for error cases ---
    error_case_features = {} # Key: (patient_id, side), Value: feature_dict_final from extraction
    # ---

    for index, row in df.iterrows():
        patient_id = row['Patient ID']
        row_data = row.to_dict()
        for side in ['Left', 'Right']:
            prediction_result = -1 # Default to error
            try:
                # --- Extract features FIRST ---
                features_list = extract_features_for_detection(row_data, side)

                if features_list is None:
                    logger.warning(f"Feature extraction failed for {patient_id} - {side}. Skipping prediction.")
                    error_count += 1
                elif len(features_list) != len(model_features):
                    logger.error(f"Feature mismatch for {patient_id} - {side}. Expected {len(model_features)}, got {len(features_list)}. Skipping prediction.")
                    error_count += 1
                else:
                    # --- Create DataFrame only if features are valid ---
                    features_df = pd.DataFrame([features_list], columns=model_features)
                    if features_df.isnull().values.any():
                        nan_cols = features_df.columns[features_df.isna().any()].tolist()
                        logger.warning(f"NaNs found in extracted features for {patient_id} - {side}: {nan_cols}. Filling with 0.")
                        features_df = features_df.fillna(0)

                    scaled_features = scaler.transform(features_df)
                    prediction = model.predict(scaled_features)[0]
                    prediction_result = int(prediction)

                    # --- Store features if it's an error case (for later comparison) ---
                    is_error = False
                    target_col = f"Target_{side}_Oral_Ocular"
                    if target_col in df.columns: # Check if target column exists
                       actual_target = df.loc[index, target_col]
                       if prediction_result != actual_target:
                           is_error = True
                           # Store the list (or convert back to dict if easier)
                           error_case_features[(patient_id, side)] = dict(zip(model_features, features_list))
                    # --- End storing features ---

            except Exception as pred_e:
                logger.error(f"Prediction failed for {patient_id} - {side}: {pred_e}", exc_info=True)
                error_count += 1
            finally:
                predictions[(patient_id, side)] = prediction_result # Store prediction or -1

    logger.info(f"Predictions complete. Encountered {error_count} errors during prediction/feature extraction.")

    # Add predictions to DataFrame
    df['Prediction_Left'] = df['Patient ID'].apply(lambda pid: predictions.get((pid, 'Left'), -1))
    df['Prediction_Right'] = df['Patient ID'].apply(lambda pid: predictions.get((pid, 'Right'), -1))

    # Identify Groups (Focus on Right Side)
    logger.info("Identifying analysis groups (FN_Right, FP_Right, TP_Right, TN_Right_Sample)...")
    df_analysis_list = []

    # False Negatives (Right)
    fn_right_df = df[(df['Target_Right_Oral_Ocular'] == 1) & (df['Prediction_Right'] == 0)].copy()
    fn_right_df['Group'] = 'FN_Right'
    if not fn_right_df.empty: df_analysis_list.append(fn_right_df)
    logger.info(f"Found {len(fn_right_df)} FN_Right cases.")

    # False Positives (Right)
    fp_right_df = df[(df['Target_Right_Oral_Ocular'] == 0) & (df['Prediction_Right'] == 1)].copy()
    fp_right_df['Group'] = 'FP_Right'
    if not fp_right_df.empty: df_analysis_list.append(fp_right_df)
    logger.info(f"Found {len(fp_right_df)} FP_Right cases.")

    # True Positives (Right)
    tp_right_df = df[(df['Target_Right_Oral_Ocular'] == 1) & (df['Prediction_Right'] == 1)].copy()
    tp_right_df['Group'] = 'TP_Right'
    if not tp_right_df.empty: df_analysis_list.append(tp_right_df)
    logger.info(f"Found {len(tp_right_df)} TP_Right cases.")

    # True Negatives (Right) - Sample
    tn_candidates_right = df[(df['Target_Right_Oral_Ocular'] == 0) & (df['Prediction_Right'] == 0)]
    sample_size_tn = min(5, len(tn_candidates_right)) # Sample up to 5 TNs
    tn_right_df_sample = tn_candidates_right.sample(n=sample_size_tn, random_state=42).copy()
    tn_right_df_sample['Group'] = 'TN_Right_Sample'
    if not tn_right_df_sample.empty: df_analysis_list.append(tn_right_df_sample)
    logger.info(f"Sampled {len(tn_right_df_sample)} TN_Right cases.")

    if not df_analysis_list:
        logger.warning("No cases found for analysis groups (FN, FP, TP, TN). Cannot generate comparison.")
        return

    df_analysis = pd.concat(df_analysis_list, ignore_index=True)

    # --- Extract RAW/NORMALIZED AUs for Comparison ---
    # Define the core AU columns needed to understand the errors
    core_aus_to_compare = [
        # BS Right Side - Trigger/Coupled Raw + Norm
        'BS_Right AU12_r', 'BS_Right AU12_r (Normalized)',
        'BS_Right AU25_r', 'BS_Right AU25_r (Normalized)',
        'BS_Right AU06_r', 'BS_Right AU06_r (Normalized)',
        'BS_Right AU45_r', 'BS_Right AU45_r (Normalized)',
        # SS Right Side - Trigger/Coupled Raw + Norm
        'SS_Right AU12_r', 'SS_Right AU12_r (Normalized)',
        'SS_Right AU25_r', 'SS_Right AU25_r (Normalized)',
        'SS_Right AU06_r', 'SS_Right AU06_r (Normalized)',
        'SS_Right AU45_r', 'SS_Right AU45_r (Normalized)',
        # Include Left side BS AUs for asymmetry context
        'BS_Left AU12_r (Normalized)', 'BS_Left AU25_r (Normalized)',
        'BS_Left AU06_r (Normalized)', 'BS_Left AU45_r (Normalized)',
    ]

    # Select only columns that actually exist in the DataFrame
    output_cols = ['Patient ID', 'Group', 'Target_Right_Oral_Ocular', 'Prediction_Right'] + \
                  [col for col in core_aus_to_compare if col in df_analysis.columns]

    if len(output_cols) <= 4: # Only the ID/Group/Target/Pred columns, no features found
         logger.error("None of the core AU columns for comparison were found in the data. Cannot generate output.")
         return

    df_output = df_analysis[output_cols].copy()
    logger.info(f"Extracting {len(output_cols)-4} core AU columns for comparison...")

    # Save Output
    output_filename = "oral_ocular_error_AU_comparison.csv" # Changed filename
    output_path = os.path.join(output_dir, output_filename)
    try:
        # Save transposed version for easier comparison
        df_output.set_index(['Group', 'Patient ID', 'Target_Right_Oral_Ocular', 'Prediction_Right']).sort_index().T.round(4).to_csv(output_path)
        logger.info(f"Error analysis CORE AU comparison saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save error analysis output: {e}", exc_info=True)

    logger.info("--- Oral-Ocular Error Analysis Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Oral-Ocular Synkinesis Model Errors.")
    parser.add_argument("--results_file", type=str, default="combined_results.csv", help="Path to the combined results CSV file.")
    parser.add_argument("--expert_file", type=str, default="FPRS FP Key.csv", help="Path to the expert key CSV file.")
    parser.add_argument("--model_dir", type=str, default="models/synkinesis/oral_ocular", help="Directory containing model.pkl, scaler.pkl, features.list.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save the analysis output.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    analyze_errors(args.results_file, args.expert_file, args.model_dir, args.output_dir)