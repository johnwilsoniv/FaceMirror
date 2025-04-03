# analyze_probabilities.py

import pandas as pd
import numpy as np
import joblib
import os
import logging
import json # Import json for loading thresholds

# --- Setup logging FIRST ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- END Logging Setup ---

# Attempt to import from config, define fallbacks if it fails
try:
    from lower_face_config import MODEL_FILENAMES, LOG_DIR, CLASS_NAMES, DETECTION_THRESHOLDS
except ImportError:
    # Define fallbacks if config import fails
    logger.warning("Could not import from lower_face_config. Using fallback definitions.") # Now logger exists
    LOG_DIR = 'logs'
    MODEL_DIR = 'models'
    MODEL_FILENAMES = {
        'base_model': os.path.join(MODEL_DIR, 'lower_face_model.pkl'),
        'base_scaler': os.path.join(MODEL_DIR, 'lower_face_scaler.pkl'),
    }
    CLASS_NAMES = {0: 'None', 1: 'Partial', 2: 'Complete'}
    # DEFINE DETECTION_THRESHOLDS HERE AS A FALLBACK
    DETECTION_THRESHOLDS = {
        'complete_confidence': 0.7,
        'none_probability': 0.3,
        'partial_probability': 0.25,
        'upgrade_to_partial': 0.4, # Provide a default value
        'specialist_complete_lower': 0.5,
        'specialist_complete_upper': 0.7,
        'specialist_partial_threshold': 0.3
    }


def analyze_test_set_probabilities():
    """
    Loads the saved test set, base model, and scaler, then analyzes
    the predicted probabilities for the 'Partial' class.
    """
    logger.info("--- Analyzing Base Model Probabilities on Test Set ---")

    # --- Load Saved Data ---
    model_path = MODEL_FILENAMES.get('base_model')
    scaler_path = MODEL_FILENAMES.get('base_scaler')
    test_data_path = os.path.join(LOG_DIR, 'lower_face_test_data.pkl')

    # Check for existence using os.path.exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        return
    if not os.path.exists(test_data_path):
        logger.error(f"Test data file not found: {test_data_path}. Please run training script first.")
        return

    try:
        base_model = joblib.load(model_path)
        base_scaler = joblib.load(scaler_path)
        test_set = joblib.load(test_data_path)
        X_test = test_set['X_test']
        y_test = test_set['y_test']
        feature_names = test_set['feature_names']
        logger.info("Loaded model, scaler, and test data.")
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return

    # Ensure X_test is a DataFrame with correct columns for scaling
    try:
        if not isinstance(X_test, pd.DataFrame):
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
        else:
            # If already a DF, ensure columns match
            X_test_df = X_test[feature_names]
    except Exception as df_e:
        logger.error(f"Error preparing X_test DataFrame: {df_e}")
        return

    # --- Scale and Predict Probabilities ---
    try:
        X_test_scaled = base_scaler.transform(X_test_df)
        probabilities = base_model.predict_proba(X_test_scaled)
        # Ensure we know which column corresponds to which class
        model_classes_list = list(base_model.classes_)
        idx_none = model_classes_list.index(0) if 0 in model_classes_list else -1
        idx_partial = model_classes_list.index(1) if 1 in model_classes_list else -1
        idx_complete = model_classes_list.index(2) if 2 in model_classes_list else -1


        if idx_partial == -1:
             logger.error("Model classes do not contain expected class 1 (Partial). Cannot analyze probabilities.")
             logger.info(f"Model classes found: {model_classes_list}")
             return

    except Exception as pred_e:
        logger.error(f"Error during scaling or prediction: {pred_e}")
        return

    # --- Analyze Probabilities for True 'Partial' Cases ---
    partial_indices = np.where(y_test == 1)[0] # Get indices where true label is Partial

    if len(partial_indices) == 0:
        logger.info("No true 'Partial' cases found in the test set.")
        # Optionally print probabilities for ALL cases if partial is missing
        logger.info("\nProbabilities for all test cases:")
        logger.info("Idx | True | P(None)  | P(Partial) | P(Complete)")
        logger.info("-" * 50)
        for i in range(len(y_test)):
             prob_none_str = f"{probabilities[i, idx_none]:.4f}" if idx_none != -1 and idx_none < probabilities.shape[1] else "N/A"
             prob_partial_str = f"{probabilities[i, idx_partial]:.4f}" if idx_partial != -1 and idx_partial < probabilities.shape[1] else "N/A"
             prob_complete_str = f"{probabilities[i, idx_complete]:.4f}" if idx_complete != -1 and idx_complete < probabilities.shape[1] else "N/A"
             logger.info(f"{i:3d} | {y_test[i]:4d} | {prob_none_str} | {prob_partial_str}   | {prob_complete_str}")
        return

    logger.info(f"\nProbabilities for {len(partial_indices)} True 'Partial' Cases (True Class = 1):")
    logger.info("SampleIdx | P(None)  | P(Partial) | P(Complete)")
    logger.info("-" * 50)

    partial_probs_for_class1 = []
    max_partial_prob = 0.0
    min_partial_prob = 1.0

    for i in partial_indices:
        # Check indices validity before accessing
        prob_none_str = f"{probabilities[i, idx_none]:.4f}" if idx_none != -1 and idx_none < probabilities.shape[1] else "N/A"
        prob_partial = probabilities[i, idx_partial] # We know idx_partial exists and is valid if we reached here
        prob_complete_str = f"{probabilities[i, idx_complete]:.4f}" if idx_complete != -1 and idx_complete < probabilities.shape[1] else "N/A"

        logger.info(f"{i:9d} | {prob_none_str} | {prob_partial:.4f}   | {prob_complete_str}")
        partial_probs_for_class1.append(prob_partial)
        max_partial_prob = max(max_partial_prob, prob_partial)
        min_partial_prob = min(min_partial_prob, prob_partial)

    logger.info("-" * 50)
    logger.info(f"Max probability assigned to 'Partial' for true 'Partial' cases: {max_partial_prob:.4f}")
    logger.info(f"Min probability assigned to 'Partial' for true 'Partial' cases: {min_partial_prob:.4f}")
    avg_partial_prob = np.mean(partial_probs_for_class1) if partial_probs_for_class1 else 0.0
    logger.info(f"Avg probability assigned to 'Partial' for true 'Partial' cases: {avg_partial_prob:.4f}")

    # --- Check against the upgrade threshold ---
    # Load the *actually saved* thresholds to get the operative upgrade threshold
    saved_thresholds_path = os.path.join(LOG_DIR, 'lower_face_optimal_thresholds.json')
    # Default from config or a reasonable fallback if config also missing
    # Use globals() check for safety in case DETECTION_THRESHOLDS wasn't defined at all
    default_upgrade_threshold = DETECTION_THRESHOLDS.get('upgrade_to_partial', 0.3) if 'DETECTION_THRESHOLDS' in globals() else 0.3
    upgrade_threshold = default_upgrade_threshold

    if os.path.exists(saved_thresholds_path):
        try:
            with open(saved_thresholds_path, 'r') as f:
                saved_thresholds = json.load(f)
                # Use .get() on the loaded dictionary for safety
                upgrade_threshold = saved_thresholds.get('upgrade_to_partial', default_upgrade_threshold)
                logger.info(f"Using 'upgrade_to_partial' threshold from saved file: {upgrade_threshold:.4f}")
        except Exception as json_e:
            logger.warning(f"Could not load saved thresholds, using default 'upgrade_to_partial': {json_e}")
            upgrade_threshold = default_upgrade_threshold # Ensure it's set
    else:
         logger.warning(f"Saved thresholds file not found at '{saved_thresholds_path}'. Using default 'upgrade_to_partial'.")


    num_below_upgrade = sum(p < upgrade_threshold for p in partial_probs_for_class1)
    logger.info(f"\nNumber of true 'Partial' cases where P(Partial) < {upgrade_threshold:.4f} (upgrade threshold): {num_below_upgrade} / {len(partial_indices)}")

    if max_partial_prob < upgrade_threshold:
        logger.warning(f"CRITICAL: The HIGHEST probability the model assigned to any true 'Partial' case ({max_partial_prob:.4f}) is BELOW the operative upgrade threshold ({upgrade_threshold:.4f}). Threshold tuning cannot fix this.")
    elif num_below_upgrade == len(partial_indices) and len(partial_indices) > 0:
         logger.warning(f"CRITICAL: ALL {len(partial_indices)} true 'Partial' cases have P(Partial) below the operative upgrade threshold ({upgrade_threshold:.4f}). Model is not separating this class well enough.")
    elif num_below_upgrade > 0:
         logger.warning(f"NOTE: {num_below_upgrade} true 'Partial' cases have P(Partial) below the operative upgrade threshold ({upgrade_threshold:.4f}). These specific cases weren't recovered by the current thresholding.")


if __name__ == "__main__":
    analyze_test_set_probabilities()