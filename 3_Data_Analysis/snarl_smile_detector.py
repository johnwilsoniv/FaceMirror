# snarl_smile_detector.py
# - Capitalization fix
# - Use DataFrame with names for scaling
# - Improved loading/validation

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore specific warning about feature names in sklearn prediction/transform
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

try:
    from snarl_smile_config import MODEL_FILENAMES, CLASS_NAMES, LOG_DIR, LOGGING_CONFIG
    from snarl_smile_features import extract_features_for_detection
except ImportError:
    logging.error("Failed imports in snarl_smile_detector")
    extract_features_for_detection = None
    MODEL_FILENAMES = {}; CLASS_NAMES = {0:'None', 1:'Synkinesis'}; LOG_DIR = '.'; LOGGING_CONFIG = {}

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)

class SnarlSmileDetector:
    """ Detects Snarl-Smile synkinesis using a trained ML model. """
    def __init__(self):
        logger.info("Initializing SnarlSmileDetector")
        self.model = None; self.scaler = None; self.feature_names = None
        self._load_models()

    def _load_models(self):
        """Load the model, scaler, and feature names list."""
        try:
            model_path = MODEL_FILENAMES.get('model')
            scaler_path = MODEL_FILENAMES.get('scaler')
            features_path = MODEL_FILENAMES.get('feature_list')

            if not all([model_path, scaler_path, features_path]):
                logger.error("Missing snarl-smile model/scaler/features paths in config.")
                return

            # Check existence first
            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path] if p)
            if not paths_exist:
                 missing = [p for p in [model_path, scaler_path, features_path] if not os.path.exists(p)]
                 logger.error(f"Snarl-Smile artifact(s) missing: {missing}")
                 self.model, self.scaler, self.feature_names = None, None, None
                 return

            # Load artifacts
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            logger.info("Snarl-Smile artifacts loaded.")

            # --- Validation ---
            if not isinstance(self.feature_names, list):
                logger.error(f"Loaded Snarl-Smile feature names is not a list ({type(self.feature_names)}). Detector unusable.")
                self.model, self.scaler, self.feature_names = None, None, None; return

            expected_features = len(self.feature_names)
            logger.debug(f"Expected {expected_features} features based on loaded list.")

            # Check Scaler features
            scaler_n_features = getattr(self.scaler, 'n_features_in_', None)
            if scaler_n_features is not None and expected_features != scaler_n_features:
                logger.warning(f"SnSm Scaler feature count mismatch: Expected {expected_features}, Scaler was fit with {scaler_n_features}")
                # Continue, but scaling might fail or be incorrect

            # Check Model features (handle calibrated models)
            model_n_features = None
            if hasattr(self.model, 'n_features_in_'): # Direct attribute (e.g., simple models)
                 model_n_features = self.model.n_features_in_
            elif hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'n_features_in_'): # CalibratedClassifierCV case
                 model_n_features = self.model.estimator.n_features_in_
                 logger.debug("Detected calibrated model structure for feature count check.")
            elif hasattr(self.model, 'feature_importances_'): # Fallback for tree models
                 model_n_features = len(self.model.feature_importances_)

            if model_n_features is not None and expected_features != model_n_features:
                 logger.warning(f"SnSm Model feature count mismatch: Expected {expected_features}, Model expects/has {model_n_features}")
                 # Continue, but prediction will likely fail

        except FileNotFoundError as fnf_e:
             logger.error(f"Load error SnSm artifacts: File not found - {fnf_e}", exc_info=False)
             self.model, self.scaler, self.feature_names = None, None, None
        except Exception as e:
             logger.error(f"Load error SnSm artifacts: {e}", exc_info=True)
             self.model, self.scaler, self.feature_names = None, None, None


    def detect_snarl_smile_synkinesis(self, row_data, side):
        """
        Detects Snarl-Smile synkinesis using ML model.

        Args:
            row_data (dict or pd.Series): Full patient data row.
            side (str): Side ('Left' or 'Right').

        Returns:
            tuple: (bool, float) -> (is_detected, confidence)
                   Returns (False, 0.0) if detection cannot be performed.
        """
        # --- Check if components are loaded ---
        if not all([self.model, self.scaler, self.feature_names]):
            logger.warning("Snarl-Smile detector components (model/scaler/features) not loaded. Cannot detect.")
            return False, 0.0
        if extract_features_for_detection is None:
             logger.error("Snarl-Smile feature extraction function is unavailable.")
             return False, 0.0

        # --- Ensure side is capitalized ---
        if side not in ['Left', 'Right']:
            logger.error(f"Invalid 'side' argument '{side}' in detect_snarl_smile_synkinesis. Must be 'Left' or 'Right'. Attempting capitalization.")
            side_label = side.capitalize()
            if side_label not in ['Left', 'Right']:
                 logger.error(f"Side '{side}' still invalid after capitalization. Aborting detection.")
                 return False, 0.0
        else:
            side_label = side
        # --- End Ensure side ---

        try:
            # --- Pass capitalized side to feature extraction ---
            features_list = extract_features_for_detection(row_data, side_label)
            # --- End Pass capitalized side ---

            if features_list is None:
                logger.error(f"SnSm feature extraction failed for side {side_label}."); return False, 0.0
            if len(features_list) != len(self.feature_names):
                logger.error(f"SnSm feature mismatch for side {side_label}: Extracted {len(features_list)}, Expected {len(self.feature_names)}. Check feature list file and extraction logic.")
                # Ensure feature list file ('features.list') is up-to-date with the trained model.
                return False, 0.0

            # --- Create DataFrame with feature names for scaling ---
            # This is crucial if the scaler was fitted on a DataFrame with names
            features_df = pd.DataFrame([features_list], columns=self.feature_names)

            # Check for NaNs/Infs *before* scaling
            if features_df.isnull().values.any() or np.isinf(features_df.values).any():
                nan_cols = features_df.columns[features_df.isna().any()].tolist()
                inf_cols = features_df.columns[np.isinf(features_df.values).any(axis=0)].tolist()
                logger.warning(f"NaNs/Infs found in features BEFORE scaling SnSm ({side_label}): NaN in {nan_cols}, Inf in {inf_cols}. Filling with 0.")
                features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0) # Fill NaNs and replace Infs

            # --- Scale using the loaded scaler ---
            # Pass the DataFrame with correct column names/order
            scaled_features = self.scaler.transform(features_df)
            # --- End scaling ---

            # Check for NaNs/Infs *after* scaling (less likely but possible with problematic data/scalers)
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                logger.warning(f"NaNs/Infs found AFTER scaling SnSm ({side_label}). Check data and scaler. Using 0 for prediction input.")
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)


            # --- Predict using the loaded (calibrated) model ---
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            # --- End Prediction ---


            # Get confidence for the predicted class
            predicted_class_index = int(prediction)
            confidence = 0.0
            if 0 <= predicted_class_index < len(probabilities):
                 confidence = probabilities[predicted_class_index]
            else:
                 logger.warning(f"SnSm ({side_label}): Predicted class index {predicted_class_index} out of bounds for probabilities array (len {len(probabilities)}). Setting confidence to 0.")
                 # This might indicate an issue with the model or prediction process

            # Determine if synkinesis is detected (assuming class 1 is 'Synkinesis')
            is_detected = bool(prediction == 1)

            logger.debug(f"ML Snarl-Smile Detect {side_label}: Pred={prediction} ({CLASS_NAMES.get(prediction,'?')}), Conf={confidence:.3f}")
            return is_detected, confidence

        except Exception as e:
            logger.error(f"Error during Snarl-Smile detection for side {side_label}: {e}", exc_info=True)
            return False, 0.0 # Return False on any error during the process