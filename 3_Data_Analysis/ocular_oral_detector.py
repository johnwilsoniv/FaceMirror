# ocular_oral_detector.py (Refactored with DataFrame Scaling & Capitalization Fix)

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

try:
    from ocular_oral_config import MODEL_FILENAMES, CLASS_NAMES, LOG_DIR, LOGGING_CONFIG
    from ocular_oral_features import extract_features_for_detection
except ImportError:
    logging.error("Failed imports in ocular_oral_detector")
    extract_features_for_detection = None
    MODEL_FILENAMES = {}; CLASS_NAMES = {0:'None', 1:'Synkinesis'}; LOG_DIR = '.'; LOGGING_CONFIG = {}

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)

class OcularOralDetector:
    """ Detects Ocular-Oral synkinesis using a trained ML model. """
    def __init__(self):
        logger.info("Initializing OcularOralDetector")
        self.model = None; self.scaler = None; self.feature_names = None
        self._load_models()

    def _load_models(self):
        """Load the model, scaler, and feature names list."""
        # ... (Identical loading logic as previous version) ...
        try:
            model_path = MODEL_FILENAMES.get('model')
            scaler_path = MODEL_FILENAMES.get('scaler')
            features_path = MODEL_FILENAMES.get('feature_list')
            if not all([model_path, scaler_path, features_path]): logger.error("Missing ocular-oral config paths."); return
            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path] if p)
            if paths_exist:
                self.model = joblib.load(model_path); self.scaler = joblib.load(scaler_path); self.feature_names = joblib.load(features_path)
                logger.info("Ocular-Oral artifacts loaded."); assert isinstance(self.feature_names, list), "Feature names not a list"
                expected = len(self.feature_names); scaler_n = getattr(self.scaler, 'n_features_in_', None); model_n = getattr(self.model, 'n_features_in_', None)
                logger.debug(f"Loaded {expected} features.")
                # Check potential length mismatches (optional)
                if scaler_n is not None and expected != scaler_n: logger.warning(f"OcOr Scaler feature mismatch: Expected {expected}, Scaler has {scaler_n}")
                # For XGBoost, n_features_in_ might not be set until fit, or might be under booster
                xgb_model_n = None
                if hasattr(self.model, 'n_features_in_'): # CalibratedClassifierCV might have it
                    xgb_model_n = self.model.n_features_in_
                elif hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'n_features_in_'): # Check base estimator
                    xgb_model_n = self.model.estimator.n_features_in_

                if xgb_model_n is not None and expected != xgb_model_n:
                     logger.warning(f"OcOr Model feature mismatch: Expected {expected}, Model has {xgb_model_n}")

            else: logger.error("Ocular-Oral artifacts missing."); self.model, self.scaler, self.feature_names = None, None, None
        except Exception as e: logger.error(f"Load error OcOr artifacts: {e}", exc_info=True); self.model, self.scaler, self.feature_names = None, None, None


    def detect_ocular_oral_synkinesis(self, row_data, side):
        """
        Detects Ocular-Oral synkinesis using ML model.

        Args:
            row_data (dict or pd.Series): Full patient data row.
            side (str): Side ('Left' or 'Right' - MUST BE CAPITALIZED).

        Returns:
            tuple: (bool, float) -> (is_detected, confidence)
        """
        if not all([self.model, self.scaler, self.feature_names]):
            logger.warning("Ocular-Oral components unavailable."); return False, 0.0
        if extract_features_for_detection is None:
             logger.error("Ocular-Oral feature extraction unavailable."); return False, 0.0

        # --- Ensure side is capitalized ---
        if side not in ['Left', 'Right']:
            logger.error(f"Invalid 'side' argument '{side}' in detect_ocular_oral_synkinesis. Must be 'Left' or 'Right'.")
            side_label = side.capitalize() # Attempt to fix, but log error
        else:
            side_label = side
        # --- End Ensure side ---

        try:
            # --- Pass capitalized side to feature extraction ---
            features_list = extract_features_for_detection(row_data, side_label)
            # --- End Pass capitalized side ---

            if features_list is None: logger.error(f"OcOr-Feat extract fail {side_label}"); return False, 0.0
            if len(features_list) != len(self.feature_names):
                logger.error(f"OcOr-Feat mismatch {side_label}: Got {len(features_list)}, Expected {len(self.feature_names)}")
                return False, 0.0

            # --- Create DataFrame with feature names for scaling ---
            features_df = pd.DataFrame([features_list], columns=self.feature_names)
            # Check for NaNs before scaling
            if features_df.isnull().values.any():
                nan_cols = features_df.columns[features_df.isna().any()].tolist()
                logger.warning(f"NaNs found in features BEFORE scaling OcOr ({side_label}): {nan_cols}. Filling with 0.")
                features_df = features_df.fillna(0) # Fill NaNs before scaling
            scaled_features = self.scaler.transform(features_df) # Pass DataFrame
            # --- End DataFrame scaling ---

            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            # Get probability of the predicted class
            # Ensure prediction index is valid for probabilities array
            predicted_class_index = int(prediction)
            if 0 <= predicted_class_index < len(probabilities):
                 confidence = probabilities[predicted_class_index]
            else:
                 logger.warning(f"OcOr ({side_label}): Predicted class index {predicted_class_index} out of bounds for probabilities array (len {len(probabilities)}). Setting confidence to 0.")
                 confidence = 0.0

            is_detected = bool(prediction == 1) # Class 1 is 'Synkinesis'

            logger.debug(f"ML Ocular-Oral Detect {side_label}: Pred={prediction}, Conf={confidence:.3f}")
            return is_detected, confidence

        except Exception as e:
            logger.error(f"Error in Ocular-Oral detect {side_label}: {e}", exc_info=True)
            return False, 0.0