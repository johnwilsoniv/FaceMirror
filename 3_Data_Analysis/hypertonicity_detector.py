# hypertonicity_detector.py
# Detects hypertonicity using a trained ML model.

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

try:
    from hypertonicity_config import MODEL_FILENAMES, CLASS_NAMES, LOG_DIR, LOGGING_CONFIG
    from hypertonicity_features import extract_features_for_detection
except ImportError:
    logging.error("Failed imports in hypertonicity_detector")
    extract_features_for_detection = None
    MODEL_FILENAMES = {}; CLASS_NAMES = {0:'None', 1:'Hypertonicity'}; LOG_DIR = '.'; LOGGING_CONFIG = {}

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)

class HypertonicityDetector:
    """ Detects resting hypertonicity using a trained ML model. """
    def __init__(self):
        logger.info("Initializing HypertonicityDetector")
        self.model = None; self.scaler = None; self.feature_names = None
        self._load_models()

    def _load_models(self):
        """Load the model, scaler, and feature names list."""
        try:
            model_path = MODEL_FILENAMES.get('model')
            scaler_path = MODEL_FILENAMES.get('scaler')
            features_path = MODEL_FILENAMES.get('feature_list')

            if not all([model_path, scaler_path, features_path]):
                logger.error("Missing hypertonicity model/scaler/features paths in config.")
                return

            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path] if p)
            if not paths_exist:
                 missing = [p for p in [model_path, scaler_path, features_path] if not os.path.exists(p)]
                 logger.error(f"Hypertonicity artifact(s) missing: {missing}")
                 self.model, self.scaler, self.feature_names = None, None, None; return

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            logger.info("Hypertonicity artifacts loaded.")

            if not isinstance(self.feature_names, list):
                logger.error(f"Loaded HyperT feature names not a list ({type(self.feature_names)})."); raise TypeError("Features not a list.")

            expected_features = len(self.feature_names)
            logger.debug(f"Expected {expected_features} features.")
            scaler_n_features = getattr(self.scaler, 'n_features_in_', None)
            if scaler_n_features is not None and expected_features != scaler_n_features: logger.warning(f"HyperT Scaler feature mismatch: Expected {expected_features}, Scaler has {scaler_n_features}")

            model_n_features = None
            if hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'n_features_in_'): model_n_features = self.model.estimator.n_features_in_
            elif hasattr(self.model, 'n_features_in_'): model_n_features = self.model.n_features_in_
            if model_n_features is not None and expected_features != model_n_features: logger.warning(f"HyperT Model feature mismatch: Expected {expected_features}, Model expects {model_n_features}")

        except Exception as e:
             logger.error(f"Load error HyperT artifacts: {e}", exc_info=True)
             self.model, self.scaler, self.feature_names = None, None, None


    def detect_hypertonicity(self, row_data, side):
        """
        Detects hypertonicity using ML model.

        Args:
            row_data (dict or pd.Series): Full patient data row.
            side (str): Side ('Left' or 'Right').

        Returns:
            tuple: (bool, float) -> (is_detected, confidence)
                   Returns (False, 0.0) if detection cannot be performed.
        """
        if not all([self.model, self.scaler, self.feature_names]):
            logger.warning("Hypertonicity detector components not loaded."); return False, 0.0
        if extract_features_for_detection is None:
             logger.error("Hypertonicity feature extraction unavailable."); return False, 0.0

        if side not in ['Left', 'Right']:
            logger.error(f"Invalid 'side': {side}. Must be 'Left' or 'Right'."); return False, 0.0
        side_label = side # Use directly

        try:
            features_list = extract_features_for_detection(row_data, side_label)

            if features_list is None: logger.error(f"HyperT feature extraction failed {side_label}"); return False, 0.0
            if len(features_list) != len(self.feature_names):
                logger.error(f"HyperT feature mismatch {side_label}: Got {len(features_list)}, Expected {len(self.feature_names)}")
                return False, 0.0

            features_df = pd.DataFrame([features_list], columns=self.feature_names)
            if features_df.isnull().values.any() or np.isinf(features_df.values).any():
                logger.warning(f"NaNs/Infs found BEFORE scaling HyperT ({side_label}). Filling/Replacing with 0.")
                features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)

            scaled_features = self.scaler.transform(features_df)

            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                 logger.warning(f"NaNs/Infs found AFTER scaling HyperT ({side_label}). Using 0 for prediction.")
                 scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            predicted_class_index = int(prediction)
            confidence = 0.0
            if 0 <= predicted_class_index < len(probabilities): confidence = probabilities[predicted_class_index]
            else: logger.warning(f"HyperT ({side_label}): Predicted class index {predicted_class_index} out of bounds for probabilities (len {len(probabilities)}).")

            is_detected = bool(prediction == 1) # Class 1 is 'Hypertonicity'

            logger.debug(f"ML Hypertonicity Detect {side_label}: Pred={prediction} ({CLASS_NAMES.get(prediction,'?')}), Conf={confidence:.3f}")
            return is_detected, confidence

        except Exception as e:
            logger.error(f"Error in Hypertonicity detect {side_label}: {e}", exc_info=True)
            return False, 0.0