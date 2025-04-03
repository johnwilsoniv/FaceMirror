# snarl_smile_detector.py (Refactored with DataFrame Scaling)

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
import warnings

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
        # ... (Identical loading logic as other detectors) ...
        try:
            model_path = MODEL_FILENAMES.get('model')
            scaler_path = MODEL_FILENAMES.get('scaler')
            features_path = MODEL_FILENAMES.get('feature_list')
            if not all([model_path, scaler_path, features_path]): logger.error("Missing snarl-smile config paths."); return
            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path] if p)
            if paths_exist:
                self.model = joblib.load(model_path); self.scaler = joblib.load(scaler_path); self.feature_names = joblib.load(features_path)
                logger.info("Snarl-Smile artifacts loaded."); assert isinstance(self.feature_names, list), "Feature names not a list"
                expected = len(self.feature_names); scaler_n = getattr(self.scaler, 'n_features_in_', None); model_n = getattr(self.model, 'n_features_in_', None)
                logger.debug(f"Loaded {expected} features.")
                if scaler_n is not None and expected != scaler_n: logger.warning(f"Scaler mismatch: {expected} vs {scaler_n}")
                if model_n is not None and expected != model_n: logger.warning(f"Model mismatch: {expected} vs {model_n}")
            else: logger.error("Snarl-Smile artifacts missing."); self.model, self.scaler, self.feature_names = None, None, None
        except Exception as e: logger.error(f"Load error: {e}", exc_info=True); self.model, self.scaler, self.feature_names = None, None, None


    def detect_snarl_smile_synkinesis(self, row_data, side):
        """ Detects Snarl-Smile synkinesis using ML model. """
        if not all([self.model, self.scaler, self.feature_names]):
            logger.warning("Snarl-Smile components unavailable."); return False, 0.0
        if extract_features_for_detection is None:
             logger.error("Snarl-Smile feature extraction unavailable."); return False, 0.0

        try:
            features_list = extract_features_for_detection(row_data, side)
            if features_list is None: logger.error(f"SnSm-Feat extract fail {side}"); return False, 0.0
            if len(features_list) != len(self.feature_names): logger.error(f"SnSm-Feat mismatch {side}"); return False, 0.0

            # --- Create DataFrame with feature names for scaling ---
            features_df = pd.DataFrame([features_list], columns=self.feature_names)
            scaled_features = self.scaler.transform(features_df) # Pass DataFrame
            # --- End DataFrame scaling ---

            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            is_detected = bool(prediction == 1)
            confidence = probabilities[prediction] if prediction < len(probabilities) else 0.0
            logger.debug(f"ML Snarl-Smile Detect {side}: Pred={prediction}, Conf={confidence:.3f}")
            return is_detected, confidence

        except Exception as e:
            logger.error(f"Error in Snarl-Smile detect {side}: {e}", exc_info=True)
            return False, 0.0