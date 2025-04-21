# hypertonicity_detector.py
# Detects hypertonicity using a trained ML model and adjustable threshold.

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

try:
    # <<< Import DETECTION_THRESHOLD >>>
    from hypertonicity_config import MODEL_FILENAMES, CLASS_NAMES, LOG_DIR, LOGGING_CONFIG, DETECTION_THRESHOLD
    from hypertonicity_features import extract_features_for_detection
except ImportError:
    logging.error("Failed imports in hypertonicity_detector")
    extract_features_for_detection = None
    MODEL_FILENAMES = {}; CLASS_NAMES = {0:'None', 1:'Hypertonicity'}; LOG_DIR = '.'; LOGGING_CONFIG = {}
    DETECTION_THRESHOLD = 0.5 # Fallback

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)

class HypertonicityDetector:
    """ Detects resting hypertonicity using a trained ML model and adjustable threshold. """
    def __init__(self):
        logger.info("Initializing HypertonicityDetector")
        self.model = None; self.scaler = None; self.feature_names = None
        # <<< Store threshold from config >>>
        self.threshold = DETECTION_THRESHOLD
        logger.info(f"Using Detection Threshold: {self.threshold}")
        self._load_models()

    def _load_models(self):
        """Load the model, scaler, and feature names list."""
        try:
            model_path = MODEL_FILENAMES.get('model')
            scaler_path = MODEL_FILENAMES.get('scaler')
            features_path = MODEL_FILENAMES.get('feature_list')

            if not all([model_path, scaler_path, features_path]): logger.error("Missing hypertonicity paths."); return
            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path] if p)
            if not paths_exist:
                 missing = [p for p in [model_path, scaler_path, features_path] if not os.path.exists(p)]; logger.error(f"HyperT artifact(s) missing: {missing}"); return

            self.model = joblib.load(model_path); self.scaler = joblib.load(scaler_path); self.feature_names = joblib.load(features_path)
            logger.info("Hypertonicity artifacts loaded.")

            if not isinstance(self.feature_names, list): logger.error(f"Features not list."); self.model, self.scaler, self.feature_names = None, None, None; return
            expected_features = len(self.feature_names); logger.debug(f"Expected {expected_features} features.")
            scaler_n_features = getattr(self.scaler, 'n_features_in_', None)
            if scaler_n_features is not None and expected_features != scaler_n_features: logger.warning(f"HyperT Scaler mismatch: Exp {expected_features}, Has {scaler_n_features}")
            model_n_features = None
            if hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'n_features_in_'): model_n_features = self.model.estimator.n_features_in_
            elif hasattr(self.model, 'n_features_in_'): model_n_features = self.model.n_features_in_
            if model_n_features is not None and expected_features != model_n_features: logger.warning(f"HyperT Model mismatch: Exp {expected_features}, Has {model_n_features}")

        except Exception as e: logger.error(f"Load error HyperT artifacts: {e}", exc_info=True); self.model, self.scaler, self.feature_names = None, None, None


    def detect_hypertonicity(self, row_data, side):
        """
        Detects hypertonicity using ML model and adjustable threshold.

        Args:
            row_data (dict or pd.Series): Full patient data row.
            side (str): Side ('Left' or 'Right').

        Returns:
            tuple: (bool, float) -> (is_detected, confidence)
                   Returns (False, 0.0) if detection cannot be performed.
        """
        if not all([self.model, self.scaler, self.feature_names]): logger.warning("Hypertonicity components not loaded."); return False, 0.0
        if extract_features_for_detection is None: logger.error("Hypertonicity feature extraction unavailable."); return False, 0.0

        if side not in ['Left', 'Right']: logger.error(f"Invalid 'side': {side}. Must be 'Left' or 'Right'."); return False, 0.0
        side_label = side # Use directly

        try:
            features_list = extract_features_for_detection(row_data, side_label)
            if features_list is None: logger.error(f"HyperT feature extraction failed {side_label}"); return False, 0.0
            if len(features_list) != len(self.feature_names): logger.error(f"HyperT feature mismatch {side_label}: Got {len(features_list)}, Expected {len(self.feature_names)}"); return False, 0.0

            features_df = pd.DataFrame([features_list], columns=self.feature_names)
            if features_df.isnull().values.any() or np.isinf(features_df.values).any(): logger.warning(f"NaNs/Infs found BEFORE scaling HyperT ({side_label}). Filling 0."); features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)

            scaled_features = self.scaler.transform(features_df)
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any(): logger.warning(f"NaNs/Infs found AFTER scaling HyperT ({side_label}). Using 0."); scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Use predict_proba and Threshold ---
            probabilities = self.model.predict_proba(scaled_features)[0]

            if len(probabilities) < 2:
                logger.error(f"HyperT ({side_label}): predict_proba returned unexpected shape: {probabilities.shape}. Cannot determine positive class probability.")
                return False, 0.0

            # Assuming Class 1 ('Hypertonicity') is the positive class (index 1)
            positive_class_proba = probabilities[1]

            # Apply the threshold
            is_detected = bool(positive_class_proba >= self.threshold)
            # --- End Threshold Logic ---

            # --- Calculate Confidence ---
            predicted_class_index = 1 if is_detected else 0
            confidence = probabilities[predicted_class_index]
            # --- End Confidence Calculation ---

            predicted_label = CLASS_NAMES.get(predicted_class_index, '?')
            logger.debug(f"ML Hypertonicity Detect {side_label}: Prob(HyperT)={positive_class_proba:.3f}, Threshold={self.threshold}, Pred={predicted_class_index} ({predicted_label}), Conf={confidence:.3f}")
            return is_detected, confidence

        except Exception as e:
            logger.error(f"Error in Hypertonicity detect {side_label}: {e}", exc_info=True)
            return False, 0.0