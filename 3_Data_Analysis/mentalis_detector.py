# mentalis_detector.py
# Detects Mentalis Synkinesis using a trained ML model.

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

try:
    from mentalis_config import MODEL_FILENAMES, CLASS_NAMES, LOG_DIR, LOGGING_CONFIG
    from mentalis_features import extract_features_for_detection
except ImportError:
    logging.error("Failed imports in mentalis_detector")
    extract_features_for_detection = None
    MODEL_FILENAMES = {}; CLASS_NAMES = {0:'None', 1:'Mentalis Synkinesis'}; LOG_DIR = '.'; LOGGING_CONFIG = {}

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)

class MentalisDetector:
    """ Detects Mentalis Synkinesis using a trained ML model. """
    def __init__(self):
        logger.info("Initializing MentalisDetector")
        self.model = None; self.scaler = None; self.feature_names = None
        self._load_models()

    def _load_models(self):
        """Load the model, scaler, and feature names list."""
        try:
            paths = MODEL_FILENAMES; model_p, scaler_p, feat_p = paths.get('model'), paths.get('scaler'), paths.get('feature_list')
            if not all([model_p, scaler_p, feat_p]): logger.error("Missing mentalis config paths."); return
            if not all(os.path.exists(p) for p in [model_p, scaler_p, feat_p] if p):
                 missing = [p for p in [model_p, scaler_p, feat_p] if not os.path.exists(p)]
                 logger.error(f"Mentalis artifact(s) missing: {missing}"); self.model, self.scaler, self.feature_names = None, None, None; return

            self.model = joblib.load(model_p); self.scaler = joblib.load(scaler_p); self.feature_names = joblib.load(feat_p)
            logger.info("Mentalis artifacts loaded.")
            if not isinstance(self.feature_names, list): logger.error("Features not list."); raise TypeError("Features not list")

            expected = len(self.feature_names); logger.debug(f"Expected {expected} features.")
            scaler_n = getattr(self.scaler, 'n_features_in_', None)
            if scaler_n is not None and expected != scaler_n: logger.warning(f"MentS Scaler mismatch: Exp {expected}, Has {scaler_n}")
            model_n = None
            if hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'n_features_in_'): model_n = self.model.estimator.n_features_in_
            elif hasattr(self.model, 'n_features_in_'): model_n = self.model.n_features_in_
            if model_n is not None and expected != model_n: logger.warning(f"MentS Model mismatch: Exp {expected}, Has {model_n}")

        except Exception as e: logger.error(f"Load error MentS artifacts: {e}", exc_info=True); self.model, self.scaler, self.feature_names = None, None, None

    def detect_mentalis_synkinesis(self, row_data, side):
        """
        Detects Mentalis Synkinesis using ML model.

        Args:
            row_data (dict or pd.Series): Full patient data row.
            side (str): Side ('Left' or 'Right').

        Returns:
            tuple: (bool, float) -> (is_detected, confidence)
        """
        if not all([self.model, self.scaler, self.feature_names]): logger.warning("Mentalis components unavailable."); return False, 0.0
        if extract_features_for_detection is None: logger.error("Mentalis feature extraction unavailable."); return False, 0.0
        if side not in ['Left', 'Right']: logger.error(f"Invalid 'side': {side}."); return False, 0.0
        side_label = side

        try:
            features_list = extract_features_for_detection(row_data, side_label)
            if features_list is None: logger.error(f"MentS feat extract fail {side_label}"); return False, 0.0
            if len(features_list) != len(self.feature_names): logger.error(f"MentS feat mismatch {side_label}: Got {len(features_list)}, Exp {len(self.feature_names)}"); return False, 0.0

            features_df = pd.DataFrame([features_list], columns=self.feature_names)
            if features_df.isnull().values.any() or np.isinf(features_df.values).any(): logger.warning(f"NaNs/Infs BEFORE scaling MentS ({side_label}). Filling 0."); features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
            scaled_features = self.scaler.transform(features_df)
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any(): logger.warning(f"NaNs/Infs AFTER scaling MentS ({side_label}). Using 0."); scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            prediction = self.model.predict(scaled_features)[0]; probabilities = self.model.predict_proba(scaled_features)[0]
            pred_idx = int(prediction); confidence = 0.0
            if 0 <= pred_idx < len(probabilities): confidence = probabilities[pred_idx]
            else: logger.warning(f"MentS ({side_label}): Pred idx {pred_idx} out of bounds for probs (len {len(probabilities)}).")
            is_detected = bool(prediction == 1) # Class 1 is 'Mentalis Synkinesis'
            logger.debug(f"ML Mentalis Detect {side_label}: Pred={prediction} ({CLASS_NAMES.get(prediction,'?')}), Conf={confidence:.3f}")
            return is_detected, confidence

        except Exception as e: logger.error(f"Error in Mentalis detect {side_label}: {e}", exc_info=True); return False, 0.0