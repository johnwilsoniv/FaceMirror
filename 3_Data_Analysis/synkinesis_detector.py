# synkinesis_detector.py (v2 - Added Interface Method, Enhanced Detect Return)

import numpy as np
import pandas as pd
import joblib
import logging
import os
import importlib
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific scikit-learn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Import central config
try:
    from synkinesis_config import SYNKINESIS_CONFIG, CLASS_NAMES
except ImportError:
    logging.critical("CRITICAL: Could not import SYNKINESIS_CONFIG. Synkinesis detector cannot function.")
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'} # Minimal fallback
    SYNKINESIS_CONFIG = {} # Detector will fail gracefully

logger = logging.getLogger(__name__) # Assume configured by calling script

class SynkinesisDetector:
    """ Generic detector for binary synkinesis types using ML models. """
    def __init__(self, synkinesis_type):
        """
        Initializes the detector for a specific synkinesis type.

        Args:
            synkinesis_type (str): The type (e.g., 'ocular_oral', 'mentalis').
                                   Must match a key in SYNKINESIS_CONFIG.
        """
        if synkinesis_type not in SYNKINESIS_CONFIG:
            raise ValueError(f"Invalid synkinesis_type '{synkinesis_type}'. Available types: {list(SYNKINESIS_CONFIG.keys())}")

        self.type = synkinesis_type
        self.config = SYNKINESIS_CONFIG[self.type]
        self.name = self.config.get('name', self.type.replace('_', ' ').title())
        logger.info(f"Initializing SynkinesisDetector for type: {self.name}")

        self.model = None; self.scaler = None; self.feature_names = None
        self.extract_features_func = None
        self.threshold = self.config.get('DETECTION_THRESHOLD', 0.5) # Load threshold
        logger.info(f"[{self.name}] Using Detection Threshold: {self.threshold}")

        self._load_artifacts()
        self._load_feature_extractor()

    def _load_artifacts(self):
        """Load the ML model, scaler, and feature names list for the type."""
        try:
            filenames = self.config.get('filenames', {})
            model_path = filenames.get('model'); scaler_path = filenames.get('scaler'); features_path = filenames.get('feature_list')
            if not all([model_path, scaler_path, features_path]): logger.error(f"[{self.name}] Missing model/scaler/features path in config."); return

            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path] if p)
            if paths_exist:
                self.model = joblib.load(model_path); self.scaler = joblib.load(scaler_path); self.feature_names = joblib.load(features_path)
                logger.info(f"[{self.name}] Artifacts loaded successfully.")
                if not isinstance(self.feature_names, list): raise TypeError("Loaded feature names is not a list.")
                # Sanity checks
                expected = len(self.feature_names); scaler_n = getattr(self.scaler, 'n_features_in_', None)
                model_n = None
                if hasattr(self.model, 'n_features_in_'): model_n = self.model.n_features_in_
                elif hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'n_features_in_'): model_n = self.model.estimator.n_features_in_ # Calibrated
                logger.debug(f"[{self.name}] Expected features: {expected}")
                if scaler_n is not None and expected != scaler_n: logger.warning(f"[{self.name}] Scaler feature count mismatch: Expected {expected}, Scaler has {scaler_n}")
                if model_n is not None and expected != model_n: logger.warning(f"[{self.name}] Model feature count mismatch: Expected {expected}, Model has {model_n}")
            else:
                missing = [p for p in [model_path, scaler_path, features_path] if p and not os.path.exists(p)]
                logger.error(f"[{self.name}] Artifacts missing: {missing}. Cannot perform detection.")
                self._reset_artifacts()
        except Exception as e:
            logger.error(f"[{self.name}] Error loading artifacts: {e}", exc_info=True)
            self._reset_artifacts()

    def _reset_artifacts(self):
        """ Resets loaded artifacts to None. """
        self.model, self.scaler, self.feature_names = None, None, None

    def _load_feature_extractor(self):
        """Dynamically loads the feature extraction function for the type."""
        try:
            module_name = f"{self.type}_features"
            feature_module = importlib.import_module(module_name)
            self.extract_features_func = getattr(feature_module, 'extract_features_for_detection')
            logger.info(f"[{self.name}] Feature extraction function loaded from {module_name}.")
        except ModuleNotFoundError: logger.error(f"[{self.name}] Feature extraction module '{module_name}.py' not found."); self.extract_features_func = None
        except AttributeError: logger.error(f"[{self.name}] Func 'extract_features_for_detection' not found in '{module_name}'."); self.extract_features_func = None
        except Exception as e: logger.error(f"[{self.name}] Error loading feature extraction function: {e}", exc_info=True); self.extract_features_func = None

    def detect(self, row_data, side):
        """
        Detects synkinesis for the configured type and specified side.

        Args:
            row_data (dict or pd.Series): Input data for a single patient/frame.
            side (str): 'Left' or 'Right'.

        Returns:
            tuple: (is_detected (bool), confidence (float), positive_proba (float), details (dict))
                   Returns (False, 0.0, 0.0, {'error': ...}) on error.
        """
        error_details = {'error': 'Detection not performed'} # Default error state
        if not all([self.model, self.scaler, self.feature_names]):
            error_details['error'] = 'Model/Scaler/Features missing'
            logger.warning(f"[{self.name}] Components unavailable for {side} {self.type}. Cannot detect.")
            return False, 0.0, 0.0, error_details
        if self.extract_features_func is None:
            error_details['error'] = 'Feature extraction function missing'
            logger.error(f"[{self.name}] Feature extraction unavailable. Cannot detect.")
            return False, 0.0, 0.0, error_details
        if side not in ['Left', 'Right']:
            error_details['error'] = f"Invalid side '{side}'. Must be 'Left' or 'Right'."
            logger.error(f"[{self.name}] {error_details['error']}")
            return False, 0.0, 0.0, error_details

        try:
            features_list = self.extract_features_func(row_data, side)
            if features_list is None:
                error_details['error'] = 'Feature extraction failed'
                logger.error(f"[{self.name}] Feature extraction failed for {side}.");
                return False, 0.0, 0.0, error_details
            if len(features_list) != len(self.feature_names):
                error_details['error'] = f"Feature count mismatch (Expected {len(self.feature_names)}, Got {len(features_list)})"
                logger.error(f"[{self.name}] {error_details['error']} for {side}.")
                return False, 0.0, 0.0, error_details

            features_df = pd.DataFrame([features_list], columns=self.feature_names)
            if features_df.isnull().values.any() or np.isinf(features_df.values).any():
                logger.warning(f"[{self.name}] NaNs/Infs found before scaling for {side}. Filling with 0.")
                features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)

            scaled_features = self.scaler.transform(features_df)
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                logger.warning(f"[{self.name}] NaNs/Infs found after scaling for {side}. Using 0 for prediction.")
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            probabilities = self.model.predict_proba(scaled_features)[0]
            if len(probabilities) < 2:
                error_details['error'] = 'Invalid probability array shape'
                logger.error(f"[{self.name}] Probabilities array shape error for {side}. Got shape {probabilities.shape}");
                return False, 0.0, 0.0, error_details

            positive_proba = probabilities[1] # Probability of the positive class (Synkinesis)
            is_detected = bool(positive_proba >= self.threshold)
            predicted_class_index = 1 if is_detected else 0
            confidence = probabilities[predicted_class_index] # Confidence is prob of predicted class

            details = {
                'raw_probabilities': probabilities.tolist(),
                'threshold_used': self.threshold,
                'error': None # No error
            }
            logger.debug(f"[{self.name}] Detect - {side}: Prob(+)={positive_proba:.3f}, Threshold={self.threshold}, Detected={is_detected}, Conf={confidence:.3f}")
            return is_detected, confidence, positive_proba, details

        except Exception as e:
            error_details['error'] = f'Detection exception: {e}'
            logger.error(f"[{self.name}] Error during detection for {side}: {e}", exc_info=True)
            return False, 0.0, 0.0, error_details


    def detect_synkinesis(self, self_orig, info, side, row_data=None, synkinesis_summary=None, **kwargs):
        """
        Interface method called by the main analysis loop.
        Updates the shared summary and action-specific info dictionaries.

        Args:
            self_orig: The original 'self' from the calling analyzer (unused here).
            info (dict): Action-specific results dictionary to update.
            side (str): 'Left' or 'Right'.
            row_data (dict or pd.Series): Full data row for the patient. REQUIRED.
            synkinesis_summary (dict): Patient-level summary dict to update. {pid: {'left': {}, 'right':{}}}
            **kwargs: Catches unused arguments.

        Returns:
            bool: True if synkinesis was detected, False otherwise.
        """
        if row_data is None:
            logger.error(f"[{self.name}] detect_synkinesis called without row_data for side {side}.")
            self._update_action_info(info, side, False, 0.0, 0.0, {'error': 'Missing row_data'})
            # Update patient summary if possible
            if synkinesis_summary:
                 # Assuming synkinesis_summary is structured like {patient_id: {'left': {}, 'right': {}}}
                 patient_id = row_data.get('Patient ID', 'Unknown') if isinstance(row_data, (dict, pd.Series)) else 'Unknown'
                 if patient_id != 'Unknown':
                    if patient_id not in synkinesis_summary: synkinesis_summary[patient_id] = {'left': {}, 'right': {}}
                    if side.lower() not in synkinesis_summary[patient_id]: synkinesis_summary[patient_id][side.lower()] = {}
                    synkinesis_summary[patient_id][side.lower()][self.type] = 'Error' # Mark error in summary
            return False

        if synkinesis_summary is None:
            logger.warning(f"[{self.name}] detect_synkinesis called without synkinesis_summary for side {side}. Cannot update patient summary.")
            # Continue detection but summary won't be updated

        try:
            is_detected, confidence, positive_proba, details = self.detect(row_data, side)
            detected_label = 1 if is_detected else 0

            # --- Update the main patient summary dictionary ---
            if synkinesis_summary:
                patient_id = row_data.get('Patient ID', 'Unknown')
                if patient_id != 'Unknown':
                    if patient_id not in synkinesis_summary: synkinesis_summary[patient_id] = {'left': {}, 'right': {}}
                    side_lower = side.lower()
                    if side_lower not in synkinesis_summary[patient_id]: synkinesis_summary[patient_id][side_lower] = {}

                    # Update summary only if detection occurred (don't overwrite previous detection)
                    # Or if an error occurred previously
                    current_summary_val = synkinesis_summary[patient_id][side_lower].get(self.type)
                    if is_detected or current_summary_val == 'Error':
                         synkinesis_summary[patient_id][side_lower][self.type] = CLASS_NAMES.get(detected_label, 'Error')
                         logger.debug(f"[{self.name}] Updated synkinesis_summary[{patient_id}][{side_lower}][{self.type}] to {CLASS_NAMES.get(detected_label, 'Error')}")
                    elif current_summary_val is None: # First time seeing this type for this patient/side
                         synkinesis_summary[patient_id][side_lower][self.type] = CLASS_NAMES.get(detected_label, 'Error')
                         logger.debug(f"[{self.name}] Initialized synkinesis_summary[{patient_id}][{side_lower}][{self.type}] to {CLASS_NAMES.get(detected_label, 'Error')}")


            # --- Update action-specific info dict ---
            self._update_action_info(info, side, is_detected, confidence, positive_proba, details)

            return is_detected

        except Exception as e:
            logger.error(f"[{self.name}] Exception in detect_synkinesis {side}: {e}", exc_info=True)
            self._update_action_info(info, side, False, 0.0, 0.0, {'error': f'detect_synkinesis exception: {e}'})
            # Update patient summary with error if possible
            if synkinesis_summary:
                 patient_id = row_data.get('Patient ID', 'Unknown') if isinstance(row_data, (dict, pd.Series)) else 'Unknown'
                 if patient_id != 'Unknown':
                    if patient_id not in synkinesis_summary: synkinesis_summary[patient_id] = {'left': {}, 'right': {}}
                    if side.lower() not in synkinesis_summary[patient_id]: synkinesis_summary[patient_id][side.lower()] = {}
                    synkinesis_summary[patient_id][side.lower()][self.type] = 'Error' # Mark error in summary
            return False


    def _update_action_info(self, info, side, is_detected, confidence, positive_proba, details):
        """ Safely updates the action-specific info dictionary. """
        try:
            if 'synkinesis' not in info: info['synkinesis'] = {}
            side_lower = side.lower()
            if side_lower not in info['synkinesis']: info['synkinesis'][side_lower] = {}

            # Store results per synkinesis type
            info['synkinesis'][side_lower][self.type] = {
                'detected': is_detected,
                'confidence': confidence,
                'positive_probability': positive_proba,
                'details': details
            }
        except Exception as e:
            logger.error(f"[{self.name}] Failed to update action info dictionary: {e}", exc_info=True)