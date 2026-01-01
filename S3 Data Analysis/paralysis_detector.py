# paralysis_detector.py (Generic Detector)

import numpy as np
import pandas as pd
import joblib
import logging
import os
import importlib # For dynamic imports
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific scikit-learn warnings about feature names during transform
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Import central config
try:
    from paralysis_config import ZONE_CONFIG, CLASS_NAMES
except ImportError:
    logging.critical("CRITICAL: Could not import ZONE_CONFIG from paralysis_config. Detector cannot function.")
    # Provide minimal fallback for CLASS_NAMES if needed, but ZONE_CONFIG is essential
    CLASS_NAMES = {0: 'Normal', 1: 'Partial', 2: 'Complete'}
    ZONE_CONFIG = {} # Detector will fail gracefully if config is missing

# Configure logging
logger = logging.getLogger(__name__) # Assuming configured by calling script/main app

class ParalysisDetector:
    """ Generic detector for facial paralysis in a specific zone. """
    def __init__(self, zone):
        """
        Initializes the detector for a specific facial zone.

        Args:
            zone (str): The facial zone ('lower', 'mid', or 'upper').
        """
        if zone not in ZONE_CONFIG:
            raise ValueError(f"Invalid zone '{zone}'. Available zones: {list(ZONE_CONFIG.keys())}")

        self.zone = zone
        self.config = ZONE_CONFIG[zone]
        self.zone_name = self.config.get('name', zone.capitalize() + ' Face') # e.g., 'Lower Face'
        logger.info(f"Initializing ParalysisDetector for zone: {self.zone_name}")

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.extract_features_func = None
        self._load_artifacts()
        self._load_feature_extractor()

    def _load_artifacts(self):
        """Load the ML model, scaler, and feature names list for the zone."""
        try:
            filenames = self.config.get('filenames', {})
            model_path = filenames.get('model')
            scaler_path = filenames.get('scaler')
            features_path = filenames.get('feature_list')

            if not all([model_path, scaler_path, features_path]):
                logger.error(f"[{self.zone_name}] Missing model/scaler/features path in config.")
                return

            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path])
            if paths_exist:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)

                # Load feature names - handle both .pkl and .list text files
                if features_path.endswith('.list'):
                    with open(features_path, 'r') as f:
                        self.feature_names = [line.strip() for line in f if line.strip()]
                else:
                    self.feature_names = joblib.load(features_path)

                logger.info(f"[{self.zone_name}] Artifacts loaded successfully.")

                if not isinstance(self.feature_names, list):
                    logger.error(f"[{self.zone_name}] Loaded feature names is not a list ({type(self.feature_names)}). Resetting artifacts.")
                    self.model, self.scaler, self.feature_names = None, None, None
                    return

                # Sanity checks
                expected_features = len(self.feature_names)
                scaler_features = getattr(self.scaler, 'n_features_in_', None)
                model_features = getattr(self.model, 'n_features_in_', None) # XGBoost might not have this directly after calibration
                logger.debug(f"[{self.zone_name}] Expected features: {expected_features}")

                if scaler_features is not None and expected_features != scaler_features:
                    logger.warning(f"[{self.zone_name}] Feature list length ({expected_features}) doesn't match scaler features ({scaler_features}).")
                # Model feature check might be less reliable depending on model type/calibration
                if model_features is not None and expected_features != model_features:
                     logger.warning(f"[{self.zone_name}] Feature list length ({expected_features}) doesn't match model features ({model_features}).")

            else:
                missing = [p for p in [model_path, scaler_path, features_path] if not os.path.exists(p)]
                logger.error(f"[{self.zone_name}] Artifacts missing: {missing}. Cannot perform detection.")
                self.model, self.scaler, self.feature_names = None, None, None

        except Exception as e:
            logger.error(f"[{self.zone_name}] Error loading artifacts: {e}", exc_info=True)
            self.model, self.scaler, self.feature_names = None, None, None

    def _load_feature_extractor(self):
        """Dynamically loads the feature extraction function for the zone."""
        try:
            module_name = f"{self.zone}_face_features"
            feature_module = importlib.import_module(module_name)
            self.extract_features_func = getattr(feature_module, 'extract_features_for_detection')
            logger.info(f"[{self.zone_name}] Feature extraction function loaded from {module_name}.")
        except ModuleNotFoundError:
            logger.error(f"[{self.zone_name}] Feature extraction module '{module_name}.py' not found.")
            self.extract_features_func = None
        except AttributeError:
            logger.error(f"[{self.zone_name}] Function 'extract_features_for_detection' not found in module '{module_name}'.")
            self.extract_features_func = None
        except Exception as e:
            logger.error(f"[{self.zone_name}] Error loading feature extraction function: {e}", exc_info=True)
            self.extract_features_func = None

    def detect(self, row_data, side):
        """
        Detects paralysis for the configured zone and specified side.

        Args:
            row_data (dict or pd.Series): Input data for a single patient/frame.
            side (str): 'left' or 'right'.

        Returns:
            tuple: (result_str, confidence_float, details_dict)
                   result_str can be 'None', 'Partial', 'Complete', 'Error'.
                   details_dict contains raw prediction, probabilities, etc.
        """
        if not all([self.model, self.scaler, self.feature_names]):
            logger.warning(f"[{self.zone_name}] Components unavailable for {side} {self.zone}. Cannot detect.")
            return 'Error', 0.0, {'error': 'Model/Scaler/Features missing'}
        if self.extract_features_func is None:
            logger.error(f"[{self.zone_name}] Feature extraction unavailable. Cannot detect.")
            return 'Error', 0.0, {'error': 'Feature extraction function missing'}

        try:
            # Call the dynamically loaded feature extraction function
            features_list = self.extract_features_func(row_data, side, self.zone)

            if features_list is None:
                logger.error(f"[{self.zone_name}] Feature extraction failed for {side} {self.zone}.")
                return 'Error', 0.0, {'error': 'Feature extraction failed'}
            if len(features_list) != len(self.feature_names):
                logger.error(f"[{self.zone_name}] Feature mismatch for {side} {self.zone}. "
                             f"Expected {len(self.feature_names)}, Got {len(features_list)}")
                return 'Error', 0.0, {'error': 'Feature count mismatch'}

            # Create DataFrame with correct feature names for scaling
            features_df = pd.DataFrame([features_list], columns=self.feature_names)

            # Scale features
            scaled_features = self.scaler.transform(features_df) # Pass DataFrame

            # Predict
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            result = CLASS_NAMES.get(prediction, 'Unknown')
            confidence = 0.0
            if 0 <= prediction < len(probabilities):
                 confidence = probabilities[prediction]
            else:
                 logger.warning(f"[{self.zone_name}] Prediction index {prediction} out of bounds for probabilities length {len(probabilities)}.")


            details = {
                'raw_prediction': int(prediction),
                'probabilities': probabilities.tolist(),
                'model_used': 'base_model', # Or potentially add specific model info later
                'zone': self.zone
            }
            logger.debug(f"[{self.zone_name}] Detect - {side}: Result={result}, Conf={confidence:.3f}")
            return result, confidence, details

        except Exception as e:
            logger.error(f"[{self.zone_name}] Error during detection for {side} {self.zone}: {e}", exc_info=True)
            return 'Error', 0.0, {'error': f'Detection exception: {e}'}

    def detect_paralysis(self, self_orig, info, side, row_data=None, zone_paralysis_summary=None, **kwargs):
        """
        Interface method called by the main analysis loop (like in facial_paralysis_detection.py).
        Updates the shared summary and action-specific info dictionaries.

        Args:
            self_orig: The original 'self' from the calling analyzer (unused here).
            info (dict): Action-specific results dictionary to update.
            side (str): 'left' or 'right'.
            row_data (dict or pd.Series): Full data row for the patient. REQUIRED.
            zone_paralysis_summary (dict): Patient-level summary dict to update.
            **kwargs: Catches unused arguments like aus, values, etc.

        Returns:
            bool: True if paralysis ('Partial' or 'Complete') was detected, False otherwise.
        """
        if row_data is None:
            logger.error(f"[{self.zone_name}] detect_paralysis called without row_data for side {side}.")
            # Update summaries with Error state
            if zone_paralysis_summary and side in zone_paralysis_summary and self.zone in zone_paralysis_summary[side]:
                zone_paralysis_summary[side][self.zone] = 'Error'
            self._update_action_info(info, side, 'Error', 0.0, {'error': 'Missing row_data'})
            return False

        if zone_paralysis_summary is None:
             logger.error(f"[{self.zone_name}] detect_paralysis called without zone_paralysis_summary for side {side}. Cannot update.")
             # We can still attempt detection and update 'info' but the main summary won't be updated.
             # Or we can return False immediately. Let's return False for safer operation.
             self._update_action_info(info, side, 'Error', 0.0, {'error': 'Missing zone_paralysis_summary'})
             return False


        try:
            result, confidence, details = self.detect(row_data, side)

            # Ensure structure exists
            if side not in zone_paralysis_summary: zone_paralysis_summary[side] = {}
            if self.zone not in zone_paralysis_summary[side]: zone_paralysis_summary[side][self.zone] = 'None' # Initialize if missing

            current_severity_str = zone_paralysis_summary[side][self.zone]
            level_map = {'Normal': 0, 'None': 0, 'Partial': 1, 'Complete': 2, 'Error': -1}
            current_level = level_map.get(current_severity_str, 0)
            result_level = level_map.get(result, -1)

            # Update only if the new result indicates higher severity
            if result_level > current_level:
                zone_paralysis_summary[side][self.zone] = result
                logger.debug(f"[{self.zone_name}] Updated zone_paralysis_summary[{side}][{self.zone}] to {result}")

            self._update_action_info(info, side, result, confidence, details)

            return result in ['Partial', 'Complete']

        except Exception as e:
            logger.error(f"[{self.zone_name}] Exception in detect_paralysis {side}: {e}", exc_info=True)
            if zone_paralysis_summary and side in zone_paralysis_summary and self.zone in zone_paralysis_summary[side]:
                 zone_paralysis_summary[side][self.zone] = 'Error' # Update summary
            self._update_action_info(info, side, 'Error', 0.0, {'error': f'detect_paralysis exception: {e}'})
            return False

    def _update_action_info(self, info, side, result, confidence, details):
        """ Safely updates the action-specific info dictionary. """
        try:
            if 'paralysis' not in info:
                info['paralysis'] = {'zones': {'left': {}, 'right': {}}, 'detection_details': {}, 'confidence': {'left': {}, 'right': {}}}
            if 'zones' not in info['paralysis']: info['paralysis']['zones'] = {'left': {}, 'right': {}}
            if 'detection_details' not in info['paralysis']: info['paralysis']['detection_details'] = {}
            if 'confidence' not in info['paralysis']: info['paralysis']['confidence'] = {'left': {}, 'right': {}}

            if side not in info['paralysis']['zones']: info['paralysis']['zones'][side] = {}
            if side not in info['paralysis']['confidence']: info['paralysis']['confidence'][side] = {}

            info['paralysis']['zones'][side][self.zone] = result
            info['paralysis']['confidence'][side][self.zone] = confidence
            info['paralysis']['detection_details'][f"{side}_{self.zone}"] = details
        except Exception as e:
            logger.error(f"[{self.zone_name}] Failed to update action info dictionary: {e}", exc_info=True)