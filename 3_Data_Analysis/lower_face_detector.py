# lower_face_detector.py (Refactored with DataFrame Scaling)

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific scikit-learn warnings about feature names during transform
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Import config carefully, providing fallbacks
try:
    from lower_face_config import (
        MODEL_FILENAMES, CLASS_NAMES, LOG_DIR,
        LOGGING_CONFIG, FEATURE_CONFIG, MODEL_DIR
    )
except ImportError:
    logging.warning("Could not import from lower_face_config. Using fallback definitions.")
    LOG_DIR = 'logs'; MODEL_DIR = 'models'
    MODEL_FILENAMES = {'base_model': os.path.join(MODEL_DIR, 'lower_face_model.pkl'),
                       'base_scaler': os.path.join(MODEL_DIR, 'lower_face_scaler.pkl'),
                       'feature_list': os.path.join(MODEL_DIR, 'lower_face_features.list')} # Add feature list path
    CLASS_NAMES = {0: 'None', 1: 'Partial', 2: 'Complete'}
    LOGGING_CONFIG = {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'}

# Use the extraction function from the features file
try:
    feature_list_path_check = MODEL_FILENAMES.get('feature_list', os.path.join(MODEL_DIR, 'lower_face_features.list'))
    if not os.path.exists(feature_list_path_check):
         logging.warning(f"Feature list {feature_list_path_check} not found during import.")
    from lower_face_features import extract_features_for_detection
except ImportError:
    logging.error("Could not import extract_features_for_detection from lower_face_features.py.")
    extract_features_for_detection = None

# Configure logging
# (Assuming configured elsewhere or use basicConfig if run standalone)
logger = logging.getLogger(__name__)

class LowerFaceParalysisDetector:
    """ Detects lower face paralysis using ML model. """
    def __init__(self):
        logger.info("Initializing LowerFaceParalysisDetector")
        self.base_model = None
        self.base_scaler = None
        self.feature_names = None # Expect a list of feature names
        self._load_models()

    def _load_models(self):
        """Load the base model, scaler, and feature names."""
        # ... (Identical loading logic as previous version) ...
        try:
            model_path = MODEL_FILENAMES.get('base_model')
            scaler_path = MODEL_FILENAMES.get('base_scaler')
            features_path = MODEL_FILENAMES.get('feature_list')
            if not all([model_path, scaler_path, features_path]): logger.error("Missing config paths."); return
            paths_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path])
            if paths_exist:
                self.base_model = joblib.load(model_path); self.base_scaler = joblib.load(scaler_path); self.feature_names = joblib.load(features_path)
                logger.info("Lower face artifacts loaded."); assert isinstance(self.feature_names, list), "Feature names not a list"
                expected = len(self.feature_names); scaler_n = getattr(self.base_scaler, 'n_features_in_', None); model_n = getattr(self.base_model, 'n_features_in_', None)
                logger.debug(f"Loaded {expected} features.")
                if scaler_n is not None and expected != scaler_n: logger.warning(f"Scaler mismatch: {expected} vs {scaler_n}")
                if model_n is not None and expected != model_n: logger.warning(f"Model mismatch: {expected} vs {model_n}")
            else: logger.error("Lower face artifacts missing."); self.base_model, self.base_scaler, self.feature_names = None, None, None
        except Exception as e: logger.error(f"Load error: {e}", exc_info=True); self.base_model, self.base_scaler, self.feature_names = None, None, None


    def detect(self, row_data, side, zone):
        """ Detect using base model prediction. Accepts full row data. """
        if not all([self.base_model, self.base_scaler, self.feature_names]):
            logger.warning(f"Lower Face components unavailable for {side} {zone}.")
            return 'Error', 0.0, {'error': 'Model/Scaler/Features missing'}
        if extract_features_for_detection is None:
             logger.error("Lower face feature extraction unavailable."); return 'Error', 0.0, {'error': 'Feature func missing'}

        try:
            features_list = extract_features_for_detection(row_data, side, zone)
            if features_list is None: logger.error(f"L-Feat extract fail {side} {zone}"); return 'Error', 0.0, {'error': 'Feat extract failed'}
            if len(features_list) != len(self.feature_names): logger.error(f"L-Feat mismatch {side} {zone}"); return 'Error', 0.0, {'error': 'Feat mismatch'}

            # --- Create DataFrame with feature names for scaling ---
            features_df = pd.DataFrame([features_list], columns=self.feature_names)
            scaled_features = self.base_scaler.transform(features_df) # Pass DataFrame
            # --- End DataFrame scaling ---

            prediction = self.base_model.predict(scaled_features)[0]
            probabilities = self.base_model.predict_proba(scaled_features)[0]
            result = CLASS_NAMES.get(prediction, 'Unknown')
            confidence = probabilities[prediction] if prediction < len(probabilities) else 0.0
            details = { 'raw_prediction': int(prediction), 'probabilities': probabilities.tolist(), 'model_used': 'base_model' }
            logger.debug(f"Lower Face Detect - {side} {zone}: Result={result}, Conf={confidence:.3f}")
            return result, confidence, details

        except Exception as e:
            logger.error(f"Error in lower face detect {side} {zone}: {e}", exc_info=True)
            return 'Error', 0.0, {'error': f'Detect exception: {e}'}

    # --- (Inside detect_paralysis method) ---
    def detect_paralysis(self, self_orig, info, zone, side, aus, values, other_values,
                         values_normalized, other_values_normalized,
                         zone_paralysis, affected_aus_by_zone_side, row_data=None, **kwargs):
        if row_data is None:
            # ... (error handling for missing row_data) ...
            logger.error(f"{zone}-detect_paralysis no row_data {side}");
            zone_paralysis[side][zone] = 'Error';
            return False

        try:
            result, confidence, details = self.detect(row_data, side, zone)

            # --- Update the main summary dictionary (zone_paralysis) ---
            current_severity_str = zone_paralysis[side][zone];
            level_map = {'None': 0, 'Partial': 1, 'Complete': 2, 'Error': -1}
            current_level = level_map.get(current_severity_str, 0);
            result_level = level_map.get(result, -1)
            if result_level > current_level: zone_paralysis[side][zone] = result

            # --- Robustly Update action-specific info dict ---
            if 'paralysis' not in info:
                # Initialize the entire expected paralysis structure if it's missing
                info['paralysis'] = {
                    'zones': {'left': {}, 'right': {}},
                    'detection_details': {},
                    'confidence': {'left': {}, 'right': {}}  # Initialize confidence dict here
                }
            # Ensure nested dictionaries exist before access
            if 'zones' not in info['paralysis']: info['paralysis']['zones'] = {'left': {}, 'right': {}}
            if 'detection_details' not in info['paralysis']: info['paralysis']['detection_details'] = {}
            if 'confidence' not in info['paralysis']: info['paralysis']['confidence'] = {'left': {},
                                                                                         'right': {}}  # Ensure again just in case

            if side not in info['paralysis']['zones']: info['paralysis']['zones'][side] = {}
            # --- CORRECTED Confidence Handling ---
            if side not in info['paralysis']['confidence']:
                info['paralysis']['confidence'][side] = {}  # Ensure the side key exists
            # Now it's safe to access and update the zone confidence
            info['paralysis']['confidence'][side][zone] = confidence
            # --- END CORRECTION ---

            # Store zone result and details
            info['paralysis']['zones'][side][zone] = result
            info['paralysis']['detection_details'][f"{side}_{zone}"] = details
            # --- End Update Action Info ---

            return result in ['Partial', 'Complete']

        except Exception as e:
            logger.error(f"Exc in {zone}-detect_paralysis {side}: {e}", exc_info=True)
            zone_paralysis[side][zone] = 'Error'  # Update summary
            # Ensure structure exists before setting error in info
            if 'paralysis' not in info: info['paralysis'] = {'zones': {'left': {}, 'right': {}},
                                                             'confidence': {'left': {}, 'right': {}}}
            if 'zones' not in info['paralysis']: info['paralysis']['zones'] = {'left': {}, 'right': {}}
            if side not in info['paralysis']['zones']: info['paralysis']['zones'][side] = {}
            info['paralysis']['zones'][side][zone] = 'Error'
            # Also ensure confidence structure exists for error case if needed elsewhere
            if 'confidence' not in info['paralysis']: info['paralysis']['confidence'] = {'left': {}, 'right': {}}
            if side not in info['paralysis']['confidence']: info['paralysis']['confidence'][side] = {}
            info['paralysis']['confidence'][side][zone] = 0.0  # Set confidence to 0 on error
            return False