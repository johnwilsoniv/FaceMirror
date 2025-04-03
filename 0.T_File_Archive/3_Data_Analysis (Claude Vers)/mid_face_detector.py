"""
Unified detector for mid face paralysis.
Combines base model and specialized detection in a single class.
Similar approach to lower face detector with minimal rule-based interventions.
Enhanced with multi-factor verification.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
from mid_face_config import (
    MODEL_FILENAMES, DETECTION_THRESHOLDS, CLASS_NAMES, LOG_DIR, LOGGING_CONFIG,
    FEATURE_CONFIG
)
from mid_face_features import extract_features_for_detection

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


class MidFaceParalysisDetector:
    """
    Unified detector for mid face paralysis.

    Primarily relies on ML model for detection with enhanced multi-factor verification.
    """

    def __init__(self):
        """Initialize the detector and load models."""
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(LOG_DIR, 'mid_face_detector.log'))
            ]
        )

        logger.info("Mid face paralysis detector initializing")

        # Initialize model variables
        self.base_model = None
        self.base_scaler = None
        self.specialist_model = None
        self.specialist_scaler = None

        # Load models
        self._load_models()

        self.thresholds = DETECTION_THRESHOLDS.copy()  # Start with defaults
        try:
            # Load optimal thresholds if available
            config_path = os.path.join(LOG_DIR, 'mid_face_optimal_thresholds.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    optimal_thresholds = json.load(f)
                    self.thresholds.update(optimal_thresholds)
                    logger.info("Loaded optimized thresholds")
                    logger.debug(f"Using thresholds: {self.thresholds}")
            else:
                logger.info("Using default thresholds from configuration")
        except Exception as e:
            logger.warning(f"Error loading optimal thresholds: {str(e)}")

    def _load_models(self):
        """Load the base and specialist models."""
        try:
            # Load base model
            if os.path.exists(MODEL_FILENAMES['base_model']):
                self.base_model = joblib.load(MODEL_FILENAMES['base_model'])
                self.base_scaler = joblib.load(MODEL_FILENAMES['base_scaler'])
                logger.info("Base model loaded successfully")
            else:
                logger.warning(f"Base model not found at {MODEL_FILENAMES['base_model']}")

            # Load specialist model if available
            if os.path.exists(MODEL_FILENAMES['specialist_model']):
                self.specialist_model = joblib.load(MODEL_FILENAMES['specialist_model'])
                self.specialist_scaler = joblib.load(MODEL_FILENAMES['specialist_scaler'])
                logger.info("Specialist model loaded successfully")
            else:
                logger.warning("Specialist model not found - will use base model only")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.base_model = None

    def detect(self, info, side, zone, aus, values, other_values,
               values_normalized, other_values_normalized):
        """
        Detect mid face paralysis using ML-based approach with enhanced multi-factor verification.

        Args:
            info (dict): Results dictionary
            side (str): Side being analyzed ('left' or 'right')
            zone (str): Zone being analyzed ('mid')
            aus (list): Action Units for this zone
            values (dict): AU values for this side
            other_values (dict): AU values for opposite side
            values_normalized (dict): Normalized AU values for this side
            other_values_normalized (dict): Normalized AU values for opposite side

        Returns:
            tuple: (result, confidence, details)
        """
        # Check if model is available
        if self.base_model is None:
            logger.warning("Model not available - attempting to load")
            self._load_models()
            if self.base_model is None:
                return 'None', 0.0, {'error': 'Model not available'}

        try:
            # Extract features
            features = extract_features_for_detection(
                info, side, zone, aus, values, other_values,
                values_normalized, other_values_normalized
            )

            # Convert to numpy array
            features_np = np.array(features).reshape(1, -1)

            # Scale features
            expected_features = getattr(self.base_model, 'n_features_in_', len(features_np[0]))

            # Adjust feature vector size if needed
            if features_np.shape[1] < expected_features:
                padded = np.zeros((1, expected_features))
                padded[0, :features_np.shape[1]] = features_np
                features_np = padded
            elif features_np.shape[1] > expected_features:
                features_np = features_np[:, :expected_features]

            # Scale features
            scaled_features = self.base_scaler.transform(features_np)

            # Get base model prediction
            base_prediction = self.base_model.predict(scaled_features)[0]
            base_proba = self.base_model.predict_proba(scaled_features)[0]

            # Apply thresholds
            adjusted_prediction = self._apply_thresholds(base_prediction, base_proba)

            # Extract AU45 values and ES/ET ratio for verification
            au45_et_value = values.get('AU45_r', 0)
            au07_et_value = values.get('AU07_r', 0)

            # Get ES value if available in info
            au45_es_value = 0
            if 'ES' in info:
                side_key = side.lower()
                if side_key in info['ES'] and 'au_values' in info['ES'][side_key]:
                    au45_es_value = info['ES'][side_key]['au_values'].get('AU45_r', 0)

            # Calculate ES/ET ratio (avoid division by zero)
            es_et_ratio = 1.0  # Default is 1:1 ratio (normal)
            if au45_et_value > FEATURE_CONFIG['min_value']:
                es_et_ratio = au45_es_value / au45_et_value

            # Calculate asymmetry for verification
            other_au45_et_value = other_values.get('AU45_r', 0)
            et_asymmetry_pct = 0
            if au45_et_value > 0 or other_au45_et_value > 0:
                et_diff = abs(au45_et_value - other_au45_et_value)
                et_avg = (au45_et_value + other_au45_et_value) / 2
                if et_avg > FEATURE_CONFIG['min_value']:
                    et_asymmetry_pct = (et_diff / et_avg) * 100

            # Apply enhanced multi-factor verification
            verified_prediction = self._enhanced_verification(
                adjusted_prediction, base_proba,
                au45_et_value, au45_es_value, es_et_ratio,
                au07_et_value, et_asymmetry_pct)

            # Use specialist classifier if appropriate (only for borderline cases)
            specialist_used = False
            final_prediction = verified_prediction

            if self.specialist_model is not None:
                # Determine if specialist should be used - only for borderline cases
                if (verified_prediction == 2 and  # Complete prediction
                        base_proba[2] >= self.thresholds['specialist_complete_lower'] and
                        base_proba[2] <= self.thresholds['specialist_complete_upper']):

                    # Use specialist for borderline Complete cases
                    specialist_used = True
                    specialist_prediction = self._get_specialist_prediction(
                        features_np, verified_prediction, base_proba
                    )

                    # Use specialist prediction if it's more conservative
                    # (prefer avoiding false positives)
                    if specialist_prediction < verified_prediction:
                        final_prediction = specialist_prediction
                elif (verified_prediction == 0 and  # None prediction
                      base_proba[1] >= self.thresholds['specialist_partial_threshold']):

                    # Use specialist for potential Partial cases
                    specialist_used = True
                    specialist_prediction = self._get_specialist_prediction(
                        features_np, verified_prediction, base_proba
                    )

                    # Only upgrade to Partial if both methods agree
                    if specialist_prediction > 0:
                        final_prediction = specialist_prediction

            # Map prediction to result
            result = CLASS_NAMES[final_prediction]

            # Calculate confidence
            confidence = base_proba[final_prediction]

            # Prepare details
            details = {
                'base_prediction': int(base_prediction),
                'adjusted_prediction': int(adjusted_prediction),
                'verified_prediction': int(verified_prediction),
                'final_prediction': int(final_prediction),
                'probabilities': base_proba.tolist(),
                'specialist_used': specialist_used,
                'au45_et_value': au45_et_value,
                'au45_es_value': au45_es_value,
                'es_et_ratio': es_et_ratio,
                'au07_et_value': au07_et_value,
                'et_asymmetry_pct': et_asymmetry_pct
            }

            logger.debug(f"{side} {zone}: {result} paralysis with {confidence:.3f} confidence")

            return result, confidence, details

        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            return 'None', 0.0, {'error': f'Detection failed: {str(e)}'}

    def _apply_thresholds(self, prediction, probabilities):
        """
        Apply thresholds to adjust prediction.
        Similar approach to lower face detector but with stricter thresholds.
        """
        adjusted = prediction

        # Complete prediction with low confidence
        if prediction == 2:  # Complete
            if probabilities[2] < self.thresholds['complete_confidence']:
                if probabilities[0] > self.thresholds['none_probability']:
                    # Significant chance of None
                    adjusted = 0
                elif probabilities[1] > self.thresholds['partial_probability']:
                    # Reasonable chance of Partial
                    adjusted = 1

        # None prediction with significant Partial probability
        elif prediction == 0 and probabilities[1] > self.thresholds['upgrade_to_partial']:
            adjusted = 1

        logger.debug(f"Original prediction: {prediction}, probabilities: {probabilities}")
        logger.debug(f"Adjusted prediction: {adjusted}")

        return adjusted

    def _enhanced_verification(self, prediction, probabilities,
                               au45_et_value, au45_es_value, es_et_ratio,
                               au07_et_value, et_asymmetry_pct):
        """
        Apply enhanced multi-factor verification to improve classification accuracy.

        Args:
            prediction (int): Current prediction (0=None, 1=Partial, 2=Complete)
            probabilities (list): Prediction probabilities
            au45_et_value (float): AU45 value from ET action
            au45_es_value (float): AU45 value from ES action
            es_et_ratio (float): Ratio of ES/ET AU45 values
            au07_et_value (float): AU07 value from ET action
            et_asymmetry_pct (float): Asymmetry percentage between sides

        Returns:
            int: Verified prediction (potentially adjusted based on verification)
        """
        # Get thresholds from config
        au45_thresholds = FEATURE_CONFIG['au45_thresholds']
        ratio_thresholds = FEATURE_CONFIG['es_et_ratio_thresholds']
        verification = FEATURE_CONFIG['verification']

        # Start with adjusted prediction
        verified = prediction

        # Check AU45_ET value against thresholds
        is_au45_complete = au45_et_value <= au45_thresholds['complete_max']
        is_au45_partial = au45_et_value <= au45_thresholds['partial_max']

        # Check ES/ET ratio against thresholds
        is_ratio_complete = es_et_ratio <= ratio_thresholds['complete_max']
        is_ratio_partial = es_et_ratio <= ratio_thresholds['partial_max']

        # Check additional verification factors
        is_au07_sufficient = au07_et_value >= verification['au07_min_partial']
        is_au07_low_enough = au07_et_value <= verification.get('au07_max_complete', float('inf'))
        is_asymmetry_partial = et_asymmetry_pct >= verification['asymmetry_min_partial']
        is_asymmetry_complete = et_asymmetry_pct >= verification['asymmetry_min_complete']

        # Check for borderline zone
        is_borderline = (au45_et_value >= au45_thresholds['borderline_lower'] and
                         au45_et_value <= au45_thresholds['borderline_upper'])

        # VERIFICATION RULES

        # Rule 1: Complete verification
        if verified == 2:  # Complete
            # If configured to require ALL conditions for Complete
            if verification.get('require_all_for_complete', False):
                # Check if ALL conditions are met
                if not (is_au45_complete and is_ratio_complete and is_asymmetry_complete and is_au07_low_enough):
                    # At least one condition failed
                    if is_au45_partial and is_ratio_partial and is_asymmetry_partial:
                        verified = 1  # Downgrade to Partial
                        logger.debug("Downgraded Complete → Partial: Not all Complete conditions met")
                    else:
                        verified = 0  # Downgrade to None
                        logger.debug("Downgraded Complete → None: Not all Complete conditions met")
            else:
                # Original logic - require either AU45 or ratio
                if not (is_au45_complete or is_ratio_complete):
                    # No evidence supports Complete
                    verified = 1  # Downgrade to Partial
                    logger.debug("Downgraded Complete → Partial: AU45_ET and ES/ET ratio too high")
                elif not is_asymmetry_partial:
                    # Insufficient asymmetry for any paralysis
                    verified = 0  # Downgrade to None
                    logger.debug(f"Downgraded Complete → None: Insufficient asymmetry ({et_asymmetry_pct:.1f}%)")

        # Rule 2: Partial verification
        elif verified == 1:  # Partial
            # For borderline Partial cases, apply stricter verification
            if is_borderline:
                # Require both AU07 and asymmetry for borderline cases
                if not (is_au07_sufficient and is_asymmetry_partial):
                    verified = 0  # Downgrade to None
                    logger.debug("Downgraded borderline Partial → None: Failed AU07/asymmetry check")
            else:
                # For non-borderline, verify Partial with AU45 and ratio
                if not (is_au45_partial or is_ratio_partial):
                    verified = 0  # Downgrade to None
                    logger.debug("Downgraded Partial → None: AU45_ET and ES/ET ratio too high")
                # Also require either sufficient AU07 or asymmetry
                elif not (is_au07_sufficient or is_asymmetry_partial):
                    verified = 0  # Downgrade to None
                    logger.debug("Downgraded Partial → None: Failed AU07/asymmetry check")

        # Rule 3: None → Partial/Complete upgrade only if strong evidence
        elif verified == 0:  # None
            # For Complete upgrade, require ALL conditions
            if (is_au45_complete and is_ratio_complete and is_asymmetry_complete and is_au07_low_enough):
                verified = 2  # Strong evidence for Complete
                logger.debug("Upgraded None → Complete: Strong evidence from all factors")
            # For Partial upgrade, slightly less strict
            elif (is_au45_partial and is_ratio_partial and is_asymmetry_partial and
                  is_au07_sufficient and not is_borderline):
                verified = 1  # Strong evidence for Partial
                logger.debug("Upgraded None → Partial: Strong evidence from all factors")

        return verified

    def _get_specialist_prediction(self, features_np, base_prediction, base_proba):
        """
        Get prediction from specialist classifier.

        Args:
            features_np (numpy.ndarray): Feature array
            base_prediction (int): Base model prediction
            base_proba (list): Base model probabilities

        Returns:
            int: Specialist prediction (0=None, 1=Partial, 2=Complete)
        """
        try:
            # Scale features for specialist
            scaled_features = self.specialist_scaler.transform(features_np)

            if base_prediction in [1, 2]:  # For Partial/Complete predictions
                # Specialist classifies between Partial (0) and Complete (1)
                spec_pred = self.specialist_model.predict(scaled_features)[0]
                spec_prob = self.specialist_model.predict_proba(scaled_features)[0]

                # Convert to full range (0=None, 1=Partial, 2=Complete)
                # Specialist only determines Partial vs Complete
                return spec_pred + 1
            else:
                # For None predictions with significant Partial probability
                spec_pred = self.specialist_model.predict(scaled_features)[0]
                spec_prob = self.specialist_model.predict_proba(scaled_features)[0]

                # If specialist says Partial with high confidence
                if spec_pred == 0 and spec_prob[0] > 0.65:  # Partial with high confidence
                    return 1  # Upgrade to Partial
                else:
                    return base_prediction  # Keep base prediction

        except Exception as e:
            logger.error(f"Error using specialist: {str(e)}")
            return base_prediction  # Fall back to base prediction

    def detect_paralysis(self, self_orig, info, zone, side, aus, values, other_values,
                         values_normalized, other_values_normalized,
                         zone_paralysis, affected_aus_by_zone_side, **kwargs):
        """
        Interface compatible with existing system.

        Args:
            self_orig: The original FacialParalysisDetector instance
            info (dict): Results dictionary for current action
            zone (str): Facial zone being analyzed ('mid')
            side (str): Side being analyzed ('left' or 'right')
            aus (list): List of Action Units for this zone
            values (dict): AU values for this side
            other_values (dict): AU values for other side
            values_normalized (dict): Normalized AU values for this side
            other_values_normalized (dict): Normalized AU values for other side
            zone_paralysis (dict): Track paralysis results at patient level
            affected_aus_by_zone_side (dict): Track affected AUs
            **kwargs: Additional arguments (not used)

        Returns:
            bool: True if paralysis was detected, False otherwise
        """
        try:
            # Use unified detection approach
            result, confidence, details = self.detect(
                info, side, zone, aus, values, other_values,
                values_normalized, other_values_normalized
            )

            # Update the info structure with detection result
            info['paralysis']['zones'][side][zone] = result

            # Track for patient-level assessment
            if result == 'Complete':
                zone_paralysis[side][zone] = 'Complete'
            elif result == 'Partial' and zone_paralysis[side][zone] == 'None':
                zone_paralysis[side][zone] = 'Partial'

            # Add affected AUs - only if paralysis detected
            if result != 'None':
                for au in ['AU45_r', 'AU07_r']:
                    if au in values:
                        affected_aus_by_zone_side[side][zone].add(au)
                        if au not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append(au)

            # Store confidence score
            info['paralysis']['confidence'][side][zone] = confidence

            # Add detection details
            if 'detection_details' not in info['paralysis']:
                info['paralysis']['detection_details'] = {}

            info['paralysis']['detection_details'][f"{side}_{zone}"] = details

            # Add contributing AUs info
            if result != 'None':
                if 'detection' not in info['paralysis']['contributing_aus'][side][zone]:
                    info['paralysis']['contributing_aus'][side][zone]['detection'] = []

                # Track AU values for detection
                for au in ['AU45_r', 'AU07_r']:
                    if au in values:
                        info['paralysis']['contributing_aus'][side][zone]['detection'].append({
                            'au': au,
                            'side_value': values.get(au, 0),
                            'other_value': other_values.get(au, 0),
                            'confidence': confidence,
                            'type': result,
                            'specialist_used': details.get('specialist_used', False)
                        })

            # Return success if paralysis was detected
            return result != 'None'

        except Exception as e:
            logger.error(f"Exception in detect_paralysis: {str(e)}")
            return False