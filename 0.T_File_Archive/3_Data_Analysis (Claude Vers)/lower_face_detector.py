"""
Unified detector for lower face paralysis.
Combines base model and specialized detection in a single class.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import StandardScaler
from lower_face_config import (
    MODEL_FILENAMES, DETECTION_THRESHOLDS, CLASS_NAMES,
    LOG_DIR, LOGGING_CONFIG
)
from lower_face_features import extract_features_for_detection

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


class LowerFaceParalysisDetector:
    """
    Unified detector for lower face paralysis.

    Integrates base model detection with specialist refinement
    without requiring separate integration steps.
    """

    def __init__(self):
        """Initialize the detector and load models."""
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(LOG_DIR, 'lower_face_detector.log'))
            ]
        )

        # Initialize model variables
        self.base_model = None
        self.base_scaler = None
        self.specialist_model = None
        self.specialist_scaler = None

        # Load models
        self._load_models()

        self.thresholds = DETECTION_THRESHOLDS.copy()  # Start with defaults
        try:
            # Updated path with consistent prefix
            config_path = os.path.join(LOG_DIR, 'lower_face_optimal_thresholds.json')
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
        Detect lower face paralysis using integrated approach.

        Args:
            info (dict): Results dictionary
            side (str): Side being analyzed ('left' or 'right')
            zone (str): Zone being analyzed ('lower')
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

            # Use specialist classifier if appropriate
            specialist_used = False
            if self.specialist_model is not None:
                # Determine if specialist should be used
                if (base_prediction == 2 and  # Complete prediction
                        base_proba[2] >= DETECTION_THRESHOLDS['specialist_complete_lower'] and
                        base_proba[2] <= DETECTION_THRESHOLDS['specialist_complete_upper']):

                    # Use specialist for borderline Complete cases
                    specialist_used = True
                    specialist_prediction = self._get_specialist_prediction(
                        features_np, base_prediction, base_proba
                    )

                    # Combine predictions
                    final_prediction = specialist_prediction
                elif (base_prediction == 0 and  # None prediction
                      base_proba[1] >= DETECTION_THRESHOLDS['specialist_partial_threshold']):

                    # Use specialist for potential Partial cases
                    specialist_used = True
                    specialist_prediction = self._get_specialist_prediction(
                        features_np, base_prediction, base_proba
                    )

                    # Combine predictions
                    final_prediction = specialist_prediction
                else:
                    final_prediction = adjusted_prediction
            else:
                final_prediction = adjusted_prediction

            # Map prediction to result
            result = CLASS_NAMES[final_prediction]

            # Calculate confidence
            confidence = base_proba[final_prediction]

            # Prepare details
            details = {
                'base_prediction': int(base_prediction),
                'adjusted_prediction': int(adjusted_prediction),
                'final_prediction': int(final_prediction),
                'probabilities': base_proba.tolist(),
                'specialist_used': specialist_used
            }

            logger.debug(f"{side} {zone}: {result} paralysis with {confidence:.3f} confidence")

            return result, confidence, details

        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            return 'None', 0.0, {'error': f'Detection failed: {str(e)}'}

    def _apply_thresholds(self, prediction, probabilities):
        """
        Apply thresholds to adjust prediction.
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
        elif prediction == 0 and probabilities[1] > self.thresholds.get('upgrade_to_partial',
                                                                        DETECTION_THRESHOLDS['upgrade_to_partial']):
            adjusted = 1

        logger.debug(f"Original prediction: {prediction}, probabilities: {probabilities}")
        logger.debug(f"Adjusted prediction: {adjusted}")

        return adjusted

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
            zone (str): Facial zone being analyzed ('lower')
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
                for au in ['AU12_r', 'AU25_r']:
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
                for au in ['AU12_r', 'AU25_r']:
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