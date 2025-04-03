""""
ML-based lower face paralysis detector.
Uses trained machine learning model to detect lower face paralysis.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import warnings

# Suppress specific scikit-learn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Configure logging to use logs subfolder
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

class LowerFaceParalysisDetector:
    """
    Machine learning-based lower face paralysis detector.
    Detects paralysis using a trained model.
    """

    def __init__(self):
        """Initialize the ML-based detector by loading model and scaler."""
        self.model = None
        self.scaler = None
        self.feature_names = None

        try:
            # Load model from models directory
            os.makedirs('models', exist_ok=True)
            self.model = joblib.load('models/lower_face_paralysis_model.pkl')
            self.scaler = joblib.load('models/lower_face_paralysis_scaler.pkl')

            # Load feature names if available
            if os.path.exists('models/feature_importance.csv'):
                feature_df = pd.read_csv('models/feature_importance.csv')
                self.feature_names = feature_df['feature'].tolist()

            logger.info("ML lower face paralysis detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML detector: {str(e)}")
            # No fallback - this is an error state that should be fixed
            raise ValueError("Failed to load ML model for lower face detection")

    def detect_lower_face_paralysis(self, self_orig, info, zone, side, aus, values, other_values,
                                    values_normalized, other_values_normalized,
                                    zone_paralysis, affected_aus_by_zone_side,
                                    asymmetry_thresholds, confidence_thresholds):
        """
        ML-based detection for lower face paralysis.
        Uses trained model to detect paralysis.

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
            asymmetry_thresholds (dict): Thresholds for asymmetry detection
            confidence_thresholds (dict): Thresholds for confidence scoring

        Returns:
            bool: True if paralysis was detected, False otherwise
        """
        # Check if model loaded successfully
        if self.model is None or self.scaler is None:
            # This is an error state - model is required
            error_msg = "ML model not available for lower face detection"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Extract features for prediction - now passing the zone parameter
            features = self._extract_features(
                info, side, zone, aus, values, other_values,
                values_normalized, other_values_normalized
            )

            # Scale features - bypass feature name validation by using numpy array directly
            # This avoids scikit-learn's feature name validation
            features_np = np.array(features).reshape(1, -1)

            # Make sure the feature vector has the right length
            expected_feature_count = getattr(self.model, 'n_features_in_',
                                             len(self.feature_names) if self.feature_names else len(features_np[0]))

            # Pad or truncate as needed
            if features_np.shape[1] < expected_feature_count:
                # Pad with zeros if too short
                padded = np.zeros((1, expected_feature_count))
                padded[0, :features_np.shape[1]] = features_np
                features_np = padded
            elif features_np.shape[1] > expected_feature_count:
                # Truncate if too long
                features_np = features_np[:, :expected_feature_count]

            # Suppress warnings during transform
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                scaled_features = self.scaler.transform(features_np)

            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            prediction_proba = self.model.predict_proba(scaled_features)[0]

            # Map prediction to result
            result_map = {0: 'None', 1: 'Partial', 2: 'Complete'}
            result = result_map[prediction]

            # Calculate confidence based on prediction probability
            confidence_score = prediction_proba[prediction]

            # Add minimum confidence threshold to avoid false positives
            min_confidence = {
                0: 0.5,  # "None" class - higher threshold to avoid false negatives
                1: 0.45,  # "Partial" class
                2: 0.4  # "Complete" class - lower threshold as it's usually rare
            }

            # If confidence is too low, consider downgrading the prediction
            if confidence_score < min_confidence[prediction]:
                if prediction == 2:  # If Complete with low confidence, downgrade to Partial
                    if prediction_proba[1] > 0.3:
                        result = 'Partial'
                        prediction = 1
                        confidence_score = prediction_proba[1]
                elif prediction == 1:  # If Partial with low confidence, check None confidence
                    if prediction_proba[0] > 0.4:
                        result = 'None'
                        prediction = 0
                        confidence_score = prediction_proba[0]

            # Update the info structure with detection result
            info['paralysis']['zones'][side][zone] = result

            # Track for patient-level assessment
            if result == 'Complete':
                zone_paralysis[side][zone] = 'Complete'
            elif result == 'Partial' and zone_paralysis[side][zone] == 'None':
                zone_paralysis[side][zone] = 'Partial'

            # Add affected AUs - only if paralysis detected
            if result != 'None':
                if 'AU12_r' in values:
                    affected_aus_by_zone_side[side][zone].add('AU12_r')
                    if 'AU12_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU12_r')

                if 'AU25_r' in values:
                    affected_aus_by_zone_side[side][zone].add('AU25_r')
                    if 'AU25_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU25_r')

            # Store confidence score
            info['paralysis']['confidence'][side][zone] = confidence_score

            # Add ML-specific information
            if 'ml_details' not in info['paralysis']:
                info['paralysis']['ml_details'] = {}

            info['paralysis']['ml_details'][f"{side}_{zone}"] = {
                'prediction': result,
                'prediction_proba': prediction_proba.tolist(),
                'confidence': confidence_score
            }

            # Add contributing AUs info for consistency
            if result != 'None':
                # Initialize ML detection list if needed
                if 'ml_detection' not in info['paralysis']['contributing_aus'][side][zone]:
                    info['paralysis']['contributing_aus'][side][zone]['ml_detection'] = []

                # Track AU values for ML detection
                info['paralysis']['contributing_aus'][side][zone]['ml_detection'].append({
                    'au': 'AU12_r',
                    'side_value': values.get('AU12_r', 0),
                    'other_value': other_values.get('AU12_r', 0),
                    'ml_confidence': confidence_score,
                    'type': result
                })

                if 'AU25_r' in values:
                    info['paralysis']['contributing_aus'][side][zone]['ml_detection'].append({
                        'au': 'AU25_r',
                        'side_value': values.get('AU25_r', 0),
                        'other_value': other_values.get('AU25_r', 0),
                        'ml_confidence': confidence_score,
                        'type': result
                    })

            # Log the detection result
            logger.debug(f"{side} lower face: {result} paralysis detected with ML confidence {confidence_score:.3f}")

            # Return success if paralysis was detected
            return result != 'None'

        except Exception as e:
            logger.error(f"Exception in ML detect_lower_face_paralysis: {str(e)}")
            raise  # Propagate error upward - no fallback

    def _extract_features(self, info, side, zone, aus, values, other_values, values_normalized,
                          other_values_normalized):
        """
        Extract features for ML model from current action data.

        Args:
            info (dict): Results dictionary for current action
            side (str): Side being analyzed ('left' or 'right')
            zone (str): Zone being analyzed (should be 'lower')
            aus (list): List of Action Units for this zone
            values (dict): AU values for this side
            other_values (dict): AU values for other side
            values_normalized (dict): Normalized AU values for this side
            other_values_normalized (dict): Normalized AU values for other side

        Returns:
            list: Feature vector for model input
        """
        # Determine which action we're analyzing
        action = self._get_current_action(info)
        opposite_side = 'Right' if side == 'Left' else 'Left'

        # Initialize empty feature dictionary
        features = {}

        # 1. Basic AU values
        au12_val = values.get('AU12_r', 0)
        au25_val = values.get('AU25_r', 0)
        au12_other = other_values.get('AU12_r', 0)
        au25_other = other_values.get('AU25_r', 0)

        features[f"{action}_AU12_raw"] = au12_val
        features[f"{action}_AU25_raw"] = au25_val

        # 2. Normalized values if available
        au12_norm = values_normalized.get('AU12_r', au12_val) if values_normalized else au12_val
        au25_norm = values_normalized.get('AU25_r', au25_val) if values_normalized else au25_val
        au12_other_norm = other_values_normalized.get('AU12_r', au12_other) if other_values_normalized else au12_other
        au25_other_norm = other_values_normalized.get('AU25_r', au25_other) if other_values_normalized else au25_other

        features[f"{action}_AU12_norm"] = au12_norm
        features[f"{action}_AU25_norm"] = au25_norm

        # 3. Calculate asymmetry metrics
        au12_ratio = self._calculate_ratio(au12_norm, au12_other_norm)
        au25_ratio = self._calculate_ratio(au25_norm, au25_other_norm)
        au12_percent_diff = self._calculate_percent_diff(au12_norm, au12_other_norm)
        au25_percent_diff = self._calculate_percent_diff(au25_norm, au25_other_norm)

        features[f"{action}_AU12_ratio"] = au12_ratio
        features[f"{action}_AU25_ratio"] = au25_ratio
        features[f"{action}_AU12_percent_diff"] = au12_percent_diff
        features[f"{action}_AU25_percent_diff"] = au25_percent_diff

        # 4. Normalized asymmetry metrics
        features[f"{action}_AU12_norm_ratio"] = au12_ratio  # Same as above since we used normalized values
        features[f"{action}_AU12_norm_percent_diff"] = au12_percent_diff
        features[f"{action}_AU25_norm_ratio"] = au25_ratio
        features[f"{action}_AU25_norm_percent_diff"] = au25_percent_diff

        # 5. Directional asymmetry (which side is weaker)
        features[f"{action}_AU12_is_weaker"] = 1 if au12_norm < au12_other_norm else 0
        features[f"{action}_AU25_is_weaker"] = 1 if au25_norm < au25_other_norm else 0

        # 6. Combined score
        features[f"{action}_combined_score"] = self._calculate_combined_score(
            au12_norm, au25_norm, au12_other_norm, au25_other_norm
        )

        # 7. Enhanced features - AU interactions
        features[f"{action}_AU12_AU25_interaction"] = au12_norm * au25_norm
        features[f"{action}_ratio_product"] = au12_ratio * au25_ratio
        features[f"{action}_asymmetry_differential"] = abs(au12_percent_diff - au25_percent_diff)

        # 8. Side indicator
        features["side"] = 1 if side == 'Right' else 0

        # 9. Add current detection if available
        if hasattr(info['paralysis']['zones'][side], 'get'):
            current_detection = info['paralysis']['zones'][side].get(zone, 'None')
            features['current_detection'] = {'None': 0, 'Partial': 1, 'Complete': 2}.get(current_detection, 0)
        else:
            features['current_detection'] = 0

        # Create feature vector
        if self.feature_names and hasattr(self.model, 'n_features_in_'):
            # If we know how many features the model expects, just return values
            # We'll handle padding/truncating in the calling method
            return list(features.values())
        elif self.feature_names:
            # Create feature vector in order of feature_names
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    # If feature not found, use a default value of 0
                    feature_vector.append(0)
            return feature_vector
        else:
            # Return values if feature names not available
            return list(features.values())

    def _get_current_action(self, info):
        """
        Determine the current action being analyzed.

        Args:
            info (dict): Results dictionary

        Returns:
            str: Current action name
        """
        # Check common action names in info keys
        for action in ['BS', 'SS', 'SO', 'SE']:
            if action in info:
                return action

        # Try checking action from the structure
        if 'action' in info:
            return info['action']

        # Default to Big Smile if action not found
        return 'BS'

    def _calculate_ratio(self, val1, val2):
        """
        Calculate ratio between values (min/max).

        Args:
            val1 (float): First value
            val2 (float): Second value

        Returns:
            float: Ratio (min/max)
        """
        if val1 <= 0 and val2 <= 0:
            return 0.0

        # Get min and max values
        min_val = min(val1, val2)
        max_val = max(val1, val2)

        # Avoid division by zero
        if max_val <= 0:
            return 0.0

        ratio = min_val / max_val

        # Cap at 1.0 to avoid floating point issues
        return min(ratio, 1.0)

    def _calculate_percent_diff(self, val1, val2):
        """
        Calculate percent difference between values.

        Args:
            val1 (float): First value
            val2 (float): Second value

        Returns:
            float: Percent difference
        """
        if val1 == 0 and val2 == 0:
            return 0.0

        # Calculate absolute difference
        abs_diff = abs(val1 - val2)

        # Calculate average
        avg = (val1 + val2) / 2.0

        # Avoid division by zero
        if avg <= 0:
            return 100.0 if val1 != val2 else 0.0

        # Calculate percent difference
        percent_diff = (abs_diff / avg) * 100.0

        # Cap at 200% for extreme differences
        return min(percent_diff, 200.0)

    def _calculate_combined_score(self, au12_val, au25_val, au12_opp, au25_opp):
        """
        Calculate combined score similar to original algorithm.

        Args:
            au12_val (float): AU12 value for current side
            au25_val (float): AU25 value for current side
            au12_opp (float): AU12 value for opposite side
            au25_opp (float): AU25 value for opposite side

        Returns:
            float: Combined score (lower values indicate more severe paralysis)
        """
        # Weights for each AU
        au12_weight = 0.6
        au25_weight = 0.4

        # Calculate ratios
        au12_ratio = self._calculate_ratio(au12_val, au12_opp)
        au25_ratio = self._calculate_ratio(au25_val, au25_opp)

        # Combined ratio
        combined_ratio = (au12_ratio * au12_weight) + (au25_ratio * au25_weight)

        # Minimum values (use the minimum of current side and opposite side)
        au12_min = min(au12_val, au12_opp)
        au25_min = min(au25_val, au25_opp)

        # Combined minimum
        combined_min = (au12_min * au12_weight) + (au25_min * au25_weight)

        # Final score
        return combined_ratio * (1 + combined_min / 5.0)