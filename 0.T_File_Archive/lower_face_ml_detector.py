""""
ML-based lower face paralysis detector.
Uses XGBoost machine learning model to detect lower face paralysis.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import warnings

# Suppress specific scikit-learn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

class LowerFaceParalysisDetector:
    """
    Machine learning-based lower face paralysis detector.
    Detects paralysis using a trained XGBoost model without threshold overrides.
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
            raise ValueError("Failed to load ML model for lower face detection")

    def detect_lower_face_paralysis(self, self_orig, info, zone, side, aus, values, other_values,
                                   values_normalized, other_values_normalized,
                                   zone_paralysis, affected_aus_by_zone_side, **kwargs):
        """
        ML-based detection for lower face paralysis.
        Uses trained XGBoost model to detect paralysis with no threshold overrides.

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
        # Check if model loaded successfully
        if self.model is None or self.scaler is None:
            error_msg = "ML model not available for lower face detection"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Extract features for prediction
            features = self._extract_features(
                info, side, zone, aus, values, other_values,
                values_normalized, other_values_normalized
            )

            # Scale features
            features_np = np.array(features).reshape(1, -1)

            # Ensure feature vector has right length
            expected_feature_count = getattr(self.model, 'n_features_in_',
                                           len(self.feature_names) if self.feature_names else len(features_np[0]))

            # Pad or truncate as needed
            if features_np.shape[1] < expected_feature_count:
                padded = np.zeros((1, expected_feature_count))
                padded[0, :features_np.shape[1]] = features_np
                features_np = padded
            elif features_np.shape[1] > expected_feature_count:
                features_np = features_np[:, :expected_feature_count]

            # Scale features
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                scaled_features = self.scaler.transform(features_np)

            # Get prediction and probabilities directly from model
            prediction = self.model.predict(scaled_features)[0]
            prediction_proba = self.model.predict_proba(scaled_features)[0]
            
            # Apply limited post-processing thresholds to reduce None-to-Complete errors
            # More conservative adjustment than before
            original_prediction = prediction
            
            # Only apply threshold for Complete predictions with low confidence
            if prediction == 2:  # If model predicts Complete
                if prediction_proba[2] < 0.5:  # But confidence is less than 60%
                    if prediction_proba[0] > 0.2:  # And there's significant chance it's None
                        prediction = 0  # Downgrade to None
                    elif prediction_proba[1] > 0.15:  # Or reasonable chance it's Partial
                        prediction = 1  # Downgrade to Partial
            
            # We won't upgrade None to Partial here - conservative approach
                
            # Map prediction to result
            result_map = {0: 'None', 1: 'Partial', 2: 'Complete'}
            result = result_map[prediction]
            
            # Store both original and adjusted predictions for analysis
            original_result = result_map[original_prediction]

            # Get confidence score from probability
            confidence_score = prediction_proba[prediction]

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

            # Store ML-specific information
            if 'ml_details' not in info['paralysis']:
                info['paralysis']['ml_details'] = {}

            info['paralysis']['ml_details'][f"{side}_{zone}"] = {
                'prediction': result,
                'original_prediction': original_result,
                'prediction_proba': prediction_proba.tolist(),
                'confidence': confidence_score,
                'threshold_adjusted': (original_prediction != prediction)
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
            raise

    # Helper method definitions first, before _extract_features
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

    def _extract_features(self, info, side, zone, aus, values, other_values, values_normalized,
                         other_values_normalized):
        """
        Extract features for ML model from current action data.
        Focuses on BS action features only.

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
        # Only using BS action
        action = 'BS'
        opposite_side = 'Right' if side == 'Left' else 'Left'

        # Initialize feature dictionary
        features = {}

        # 1. Raw AU values
        au12_val = values.get('AU12_r', 0)
        au25_val = values.get('AU25_r', 0)
        au12_other = other_values.get('AU12_r', 0)
        au25_other = other_values.get('AU25_r', 0)

        features[f"{action}_AU12_raw"] = au12_val
        features[f"{action}_AU25_raw"] = au25_val
        features[f"{action}_opp_AU12_raw"] = au12_other
        features[f"{action}_opp_AU25_raw"] = au25_other

        # 2. Normalized values
        au12_norm = values_normalized.get('AU12_r', au12_val) if values_normalized else au12_val
        au25_norm = values_normalized.get('AU25_r', au25_val) if values_normalized else au25_val
        au12_other_norm = other_values_normalized.get('AU12_r', au12_other) if other_values_normalized else au12_other
        au25_other_norm = other_values_normalized.get('AU25_r', au25_other) if other_values_normalized else au25_other

        features[f"{action}_AU12_norm"] = au12_norm
        features[f"{action}_AU25_norm"] = au25_norm
        features[f"{action}_opp_AU12_norm"] = au12_other_norm
        features[f"{action}_opp_AU25_norm"] = au25_other_norm

        # 3. Calculate asymmetry metrics
        # Ratio metrics (min/max)
        features[f"{action}_AU12_ratio"] = self._calculate_ratio(au12_norm, au12_other_norm)
        features[f"{action}_AU25_ratio"] = self._calculate_ratio(au25_norm, au25_other_norm)
        
        # Percent difference metrics
        features[f"{action}_AU12_percent_diff"] = self._calculate_percent_diff(au12_norm, au12_other_norm)
        features[f"{action}_AU25_percent_diff"] = self._calculate_percent_diff(au25_norm, au25_other_norm)

        # 4. Directional asymmetry (which side is weaker)
        features[f"{action}_AU12_is_weaker"] = 1 if au12_norm < au12_other_norm else 0
        features[f"{action}_AU25_is_weaker"] = 1 if au25_norm < au25_other_norm else 0

        # 5. Enhanced features - AU interactions
        features[f"{action}_AU12_AU25_interaction"] = au12_norm * au25_norm
        features[f"{action}_ratio_product"] = features[f"{action}_AU12_ratio"] * features[f"{action}_AU25_ratio"]
        features[f"{action}_asymmetry_differential"] = abs(
            features[f"{action}_AU12_percent_diff"] - features[f"{action}_AU25_percent_diff"]
        )

        # 6. Side indicator
        features["side"] = 1 if side == 'Right' else 0

        # 7. Additional features that might help with borderline cases
        # AU intensity sum (overall expressiveness)
        features[f"{action}_AU_intensity_sum"] = au12_norm + au25_norm
        
        # AU intensity ratio (relative contribution of AU12 vs AU25)
        denominator = au12_norm + au25_norm
        features[f"{action}_AU12_contribution"] = au12_norm / denominator if denominator > 0 else 0

        # Average ratio
        features[f"{action}_avg_ratio"] = features[f"{action}_AU12_ratio"] * 0.6 + features[f"{action}_AU25_ratio"] * 0.4
        
        # Average percent difference
        features[f"{action}_avg_percent_diff"] = (
            features[f"{action}_AU12_percent_diff"] * 0.6 + features[f"{action}_AU25_percent_diff"] * 0.4
        )

        # Create feature vector - return all features without filtering by known features
        # The model will handle feature alignment
        return list(features.values())