"""
First-stage ML model for lower face paralysis detection.
Determines presence or absence of paralysis in binary classification.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = logging.getLogger(__name__)

class MLLowerFaceBinaryDetector:
    """
    First-stage ML model to detect presence or absence of facial paralysis.
    """

    def __init__(self):
        """Initialize the binary detector model."""
        self.model = None
        self.scaler = None
        self.feature_names = None

        try:
            # Try to load model from models directory first
            if os.path.exists('models/lower_face_binary_model.pkl'):
                self.model = joblib.load('models/lower_face_binary_model.pkl')
                self.scaler = joblib.load('models/lower_face_binary_scaler.pkl')

                # Load feature names if available
                if os.path.exists('models/binary_feature_importance.csv'):
                    feature_df = pd.read_csv('models/binary_feature_importance.csv')
                    self.feature_names = feature_df['feature'].tolist()

            logger.info("ML lower face binary detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing binary detector: {str(e)}")
            raise ValueError("Failed to load ML model for binary lower face detection")

    def detect_paralysis_presence(self, info, side, zone, aus, values, other_values,
                                values_normalized, other_values_normalized):
        """
        Detect presence of lower face paralysis using binary classification.

        Args:
            info (dict): Results dictionary for current action
            side (str): Side being analyzed ('left' or 'right')
            zone (str): Facial zone being analyzed ('lower')
            aus (list): List of Action Units for this zone
            values (dict): AU values for this side
            other_values (dict): AU values for other side
            values_normalized (dict): Normalized AU values for this side
            other_values_normalized (dict): Normalized AU values for other side

        Returns:
            tuple: (has_paralysis, confidence_score)
        """
        if self.model is None or self.scaler is None:
            logger.error("Binary model not available")
            return False, 0.0

        try:
            # Extract features focused on paralysis detection
            features = self._extract_binary_features(
                info, side, zone, aus, values, other_values,
                values_normalized, other_values_normalized
            )

            # Scale features
            features_np = np.array(features).reshape(1, -1)

            # Ensure feature vector length matches expected
            expected_feature_count = getattr(self.model, 'n_features_in_',
                                         len(self.feature_names) if self.feature_names else len(features_np[0]))

            # Pad or truncate as needed
            if features_np.shape[1] < expected_feature_count:
                padded = np.zeros((1, expected_feature_count))
                padded[0, :features_np.shape[1]] = features_np
                features_np = padded
            elif features_np.shape[1] > expected_feature_count:
                features_np = features_np[:, :expected_feature_count]

            # Suppress warnings during transform
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                scaled_features = self.scaler.transform(features_np)

            # Make prediction (binary classification)
            prediction = self.model.predict(scaled_features)[0]
            prediction_proba = self.model.predict_proba(scaled_features)[0]

            # Get confidence score (probability of positive class)
            confidence_score = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]

            # Return result: has_paralysis, confidence_score
            return prediction == 1, confidence_score

        except Exception as e:
            logger.error(f"Exception in binary detection: {str(e)}")
            return False, 0.0

    def _extract_binary_features(self, info, side, zone, aus, values, other_values,
                               values_normalized, other_values_normalized):
        """
        Extract features specifically tailored for binary paralysis detection.
        Focus on asymmetry metrics and BS action features.
        """
        # Get current action
        action = self._get_current_action(info)

        # Initialize feature dictionary
        features = {}

        # 1. Strong asymmetry features
        for au in ['AU12_r', 'AU25_r']:
            if au in values and au in other_values:
                # Get values (prefer normalized if available)
                side_val = values_normalized.get(au, values.get(au, 0)) if values_normalized else values.get(au, 0)
                other_val = other_values_normalized.get(au, other_values.get(au, 0)) if other_values_normalized else other_values.get(au, 0)

                # Calculate asymmetry metrics
                percent_diff = self._calculate_percent_diff(side_val, other_val)
                ratio = self._calculate_ratio(side_val, other_val)

                # Store features
                features[f"{action}_{au}_percent_diff"] = percent_diff
                features[f"{action}_{au}_ratio"] = ratio
                features[f"{action}_{au}_side_value"] = side_val
                features[f"{action}_{au}_other_value"] = other_val

                # Extreme asymmetry flag (>120% difference)
                features[f"{action}_{au}_extreme_asymmetry"] = 1 if percent_diff > 120 else 0

        # 2. Action-specific features (emphasize BS features)
        if action == 'BS':
            # BS is most important for paralysis detection
            bs_au12_side = values_normalized.get('AU12_r', values.get('AU12_r', 0)) if values_normalized else values.get('AU12_r', 0)
            bs_au12_other = other_values_normalized.get('AU12_r', other_values.get('AU12_r', 0)) if other_values_normalized else other_values.get('AU12_r', 0)

            # Calculate additional BS-specific features
            features['BS_AU12_absolute_diff'] = abs(bs_au12_side - bs_au12_other)
            features['BS_AU12_side_to_other_ratio'] = bs_au12_side / max(bs_au12_other, 0.1)  # Avoid division by zero

            # Add AU25_r features for BS
            bs_au25_side = values_normalized.get('AU25_r', values.get('AU25_r', 0)) if values_normalized else values.get('AU25_r', 0)
            bs_au25_other = other_values_normalized.get('AU25_r', other_values.get('AU25_r', 0)) if other_values_normalized else other_values.get('AU25_r', 0)

            features['BS_AU25_absolute_diff'] = abs(bs_au25_side - bs_au25_other)
            features['BS_AU25_side_to_other_ratio'] = bs_au25_side / max(bs_au25_other, 0.1)

        # 3. Combined AU behavior features
        if 'AU12_r' in values and 'AU25_r' in values:
            # AU interaction metrics
            au12_val = values_normalized.get('AU12_r', values.get('AU12_r', 0)) if values_normalized else values.get('AU12_r', 0)
            au25_val = values_normalized.get('AU25_r', values.get('AU25_r', 0)) if values_normalized else values.get('AU25_r', 0)

            features[f"{action}_AU12_AU25_interaction"] = au12_val * au25_val
            features[f"{action}_AU12_AU25_ratio"] = au12_val / max(au25_val, 0.1)

            # Same for other side
            au12_other = other_values_normalized.get('AU12_r', other_values.get('AU12_r', 0)) if other_values_normalized else other_values.get('AU12_r', 0)
            au25_other = other_values_normalized.get('AU25_r', other_values.get('AU25_r', 0)) if other_values_normalized else other_values.get('AU25_r', 0)

            # Interaction pattern difference between sides
            side_interaction = au12_val * au25_val
            other_interaction = au12_other * au25_other

            features[f"{action}_interaction_asymmetry"] = abs(side_interaction - other_interaction) / max(0.1, (side_interaction + other_interaction) / 2)

        # 4. Side indicator
        features["side"] = 1 if side == 'Right' else 0

        # Convert features dict to feature vector
        if self.feature_names:
            # Create feature vector in order of feature_names
            feature_vector = []
            for name in self.feature_names:
                feature_vector.append(features.get(name, 0))
            return feature_vector
        else:
            # Return values if feature names not available
            return list(features.values())

    def _get_current_action(self, info):
        """Get the current action being analyzed."""
        for action in ['BS', 'SS', 'SO', 'SE']:
            if action in info:
                return action

        if 'action' in info:
            return info['action']

        return 'BS'

    def _calculate_percent_diff(self, val1, val2):
        """Calculate percent difference between values."""
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

    def _calculate_ratio(self, val1, val2):
        """Calculate ratio between values (min/max)."""
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