"""
Second-stage ML model for lower face paralysis severity detection.
Determines partial vs. complete paralysis severity.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = logging.getLogger(__name__)


class MLLowerFaceSeverityDetector:
    """
    Second-stage ML model to classify paralysis severity into partial or complete.
    """

    def __init__(self):
        """Initialize the severity detector model."""
        self.model = None
        self.scaler = None
        self.feature_names = None

        try:
            # Try to load model from models directory first
            if os.path.exists('models/lower_face_severity_model.pkl'):
                self.model = joblib.load('models/lower_face_severity_model.pkl')
                self.scaler = joblib.load('models/lower_face_severity_scaler.pkl')

                # Load feature names if available
                if os.path.exists('models/severity_feature_importance.csv'):
                    feature_df = pd.read_csv('models/severity_feature_importance.csv')
                    self.feature_names = feature_df['feature'].tolist()

            logger.info("ML lower face severity detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing severity detector: {str(e)}")
            raise ValueError("Failed to load ML model for lower face severity detection")

    def classify_paralysis_severity(self, info, side, zone, aus, values, other_values,
                                    values_normalized, other_values_normalized):
        """
        Classify paralysis severity (partial vs. complete).

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
            tuple: (severity, confidence_score) - severity is either 'Partial' or 'Complete'
        """
        if self.model is None or self.scaler is None:
            logger.error("Severity model not available")
            return 'Partial', 0.0

        try:
            # Extract features focused on severity classification
            features = self._extract_severity_features(
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

            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            prediction_proba = self.model.predict_proba(scaled_features)[0]

            # Map prediction to result (0 = Partial, 1 = Complete)
            result = 'Complete' if prediction == 1 else 'Partial'

            # Get confidence score
            confidence_score = prediction_proba[prediction]

            # Return result: severity, confidence_score
            return result, confidence_score

        except Exception as e:
            logger.error(f"Exception in severity classification: {str(e)}")
            return 'Partial', 0.0

    def _extract_severity_features(self, info, side, zone, aus, values, other_values,
                                   values_normalized, other_values_normalized):
        """
        Extract features specifically for severity classification.
        Focus on functional movement and fine-grained asymmetry features.
        """
        # Get current action
        action = self._get_current_action(info)

        # Initialize feature dictionary
        features = {}

        # 1. Functional movement measures
        for au in ['AU12_r', 'AU25_r']:
            if au in values:
                # Get values (prefer normalized if available)
                side_val = values_normalized.get(au, values.get(au, 0)) if values_normalized else values.get(au, 0)

                # Store absolute activation values (critical for severity)
                features[f"{action}_{au}_absolute_activation"] = side_val

                # Functional threshold features
                # Complete paralysis typically has very low AU activation
                features[f"{action}_{au}_below_threshold_04"] = 1 if side_val < 0.4 else 0
                features[f"{action}_{au}_below_threshold_08"] = 1 if side_val < 0.8 else 0
                features[f"{action}_{au}_below_threshold_12"] = 1 if side_val < 1.2 else 0

                # Movement capacity feature
                features[f"{action}_{au}_movement_capacity"] = min(1.0, side_val / 3.0)  # Normalize to 0-1 range

        # 2. Fine-grained asymmetry features
        for au in ['AU12_r', 'AU25_r']:
            if au in values and au in other_values:
                # Get values (prefer normalized if available)
                side_val = values_normalized.get(au, values.get(au, 0)) if values_normalized else values.get(au, 0)
                other_val = other_values_normalized.get(au, other_values.get(au,
                                                                             0)) if other_values_normalized else other_values.get(
                    au, 0)

                # Calculate fine-grained asymmetry metrics
                percent_diff = self._calculate_percent_diff(side_val, other_val)
                ratio = self._calculate_ratio(side_val, other_val)

                # Store basic asymmetry features
                features[f"{action}_{au}_percent_diff"] = percent_diff
                features[f"{action}_{au}_ratio"] = ratio

                # Severity-weighted asymmetry scores
                severity_weight = 1.0
                if au == 'AU12_r':  # More weight to AU12_r
                    severity_weight = 1.5

                features[f"{action}_{au}_weighted_asym"] = percent_diff * severity_weight

                # Extreme asymmetry flags with specific thresholds per AU
                au_threshold = 150.0 if au == 'AU12_r' else 130.0
                features[f"{action}_{au}_extreme_asym"] = 1 if percent_diff > au_threshold else 0

                # Lower threshold flags
                features[f"{action}_{au}_severe_asym"] = 1 if percent_diff > 80.0 else 0
                features[f"{action}_{au}_moderate_asym"] = 1 if percent_diff > 50.0 else 0

                # Ratio-based severity scores
                features[f"{action}_{au}_severity_score"] = 1.0 - ratio  # Higher value = more severe

        # 3. Confidence modifiers
        # Calculate consistency across multiple indicators
        au12_metrics = [
            features.get(f"{action}_AU12_r_below_threshold_04", 0),
            features.get(f"{action}_AU12_r_severe_asym", 0),
            features.get(f"{action}_AU12_r_extreme_asym", 0)
        ]

        au25_metrics = [
            features.get(f"{action}_AU25_r_below_threshold_04", 0),
            features.get(f"{action}_AU25_r_severe_asym", 0),
            features.get(f"{action}_AU25_r_extreme_asym", 0)
        ]

        # Calculate consistency scores
        features[f"{action}_AU12_consistency"] = sum(au12_metrics) / len(au12_metrics)
        features[f"{action}_AU25_consistency"] = sum(au25_metrics) / len(au25_metrics)
        features[f"{action}_overall_consistency"] = (features[f"{action}_AU12_consistency"] +
                                                     features[f"{action}_AU25_consistency"]) / 2

        # Borderline case detection
        au12_val = values_normalized.get('AU12_r', values.get('AU12_r', 0)) if values_normalized else values.get(
            'AU12_r', 0)
        au12_ratio = features.get(f"{action}_AU12_r_ratio", 0)

        # Borderline cases: moderate asymmetry with some movement
        features[f"{action}_borderline_case"] = 1 if (50 < features.get(f"{action}_AU12_r_percent_diff", 0) < 90 and
                                                      0.4 < au12_val < 1.2) else 0

        # 4. Secondary pattern features
        # Check for compensatory movement patterns
        if 'AU14_r' in values and 'AU14_r' in other_values:
            au14_side = values.get('AU14_r', 0)
            au14_other = other_values.get('AU14_r', 0)

            # Compensatory movement: increased AU14_r on affected side
            features[f"{action}_AU14_compensatory"] = 1 if au14_side > au14_other * 1.5 else 0

        # 5. Cross-action comparisons
        # We would need data from other actions to implement this
        # For now, use what we have available from the current action

        # Side indicator
        features["side"] = 1 if side == 'Right' else 0

        # Convert features dict to vector
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