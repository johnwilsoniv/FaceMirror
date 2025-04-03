"""
ML-based Snarl-Smile synkinesis detector.
Uses trained machine learning model to detect smile-to-snarl synkinesis.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import warnings

# Suppress specific scikit-learn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = logging.getLogger(__name__)


class MLSnarlSmileDetector:
    """
    Machine learning-based Snarl-Smile synkinesis detector.
    Detects synkinesis where smile actions cause unwanted nose wrinkling.
    """

    def __init__(self):
        """Initialize the ML-based detector by loading model and scaler."""
        self.model = None
        self.scaler = None
        self.feature_names = None

        try:
            # Try to load model from models directory first (preferred)
            if os.path.exists('models/synkinesis/snarl_smile/model.pkl'):
                self.model = joblib.load('models/synkinesis/snarl_smile/model.pkl')
                self.scaler = joblib.load('models/synkinesis/snarl_smile/scaler.pkl')

                # Load feature names if available
                if os.path.exists('models/synkinesis/snarl_smile/feature_importance.csv'):
                    feature_df = pd.read_csv('models/synkinesis/snarl_smile/feature_importance.csv')
                    self.feature_names = feature_df['feature'].tolist()
            # Fall back to root directory
            elif os.path.exists('snarl_smile_synkinesis_model.pkl'):
                self.model = joblib.load('snarl_smile_synkinesis_model.pkl')
                self.scaler = joblib.load('snarl_smile_synkinesis_scaler.pkl')

                # Load feature names if available
                if os.path.exists('snarl_smile_feature_importance.csv'):
                    feature_df = pd.read_csv('snarl_smile_feature_importance.csv')
                    self.feature_names = feature_df['feature'].tolist()

            logger.info("ML Snarl-Smile synkinesis detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML Snarl-Smile detector: {str(e)}")
            # No fallback - this is an error state that should be fixed
            logger.warning("ML Snarl-Smile detector will not be available")
            self.model = None
            self.scaler = None

    def detect_snarl_smile_synkinesis(self, action, info, side):
        """
        ML-based detection for Snarl-Smile synkinesis.
        Uses trained model to detect synkinesis.

        Args:
            action (str): The current facial action being analyzed
            info (dict): Results dictionary for current action
            side (str): Side being analyzed ('left' or 'right')

        Returns:
            tuple: (is_detected, confidence_score, contributing_aus)
        """
        # Check if model loaded successfully
        if self.model is None or self.scaler is None:
            # Skip ML detection if model not available
            logger.warning(f"ML model not available for Snarl-Smile synkinesis detection")
            return False, 0.0, {'trigger': [], 'response': []}

        try:
            # Only process relevant actions for Snarl-Smile synkinesis (smile actions)
            relevant_actions = ['BS', 'SS']
            if action not in relevant_actions:
                return False, 0.0, {'trigger': [], 'response': []}

            # Extract features for prediction
            features = self._extract_features(action, info, side)

            # Scale features - bypass feature name validation by using numpy array directly
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

            # Get confidence score and binary result
            if len(prediction_proba) >= 2:  # Binary or multi-class
                is_detected = prediction > 0  # Any class > 0 means synkinesis detected
                confidence_score = prediction_proba[prediction]
            else:
                is_detected = prediction > 0.5  # For regression-like output
                confidence_score = prediction

            # Determine contributing AUs
            contributing_aus = self._determine_contributing_aus(action, info, side, is_detected)

            # Log the detection result
            if is_detected:
                logger.debug(f"ML Snarl-Smile synkinesis detected on {side} side during {action} with confidence {confidence_score:.3f}")

            return is_detected, confidence_score, contributing_aus

        except Exception as e:
            logger.error(f"Exception in ML detect_snarl_smile_synkinesis: {str(e)}")
            # Return no detection on error
            return False, 0.0, {'trigger': [], 'response': []}

    def _extract_features(self, action, info, side):
        """
        Extract features for ML model from current action data.

        Args:
            action (str): The facial action being analyzed
            info (dict): Results dictionary for current action
            side (str): Side being analyzed ('left' or 'right')

        Returns:
            list: Feature vector for model input
        """
        features = {}
        
        # Relevant action data for Snarl-Smile synkinesis
        # Trigger AU (smile related)
        trigger_au = 'AU12_r'
        # Coupled AUs (snarl related)
        coupled_aus = ['AU09_r', 'AU10_r', 'AU14_r']
        
        # Get opposite side
        opposite_side = 'right' if side == 'left' else 'left'
        
        # Extract feature for trigger AU (smile action)
        if trigger_au in info[side]['normalized_au_values']:
            features[f"{action}_{trigger_au}_norm"] = info[side]['normalized_au_values'][trigger_au]
        else:
            features[f"{action}_{trigger_au}_norm"] = 0
            
        # Extract features for coupled AUs (snarl response)
        for au in coupled_aus:
            if au in info[side]['normalized_au_values']:
                features[f"{action}_{au}_norm"] = info[side]['normalized_au_values'][au]
            else:
                features[f"{action}_{au}_norm"] = 0
                
        # Calculate interaction features
        for coupled_au in coupled_aus:
            trigger_val = features.get(f"{action}_{trigger_au}_norm", 0)
            coupled_val = features.get(f"{action}_{coupled_au}_norm", 0)
            
            # Calculate ratio and product
            features[f"{action}_{trigger_au}_{coupled_au}_ratio"] = min(trigger_val, coupled_val) / max(trigger_val, coupled_val) if max(trigger_val, coupled_val) > 0 else 0
            features[f"{action}_{trigger_au}_{coupled_au}_product"] = trigger_val * coupled_val
            
        # Calculate summary features
        coupled_vals = [features.get(f"{action}_{au}_norm", 0) for au in coupled_aus]
        
        features[f"{action}_coupled_avg"] = sum(coupled_vals) / len(coupled_vals) if coupled_vals else 0
        features[f"{action}_coupled_max"] = max(coupled_vals) if coupled_vals else 0
        
        # Ratio of trigger to coupled activation
        trigger_val = features.get(f"{action}_{trigger_au}_norm", 0)
        coupled_avg = features.get(f"{action}_coupled_avg", 0)
        features[f"{action}_trigger_coupled_ratio"] = min(trigger_val, coupled_avg) / max(trigger_val, coupled_avg) if max(trigger_val, coupled_avg) > 0 else 0
        
        # Weighted score based on importance weights
        au09_weight = 0.4  # Nose wrinkle importance
        au10_weight = 0.3  # Upper lip raiser importance
        au14_weight = 0.3  # Dimpler importance
        
        # Calculate weighted score
        weighted_score = (
            features.get(f"{action}_AU09_r_norm", 0) * au09_weight +
            features.get(f"{action}_AU10_r_norm", 0) * au10_weight +
            features.get(f"{action}_AU14_r_norm", 0) * au14_weight
        )
        features[f"{action}_weighted_score"] = weighted_score
        
        # Count number of active components (AUs exceeding thresholds)
        thresholds = {
            'AU09_r': 0.8,
            'AU10_r': 0.8,
            'AU14_r': 1.2
        }
        
        active_component_count = sum(1 for au in coupled_aus if features.get(f"{action}_{au}_norm", 0) > thresholds.get(au, 0.8))
        features['active_component_count'] = active_component_count
        
        # Add side indicator
        features["side"] = 1 if side == 'right' else 0
        
        # Handle feature names for the model
        if self.feature_names:
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
            # Return all feature values if feature names not available
            return list(features.values())
            
    def _determine_contributing_aus(self, action, info, side, is_detected):
        """
        Determine which AUs contributed to synkinesis detection.
        
        Args:
            action (str): The facial action being analyzed
            info (dict): Results dictionary for current action
            side (str): Side being analyzed ('left' or 'right')
            is_detected (bool): Whether synkinesis was detected
            
        Returns:
            dict: Dictionary with lists of contributing trigger and response AUs
        """
        if not is_detected:
            return {'trigger': [], 'response': []}
            
        # Define thresholds for significant AU activation
        trigger_threshold = 2.0  # Significant smile action
        coupled_thresholds = {
            'AU09_r': 0.8,  # Nose wrinkler
            'AU10_r': 0.8,  # Upper lip raiser
            'AU14_r': 1.2   # Dimpler
        }
        
        # Define potential AUs
        trigger_au = 'AU12_r'
        coupled_aus = ['AU09_r', 'AU10_r', 'AU14_r']
        
        # Find active trigger AU
        active_trigger_aus = []
        if (trigger_au in info[side]['normalized_au_values'] and 
                info[side]['normalized_au_values'][trigger_au] > trigger_threshold):
            active_trigger_aus.append(trigger_au)
                
        # Find active coupled AUs (unwanted responses)
        active_coupled_aus = []
        for au in coupled_aus:
            if (au in info[side]['normalized_au_values'] and 
                    info[side]['normalized_au_values'][au] > coupled_thresholds.get(au, 0.8)):
                active_coupled_aus.append(au)
                
        return {
            'trigger': active_trigger_aus,
            'response': active_coupled_aus
        }