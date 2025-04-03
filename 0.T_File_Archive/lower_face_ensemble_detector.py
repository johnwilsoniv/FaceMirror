"""
Ensemble detector for facial paralysis detection.
Implements a two-stage approach: first detecting normal vs. abnormal,
then classifying abnormal cases as partial or complete.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lower_face_ml_detector import LowerFaceParalysisDetector

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

class EnsembleLowerFaceDetector:
    """
    Two-stage ensemble detector for lower face paralysis.
    First detects normal vs. abnormal, then classifies abnormal cases.
    """
    
    def __init__(self):
        """Initialize the ensemble detector."""
        # Try to load the main ML detector
        try:
            self.ml_detector = LowerFaceParalysisDetector()
            logger.info("Main ML detector loaded successfully")
        except Exception as e:
            logger.error(f"Error loading main ML detector: {str(e)}")
            self.ml_detector = None
        
        # Initialize binary classifier (normal vs. abnormal)
        self.binary_classifier = None
        self.binary_scaler = None
        
        # Initialize multiclass classifier (None, Partial, Complete)
        self.multiclass_classifier = None
        self.multiclass_scaler = None
        
        # Try to load ensemble components if available
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models/ensemble', exist_ok=True)
            
            # Load binary classifier if available
            if os.path.exists('models/ensemble/binary_classifier.pkl'):
                self.binary_classifier = joblib.load('models/ensemble/binary_classifier.pkl')
                self.binary_scaler = joblib.load('models/ensemble/binary_scaler.pkl')
                logger.info("Binary classifier loaded successfully")
            
            # Load multiclass classifier if available
            if os.path.exists('models/ensemble/multiclass_classifier.pkl'):
                self.multiclass_classifier = joblib.load('models/ensemble/multiclass_classifier.pkl')
                self.multiclass_scaler = joblib.load('models/ensemble/multiclass_scaler.pkl')
                logger.info("Multiclass classifier loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading ensemble components: {str(e)}")
    
    def detect(self, info, side, zone, aus, values, other_values, 
              values_normalized, other_values_normalized):
        """
        Detect lower face paralysis using the ensemble approach.
        
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
        # Use main ML detector first
        if self.ml_detector:
            try:
                # Extract features using the ML detector's method
                features = self.ml_detector._extract_features(
                    info, side, zone, aus, values, other_values,
                    values_normalized, other_values_normalized
                )
                
                # Get predictions from the ML detector
                features_np = np.array(features).reshape(1, -1)
                expected_features = getattr(self.ml_detector.model, 'n_features_in_', 
                                          len(features_np[0]))
                
                # Pad or truncate as needed
                if features_np.shape[1] < expected_features:
                    padded = np.zeros((1, expected_features))
                    padded[0, :features_np.shape[1]] = features_np
                    features_np = padded
                elif features_np.shape[1] > expected_features:
                    features_np = features_np[:, :expected_features]
                
                # Scale features
                scaled_features = self.ml_detector.scaler.transform(features_np)
                
                # Get ML detector prediction and probabilities
                ml_prediction = self.ml_detector.model.predict(scaled_features)[0]
                ml_proba = self.ml_detector.model.predict_proba(scaled_features)[0]
                
                # Apply post-processing
                ml_adjusted = self._apply_thresholds(ml_prediction, ml_proba)
                
                # Check if ensemble components are available
                if self.binary_classifier and self.multiclass_classifier:
                    try:
                        # Get binary classifier prediction (normal vs. abnormal)
                        binary_scaled = self.binary_scaler.transform(features_np)
                        is_abnormal = self.binary_classifier.predict(binary_scaled)[0]
                        binary_proba = self.binary_classifier.predict_proba(binary_scaled)[0]
                        
                        if is_abnormal:
                            # If abnormal, use multiclass classifier for Partial vs. Complete
                            multiclass_scaled = self.multiclass_scaler.transform(features_np)
                            severity = self.multiclass_classifier.predict(multiclass_scaled)[0]
                            severity_proba = self.multiclass_classifier.predict_proba(multiclass_scaled)[0]
                            
                            # Map to result (1 = Partial, 2 = Complete)
                            ensemble_prediction = severity + 1  # Map 0->1 (Partial), 1->2 (Complete)
                        else:
                            # Not abnormal
                            ensemble_prediction = 0  # None
                            severity_proba = [1.0, 0.0, 0.0]  # Placeholder probabilities
                        
                        # Combine ML and ensemble predictions
                        final_prediction = self._combine_predictions(
                            ml_adjusted, ml_proba,
                            ensemble_prediction, 
                            [binary_proba[0], severity_proba[0] if is_abnormal else 0, 
                             severity_proba[1] if is_abnormal else 0]
                        )
                    except Exception as e:
                        logger.error(f"Error in ensemble prediction: {str(e)}")
                        final_prediction = ml_adjusted
                else:
                    # Fallback to ML detector prediction
                    final_prediction = ml_adjusted
                
                # Map prediction to result
                result_map = {0: 'None', 1: 'Partial', 2: 'Complete'}
                result = result_map[final_prediction]
                
                # Calculate confidence
                confidence = ml_proba[final_prediction]
                
                # Prepare details
                details = {
                    'ml_prediction': int(ml_prediction),
                    'ml_adjusted': int(ml_adjusted),
                    'final_prediction': int(final_prediction),
                    'probabilities': ml_proba.tolist(),
                    'ensemble_used': self.binary_classifier is not None and self.multiclass_classifier is not None
                }
                
                return result, confidence, details
                
            except Exception as e:
                logger.error(f"Error in ensemble detection: {str(e)}")
        
        # Fallback to conservative result
        return 'None', 0.0, {'error': 'Detection failed'}
    
    def _apply_thresholds(self, prediction, probabilities):
        """
        Apply custom thresholds to improve prediction accuracy.
        
        Args:
            prediction (int): Original prediction class
            probabilities (list): Class probabilities
            
        Returns:
            int: Adjusted prediction
        """
        # Apply the same thresholds as in the ML detector
        if prediction == 2:  # If model predicts Complete
            if probabilities[2] < 0.7:  # But confidence is less than 70%
                if probabilities[1] > 0.2:  # And there's reasonable chance of Partial
                    return 1  # Downgrade to Partial
                elif probabilities[0] > 0.35:  # Or significant chance of None
                    return 0  # Downgrade to None
        elif prediction == 0 and probabilities[1] > 0.3:  # If None but reasonable Partial probability
            return 1  # Upgrade to Partial
            
        return prediction
    
    def _combine_predictions(self, ml_prediction, ml_proba, ensemble_prediction, ensemble_proba):
        """
        Combine predictions from ML detector and ensemble model.
        
        Args:
            ml_prediction (int): ML detector prediction
            ml_proba (list): ML detector probabilities
            ensemble_prediction (int): Ensemble model prediction
            ensemble_proba (list): Ensemble model probabilities
            
        Returns:
            int: Combined prediction
        """
        # If predictions agree, use that prediction
        if ml_prediction == ensemble_prediction:
            return ml_prediction
        
        # If predictions differ, consider confidence levels
        ml_confidence = ml_proba[ml_prediction]
        ensemble_confidence = ensemble_proba[ensemble_prediction]
        
        # Use prediction with higher confidence
        if ml_confidence >= ensemble_confidence:
            return ml_prediction
        else:
            return ensemble_prediction
            
    def train_ensemble(self, features, targets):
        """
        Train the ensemble components using provided data.
        
        Args:
            features (pandas.DataFrame): Feature data
            targets (numpy.ndarray): Target labels (0=None, 1=Partial, 2=Complete)
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("Training ensemble detector components")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.25, random_state=42, stratify=targets
            )
            
            # Create binary labels (0=Normal, 1=Abnormal)
            y_binary_train = (y_train > 0).astype(int)
            y_binary_test = (y_test > 0).astype(int)
            
            # Train binary classifier (Normal vs. Abnormal)
            logger.info("Training binary classifier (Normal vs. Abnormal)")
            self.binary_scaler = StandardScaler()
            X_binary_train = self.binary_scaler.fit_transform(X_train)
            
            self.binary_classifier = RandomForestClassifier(
                n_estimators=300,
                class_weight={0: 1.0, 1: 2.0},  # Weight abnormal cases more
                random_state=42
            )
            
            self.binary_classifier.fit(X_binary_train, y_binary_train)
            
            # Evaluate binary classifier
            X_binary_test = self.binary_scaler.transform(X_test)
            binary_acc = self.binary_classifier.score(X_binary_test, y_binary_test)
            logger.info(f"Binary classifier accuracy: {binary_acc:.4f}")
            
            # Train multiclass classifier for abnormal cases (Partial vs. Complete)
            logger.info("Training multiclass classifier (Partial vs. Complete)")
            
            # Filter for abnormal cases only
            abnormal_idx_train = np.where(y_train > 0)[0]
            X_abnormal_train = X_train.iloc[abnormal_idx_train]
            y_abnormal_train = y_train[abnormal_idx_train] - 1  # Subtract 1 to get 0=Partial, 1=Complete
            
            self.multiclass_scaler = StandardScaler()
            X_multi_train = self.multiclass_scaler.fit_transform(X_abnormal_train)
            
            self.multiclass_classifier = RandomForestClassifier(
                n_estimators=300,
                class_weight={0: 2.0, 1: 1.0},  # Weight partial cases more
                random_state=42
            )
            
            self.multiclass_classifier.fit(X_multi_train, y_abnormal_train)
            
            # Save ensemble components
            logger.info("Saving ensemble components")
            joblib.dump(self.binary_classifier, 'models/ensemble/binary_classifier.pkl')
            joblib.dump(self.binary_scaler, 'models/ensemble/binary_scaler.pkl')
            joblib.dump(self.multiclass_classifier, 'models/ensemble/multiclass_classifier.pkl')
            joblib.dump(self.multiclass_scaler, 'models/ensemble/multiclass_scaler.pkl')
            
            logger.info("Ensemble components trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training ensemble components: {str(e)}", exc_info=True)