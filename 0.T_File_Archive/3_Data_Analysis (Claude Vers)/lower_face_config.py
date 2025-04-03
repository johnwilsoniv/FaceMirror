"""
Configuration for lower face paralysis detection.
Contains model parameters, thresholds, and file paths.
"""

import os

# Base paths
MODEL_DIR = 'models'
LOG_DIR = 'logs'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model filenames - Updated with consistent lower_face prefix
MODEL_FILENAMES = {
    'base_model': os.path.join(MODEL_DIR, 'lower_face_model.pkl'),
    'base_scaler': os.path.join(MODEL_DIR, 'lower_face_scaler.pkl'),
    'specialist_model': os.path.join(MODEL_DIR, 'lower_face_specialist_model.pkl'),
    'specialist_scaler': os.path.join(MODEL_DIR, 'lower_face_specialist_scaler.pkl'),
    'feature_importance': os.path.join(MODEL_DIR, 'lower_face_feature_importance.csv')
}

# Feature extraction parameters - Added to match structure of mid_face_config.py
FEATURE_CONFIG = {
    'actions': ['BS'],  # Big Smile - primary action for lower face
    'use_normalized': True,  # Always use normalized values when available
    'au_weights': {  # Relative importance of AUs for lower face
        'AU12_r': 0.6,  # Smile (primary)
        'AU25_r': 0.4   # Mouth open (secondary)
    },
    'percent_diff_cap': 200.0,  # Cap percent difference values
    'min_value': 0.0001  # Minimum value to avoid division by zero
}

# ML model training parameters
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    # Base model parameters (XGBoost)
    'base_model': {
        'objective': 'multi:softprob',
        'num_class': 3,  # None, Partial, Complete
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'n_estimators': 300,
        'class_weights': {0: 1.0, 1: 3.0, 2: 5.0}  # Emphasis on minority classes
    },

    # Specialist model parameters (Random Forest)
    'specialist_model': {
        'n_estimators': 200,
        'max_depth': 8,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'bootstrap': True
    },

    # SMOTE parameters for handling class imbalance
    'smote': {
        'enabled': True,
        'sampling_multiplier': 2.0,  # Multiply minority class samples
        'k_neighbors': 5
    }
}

# Detection thresholds
DETECTION_THRESHOLDS = {
    # Confidence threshold for complete paralysis prediction
    'complete_confidence': 0.7,

    # Thresholds for adjusting Complete predictions with low confidence
    'none_probability': 0.3,  # If None probability exceeds this, downgrade to None
    'partial_probability': 0.25,  # If Partial probability exceeds this, downgrade to Partial

    # Threshold for upgrading None to Partial (missing parameter that's causing the error)
    'upgrade_to_partial': 0.4,  # If Partial probability exceeds this, upgrade to Partial

    # Specialist model invocation thresholds
    'specialist_complete_lower': 0.5,  # Lower bound for Complete confidence
    'specialist_complete_upper': 0.7,  # Upper bound for Complete confidence
    'specialist_partial_threshold': 0.3  # Threshold for considering Partial
}

# Class name mapping
CLASS_NAMES = {
    0: 'None',
    1: 'Partial',
    2: 'Complete'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}
