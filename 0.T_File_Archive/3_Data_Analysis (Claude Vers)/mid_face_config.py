"""
Central configuration for mid face paralysis detection.
Contains all thresholds, parameters, and settings in one place.
"""

import os

# Base paths
MODEL_DIR = 'models'
LOG_DIR = 'logs'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model filenames
MODEL_FILENAMES = {
    'base_model': os.path.join(MODEL_DIR, 'mid_face_model.pkl'),
    'base_scaler': os.path.join(MODEL_DIR, 'mid_face_scaler.pkl'),
    'specialist_model': os.path.join(MODEL_DIR, 'mid_face_specialist_model.pkl'),
    'specialist_scaler': os.path.join(MODEL_DIR, 'mid_face_specialist_scaler.pkl'),
    'feature_importance': os.path.join(MODEL_DIR, 'mid_face_feature_importance.csv')
}

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': ['ES', 'ET', 'RE'],  # Use Close Eyes Softly, Close Eyes Tightly, and Raise Eyebrows
    'use_normalized': True,  # Always use normalized values when available
    'au_weights': {  # Relative importance of AUs for midface
        'AU45_r': 0.7,  # Blink (primary)
        'AU07_r': 0.3  # Lid Tightener (secondary)
    },
    'percent_diff_cap': 200.0,  # Cap percent difference values
    'min_value': 0.0001,  # Minimum value to avoid division by zero
    'et_es_ratio_factor': 0.5,  # Weight factor for ET/ES ratio importance

    # AU45 value thresholds based on data analysis
    'au45_thresholds': {
        'complete_max': 1.65,  # Maximum AU45_ET value for Complete classification (more strict)
        'partial_max': 2.2,  # Maximum AU45_ET value for Partial classification
        'borderline_lower': 2.5,  # Lower bound for borderline zone
        'borderline_upper': 3.4  # Upper bound for borderline zone
    },

    # ES/ET ratio thresholds based on data analysis
    'es_et_ratio_thresholds': {
        'complete_max': 0.6,  # Maximum ES/ET ratio for Complete classification (more strict)
        'partial_max': 0.85  # Maximum ES/ET ratio for Partial classification
    },

    # Additional verification thresholds
    'verification': {
        'au07_min_partial': 1.5,  # Minimum AU07_ET value to confirm Partial classification
        'au07_max_complete': 1.8,  # NEW: Maximum AU07_ET for Complete classification
        'asymmetry_min_partial': 40.0,  # Minimum ET asymmetry (%) to confirm Partial
        'asymmetry_min_complete': 70.0,  # Increased: Minimum ET asymmetry (%) to confirm Complete
        'require_all_for_complete': True  # NEW: Require ALL conditions for Complete classification
    }
}

# ML model training parameters
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    # XGBoost model parameters
    'base_model': {
        'objective': 'multi:softprob',
        'num_class': 3,  # None, Partial, Complete
        'learning_rate': 0.05,
        'max_depth': 4,  # Reduced to prevent overfitting on small dataset
        'min_child_weight': 3,  # Increased to prevent overfitting
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.2,  # Increased regularization
        'n_estimators': 200,
        'class_weights': {0: 1.0, 1: 3.0, 2: 5.0}  # Adjusted to prevent Complete over-prediction
    },

    # Specialist model parameters (XGBoost binary classifier)
    'specialist_model': {
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'n_estimators': 200,
        'scale_pos_weight': 1.0  # Reduced to prevent Complete over-prediction
    },

    # SMOTE parameters for handling class imbalance
    'smote': {
        'enabled': True,
        'sampling_strategy': 'auto',  # This will only increase minority classes
        'k_neighbors': 3  # Keep small for small dataset
    }
}

# Detection thresholds - Significantly adjusted based on data analysis
DETECTION_THRESHOLDS = {
    'complete_confidence': 0.85,  # Increased from 0.7 (more strict)
    'none_probability': 0.6,  # Increased from 0.4 (more weight to None class)
    'partial_probability': 0.7,  # Increased from 0.3 (more strict)
    'upgrade_to_partial': 0.8,  # Increased from 0.4 (more strict)
    'specialist_complete_lower': 0.7,  # Increased from 0.6 (more strict)
    'specialist_complete_upper': 0.9,  # Increased from 0.8 (more strict)
    'specialist_partial_threshold': 0.7  # Increased from 0.3 (more strict)
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