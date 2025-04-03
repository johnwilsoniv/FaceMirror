# mid_face_config.py (Added Hyperparameter Tuning Toggle)

import os
from scipy.stats import uniform, randint # Needed here for distributions

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
    'feature_importance': os.path.join(MODEL_DIR, 'mid_face_feature_importance.csv'),
    'feature_list': os.path.join(MODEL_DIR, 'mid_face_features.list')
}

# Define Core Mid Face Actions and AUs
MID_FACE_ACTIONS = ['ES', 'ET', 'BS']
MID_FACE_AUS = ['AU45_r', 'AU07_r', 'AU06_r']

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': MID_FACE_ACTIONS,
    'aus': MID_FACE_AUS,
    'use_normalized': True,
    'percent_diff_cap': 200.0,
    'min_value': 0.0001
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    'enabled': True,
    'top_n_features': 25,
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    # Base model parameters (Defaults if tuning is OFF)
    'base_model': {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'gamma': 0,
        'n_estimators': 100,
        'class_weights': {0: 1.0, 1: 3.0, 2: 3.0} # Moderate weights
    },

    # --- Hyperparameter Tuning Configuration ---
    'hyperparameter_tuning': {
        'enabled': True,           # <<< SET TO True TO TUNE, False TO USE DEFAULTS
        'n_iter': 50,              # Number of parameter settings to sample
        'cv_folds': 3,             # Folds for cross-validation during tuning
        'scoring': 'f1_weighted',  # Metric to optimize during tuning
        # Parameter distributions for RandomizedSearchCV
        'param_distributions': {
            'learning_rate': uniform(0.01, 0.29),
            'max_depth': randint(3, 11),
            'n_estimators': randint(100, 600),
            'min_child_weight': randint(1, 11),
            'gamma': uniform(0, 0.6),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }
    },
    # --- End Tuning Config ---

    # SMOTE parameters - DISABLED
    'smote': {
        'enabled': True,
        'sampling_multiplier': 1.0,
        'k_neighbors': 5
    }
}

# Detection thresholds REMOVED

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