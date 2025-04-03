# snarl_smile_config.py (Mirrors oral_ocular_config.py)

import os

# Base paths
MODEL_PARENT_DIR = 'models/synkinesis'
MODEL_DIR = os.path.join(MODEL_PARENT_DIR, 'snarl_smile') # Specific path
LOG_DIR = 'logs'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model filenames
MODEL_FILENAMES = {
    'model': os.path.join(MODEL_DIR, 'model.pkl'),
    'scaler': os.path.join(MODEL_DIR, 'scaler.pkl'),
    'feature_importance': os.path.join(MODEL_DIR, 'feature_importance.csv'),
    'feature_list': os.path.join(MODEL_DIR, 'features.list')
}

# Define Core Snarl-Smile Actions and AUs
SNARL_SMILE_ACTIONS = ['BS', 'SS'] # Actions likely to trigger smile
TRIGGER_AUS = ['AU12_r'] # Primary smile AU (Lip Corner Puller). Can add AU06_r if needed later.
COUPLED_AUS = ['AU09_r', 'AU10_r', 'AU14_r'] # Nose/Upper Lip/Dimple AUs involved in snarl

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': SNARL_SMILE_ACTIONS,
    'trigger_aus': TRIGGER_AUS,
    'coupled_aus': COUPLED_AUS,
    'use_normalized': True,
    'min_value_for_ratio': 0.05,
    # Weights for weighted score feature (optional, can be added in feature extraction)
    # 'weights': {'AU09_r': 0.4, 'AU10_r': 0.3, 'AU14_r': 0.3}
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    # Set enabled=True AFTER the first run generates feature_importance.csv
    'enabled': False, # <<< SET TO False FOR FIRST RUN >>>
    'top_n_features': 40, # Example (Adjust after first run)
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    # Model parameters
    'model': {
        'type': 'random_forest',
        'n_estimators': 150,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42
    },

    # SMOTE parameters - ENABLED
    'smote': {
        'enabled': True,
        # k_neighbors must be <= smallest class count in training data - 1.
        # Adjust if Snarl-Smile minority count in train split is < 6.
        'k_neighbors': 5, # ASSUMPTION: minority count will be >= 6
        'random_state': 42
    }
}

# Class name mapping (Binary: Synkinesis vs. None)
CLASS_NAMES = {
    0: 'None',
    1: 'Synkinesis'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}