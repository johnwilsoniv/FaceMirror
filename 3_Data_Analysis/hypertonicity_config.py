# hypertonicity_config.py
# Config for detecting hypertonicity/dysfunctional movement patterns involving resting tone.
# REVERTED: Using BL + BS actions, Normalized values for BS, SMOTE enabled.

import os

# Base paths
MODEL_PARENT_DIR = 'models/synkinesis'
MODEL_DIR = os.path.join(MODEL_PARENT_DIR, 'hypertonicity') # Specific path
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

# --- Define Core Hypertonicity Actions and AUs ---
# Use Baseline (BL) for resting tone and Big Smile (BS) for movement context
HYPERTONICITY_ACTIONS = ['BL', 'BS'] # <<< BOTH BL and BS ACTIONS >>>
# AUs potentially involved in hypertonicity or related smile dysfunction
INTEREST_AUS = ['AU12_r', 'AU14_r', 'AU06_r', 'AU07_r'] # Renamed for clarity
# --- END DEFINITIONS ---

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': HYPERTONICITY_ACTIONS,
    'interest_aus': INTEREST_AUS, # Renamed key
    'use_normalized': True,      # <<< USE NORMALIZED for BS action features >>>
                                 # (Feature extractor will handle using RAW for BL)
    'min_value': 0.0001,
    'percent_diff_cap': 200.0
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    # Keep disabled until we see the initial feature importance
    'enabled': False,
    'top_n_features': 20, # Placeholder, adjust later
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters (XGBoost)
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    # --- Model parameters (XGBoost) ---
    'model': {
        'type': 'xgboost',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.1,
        'max_depth': 4, # Start with 4 again
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        # 'reg_alpha': 0.1, # Keep commented out for now
        # No scale_pos_weight when SMOTE is enabled
    },
    # --- END Model parameters ---

    # --- SMOTE parameters ---
    'smote': {
        'enabled': True, # <<< ENABLED >>>
        'k_neighbors': 5,
        'random_state': 42
    }
    # --- END SMOTE parameters ---
}

# Class name mapping (Binary: Hypertonicity vs. None)
CLASS_NAMES = {
    0: 'None',
    1: 'Hypertonicity' # Or maybe "Dysfunctional Movement Pattern"? Keep as Hypertonicity for now.
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}