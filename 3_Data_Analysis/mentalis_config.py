# mentalis_config.py
# Config for detecting Mentalis Synkinesis.
# FINAL CONFIG (Pending Error Analysis): Full Features (BS+SE+Context), SMOTE Enabled.
# Added adjustable DETECTION_THRESHOLD.

import os

# Base paths
MODEL_PARENT_DIR = 'models/synkinesis'
MODEL_DIR = os.path.join(MODEL_PARENT_DIR, 'mentalis') # Specific path
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

# --- Define Core Mentalis Synkinesis Actions and AUs ---
MENTALIS_ACTIONS = ['BS', 'SE'] # Actions: Big Smile, Say E
COUPLED_AUS = ['AU17_r']        # Mentalis / Chin Raiser (Target Synkinesis)
CONTEXT_AUS = ['AU12_r', 'AU15_r', 'AU16_r'] # Lip Corner Puller, Lip Corner Depressor, Lower Lip Depressor
# --- END DEFINITIONS ---

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': MENTALIS_ACTIONS,
    'coupled_aus': COUPLED_AUS,
    'context_aus': CONTEXT_AUS,
    'use_normalized': True,      # Use normalized values (baseline subtracted)
    'min_value': 0.0001,
    'percent_diff_cap': 200.0
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    'enabled': False, # Keep disabled for now
    'top_n_features': 15, # Note: Actual features used determined by features.list
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters (XGBoost)
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    'model': {
        'type': 'xgboost',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'n_estimators': 100,
        'random_state': 42,
    },

    'smote': {
        'enabled': True,
        'k_neighbors': 5,
        'random_state': 42
    }
}

# --- Detection Threshold Configuration ---
# Adjust this threshold to trade off precision and recall.
# Start lower due to very low recall seen in training logs (0.20).
DETECTION_THRESHOLD = 0.5
# --- End Detection Threshold ---

# Class name mapping
CLASS_NAMES = {
    0: 'None',
    1: 'Mentalis Synkinesis'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}