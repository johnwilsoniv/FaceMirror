# oral_ocular_config.py
# - Feature selection DISABLED (to retrain/save artifacts for 50 features)
# - Model is XGBoost
# - SMOTE Enabled

import os

# Base paths
MODEL_PARENT_DIR = 'models/synkinesis'
MODEL_DIR = os.path.join(MODEL_PARENT_DIR, 'oral_ocular')
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

# Define Core Oral-Ocular Actions and AUs
ORAL_OCULAR_ACTIONS = ['BS', 'SS', 'SO', 'SE']
TRIGGER_AUS = ['AU12_r', 'AU25_r']
COUPLED_AUS = ['AU06_r', 'AU45_r']

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': ORAL_OCULAR_ACTIONS,
    'trigger_aus': TRIGGER_AUS,
    'coupled_aus': COUPLED_AUS,
    'use_normalized': True,
    'min_value': 0.0001,
    'percent_diff_cap': 200.0
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    'enabled': False, # <<< DISABLE to ensure training uses all 50 features >>>
    'top_n_features': 40, # Value not used when disabled
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
        'max_depth': 5,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'n_estimators': 150,
        'random_state': 42
    },

    'smote': {
        'enabled': True, # Keep enabled
        'k_neighbors': 5,
        'random_state': 42
    }
}

# Class name mapping
CLASS_NAMES = {
    0: 'None',
    1: 'Synkinesis'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}