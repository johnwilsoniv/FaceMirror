# ocular_oral_config.py (Mirrors oral_ocular_config.py)
# - Model type is XGBoost
# - SMOTE RE-ENABLED
# - scale_pos_weight REMOVED
# - Feature selection ENABLED

import os

# Base paths
MODEL_PARENT_DIR = 'models/synkinesis' # Parent directory for synkinesis models
MODEL_DIR = os.path.join(MODEL_PARENT_DIR, 'ocular_oral') # Specific path
LOG_DIR = 'logs' # Assuming shared logs

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

# Define Core Ocular-Oral Actions and AUs
OCULAR_ORAL_ACTIONS = ['ET', 'ES', 'RE', 'BL'] # Actions likely to trigger eye/brow movement
TRIGGER_AUS = ['AU01_r', 'AU02_r', 'AU45_r'] # Primary eye/brow movement AUs
COUPLED_AUS = ['AU12_r', 'AU25_r', 'AU14_r'] # Mouth AUs that might co-activate unwantedly

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': OCULAR_ORAL_ACTIONS,
    'trigger_aus': TRIGGER_AUS,
    'coupled_aus': COUPLED_AUS,
    'use_normalized': True,
    'min_value': 0.0001,          # Standard small value for safe division
    'percent_diff_cap': 200.0     # Cap for percentage difference
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    'enabled': True, # <<< Keep ENABLED >>>
    'top_n_features': 40,
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters
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
        'max_depth': 5,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'n_estimators': 150,
        # 'scale_pos_weight': 11, # <<< REMOVED >>>
        'random_state': 42
    },
    # --- End Model parameters ---

    # --- SMOTE parameters - RE-ENABLED ---
    'smote': {
        'enabled': True, # <<< RE-ENABLED >>>
        'k_neighbors': 5, # Will be adjusted dynamically in training script
        'random_state': 42
    }
    # --- END SMOTE parameters ---
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