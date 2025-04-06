# snarl_smile_config.py
# - Focus ONLY on BS action.
# - Updated AUs based on description (AU14, AU15 coupled)
# - Switched model to XGBoost
# - ENABLED SMOTE (removed scale_pos_weight if previously added)
# - Feature selection DISABLED for first run with new features

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

# --- UPDATED Define Core Snarl-Smile Actions and AUs ---
SNARL_SMILE_ACTIONS = ['BS'] # <<< Actions: ONLY Big Smile >>>
TRIGGER_AUS = ['AU12_r']           # Primary smile AU (Lip Corner Puller)
COUPLED_AUS = ['AU14_r', 'AU15_r'] # Unwanted coupled: Dimpler/Buccinator (AU14), DAO (AU15)
# --- END UPDATED AUs ---

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': SNARL_SMILE_ACTIONS, # <<< UPDATED >>>
    'trigger_aus': TRIGGER_AUS,
    'coupled_aus': COUPLED_AUS,
    'use_normalized': True,      # Use normalized for BS features
    'min_value': 0.0001,
    'percent_diff_cap': 200.0
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    'enabled': False, # <<< KEEP False FOR FIRST RUN >>>
    'top_n_features': 15, # Placeholder, adjust after first run if enabling FS
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    # --- UPDATED Model parameters (XGBoost) ---
    'model': {
        'type': 'xgboost',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.1,
        'max_depth': 5, # Keep other params same for now
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'n_estimators': 150,
        'random_state': 42
    },
    # --- END UPDATED Model parameters ---

    # --- SMOTE parameters ---
    'smote': {
        'enabled': True, # <<< ENABLE SMOTE >>>
        'k_neighbors': 5, # Will be adjusted dynamically if needed
        'random_state': 42
    }
    # --- END SMOTE parameters ---
}

# Class name mapping (Binary: Synkinesis vs. None)
CLASS_NAMES = {
    0: 'None',
    1: 'Synkinesis' # Assuming 1 indicates Snarl-Smile Synkinesis
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}