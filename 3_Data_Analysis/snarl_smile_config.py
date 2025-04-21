# snarl_smile_config.py
# - Focus ONLY on BS action.
# - Coupled AUs: AU10, AU14, AU15 (Needed for Max/Asym features)
# - Switched model to XGBoost.
# - SMOTE ENABLED, scale_pos_weight DISABLED.
# - INCREASED XGBoost Regularization.
# - Feature selection DISABLED.
# - Using FINAL V7 feature set (15 features: Removed side_indicator).
# - Added adjustable DETECTION_THRESHOLD.

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

# --- Define Core Snarl-Smile Actions and AUs ---
SNARL_SMILE_ACTIONS = ['BS'] # <<< Actions: ONLY Big Smile >>>
TRIGGER_AUS = ['AU12_r']           # Primary smile AU (Lip Corner Puller)
# <<< Coupled AUs needed for calculations >>>
COUPLED_AUS = ['AU10_r', 'AU14_r', 'AU15_r'] # Upper Lip Raiser, Dimpler, DAO
# --- END UPDATED AUs ---

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': SNARL_SMILE_ACTIONS,
    'trigger_aus': TRIGGER_AUS,
    'coupled_aus': COUPLED_AUS,
    'use_normalized': True,      # Use normalized for BS features
    'min_value': 0.0001,
    'percent_diff_cap': 200.0 # Needed for PercDiff feature
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    'enabled': False, # Keep False, using a curated feature set now
    'top_n_features': 15, # Expected feature count for V7
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    # --- Model parameters (XGBoost with Regularization) ---
    'model': {
        'type': 'xgboost',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.5,
        'reg_alpha': 0.5,
        'n_estimators': 100,
        'random_state': 42,
    },
    # --- END Model parameters ---

    # --- SMOTE parameters ---
    'smote': {
        'enabled': True,
        'k_neighbors': 5,
        'random_state': 42
    }
    # --- END SMOTE parameters ---
}

# --- Detection Threshold Configuration ---
# Adjust this threshold to trade off precision and recall.
# Lower threshold -> Higher Recall (fewer FNs), Lower Precision (more FPs)
# Higher threshold -> Lower Recall (more FNs), Higher Precision (fewer FPs)
# Default was effectively 0.5. Let's try 0.45 as a starting point.
DETECTION_THRESHOLD = 0.45
# --- End Detection Threshold ---


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