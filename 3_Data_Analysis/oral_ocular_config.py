# ocular_oral_config.py (Mirrors oral_ocular_config.py)
# - Changed model type to XGBoost
# - Updated feature config defaults
# - Disabled feature selection for first run with new features

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
    # 'min_value_for_ratio' removed, use 'min_value'
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    # Set enabled=True AFTER the first run with NEW features generates feature_importance.csv
    'enabled': True, # <<< SET TO False FOR FIRST RUN WITH NEW FEATURES >>>
    'top_n_features': 40, # Example: Keep top 40 (Adjust after first run)
    'importance_file': MODEL_FILENAMES['feature_importance']
}

# ML model training parameters
TRAINING_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,

    # --- UPDATED Model parameters (XGBoost) ---
    'model': {
        'type': 'xgboost', # Changed type
        'objective': 'binary:logistic', # Objective for binary classification
        'eval_metric': 'logloss',      # Evaluation metric for binary logistic
        'learning_rate': 0.1,          # Common default
        'max_depth': 5,                # Reasonable starting depth
        'min_child_weight': 1,         # Default XGBoost
        'subsample': 0.8,              # Introduce some randomness
        'colsample_bytree': 0.8,       # Introduce some randomness
        'gamma': 0,                    # Default XGBoost
        'n_estimators': 150,           # Number of trees
        # scale_pos_weight can be used for imbalance in binary, calculate if needed
        # e.g., scale_pos_weight = count(negative class) / count(positive class)
        # Let's keep SMOTE enabled for now, but this is an alternative.
        'random_state': 42
    },
    # --- END UPDATED Model parameters ---

    # SMOTE parameters - ENABLED (Keep enabled, adjust k_neighbors dynamically in training)
    'smote': {
        'enabled': True,
        # k_neighbors must be <= smallest class count in training data - 1.
        # Will be adjusted dynamically in the training script.
        'k_neighbors': 5,
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