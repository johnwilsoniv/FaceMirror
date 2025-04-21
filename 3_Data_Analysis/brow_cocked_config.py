# brow_cocked_config.py
# Config for detecting Brow Cocked phenomenon (resting asymmetry + dynamic change).

import os

# Base paths
MODEL_PARENT_DIR = 'models/synkinesis'
MODEL_DIR = os.path.join(MODEL_PARENT_DIR, 'brow_cocked') # Specific path
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

# --- Define Core Brow Cocked Actions and AUs ---
BROW_COCKED_ACTIONS = ['BL', 'ET'] # Baseline (Raw), Eyes Tight (Normalized Change)
INTEREST_AUS = ['AU01_r', 'AU02_r'] # Brow Raisers
CONTEXT_AUS = ['AU07_r']           # Lid Tightener (Eye Closure Context)
ALL_RELEVANT_AUS = INTEREST_AUS + CONTEXT_AUS
# --- END DEFINITIONS ---

# Feature extraction parameters
FEATURE_CONFIG = {
    'actions': BROW_COCKED_ACTIONS,
    'interest_aus': INTEREST_AUS,
    'context_aus': CONTEXT_AUS,
    # 'use_normalized': True, # This is ambiguous here; logic is handled in feature extractor
                              # BL features are RAW, ET features are NORMALIZED
    'min_value': 0.0001,      # For ratio/perc_diff calculations
    'percent_diff_cap': 200.0 # For perc_diff calculations
}

# Feature Selection Configuration
FEATURE_SELECTION = {
    'enabled': False, # Keep False for initial run
    'top_n_features': 15, # Target count for V1 feature set
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
        'max_depth': 4, # Start reasonably shallow
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1, # Default regularization
        'n_estimators': 100,
        'random_state': 42,
        # No scale_pos_weight when SMOTE is enabled
    },
    # --- END Model parameters ---

    # --- SMOTE parameters ---
    'smote': {
        'enabled': True, # Enable SMOTE as hypertonicity can be less frequent
        'k_neighbors': 5,
        'random_state': 42
    }
    # --- END SMOTE parameters ---
}

# --- Detection Threshold Configuration ---
# Start with standard 0.5, adjust after evaluation if needed
DETECTION_THRESHOLD = 0.25
# --- End Detection Threshold ---

# Class name mapping (Binary: Brow Cocked vs. None)
CLASS_NAMES = {
    0: 'None',
    1: 'Brow Cocked'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}