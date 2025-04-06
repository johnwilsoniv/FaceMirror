# snarl_smile_training.py
# - Switched to XGBoost
# - Added Calibration
# - Dynamic SMOTE k_neighbors (if enabled)

import pandas as pd
import numpy as np
import logging
import joblib
import os
import xgboost as xgb # Import XGBoost
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# Removed RandomForestClassifier import
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV # Import calibration

# Import specific components for Snarl-Smile
try:
    from snarl_smile_features import prepare_data
    from snarl_smile_config import TRAINING_CONFIG, MODEL_FILENAMES, LOG_DIR, LOGGING_CONFIG, CLASS_NAMES
except ImportError:
    logging.error("Failed import from snarl_smile_features or snarl_smile_config.")
    raise SystemExit("Cannot proceed.")

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'snarl_smile_training.log') # Specific log file
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get('level', 'INFO')),
    format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'),
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
    force=True
)
logger = logging.getLogger(__name__)


def train_and_evaluate():
    """ Trains and evaluates the Snarl-Smile synkinesis model using XGBoost. """
    logger.info("--- Starting Snarl-Smile Synkinesis Model Training (XGBoost) ---")

    # 1. Prepare Data
    try:
        # prepare_data now handles feature selection based on config internally
        features, targets = prepare_data()
        if features is None or targets is None or features.empty:
            raise ValueError("Data prep failed or returned empty features.")
        feature_names = features.columns.tolist() # Get feature names *after* potential selection
        logger.info(f"Data prepared: {features.shape[0]} samples, {features.shape[1]} features.")
        unique_targets, counts = np.unique(targets, return_counts=True)
        target_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(unique_targets, counts)}
        logger.info(f"Target distribution in final prepared data: {target_dist}")
    except Exception as e: logger.error(f"Data prep error: {e}", exc_info=True); return

    # 2. Data Splitting
    test_size = TRAINING_CONFIG.get('test_size', 0.25)
    random_state = TRAINING_CONFIG.get('random_state', 42)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state, stratify=targets)
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        train_target_counts = dict(zip(*np.unique(y_train, return_counts=True)))
        train_target_dist = {CLASS_NAMES.get(k, k): v for k, v in train_target_counts.items()}
        logger.info(f"Training target distribution (before SMOTE/Weighting): {train_target_dist}")
        logger.info(f"Testing target distribution: { {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_test, return_counts=True))} }")

        # Determine safe k_neighbors for SMOTE if it's enabled
        min_class_count_train = min(train_target_counts.values()) if train_target_counts else 0
        smote_config = TRAINING_CONFIG.get('smote', {})
        smote_k_neighbors = 1 # Default safe value
        if smote_config.get('enabled', False):
            k_neighbors_config = smote_config.get('k_neighbors', 5)
            # k_neighbors must be less than the number of samples in the smallest class
            smote_k_neighbors = min(k_neighbors_config, min_class_count_train - 1) if min_class_count_train > 1 else 1
            if smote_k_neighbors != k_neighbors_config and k_neighbors_config > 1:
                 logger.warning(f"Adjusted SMOTE k_neighbors from {k_neighbors_config} to {smote_k_neighbors} due to small minority class size ({min_class_count_train}).")
            if smote_k_neighbors < 1: # Ensure k_neighbors is at least 1
                smote_k_neighbors = 1
                logger.warning(f"Minority class size ({min_class_count_train}) is <= 1. Setting SMOTE k_neighbors to 1.")


    except Exception as e: logger.error(f"Split/k_neighbor error: {e}", exc_info=True); return

    # 3. SMOTE / Scaling
    scaler = StandardScaler()
    # Use feature names for DataFrame creation
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_train_fit, y_train_fit = X_train_df, y_train # Default if SMOTE fails or disabled

    smote_enabled = smote_config.get('enabled', False)
    # Check if scale_pos_weight is set in the XGBoost config
    scale_pos_weight = TRAINING_CONFIG.get('model', {}).get('scale_pos_weight', None)

    if smote_enabled and scale_pos_weight is not None:
        logger.warning("Both SMOTE ('enabled': True) and XGBoost 'scale_pos_weight' are configured. SMOTE will be applied, and scale_pos_weight might be redundant or have unintended interactions. Consider using only one balancing method.")
        # Typically, you'd use one or the other. If SMOTE is on, don't manually set scale_pos_weight later.

    if smote_enabled:
        if min_class_count_train <= 1 or smote_k_neighbors < 1:
             logger.warning(f"Cannot apply SMOTE (min_class_size={min_class_count_train}, k_neighbors={smote_k_neighbors}). Training on original scaled data.")
             X_train_scaled = scaler.fit_transform(X_train_df)
             X_train_fit, y_train_fit = X_train_scaled, y_train # Use original scaled data
        else:
            logger.info(f"Applying SMOTE to training data with k_neighbors={smote_k_neighbors}...")
            try:
                 # Pass only relevant SMOTE parameters
                 smote_params = {k: v for k, v in smote_config.items() if k not in ['enabled', 'k_neighbors']}
                 smote_params['k_neighbors'] = smote_k_neighbors # Use adjusted value
                 smote = SMOTE(**smote_params)
                 logger.info(f"Instantiating SMOTE with parameters: {smote_params}")
                 # SMOTE works better on scaled data sometimes, but standard practice is fit_resample then scale
                 X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
                 logger.info(f"SMOTE applied. Original shape: {X_train_df.shape}, Resampled shape: {X_train_resampled.shape}")
                 resampled_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_train_resampled, return_counts=True))}
                 logger.info(f"Resampled training target distribution: {resampled_dist}")
                 # Scale the resampled data
                 X_train_scaled = scaler.fit_transform(X_train_resampled) # Fit scaler on resampled data
                 X_train_fit, y_train_fit = X_train_scaled, y_train_resampled # Use resampled data for fitting
            except Exception as smote_e:
                 logger.error(f"Error during SMOTE: {smote_e}. Training on original scaled data.", exc_info=True)
                 X_train_scaled = scaler.fit_transform(X_train_df) # Fit on original
                 X_train_fit, y_train_fit = X_train_scaled, y_train # Fallback to original scaled
    else:
        logger.info("SMOTE is disabled.")
        X_train_scaled = scaler.fit_transform(X_train_df) # Fit scaler on original training data
        X_train_fit, y_train_fit = X_train_scaled, y_train # Use original scaled data

    # Scale the original test set using the *fitted* scaler
    X_test_scaled = scaler.transform(X_test_df)

    # 4. Model Training (XGBoost)
    model_config = TRAINING_CONFIG.get('model', {})
    model_type = model_config.get('type', 'xgboost')
    model_params = {k: v for k, v in model_config.items() if k != 'type'}

    # Prepare XGBoost parameters, filter allowed ones
    allowed_xgb_params = ['objective', 'eval_metric', 'learning_rate', 'max_depth',
                          'min_child_weight', 'subsample', 'colsample_bytree',
                          'gamma', 'n_estimators', 'random_state', 'scale_pos_weight',
                          'reg_alpha', 'reg_lambda', 'n_jobs', 'booster', 'tree_method',
                          'use_label_encoder'] # Add if needed, default changes in XGB >= 1.6
    xgb_params_for_init = {k: v for k, v in model_params.items() if k in allowed_xgb_params}

    # Handle potential scale_pos_weight conflict with SMOTE
    if scale_pos_weight is not None and smote_enabled:
        logger.warning("SMOTE is enabled, ignoring 'scale_pos_weight' from config for XGBoost initialization.")
        if 'scale_pos_weight' in xgb_params_for_init:
            del xgb_params_for_init['scale_pos_weight']
    elif scale_pos_weight is not None and not smote_enabled:
        logger.info(f"SMOTE is disabled. Using scale_pos_weight={scale_pos_weight} for XGBoost.")
        # Keep scale_pos_weight in xgb_params_for_init
    # else: scale_pos_weight is None, proceed without it

    # Add use_label_encoder=False for newer XGBoost versions if objective is binary
    if xgb_params_for_init.get('objective', '').startswith('binary:') and 'use_label_encoder' not in xgb_params_for_init:
        try:
            xgb_version = tuple(map(int, (xgb.__version__.split('.'))))
            if xgb_version >= (1, 6, 0):
                # logger.debug("Setting use_label_encoder=False for XGBoost >= 1.6")
                xgb_params_for_init['use_label_encoder'] = False # Recommended for XGB 1.6+
        except: pass # Ignore version check errors

    try:
        if model_type == 'xgboost':
            logger.info(f"Training XGBoostClassifier with effective params: {xgb_params_for_init}")
            # Base model for feature importance and calibration
            base_model = xgb.XGBClassifier(**xgb_params_for_init)
        else: logger.error(f"Unsupported model type '{model_type}' in config. Expected 'xgboost'."); return

        cv_folds = TRAINING_CONFIG.get('cv_folds', 5)
        if cv_folds > 1:
            logger.info(f"Performing {cv_folds}-fold Cross-Validation...")
            # CV uses the base model config on the data prepared for final fit (could be SMOTEd)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            # Use F1 score, suitable for imbalanced data
            cv_scores = cross_val_score(base_model, X_train_fit, y_train_fit, cv=cv, scoring='f1', n_jobs=-1)
            logger.info(f"CV F1 Scores: {cv_scores}")
            logger.info(f"Mean CV F1: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        logger.info("Training final base XGBoost model...")
        # Train the base model on the prepared training data (potentially SMOTEd and scaled)
        # Note: If using early stopping, need an eval set. Here we train fully.
        base_model.fit(X_train_fit, y_train_fit, verbose=False)
        logger.info("Base model training complete.")

        # Calibrate Model using the trained base model
        logger.info("Calibrating model probabilities using Isotonic Regression...")
        # Use cv='prefit' because base_model is already trained
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
        # Fit the calibrator on the same data the base model was trained on
        calibrated_model.fit(X_train_fit, y_train_fit)
        logger.info("Model calibration complete.")

    except Exception as e: logger.error(f"Error during model training or calibration: {e}", exc_info=True); return

    # 5. Evaluation (using the CALIBRATED model)
    try:
        logger.info("Evaluating CALIBRATED model on the original (unseen) scaled test set...")
        y_pred = calibrated_model.predict(X_test_scaled)
        y_proba = calibrated_model.predict_proba(X_test_scaled) # Probabilities from calibrated model

        report = classification_report(y_test, y_pred, target_names=[CLASS_NAMES.get(i) for i in sorted(CLASS_NAMES.keys())], zero_division=0)
        logger.info("Classification Report (Test Set - Calibrated):\n" + report)

        cm = confusion_matrix(y_test, y_pred, labels=sorted(CLASS_NAMES.keys()))
        logger.info("Confusion Matrix (Test Set - Calibrated):\n" + str(cm))

        # Calculate ROC AUC using probabilities of the positive class (class 1)
        if 1 in CLASS_NAMES and y_proba.shape[1] > 1:
            try:
                # Ensure there are samples of both classes in the test set for AUC
                if len(np.unique(y_test)) > 1:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1]) # Use probability of class 1
                    logger.info(f"ROC AUC Score (Test Set - Calibrated): {roc_auc:.4f}")
                else:
                    logger.warning(f"Only one class ({np.unique(y_test)[0]}) present in y_test. ROC AUC score is not defined.")
            except ValueError as roc_ve:
                logger.warning(f"Could not calculate ROC AUC score (ValueError): {roc_ve}") # Handles cases like only one class predicted
            except Exception as roc_e: logger.warning(f"Could not calculate ROC AUC score (Other Error): {roc_e}")
        else: logger.warning("Cannot calculate ROC AUC - Positive class '1' not defined or insufficient probability columns.")
    except Exception as e: logger.error(f"Error during model evaluation: {e}", exc_info=True)

    # 6. Feature Importance (from the base UNCALIBRATED XGBoost model)
    feature_importance_df = pd.DataFrame()
    try:
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
            if len(importances) == len(feature_names):
                 feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
                 logger.info("Top 15 Feature Importances (from base XGBoost model):"); logger.info("\n" + feature_importance_df.head(15).to_string())
            else: logger.warning(f"SnSm Importance ({len(importances)}) / Feature name ({len(feature_names)}) length mismatch. Cannot reliably map importances.")
        else: logger.warning("Base XGBoost model does not provide feature_importances_ attribute.")
    except Exception as e: logger.error(f"Error extracting SnSm feature importance: {e}", exc_info=True)

    # 7. Save Artifacts
    try:
        model_path = MODEL_FILENAMES.get('model'); scaler_path = MODEL_FILENAMES.get('scaler')
        importance_path = MODEL_FILENAMES.get('feature_importance'); feature_list_path = MODEL_FILENAMES.get('feature_list')

        if not all([model_path, scaler_path, importance_path, feature_list_path]):
            raise ValueError("SnSm Artifact paths missing in config.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the CALIBRATED model for detection
        joblib.dump(calibrated_model, model_path); logger.info(f"Calibrated SnSm model saved: {model_path}")
        # Save the scaler fitted on the (potentially SMOTEd) training data
        joblib.dump(scaler, scaler_path); logger.info(f"SnSm Scaler saved: {scaler_path}")
        # Save feature importance if generated
        if not feature_importance_df.empty:
             feature_importance_df.to_csv(importance_path, index=False); logger.info(f"SnSm Importance saved: {importance_path}")
        else: logger.warning(f"SnSm Feature importance was not generated or saved ({importance_path}).")
        # Save the list of features USED for training (could be selected or all)
        # This was already saved in prepare_data, but saving again here ensures it matches the trained model exactly.
        joblib.dump(feature_names, feature_list_path); logger.info(f"SnSm Feature names list saved (used for training): {feature_list_path}")

    except Exception as e: logger.error(f"Error saving SnSm artifacts: {e}", exc_info=True)

    logger.info("--- Snarl-Smile Synkinesis Model Training Finished ---")

if __name__ == "__main__":
    train_and_evaluate()