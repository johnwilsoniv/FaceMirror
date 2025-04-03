# ocular_oral_training.py (Mirrors oral_ocular_training.py)
# - Changed model to XGBoost
# - Added CalibratedClassifierCV
# - Dynamic k_neighbors for SMOTE
# - Removed early_stopping_rounds from model.fit

import pandas as pd
import numpy as np
import logging
import joblib
import os
import xgboost as xgb # Import XGBoost
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# Removed RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV # Import calibration

# Import specific components for Ocular-Oral
try:
    from ocular_oral_features import prepare_data
    from ocular_oral_config import TRAINING_CONFIG, MODEL_FILENAMES, LOG_DIR, LOGGING_CONFIG, CLASS_NAMES
except ImportError:
    logging.error("Failed to import necessary components from ocular_oral_features or ocular_oral_config.")
    raise SystemExit("Cannot proceed without configuration and feature preparation modules.")

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'ocular_oral_training.log') # Specific log file
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get('level', 'INFO')),
    format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)


def train_and_evaluate():
    """ Trains and evaluates the Ocular-Oral synkinesis model using XGBoost. """
    logger.info("--- Starting Ocular-Oral Synkinesis Model Training (XGBoost) ---")

    # 1. Prepare Data
    try:
        features, targets = prepare_data()
        if features is None or targets is None or features.empty: # Check if features empty
             raise ValueError("Data preparation failed or returned empty features.")
        feature_names = features.columns.tolist() # Get feature names *after* potential selection
        logger.info(f"Data prepared: {features.shape[0]} samples, {features.shape[1]} features.")
        unique_targets, counts = np.unique(targets, return_counts=True)
        target_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(unique_targets, counts)}
        logger.info(f"Target distribution in final prepared data: {target_dist}")
    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True); return

    # 2. Data Splitting
    test_size = TRAINING_CONFIG.get('test_size', 0.25)
    random_state = TRAINING_CONFIG.get('random_state', 42)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state, stratify=targets)
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        train_target_counts = dict(zip(*np.unique(y_train, return_counts=True)))
        train_target_dist = {CLASS_NAMES.get(k, k): v for k, v in train_target_counts.items()}
        logger.info(f"Training target distribution (before SMOTE): {train_target_dist}")
        logger.info(f"Testing target distribution: { {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_test, return_counts=True))} }")

        # --- Determine k_neighbors dynamically and safely ---
        min_class_count_train = min(train_target_counts.values()) if train_target_counts else 0
        smote_config = TRAINING_CONFIG.get('smote', {})
        k_neighbors_config = smote_config.get('k_neighbors', 5)
        # Adjust k_neighbors if it's too large for the minority class
        safe_k_neighbors = min(k_neighbors_config, min_class_count_train - 1) if min_class_count_train > 1 else 1
        if safe_k_neighbors != k_neighbors_config:
             logger.warning(f"Adjusted SMOTE k_neighbors from {k_neighbors_config} to {safe_k_neighbors} due to small minority class size ({min_class_count_train}) in training split.")
        # Store safe k_neighbors for SMOTE instantiation
        smote_k_neighbors = safe_k_neighbors

    except Exception as e:
        logger.error(f"Error during data splitting or k_neighbors calculation: {e}", exc_info=True); return

    # 3. SMOTE & Scaling Block
    scaler = StandardScaler()
    # Convert to DataFrame before SMOTE/Scaling to preserve feature names if needed later
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_train_fit, y_train_fit = X_train_df, y_train # Default to original data

    if smote_config.get('enabled', False) and min_class_count_train > 1: # Only run SMOTE if enabled and minority class has >1 sample
        logger.info("Applying SMOTE to training data...")
        try:
            # Ensure smote_k_neighbors >= 1
            if smote_k_neighbors < 1:
                 logger.warning(f"Cannot apply SMOTE with k_neighbors={smote_k_neighbors}. Skipping.")
            else:
                 smote_params = {k: v for k, v in smote_config.items() if k not in ['enabled', 'k_neighbors']}
                 smote_params['k_neighbors'] = smote_k_neighbors # Use safe value
                 smote = SMOTE(**smote_params)
                 logger.info(f"Instantiating SMOTE with parameters: {smote_params}")
                 X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
                 logger.info(f"SMOTE applied successfully. Original training shape: {X_train_df.shape}, Resampled training shape: {X_train_resampled.shape}")
                 resampled_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_train_resampled, return_counts=True))}
                 logger.info(f"Resampled training target distribution: {resampled_dist}")
                 # Scale the data: Fit on RESAMPLED training data
                 # NOTE: SMOTE returns numpy arrays, convert back to DF if needed, or scale directly
                 X_train_scaled = scaler.fit_transform(X_train_resampled)
                 logger.info("Features scaled after SMOTE (Scaler fit on resampled training data).")
                 # Use resampled data for fitting (can keep as numpy for XGBoost)
                 X_train_fit, y_train_fit = X_train_scaled, y_train_resampled
        except Exception as smote_e:
            logger.error(f"Error during SMOTE application: {smote_e}. Training on original imbalanced data.", exc_info=True)
            # Fallback: Scale original training data
            X_train_scaled = scaler.fit_transform(X_train_df)
            logger.info("Features scaled (Scaler fit on original training data due to SMOTE error).")
            X_train_fit, y_train_fit = X_train_scaled, y_train # Use original scaled data
    else:
        if not smote_config.get('enabled', False): logger.info("SMOTE is disabled.")
        elif min_class_count_train <= 1: logger.warning(f"SMOTE skipped: Minority class in training split has {min_class_count_train} samples (must be >1).")
        # Scale original training data
        X_train_scaled = scaler.fit_transform(X_train_df)
        logger.info("Features scaled (Scaler fit on original training data).")
        X_train_fit, y_train_fit = X_train_scaled, y_train # Use original scaled data

    # Scale the original test set regardless of SMOTE
    X_test_scaled = scaler.transform(X_test_df)

    # 4. Model Training (XGBoost)
    model_config = TRAINING_CONFIG.get('model', {})
    model_type = model_config.get('type', 'xgboost') # Default to xgboost now
    model_params = {k: v for k, v in model_config.items() if k != 'type'}

    # Remove parameters not directly used by XGBClassifier constructor if they exist
    allowed_xgb_params = ['objective', 'eval_metric', 'learning_rate', 'max_depth',
                          'min_child_weight', 'subsample', 'colsample_bytree',
                          'gamma', 'n_estimators', 'random_state', 'scale_pos_weight',
                          'reg_alpha', 'reg_lambda', 'n_jobs', 'booster', 'tree_method'] # Add more if used
    xgb_params_for_init = {k: v for k, v in model_params.items() if k in allowed_xgb_params}

    try:
        if model_type == 'xgboost':
            logger.info(f"Training XGBoostClassifier with params: {xgb_params_for_init}")
            # --- Instantiate XGBoost ---
            model = xgb.XGBClassifier(**xgb_params_for_init)
            # --- End Instantiate XGBoost ---
        else: logger.error(f"Unsupported model type '{model_type}'. Expected 'xgboost'."); return

        # Cross-validation (Optional but recommended)
        cv_folds = TRAINING_CONFIG.get('cv_folds', 5)
        if cv_folds > 1:
            logger.info(f"Performing {cv_folds}-fold CV on data used for final fit...")
            # Note: CV uses the *uncalibrated* model
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            # Use appropriate scoring for binary classification
            cv_scores = cross_val_score(model, X_train_fit, y_train_fit, cv=cv, scoring='f1', n_jobs=-1) # F1 for positive class
            logger.info(f"CV F1 Scores: {cv_scores}")
            logger.info(f"Mean CV F1: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        logger.info("Training final XGBoost model...")
        # Train the base model before calibration
        # --- REMOVED early_stopping_rounds ---
        model.fit(X_train_fit, y_train_fit,
                  # eval_set=[(X_test_scaled, y_test)], # Can optionally remove eval_set too if not stopping early
                  verbose=False)
        # --- END REMOVED early_stopping_rounds ---
        logger.info("Base model training complete.")

        # --- Calibrate Model ---
        logger.info("Calibrating model probabilities using Isotonic Regression...")
        calibrated_model = CalibratedClassifierCV(
            estimator=model, # Pass the trained base model
            method='isotonic',
            cv='prefit' # Use the already trained model
        )
        # Fit calibrator on the same data model was trained on (or a dedicated calibration set if available)
        calibrated_model.fit(X_train_fit, y_train_fit)
        logger.info("Model calibration complete.")
        # --- End Calibrate Model ---

    except Exception as e:
        logger.error(f"Error during model training or calibration: {e}", exc_info=True); return

    # 5. Evaluation (using CALIBRATED model)
    try:
        logger.info("Evaluating CALIBRATED model on the original (unseen) test set...")
        y_pred = calibrated_model.predict(X_test_scaled)
        y_proba = calibrated_model.predict_proba(X_test_scaled) # Probabilities from calibrated model
        report = classification_report(y_test, y_pred, target_names=[CLASS_NAMES.get(i) for i in sorted(CLASS_NAMES.keys())], zero_division=0)
        logger.info("Classification Report (Test Set - Calibrated):\n" + report)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(CLASS_NAMES.keys()))
        logger.info("Confusion Matrix (Test Set - Calibrated):\n" + str(cm))
        # Calculate ROC AUC using probabilities of the positive class (class 1)
        if 1 in CLASS_NAMES and y_proba.shape[1] > 1:
            try:
                # Ensure there are multiple classes present in y_test for ROC AUC
                if len(np.unique(y_test)) > 1:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    logger.info(f"ROC AUC Score (Test Set - Calibrated): {roc_auc:.4f}")
                else:
                    logger.warning(f"Only one class ({np.unique(y_test)[0]}) present in y_test. ROC AUC score is not defined.")
            except Exception as roc_e: logger.warning(f"Could not calculate ROC AUC: {roc_e}")
        else: logger.warning("Cannot calculate ROC AUC - Positive class '1' not defined or insufficient probability columns.")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)

    # 6. Feature Importance (from the base UNCALIBRATED model)
    feature_importance_df = pd.DataFrame()
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Use feature_names captured *after* data prep (which includes selection)
            if len(importances) == len(feature_names):
                 feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
                 logger.info("Top 15 Feature Importances (from base XGBoost model):"); logger.info("\n" + feature_importance_df.head(15).to_string())
            else: logger.warning(f"Importance ({len(importances)}) / Feature name ({len(feature_names)}) length mismatch. Skipping importance report.")
        else: logger.warning("Base XGBoost model does not have feature_importances_ attribute.")
    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}", exc_info=True)

    # 7. Save Artifacts (CALIBRATED Model, Scaler, Importance)
    try:
        model_path = MODEL_FILENAMES.get('model')
        scaler_path = MODEL_FILENAMES.get('scaler')
        importance_path = MODEL_FILENAMES.get('feature_importance')
        feature_list_path = MODEL_FILENAMES.get('feature_list') # Get feature list path from config

        if not all([model_path, scaler_path, importance_path, feature_list_path]):
            raise ValueError("Artifact paths not defined in config.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True) # Ensure directory exists

        # --- Save the CALIBRATED model ---
        joblib.dump(calibrated_model, model_path); logger.info(f"Calibrated model saved: {model_path}")
        # --- End Save Calibrated ---
        joblib.dump(scaler, scaler_path); logger.info(f"Scaler saved: {scaler_path}")
        # Save feature importance if generated
        if not feature_importance_df.empty:
             feature_importance_df.to_csv(importance_path, index=False); logger.info(f"Importance saved: {importance_path}")
        # Save the feature list used for training (redundant with prepare_data, but safe)
        joblib.dump(feature_names, feature_list_path); logger.info(f"Feature names list saved: {feature_list_path}")

    except Exception as e:
        logger.error(f"Error saving artifacts: {e}", exc_info=True)

    logger.info("--- Ocular-Oral Synkinesis Model Training Finished ---")

if __name__ == "__main__":
    train_and_evaluate()