# lower_face_training.py (Simplified with Feature Selection)

import pandas as pd
import numpy as np
import logging
import os
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
# Removed RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Import necessary components from other files
from lower_face_features import prepare_data
from lower_face_config import (
    MODEL_FILENAMES, TRAINING_CONFIG, # DETECTION_THRESHOLDS removed
    LOG_DIR, LOGGING_CONFIG, CLASS_NAMES, MODEL_DIR,
    FEATURE_SELECTION # Import feature selection config
)

# Configure logging (within train_models)

# --- train_base_model function remains the same (no tuning logic here) ---
def train_base_model(X_train, y_train, X_test, y_test, feature_names):
    """ Train base XGBoost model using default or specified parameters. """
    logger = logging.getLogger(__name__)

    # Ensure data is DataFrame with correct columns
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get model parameters from config (default-like values set in config now)
    params = TRAINING_CONFIG.get('base_model', {})

    model = xgb.XGBClassifier(
        objective=params.get('objective', 'multi:softprob'),
        num_class=params.get('num_class', 3),
        learning_rate=params.get('learning_rate', 0.1),
        max_depth=params.get('max_depth', 6),
        min_child_weight=params.get('min_child_weight', 1),
        subsample=params.get('subsample', 1.0),
        colsample_bytree=params.get('colsample_bytree', 1.0),
        gamma=params.get('gamma', 0),
        random_state=TRAINING_CONFIG.get('random_state', 42),
        n_estimators=params.get('n_estimators', 100),
        eval_metric='mlogloss'
    )

    # Create sample weights
    class_weights = params.get('class_weights', {0: 1.0, 1: 1.0, 2: 1.0})
    sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])

    # Train the model
    logger.info("Training final model (using specified/default hyperparameters)...")
    # **** CORRECTED LINE: Removed early_stopping_rounds ****
    model.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False # Keep verbose False to avoid excessive output during eval
        # early_stopping_rounds=20 # REMOVED THIS ARGUMENT AGAIN
    )
    # **** END CORRECTION ****

    # Calibrate model
    logger.info("Calibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(
        estimator=model, method='isotonic', cv='prefit'
    )
    calibrated_model.fit(X_test_scaled, y_test)

    # Evaluate on test set using the CALIBRATED model
    logger.info("Evaluating final calibrated model on test set...")
    y_pred = calibrated_model.predict(X_test_scaled)
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names_list = [CLASS_NAMES.get(i, f"Class_{i}") for i in present_labels]
    report_str = "No labels to report on."
    if target_names_list:
        report_str = classification_report(
            y_test, y_pred, labels=present_labels, target_names=target_names_list, zero_division=0
        )
    logger.info("Final Test Set Classification Report (Calibrated Model):\n" + report_str)
    if present_labels:
        conf_matrix = confusion_matrix(y_test, y_pred, labels=present_labels)
        logger.info("Final Test Set Confusion Matrix:\n" + str(conf_matrix))

    # Feature Importance from the *uncalibrated* model
    if hasattr(model, 'feature_importances_'):
        # Ensure feature names match the number of features the model saw
        model_num_features = model.n_features_in_
        if len(feature_names) == model_num_features:
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info("Top 15 Most Important Features:")
            logger.info("\n" + feature_importance.head(15).to_string(index=False))
        else:
            logger.warning(f"Feature name count ({len(feature_names)}) doesn't match model expected features ({model_num_features}). Skipping importance report.")
            feature_importance = pd.DataFrame(columns=['feature', 'importance'])
    else:
        logger.warning("Base model does not have feature_importances_ attribute.")
        feature_importance = pd.DataFrame(columns=['feature', 'importance'])

    return calibrated_model, scaler, feature_importance

# --- Main Training Function ---
def train_models():
    """
    Simplified training pipeline: Prepares data (with optional feature selection),
    trains base model only (no hyperparameter tuning in this run).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG.get('level', 'INFO')),
        format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, 'lower_face_training.log'), mode='w')
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting SIMPLIFIED lower face training (Feature Selection + Base Model Only)")

    try:
        # Step 1: Prepare data (This now includes potential feature selection)
        logger.info("Preparing data (Feature Selection Enabled = {})...".format(FEATURE_SELECTION.get('enabled', False)))
        results_file='combined_results.csv'
        expert_file='FPRS FP Key.csv'
        if not os.path.exists(results_file) or not os.path.exists(expert_file):
             logger.error(f"Input CSV files not found. Aborting.")
             return
        # *** Crucially, get the feature names *after* potential selection ***
        features, targets = prepare_data(results_file=results_file, expert_file=expert_file)
        if features.empty:
            logger.error("Feature preparation resulted in an empty DataFrame. Aborting.")
            return
        feature_names = features.columns.tolist() # Get column names *after* selection

        # Log class distribution before SMOTE (based on selected features data)
        unique_before, counts_before = np.unique(targets, return_counts=True)
        class_dist_before = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in unique_before], counts_before))
        logger.info(f"Class distribution before SMOTE: {class_dist_before}")
        min_class_size_before = min(counts_before) if len(counts_before) > 0 else 0

        # Step 2: Apply SMOTE if enabled (Using the *selected* features)
        X_resampled, y_resampled = features, targets
        if TRAINING_CONFIG.get('smote', {}).get('enabled', False): # Check if SMOTE is enabled in config
            logger.info("Applying SMOTE for class balancing...")
            k_neighbors_config = TRAINING_CONFIG.get('smote', {}).get('k_neighbors', 5)
            actual_k_neighbors = min(k_neighbors_config, min_class_size_before - 1) if min_class_size_before > 1 else 1
            if actual_k_neighbors < k_neighbors_config: logger.warning(f"Adjusted SMOTE k_neighbors to {actual_k_neighbors}.")
            if min_class_size_before <= 1 or actual_k_neighbors < 1: logger.warning(f"Cannot apply SMOTE. Skipping.")
            else:
                 sampling_strategy = {}
                 majority_class_index = np.argmax(counts_before)
                 majority_size = counts_before[majority_class_index]
                 multiplier = TRAINING_CONFIG.get('smote', {}).get('sampling_multiplier', 1.0)
                 for i, class_label in enumerate(unique_before):
                     if i != majority_class_index: target_size = max(counts_before[i], int(majority_size * multiplier))
                     else: target_size = counts_before[i]
                     sampling_strategy[class_label] = max(counts_before[i], target_size)
                 logger.info(f"Attempting SMOTE with strategy: {sampling_strategy}, k_neighbors={actual_k_neighbors}")
                 try:
                     smote = SMOTE(random_state=TRAINING_CONFIG.get('random_state', 42), k_neighbors=actual_k_neighbors, sampling_strategy=sampling_strategy)
                     X_resampled, y_resampled = smote.fit_resample(features, targets) # Use selected features
                     new_unique, new_counts = np.unique(y_resampled, return_counts=True)
                     new_class_dist = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in new_unique], new_counts))
                     logger.info(f"Class distribution after SMOTE: {new_class_dist}")
                 except Exception as smote_error:
                     logger.error(f"SMOTE failed: {smote_error}. Continuing without SMOTE.")
                     X_resampled, y_resampled = features, targets # Revert
        else:
            logger.info("SMOTE is disabled.") # Log if disabled by config

        features_to_split = X_resampled
        targets_to_split = y_resampled

        # Step 3: Split data for training and testing only
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            features_to_split, targets_to_split,
            test_size=TRAINING_CONFIG.get('test_size', 0.25),
            random_state=TRAINING_CONFIG.get('random_state', 42),
            stratify=targets_to_split
        )
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")

        # Step 4: Train base model (No hyperparameter tuning in this specific run)
        logger.info("Training base model (using config hyperparameters)...")
        # Pass the potentially reduced feature list
        base_model, base_scaler, feature_importance = train_base_model(X_train, y_train, X_test, y_test, feature_names)

        # Step 5: Save models and the potentially reduced feature list
        logger.info("Saving models and feature list...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(base_model, MODEL_FILENAMES['base_model'])
        joblib.dump(base_scaler, MODEL_FILENAMES['base_scaler'])
        # *** Save the list of features ACTUALLY used for training ***
        joblib.dump(feature_names, os.path.join(MODEL_DIR, 'lower_face_features.list'))
        if not feature_importance.empty:
            # Ensure importance df uses the correct feature names if selection happened
            if FEATURE_SELECTION.get('enabled', False):
                 feature_importance = feature_importance[feature_importance['feature'].isin(feature_names)]
            feature_importance.to_csv(MODEL_FILENAMES['feature_importance'], index=False)

        # Step 6: No threshold tuning needed for this simplified run

        logger.info("Simplified model training (with feature selection) complete.")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)


if __name__ == "__main__":
    train_models()