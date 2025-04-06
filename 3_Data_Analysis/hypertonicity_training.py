# hypertonicity_training.py
# Trains an XGBoost model for Hypertonicity detection.

import pandas as pd
import numpy as np
import logging
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

# Import specific components for Hypertonicity
try:
    from hypertonicity_features import prepare_data
    from hypertonicity_config import TRAINING_CONFIG, MODEL_FILENAMES, LOG_DIR, LOGGING_CONFIG, CLASS_NAMES
except ImportError:
    logging.error("Failed import from hypertonicity_features or hypertonicity_config.")
    raise SystemExit("Cannot proceed.")

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'hypertonicity_training.log') # Specific log file
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get('level', 'INFO')),
    format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'),
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
    force=True
)
logger = logging.getLogger(__name__)


def train_and_evaluate():
    """ Trains and evaluates the Hypertonicity detection model using XGBoost. """
    logger.info("--- Starting Hypertonicity Detection Model Training (XGBoost) ---")

    # 1. Prepare Data
    try:
        features, targets = prepare_data()
        if features is None or targets is None or features.empty:
            raise ValueError("Data prep failed or returned empty features/targets.")
        feature_names = features.columns.tolist()
        logger.info(f"Data prepared: {features.shape[0]} samples, {features.shape[1]} features.")
        unique_targets, counts = np.unique(targets, return_counts=True)
        target_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(unique_targets, counts)}
        logger.info(f"Target distribution in final prepared data: {target_dist}")
        if len(unique_targets) < 2:
             logger.warning("Only one class present in the target data. Model training might fail or be meaningless.")
             # Optionally exit early if only one class: raise SystemExit("Cannot train with only one class.")
    except Exception as e: logger.error(f"Data prep error: {e}", exc_info=True); return

    # 2. Data Splitting
    test_size = TRAINING_CONFIG.get('test_size', 0.25)
    random_state = TRAINING_CONFIG.get('random_state', 42)
    # Ensure stratification is possible
    if len(np.unique(targets)) < 2:
        logger.warning("Cannot stratify split with only one class. Performing standard split.")
        stratify_param = None
    else:
        stratify_param = targets

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state, stratify=stratify_param)
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        train_target_counts = dict(zip(*np.unique(y_train, return_counts=True)))
        train_target_dist = {CLASS_NAMES.get(k, k): v for k, v in train_target_counts.items()}
        logger.info(f"Training target distribution (before SMOTE/Weighting): {train_target_dist}")
        test_target_dist = {CLASS_NAMES.get(k,k): v for k,v in zip(*np.unique(y_test, return_counts=True))}
        logger.info(f"Testing target distribution: { test_target_dist }")

        min_class_count_train = min(train_target_counts.values()) if train_target_counts else 0
        smote_config = TRAINING_CONFIG.get('smote', {})
        smote_k_neighbors = 1
        if smote_config.get('enabled', False):
            k_neighbors_config = smote_config.get('k_neighbors', 5)
            smote_k_neighbors = min(k_neighbors_config, min_class_count_train - 1) if min_class_count_train > 1 else 1
            if smote_k_neighbors != k_neighbors_config and k_neighbors_config > 1: logger.warning(f"Adjusted SMOTE k_neighbors to {smote_k_neighbors} from {k_neighbors_config}.")
            if smote_k_neighbors < 1: smote_k_neighbors = 1; logger.warning("Set SMOTE k_neighbors to 1.")

    except Exception as e: logger.error(f"Split/k_neighbor error: {e}", exc_info=True); return

    # 3. SMOTE / Scaling
    scaler = StandardScaler()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_train_fit, y_train_fit = X_train_df, y_train

    smote_enabled = smote_config.get('enabled', False)
    scale_pos_weight = TRAINING_CONFIG.get('model', {}).get('scale_pos_weight', None)

    if smote_enabled and scale_pos_weight is not None: logger.warning("Both SMOTE and scale_pos_weight configured. Prioritizing SMOTE.")

    if smote_enabled:
        # Check if SMOTE is applicable (more than 1 class and sufficient samples)
        if len(train_target_counts) < 2:
             logger.warning("Cannot apply SMOTE with only one class in the training data. Training on original.")
             X_train_scaled = scaler.fit_transform(X_train_df)
             X_train_fit, y_train_fit = X_train_scaled, y_train
        elif min_class_count_train <= 1 or smote_k_neighbors < 1:
             logger.warning(f"Cannot apply SMOTE (min_class={min_class_count_train}, k={smote_k_neighbors}). Training on original.")
             X_train_scaled = scaler.fit_transform(X_train_df)
             X_train_fit, y_train_fit = X_train_scaled, y_train
        else:
            logger.info(f"Applying SMOTE with k_neighbors={smote_k_neighbors}...")
            try:
                 smote_params = {k: v for k, v in smote_config.items() if k not in ['enabled', 'k_neighbors']}
                 smote_params['k_neighbors'] = smote_k_neighbors
                 smote = SMOTE(**smote_params)
                 logger.info(f"SMOTE params: {smote_params}")
                 X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
                 logger.info(f"SMOTE applied. Original: {X_train_df.shape}, Resampled: {X_train_resampled.shape}")
                 resampled_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_train_resampled, return_counts=True))}
                 logger.info(f"Resampled train distribution: {resampled_dist}")
                 X_train_scaled = scaler.fit_transform(X_train_resampled)
                 X_train_fit, y_train_fit = X_train_scaled, y_train_resampled
            except Exception as smote_e:
                 logger.error(f"SMOTE error: {smote_e}. Training on original.", exc_info=True)
                 X_train_scaled = scaler.fit_transform(X_train_df)
                 X_train_fit, y_train_fit = X_train_scaled, y_train
    else:
        logger.info("SMOTE disabled.")
        X_train_scaled = scaler.fit_transform(X_train_df)
        X_train_fit, y_train_fit = X_train_scaled, y_train

    X_test_scaled = scaler.transform(X_test_df)

    # 4. Model Training (XGBoost)
    model_config = TRAINING_CONFIG.get('model', {})
    model_type = model_config.get('type', 'xgboost')
    model_params = {k: v for k, v in model_config.items() if k != 'type'}

    allowed_xgb_params = ['objective', 'eval_metric', 'learning_rate', 'max_depth',
                          'min_child_weight', 'subsample', 'colsample_bytree',
                          'gamma', 'n_estimators', 'random_state', 'scale_pos_weight',
                          'reg_alpha', 'reg_lambda', 'n_jobs', 'booster', 'tree_method', 'use_label_encoder']
    xgb_params_for_init = {k: v for k, v in model_params.items() if k in allowed_xgb_params}

    if scale_pos_weight is not None and smote_enabled: logger.warning("Prioritizing SMOTE. scale_pos_weight from config ignored."); del xgb_params_for_init['scale_pos_weight']
    elif scale_pos_weight is not None and not smote_enabled: logger.info(f"Using scale_pos_weight={scale_pos_weight} for XGBoost.") # Keep it
    # Add use_label_encoder=False for newer XGBoost versions
    if xgb_params_for_init.get('objective', '').startswith('binary:') and 'use_label_encoder' not in xgb_params_for_init:
        try:
            xgb_version = tuple(map(int, (xgb.__version__.split('.'))));
            if xgb_version >= (1, 6, 0): xgb_params_for_init['use_label_encoder'] = False
        except: pass # Ignore version check errors


    try:
        if model_type == 'xgboost':
            logger.info(f"Training XGBoostClassifier: {xgb_params_for_init}")
            base_model = xgb.XGBClassifier(**xgb_params_for_init)
        else: logger.error(f"Unsupported model type '{model_type}'."); return

        cv_folds = TRAINING_CONFIG.get('cv_folds', 5)
        # Ensure cross-validation is possible (more than 1 class in y_train_fit)
        if cv_folds > 1 and len(np.unique(y_train_fit)) > 1 :
            logger.info(f"Performing {cv_folds}-fold CV...")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(base_model, X_train_fit, y_train_fit, cv=cv, scoring='f1', n_jobs=-1)
            logger.info(f"CV F1 Scores: {cv_scores}")
            logger.info(f"Mean CV F1: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        elif cv_folds > 1:
            logger.warning(f"Skipping CV: Only one class present in the training data (after potential SMOTE).")


        logger.info("Training final model...")
        base_model.fit(X_train_fit, y_train_fit, verbose=False)
        logger.info("Base training complete.")

        logger.info("Calibrating model probabilities...")
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_train_fit, y_train_fit)
        logger.info("Calibration complete.")

    except Exception as e: logger.error(f"Training/Calibration error: {e}", exc_info=True); return

    # 5. Evaluation
    try:
        logger.info("Evaluating calibrated model on test set...")
        y_pred = calibrated_model.predict(X_test_scaled); y_proba = calibrated_model.predict_proba(X_test_scaled)
        # Ensure labels for report match the actual unique classes present in y_test
        test_labels = sorted(np.unique(y_test))
        target_names_test = [CLASS_NAMES.get(i) for i in test_labels]

        report = classification_report(y_test, y_pred, labels=test_labels, target_names=target_names_test, zero_division=0)
        logger.info("Report (Test Set - Calibrated):\n" + report)
        # Use all potential labels for confusion matrix structure if desired, or just test_labels
        cm_labels = sorted(CLASS_NAMES.keys()) # Use predefined 0, 1
        cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
        logger.info("CM (Test Set - Calibrated):\n" + str(cm))

        if 1 in CLASS_NAMES and y_proba.shape[1] > 1 and len(test_labels) > 1:
            try:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1]); logger.info(f"ROC AUC (Test Set - Calibrated): {roc_auc:.4f}")
            except ValueError as roc_ve: logger.warning(f"Could not calculate ROC AUC (ValueError): {roc_ve}") # Handles if only one class predicted/present
            except Exception as roc_e: logger.warning(f"ROC AUC calc failed: {roc_e}")
        else: logger.warning("Cannot calculate ROC AUC (only one class in test set or missing probabilities).")
    except Exception as e: logger.error(f"Evaluation error: {e}", exc_info=True)

    # 6. Feature Importance
    feature_importance_df = pd.DataFrame()
    try:
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
            if len(importances) == len(feature_names):
                 feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
                 logger.info("Top 15 Importances:"); logger.info("\n" + feature_importance_df.head(15).to_string())
            else: logger.warning(f"HyperT Importance ({len(importances)}) / Feature ({len(feature_names)}) length mismatch.")
        else: logger.warning("Base XGBoost model lacks feature_importances_.")
    except Exception as e: logger.error(f"Importance error: {e}", exc_info=True)

    # 7. Save Artifacts
    try:
        model_path = MODEL_FILENAMES.get('model'); scaler_path = MODEL_FILENAMES.get('scaler')
        importance_path = MODEL_FILENAMES.get('feature_importance'); feature_list_path = MODEL_FILENAMES.get('feature_list')
        if not all([model_path, scaler_path, importance_path, feature_list_path]): raise ValueError("HyperT Artifact paths missing.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(calibrated_model, model_path); logger.info(f"Calibrated model saved: {model_path}")
        joblib.dump(scaler, scaler_path); logger.info(f"Scaler saved: {scaler_path}")
        if not feature_importance_df.empty: feature_importance_df.to_csv(importance_path, index=False); logger.info(f"Importance saved: {importance_path}")
        else: logger.warning(f"Importance df empty, not saving file {importance_path}")
        joblib.dump(feature_names, feature_list_path); logger.info(f"Feature list saved: {feature_list_path}")
    except Exception as e: logger.error(f"Saving error: {e}", exc_info=True)

    logger.info("--- Hypertonicity Detection Model Training Finished ---")

if __name__ == "__main__":
    train_and_evaluate()