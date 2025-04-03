# upper_face_training.py (Added Tuning Logic + Calibration)

import pandas as pd
import numpy as np
import logging
import os
import joblib
import json
# Added RandomizedSearchCV and distributions
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV # Added KFold
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Import necessary components from UPPER FACE files
from upper_face_features import prepare_data
from upper_face_config import (
    MODEL_FILENAMES, TRAINING_CONFIG, # DETECTION_THRESHOLDS removed
    LOG_DIR, LOGGING_CONFIG, CLASS_NAMES, MODEL_DIR,
    FEATURE_SELECTION # Import feature selection config
)

# Configure logging (within train_models)

# --- train_base_model function with Tuning Logic and Calibration ---
def train_base_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Optionally tune hyperparameters using RandomizedSearchCV, trains the base
    XGBoost model, and then calibrates it using CalibratedClassifierCV.
    """
    logger = logging.getLogger(__name__)

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_params_config = TRAINING_CONFIG.get('base_model', {})
    class_weights = base_params_config.get('class_weights', {0: 1.0, 1: 1.0, 2: 1.0})
    unique_train_labels = np.unique(y_train)
    if len(unique_train_labels) == 1:
         logger.warning("Only one class present in y_train. Sample weights will be uniform.")
         sample_weights = np.ones(len(y_train))
    else: sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])
    fit_params = {'sample_weight': sample_weights}

    # --- Conditional Hyperparameter Tuning ---
    tuning_config = TRAINING_CONFIG.get('hyperparameter_tuning', {})
    tune_enabled = tuning_config.get('enabled', False)

    final_model_untuned = None

    if tune_enabled:
        logger.info("Hyperparameter tuning ENABLED.")
        param_distributions = tuning_config.get('param_distributions', {})
        if not param_distributions: logger.error("Tuning enabled but 'param_distributions' missing!"); return None, None, None

        xgb_base = xgb.XGBClassifier(
            objective=base_params_config.get('objective', 'multi:softprob'),
            num_class=base_params_config.get('num_class', 3),
            random_state=TRAINING_CONFIG.get('random_state', 42),
            eval_metric='mlogloss')

        n_iter_search = tuning_config.get('n_iter', 50); cv_folds_tuning = tuning_config.get('cv_folds', 3)
        scoring_metric = tuning_config.get('scoring', 'f1_weighted')
        logger.info(f"Running RandomizedSearchCV with n_iter={n_iter_search}, cv={cv_folds_tuning}, scoring='{scoring_metric}'...")

        cv_strategy = None
        if len(np.unique(y_train)) > 1:
             cv_strategy = StratifiedKFold(n_splits=cv_folds_tuning, shuffle=True, random_state=TRAINING_CONFIG.get('random_state', 42)); logger.info("Using StratifiedKFold.")
        else: logger.warning("Only one class in training data, using standard KFold."); cv_strategy = KFold(n_splits=cv_folds_tuning, shuffle=True, random_state=TRAINING_CONFIG.get('random_state', 42))

        random_search = RandomizedSearchCV(
            estimator=xgb_base, param_distributions=param_distributions, n_iter=n_iter_search,
            scoring=scoring_metric, cv=cv_strategy, n_jobs=-1,
            random_state=TRAINING_CONFIG.get('random_state', 42), verbose=1, error_score='raise')

        try:
            random_search.fit(X_train_scaled, y_train, **fit_params)
            logger.info(f"Best parameters found: {random_search.best_params_}")
            logger.info(f"Best cross-validation score ({scoring_metric}): {random_search.best_score_:.4f}")
            final_model_untuned = random_search.best_estimator_
        except ValueError as e:
             logger.error(f"Error during RandomizedSearchCV fit: {e}"); return None, scaler, pd.DataFrame(columns=['feature', 'importance'])

    else: # --- Train directly using config parameters ---
        logger.info("Hyperparameter tuning DISABLED. Training with parameters from config.")
        final_model_untuned = xgb.XGBClassifier(
            objective=base_params_config.get('objective', 'multi:softprob'), num_class=base_params_config.get('num_class', 3),
            learning_rate=base_params_config.get('learning_rate', 0.1), max_depth=base_params_config.get('max_depth', 6),
            min_child_weight=base_params_config.get('min_child_weight', 1), subsample=base_params_config.get('subsample', 1.0),
            colsample_bytree=base_params_config.get('colsample_bytree', 1.0), gamma=base_params_config.get('gamma', 0),
            random_state=TRAINING_CONFIG.get('random_state', 42), n_estimators=base_params_config.get('n_estimators', 100),
            eval_metric='mlogloss')
        try:
            eval_set_list = []; use_early_stopping = False
            if len(np.unique(y_test)) > 1: eval_set_list = [(X_test_scaled, y_test)]; use_early_stopping = True; logger.info("Using evaluation set.")
            else: logger.warning("Only one class in test data, cannot use eval set.")
            fit_kwargs = {}
            if use_early_stopping: fit_kwargs['eval_set'] = eval_set_list; fit_kwargs['early_stopping_rounds'] = 20; fit_kwargs['verbose'] = False
            final_model_untuned.fit(X_train_scaled, y_train, sample_weight=sample_weights, **fit_kwargs)
        except ValueError as e: logger.error(f"Error during direct XGBoost fit: {e}"); return None, scaler, pd.DataFrame(columns=['feature', 'importance'])

    if final_model_untuned is None: logger.error("Base model not trained."); return None, scaler, pd.DataFrame(columns=['feature', 'importance'])

    # --- Calibration ---
    logger.info("Calibrating the final model...")
    calibrated_model_iso = CalibratedClassifierCV(estimator=final_model_untuned, method='isotonic', cv='prefit')
    calibrated_model_sig = CalibratedClassifierCV(estimator=final_model_untuned, method='sigmoid', cv='prefit')
    calibrated_model = None
    try:
        calibrated_model_iso.fit(X_test_scaled, y_test); calibrated_model = calibrated_model_iso; logger.info("Isotonic calibration successful.")
    except ValueError as cal_e:
        logger.warning(f"Isotonic calibration failed: {cal_e}. Falling back to sigmoid.")
        try: calibrated_model_sig.fit(X_test_scaled, y_test); calibrated_model = calibrated_model_sig; logger.info("Sigmoid calibration successful.")
        except ValueError as sig_e: logger.error(f"Sigmoid calibration also failed: {sig_e}. Using uncalibrated model."); calibrated_model = final_model_untuned
    except Exception as e: logger.error(f"Unexpected error during calibration: {e}. Using uncalibrated model.", exc_info=True); calibrated_model = final_model_untuned
    model_to_evaluate = calibrated_model
    model_name_for_log = "Calibrated" if calibrated_model != final_model_untuned else "Uncalibrated"
    # --- End Calibration ---

    # --- Final Evaluation ---
    logger.info(f"Evaluating final {model_name_for_log} model on test set...")
    try:
        y_pred = model_to_evaluate.predict(X_test_scaled)
        present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        target_names_list = [CLASS_NAMES.get(i, f"Class_{i}") for i in present_labels]
        report_str = "No labels."
        if present_labels:
            try: report_str = classification_report(y_test, y_pred, labels=present_labels, target_names=target_names_list, zero_division=0)
            except ValueError as report_err: report_str = f"Report error: {report_err}"
        logger.info(f"Final Test Set Classification Report ({model_name_for_log} Model):\n" + report_str)
        if present_labels and len(present_labels) > 1:
            try: conf_matrix = confusion_matrix(y_test, y_pred, labels=present_labels); logger.info("Final Test Set Confusion Matrix:\n" + str(conf_matrix))
            except ValueError as cm_err: logger.error(f"CM error: {cm_err}")
        elif present_labels: logger.info(f"Only one class ({CLASS_NAMES.get(present_labels[0], present_labels[0])}) present. CM N/A.")
        else: logger.info("No labels found.")
    except Exception as eval_e: logger.error(f"Eval error: {eval_e}", exc_info=True); return model_to_evaluate, scaler, pd.DataFrame(columns=['feature', 'importance'])

    # --- Feature Importance ---
    feature_importance = pd.DataFrame(columns=['feature', 'importance'])
    base_estimator_for_fi = final_model_untuned
    if hasattr(base_estimator_for_fi, 'feature_importances_'):
        try:
            model_num_features = base_estimator_for_fi.n_features_in_
            if len(feature_names) == model_num_features:
                feature_importance = pd.DataFrame({'feature': feature_names, 'importance': base_estimator_for_fi.feature_importances_}).sort_values('importance', ascending=False)
                logger.info("Top 15 Features (base model):\n" + feature_importance.head(15).to_string(index=False))
            else: logger.warning(f"Feat name count {len(feature_names)} != model expected {model_num_features}. Skip FI.")
        except Exception as fi_e: logger.error(f"FI error: {fi_e}", exc_info=True)
    else: logger.warning("Base model lacks feature_importances_.")

    return model_to_evaluate, scaler, feature_importance


# --- Main Training Function ---
def train_models():
    """ Training pipeline for UPPER FACE. """
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=getattr(logging, LOGGING_CONFIG.get('level', 'INFO')), format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'), handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, 'upper_face_training.log'), mode='w')], force=True)
    logger = logging.getLogger(__name__)
    tuning_enabled_log = TRAINING_CONFIG.get('hyperparameter_tuning',{}).get('enabled', False)
    feat_sel_enabled_log = FEATURE_SELECTION.get('enabled', False)
    smote_enabled_log = TRAINING_CONFIG.get('smote',{}).get('enabled', False)
    logger.info(f"Starting UPPER FACE training (Tuning: {tuning_enabled_log}, FeatSel: {feat_sel_enabled_log}, SMOTE: {smote_enabled_log})")

    try:
        logger.info("Preparing upper face data...")
        results_file='combined_results.csv'; expert_file='FPRS FP Key.csv'
        if not os.path.exists(results_file) or not os.path.exists(expert_file): logger.error("Input CSV files not found."); return
        features, targets = prepare_data(results_file=results_file, expert_file=expert_file)
        if features is None or features.empty: logger.error("Feature prep failed."); return
        feature_names = features.columns.tolist(); logger.info(f"Using {len(feature_names)} features for training.")

        unique_before, counts_before = np.unique(targets, return_counts=True)
        if not unique_before.size: logger.error("No target labels found."); return
        class_dist_before = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in unique_before], counts_before))
        logger.info(f"Upper face class distribution before SMOTE: {class_dist_before}")
        min_class_size_before = min(counts_before) if len(counts_before) > 0 else 0

        X_resampled, y_resampled = features, targets
        smote_config = TRAINING_CONFIG.get('smote', {})
        if smote_config.get('enabled', False):
            logger.info("Applying SMOTE...")
            k_neighbors_config = smote_config.get('k_neighbors', 5)
            actual_k_neighbors = k_neighbors_config
            if len(unique_before) > 1 and min_class_size_before <= k_neighbors_config:
                actual_k_neighbors = max(1, min_class_size_before - 1); logger.warning(f"Adjusted SMOTE k_neighbors to {actual_k_neighbors}.")
            if len(unique_before) <= 1 or actual_k_neighbors < 1 : logger.warning(f"Cannot apply SMOTE. Skipping.")
            else:
                 sampling_strategy = 'auto' # Or customize as needed
                 logger.info(f"Attempting SMOTE with strategy: {sampling_strategy}, k_neighbors={actual_k_neighbors}")
                 try:
                     smote = SMOTE(random_state=TRAINING_CONFIG.get('random_state', 42), k_neighbors=actual_k_neighbors, sampling_strategy=sampling_strategy)
                     X_resampled, y_resampled = smote.fit_resample(features, targets)
                     new_unique, new_counts = np.unique(y_resampled, return_counts=True); new_class_dist = dict(zip([CLASS_NAMES.get(i, f"Class_{i}") for i in new_unique], new_counts))
                     logger.info(f"Class distribution after SMOTE: {new_class_dist}")
                 except Exception as smote_error: logger.error(f"SMOTE failed: {smote_error}. Continuing without.", exc_info=True); X_resampled, y_resampled = features, targets
        else: logger.info("SMOTE is disabled.")

        features_to_split = X_resampled; targets_to_split = y_resampled

        logger.info("Splitting data into train and test sets...")
        stratify_targets = targets_to_split if len(np.unique(targets_to_split)) > 1 else None
        if stratify_targets is not None: logger.info("Using stratification.")
        else: logger.warning("Cannot stratify split (only one class?).")
        X_train, X_test, y_train, y_test = train_test_split(features_to_split, targets_to_split, test_size=TRAINING_CONFIG.get('test_size', 0.25), random_state=TRAINING_CONFIG.get('random_state', 42), stratify=stratify_targets)
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        logger.info(f"Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info(f"Test dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        # Train base model (now includes tuning logic) and calibrate
        final_model, final_scaler, feature_importance = train_base_model(X_train, y_train, X_test, y_test, feature_names)

        if final_model and final_scaler and feature_names:
            logger.info("Saving upper face models, scaler, and feature list...")
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(final_model, MODEL_FILENAMES['base_model'])
            joblib.dump(final_scaler, MODEL_FILENAMES['base_scaler'])
            if not feature_importance.empty: feature_importance.to_csv(MODEL_FILENAMES['feature_importance'], index=False); logger.info(f"FI saved to {MODEL_FILENAMES['feature_importance']}")
            else: logger.warning("FI DataFrame empty.")
            logger.info("Upper face model training complete.")
        else: logger.error("Base model training/tuning/calibration failed. Models not saved.")

    except Exception as e: logger.error(f"Error in upper face training pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    train_models()