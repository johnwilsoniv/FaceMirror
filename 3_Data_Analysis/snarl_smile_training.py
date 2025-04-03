# snarl_smile_training.py (Mirrors oral_ocular_training.py)

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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
    """ Trains and evaluates the Snarl-Smile synkinesis model. """
    logger.info("--- Starting Snarl-Smile Synkinesis Model Training ---")

    # 1. Prepare Data
    try:
        features, targets = prepare_data()
        if features is None or targets is None: raise ValueError("Data prep failed.")
        logger.info(f"Data prepared: {features.shape[0]} samples, {features.shape[1]} features.")
        unique_targets, counts = np.unique(targets, return_counts=True)
        target_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(unique_targets, counts)}
        logger.info(f"Target distribution in final data: {target_dist}")
    except Exception as e: logger.error(f"Data prep error: {e}", exc_info=True); return

    # 2. Data Splitting
    test_size = TRAINING_CONFIG.get('test_size', 0.25)
    random_state = TRAINING_CONFIG.get('random_state', 42)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state, stratify=targets)
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        train_target_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_train, return_counts=True))}
        logger.info(f"Training target distribution (before SMOTE): {train_target_dist}")
        logger.info(f"Testing target distribution: { {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_test, return_counts=True))} }")

        min_class_count_train = min(train_target_dist.values()) if train_target_dist else 0
        smote_config = TRAINING_CONFIG.get('smote', {})
        k_neighbors_config = smote_config.get('k_neighbors', 5)
        safe_k_neighbors = min(k_neighbors_config, min_class_count_train - 1) if min_class_count_train > 1 else 1
        if safe_k_neighbors != k_neighbors_config: logger.warning(f"Adjusted SMOTE k_neighbors to {safe_k_neighbors} from {k_neighbors_config}.")
        smote_config['k_neighbors'] = safe_k_neighbors

    except Exception as e: logger.error(f"Split/k_neighbor error: {e}", exc_info=True); return

    # 3. SMOTE & Scaling
    scaler = StandardScaler(); X_train_fit, y_train_fit = X_train, y_train
    if smote_config.get('enabled', False) and min_class_count_train > 1:
        logger.info("Applying SMOTE...")
        try:
            smote_params = {k: v for k, v in smote_config.items() if k != 'enabled'}
            smote = SMOTE(**smote_params); logger.info(f"SMOTE params: {smote_params}")
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE applied. Original: {X_train.shape}, Resampled: {X_train_resampled.shape}")
            resampled_dist = {CLASS_NAMES.get(k, k): v for k, v in zip(*np.unique(y_train_resampled, return_counts=True))}
            logger.info(f"Resampled train distribution: {resampled_dist}")
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            logger.info("Scaled after SMOTE (fit on resampled).")
            X_train_fit, y_train_fit = X_train_scaled, y_train_resampled
        except Exception as smote_e:
            logger.error(f"SMOTE error: {smote_e}. Training on original.", exc_info=True)
            X_train_scaled = scaler.fit_transform(X_train)
            logger.info("Scaled (fit on original due to SMOTE error).")
            X_train_fit, y_train_fit = X_train_scaled, y_train
    else:
        if not smote_config.get('enabled', False): logger.info("SMOTE disabled.")
        elif min_class_count_train <= 1: logger.warning(f"SMOTE skipped: Minority class <= 1 sample.")
        X_train_scaled = scaler.fit_transform(X_train)
        logger.info("Scaled (fit on original).")
        X_train_fit, y_train_fit = X_train_scaled, y_train
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training
    model_config = TRAINING_CONFIG.get('model', {}); model_type = model_config.get('type', 'random_forest')
    model_params = {k: v for k, v in model_config.items() if k != 'type'}
    try:
        if model_type == 'random_forest':
            logger.info(f"Training RandomForestClassifier: {model_params}")
            model = RandomForestClassifier(**model_params)
        else: logger.error(f"Unsupported model type '{model_type}'."); return

        cv_folds = TRAINING_CONFIG.get('cv_folds', 5)
        if cv_folds > 1:
            logger.info(f"Performing {cv_folds}-fold CV...")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train_fit, y_train_fit, cv=cv, scoring='f1_weighted', n_jobs=-1)
            logger.info(f"CV F1 Weighted Scores: {cv_scores}")
            logger.info(f"Mean CV F1 Weighted: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        logger.info("Training final model...")
        model.fit(X_train_fit, y_train_fit)
        logger.info("Training complete.")
    except Exception as e: logger.error(f"Training error: {e}", exc_info=True); return

    # 5. Evaluation
    try:
        logger.info("Evaluating on test set...")
        y_pred = model.predict(X_test_scaled); y_proba = model.predict_proba(X_test_scaled)
        report = classification_report(y_test, y_pred, target_names=[CLASS_NAMES.get(i) for i in sorted(CLASS_NAMES.keys())], zero_division=0)
        logger.info("Report (Test Set):\n" + report)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(CLASS_NAMES.keys()))
        logger.info("CM (Test Set):\n" + str(cm))
        if len(CLASS_NAMES) == 2 and 1 in CLASS_NAMES:
            try: roc_auc = roc_auc_score(y_test, y_proba[:, 1]); logger.info(f"ROC AUC (Test Set): {roc_auc:.4f}")
            except Exception as roc_e: logger.warning(f"ROC AUC calc failed: {roc_e}")
    except Exception as e: logger.error(f"Evaluation error: {e}", exc_info=True)

    # 6. Feature Importance
    feature_importance_df = pd.DataFrame()
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_; feature_names = features.columns
            if len(importances) == len(feature_names):
                 feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
                 logger.info("Top 15 Importances:"); logger.info("\n" + feature_importance_df.head(15).to_string())
            else: logger.warning("Importance/Feature length mismatch.")
        else: logger.warning("Model lacks feature_importances_.")
    except Exception as e: logger.error(f"Importance error: {e}", exc_info=True)

    # 7. Save Artifacts
    try:
        model_path = MODEL_FILENAMES.get('model'); scaler_path = MODEL_FILENAMES.get('scaler')
        importance_path = MODEL_FILENAMES.get('feature_importance')
        if not all([model_path, scaler_path, importance_path]): raise ValueError("Artifact paths missing.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path); logger.info(f"Model saved: {model_path}")
        joblib.dump(scaler, scaler_path); logger.info(f"Scaler saved: {scaler_path}")
        if not feature_importance_df.empty and not feature_importance_df['importance'].isnull().all():
             feature_importance_df.to_csv(importance_path, index=False); logger.info(f"Importance saved: {importance_path}")
    except Exception as e: logger.error(f"Saving error: {e}", exc_info=True)

    logger.info("--- Snarl-Smile Synkinesis Model Training Finished ---")

if __name__ == "__main__":
    train_and_evaluate()