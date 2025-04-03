"""
Unified training pipeline for lower face paralysis detection.
Handles base model training, specialist classifier, and threshold optimization.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from lower_face_features import prepare_data
from lower_face_config import (
    MODEL_FILENAMES, TRAINING_CONFIG, DETECTION_THRESHOLDS,
    LOG_DIR, LOGGING_CONFIG, CLASS_NAMES
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score
)

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


def train_models():
    """
    Comprehensive training pipeline that handles:
    - Base model training
    - Specialist classifier training
    - Threshold optimization
    - Performance validation

    All models are saved in the configured directories.
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, 'lower_face_training.log'))
        ]
    )

    logger.info("Starting unified lower face paralysis detection training")

    try:
        # Step 1: Prepare data
        logger.info("Preparing data...")
        features, targets = prepare_data()

        # Log class distribution
        unique, counts = np.unique(targets, return_counts=True)
        class_dist = dict(zip([CLASS_NAMES[i] for i in unique], counts))
        logger.info(f"Class distribution: {class_dist}")

        # Step 2: Apply SMOTE if enabled
        if TRAINING_CONFIG['smote']['enabled']:
            logger.info("Applying SMOTE for class balancing...")
            # Calculate sampling strategy
            if len(counts) >= 3:  # Make sure we have all three classes
                multiplier = TRAINING_CONFIG['smote']['sampling_multiplier']
                sampling_strategy = {
                    0: counts[0],  # Keep None class as is
                    1: int(counts[0] * multiplier),  # Boost Partial class
                    2: counts[2]  # Keep Complete class as is
                }

                # Apply SMOTE
                k_neighbors = min(TRAINING_CONFIG['smote']['k_neighbors'], min(counts) - 1)
                smote = SMOTE(
                    random_state=TRAINING_CONFIG['random_state'],
                    k_neighbors=k_neighbors,
                    sampling_strategy=sampling_strategy
                )
                features, targets = smote.fit_resample(features, targets)

                # Log new distribution
                new_unique, new_counts = np.unique(targets, return_counts=True)
                new_class_dist = dict(zip([CLASS_NAMES[i] for i in new_unique], new_counts))
                logger.info(f"Class distribution after SMOTE: {new_class_dist}")

        # Step 3: Split data for training and testing
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets,
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state'],
            stratify=targets
        )

        logger.info(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")

        # Step 4: Train base model
        logger.info("Training base model...")
        base_model, base_scaler, feature_importance = train_base_model(X_train, y_train, X_test, y_test)

        # Step 5: Train specialist model
        logger.info("Training specialist model...")
        specialist_model, specialist_scaler = train_specialist_model(features, targets)

        # Step 6: Save models
        logger.info("Saving models...")
        os.makedirs(os.path.dirname(MODEL_FILENAMES['base_model']), exist_ok=True)

        # Ensure all model files have lower_face prefix
        joblib.dump(base_model, MODEL_FILENAMES['base_model'])
        joblib.dump(base_scaler, MODEL_FILENAMES['base_scaler'])
        feature_importance.to_csv(MODEL_FILENAMES['feature_importance'], index=False)

        joblib.dump(specialist_model, MODEL_FILENAMES['specialist_model'])
        joblib.dump(specialist_scaler, MODEL_FILENAMES['specialist_scaler'])

        # Step 7: Tune detection thresholds
        logger.info("Tuning detection thresholds...")
        optimal_thresholds = tune_thresholds(base_model, base_scaler, features, targets)

        # Save the optimal thresholds
        config_path = os.path.join(LOG_DIR, 'lower_face_optimal_thresholds.json')  # Updated filename
        import json
        with open(config_path, 'w') as f:
            json.dump(optimal_thresholds, f, indent=4)
        logger.info(f"Saved optimal thresholds to {config_path}")

        logger.info("Model training complete.")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)


def train_base_model(X_train, y_train, X_test, y_test):
    """
    Train the base XGBoost model.

    Args:
        X_train (DataFrame): Training features
        y_train (ndarray): Training targets
        X_test (DataFrame): Testing features
        y_test (ndarray): Testing targets

    Returns:
        tuple: (trained model, feature scaler, feature importance DataFrame)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create feature names list
    feature_names = X_train.columns.tolist()

    # Get model parameters
    params = TRAINING_CONFIG['base_model']

    # Create and train model
    model = xgb.XGBClassifier(
        objective=params['objective'],
        num_class=params['num_class'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        random_state=TRAINING_CONFIG['random_state'],
        n_estimators=params['n_estimators']
    )

    # Create sample weights based on class weights
    class_weights = params['class_weights']
    sample_weights = np.ones(len(y_train))
    for i, y in enumerate(y_train):
        sample_weights[i] = class_weights[y]

    # Perform cross-validation
    logger.info("Performing cross-validation...")
    cv = StratifiedKFold(
        n_splits=TRAINING_CONFIG['cv_folds'],
        shuffle=True,
        random_state=TRAINING_CONFIG['random_state']
    )
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
    logger.info(f"Cross-validation F1 scores: {cv_scores}")
    logger.info(f"Mean CV F1 score: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")

    # Train the model
    logger.info("Training final model...")
    model.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # For better calibrated probabilities
    logger.info("Calibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',  # or 'sigmoid'
        cv='prefit'  # Use prefit model
    )
    calibrated_model.fit(X_test_scaled, y_test)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = calibrated_model.predict(X_test_scaled)

    # Get classification report
    report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in range(3)],
        zero_division=0
    )
    logger.info("Classification Report:\n" + report)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info("\n" + str(conf_matrix))

    # Calculate ROC AUC
    try:
        # Get prediction probabilities
        y_proba = calibrated_model.predict_proba(X_test_scaled)

        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        logger.info(f"ROC AUC (weighted OvR): {roc_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {str(e)}")

    # Extract feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("Top 15 Most Important Features:")
    logger.info("\n" + feature_importance.head(15).to_string(index=False))

    return calibrated_model, scaler, feature_importance


def train_specialist_model(features, targets):
    """
    Train specialist classifier for distinguishing between Partial and Complete.

    Args:
        features (DataFrame): Feature data
        targets (ndarray): Target labels

    Returns:
        tuple: (trained specialist model, feature scaler)
    """
    # Filter for only Partial/Complete cases
    is_paralysis = (targets > 0)
    specialist_features = features[is_paralysis]
    specialist_targets = targets[is_paralysis] - 1  # Shift to 0=Partial, 1=Complete

    logger.info(f"Specialist training data: {len(specialist_features)} samples")
    unique, counts = np.unique(specialist_targets, return_counts=True)
    logger.info(f"Specialist class distribution: {{0 (Partial): {counts[0]}, 1 (Complete): {counts[1]}}}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        specialist_features, specialist_targets,
        test_size=TRAINING_CONFIG['test_size'],
        random_state=TRAINING_CONFIG['random_state'] + 1,  # Different seed than base model
        stratify=specialist_targets
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get specialist parameters
    params = TRAINING_CONFIG['specialist_model']

    # Create and train Random Forest
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        class_weight=params['class_weight'],
        bootstrap=params['bootstrap'],
        random_state=TRAINING_CONFIG['random_state']
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    logger.info(f"Specialist training accuracy: {train_acc:.4f}")
    logger.info(f"Specialist testing accuracy: {test_acc:.4f}")

    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)

    # Get classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['Partial', 'Complete'],
        zero_division=0
    )
    logger.info("Specialist Classification Report:\n" + report)

    return model, scaler


def tune_thresholds(base_model, base_scaler, features, targets):
    """
    Tune post-processing thresholds to minimize critical errors.

    Args:
        base_model: Trained base model
        base_scaler: Feature scaler
        features: Feature data
        targets: Target labels

    Returns:
        dict: Optimal threshold values
    """
    logger.info("Tuning detection thresholds...")

    # Split data specifically for threshold tuning
    # Use a different random seed than for model training
    _, X_val, _, y_val = train_test_split(
        features, targets, test_size=0.3, random_state=43
    )

    # Scale features
    X_val_scaled = base_scaler.transform(X_val)

    # Get raw predictions and probabilities
    raw_predictions = base_model.predict(X_val_scaled)
    prediction_probas = base_model.predict_proba(X_val_scaled)

    # Define threshold ranges to search
    complete_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    none_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    partial_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35]

    # Track best configuration and metrics
    best_config = None
    best_score = -1
    results = []

    # Grid search through threshold combinations
    logger.info("Performing grid search for thresholds...")
    for complete_thresh in complete_thresholds:
        for none_thresh in none_thresholds:
            for partial_thresh in partial_thresholds:
                # Apply thresholds
                adjusted_preds = raw_predictions.copy()

                for i in range(len(adjusted_preds)):
                    if adjusted_preds[i] == 2:  # Complete prediction
                        if prediction_probas[i][2] < complete_thresh:  # Low confidence
                            if prediction_probas[i][0] > none_thresh:  # High None probability
                                adjusted_preds[i] = 0  # Downgrade to None
                            elif prediction_probas[i][1] > partial_thresh:  # High Partial probability
                                adjusted_preds[i] = 1  # Downgrade to Partial

                # Calculate key metrics
                accuracy = accuracy_score(y_val, adjusted_preds)
                f1 = f1_score(y_val, adjusted_preds, average='weighted')

                # Count critical error types - these are the most important to minimize
                none_to_complete = sum((y_val == 0) & (adjusted_preds == 2))
                complete_to_none = sum((y_val == 2) & (adjusted_preds == 0))
                critical_errors = none_to_complete + complete_to_none

                # Calculate partial class metrics
                partial_recall = recall_score(y_val == 1, adjusted_preds == 1, zero_division=0)

                # Custom score that penalizes critical errors heavily
                # This formula can be adjusted based on priorities
                custom_score = f1 - (critical_errors * 0.1) + (partial_recall * 0.05)

                # Track result
                results.append({
                    'complete_threshold': complete_thresh,
                    'none_threshold': none_thresh,
                    'partial_threshold': partial_thresh,
                    'accuracy': accuracy,
                    'f1_weighted': f1,
                    'none_to_complete': none_to_complete,
                    'complete_to_none': complete_to_none,
                    'partial_recall': partial_recall,
                    'custom_score': custom_score
                })

                # Check if this is the best configuration so far
                if custom_score > best_score:
                    best_score = custom_score
                    best_config = {
                        'complete_confidence': complete_thresh,
                        'none_probability': none_thresh,
                        'partial_probability': partial_thresh
                    }

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Sort by custom score
    results_df = results_df.sort_values(by='custom_score', ascending=False)

    # Log top 5 configurations
    logger.info("Top 5 threshold configurations:")
    for i, config in results_df.head(5).iterrows():
        logger.info(f"Configuration {i + 1}:")
        logger.info(f"  Complete threshold: {config['complete_threshold']}")
        logger.info(f"  None threshold: {config['none_threshold']}")
        logger.info(f"  Partial threshold: {config['partial_threshold']}")
        logger.info(f"  Accuracy: {config['accuracy']:.4f}")
        logger.info(f"  F1 weighted: {config['f1_weighted']:.4f}")
        logger.info(f"  None-to-Complete errors: {config['none_to_complete']}")
        logger.info(f"  Complete-to-None errors: {config['complete_to_none']}")
        logger.info(f"  Partial recall: {config['partial_recall']:.4f}")

    # Save results for reference
    results_df.to_csv(os.path.join(LOG_DIR, 'lower_face_threshold_tuning_results.csv'), index=False)  # Updated filename

    logger.info(f"Selected optimal thresholds: {best_config}")
    return best_config

if __name__ == "__main__":
    train_models()
