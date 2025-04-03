"""
Unified training pipeline for mid face paralysis detection.
Handles base model training, specialist classifier, and threshold optimization.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score
)

from mid_face_features import prepare_data
from mid_face_config import (
    MODEL_FILENAMES, TRAINING_CONFIG, DETECTION_THRESHOLDS,
    LOG_DIR, LOGGING_CONFIG, CLASS_NAMES, FEATURE_CONFIG
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
            logging.FileHandler(os.path.join(LOG_DIR, 'mid_face_training.log'))
        ]
    )

    logger.info("Starting unified mid face paralysis detection training")

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
            try:
                sampling_strategy = TRAINING_CONFIG['smote'].get('sampling_strategy', 'auto')
                k_neighbors = TRAINING_CONFIG['smote'].get('k_neighbors', 3)

                # Ensure k_neighbors doesn't exceed minimum class size
                min_class_size = min(counts)
                k_neighbors = min(k_neighbors, min_class_size - 1)

                # Create SMOTE instance
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
            except Exception as e:
                logger.warning(f"SMOTE failed, continuing with original data: {str(e)}")

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

        # Step 5: Train specialist model for borderline cases
        logger.info("Training specialist model for borderline cases...")
        specialist_model, specialist_scaler = train_specialist_model(features, targets)

        # Step 6: Save models
        logger.info("Saving models...")
        os.makedirs(os.path.dirname(MODEL_FILENAMES['base_model']), exist_ok=True)

        # Ensure all model files have mid_face prefix
        joblib.dump(base_model, MODEL_FILENAMES['base_model'])
        joblib.dump(base_scaler, MODEL_FILENAMES['base_scaler'])
        feature_importance.to_csv(MODEL_FILENAMES['feature_importance'], index=False)

        joblib.dump(specialist_model, MODEL_FILENAMES['specialist_model'])
        joblib.dump(specialist_scaler, MODEL_FILENAMES['specialist_scaler'])

        # Step 7: Tune detection thresholds
        logger.info("Tuning detection thresholds...")
        optimal_thresholds = tune_thresholds(base_model, base_scaler, features, targets)

        # Save the optimal thresholds - ensure mid_face prefix
        config_path = os.path.join(LOG_DIR, 'mid_face_optimal_thresholds.json')
        with open(config_path, 'w') as f:
            json.dump(optimal_thresholds, f, indent=4)
        logger.info(f"Saved optimal thresholds to {config_path}")

        # Step 8: Optimize ES/ET ratio thresholds
        logger.info("Optimizing ES/ET ratio thresholds...")
        optimal_ratio_thresholds = tune_es_et_ratio_thresholds(features, targets)

        # Save optimal ES/ET ratio thresholds
        ratio_config_path = os.path.join(LOG_DIR, 'mid_face_optimal_ratio_thresholds.json')
        with open(ratio_config_path, 'w') as f:
            json.dump(optimal_ratio_thresholds, f, indent=4)
        logger.info(f"Saved optimal ES/ET ratio thresholds to {ratio_config_path}")

        logger.info("Model training complete.")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)


def train_base_model(X_train, y_train, X_test, y_test):
    """
    Train the base XGBoost model for mid face paralysis detection.

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

    # Create class weights dict - higher weights for minority classes
    class_weights = params['class_weights']

    # Create XGBoost DMatrix with weights
    # Calculate sample weights based on class
    sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])

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

    # Perform cross-validation
    logger.info("Performing cross-validation...")
    cv = StratifiedKFold(
        n_splits=TRAINING_CONFIG['cv_folds'],
        shuffle=True,
        random_state=TRAINING_CONFIG['random_state']
    )

    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=cv, scoring='f1_weighted'
    )

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

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)

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
        y_proba = model.predict_proba(X_test_scaled)

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

    return model, scaler, feature_importance


def train_specialist_model(features, targets):
    """
    Train specialist classifier for distinguishing between borderline cases.
    This model focuses on the border between None/Partial and Partial/Complete.

    Args:
        features (DataFrame): Feature data
        targets (ndarray): Target labels

    Returns:
        tuple: (trained specialist model, feature scaler)
    """
    # Create binary classification problem for the specialist
    # Combine None/Partial cases vs Complete cases
    is_borderline = np.logical_or(
        # None vs Partial cases (0 vs 1)
        np.logical_and(targets <= 1, np.random.rand(len(targets)) < 0.7),  # Take subset of None cases
        # Partial vs Complete cases (1 vs 2)
        targets >= 1
    )

    specialist_features = features[is_borderline]
    specialist_targets = (targets[is_borderline] >= 1.5).astype(int)  # Binary: 0=None/Partial, 1=Complete

    logger.info(f"Specialist training data: {len(specialist_features)} samples")
    unique, counts = np.unique(specialist_targets, return_counts=True)
    logger.info(f"Specialist class distribution: {dict(zip(unique, counts))}")

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

    # Create and train XGBoost classifier for binary classification
    model = xgb.XGBClassifier(
        objective=params['objective'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        n_estimators=params['n_estimators'],
        scale_pos_weight=params['scale_pos_weight'],
        random_state=TRAINING_CONFIG['random_state']
    )

    # Train model
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Specialist model accuracy: {accuracy:.4f}")
    logger.info(f"Specialist model F1 score: {f1:.4f}")

    # Detailed evaluation
    report = classification_report(
        y_test, y_pred,
        target_names=['None/Partial', 'Complete'],
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

    # Define threshold ranges to search - more stringent ranges to prevent overdetection
    complete_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    none_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    partial_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    upgrade_thresholds = [0.6, 0.7, 0.8, 0.9]
    specialist_lower_thresholds = [0.6, 0.7, 0.8]
    specialist_upper_thresholds = [0.8, 0.9, 0.95]
    specialist_partial_thresholds = [0.5, 0.6, 0.7, 0.8]

    # Track best configuration and metrics
    best_config = None
    best_score = -1
    results = []

    # Grid search through threshold combinations
    logger.info("Performing grid search for thresholds...")

    # First round - tune basic thresholds
    for complete_thresh in complete_thresholds:
        for none_thresh in none_thresholds:
            for partial_thresh in partial_thresholds:
                for upgrade_thresh in upgrade_thresholds:
                    # Apply thresholds
                    adjusted_preds = raw_predictions.copy()

                    for i in range(len(adjusted_preds)):
                        # Adjust Complete predictions with low confidence
                        if adjusted_preds[i] == 2:  # Complete prediction
                            if prediction_probas[i][2] < complete_thresh:  # Low confidence
                                if prediction_probas[i][0] > none_thresh:  # High None probability
                                    adjusted_preds[i] = 0  # Downgrade to None
                                elif prediction_probas[i][1] > partial_thresh:  # High Partial probability
                                    adjusted_preds[i] = 1  # Downgrade to Partial

                        # None prediction with significant Partial probability
                        elif adjusted_preds[i] == 0 and prediction_probas[i][1] > upgrade_thresh:
                            adjusted_preds[i] = 1  # Upgrade to Partial

                    # Calculate key metrics
                    accuracy = accuracy_score(y_val, adjusted_preds)
                    f1 = f1_score(y_val, adjusted_preds, average='weighted')

                    # Calculate class-specific metrics
                    class_recalls = {}
                    for cls in range(3):
                        if cls in y_val:
                            class_recalls[cls] = recall_score(
                                y_val == cls,
                                adjusted_preds == cls,
                                zero_division=0
                            )
                        else:
                            class_recalls[cls] = 0

                    # Count critical error types - these are the most important to minimize
                    none_to_complete = sum((y_val == 0) & (adjusted_preds == 2))
                    complete_to_none = sum((y_val == 2) & (adjusted_preds == 0))
                    critical_errors = none_to_complete + complete_to_none

                    # Custom score that heavily penalizes false positives (None classified as Complete)
                    # This is the key change to prevent the overdetection issue
                    custom_score = (
                        f1 * 0.3 +
                        (class_recalls.get(0, 0) * 0.4) +  # Emphasize None class recall
                        (class_recalls.get(1, 0) * 0.2) +  # Consider partial recall
                        (class_recalls.get(2, 0) * 0.1) +  # Lower weight for complete recall
                        (1.0 - (none_to_complete / max(sum(y_val == 0), 1) * 2)) * 0.2  # Heavily penalize None→Complete errors
                    )

                    # Track result
                    results.append({
                        'complete_threshold': complete_thresh,
                        'none_threshold': none_thresh,
                        'partial_threshold': partial_thresh,
                        'upgrade_to_partial': upgrade_thresh,
                        'accuracy': accuracy,
                        'f1_weighted': f1,
                        'none_recall': class_recalls.get(0, 0),
                        'partial_recall': class_recalls.get(1, 0),
                        'complete_recall': class_recalls.get(2, 0),
                        'none_to_complete': none_to_complete,
                        'complete_to_none': complete_to_none,
                        'critical_errors': critical_errors,
                        'custom_score': custom_score
                    })

    # Find best basic thresholds
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='custom_score', ascending=False)
    best_basic = results_df.iloc[0]

    # Set initial best config from basic thresholds
    best_config = {
        'complete_confidence': best_basic['complete_threshold'],
        'none_probability': best_basic['none_threshold'],
        'partial_probability': best_basic['partial_threshold'],
        'upgrade_to_partial': best_basic['upgrade_to_partial']
    }
    best_score = best_basic['custom_score']

    # Second round - tune specialist thresholds using best basic thresholds
    specialist_results = []
    for spec_lower in specialist_lower_thresholds:
        for spec_upper in specialist_upper_thresholds:
            for spec_partial in specialist_partial_thresholds:
                # Skip invalid combinations
                if spec_lower >= spec_upper:
                    continue

                # Create full threshold set
                thresholds = {
                    'complete_confidence': best_basic['complete_threshold'],
                    'none_probability': best_basic['none_threshold'],
                    'partial_probability': best_basic['partial_threshold'],
                    'upgrade_to_partial': best_basic['upgrade_to_partial'],
                    'specialist_complete_lower': spec_lower,
                    'specialist_complete_upper': spec_upper,
                    'specialist_partial_threshold': spec_partial
                }

                # Evaluate these thresholds (in a real system, this would involve the specialist model)
                # For now, we'll just track the threshold values
                specialist_results.append({
                    **thresholds,
                    'basic_custom_score': best_basic['custom_score']
                })

    # In a full implementation, we would evaluate these specialist thresholds
    # For now, we'll just use sensible defaults based on our analysis
    best_specialist_config = {
        'specialist_complete_lower': 0.7,  # More stringent lower bound
        'specialist_complete_upper': 0.9,  # More stringent upper bound
        'specialist_partial_threshold': 0.7  # More stringent threshold
    }

    # Combine best basic and specialist thresholds
    best_config.update(best_specialist_config)

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
        logger.info(f"  Upgrade threshold: {config['upgrade_to_partial']}")
        logger.info(f"  Accuracy: {config['accuracy']:.4f}")
        logger.info(f"  F1 weighted: {config['f1_weighted']:.4f}")
        logger.info(f"  None recall: {config['none_recall']:.4f}")
        logger.info(f"  Partial recall: {config['partial_recall']:.4f}")
        logger.info(f"  Complete recall: {config['complete_recall']:.4f}")
        logger.info(f"  Critical errors: {config['critical_errors']}")
        logger.info(f"  Custom score: {config['custom_score']:.4f}")

    # Save results for reference - ensure mid_face prefix
    results_df.to_csv(os.path.join(LOG_DIR, 'mid_face_threshold_tuning_results.csv'), index=False)

    logger.info(f"Selected optimal thresholds: {best_config}")
    return best_config


def tune_es_et_ratio_thresholds(features, targets):
    """
    Find optimal ES/ET ratio thresholds for classification.

    Args:
        features (DataFrame): Feature data
        targets (ndarray): Target labels

    Returns:
        dict: Optimal ES/ET ratio thresholds
    """
    logger.info("Analyzing ES/ET ratio distribution by class...")

    # Try to find any columns related to ES/ET ratio
    ratio_columns = [col for col in features.columns if 'ratio' in col.lower() and ('es' in col.lower() or 'et' in col.lower())]

    if not ratio_columns:
        logger.warning("No ES/ET ratio columns found in features")
        # Return default thresholds from config
        return {
            'complete_max': FEATURE_CONFIG['es_et_ratio_thresholds']['complete_max'],
            'partial_max': FEATURE_CONFIG['es_et_ratio_thresholds']['partial_max']
        }

    # Extract the most relevant ratio feature (prioritize weighted ratio if available)
    ratio_feature = next((col for col in ratio_columns if 'weight' in col.lower()), ratio_columns[0])
    logger.info(f"Using ratio feature: {ratio_feature}")

    # Group ratio values by class
    class_ratios = {
        0: features.loc[targets == 0, ratio_feature],
        1: features.loc[targets == 1, ratio_feature],
        2: features.loc[targets == 2, ratio_feature]
    }

    # Calculate percentiles for each class
    class_percentiles = {}
    for cls, values in class_ratios.items():
        if len(values) == 0:
            class_percentiles[cls] = {'p25': None, 'p50': None, 'p75': None}
            continue

        class_percentiles[cls] = {
            'p25': np.percentile(values, 25),
            'p50': np.percentile(values, 50),
            'p75': np.percentile(values, 75)
        }

        logger.info(f"Class {CLASS_NAMES[cls]} ES/ET ratio percentiles: " +
                   f"25th={class_percentiles[cls]['p25']:.3f}, " +
                   f"50th={class_percentiles[cls]['p50']:.3f}, " +
                   f"75th={class_percentiles[cls]['p75']:.3f}")

    # Find optimal threshold between Complete and Partial classes
    complete_max = 0.5  # Default
    if class_percentiles[1]['p25'] is not None and class_percentiles[2]['p75'] is not None:
        # Find a threshold between 75th percentile of Complete and 25th percentile of Partial
        complete_max = (class_percentiles[2]['p75'] + class_percentiles[1]['p25']) / 2
        logger.info(f"Determined threshold between Complete and Partial: {complete_max:.3f}")

    # Find optimal threshold between Partial and None classes
    partial_max = 0.95  # Default
    if class_percentiles[0]['p25'] is not None and class_percentiles[1]['p75'] is not None:
        # Find a threshold between 75th percentile of Partial and 25th percentile of None
        partial_max = (class_percentiles[1]['p75'] + class_percentiles[0]['p25']) / 2
        logger.info(f"Determined threshold between Partial and None: {partial_max:.3f}")

    # Ensure thresholds are reasonable
    complete_max = min(max(complete_max, 0.3), 0.7)  # Keep between 0.3 and 0.7
    partial_max = min(max(partial_max, 0.7), 0.95)  # Keep between 0.7 and 0.95

    # Ensure complete_max < partial_max
    if complete_max >= partial_max:
        logger.warning(f"Adjusting thresholds because complete_max ({complete_max:.3f}) >= partial_max ({partial_max:.3f})")

        # Use defaults from config but ensure their relationship
        complete_max = FEATURE_CONFIG['es_et_ratio_thresholds']['complete_max']
        partial_max = FEATURE_CONFIG['es_et_ratio_thresholds']['partial_max']

        if complete_max >= partial_max:
            complete_max = 0.5
            partial_max = 0.95

    optimal_thresholds = {
        'complete_max': complete_max,
        'partial_max': partial_max
    }

    logger.info(f"Optimal ES/ET ratio thresholds: {optimal_thresholds}")
    return optimal_thresholds


if __name__ == "__main__":
    train_models()