"""
Model training for ML-based lower face paralysis detection.
Trains and evaluates an XGBoost model using extracted features.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from lower_face_feature_extraction import prepare_data

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

def train_model(features, targets):
    """
    Train an XGBoost model for lower face paralysis detection.

    Args:
        features (pandas.DataFrame): Feature data
        targets (numpy.ndarray): Target labels

    Returns:
        tuple: (trained model, feature scaler, feature importance DataFrame)
    """
    # Print class distribution
    unique, counts = np.unique(targets, return_counts=True)
    class_names = ['None', 'Partial', 'Complete']
    class_dist = dict(zip([class_names[i] if i < len(class_names) else f"Class {i}" for i in unique], counts))
    logger.info(f"Class distribution in dataset: {class_dist}")

    # Split data with stratification to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42, stratify=targets)

    logger.info(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")

    # Scale features - crucial for optimal model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create feature names list for XGBoost
    feature_names = features.columns.tolist()

    # Train XGBoost model
    logger.info("Training XGBoost model...")
    
    # Initial hyperparameters - these can be tuned via grid search
    # Initial hyperparameters - using values closer to the original model that achieved 63% accuracy
    xgb_params = {
        'objective': 'multi:softprob',  # Multiclass probability output
        'num_class': 3,                 # 3 classes: None, Partial, Complete
        'learning_rate': 0.05,          # Back to original learning rate
        'max_depth': 6,                 # Original tree depth
        'min_child_weight': 2,          # Original value
        'subsample': 0.8,               # Use subsampling to reduce overfitting
        'colsample_bytree': 0.8,        # Feature subsampling
        'gamma': 0.1,                   # Original gamma value
        'random_state': 42,             # For reproducibility
        'n_estimators': 300             # Original number of boosting rounds
    }
    
    # Create and train XGBoost model
    model = xgb.XGBClassifier(**xgb_params)
    
    # Calculate moderate class weights - less extreme than before
    class_weights = {0: 1.0, 1: 2.0, 2: 0.8}  # More moderate: emphasize partial (2x), slightly de-emphasize complete (0.8x)
    sample_weights = np.ones(len(y_train))
    
    # Apply class weights to sample weights
    for i, y in enumerate(y_train):
        sample_weights[i] = class_weights[y]
    
    # Create and train XGBoost model
    model = xgb.XGBClassifier(**xgb_params)
    
    # Perform cross-validation
    logger.info("Performing cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
    logger.info(f"Cross-validation F1 scores: {cv_scores}")
    logger.info(f"Mean CV F1 score: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")

    # Train the final model on all training data
    logger.info("Training final model on all training data...")
    # XGBoost API changed - eval_metric is set in parameters, not in fit method
    # Train the final model on all training data
    logger.info("Training final model on all training data...")
    # Apply class weights through sample weights
    model.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weights,  # Use sample weights for class weighting
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )

    # For better calibrated probabilities, use CalibratedClassifierCV
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
        target_names=['None', 'Partial', 'Complete'],
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

        # Calculate ROC AUC - use OvR for multiclass
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        logger.info(f"ROC AUC (weighted OvR): {roc_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {str(e)}")

    # Extract and save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("Top 15 Most Important Features:")
    logger.info("\n" + feature_importance.head(15).to_string(index=False))

    return calibrated_model, scaler, feature_importance

def save_model_artifacts(model, scaler, feature_importance):
    """
    Save model and related artifacts for later use.

    Args:
        model: Trained model
        scaler: Feature scaler
        feature_importance (pandas.DataFrame): Feature importance data
    """
    try:
        # Create output directory
        os.makedirs('models', exist_ok=True)

        # Save model and artifacts
        joblib.dump(model, 'models/lower_face_paralysis_model.pkl')
        joblib.dump(scaler, 'models/lower_face_paralysis_scaler.pkl')
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        
        logger.info("Model artifacts saved to models/ directory")

    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise

def main():
    """Main function to run the model training process."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/lower_face_model_training.log')
        ]
    )

    logger.info("Starting lower face paralysis XGBoost model training...")

    # Prepare data
    logger.info("Preparing data...")
    features, targets = prepare_data()

    # Train XGBoost model
    logger.info("Training XGBoost model...")
    model, scaler, feature_importance = train_model(features, targets)

    # Save model artifacts
    logger.info("Saving model artifacts...")
    save_model_artifacts(model, scaler, feature_importance)

    logger.info("ML model training complete. XGBoost model ready for integration.")

if __name__ == "__main__":
    main()