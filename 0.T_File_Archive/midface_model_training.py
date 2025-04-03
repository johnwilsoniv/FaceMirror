"""
Model training for ML-based mid face paralysis detection.
Trains and evaluates a machine learning model using extracted features.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from midface_feature_extraction import prepare_data

logger = logging.getLogger(__name__)

def train_model(features, targets, model_type='random_forest'):
    """
    Train a machine learning model for mid face paralysis detection with improved
    handling of class imbalance.

    Args:
        features (pandas.DataFrame): Feature data
        targets (numpy.ndarray): Target labels
        model_type (str): Type of model to train ('random_forest' or 'gradient_boosting')

    Returns:
        tuple: (trained model, feature scaler, feature importance DataFrame)
    """
    # Print class distribution
    unique, counts = np.unique(targets, return_counts=True)
    class_names = ['None', 'Partial', 'Complete']
    class_dist = dict(zip([class_names[i] if i < len(class_names) else f"Class {i}" for i in unique], counts))
    logger.info(f"Class distribution in dataset: {class_dist}")

    # Calculate class weights inversely proportional to class frequencies
    n_samples = len(targets)
    class_weights = {}
    for i in unique:
        class_count = counts[np.where(unique == i)[0][0]]
        # More aggressive weighting for rare classes
        class_weights[i] = n_samples / (len(unique) * class_count)

    logger.info(f"Using class weights: {class_weights}")

    # Split data with stratification to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42, stratify=targets)

    logger.info(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model based on selected type with class weighting
    if model_type == 'gradient_boosting':
        logger.info("Training Gradient Boosting Classifier...")
        model = GradientBoostingClassifier(
            n_estimators=200,          # More trees for better learning
            learning_rate=0.05,        # Lower learning rate for better generalization
            max_depth=6,               # Moderate tree depth to avoid overfitting
            subsample=0.8,             # Use subsampling to reduce overfitting
            min_samples_split=5,       # Minimum samples to split an internal node
            min_samples_leaf=2,        # Minimum samples in a leaf node
            random_state=42
        )
    else:  # Default to random forest
        logger.info("Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=200,          # More trees for better learning
            max_depth=None,            # Let trees grow to their full depth
            min_samples_split=2,       # Minimum samples to split an internal node
            min_samples_leaf=1,        # Minimum samples in a leaf node
            class_weight=class_weights, # Apply class weights
            bootstrap=True,            # Use bootstrap samples
            random_state=42
        )

    # Perform cross-validation
    logger.info("Performing cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
    logger.info(f"Cross-validation F1 scores: {cv_scores}")
    logger.info(f"Mean CV F1 score: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")

    # Train the final model on all training data
    logger.info("Training final model on all training data...")
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)

    # Get classification report with zero_division=0 to handle missing predictions
    report = classification_report(
        y_test, y_pred,
        target_names=['None', 'Partial', 'Complete'] if len(unique) <= 3 else None,
        zero_division=0
    )
    logger.info("Classification Report:\n" + report)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info("\n" + str(conf_matrix))

    # Calculate ROC AUC if possible
    try:
        # Get prediction probabilities
        y_proba = model.predict_proba(X_test_scaled)

        # Calculate ROC AUC - use OvR for multiclass
        if len(unique) > 2:
            # One-vs-Rest ROC AUC for multiclass
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            logger.info(f"ROC AUC (weighted OvR): {roc_auc:.4f}")
        else:
            # Binary ROC AUC
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            logger.info(f"ROC AUC (binary): {roc_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {str(e)}")

    # Extract and save feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("Top 15 Most Important Features:")
        logger.info("\n" + feature_importance.head(15).to_string(index=False))
    else:
        # Create empty feature importance if not available
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': np.zeros(len(features.columns))
        })

    return model, scaler, feature_importance

def save_model_artifacts(model, scaler, feature_importance):
    """
    Save model and related artifacts for later use.

    Args:
        model: Trained model
        scaler: Feature scaler
        feature_importance (pandas.DataFrame): Feature importance data
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save model
        joblib.dump(model, 'models/mid_face_paralysis_model.pkl')
        logger.info("Model saved to models/mid_face_paralysis_model.pkl")

        # Save scaler
        joblib.dump(scaler, 'models/mid_face_paralysis_scaler.pkl')
        logger.info("Scaler saved to models/mid_face_paralysis_scaler.pkl")

        # Save feature importance
        feature_importance.to_csv('models/mid_face_feature_importance.csv', index=False)
        logger.info("Feature importance saved to models/mid_face_feature_importance.csv")

        # Also save in root directory for backward compatibility
        joblib.dump(model, 'mid_face_paralysis_model.pkl')
        joblib.dump(scaler, 'mid_face_paralysis_scaler.pkl')
        feature_importance.to_csv('mid_face_feature_importance.csv', index=False)

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
            logging.FileHandler('midface_model_training.log')
        ]
    )

    logger.info("Starting mid face paralysis ML model training...")

    # Prepare data
    logger.info("Preparing data...")
    features, targets = prepare_data()

    # Train model with random forest (default)
    logger.info("Training Random Forest model...")
    rf_model, rf_scaler, rf_importance = train_model(
        features, targets, model_type='random_forest'
    )

    # Save random forest model artifacts
    logger.info("Saving Random Forest model artifacts...")
    save_model_artifacts(rf_model, rf_scaler, rf_importance)

    # Train gradient boosting model as alternative
    logger.info("Training Gradient Boosting model...")
    gb_model, gb_scaler, gb_importance = train_model(
        features, targets, model_type='gradient_boosting'
    )

    # Save gradient boosting model artifacts
    logger.info("Saving Gradient Boosting model artifacts...")
    os.makedirs('models/gradient_boosting', exist_ok=True)
    joblib.dump(gb_model, 'models/gradient_boosting/mid_face_paralysis_model.pkl')
    joblib.dump(gb_scaler, 'models/gradient_boosting/mid_face_paralysis_scaler.pkl')
    gb_importance.to_csv('models/gradient_boosting/mid_face_feature_importance.csv', index=False)

    logger.info("ML model training complete. Models ready for integration.")

if __name__ == "__main__":
    main()