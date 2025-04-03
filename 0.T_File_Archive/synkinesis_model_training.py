"""
Model training for ML-based synkinesis detection.
Trains and evaluates machine learning models for detecting different types of synkinesis.
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
from synkinesis_feature_extraction import prepare_synkinesis_data

logger = logging.getLogger(__name__)

def train_model(features, targets, model_type='random_forest'):
    """
    Train a machine learning model for synkinesis detection with 
    improved handling of class imbalance.
    
    Args:
        features (pandas.DataFrame): Feature data
        targets (numpy.ndarray): Target labels
        model_type (str): Type of model to train ('random_forest' or 'gradient_boosting')
        
    Returns:
        tuple: (trained model, feature scaler, feature importance DataFrame)
    """
    # Print class distribution
    unique, counts = np.unique(targets, return_counts=True)
    class_names = ['None', 'Synkinesis']
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
    
    # Get classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['None', 'Synkinesis'] if len(unique) <= 2 else None,
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
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        logger.info(f"ROC AUC: {roc_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {str(e)}")
    
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 15 Most Important Features:")
        logger.info("\n" + feature_importance.head(15).to_string(index=False))
    else:
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': np.zeros(len(features.columns))
        })
        
    return model, scaler, feature_importance

def save_model_artifacts(model, scaler, feature_importance, synkinesis_type):
    """
    Save model and related artifacts for later use.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_importance (pandas.DataFrame): Feature importance data
        synkinesis_type (str): Type of synkinesis ('ocular_oral', 'oral_ocular', or 'snarl_smile')
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs('models/synkinesis', exist_ok=True)
        
        # Model-specific directory
        model_dir = f'models/synkinesis/{synkinesis_type}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        joblib.dump(model, f'{model_dir}/model.pkl')
        logger.info(f"Model saved to {model_dir}/model.pkl")
        
        # Save scaler
        joblib.dump(scaler, f'{model_dir}/scaler.pkl')
        logger.info(f"Scaler saved to {model_dir}/scaler.pkl")
        
        # Save feature importance
        feature_importance.to_csv(f'{model_dir}/feature_importance.csv', index=False)
        logger.info(f"Feature importance saved to {model_dir}/feature_importance.csv")
        
        # Also save in root directory for backward compatibility
        joblib.dump(model, f'{synkinesis_type}_synkinesis_model.pkl')
        joblib.dump(scaler, f'{synkinesis_type}_synkinesis_scaler.pkl')
        feature_importance.to_csv(f'{synkinesis_type}_feature_importance.csv', index=False)
        
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
            logging.FileHandler('synkinesis_model_training.log')
        ]
    )

    logger.info("Starting synkinesis ML model training...")

    # Prepare data for all synkinesis types
    logger.info("Preparing data...")
    synkinesis_datasets = prepare_synkinesis_data()
    
    # Train models for each synkinesis type
    for synkinesis_type, (features, targets) in synkinesis_datasets.items():
        logger.info(f"\n{'='*20} Training {synkinesis_type} model {'='*20}")
        
        # Convert binary targets - combine Partial and Complete as positive class (1)
        binary_targets = np.array([1 if t > 0 else 0 for t in targets])
        
        # Train Random Forest model
        logger.info(f"Training Random Forest model for {synkinesis_type}...")
        rf_model, rf_scaler, rf_importance = train_model(
            features, binary_targets, model_type='random_forest'
        )
        
        # Save model artifacts
        logger.info(f"Saving Random Forest model artifacts for {synkinesis_type}...")
        save_model_artifacts(rf_model, rf_scaler, rf_importance, synkinesis_type)
        
        # Train Gradient Boosting model as alternative
        logger.info(f"Training Gradient Boosting model for {synkinesis_type}...")
        gb_model, gb_scaler, gb_importance = train_model(
            features, binary_targets, model_type='gradient_boosting'
        )
        
        # Save gradient boosting model artifacts
        logger.info(f"Saving Gradient Boosting model artifacts for {synkinesis_type}...")
        os.makedirs(f'models/synkinesis/{synkinesis_type}/gradient_boosting', exist_ok=True)
        joblib.dump(gb_model, f'models/synkinesis/{synkinesis_type}/gradient_boosting/model.pkl')
        joblib.dump(gb_scaler, f'models/synkinesis/{synkinesis_type}/gradient_boosting/scaler.pkl')
        gb_importance.to_csv(f'models/synkinesis/{synkinesis_type}/gradient_boosting/feature_importance.csv', index=False)
        
    logger.info("ML synkinesis model training complete. All models ready for integration.")

if __name__ == "__main__":
    main()