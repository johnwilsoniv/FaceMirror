"""
Script to validate the trained model on test data.
Analyzes both raw model outputs and threshold-adjusted predictions.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from lower_face_feature_extraction import prepare_data

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

def validate_model():
    """
    Validate the trained model on test data.
    
    Analyzes both raw model outputs and threshold-adjusted predictions.
    Outputs detailed metrics for both approaches.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/model_validation.log')
        ]
    )
    
    logger.info("Starting model validation")
    
    try:
        # Load model and scaler
        logger.info("Loading model and scaler")
        model_path = 'models/lower_face_paralysis_model.pkl'
        scaler_path = 'models/lower_face_paralysis_scaler.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.error("Model or scaler file not found. Please train the model first.")
            return
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load data
        logger.info("Loading data")
        features, targets = prepare_data()
        
        # Split data with a different random seed to create a "new" test set
        _, X_test, _, y_test = train_test_split(
            features, targets, test_size=0.3, random_state=24
        )
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions using raw model
        logger.info("Making raw model predictions")
        raw_preds = model.predict(X_test_scaled)
        raw_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics for raw predictions
        raw_report = classification_report(
            y_test, raw_preds, 
            target_names=['None', 'Partial', 'Complete'],
            output_dict=True
        )
        
        raw_cm = confusion_matrix(y_test, raw_preds)
        
        # Make threshold-adjusted predictions
        logger.info("Making threshold-adjusted predictions")
        adjusted_preds = raw_preds.copy()
        
        # Apply thresholds from conservative approach
        for i in range(len(adjusted_preds)):
            if adjusted_preds[i] == 2:  # If prediction is Complete
                if raw_proba[i][2] < 0.6:  # But confidence is less than 60%
                    if raw_proba[i][0] > 0.3:  # And significant chance it's None
                        adjusted_preds[i] = 0  # Downgrade to None
                    elif raw_proba[i][1] > 0.25:  # Or chance it's Partial
                        adjusted_preds[i] = 1  # Downgrade to Partial
        
        # Calculate metrics for adjusted predictions
        adjusted_report = classification_report(
            y_test, adjusted_preds,
            target_names=['None', 'Partial', 'Complete'],
            output_dict=True
        )
        
        adjusted_cm = confusion_matrix(y_test, adjusted_preds)
        
        # Calculate error patterns
        raw_errors = calculate_error_patterns(y_test, raw_preds)
        adjusted_errors = calculate_error_patterns(y_test, adjusted_preds)
        
        # Log raw model performance
        logger.info("\n--- Raw Model Performance ---")
        logger.info(f"Accuracy: {raw_report['accuracy']:.4f}")
        logger.info(f"Weighted F1: {raw_report['weighted avg']['f1-score']:.4f}")
        
        logger.info("\nClass-specific metrics:")
        logger.info(f"None - Precision: {raw_report['None']['precision']:.4f}, " +
                   f"Recall: {raw_report['None']['recall']:.4f}, " +
                   f"F1: {raw_report['None']['f1-score']:.4f}")
        logger.info(f"Partial - Precision: {raw_report['Partial']['precision']:.4f}, " +
                   f"Recall: {raw_report['Partial']['recall']:.4f}, " +
                   f"F1: {raw_report['Partial']['f1-score']:.4f}")
        logger.info(f"Complete - Precision: {raw_report['Complete']['precision']:.4f}, " +
                   f"Recall: {raw_report['Complete']['recall']:.4f}, " +
                   f"F1: {raw_report['Complete']['f1-score']:.4f}")
        
        logger.info("\nRaw Model Error Patterns:")
        for pattern, count in raw_errors.items():
            if count > 0:
                logger.info(f"{pattern}: {count} cases")
        
        # Log adjusted model performance
        logger.info("\n--- Threshold-Adjusted Performance ---")
        logger.info(f"Accuracy: {adjusted_report['accuracy']:.4f}")
        logger.info(f"Weighted F1: {adjusted_report['weighted avg']['f1-score']:.4f}")
        
        logger.info("\nClass-specific metrics:")
        logger.info(f"None - Precision: {adjusted_report['None']['precision']:.4f}, " +
                   f"Recall: {adjusted_report['None']['recall']:.4f}, " +
                   f"F1: {adjusted_report['None']['f1-score']:.4f}")
        logger.info(f"Partial - Precision: {adjusted_report['Partial']['precision']:.4f}, " +
                   f"Recall: {adjusted_report['Partial']['recall']:.4f}, " +
                   f"F1: {adjusted_report['Partial']['f1-score']:.4f}")
        logger.info(f"Complete - Precision: {adjusted_report['Complete']['precision']:.4f}, " +
                   f"Recall: {adjusted_report['Complete']['recall']:.4f}, " +
                   f"F1: {adjusted_report['Complete']['f1-score']:.4f}")
        
        logger.info("\nAdjusted Model Error Patterns:")
        for pattern, count in adjusted_errors.items():
            if count > 0:
                logger.info(f"{pattern}: {count} cases")
        
        # Compare improvements
        logger.info("\n--- Improvements from Threshold Adjustment ---")
        accuracy_diff = adjusted_report['accuracy'] - raw_report['accuracy']
        f1_diff = adjusted_report['weighted avg']['f1-score'] - raw_report['weighted avg']['f1-score']
        
        logger.info(f"Accuracy change: {accuracy_diff:.4f} ({'+' if accuracy_diff >= 0 else ''}{accuracy_diff*100:.2f}%)")
        logger.info(f"F1 score change: {f1_diff:.4f} ({'+' if f1_diff >= 0 else ''}{f1_diff*100:.2f}%)")
        
        # Calculate change in None-to-Complete errors
        raw_none_to_complete = raw_errors.get('0_to_2', 0)
        adjusted_none_to_complete = adjusted_errors.get('0_to_2', 0)
        n2c_reduction = raw_none_to_complete - adjusted_none_to_complete
        
        logger.info(f"None-to-Complete errors reduced by: {n2c_reduction} ({n2c_reduction/max(raw_none_to_complete, 1)*100:.2f}%)")
        
        # Calculate change in Partial recall
        partial_recall_diff = adjusted_report['Partial']['recall'] - raw_report['Partial']['recall']
        logger.info(f"Partial recall change: {partial_recall_diff:.4f} ({'+' if partial_recall_diff >= 0 else ''}{partial_recall_diff*100:.2f}%)")
        
        logger.info("\nModel validation complete.")
        
    except Exception as e:
        logger.error(f"Error in model validation: {str(e)}", exc_info=True)
        
def calculate_error_patterns(true_labels, predictions):
    """
    Calculate the patterns of errors in predictions.
    
    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        
    Returns:
        dict: Counts of each error pattern
    """
    error_patterns = {}
    
    for true, pred in zip(true_labels, predictions):
        if true != pred:
            pattern = f"{true}_to_{pred}"
            if pattern not in error_patterns:
                error_patterns[pattern] = 0
            error_patterns[pattern] += 1
    
    return error_patterns

if __name__ == "__main__":
    validate_model()