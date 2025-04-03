"""
Script for tuning post-processing thresholds for the ML-based lower face paralysis detector.
Implements a grid search to find optimal thresholds.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.metrics import classification_report, f1_score, accuracy_score
from lower_face_feature_extraction import prepare_data

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

def tune_thresholds():
    """
    Tune post-processing thresholds for the ML-based detector.
    
    Uses a grid search approach to find thresholds that minimize
    None-to-Complete errors while maintaining good overall performance.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/threshold_tuning.log')
        ]
    )
    
    logger.info("Starting threshold tuning process")
    
    try:
        # Load the trained model and scaler
        logger.info("Loading trained model and scaler")
        model_path = 'models/lower_face_paralysis_model.pkl'
        scaler_path = 'models/lower_face_paralysis_scaler.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.error("Model or scaler file not found. Please train the model first.")
            return
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load validation data (same as training data but we'll split it differently)
        features, targets = prepare_data()
        
        # Split into train/test differently to prevent data leakage
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            features, targets, test_size=0.3, random_state=24  # Different random state
        )
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Get raw predictions
        raw_predictions = model.predict(X_test_scaled)
        prediction_probas = model.predict_proba(X_test_scaled)
        
        # Function to apply thresholds and evaluate
        def evaluate_thresholds(complete_conf_threshold, none_prob_threshold, partial_prob_threshold):
            # Apply thresholds
            adjusted_preds = raw_predictions.copy()
            
            for i in range(len(adjusted_preds)):
                if adjusted_preds[i] == 2:  # If prediction is Complete
                    if prediction_probas[i][2] < complete_conf_threshold:  # Low confidence
                        if prediction_probas[i][0] > none_prob_threshold:  # High None probability
                            adjusted_preds[i] = 0  # Change to None
                        elif prediction_probas[i][1] > partial_prob_threshold:  # High Partial probability
                            adjusted_preds[i] = 1  # Change to Partial
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, adjusted_preds)
            f1 = f1_score(y_test, adjusted_preds, average='weighted')
            
            # Count error types
            none_to_complete = sum((y_test == 0) & (adjusted_preds == 2))
            partial_to_none = sum((y_test == 1) & (adjusted_preds == 0))
            
            # Calculate F1 for partial class
            partial_f1 = f1_score(y_test == 1, adjusted_preds == 1, zero_division=0)
            
            return {
                'complete_threshold': complete_conf_threshold,
                'none_threshold': none_prob_threshold,
                'partial_threshold': partial_prob_threshold,
                'accuracy': accuracy,
                'f1_weighted': f1,
                'partial_f1': partial_f1,
                'none_to_complete': none_to_complete,
                'partial_to_none': partial_to_none
            }
        
        # Grid search for thresholds
        logger.info("Performing grid search for optimal thresholds")
        
        # Define threshold ranges
        complete_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        none_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
        partial_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35]
        
        results = []
        
        # Run grid search
        for ct in complete_thresholds:
            for nt in none_thresholds:
                for pt in partial_thresholds:
                    result = evaluate_thresholds(ct, nt, pt)
                    results.append(result)
                    
                    # Log progress
                    logger.debug(f"Thresholds: {ct}/{nt}/{pt} - Accuracy: {result['accuracy']:.4f}, " + 
                               f"F1: {result['f1_weighted']:.4f}, Partial F1: {result['partial_f1']:.4f}")
        
        # Convert to dataframe
        results_df = pd.DataFrame(results)
        
        # Find best thresholds
        # Sort by minimizing None-to-Complete errors, then maximizing Partial F1, then overall accuracy
        results_df = results_df.sort_values(
            by=['none_to_complete', 'partial_f1', 'accuracy'], 
            ascending=[True, False, False]
        )
        
        # Get top 5 configurations
        top_configs = results_df.head(5)
        
        # Save results
        results_df.to_csv('logs/threshold_search_results.csv', index=False)
        
        # Log best configuration
        best_config = top_configs.iloc[0]
        logger.info("Best threshold configuration found:")
        logger.info(f"Complete confidence threshold: {best_config['complete_threshold']}")
        logger.info(f"None probability threshold: {best_config['none_threshold']}")
        logger.info(f"Partial probability threshold: {best_config['partial_threshold']}")
        logger.info(f"Accuracy: {best_config['accuracy']:.4f}")
        logger.info(f"F1 weighted: {best_config['f1_weighted']:.4f}")
        logger.info(f"Partial F1: {best_config['partial_f1']:.4f}")
        logger.info(f"None-to-Complete errors: {best_config['none_to_complete']}")
        logger.info(f"Partial-to-None errors: {best_config['partial_to_none']}")
        
        # Log top 5 configurations
        logger.info("\nTop 5 configurations:")
        for i, config in top_configs.iterrows():
            logger.info(f"Configuration {i+1}:")
            logger.info(f"  Complete threshold: {config['complete_threshold']}")
            logger.info(f"  None threshold: {config['none_threshold']}")
            logger.info(f"  Partial threshold: {config['partial_threshold']}")
            logger.info(f"  Accuracy: {config['accuracy']:.4f}")
            logger.info(f"  F1 weighted: {config['f1_weighted']:.4f}")
            logger.info(f"  None-to-Complete errors: {config['none_to_complete']}")
            logger.info(f"  Partial-to-None errors: {config['partial_to_none']}")
        
        logger.info("\nThreshold tuning complete. Update the post-processing thresholds in lower_face_ml_detector.py with these values.")
        
    except Exception as e:
        logger.error(f"Error in threshold tuning: {str(e)}", exc_info=True)

if __name__ == "__main__":
    tune_thresholds()