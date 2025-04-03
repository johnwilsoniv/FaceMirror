"""
Script to train specialist classifier for partial vs complete paralysis cases.
Should be run after training the base ML model.
"""

import logging
import pandas as pd
import numpy as np
import os
from lower_face_feature_extraction import prepare_data
from lower_face_ensemble_detector import EnsembleLowerFaceDetector

def main():
    """Main function to train the specialist classifier."""
    # Configure logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/specialist_training.log')
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # Load data
        logger.info("Loading data for specialist training...")
        features, targets = prepare_data()
        
        # Get some basic info about the data
        unique, counts = np.unique(targets, return_counts=True)
        class_dist = dict(zip(['None', 'Partial', 'Complete'], counts))
        logger.info(f"Full dataset class distribution: {class_dist}")
        
        # Initialize the ensemble detector
        logger.info("Training specialist classifier...")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Filter for only cases that are Partial or Complete
        is_paralysis = (targets > 0)
        specialist_features = features[is_paralysis]
        specialist_targets = targets[is_paralysis] - 1  # Shift to 0=Partial, 1=Complete
        
        logger.info(f"Specialist training data size: {len(specialist_features)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            specialist_features, specialist_targets, test_size=0.25, random_state=42, stratify=specialist_targets
        )
        
        # Scale features
        specialist_scaler = StandardScaler()
        X_train_scaled = specialist_scaler.fit_transform(X_train)
        X_test_scaled = specialist_scaler.transform(X_test)
        
        # Train Random Forest with emphasis on Partial class
        specialist_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight={0: 2.0, 1: 1.0},  # Emphasize Partial class
            bootstrap=True,
            random_state=42
        )
        
        # Fit model
        specialist_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = specialist_classifier.score(X_train_scaled, y_train)
        test_acc = specialist_classifier.score(X_test_scaled, y_test)
        
        logger.info(f"Specialist classifier training accuracy: {train_acc:.4f}")
        logger.info(f"Specialist classifier testing accuracy: {test_acc:.4f}")
        
        # Save specialist components
        logger.info("Saving specialist components")
        os.makedirs('models/ensemble', exist_ok=True)
        joblib.dump(specialist_classifier, 'models/ensemble/specialist_classifier.pkl')
        joblib.dump(specialist_scaler, 'models/ensemble/specialist_scaler.pkl')
        
        logger.info("Specialist training complete. Models saved to models/ensemble/ directory")
        
    except Exception as e:
        logger.error(f"Error training specialist: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main()