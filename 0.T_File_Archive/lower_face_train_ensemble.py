"""
Script to train ensemble detector components.
Should be run after training the base ML model.
"""

import logging
import pandas as pd
import numpy as np
import os
from lower_face_feature_extraction import prepare_data
from ensemble_detector import EnsembleLowerFaceDetector

def main():
    """Main function to train the ensemble detector."""
    # Configure logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ensemble_training.log')
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # Load data
        logger.info("Loading data for ensemble training...")
        features, targets = prepare_data()
        
        # Get some basic info about the data
        unique, counts = np.unique(targets, return_counts=True)
        class_dist = dict(zip(['None', 'Partial', 'Complete'], counts))
        logger.info(f"Class distribution: {class_dist}")
        
        # Train ensemble
        logger.info("Training ensemble components...")
        ensemble = EnsembleLowerFaceDetector()
        ensemble.train_ensemble(features, targets)
        
        logger.info("Ensemble training complete. Models saved to models/ensemble/ directory")
        
    except Exception as e:
        logger.error(f"Error training ensemble: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main()