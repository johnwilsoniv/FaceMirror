"""
Facial synkinesis detection module.
Analyzes facial Action Units to detect possible synkinesis using ML-based detection.
"""

import logging
import sys
from facial_au_constants import SYNKINESIS_TYPES

# Import ML-based detector
from ml_synkinesis_detector import MLSynkinesisDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialSynkinesisDetector:
    """
    Detects facial synkinesis (unwanted co-activation of muscles)
    using ML-based detection methods.
    """
    
    def __init__(self):
        """Initialize the facial synkinesis detector with ML models."""
        # Initialize synkinesis types
        self.SYNKINESIS_TYPES = SYNKINESIS_TYPES

        # Initialize ML-based detector
        try:
            self.ml_detector = MLSynkinesisDetector()
            logger.info("ML-based synkinesis detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML-based synkinesis detector: {str(e)}")
            logger.error("Synkinesis detection will not be available.")
            self.ml_detector = None

    def detect_synkinesis(self, results):
        """
        Detect potential synkinesis (unwanted co-activation of muscles)
        using machine learning models.

        Args:
            results (dict): Results dictionary with AU values for each action

        Returns:
            None: Updates results dictionary in place
        """
        if not results:
            logger.warning("No results to analyze for synkinesis detection")
            return

        # Check if ML detector is available
        if self.ml_detector is None:
            logger.error("ML synkinesis detector not available. Cannot perform synkinesis detection.")
            return

        # Initialize the synkinesis data structure in results
        self._initialize_synkinesis_structure(results)
        
        # Run ML-based detection
        try:
            logger.info("Detecting synkinesis using ML-based models")
            self.ml_detector.detect_synkinesis(results)
            
            # Log detection summary
            self._log_detection_summary(results)
        except Exception as e:
            logger.error(f"Error in ML-based synkinesis detection: {str(e)}")
            logger.error("Synkinesis detection failed.")
            
    def _initialize_synkinesis_structure(self, results):
        """
        Initialize synkinesis data structure in results if not already present.
        
        Args:
            results (dict): Results dictionary with AU values for each action
        """
        for action, info in results.items():
            # Initialize synkinesis structure with Yes/No for each type and side
            if 'synkinesis' not in info:
                info['synkinesis'] = {
                    'detected': False,
                    'types': [],
                    'side_specific': {
                        synk_type: {'left': False, 'right': False}
                        for synk_type in self.SYNKINESIS_TYPES
                    },
                    'confidence': {
                        synk_type: {'left': 0, 'right': 0}
                        for synk_type in self.SYNKINESIS_TYPES
                    },
                    'contributing_aus': {}  # Initialize the structure for tracking contributing AUs
                }

            # Make sure contributing_aus exists even if this is older data
            if 'contributing_aus' not in info['synkinesis']:
                info['synkinesis']['contributing_aus'] = {}

    def _log_detection_summary(self, results):
        """
        Log a summary of detected synkinesis patterns.
        
        Args:
            results (dict): Results dictionary with AU values for each action
        """
        detected_types = {}
        for action, info in results.items():
            if info.get('synkinesis', {}).get('detected', False):
                for synk_type in info['synkinesis'].get('types', []):
                    if synk_type not in detected_types:
                        detected_types[synk_type] = []
                    
                    # Check which side(s) had this synkinesis type
                    sides = []
                    for side in ['left', 'right']:
                        if info['synkinesis']['side_specific'][synk_type][side]:
                            sides.append(side)
                            
                    detected_types[synk_type].append({
                        'action': action,
                        'sides': sides,
                        'confidence': {
                            side: info['synkinesis']['confidence'][synk_type][side]
                            for side in sides
                        }
                    })
        
        if detected_types:
            logger.info("========== SYNKINESIS DETECTION SUMMARY ==========")
            for synk_type, occurrences in detected_types.items():
                logger.info(f"{synk_type} synkinesis detected {len(occurrences)} times:")
                for occurrence in occurrences:
                    sides_str = ", ".join(occurrence['sides'])
                    confidence_str = ", ".join([f"{side}: {conf:.2f}" for side, conf in occurrence['confidence'].items()])
                    logger.info(f"  - Action: {occurrence['action']}, Sides: {sides_str}, Confidence: {confidence_str}")
        else:
            logger.info("No synkinesis detected in any action")
