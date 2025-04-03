"""
Script to integrate the ensemble detector into the facial paralysis detection system.
Demonstrates how to use the ensemble detector in the main system.
"""

import logging
import numpy as np
import pandas as pd
import os
import joblib
from lower_face_ensemble_detector import EnsembleLowerFaceDetector

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

def integrate_ensemble_detector():
    """
    Integrate the ensemble detector into the paralysis detection system.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ensemble_integration.log')
        ]
    )
    
    logger.info("Setting up ensemble integration")
    
    try:
        # Initialize the ensemble detector
        logger.info("Initializing ensemble detector...")
        detector = EnsembleLowerFaceDetector()
        
        # Check if the detector is loaded
        if detector.ml_detector is None:
            logger.error("Failed to load base ML detector")
            return False
            
        logger.info("Base ML detector loaded successfully")
        
        # Check if specialist models exist
        specialist_exists = os.path.exists('models/ensemble/specialist_classifier.pkl')
        if not specialist_exists:
            logger.warning("Specialist classifier not found - will use base model only")
        else:
            logger.info("Specialist classifier found")
            # Load specialist models
            try:
                detector.specialist_classifier = joblib.load('models/ensemble/specialist_classifier.pkl')
                detector.specialist_scaler = joblib.load('models/ensemble/specialist_scaler.pkl')
                logger.info("Specialist models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading specialist models: {str(e)}")
        
        # Define the integration function
        logger.info("Defining integration function to be called from main system...")
        
        def detect_lower_face_paralysis(self_orig, info, zone, side, aus, values, other_values,
                                       values_normalized, other_values_normalized,
                                       zone_paralysis, affected_aus_by_zone_side, **kwargs):
            """
            Ensemble-based detection for lower face paralysis.
            
            Args:
                self_orig: The original FacialParalysisDetector instance
                info (dict): Results dictionary for current action
                zone (str): Facial zone being analyzed ('lower')
                side (str): Side being analyzed ('left' or 'right')
                aus (list): List of Action Units for this zone
                values (dict): AU values for this side
                other_values (dict): AU values for other side
                values_normalized (dict): Normalized AU values for this side
                other_values_normalized (dict): Normalized AU values for other side
                zone_paralysis (dict): Track paralysis results at patient level
                affected_aus_by_zone_side (dict): Track affected AUs
                **kwargs: Additional arguments (not used)
                
            Returns:
                bool: True if paralysis was detected, False otherwise
            """
            try:
                # Use ensemble detector
                result, confidence, details = detector.detect(
                    info, side, zone, aus, values, other_values,
                    values_normalized, other_values_normalized
                )
                
                # Update the info structure with detection result
                info['paralysis']['zones'][side][zone] = result
                
                # Track for patient-level assessment
                if result == 'Complete':
                    zone_paralysis[side][zone] = 'Complete'
                elif result == 'Partial' and zone_paralysis[side][zone] == 'None':
                    zone_paralysis[side][zone] = 'Partial'
                
                # Add affected AUs - only if paralysis detected
                if result != 'None':
                    if 'AU12_r' in values:
                        affected_aus_by_zone_side[side][zone].add('AU12_r')
                        if 'AU12_r' not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append('AU12_r')
                
                    if 'AU25_r' in values:
                        affected_aus_by_zone_side[side][zone].add('AU25_r')
                        if 'AU25_r' not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append('AU25_r')
                
                # Store confidence score
                info['paralysis']['confidence'][side][zone] = confidence
                
                # Add ensemble details
                if 'ensemble_details' not in info['paralysis']:
                    info['paralysis']['ensemble_details'] = {}
                
                info['paralysis']['ensemble_details'][f"{side}_{zone}"] = details
                
                # Add contributing AUs info for consistency
                if result != 'None':
                    # Initialize ensemble detection list if needed
                    if 'ensemble_detection' not in info['paralysis']['contributing_aus'][side][zone]:
                        info['paralysis']['contributing_aus'][side][zone]['ensemble_detection'] = []
                
                    # Track AU values for ensemble detection
                    info['paralysis']['contributing_aus'][side][zone]['ensemble_detection'].append({
                        'au': 'AU12_r',
                        'side_value': values.get('AU12_r', 0),
                        'other_value': other_values.get('AU12_r', 0),
                        'ensemble_confidence': confidence,
                        'type': result,
                        'specialist_used': details.get('specialist_used', False)
                    })
                
                    if 'AU25_r' in values:
                        info['paralysis']['contributing_aus'][side][zone]['ensemble_detection'].append({
                            'au': 'AU25_r',
                            'side_value': values.get('AU25_r', 0),
                            'other_value': other_values.get('AU25_r', 0),
                            'ensemble_confidence': confidence,
                            'type': result,
                            'specialist_used': details.get('specialist_used', False)
                        })
                
                # Log the detection result
                logger.debug(f"{side} lower face: {result} paralysis detected with ensemble confidence {confidence:.3f}")
                
                # Return success if paralysis was detected
                return result != 'None'
            
            except Exception as e:
                logger.error(f"Exception in ensemble detect_lower_face_paralysis: {str(e)}")
                # Fall back to the original ML detector method
                if hasattr(self_orig, 'lower_face_ml_detector') and self_orig.lower_face_ml_detector:
                    logger.info("Falling back to original ML detector")
                    return self_orig.lower_face_ml_detector.detect_lower_face_paralysis(
                        self_orig, info, zone, side, aus, values, other_values,
                        values_normalized, other_values_normalized,
                        zone_paralysis, affected_aus_by_zone_side, **kwargs
                    )
                return False
        
        return detect_lower_face_paralysis
        
    except Exception as e:
        logger.error(f"Error setting up ensemble integration: {str(e)}", exc_info=True)
        return None

def integrate_with_system():
    """
    Integrate the ensemble detector with the main system.
    
    This should be called at system initialization.
    """
    try:
        from facial_au_paralysis_detector import FacialParalysisDetector
        
        # Get the ensemble detection function
        ensemble_detect = integrate_ensemble_detector()
        
        if ensemble_detect:
            # Check different ways the ML detector might be accessed
            # First, check if it's an instance attribute
            found = False
            
            # Create a test instance to check its attributes
            test_instance = FacialParalysisDetector()
            
            # Method 1: Check if lower_face_ml_detector is a direct instance attribute
            if hasattr(test_instance, 'lower_face_ml_detector') and test_instance.lower_face_ml_detector:
                logger.info("Found ML detector as instance attribute")
                # Store the original method for fallback
                test_instance.lower_face_ml_detector._original_detect = \
                    test_instance.lower_face_ml_detector.detect_lower_face_paralysis
                
                # Create a function that will be called for each instance
                def detector_patch(self):
                    if not hasattr(self.lower_face_ml_detector, '_original_detect'):
                        self.lower_face_ml_detector._original_detect = \
                            self.lower_face_ml_detector.detect_lower_face_paralysis
                    
                    self.lower_face_ml_detector.detect_lower_face_paralysis = ensemble_detect
                    logger.info("Patched instance ML detector")
                
                # Patch the __init__ method to apply our patch to each new instance
                original_init = FacialParalysisDetector.__init__
                def patched_init(self, *args, **kwargs):
                    original_init(self, *args, **kwargs)
                    detector_patch(self)
                
                FacialParalysisDetector.__init__ = patched_init
                found = True
            
            # Method 2: Check for a class attribute (could be used in factory pattern)
            elif hasattr(FacialParalysisDetector, 'lower_face_ml_detector') and FacialParalysisDetector.lower_face_ml_detector:
                logger.info("Found ML detector as class attribute")
                # Store the original method for fallback
                FacialParalysisDetector.lower_face_ml_detector._original_detect = \
                    FacialParalysisDetector.lower_face_ml_detector.detect_lower_face_paralysis
                
                # Replace with ensemble method
                FacialParalysisDetector.lower_face_ml_detector.detect_lower_face_paralysis = ensemble_detect
                found = True
                
            # Method 3: Try to find it through imports
            else:
                try:
                    import lower_face_ml_detector as detector_module
                    logger.info("Found ML detector through import")
                    
                    # Check if LowerFaceParalysisDetector class exists
                    if hasattr(detector_module, 'LowerFaceParalysisDetector'):
                        # Store the original method for fallback
                        detector_module.LowerFaceParalysisDetector.detect_lower_face_paralysis._original = \
                            detector_module.LowerFaceParalysisDetector.detect_lower_face_paralysis
                        
                        # Replace with ensemble method
                        detector_module.LowerFaceParalysisDetector.detect_lower_face_paralysis = ensemble_detect
                        found = True
                except ImportError:
                    logger.warning("Could not import lower_face_ml_detector module")
            
            if found:
                logger.info("Ensemble detector successfully integrated")
                return True
            else:
                logger.error("ML detector not found through any known method")
                return False
        
        return False
    
    except Exception as e:
        logger.error(f"Error during system integration: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ensemble_integration.log')
        ]
    )
    
    # Perform integration
    logger.info("Starting ensemble integration")
    success = integrate_with_system()
    
    if success:
        logger.info("Ensemble integration completed successfully")
    else:
        logger.error("Ensemble integration failed")