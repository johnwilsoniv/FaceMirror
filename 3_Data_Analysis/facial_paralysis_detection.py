# facial_paralysis_detection.py (Modified with Full Input Logging)

"""
Unified facial paralysis detection implementation.
Contains detection logic dispatcher. Assumes ML flags in constants.
"""

import logging
import numpy as np
import os
import json # <--- Added import

# Import constants needed for USE_ML flags
try:
    from facial_au_constants import (
        USE_ML_FOR_UPPER_FACE, USE_ML_FOR_MIDFACE, USE_ML_FOR_LOWER_FACE
    )
except ImportError: # Fallback if constants aren't found
     logging.warning("Could not import ML flags from facial_au_constants. Assuming ML is disabled.")
     USE_ML_FOR_UPPER_FACE, USE_ML_FOR_MIDFACE, USE_ML_FOR_LOWER_FACE = False, False, False

# Import zone-specific detectors (ML versions)
# Handle potential ImportErrors gracefully
try: from lower_face_detector import LowerFaceParalysisDetector
except ImportError: logging.error("Failed to import LowerFaceParalysisDetector"); LowerFaceParalysisDetector = None
try: from mid_face_detector import MidFaceParalysisDetector
except ImportError: logging.error("Failed to import MidFaceParalysisDetector"); MidFaceParalysisDetector = None
try: from upper_face_detector import UpperFaceParalysisDetector
except ImportError: logging.error("Failed to import UpperFaceParalysisDetector"); UpperFaceParalysisDetector = None


# Configure logging
# (Assuming logging is configured elsewhere, e.g., in main or analyzer)
logger = logging.getLogger(__name__)

# --- Instantiate ML detectors ---
# These are instantiated once when the module is imported
ml_lower_face_detector = LowerFaceParalysisDetector() if LowerFaceParalysisDetector else None
ml_midface_detector = MidFaceParalysisDetector() if MidFaceParalysisDetector else None
ml_upper_face_detector = UpperFaceParalysisDetector() if UpperFaceParalysisDetector else None
# --- End Instantiate ---


# --- MODIFIED function signature to accept row_data ---
def detect_side_paralysis(
    analyzer_instance, # 'self' from the calling FacialAUAnalyzer
    info, # Results dict for the representative action (for storing details)
    zone, side, aus, values, other_values,
    values_normalized, other_values_normalized,
    side_avg, other_avg, # Keep for potential non-ML fallback? Or remove if purely ML
    zone_paralysis_summary, # The main summary dict to update
    affected_aus_summary, # Summary set/dict to update
    row_data=None, # *** ADDED: Expect the full patient data row dict ***
    **kwargs # Catch extra args
    ):
    """
    Unified function to detect paralysis for a specific side and zone using ML.

    Args:
        analyzer_instance: The FacialAUAnalyzer instance.
        info (dict): Results dictionary for a representative action (used to store details).
        zone (str): Facial zone ('upper', 'mid', or 'lower').
        side (str): Side ('left' or 'right').
        aus (list): List of Action Units for this zone.
        values (dict): AU values for this side (from representative action).
        other_values (dict): AU values for other side (from representative action).
        values_normalized (dict): Normalized AU values for this side (from representative action).
        other_values_normalized (dict): Normalized AU values for other side (from representative action).
        side_avg (float): Weighted average (potentially unused by ML).
        other_avg (float): Weighted average (potentially unused by ML).
        zone_paralysis_summary (dict): Patient-level summary dict to update with detection result.
        affected_aus_summary (dict): Patient-level summary dict for affected AUs.
        row_data (dict or pd.Series, optional): Full data row for the patient. REQUIRED for ML.
        **kwargs: Catches any extra arguments.

    Returns:
        bool: True if paralysis was detected (result is not 'None' or 'Error'), False otherwise.
               (Note: The main result is written into zone_paralysis_summary).
    """
    ml_needed = (
        (zone == 'upper' and USE_ML_FOR_UPPER_FACE) or
        (zone == 'mid' and USE_ML_FOR_MIDFACE) or
        (zone == 'lower' and USE_ML_FOR_LOWER_FACE)
    )

    # --- Check if row_data was provided if ML is needed ---
    if ml_needed and row_data is None:
        logger.error(f"detect_side_paralysis ({analyzer_instance.patient_id}) called for {side} {zone} REQUIRING ML but WITHOUT row_data. Cannot use ML detectors.")
        # Set error state in the summary dict
        if side in zone_paralysis_summary and zone in zone_paralysis_summary[side]:
            zone_paralysis_summary[side][zone] = 'Error'
        # Also set error in the action-specific info dict for traceability
        if 'paralysis' not in info: info['paralysis'] = {}
        if 'zones' not in info['paralysis']: info['paralysis']['zones'] = {'left': {}, 'right': {}}
        if side not in info['paralysis']['zones']: info['paralysis']['zones'][side] = {}
        info['paralysis']['zones'][side][zone] = 'Error'
        return False # Indicate failure

    # --- START ADDED LOGGING ---
    # Log the full input dictionary for the specific patient and zone
    test_patient_id = 'IMG_0422' # <<< CHANGE THIS if your test patient is different
    if zone == 'lower' and analyzer_instance.patient_id == test_patient_id:
        try:
            logger.debug(f"MAIN SCRIPT - Full input row_data (ml_input_dict) for {analyzer_instance.patient_id} ({side}, {zone}):")
            # Use default=str to handle potential non-serializable types like NaN (though generate_ml should remove them)
            logger.debug(json.dumps(row_data, indent=2, default=str))
        except Exception as log_e:
            logger.error(f"Error logging full row_data in main script: {log_e}")
    # --- END ADDED LOGGING ---

    try:
        detector_instance = None
        use_ml = False

        # Determine which detector instance to use based on zone and config flags
        if zone == 'upper' and USE_ML_FOR_UPPER_FACE and ml_upper_face_detector:
            detector_instance = ml_upper_face_detector
            use_ml = True
            logger.debug(f"Using UpperFaceParalysisDetector (ML) for {side} {zone}")
        elif zone == 'mid' and USE_ML_FOR_MIDFACE and ml_midface_detector:
            detector_instance = ml_midface_detector
            use_ml = True
            logger.debug(f"Using MidFaceParalysisDetector (ML) for {side} {zone}")
        elif zone == 'lower' and USE_ML_FOR_LOWER_FACE and ml_lower_face_detector:
            detector_instance = ml_lower_face_detector
            use_ml = True
            logger.debug(f"Using LowerFaceParalysisDetector (ML) for {side} {zone}")
        else:
            # --- Optional: Non-ML Fallback Logic ---
            if not ml_needed:
                 logger.debug(f"ML not enabled for {zone}. Skipping ML detection.")
                 # If you had rule-based logic, call it here and update zone_paralysis_summary
                 # e.g., result = rule_based_detector(...)
                 # zone_paralysis_summary[side][zone] = result
                 # return result != 'None' and result != 'Error'
                 return False # Return False if only ML is implemented and it's disabled
            else:
                 logger.warning(f"ML detector configured but instance not available for zone: {zone}. Skipping detection.")
                 zone_paralysis_summary[side][zone] = 'Error' # Mark as error if ML was expected but unavailable
                 return False

        # --- Call the selected detector's interface method ---
        if detector_instance:
             # Ensure the detector has the expected interface method
             if not hasattr(detector_instance, 'detect_paralysis'):
                  logger.error(f"Detector instance for {zone} does not have a 'detect_paralysis' method.")
                  zone_paralysis_summary[side][zone] = 'Error'
                  return False

             # --- MODIFIED: Pass row_data explicitly to the detector's interface method ---
             # The detector's method is responsible for calling its internal 'detect' with row_data
             detection_successful = detector_instance.detect_paralysis(
                 self_orig=analyzer_instance, # Pass the 'self' from the calling analyzer instance
                 info=info, # Pass action-specific dict for storing details
                 zone=zone,
                 side=side,
                 aus=aus, # Pass these for context if detector needs them
                 values=values,
                 other_values=other_values,
                 values_normalized=values_normalized,
                 other_values_normalized=other_values_normalized,
                 zone_paralysis=zone_paralysis_summary, # Pass the summary dict TO BE UPDATED
                 affected_aus_by_zone_side=affected_aus_summary, # Pass summary dict TO BE UPDATED
                 row_data=row_data, # *** THE CRUCIAL ADDITION ***
                 **kwargs # Pass any other optional arguments
             )
             # The detect_paralysis method should have updated zone_paralysis_summary
             final_result = zone_paralysis_summary[side][zone]
             return final_result != 'None' and final_result != 'Error'
             # --- END MODIFICATION ---
        else:
            # This case should be caught earlier, but handle defensively
            logger.error(f"Detector instance for zone {zone} is None unexpectedly.")
            zone_paralysis_summary[side][zone] = 'Error'
            return False

    except Exception as e:
        logger.error(f"Exception in detect_side_paralysis for {side} {zone}: {str(e)}", exc_info=True)
        # Ensure summary dict is updated with error status
        if side in zone_paralysis_summary and zone in zone_paralysis_summary[side]:
             zone_paralysis_summary[side][zone] = 'Error'
        # Also update action-specific info
        if 'paralysis' not in info: info['paralysis'] = {}
        if 'zones' not in info['paralysis']: info['paralysis']['zones'] = {'left': {}, 'right': {}}
        if side not in info['paralysis']['zones']: info['paralysis']['zones'][side] = {}
        info['paralysis']['zones'][side][zone] = 'Error'
        return False