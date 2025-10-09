# facial_paralysis_detection.py (Modified to use Generic Detector)

"""
Unified facial paralysis detection implementation.
Contains detection logic dispatcher. Uses ML flags from constants.
"""

import logging
import numpy as np
import os
import json

# Import constants needed for USE_ML flags (Keep this)
try:
    from facial_au_constants import (
        USE_ML_FOR_UPPER_FACE, USE_ML_FOR_MIDFACE, USE_ML_FOR_LOWER_FACE
    )
except ImportError: # Fallback if constants aren't found
     logging.warning("Could not import ML flags from facial_au_constants. Assuming ML is disabled.")
     USE_ML_FOR_UPPER_FACE, USE_ML_FOR_MIDFACE, USE_ML_FOR_LOWER_FACE = False, False, False

# --- REMOVE individual detector imports ---
# try: from lower_face_detector import LowerFaceParalysisDetector
# except ImportError: logging.error("Failed to import LowerFaceParalysisDetector"); LowerFaceParalysisDetector = None
# try: from mid_face_detector import MidFaceParalysisDetector
# except ImportError: logging.error("Failed to import MidFaceParalysisDetector"); MidFaceParalysisDetector = None
# try: from upper_face_detector import UpperFaceParalysisDetector
# except ImportError: logging.error("Failed to import UpperFaceParalysisDetector"); UpperFaceParalysisDetector = None

# --- IMPORT the new generic detector ---
try:
    from paralysis_detector import ParalysisDetector
except ImportError:
    logging.critical("CRITICAL: Failed to import ParalysisDetector. ML detection will not function.")
    ParalysisDetector = None

# Configure logging
logger = logging.getLogger(__name__) # Assuming configured elsewhere

# --- REMOVE instantiation of individual detectors ---
# ml_lower_face_detector = LowerFaceParalysisDetector() if LowerFaceParalysisDetector else None
# ml_midface_detector = MidFaceParalysisDetector() if MidFaceParalysisDetector else None
# ml_upper_face_detector = UpperFaceParalysisDetector() if UpperFaceParalysisDetector else None

# Keep function signature (includes row_data)
def detect_side_paralysis(
    analyzer_instance, # 'self' from the calling FacialAUAnalyzer
    info, # Results dict for the representative action (for storing details)
    zone, side, aus, values, other_values,
    values_normalized, other_values_normalized,
    side_avg, other_avg, # Keep for potential non-ML fallback? Or remove if purely ML
    zone_paralysis_summary, # The main summary dict to update
    affected_aus_summary, # Summary set/dict to update
    row_data=None, # *** Expect the full patient data row dict ***
    **kwargs # Catch extra args
    ):
    """
    Unified function to detect paralysis for a specific side and zone using ML.
    Uses the generic ParalysisDetector.

    (Args documentation remains the same)

    Returns:
        bool: True if paralysis was detected (result is not 'None' or 'Error'), False otherwise.
               (Note: The main result is written into zone_paralysis_summary).
    """
    if ParalysisDetector is None: # Global check if the detector class failed to import
         logger.error(f"ParalysisDetector class unavailable. Cannot perform ML detection for {side} {zone}.")
         zone_paralysis_summary[side][zone] = 'Error'
         return False

    ml_enabled_flags = {
        'upper': USE_ML_FOR_UPPER_FACE,
        'mid': USE_ML_FOR_MIDFACE,
        'lower': USE_ML_FOR_LOWER_FACE
    }
    ml_enabled = ml_enabled_flags.get(zone, False)

    # --- Check if row_data was provided if ML is enabled ---
    if ml_enabled and row_data is None:
        logger.error(f"detect_side_paralysis ({analyzer_instance.patient_id}) called for {side} {zone} REQUIRING ML but WITHOUT row_data. Cannot use ML detector.")
        # Set error state in the summary dict
        if side in zone_paralysis_summary and zone in zone_paralysis_summary[side]:
            zone_paralysis_summary[side][zone] = 'Error'
        # Also set error in the action-specific info dict for traceability
        if 'paralysis' not in info: info['paralysis'] = {}
        if 'zones' not in info['paralysis']: info['paralysis']['zones'] = {'left': {}, 'right': {}}
        if side not in info['paralysis']['zones']: info['paralysis']['zones'][side] = {}
        info['paralysis']['zones'][side][zone] = 'Error'
        return False # Indicate failure

    # --- Optional Input Logging (Keep if needed) ---
    # test_patient_id = 'IMG_0422'
    # if zone == 'lower' and analyzer_instance.patient_id == test_patient_id:
    #     try: logger.debug(f"MAIN SCRIPT - Full input row_data for {analyzer_instance.patient_id} ({side}, {zone}):\n{json.dumps(row_data, indent=2, default=str)}")
    #     except Exception as log_e: logger.error(f"Error logging full row_data in main script: {log_e}")
    # --- End Logging ---

    try:
        detector_instance = None
        if ml_enabled:
            try:
                # --- Instantiate the GENERIC detector for the required zone ---
                detector_instance = ParalysisDetector(zone=zone)
                # --- Check if detector initialized correctly (loaded artifacts) ---
                if not all([detector_instance.model, detector_instance.scaler, detector_instance.feature_names, detector_instance.extract_features_func]):
                     logger.warning(f"ML detector for {zone} initialized but artifacts/extractor missing. Skipping detection.")
                     zone_paralysis_summary[side][zone] = 'Error'
                     detector_instance = None # Prevent further use
                else:
                     logger.debug(f"Using generic ParalysisDetector(zone='{zone}') for {side}")
            except Exception as e:
                logger.error(f"Failed to instantiate ParalysisDetector for zone '{zone}': {e}", exc_info=True)
                zone_paralysis_summary[side][zone] = 'Error'
                detector_instance = None
        else:
            logger.debug(f"ML not enabled for {zone}. Skipping ML detection.")
            # Optional: Non-ML Fallback Logic Here
            # zone_paralysis_summary[side][zone] = 'None' # Or result from rules
            return False # Return False if only ML is implemented and it's disabled

        # --- Call the detector's interface method ---
        if detector_instance:
             # The detector's detect_paralysis method handles calling internal detect
             # and updating the dictionaries.
             detection_successful = detector_instance.detect_paralysis(
                 self_orig=analyzer_instance, # Pass the 'self' from the calling analyzer instance
                 info=info, # Pass action-specific dict for storing details
                 side=side,
                 row_data=row_data, # *** Pass the crucial row_data ***
                 zone_paralysis_summary=zone_paralysis_summary, # Pass the summary dict TO BE UPDATED
                 # Pass other args that might be needed contextually, though the generic detector might ignore them
                 # aus=aus, values=values, other_values=other_values,
                 # values_normalized=values_normalized, other_values_normalized=other_values_normalized,
                 # affected_aus_by_zone_side=affected_aus_summary # Pass if detector needs it
                 **kwargs
             )
             # The detect_paralysis method should have updated zone_paralysis_summary
             final_result = zone_paralysis_summary.get(side, {}).get(zone, 'Error') # Safely get result
             return final_result in ['Partial', 'Complete'] # Return True if paralysis detected
        else:
            # This case implies ML was enabled but detector failed to load/init
            logger.error(f"Detector instance for zone {zone} is None unexpectedly after check.")
            # Error state should already be set in the summary
            if side in zone_paralysis_summary and zone in zone_paralysis_summary[side]:
                if zone_paralysis_summary[side][zone] != 'Error': # Only set if not already Error
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