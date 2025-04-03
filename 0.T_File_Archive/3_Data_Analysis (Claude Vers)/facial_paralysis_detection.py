"""
Unified facial paralysis detection implementation.
Contains detection logic for all facial zones with ML-based approach for all zones.
"""

import logging
import numpy as np
import os
from facial_au_constants import (
    EXTREME_ASYMMETRY_THRESHOLD, MID_FACE_FUNCTIONAL_THRESHOLD,
    MID_FACE_FUNCTION_RATIO_OVERRIDE,
    USE_ML_FOR_UPPER_FACE, USE_ML_FOR_MIDFACE, USE_ML_FOR_LOWER_FACE
)

from facial_paralysis_helpers import calculate_percent_difference
from lower_face_detector import LowerFaceParalysisDetector  # Updated import
from mid_face_detector import MidFaceParalysisDetector
from ml_upper_face_detector import MLUpperFaceParalysisDetector

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

# Import the ML-based detectors
ml_lower_face_detector = LowerFaceParalysisDetector()  # Updated class name
ml_midface_detector = MidFaceParalysisDetector()
ml_upper_face_detector = MLUpperFaceParalysisDetector()


def detect_side_paralysis(self, info, zone, side, aus, values, other_values,
                          values_normalized, other_values_normalized,
                          side_avg, other_avg, zone_paralysis, affected_aus_by_zone_side,
                          partial_thresholds, complete_thresholds, asymmetry_thresholds, confidence_thresholds):
    """
    Unified function to detect paralysis for a specific side and zone.
    Uses ML-based detection for all facial zones.

    Args:
        self: The FacialParalysisDetector instance
        info (dict): Results dictionary for current action
        zone (str): Facial zone being analyzed ('upper', 'mid', or 'lower')
        side (str): Side being analyzed ('left' or 'right')
        aus (list): List of Action Units for this zone
        values (dict): AU values for this side
        other_values (dict): AU values for other side
        values_normalized (dict): Normalized AU values for this side
        other_values_normalized (dict): Normalized AU values for other side
        side_avg (float): Weighted average activation for this side
        other_avg (float): Weighted average activation for the other side
        zone_paralysis (dict): Track paralysis results at patient level
        affected_aus_by_zone_side (dict): Track affected AUs
        partial_thresholds (dict): Thresholds for partial paralysis detection
        complete_thresholds (dict): Thresholds for complete paralysis detection
        asymmetry_thresholds (dict): Thresholds for asymmetry detection
        confidence_thresholds (dict): Thresholds for confidence scoring

    Returns:
        bool: True if paralysis was detected, False otherwise
    """
    try:
        # Dispatch to zone-specific ML detection function
        if zone == 'upper':
            # Use ML-based detector for upper face
            result = ml_upper_face_detector.detect_upper_face_paralysis(
                self, info, zone, side, aus, values, other_values,
                values_normalized, other_values_normalized,
                zone_paralysis, affected_aus_by_zone_side,
                asymmetry_thresholds, confidence_thresholds
            )
        elif zone == 'mid':
            # Use ML-based detector for midface
            result = ml_midface_detector.detect_paralysis(
                self, info, zone, side, aus, values, other_values,
                values_normalized, other_values_normalized,
                zone_paralysis, affected_aus_by_zone_side,
                asymmetry_thresholds, confidence_thresholds
            )
        elif zone == 'lower':
            # Use ML-based detector for lower face
            result = ml_lower_face_detector.detect_paralysis(
                self, info, zone, side, aus, values, other_values,
                values_normalized, other_values_normalized,
                zone_paralysis, affected_aus_by_zone_side,
                asymmetry_thresholds, confidence_thresholds
            )
        else:
            logger.warning(f"Unknown zone: {zone}")
            return False

        return result

    except Exception as e:
        logger.error(f"Exception in detect_side_paralysis for {side} {zone}: {str(e)}")
        raise  # Propagate the error upward since we don't have fallback mechanisms