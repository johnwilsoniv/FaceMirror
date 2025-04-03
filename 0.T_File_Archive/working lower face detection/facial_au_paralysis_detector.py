"""
Facial paralysis detection module.
Analyzes facial Action Units to detect possible facial paralysis.
"""

import numpy as np
import logging
from facial_au_constants import (
    FACIAL_ZONES, ZONE_SPECIFIC_ACTIONS, PARALYSIS_THRESHOLDS,
    ASYMMETRY_THRESHOLDS, CONFIDENCE_THRESHOLDS, FACIAL_ZONE_WEIGHTS,
    EXTREME_ASYMMETRY_THRESHOLD, AU_ZONE_DETECTION_MODIFIERS,
    PARALYSIS_DETECTION_AU_IMPORTANCE, BASELINE_AU_ACTIVATIONS,
    AU12_EXTREME_ASYMMETRY_THRESHOLD,
    AU45_EXTREME_ASYMMETRY_THRESHOLD, AU07_EXTREME_ASYMMETRY_THRESHOLD,  # Added AU07 threshold
    # Lower face constants
    USE_ML_FOR_MIDFACE, USE_ML_FOR_LOWER_FACE,  # Added ML flags
    FLAG_BORDERLINE_FOR_REVIEW
)

# Configure minimal logging - only errors and critical info
logger = logging.getLogger(__name__)


class FacialParalysisDetector:
    """
    Detects facial paralysis by analyzing AU asymmetry patterns by facial zone.
    """

    def __init__(self):
        """Initialize the facial paralysis detector."""
        self.facial_zones = FACIAL_ZONES
        self.zone_specific_actions = ZONE_SPECIFIC_ACTIONS
        self.paralysis_thresholds = PARALYSIS_THRESHOLDS
        self.asymmetry_thresholds = ASYMMETRY_THRESHOLDS
        self.confidence_thresholds = CONFIDENCE_THRESHOLDS
        self.facial_zone_weights = FACIAL_ZONE_WEIGHTS

        # Add reference to ML detection flags
        self.use_ml_for_midface = USE_ML_FOR_MIDFACE
        self.use_ml_for_lower_face = USE_ML_FOR_LOWER_FACE

        # Add reference to new constants
        self.au_zone_detection_modifiers = AU_ZONE_DETECTION_MODIFIERS
        self.paralysis_detection_au_importance = PARALYSIS_DETECTION_AU_IMPORTANCE
        self.baseline_au_activations = BASELINE_AU_ACTIVATIONS

        # For AU12_r in the lower face, extreme asymmetry threshold is lower
        self.au12_extreme_asymmetry_threshold = AU12_EXTREME_ASYMMETRY_THRESHOLD
        # For AU01_r in the upper face, extreme asymmetry threshold is also lower
        # For AU45_r in the mid face, extreme asymmetry threshold is also lower
        self.au45_extreme_asymmetry_threshold = AU45_EXTREME_ASYMMETRY_THRESHOLD
        # For AU07_r in the mid face, extreme asymmetry threshold is also lower
        self.au07_extreme_asymmetry_threshold = AU07_EXTREME_ASYMMETRY_THRESHOLD

        # Flag for expert review
        self.flag_borderline_for_review = FLAG_BORDERLINE_FOR_REVIEW

        # Import helper functions
        from facial_paralysis_helpers import (
            calculate_weighted_activation, calculate_percent_difference,
            calculate_confidence_score
        )

        self._calculate_weighted_activation = lambda *args, **kwargs: calculate_weighted_activation(self, *args,
                                                                                                    **kwargs)
        self._calculate_percent_difference = calculate_percent_difference
        self._calculate_confidence_score = lambda *args, **kwargs: calculate_confidence_score(self, *args, **kwargs)

        from facial_paralysis_analysis import check_for_extreme_asymmetry, check_for_borderline_cases
        self._check_for_extreme_asymmetry = lambda *args, **kwargs: check_for_extreme_asymmetry(self, *args, **kwargs)
        self._check_for_borderline_cases = lambda *args, **kwargs: check_for_borderline_cases(self, *args, **kwargs)

    # facial_au_paralysis_detector.py - detect_paralysis method
    def detect_paralysis(self, results):
        """
        Detect potential facial paralysis by analyzing asymmetry patterns by facial zone.
        Independently assesses each zone on both left and right sides.
        Uses zone-specific thresholds for more accurate detection.
        Uses a dual-metric approach for improved detection of partial vs complete paralysis.
        Includes confidence scoring to reduce false positives.

        Args:
            results (dict): Results dictionary with AU values for each action

        Returns:
            None: Updates results dictionary in place
        """

        if not results:
            logger.warning("No results to analyze for paralysis detection")
            return

        try:
            # Track affected zones and severity for each side separately
            zone_paralysis = {
                'left': {'upper': 'None', 'mid': 'None', 'lower': 'None'},
                'right': {'upper': 'None', 'mid': 'None', 'lower': 'None'}
            }

            # Track which AUs show asymmetry by zone and side
            affected_aus_by_zone_side = {
                'left': {'upper': set(), 'mid': set(), 'lower': set()},
                'right': {'upper': set(), 'mid': set(), 'lower': set()}
            }

            # Analyze each action for paralysis by facial zone
            for action, info in results.items():
                # Reset the zone-specific paralysis data for this action
                info['paralysis'] = {
                    'detected': False,
                    'zones': {
                        'left': {'upper': 'None', 'mid': 'None', 'lower': 'None'},
                        'right': {'upper': 'None', 'mid': 'None', 'lower': 'None'}
                    },
                    'affected_aus': {'left': [], 'right': []},
                    'contributing_aus': {'left': {}, 'right': {}},  # Structure to track thresholds and values
                    'action_relevance': {},  # Track if this action is relevant for each zone
                    'confidence': {'left': {}, 'right': {}}  # Track confidence scores
                }

                # Analyze each facial zone for both sides
                for zone, aus in self.facial_zones.items():
                    # Determine if this action is relevant for this zone
                    is_relevant = action in self.zone_specific_actions[zone]
                    info['paralysis']['action_relevance'][zone] = is_relevant

                    # Only analyze zones if action is relevant for that zone
                    if not is_relevant:
                        continue

                    # Get the zone-specific thresholds for both severity levels
                    partial_thresholds = self.paralysis_thresholds[zone]['partial']
                    complete_thresholds = self.paralysis_thresholds[zone]['complete']
                    asymmetry_thresholds = self.asymmetry_thresholds[zone]
                    confidence_thresholds = self.confidence_thresholds[zone]

                    # Calculate AU values for left and right sides in this zone
                    left_values = {}
                    right_values = {}
                    left_values_normalized = {}
                    right_values_normalized = {}

                    for au in aus:
                        if au in info['left']['au_values'] and au in info['right']['au_values']:
                            # Use raw values for basic collection
                            left_values[au] = info['left']['au_values'][au]
                            right_values[au] = info['right']['au_values'][au]

                            # Also collect normalized values if available
                            if 'normalized_au_values' in info['left'] and 'normalized_au_values' in info['right']:
                                if au in info['left']['normalized_au_values'] and au in info['right'][
                                    'normalized_au_values']:
                                    left_values_normalized[au] = info['left']['normalized_au_values'][au]
                                    right_values_normalized[au] = info['right']['normalized_au_values'][au]

                    if not left_values or not right_values:
                        continue  # Skip if no data for this zone

                    # Store normalized values for access in other methods
                    self.current_normalized_values = {
                        'left': left_values_normalized,
                        'right': right_values_normalized
                    }

                    # Initialize contributing AU tracking for this zone if not already present
                    for side in ['left', 'right']:
                        if zone not in info['paralysis']['contributing_aus'][side]:
                            info['paralysis']['contributing_aus'][side][zone] = {
                                'minimal_movement': [],
                                'asymmetry': [],
                                'percent_diff': [],
                                'extreme_asymmetry': [],
                                'borderline_case': [],
                                'normalized_ratio': [],
                                'combined_score': [],
                                'individual_au': [],
                                'ml_detection': []  # Added for ML detection results
                            }

                        # Initialize confidence score for this zone
                        if zone not in info['paralysis']['confidence'][side]:
                            info['paralysis']['confidence'][side][zone] = 0.0

                    # Calculate weighted average activations with normalized values if available
                    left_avg = self._calculate_weighted_activation('left', zone, left_values, left_values_normalized)
                    right_avg = self._calculate_weighted_activation('right', zone, right_values,
                                                                    right_values_normalized)

                    # Import detection function to avoid circular imports
                    from facial_paralysis_detection import detect_side_paralysis

                    # Detect left side paralysis
                    left_result = detect_side_paralysis(
                        self, info, zone, 'left', aus, left_values, right_values,
                        left_values_normalized, right_values_normalized,
                        left_avg, right_avg, zone_paralysis, affected_aus_by_zone_side,
                        partial_thresholds, complete_thresholds, asymmetry_thresholds, confidence_thresholds
                    )

                    # Detect right side paralysis
                    right_result = detect_side_paralysis(
                        self, info, zone, 'right', aus, right_values, left_values,
                        right_values_normalized, left_values_normalized,
                        right_avg, left_avg, zone_paralysis, affected_aus_by_zone_side,
                        partial_thresholds, complete_thresholds, asymmetry_thresholds, confidence_thresholds
                    )

                    # Update the overall detection flag for this action if any zone shows paralysis
                    for side in ['left', 'right']:
                        for z in ['upper', 'mid', 'lower']:
                            if info['paralysis']['zones'][side][z] != 'None':
                                info['paralysis']['detected'] = True

            # Update patient-level paralysis information across all actions
            for action, info in results.items():
                # Copy the zone-level paralysis information determined across all actions
                for side in ['left', 'right']:
                    for zone in ['upper', 'mid', 'lower']:
                        info['paralysis']['zones'][side][zone] = zone_paralysis[side][zone]

                # Update the overall detection flag
                for side in ['left', 'right']:
                    for zone in ['upper', 'mid', 'lower']:
                        if zone_paralysis[side][zone] != 'None':
                            info['paralysis']['detected'] = True

            # Only log the final summary of detected paralysis
            detected_zones = [f'{side} {zone}={zone_paralysis[side][zone]}'
                              for side in ['left', 'right']
                              for zone in ['upper', 'mid', 'lower']
                              if zone_paralysis[side][zone] != 'None']

            if detected_zones:
                logger.info(f"Detected paralysis: {', '.join(detected_zones)}")
            else:
                logger.info("No paralysis detected")

        except Exception as e:
            logger.error(f"Exception in detect_paralysis: {str(e)}", exc_info=True)