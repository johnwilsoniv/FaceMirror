"""
Facial paralysis detection module.
Analyzes facial Action Units to detect possible facial paralysis.
"""

import numpy as np
import logging
import os
from facial_paralysis_analysis import analyze_paralysis_results
from lower_face_detector import LowerFaceParalysisDetector
from mid_face_detector import MidFaceParalysisDetector  # Added import

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)


class FacialParalysisDetector:
    """
    Detects facial paralysis by analyzing AU patterns.
    """

    def __init__(self):
        """Initialize the facial paralysis detector."""
        # Initialize zone definitions
        self.facial_zones = {
            'upper': ['AU01_r', 'AU02_r'],
            'mid': ['AU45_r', 'AU07_r'],
            'lower': ['AU12_r', 'AU25_r']
        }

        # Initialize zone-specific actions
        self.zone_specific_actions = {
            'upper': ['RE'],  # Raise Eyebrows
            'mid': ['ES', 'ET'],  # Close Eyes Softly and Tightly
            'lower': ['BS']  # Big Smile
        }

        # Initialize all zone-specific detectors
        try:
            self.lower_face_detector = LowerFaceParalysisDetector()
            logger.info("Lower face paralysis detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize lower face detector: {str(e)}")
            self.lower_face_detector = None
            
        try:
            self.mid_face_detector = MidFaceParalysisDetector()
            logger.info("Mid face paralysis detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize mid face detector: {str(e)}")
            self.mid_face_detector = None

    def detect_paralysis(self, results):
        """
        Detect potential facial paralysis by analyzing asymmetry patterns by facial zone.

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
                    'contributing_aus': {'left': {}, 'right': {}},
                    'action_relevance': {},
                    'confidence': {'left': {}, 'right': {}}
                }

                # Process each zone
                for zone in ['upper', 'mid', 'lower']:
                    # Only analyze if this action is relevant for the zone
                    is_relevant = action in self.zone_specific_actions[zone]
                    info['paralysis']['action_relevance'][zone] = is_relevant

                    if not is_relevant:
                        continue

                    aus = self.facial_zones[zone]

                    # Initialize contributing AU tracking for this zone
                    for side in ['left', 'right']:
                        if zone not in info['paralysis']['contributing_aus'][side]:
                            info['paralysis']['contributing_aus'][side][zone] = {}

                        # Initialize confidence score
                        if zone not in info['paralysis']['confidence'][side]:
                            info['paralysis']['confidence'][side][zone] = 0.0

                    # Process each side
                    for side in ['left', 'right']:
                        other_side = 'right' if side == 'left' else 'left'
                        
                        # Get AU values for this side and opposite side
                        side_values = {au: info[side]['au_values'].get(au, 0) for au in aus if
                                    au in info[side]['au_values']}
                        other_values = {au: info[other_side]['au_values'].get(au, 0) for au in aus if
                                        au in info[other_side]['au_values']}

                        # Get normalized values if available
                        side_values_normalized = {}
                        other_values_normalized = {}

                        if 'normalized_au_values' in info[side] and 'normalized_au_values' in info[other_side]:
                            side_values_normalized = {au: info[side]['normalized_au_values'].get(au, 0) for au in aus if
                                                    au in info[side]['normalized_au_values']}
                            other_values_normalized = {au: info[other_side]['normalized_au_values'].get(au, 0) for au in aus if
                                                    au in info[other_side]['normalized_au_values']}

                        # Use the appropriate detector for each zone
                        try:
                            detection_result = False
                            
                            if zone == 'lower' and self.lower_face_detector:
                                detection_result = self.lower_face_detector.detect_paralysis(
                                    self, info, zone, side, aus, side_values, other_values,
                                    side_values_normalized, other_values_normalized,
                                    zone_paralysis, affected_aus_by_zone_side
                                )
                            elif zone == 'mid' and self.mid_face_detector:
                                detection_result = self.mid_face_detector.detect_paralysis(
                                    self, info, zone, side, aus, side_values, other_values,
                                    side_values_normalized, other_values_normalized,
                                    zone_paralysis, affected_aus_by_zone_side
                                )
                            
                            if detection_result:
                                logger.info(f"Detected {zone_paralysis[side][zone]} paralysis in {side} {zone} zone")
                        except Exception as e:
                            logger.error(f"Error in {side} {zone} detection: {str(e)}")

                # Update overall detection flag
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

            # Analyze results for additional insights
            patient_id = next(iter(results.values())).get('patient_id', None) if results else None
            analysis = analyze_paralysis_results(results, patient_id)

            # Store analysis in results
            for action, info in results.items():
                info['paralysis_analysis'] = analysis

        except Exception as e:
            logger.error(f"Exception in detect_paralysis: {str(e)}", exc_info=True)
