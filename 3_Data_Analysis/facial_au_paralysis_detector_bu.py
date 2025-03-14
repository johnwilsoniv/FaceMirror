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
    AU12_EXTREME_ASYMMETRY_THRESHOLD, AU01_EXTREME_ASYMMETRY_THRESHOLD,
    AU45_EXTREME_ASYMMETRY_THRESHOLD, AU07_EXTREME_ASYMMETRY_THRESHOLD,
    MIDFACE_FUNCTIONAL_THRESHOLDS, USE_FUNCTIONAL_APPROACH_FOR_MIDFACE,
    MIDFACE_COMPONENT_THRESHOLDS, USE_DUAL_CRITERIA_FOR_MIDFACE,
    MIDFACE_COMBINED_SCORE_THRESHOLDS, USE_COMBINED_SCORE_FOR_MIDFACE,
    # Lower face constants
    LOWER_FACE_COMBINED_SCORE_THRESHOLDS, USE_COMBINED_SCORE_FOR_LOWER_FACE,
    LOWER_FACE_PARALYSIS_THRESHOLDS, LOWER_FACE_ASYMMETRY_THRESHOLDS,
    LOWER_FACE_CONFIDENCE_THRESHOLDS
)

# Import helper modules
from facial_paralysis_helpers import (
    calculate_weighted_activation, calculate_percent_difference,
    calculate_confidence_score, calculate_midface_combined_score,
    calculate_combined_midface_score,  # New helper function for combined midface score
    calculate_lower_face_combined_score,
    calculate_functional_component, calculate_asymmetry_component,
    calculate_midface_functional_score
)
from facial_paralysis_analysis import check_for_extreme_asymmetry, check_for_borderline_cases

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        # Add reference to the zone weights
        self.facial_zone_weights = FACIAL_ZONE_WEIGHTS
        # Add reference to new constants
        self.au_zone_detection_modifiers = AU_ZONE_DETECTION_MODIFIERS
        self.paralysis_detection_au_importance = PARALYSIS_DETECTION_AU_IMPORTANCE
        self.baseline_au_activations = BASELINE_AU_ACTIVATIONS
        # Extreme asymmetry threshold
        self.extreme_asymmetry_threshold = EXTREME_ASYMMETRY_THRESHOLD
        # For AU12_r in the lower face, extreme asymmetry threshold is lower
        self.au12_extreme_asymmetry_threshold = AU12_EXTREME_ASYMMETRY_THRESHOLD
        # For AU01_r in the upper face, extreme asymmetry threshold is also lower
        self.au01_extreme_asymmetry_threshold = AU01_EXTREME_ASYMMETRY_THRESHOLD
        # For AU45_r in the mid face, extreme asymmetry threshold is also lower
        self.au45_extreme_asymmetry_threshold = AU45_EXTREME_ASYMMETRY_THRESHOLD
        # For AU07_r in the mid face, extreme asymmetry threshold
        self.au07_extreme_asymmetry_threshold = AU07_EXTREME_ASYMMETRY_THRESHOLD

        # NEW: Add combined score related attributes for midface
        self.midface_combined_score_thresholds = MIDFACE_COMBINED_SCORE_THRESHOLDS
        self.use_combined_score_for_midface = USE_COMBINED_SCORE_FOR_MIDFACE
        # Add reference to component thresholds for mid face
        self.midface_component_thresholds = MIDFACE_COMPONENT_THRESHOLDS
        self.use_dual_criteria_for_midface = USE_DUAL_CRITERIA_FOR_MIDFACE
        # NEW: Add combined score related attributes for lower face
        self.lower_face_combined_score_thresholds = LOWER_FACE_COMBINED_SCORE_THRESHOLDS
        self.use_combined_score_for_lower_face = USE_COMBINED_SCORE_FOR_LOWER_FACE

        # NEW: Add lower face specific thresholds
        self.lower_face_paralysis_thresholds = LOWER_FACE_PARALYSIS_THRESHOLDS
        self.lower_face_asymmetry_thresholds = LOWER_FACE_ASYMMETRY_THRESHOLDS
        self.lower_face_confidence_thresholds = LOWER_FACE_CONFIDENCE_THRESHOLDS

        self.midface_functional_thresholds = MIDFACE_FUNCTIONAL_THRESHOLDS
        self.use_functional_approach_for_midface = USE_FUNCTIONAL_APPROACH_FOR_MIDFACE

        # Make helper functions available as methods by binding them to self
        from facial_paralysis_helpers import (
            calculate_weighted_activation, calculate_percent_difference,
            calculate_confidence_score, calculate_midface_combined_score,
            calculate_combined_midface_score, calculate_lower_face_combined_score,
            calculate_functional_component, calculate_asymmetry_component,
            calculate_midface_functional_score
        )

        self._calculate_weighted_activation = lambda *args, **kwargs: calculate_weighted_activation(self, *args,
                                                                                                    **kwargs)
        self._calculate_percent_difference = calculate_percent_difference
        self._calculate_confidence_score = lambda *args, **kwargs: calculate_confidence_score(self, *args, **kwargs)
        self._calculate_midface_combined_score = calculate_midface_combined_score
        self._calculate_combined_midface_score = calculate_combined_midface_score
        self._calculate_lower_face_combined_score = calculate_lower_face_combined_score
        self._calculate_midface_functional_score = calculate_midface_functional_score
        self._calculate_functional_component = calculate_functional_component
        self._calculate_asymmetry_component = calculate_asymmetry_component

        from facial_paralysis_analysis import check_for_extreme_asymmetry, check_for_borderline_cases
        self._check_for_extreme_asymmetry = lambda *args, **kwargs: check_for_extreme_asymmetry(self, *args, **kwargs)
        self._check_for_borderline_cases = lambda *args, **kwargs: check_for_borderline_cases(self, *args, **kwargs)

    # Fix for process_midface_combined_score indentation issue
    def process_midface_combined_score(self, info, zone, aus, side,
                                       current_side_values, other_side_values,
                                       current_side_normalized, other_side_normalized,
                                       confidence_score, zone_paralysis, affected_aus_by_zone_side,
                                       criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None):
        """Process midface zone using the combined score approach."""
        # Check if we have the necessary AUs for combined score calculation
        if 'AU45_r' in current_side_values and 'AU45_r' in other_side_values:
            # Get the appropriate values for AU45_r (eyelid closure)
            current_au45 = current_side_values['AU45_r']
            other_au45 = other_side_values['AU45_r']

            if current_side_normalized and other_side_normalized and 'AU45_r' in current_side_normalized and 'AU45_r' in other_side_normalized:
                current_au45 = current_side_normalized['AU45_r']
                other_au45 = other_side_normalized['AU45_r']

            # Also check for AU07_r if available (lid tightener) from BS action
            au7_available = False
            current_au7 = 0
            other_au7 = 0

            # Try to get AU07_r values from the BS action if available
            bs_data = None
            for action_name, action_data in info.items():
                if action_name == 'BS':
                    bs_data = action_data
                    break

            # If BS data is available, extract AU7_r values
            if bs_data and 'left' in bs_data and 'right' in bs_data:
                bs_current_side = bs_data[side]['au_values'] if side in bs_data else None
                bs_other_side = bs_data['left' if side == 'right' else 'right']['au_values']

                bs_current_normalized = bs_data[side].get('normalized_au_values', {}) if side in bs_data else {}
                bs_other_normalized = bs_data['left' if side == 'right' else 'right'].get('normalized_au_values', {})

                if bs_current_normalized and bs_other_normalized and 'AU07_r' in bs_current_normalized and 'AU07_r' in bs_other_normalized:
                    current_au7 = bs_current_normalized['AU07_r']
                    other_au7 = bs_other_normalized['AU07_r']
                    au7_available = True

            # Calculate ratios for AU45_r and AU07_r (if available)
            au45_ratio = min(current_au45, other_au45) / max(current_au45, other_au45) if max(current_au45,
                                                                                              other_au45) > 0 else 0
            au7_ratio = min(current_au7, other_au7) / max(current_au7, other_au7) if au7_available and max(current_au7,
                                                                                                           other_au7) > 0 else 0

            # Calculate combined score - lower values indicate more paralysis
            combined_score = 0
            if au7_available:
                # If AU07_r is available, use weighted average of both AU scores
                au45_weight = 0.7  # Higher weight for primary AU45_r
                au7_weight = 0.3  # Lower weight for supporting AU07_r

                # Calculate combined score with both AU45_r and AU07_r
                au45_component = au45_ratio * (1 + current_au45 / 3.0)
                au7_component = au7_ratio * (1 + current_au7 / 3.0)
                combined_score = (au45_component * au45_weight) + (au7_component * au7_weight)
            else:
                # If only AU45_r is available, use only it for score
                combined_score = au45_ratio * (1 + current_au45 / 3.0)

            # Store confidence score
            info['paralysis']['confidence'][side][zone] = confidence_score

            # Initialize tracking structures if needed
            if 'combined_score' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['combined_score'] = []

            # Use thresholds from constants - these determine severity levels
            complete_threshold = self.midface_combined_score_thresholds['complete']
            partial_threshold = self.midface_combined_score_thresholds['partial']

            # Adjust confidence for midface - often needs a boost
            adjusted_confidence = max(confidence_score, 0.3)

            # Determine paralysis severity based on combined score and confidence
            if combined_score < complete_threshold and adjusted_confidence >= self.confidence_thresholds[zone][
                'complete']:
                # Complete paralysis
                info['paralysis']['zones'][side][zone] = 'Complete'

                # Track for patient-level assessment
                if zone_paralysis[side][zone] != 'Complete':
                    zone_paralysis[side][zone] = 'Complete'

                # Track contributing AUs
                affected_aus_by_zone_side[side][zone].add('AU45_r')
                if 'AU45_r' not in info['paralysis']['affected_aus'][side]:
                    info['paralysis']['affected_aus'][side].append('AU45_r')

                if au7_available:
                    affected_aus_by_zone_side[side][zone].add('AU07_r')
                    if 'AU07_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU07_r')

                # Track contribution details
                info['paralysis']['contributing_aus'][side][zone]['combined_score'].append({
                    'au': 'AU45_r + AU07_r' if au7_available else 'AU45_r',
                    'current_au45': current_au45,
                    'other_au45': other_au45,
                    'au45_ratio': au45_ratio,
                    'current_au7': current_au7 if au7_available else 'NA',
                    'other_au7': other_au7 if au7_available else 'NA',
                    'au7_ratio': au7_ratio if au7_available else 'NA',
                    'combined_score': combined_score,
                    'threshold': complete_threshold,
                    'type': 'Complete'
                })
            elif combined_score < partial_threshold and adjusted_confidence >= self.confidence_thresholds[zone][
                'partial']:
                # Partial paralysis - only if not already Complete
                if info['paralysis']['zones'][side][zone] == 'None':
                    info['paralysis']['zones'][side][zone] = 'Partial'

                # Track for patient-level assessment (if not already Complete)
                if zone_paralysis[side][zone] == 'None':
                    zone_paralysis[side][zone] = 'Partial'

                # Track contributing AUs
                affected_aus_by_zone_side[side][zone].add('AU45_r')
                if 'AU45_r' not in info['paralysis']['affected_aus'][side]:
                    info['paralysis']['affected_aus'][side].append('AU45_r')

                if au7_available:
                    affected_aus_by_zone_side[side][zone].add('AU07_r')
                    if 'AU07_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU07_r')

                # Track contribution details
                info['paralysis']['contributing_aus'][side][zone]['combined_score'].append({
                    'au': 'AU45_r + AU07_r' if au7_available else 'AU45_r',
                    'current_au45': current_au45,
                    'other_au45': other_au45,
                    'au45_ratio': au45_ratio,
                    'current_au7': current_au7 if au7_available else 'NA',
                    'other_au7': other_au7 if au7_available else 'NA',
                    'au7_ratio': au7_ratio if au7_available else 'NA',
                    'combined_score': combined_score,
                    'threshold': partial_threshold,
                    'type': 'Partial'
                })
            else:
                # No paralysis detected
                info['paralysis']['zones'][side][zone] = 'None'

            return True  # Successfully processed

        return False  # Could not process with combined score approach

    def process_mid_face_zone(self, info, zone, aus, side,
                                current_values, other_values,
                                current_values_normalized, other_values_normalized,
                                confidence_score, zone_paralysis, affected_aus_by_zone_side,
                                criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None,
                                partial_thresholds=None, complete_thresholds=None):
        """Process mid face zone for paralysis detection.

        Args:
            info (dict): Current action info dictionary to update
            zone (str): Facial zone being analyzed ('mid')
            aus (list): Action units in this zone
            side (str): 'left' or 'right' - which side we're checking for paralysis
            current_values (dict): AU values for the side being checked
            other_values (dict): AU values for the opposite side
            current_values_normalized (dict): Normalized AU values for the side being checked
            other_values_normalized (dict): Normalized AU values for the opposite side
            confidence_score (float): Confidence score for detection
            zone_paralysis (dict): Tracking paralysis by zone
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side
            criteria_met (dict, optional): Dictionary of detection criteria that were met
            asymmetry_thresholds (dict, optional): Thresholds for asymmetry detection
            confidence_thresholds (dict, optional): Thresholds for confidence scores
            partial_thresholds (dict, optional): Thresholds for partial paralysis
            complete_thresholds (dict, optional): Thresholds for complete paralysis

        Returns:
            bool: True if successfully processed, False otherwise
        """
        if not criteria_met:
            criteria_met = {}
        if not asymmetry_thresholds:
            asymmetry_thresholds = self.asymmetry_thresholds[zone]
        if not confidence_thresholds:
            confidence_thresholds = self.confidence_thresholds[zone]
        if not partial_thresholds:
            partial_thresholds = self.paralysis_thresholds[zone]['partial']
        if not complete_thresholds:
            complete_thresholds = self.paralysis_thresholds[zone]['complete']
        
        # Try combined score approach first
        if hasattr(self, 'use_combined_score_for_midface') and self.use_combined_score_for_midface:
            processed = self.process_midface_combined_score(
                info, zone, aus, side, 
                current_values, other_values,
                current_values_normalized, other_values_normalized,
                confidence_score, zone_paralysis, affected_aus_by_zone_side,
                criteria_met, asymmetry_thresholds, confidence_thresholds
            )
            
            # If combined score approach successful, proceed with functional approach as an additional check
            # (Not returning early - we want to continue with functional approach for completeness)
        
        # Always perform functional approach for midface
        processed_functional = self.process_midface_functional_approach(
            info, zone, aus, side, 
            current_values, other_values,
            current_values_normalized, other_values_normalized,
            confidence_score, zone_paralysis, affected_aus_by_zone_side
        )
        
        # The rest of the method is similar to the standard detection logic in other zones,
        # but we only apply it if neither combined score nor functional approach detected paralysis
        if info['paralysis']['zones'][side][zone] == 'None':
            # Get specific AU normalized percent difference and ratio for mid face: AU45_r
            au45_percent_diff = None
            au45_ratio = None
            au07_percent_diff = None
            au07_ratio = None

            if 'AU45_r' in current_values and 'AU45_r' in other_values:
                # Use normalized values if available
                if 'AU45_r' in current_values_normalized and 'AU45_r' in other_values_normalized:
                    current_au45 = current_values_normalized['AU45_r']
                    other_au45 = other_values_normalized['AU45_r']

                    if current_au45 > 0 or other_au45 > 0:
                        if current_au45 > 0 and other_au45 > 0:
                            au45_percent_diff = calculate_percent_difference(current_au45, other_au45)
                            au45_ratio = min(current_au45, other_au45) / max(current_au45, other_au45)
                        else:
                            au45_percent_diff = 100  # One side has zero movement
                            au45_ratio = 0
                        logger.debug(f"AU45_r normalized: percent_diff={au45_percent_diff:.1f}%, ratio={au45_ratio:.3f}")

            # Try to get AU07_r values from the BS action if available
            bs_data = None
            for action_name, action_data in info.items():
                if action_name == 'BS':
                    bs_data = action_data
                    break

            # If BS data is available, extract AU7_r values
            if bs_data and 'left' in bs_data and 'right' in bs_data:
                bs_current_side = bs_data[side]['au_values'] if side in bs_data else None
                bs_other_side = bs_data['left' if side == 'right' else 'right']['au_values']

                bs_current_normalized = bs_data[side].get('normalized_au_values', {}) if side in bs_data else {}
                bs_other_normalized = bs_data['left' if side == 'right' else 'right'].get('normalized_au_values', {})

                if bs_current_normalized and bs_other_normalized and 'AU07_r' in bs_current_normalized and 'AU07_r' in bs_other_normalized:
                    current_au07 = bs_current_normalized['AU07_r']
                    other_au07 = bs_other_normalized['AU07_r']

                    if current_au07 > 0 or other_au07 > 0:
                        if current_au07 > 0 and other_au07 > 0:
                            au07_percent_diff = calculate_percent_difference(current_au07, other_au07)
                            au07_ratio = min(current_au07, other_au07) / max(current_au07, other_au07)
                        else:
                            au07_percent_diff = 100  # One side has zero movement
                            au07_ratio = 0
                        logger.debug(f"AU07_r normalized: percent_diff={au07_percent_diff:.1f}%, ratio={au07_ratio:.3f}")

            # Check for extreme asymmetry in any individual AU with normalized values if available
            has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, confidence_boost = self._check_for_extreme_asymmetry(
                current_values, other_values, current_values_normalized, other_values_normalized, zone
            )

            # Get weighted average activations
            current_avg = self._calculate_weighted_activation(side, zone, current_values, current_values_normalized)
            other_avg = self._calculate_weighted_activation('left' if side == 'right' else 'right', zone, other_values, other_values_normalized)
            
            # Calculate asymmetry ratio
            ratio = 0
            if current_avg > 0 and other_avg > 0:
                ratio = min(current_avg, other_avg) / max(current_avg, other_avg)

            # Calculate percent difference between sides for each AU
            au_percent_diffs = []
            for au in aus:
                if au in current_values and au in other_values:
                    # Determine if we should use normalized values for this AU
                    use_normalized = False
                    au_base = au.split('_')[0] + '_r'
                    if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                        use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                    # Get the appropriate values
                    current_val = current_values[au]
                    other_val = other_values[au]

                    if use_normalized and au in current_values_normalized and au in other_values_normalized:
                        current_val = current_values_normalized[au]
                        other_val = other_values_normalized[au]
                        logger.debug(f"Using normalized values for {au} percent diff: current={current_val}, other={other_val}")

                    percent_diff = self._calculate_percent_difference(current_val, other_val)
                    au_percent_diffs.append(percent_diff)

            # Get maximum percent difference (most asymmetric AU)
            max_percent_diff = max(au_percent_diffs) if au_percent_diffs else 0

            # Check each criterion separately and track which ones are met
            criteria_met = {
                'minimal_movement': current_avg < complete_thresholds['minimal_movement'],
                'ratio': (current_avg > 0 and other_avg > 0 and
                        ratio < asymmetry_thresholds['complete']['ratio'] and
                        current_avg < other_avg),
                'percent_diff': (max_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and
                               current_avg < other_avg),
                'extreme_asymmetry': (has_extreme_asymmetry and weaker_side == side)
            }

            # Special case for AU45_r normalized ratio (mid face)
            if au45_ratio is not None and au45_percent_diff is not None:
                # If AU45_r normalized ratio is below threshold and this side is weaker
                if (au45_ratio < asymmetry_thresholds['complete']['ratio'] and 
                    current_values_normalized['AU45_r'] < other_values_normalized['AU45_r']):
                    criteria_met['au45_normalized_ratio'] = True
                    logger.debug(
                        f"{side.upper()}: AU45_r normalized ratio is below threshold: {au45_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                side, zone, current_values, other_values, criteria_met
            )

            # Apply any confidence boost from extreme asymmetry detection
            confidence_score += confidence_boost

            # Check for borderline cases
            is_borderline, adjusted_confidence = self._check_for_borderline_cases(
                zone, side, current_values, other_values, ratio, max_percent_diff, confidence_score
            )

            # Update confidence score if borderline case
            if is_borderline:
                confidence_score = adjusted_confidence
                criteria_met['borderline_case'] = True

            # Store confidence score
            info['paralysis']['confidence'][side][zone] = confidence_score

            # Initialize tracking structures if they don't exist yet
            if 'extreme_asymmetry' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['extreme_asymmetry'] = []
            if 'normalized_ratio' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['normalized_ratio'] = []
            if 'borderline_case' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['borderline_case'] = []
            if 'minimal_movement' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['minimal_movement'] = []
            if 'asymmetry' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['asymmetry'] = []
            if 'percent_diff' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['percent_diff'] = []

            # Decision logic for mid face paralysis detection - only applied if neither
            # combined score nor functional approach detected any paralysis
                
            # Standard detection logic - only apply if zone is still 'None'
            if info['paralysis']['zones'][side][zone] == 'None':
                # First priority: Extreme asymmetry
                if criteria_met['extreme_asymmetry']:
                    # If zone has extreme asymmetry, set as Complete
                    info['paralysis']['zones'][side][zone] = 'Complete'

                    # Track for patient-level assessment
                    if zone_paralysis[side][zone] != 'Complete':
                        zone_paralysis[side][zone] = 'Complete'

                    # Track contributing AUs and add to affected_aus
                    affected_aus_by_zone_side[side][zone].add(extreme_au)
                    if extreme_au not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append(extreme_au)

                    # Track AU values and thresholds for extreme asymmetry
                    info['paralysis']['contributing_aus'][side][zone]['extreme_asymmetry'].append({
                        'au': extreme_au,
                        'current_value': current_values[extreme_au],
                        'other_value': other_values[extreme_au],
                        'percent_diff': extreme_percent_diff,
                        'threshold': self.au45_extreme_asymmetry_threshold if extreme_au == 'AU45_r' else self.extreme_asymmetry_threshold,
                        'type': 'Complete'
                    })
                
                # Second priority: AU45_r normalized ratio
                elif criteria_met.get('au45_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
                    # AU45_r normalized ratio indicates complete paralysis
                    info['paralysis']['zones'][side][zone] = 'Complete'

                    # Track for patient-level assessment
                    if zone_paralysis[side][zone] != 'Complete':
                        zone_paralysis[side][zone] = 'Complete'

                    # Track contributing AUs
                    affected_aus_by_zone_side[side][zone].add('AU45_r')
                    if 'AU45_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU45_r')

                    # Track values for this detection
                    info['paralysis']['contributing_aus'][side][zone]['normalized_ratio'].append({
                        'au': 'AU45_r',
                        'current_value': current_values_normalized['AU45_r'],
                        'other_value': other_values_normalized['AU45_r'],
                        'ratio': au45_ratio,
                        'threshold': asymmetry_thresholds['complete']['ratio'],
                        'type': 'Complete',
                        'normalized': True
                    })
                
                # Third priority: Borderline cases
                elif criteria_met.get('borderline_case', False) and confidence_score >= confidence_thresholds['complete']:
                    # Handle borderline cases with specific checks
                    info['paralysis']['zones'][side][zone] = 'Complete'
                    if zone_paralysis[side][zone] != 'Complete':
                        zone_paralysis[side][zone] = 'Complete'

                    # Track contributing AUs for borderline case
                    for au in aus:
                        if au in current_values and au in other_values:
                            percent_diff = self._calculate_percent_difference(current_values[au], other_values[au])
                            if percent_diff > 70.0 and current_values[au] < other_values[au]:
                                affected_aus_by_zone_side[side][zone].add(au)
                                if au not in info['paralysis']['affected_aus'][side]:
                                    info['paralysis']['affected_aus'][side].append(au)

                                # Track AU values for borderline case
                                info['paralysis']['contributing_aus'][side][zone]['borderline_case'].append({
                                    'au': au,
                                    'current_value': current_values[au],
                                    'other_value': other_values[au],
                                    'percent_diff': percent_diff,
                                    'ratio': current_values[au] / other_values[au] if other_values[au] > 0 else 0,
                                    'type': 'Complete'
                                })
                
                # Fourth priority: Standard thresholds for complete paralysis
                elif ((criteria_met['minimal_movement'] or criteria_met['ratio']) and
                      confidence_score >= confidence_thresholds['complete']):

                    # Set full paralysis
                    info['paralysis']['zones'][side][zone] = 'Complete'

                    # Track for patient-level assessment
                    if zone_paralysis[side][zone] != 'Complete':
                        zone_paralysis[side][zone] = 'Complete'

                    # Track contributing AUs and add to affected_aus
                    for au in aus:
                        if au in current_values:
                            # Check if we should use normalized value for this AU
                            use_normalized = False
                            au_base = au.split('_')[0] + '_r'
                            if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                                use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                            # Get the appropriate value
                            curr_val = current_values[au]
                            if use_normalized and au in current_values_normalized:
                                curr_val = current_values_normalized[au]

                            # Check if AU is below threshold
                            if curr_val < complete_thresholds['minimal_movement']:
                                affected_aus_by_zone_side[side][zone].add(au)
                                if au not in info['paralysis']['affected_aus'][side]:
                                    info['paralysis']['affected_aus'][side].append(au)

                                # Track AU values and thresholds
                                info['paralysis']['contributing_aus'][side][zone]['minimal_movement'].append({
                                    'au': au,
                                    'value': curr_val,
                                    'threshold': complete_thresholds['minimal_movement'],
                                    'type': 'Complete',
                                    'normalized': use_normalized
                                })

                            # Check if AU contributes to ratio-based detection
                            elif au in other_values:
                                # Get the appropriate other value
                                other_val = other_values[au]
                                if use_normalized and au in other_values_normalized:
                                    other_val = other_values_normalized[au]

                                if other_val > 0 and curr_val > 0:
                                    au_ratio = min(curr_val, other_val) / max(curr_val, other_val)
                                    if au_ratio < asymmetry_thresholds['complete']['ratio'] and curr_val < other_val:
                                        affected_aus_by_zone_side[side][zone].add(au)
                                        if au not in info['paralysis']['affected_aus'][side]:
                                            info['paralysis']['affected_aus'][side].append(au)

                                        # Track AU values and thresholds
                                        info['paralysis']['contributing_aus'][side][zone]['asymmetry'].append({
                                            'au': au,
                                            'current_value': curr_val,
                                            'other_value': other_val,
                                            'ratio': au_ratio,
                                            'threshold': asymmetry_thresholds['complete']['ratio'],
                                            'type': 'Complete',
                                            'normalized': use_normalized
                                        })

                # Fifth priority: Partial paralysis detection
                elif ((criteria_met.get('minimal_movement', False) or criteria_met.get('percent_diff', False)) and
                      confidence_score >= confidence_thresholds['partial']):
                      
                    # Only assign if not already marked as complete
                    if info['paralysis']['zones'][side][zone] == 'None':
                        info['paralysis']['zones'][side][zone] = 'Partial'

                    # Track for patient-level assessment - only update if not already complete
                    if zone_paralysis[side][zone] == 'None':
                        zone_paralysis[side][zone] = 'Partial'

                    # Track contributing AUs and add to affected_aus
                    for au in aus:
                        if au in current_values:
                            # Check if we should use normalized value for this AU
                            use_normalized = False
                            au_base = au.split('_')[0] + '_r'
                            if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                                use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                            # Get the appropriate values
                            curr_val = current_values[au]
                            if use_normalized and au in current_values_normalized:
                                curr_val = current_values_normalized[au]

                            # Check if AU is below threshold but above complete threshold
                            if (curr_val < partial_thresholds['minimal_movement'] and
                                    curr_val >= complete_thresholds['minimal_movement']):
                                affected_aus_by_zone_side[side][zone].add(au)
                                if au not in info['paralysis']['affected_aus'][side]:
                                    info['paralysis']['affected_aus'][side].append(au)

                                # Track AU values and thresholds
                                info['paralysis']['contributing_aus'][side][zone]['minimal_movement'].append({
                                    'au': au,
                                    'value': curr_val,
                                    'threshold': partial_thresholds['minimal_movement'],
                                    'type': 'Partial',
                                    'normalized': use_normalized
                                })

                            # Check if AU contributes to percent difference detection
                            elif au in other_values:
                                # Get the appropriate other value
                                other_val = other_values[au]
                                if use_normalized and au in other_values_normalized:
                                    other_val = other_values_normalized[au]

                                au_percent_diff = self._calculate_percent_difference(curr_val, other_val)
                                if au_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and curr_val < other_val:
                                    affected_aus_by_zone_side[side][zone].add(au)
                                    if au not in info['paralysis']['affected_aus'][side]:
                                        info['paralysis']['affected_aus'][side].append(au)

                                    # Track AU values and thresholds
                                    info['paralysis']['contributing_aus'][side][zone]['percent_diff'].append({
                                        'au': au,
                                        'current_value': curr_val,
                                        'other_value': other_val,
                                        'percent_diff': au_percent_diff,
                                        'threshold': asymmetry_thresholds['partial']['percent_diff'],
                                        'type': 'Partial',
                                        'normalized': use_normalized
                                    })

        return True
    
    def detect_side_paralysis(self, info, zone, aus, side,
                             current_values, other_values,
                             current_values_normalized, other_values_normalized,
                             current_avg, other_avg, zone_paralysis, affected_aus_by_zone_side,
                             partial_thresholds, complete_thresholds,
                             asymmetry_thresholds, confidence_thresholds):
        """Unified detection function for both left and right sides of the face.

        Args:
            info (dict): Current action info dictionary to update
            zone (str): Facial zone being analyzed
            aus (list): Action units in this zone
            side (str): 'left' or 'right' - which side we're checking for paralysis
            current_values (dict): AU values for the side being checked
            other_values (dict): AU values for the opposite side
            current_values_normalized (dict): Normalized AU values for the side being checked
            other_values_normalized (dict): Normalized AU values for the opposite side
            current_avg (float): Weighted average for the side being checked
            other_avg (float): Weighted average for the opposite side
            zone_paralysis (dict): Tracking paralysis by zone
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side
            partial_thresholds (dict): Thresholds for partial paralysis
            complete_thresholds (dict): Thresholds for complete paralysis
            asymmetry_thresholds (dict): Thresholds for asymmetry detection
            confidence_thresholds (dict): Thresholds for confidence scores
        """
        # First, check for extreme asymmetry
        has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, _ = self._check_for_extreme_asymmetry(
            current_values, other_values, current_values_normalized, other_values_normalized, zone
        )

        # If there is extreme asymmetry and the weaker side is NOT the current side,
        # then this side is the stronger side and should be protected from paralysis detection
        if has_extreme_asymmetry and weaker_side != side and extreme_percent_diff > 120:
            # Log the protection
            logger.debug(
                f"Protected {side} side from paralysis detection in {zone} zone - extreme asymmetry: {extreme_percent_diff:.1f}% (weaker side: {weaker_side})")
            return

        # Process each zone using the appropriate zone-specific method
        if zone == 'upper':
            self.process_upper_face_zone(
                info, zone, aus, side,
                current_values, other_values,
                current_values_normalized, other_values_normalized,
                confidence_score=self._calculate_confidence_score(side, zone, current_values, other_values, {}),
                zone_paralysis=zone_paralysis,
                affected_aus_by_zone_side=affected_aus_by_zone_side,
                partial_thresholds=partial_thresholds,
                complete_thresholds=complete_thresholds,
                asymmetry_thresholds=asymmetry_thresholds,
                confidence_thresholds=confidence_thresholds
            )
        elif zone == 'mid':
            self.process_mid_face_zone(
                info, zone, aus, side,
                current_values, other_values,
                current_values_normalized, other_values_normalized,
                confidence_score=self._calculate_confidence_score(side, zone, current_values, other_values, {}),
                zone_paralysis=zone_paralysis,
                affected_aus_by_zone_side=affected_aus_by_zone_side,
                partial_thresholds=partial_thresholds,
                complete_thresholds=complete_thresholds,
                asymmetry_thresholds=asymmetry_thresholds,
                confidence_thresholds=confidence_thresholds
            )
        elif zone == 'lower':
            self.process_lower_face_zone(
                info, zone, aus, side,
                current_values, other_values,
                current_values_normalized, other_values_normalized,
                confidence_score=self._calculate_confidence_score(side, zone, current_values, other_values, {}),
                zone_paralysis=zone_paralysis,
                affected_aus_by_zone_side=affected_aus_by_zone_side,
                partial_thresholds=partial_thresholds,
                complete_thresholds=complete_thresholds,
                asymmetry_thresholds=asymmetry_thresholds,
                confidence_thresholds=confidence_thresholds
            )

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

            # Analyze each facial zone
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

                # Get the asymmetry thresholds for this zone
                asymmetry_thresholds = self.asymmetry_thresholds[zone]

                # Get confidence thresholds for this zone
                confidence_thresholds = self.confidence_thresholds[zone]

                # Calculate AU values for left and right sides in this zone
                left_values = {}
                right_values = {}
                left_values_normalized = {}
                right_values_normalized = {}

                for au in aus:
                    if au in info['left']['au_values'] and au in info['right']['au_values']:
                        # Use raw values for basic collection
                        left_value = info['left']['au_values'][au]
                        right_value = info['right']['au_values'][au]

                        left_values[au] = left_value
                        right_values[au] = right_value

                        # Also collect normalized values if available
                        if 'normalized_au_values' in info['left'] and 'normalized_au_values' in info['right']:
                            if au in info['left']['normalized_au_values'] and au in info['right']['normalized_au_values']:
                                left_norm = info['left']['normalized_au_values'][au]
                                right_norm = info['right']['normalized_au_values'][au]

                                left_values_normalized[au] = left_norm
                                right_values_normalized[au] = right_norm

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
                            'percent_diff': [],  # Category for percent difference
                            'extreme_asymmetry': [],  # Category for extreme asymmetry detection
                            'borderline_case': [],  # Category for borderline cases
                            'normalized_ratio': [],  # Category for normalized ratio detection
                            'combined_score': [],  # Category for combined score detection
                            'individual_au': []  # Category for individual AU detection
                        }

                    # Initialize confidence score for this zone
                    if zone not in info['paralysis']['confidence'][side]:
                        info['paralysis']['confidence'][side][zone] = 0.0

                # Calculate weighted average activations with normalized values if available
                left_avg = self._calculate_weighted_activation('left', zone, left_values, left_values_normalized)
                right_avg = self._calculate_weighted_activation('right', zone, right_values, right_values_normalized)

                # Process each side using the unified method
                for side in ['left', 'right']:
                    # If side is left, current = left, other = right
                    # If side is right, current = right, other = left
                    current_values = left_values if side == 'left' else right_values
                    other_values = right_values if side == 'left' else left_values
                    
                    current_values_normalized = left_values_normalized if side == 'left' else right_values_normalized
                    other_values_normalized = right_values_normalized if side == 'left' else left_values_normalized
                    
                    current_avg = left_avg if side == 'left' else right_avg
                    other_avg = right_avg if side == 'left' else left_avg
                    
                    # Call unified detection method
                    self.detect_side_paralysis(
                        info, zone, aus, side, 
                        current_values, other_values,
                        current_values_normalized, other_values_normalized,
                        current_avg, other_avg, zone_paralysis, affected_aus_by_zone_side,
                        partial_thresholds, complete_thresholds, 
                        asymmetry_thresholds, confidence_thresholds
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

        logger.info("Completed paralysis detection")
        return True  # Successfully processed

            # Check for partial paralysis
        elif combined_score < partial_threshold and adjusted_confidence >= self.confidence_thresholds[zone]['partial']:
                # Partial paralysis - only if not already Complete
                if info['paralysis']['zones'][side][zone] == 'None':
                    info['paralysis']['zones'][side][zone] = 'Partial'  # String literal

                # Track for patient-level assessment (if not already Complete)
                if zone_paralysis[side][zone] == 'None':
                    zone_paralysis[side][zone] = 'Partial'  # String literal

                # Track contributing AUs
                affected_aus_by_zone_side[side][zone].add('AU45_r')
                if 'AU45_r' not in info['paralysis']['affected_aus'][side]:
                    info['paralysis']['affected_aus'][side].append('AU45_r')

                if au7_available:
                    affected_aus_by_zone_side[side][zone].add('AU07_r')
                    if 'AU07_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU07_r')

                # Track contribution details
                info['paralysis']['contributing_aus'][side][zone]['combined_score'].append({
                    'au': 'AU45_r + AU07_r' if au7_available else 'AU45_r',
                    'current_au45': current_au45,
                    'other_au45': other_au45,
                    'au45_ratio': au45_ratio,
                    'current_au7': current_au7 if au7_available else 'NA',
                    'other_au7': other_au7 if au7_available else 'NA',
                    'au7_ratio': au7_ratio if au7_available else 'NA',
                    'combined_score': combined_score,
                    'threshold': partial_threshold,
                    'type': 'Partial'
                })

                return True  # Successfully processed
            else:
                # No paralysis detected
                info['paralysis']['zones'][side][zone] = 'None'  # String literal
                return True  # Still successfully processed even if no paralysis

        return False  # Couldn't process with combined score approach


# Implementation of process_midface_functional_approach combining left and right versions
def process_midface_functional_approach(self, info, zone, aus, side,
                                        current_values, other_values,
                                        current_values_normalized, other_values_normalized,
                                        confidence_score, zone_paralysis, affected_aus_by_zone_side):
    """
    Process midface zone using a dual-criteria approach based on both AU45_r and AU7_r.

    Args:
        self: The FacialParalysisDetector instance
        info (dict): Current action info dictionary to update
        zone (str): Facial zone being analyzed ('mid')
        aus (list): Action units in this zone
        side (str): 'left' or 'right' - which side we're checking for paralysis
        current_values (dict): AU values for the side being evaluated
        other_values (dict): AU values for the opposite side
        current_values_normalized (dict): Normalized AU values for side being evaluated
        other_values_normalized (dict): Normalized AU values for opposite side
        confidence_score (float): Confidence score for detection
        zone_paralysis (dict): Tracking paralysis by zone
        affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side

    Returns:
        bool: True if successfully processed, False otherwise
    """
    # Double check that we have the mid zone
    if zone != 'mid':
        return False

    # Process the ES action data for AU45_r
    au45_processed = False
    au45_value = 0
    au45_ratio = 0
    other_au45_value = 0

    if 'AU45_r' in current_values_normalized and 'AU45_r' in other_values_normalized:
        au45_value = current_values_normalized['AU45_r']
        other_au45_value = other_values_normalized['AU45_r']

        # Calculate ratio (current side / other side)
        au45_ratio = au45_value / other_au45_value if other_au45_value > 0 else 0
        au45_processed = True

    # Process the BS action data for AU7_r
    au7_processed = False
    au7_value = 0
    au7_ratio = 0
    other_au7_value = 0

    # Get the BS action data if available
    bs_data = None
    for action_name, action_data in info.items():
        if action_name == 'BS':
            bs_data = action_data
            break

    # If BS data is available, extract AU7_r values
    if bs_data and 'left' in bs_data and 'right' in bs_data:
        bs_side_values = bs_data[side]['au_values'] if side in bs_data else None
        bs_other_values = bs_data['left' if side == 'right' else 'right']['au_values']

        bs_side_normalized = bs_data[side].get('normalized_au_values', {}) if side in bs_data else {}
        bs_other_normalized = bs_data['left' if side == 'right' else 'right'].get('normalized_au_values', {})

        if bs_side_normalized and bs_other_normalized and 'AU07_r' in bs_side_normalized and 'AU07_r' in bs_other_normalized:
            au7_value = bs_side_normalized['AU07_r']
            other_au7_value = bs_other_normalized['AU07_r']

            # Calculate ratio (current side / other side)
            au7_ratio = au7_value / other_au7_value if other_au7_value > 0 else 0
            au7_processed = True

    # For backward compatibility, also calculate functional score using AU45_r only
    functional_score = self._calculate_midface_functional_score(au45_value, au45_ratio)

    # If both AU45_r and AU7_r are available, calculate the combined score
    combined_score = functional_score  # Default to AU45_r score
    if au45_processed and au7_processed:
        combined_score = self._calculate_combined_midface_score(au45_value, au45_ratio, au7_value, au7_ratio)

    # Apply looser confidence threshold for midface detection
    adjusted_confidence = max(confidence_score, 0.3)  # Ensure minimum confidence score of 0.3

    # Store confidence score
    info['paralysis']['confidence'][side][zone] = adjusted_confidence

    # Initialize component tracking if not already present
    if 'components' not in info['paralysis']['contributing_aus'][side][zone]:
        info['paralysis']['contributing_aus'][side][zone]['components'] = []

    # For backward compatibility
    if 'functional_score' not in info['paralysis']['contributing_aus'][side][zone]:
        info['paralysis']['contributing_aus'][side][zone]['functional_score'] = []

    # Initialize AU7 tracking if not already present
    if 'au7_components' not in info['paralysis']['contributing_aus'][side][zone]:
        info['paralysis']['contributing_aus'][side][zone]['au7_components'] = []

    # Debug logging to verify values
    logger.debug(
        f"{side} mid face assessment: AU45={au45_value:.2f}, AU45_ratio={au45_ratio:.2f}, " +
        f"AU7={au7_value:.2f}, AU7_ratio={au7_ratio:.2f}, " +
        f"combined_score={combined_score:.2f}, confidence={adjusted_confidence:.2f}"
    )

    # Thresholds for combined score
    combined_complete_threshold = self.midface_combined_score_thresholds['complete']
    combined_partial_threshold = self.midface_combined_score_thresholds['partial']

    # Determine severity based on combined score - complete paralysis
    if (combined_score < combined_complete_threshold and
            adjusted_confidence >= self.confidence_thresholds[zone]['complete']):

        # Complete paralysis - CRITICAL FIX: ALWAYS use string literal for consistency with other zones
        info['paralysis']['zones'][side][zone] = 'Complete'  # Use string 'Complete', not numerical value

        # Track for patient-level assessment - CRITICAL FIX: ALWAYS use string literal
        if zone_paralysis[side][zone] != 'Complete':
            zone_paralysis[side][zone] = 'Complete'  # Use string 'Complete', not numerical value

        # Add to affected AUs
        if au45_processed:
            affected_aus_by_zone_side[side][zone].add('AU45_r')
            if 'AU45_r' not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append('AU45_r')

        if au7_processed:
            affected_aus_by_zone_side[side][zone].add('AU07_r')
            if 'AU07_r' not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append('AU07_r')

        # Track contribution details for AU45
        if au45_processed:
            info['paralysis']['contributing_aus'][side][zone]['components'].append({
                'au': 'AU45_r',
                'value': au45_value,
                'other_value': other_au45_value,
                'ratio': au45_ratio,
                'threshold': self.midface_component_thresholds['functional']['complete'],
                'type': 'Complete'
            })

        # Track contribution details for AU7
        if au7_processed:
            info['paralysis']['contributing_aus'][side][zone]['au7_components'].append({
                'au': 'AU07_r',
                'value': au7_value,
                'other_value': other_au7_value,
                'ratio': au7_ratio,
                'threshold': self.midface_component_thresholds['au7']['complete'],
                'type': 'Complete',
                'action': 'BS'
            })

        # For backward compatibility
        info['paralysis']['contributing_aus'][side][zone]['functional_score'].append({
            'au': 'Combined AU45_r and AU07_r',
            'value': combined_score,
            'threshold': combined_complete_threshold,
            'type': 'Complete'
        })

        logger.debug(f"{side} mid face: COMPLETE paralysis detected with combined AU45_r and AU07_r approach")

    # Determine severity based on component thresholds - partial paralysis
    elif (combined_score < combined_partial_threshold and
          adjusted_confidence >= self.confidence_thresholds[zone]['partial']):

        # Partial paralysis (if not already marked as Complete) - CRITICAL FIX: ALWAYS use string literal
        if info['paralysis']['zones'][side][zone] == 'None':
            info['paralysis']['zones'][side][zone] = 'Partial'  # Use string 'Partial', not numerical value

        # Track for patient-level assessment (if not already Complete) - CRITICAL FIX: ALWAYS use string literal
        if zone_paralysis[side][zone] == 'None':
            zone_paralysis[side][zone] = 'Partial'  # Use string 'Partial', not numerical value

        # Add to affected AUs
        if au45_processed:
            affected_aus_by_zone_side[side][zone].add('AU45_r')
            if 'AU45_r' not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append('AU45_r')

        if au7_processed:
            affected_aus_by_zone_side[side][zone].add('AU07_r')
            if 'AU07_r' not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append('AU07_r')

        # Track contribution details for AU45
        if au45_processed:
            info['paralysis']['contributing_aus'][side][zone]['components'].append({
                'au': 'AU45_r',
                'value': au45_value,
                'other_value': other_au45_value,
                'ratio': au45_ratio,
                'threshold': self.midface_component_thresholds['functional']['partial'],
                'type': 'Partial'
            })

        # Track contribution details for AU7
        if au7_processed:
            info['paralysis']['contributing_aus'][side][zone]['au7_components'].append({
                'au': 'AU07_r',
                'value': au7_value,
                'other_value': other_au7_value,
                'ratio': au7_ratio,
                'threshold': self.midface_component_thresholds['au7']['partial'],
                'type': 'Partial',
                'action': 'BS'
            })

        # For backward compatibility
        info['paralysis']['contributing_aus'][side][zone]['functional_score'].append({
            'au': 'Combined AU45_r and AU07_r',
            'value': combined_score,
            'threshold': combined_partial_threshold,
            'type': 'Partial'
        })

        logger.debug(f"{side} mid face: PARTIAL paralysis detected with combined AU45_r and AU07_r approach")
    else:
        # CRITICAL FIX: Make sure 'None' is explicitly set as a string
        # If no paralysis is detected, ensure we set an explicit 'None' string, not a null value or number
        info['paralysis']['zones'][side][zone] = 'None'
        logger.debug(f"{side} mid face: NO paralysis detected with combined AU45_r and AU07_r approach")

    # Always return True to indicate we processed this zone
    return True

    def process_lower_face_combined_score(self, info, zone, aus, side,
                                          current_side_values, other_side_values,
                                          current_side_normalized, other_side_normalized,
                                          confidence_score, zone_paralysis, affected_aus_by_zone_side,
                                          criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None):
        """Process lower face zone using the combined score approach.

        Args:
            info (dict): Current action info dictionary to update
            zone (str): Facial zone being analyzed ('lower')
            aus (list): Action units in this zone
            side (str): 'left' or 'right' - which side we're checking for paralysis
            current_side_values (dict): AU values for the side being checked
            other_side_values (dict): AU values for the opposite side
            current_side_normalized (dict): Normalized AU values for the side being checked
            other_side_normalized (dict): Normalized AU values for the opposite side
            confidence_score (float): Confidence score for detection
            zone_paralysis (dict): Tracking paralysis by zone
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side
            criteria_met (dict, optional): Dictionary of detection criteria that were met
            asymmetry_thresholds (dict, optional): Thresholds for asymmetry detection
            confidence_thresholds (dict, optional): Thresholds for confidence scores

        Returns:
            bool: True if successfully processed, False otherwise
        """
        if not criteria_met:
            criteria_met = {}
        if not asymmetry_thresholds:
            asymmetry_thresholds = self.asymmetry_thresholds[zone]
        if not confidence_thresholds:
            confidence_thresholds = self.confidence_thresholds[zone]
            
        # Check if we have the necessary AUs for combined score calculation
        if 'AU12_r' in current_side_values and 'AU12_r' in other_side_values:
            # Get the appropriate values for AU12_r
            current_au12 = current_side_values['AU12_r']
            other_au12 = other_side_values['AU12_r']

            if current_side_normalized and other_side_normalized and 'AU12_r' in current_side_normalized and 'AU12_r' in other_side_normalized:
                current_au12 = current_side_normalized['AU12_r']
                other_au12 = other_side_normalized['AU12_r']

            # SPECIAL HANDLING: Check if AU25_r is missing or zero on both sides
            has_au25 = ('AU25_r' in current_side_values and 'AU25_r' in other_side_values and
                        current_side_values['AU25_r'] > 0 and other_side_values['AU25_r'] > 0)

            if not has_au25 and current_au12 > 0 and other_au12 > 0:
                # Special case: Calculate score using only AU12_r
                au12_ratio = min(current_au12, other_au12) / max(current_au12, other_au12)
                au12_min = current_au12  # Since we're checking current side weakness

                # Calculate simplified score for AU12-only
                simplified_score = au12_ratio * (1 + au12_min / 5)

                # Calculate percent difference for AU12
                au12_percent_diff = abs(current_au12 - other_au12) / ((current_au12 + other_au12) / 2) * 100

                # FIX for IMG_4923: Use a special threshold (0.80) for AU12-only detection
                special_threshold = 0.80

                # FIX for IMG_4923: Detect based on either simplified score OR strong percent difference
                if ((simplified_score < special_threshold and current_au12 < other_au12) or
                        (au12_percent_diff > 40 and current_au12 < other_au12)):

                    # FIX for IMG_4923: Special confidence boost for missing AU25 cases
                    adjusted_confidence = max(confidence_score, 0.4)  # Ensure minimum confidence of 0.4

                    # Check confidence with adjusted value
                    if adjusted_confidence >= confidence_thresholds['partial']:
                        # It's partial paralysis
                        info['paralysis']['zones'][side][zone] = 'Partial'

                        # Track for patient-level assessment
                        if zone_paralysis[side][zone] == 'None':
                            zone_paralysis[side][zone] = 'Partial'

                        # Add affected AUs
                        affected_aus_by_zone_side[side][zone].add('AU12_r')
                        if 'AU12_r' not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append('AU12_r')

                        # Initialize the combined_score array if it doesn't exist
                        if 'combined_score' not in info['paralysis']['contributing_aus'][side][zone]:
                            info['paralysis']['contributing_aus'][side][zone]['combined_score'] = []

                        # Track special AU12-only detection with enhanced logging
                        info['paralysis']['contributing_aus'][side][zone]['combined_score'].append({
                            'au': 'AU12_r only (missing AU25)',
                            'current_au12': current_au12,
                            'other_au12': other_au12,
                            'au12_ratio': au12_ratio,
                            'au12_percent_diff': au12_percent_diff,
                            'simplified_score': simplified_score,
                            'threshold': special_threshold,
                            'adjusted_confidence': adjusted_confidence,
                            'original_confidence': confidence_score,
                            'type': 'Partial'
                        })

                        return True  # Successfully processed

            # Continue with standard processing if AU25_r is present or if special case didn't detect paralysis
            if 'AU25_r' in current_side_values and 'AU25_r' in other_side_values:
                # Get the appropriate values for AU25_r
                current_au25 = current_side_values['AU25_r']
                other_au25 = other_side_values['AU25_r']

                if current_side_normalized and other_side_normalized and 'AU25_r' in current_side_normalized and 'AU25_r' in other_side_normalized:
                    current_au25 = current_side_normalized['AU25_r']
                    other_au25 = other_side_normalized['AU25_r']

                # Calculate ratios
                au12_ratio = current_au12 / other_au12 if other_au12 > 0 else 0
                au25_ratio = current_au25 / other_au25 if other_au25 > 0 else 0

                # For detection purposes we want the smaller/larger ratio
                au12_ratio = min(au12_ratio, 1.0) if au12_ratio > 0 else 0
                au25_ratio = min(au25_ratio, 1.0) if au25_ratio > 0 else 0

                # Minimum value (from current side since we're checking current side paralysis)
                au12_min = current_au12
                au25_min = current_au25

                # Calculate combined score
                combined_score = self._calculate_lower_face_combined_score(au12_ratio, au25_ratio, au12_min, au25_min)
                logger.debug(
                    f"{side.capitalize()} lower face combined score: {combined_score:.3f} (AU12_ratio={au12_ratio:.3f}, AU25_ratio={au25_ratio:.3f})")

                # Use thresholds from constants
                partial_threshold = self.lower_face_combined_score_thresholds['partial']
                complete_threshold = self.lower_face_combined_score_thresholds['complete']

                # Initialize the combined_score array if it doesn't exist
                if 'combined_score' not in info['paralysis']['contributing_aus'][side][zone]:
                    info['paralysis']['contributing_aus'][side][zone]['combined_score'] = []

                # Check if meets partial criteria
                if combined_score < partial_threshold and confidence_score >= confidence_thresholds['partial']:
                    # Track contribution for combined score calculation
                    info['paralysis']['contributing_aus'][side][zone]['combined_score'].append({
                        'au': 'Combined Lower Face',
                        'current_au12': current_au12,
                        'other_au12': other_au12,
                        'current_au25': current_au25,
                        'other_au25': other_au25,
                        'au12_ratio': au12_ratio,
                        'au25_ratio': au25_ratio,
                        'combined_score': combined_score,
                        'threshold': partial_threshold,
                        'type': 'Partial'
                    })

                    # See if it meets complete criteria
                    if combined_score < complete_threshold and confidence_score >= confidence_thresholds['complete']:
                        # Complete paralysis
                        info['paralysis']['zones'][side][zone] = 'Complete'

                        # Track for patient-level assessment
                        if zone_paralysis[side][zone] != 'Complete':
                            zone_paralysis[side][zone] = 'Complete'

                        # Update tracking information
                        info['paralysis']['contributing_aus'][side][zone]['combined_score'][-1]['type'] = 'Complete'
                        info['paralysis']['contributing_aus'][side][zone]['combined_score'][-1][
                            'threshold'] = complete_threshold
                    else:
                        # Partial paralysis (if not already marked as Complete)
                        if info['paralysis']['zones'][side][zone] == 'None':
                            info['paralysis']['zones'][side][zone] = 'Partial'

                        # Track for patient-level assessment
                        if zone_paralysis[side][zone] == 'None':
                            zone_paralysis[side][zone] = 'Partial'

                    # Add affected AUs
                    affected_aus_by_zone_side[side][zone].add('AU12_r')
                    if 'AU12_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU12_r')

                    affected_aus_by_zone_side[side][zone].add('AU25_r')
                    if 'AU25_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU25_r')

                    return True  # Successfully processed

        return False  # Could not process with combined score

    def process_upper_face_zone(self, info, zone, aus, side, 
                               current_values, other_values,
                               current_values_normalized, other_values_normalized,
                               confidence_score, zone_paralysis, affected_aus_by_zone_side,
                               criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None,
                               partial_thresholds=None, complete_thresholds=None):
        """Process upper face zone for paralysis detection.

        Args:
            info (dict): Current action info dictionary to update
            zone (str): Facial zone being analyzed ('upper')
            aus (list): Action units in this zone
            side (str): 'left' or 'right' - which side we're checking for paralysis
            current_values (dict): AU values for the side being checked
            other_values (dict): AU values for the opposite side
            current_values_normalized (dict): Normalized AU values for the side being checked
            other_values_normalized (dict): Normalized AU values for the opposite side
            confidence_score (float): Confidence score for detection
            zone_paralysis (dict): Tracking paralysis by zone
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side
            criteria_met (dict, optional): Dictionary of detection criteria that were met
            asymmetry_thresholds (dict, optional): Thresholds for asymmetry detection
            confidence_thresholds (dict, optional): Thresholds for confidence scores
            partial_thresholds (dict, optional): Thresholds for partial paralysis
            complete_thresholds (dict, optional): Thresholds for complete paralysis

        Returns:
            bool: True if successfully processed, False otherwise
        """
        if not criteria_met:
            criteria_met = {}
        if not asymmetry_thresholds:
            asymmetry_thresholds = self.asymmetry_thresholds[zone]
        if not confidence_thresholds:
            confidence_thresholds = self.confidence_thresholds[zone]
        if not partial_thresholds:
            partial_thresholds = self.paralysis_thresholds[zone]['partial']
        if not complete_thresholds:
            complete_thresholds = self.paralysis_thresholds[zone]['complete']

        # Get specific AU normalized percent difference and ratio for key AUs
        # For upper face: AU01_r
        au01_percent_diff = None
        au01_ratio = None
        au02_percent_diff = None
        au02_ratio = None

        if 'AU01_r' in current_values and 'AU01_r' in other_values:
            # Use normalized values if available
            if 'AU01_r' in current_values_normalized and 'AU01_r' in other_values_normalized:
                current_au01 = current_values_normalized['AU01_r']
                other_au01 = other_values_normalized['AU01_r']

                if current_au01 > 0 or other_au01 > 0:
                    if current_au01 > 0 and other_au01 > 0:
                        au01_percent_diff = calculate_percent_difference(current_au01, other_au01)
                        au01_ratio = min(current_au01, other_au01) / max(current_au01, other_au01)
                    else:
                        au01_percent_diff = 100  # One side has zero movement
                        au01_ratio = 0
                    logger.debug(f"AU01_r normalized: percent_diff={au01_percent_diff:.1f}%, ratio={au01_ratio:.3f}")

        if 'AU02_r' in current_values and 'AU02_r' in other_values:
            # Use normalized values if available
            if 'AU02_r' in current_values_normalized and 'AU02_r' in other_values_normalized:
                current_au02 = current_values_normalized['AU02_r']
                other_au02 = other_values_normalized['AU02_r']

                if current_au02 > 0 or other_au02 > 0:
                    if current_au02 > 0 and other_au02 > 0:
                        au02_percent_diff = calculate_percent_difference(current_au02, other_au02)
                        au02_ratio = min(current_au02, other_au02) / max(current_au02, other_au02)
                    else:
                        au02_percent_diff = 100  # One side has zero movement
                        au02_ratio = 0
                    logger.debug(f"AU02_r normalized: percent_diff={au02_percent_diff:.1f}%, ratio={au02_ratio:.3f}")

        # Check each criterion separately and track which ones are met
        criteria_met = {}
        
        # Get weighted average activations
        current_avg = self._calculate_weighted_activation(side, zone, current_values, current_values_normalized)
        other_avg = self._calculate_weighted_activation('left' if side == 'right' else 'right', zone, other_values, other_values_normalized)
        
        # Calculate asymmetry ratio
        ratio = 0
        if current_avg > 0 and other_avg > 0:
            ratio = min(current_avg, other_avg) / max(current_avg, other_avg)

        # Calculate percent difference between sides for each AU
        au_percent_diffs = []
        for au in aus:
            if au in current_values and au in other_values:
                # Determine if we should use normalized values for this AU
                use_normalized = False
                au_base = au.split('_')[0] + '_r'
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                # Get the appropriate values
                current_val = current_values[au]
                other_val = other_values[au]

                if use_normalized and au in current_values_normalized and au in other_values_normalized:
                    current_val = current_values_normalized[au]
                    other_val = other_values_normalized[au]
                    logger.debug(f"Using normalized values for {au} percent diff: current={current_val}, other={other_val}")

                percent_diff = calculate_percent_difference(current_val, other_val)
                au_percent_diffs.append(percent_diff)

        # Get maximum percent difference (most asymmetric AU)
        max_percent_diff = max(au_percent_diffs) if au_percent_diffs else 0

        # Check for extreme asymmetry in any individual AU with normalized values if available
        has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, confidence_boost = self._check_for_extreme_asymmetry(
            current_values, other_values, current_values_normalized, other_values_normalized, zone
        )

        # Basic criteria
        criteria_met['minimal_movement'] = current_avg < complete_thresholds['minimal_movement']
        criteria_met['ratio'] = (current_avg > 0 and other_avg > 0 and
                               ratio < asymmetry_thresholds['complete']['ratio'] and
                               current_avg < other_avg)
        criteria_met['percent_diff'] = (max_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and
                                      current_avg < other_avg)
        criteria_met['extreme_asymmetry'] = (has_extreme_asymmetry and weaker_side == side)

        # Special case for AU01_r normalized ratio (upper face)
        if au01_ratio is not None and au01_percent_diff is not None:
            # If AU01_r normalized ratio is below threshold and this side is weaker
            if (au01_ratio < asymmetry_thresholds['complete']['ratio'] and 
                current_values_normalized['AU01_r'] < other_values_normalized['AU01_r']):
                criteria_met['au01_normalized_ratio'] = True
                logger.debug(
                    f"{side.upper()}: AU01_r normalized ratio is below threshold: {au01_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            side, zone, current_values, other_values, criteria_met
        )

        # Apply any confidence boost from extreme asymmetry detection
        confidence_score += confidence_boost

        # Check for borderline cases
        is_borderline, adjusted_confidence = self._check_for_borderline_cases(
            zone, side, current_values, other_values, ratio, max_percent_diff, confidence_score
        )

        # Update confidence score if borderline case
        if is_borderline:
            confidence_score = adjusted_confidence
            criteria_met['borderline_case'] = True

        # Store confidence score
        info['paralysis']['confidence'][side][zone] = confidence_score

        # Two-tier asymmetry detection for upper face
        # Check if primary AU (AU01_r) has extreme asymmetry and any asymmetry in other AUs
        primary_au_extreme = False
        other_au_moderate = False

        for au in aus:
            au_base = au.split('_')[0] + '_r'
            if au in current_values and au in other_values:
                # Get the appropriate values (normalized if available)
                current_val = current_values[au]
                other_val = other_values[au]

                use_normalized = False
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                if use_normalized and au in current_values_normalized and au in other_values_normalized:
                    current_val = current_values_normalized[au]
                    other_val = other_values_normalized[au]

                percent_diff = calculate_percent_difference(current_val, other_val)

                # Set threshold for AU01_r
                au01_threshold = 120

                # Check primary AU (AU01_r)
                if au_base == 'AU01_r' and percent_diff > au01_threshold and current_val < other_val:
                    primary_au_extreme = True
                    logger.debug(f"{side} primary AU01_r has extreme asymmetry: {percent_diff:.1f}% > {au01_threshold}%")
                # Check other AUs for any asymmetry
                elif au_base != 'AU01_r' and percent_diff > 25 and current_val < other_val:
                    other_au_moderate = True

        # Initialize the tracking structures if they don't exist yet
        if 'extreme_asymmetry' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['extreme_asymmetry'] = []
        if 'normalized_ratio' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['normalized_ratio'] = []
        if 'borderline_case' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['borderline_case'] = []
        if 'minimal_movement' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['minimal_movement'] = []
        if 'asymmetry' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['asymmetry'] = []
        if 'percent_diff' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['percent_diff'] = []
            
        # Decision logic for upper face paralysis detection
        # Process through each detection path and update paralysis status
        
        # First priority: Extreme asymmetry
        if criteria_met['extreme_asymmetry']:
            # If zone has extreme asymmetry, set as Complete
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Track contributing AUs and add to affected_aus
            affected_aus_by_zone_side[side][zone].add(extreme_au)
            if extreme_au not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append(extreme_au)

            # Track AU values and thresholds for extreme asymmetry
            info['paralysis']['contributing_aus'][side][zone]['extreme_asymmetry'].append({
                'au': extreme_au,
                'current_value': current_values[extreme_au],
                'other_value': other_values[extreme_au],
                'percent_diff': extreme_percent_diff,
                'threshold': self.au01_extreme_asymmetry_threshold if extreme_au == 'AU01_r' else EXTREME_ASYMMETRY_THRESHOLD,
                'type': 'Complete'
            })
            
        # Second priority: AU01_r normalized ratio
        elif criteria_met.get('au01_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
            # Upper Face verification - check if the other side actually has more movement
            # Only mark as paralyzed if this side actually has less movement
            if 'AU01_r' in current_values and 'AU01_r' in other_values:
                if current_values['AU01_r'] < other_values['AU01_r']:
                    # AU01_r normalized ratio indicates complete paralysis when this side is actually weaker
                    info['paralysis']['zones'][side][zone] = 'Complete'

                    # Track for patient-level assessment
                    if zone_paralysis[side][zone] != 'Complete':
                        zone_paralysis[side][zone] = 'Complete'

                    # Track contributing AUs
                    affected_aus_by_zone_side[side][zone].add('AU01_r')
                    if 'AU01_r' not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append('AU01_r')

                    # Track values for this detection
                    info['paralysis']['contributing_aus'][side][zone]['normalized_ratio'].append({
                        'au': 'AU01_r',
                        'current_value': current_values_normalized['AU01_r'],
                        'other_value': other_values_normalized['AU01_r'],
                        'ratio': au01_ratio,
                        'threshold': asymmetry_thresholds['complete']['ratio'],
                        'type': 'Complete',
                        'normalized': True
                    })
                else:
                    logger.debug(f"Skipping upper face detection - current AU01_r isn't weaker ({current_values['AU01_r']:.3f} vs {other_values['AU01_r']:.3f})")
                    
        # Third priority: Borderline cases
        elif criteria_met.get('borderline_case', False) and confidence_score >= confidence_thresholds['complete']:
            # For upper face, verify side direction
            if 'AU01_r' in current_values and 'AU01_r' in other_values:
                if current_values['AU01_r'] < other_values['AU01_r']:
                    info['paralysis']['zones'][side][zone] = 'Complete'
                    if zone_paralysis[side][zone] != 'Complete':
                        zone_paralysis[side][zone] = 'Complete'
                else:
                    logger.debug(f"Skipping upper face borderline case - current AU01_r isn't weaker")
                    return True

            # Track contributing AUs for borderline case (if we proceed with detection)
            if info['paralysis']['zones'][side][zone] == 'Complete':
                for au in aus:
                    if au in current_values and au in other_values:
                        percent_diff = calculate_percent_difference(current_values[au], other_values[au])
                        if percent_diff > 70.0 and current_values[au] < other_values[au]:
                            affected_aus_by_zone_side[side][zone].add(au)
                            if au not in info['paralysis']['affected_aus'][side]:
                                info['paralysis']['affected_aus'][side].append(au)

                            # Track AU values for borderline case
                            info['paralysis']['contributing_aus'][side][zone]['borderline_case'].append({
                                'au': au,
                                'current_value': current_values[au],
                                'other_value': other_values[au],
                                'percent_diff': percent_diff,
                                'ratio': current_values[au] / other_values[au] if other_values[au] > 0 else 0,
                                'type': 'Complete'
                            })
                            
        # Fourth priority: Standard thresholds for complete paralysis
        elif ((criteria_met['minimal_movement'] or criteria_met['ratio']) and
              confidence_score >= confidence_thresholds['complete']):

            # For upper face, verify we're detecting the correct side
            if 'AU01_r' in current_values and 'AU01_r' in other_values and 'AU02_r' in current_values and 'AU02_r' in other_values:
                au01_current_weaker = current_values['AU01_r'] < other_values['AU01_r']
                au02_current_weaker = current_values['AU02_r'] < other_values['AU02_r']

                # At least AU01_r (more important) must show this side being weaker
                if not au01_current_weaker:
                    logger.debug(f"Skipping upper face detection - current AU01_r isn't weaker ({current_values['AU01_r']:.3f} vs {other_values['AU01_r']:.3f})")
                    # Don't mark this side as paralyzed when it shows more movement
                    return True

                # Set full paralysis
                info['paralysis']['zones'][side][zone] = 'Complete'

                # Track for patient-level assessment
                if zone_paralysis[side][zone] != 'Complete':
                    zone_paralysis[side][zone] = 'Complete'

                # Track contributing AUs and add to affected_aus
                for au in aus:
                    if au in current_values:
                        # Check if we should use normalized value for this AU
                        use_normalized = False
                        au_base = au.split('_')[0] + '_r'
                        if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                            use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                        # Get the appropriate value
                        curr_val = current_values[au]
                        if use_normalized and au in current_values_normalized:
                            curr_val = current_values_normalized[au]

                        # Check if AU is below threshold
                        if curr_val < complete_thresholds['minimal_movement']:
                            affected_aus_by_zone_side[side][zone].add(au)
                            if au not in info['paralysis']['affected_aus'][side]:
                                info['paralysis']['affected_aus'][side].append(au)

                            # Track AU values and thresholds
                            info['paralysis']['contributing_aus'][side][zone]['minimal_movement'].append({
                                'au': au,
                                'value': curr_val,
                                'threshold': complete_thresholds['minimal_movement'],
                                'type': 'Complete',
                                'normalized': use_normalized
                            })

                        # Check if AU contributes to ratio-based detection
                        elif au in other_values:
                            # Get the appropriate other value
                            other_val = other_values[au]
                            if use_normalized and au in other_values_normalized:
                                other_val = other_values_normalized[au]

                            if other_val > 0 and curr_val > 0:
                                au_ratio = min(curr_val, other_val) / max(curr_val, other_val)
                                if au_ratio < asymmetry_thresholds['complete']['ratio'] and curr_val < other_val:
                                    affected_aus_by_zone_side[side][zone].add(au)
                                    if au not in info['paralysis']['affected_aus'][side]:
                                        info['paralysis']['affected_aus'][side].append(au)

                                    # Track AU values and thresholds
                                    info['paralysis']['contributing_aus'][side][zone]['asymmetry'].append({
                                        'au': au,
                                        'current_value': curr_val,
                                        'other_value': other_val,
                                        'ratio': au_ratio,
                                        'threshold': asymmetry_thresholds['complete']['ratio'],
                                        'type': 'Complete',
                                        'normalized': use_normalized
                                    })

        # Fifth priority: Partial paralysis detection
        elif ((criteria_met.get('minimal_movement', False) or criteria_met.get('percent_diff', False)) and
              confidence_score >= confidence_thresholds['partial']):
            
            # Only assign if not already marked as complete
            if info['paralysis']['zones'][side][zone] == 'None':
                info['paralysis']['zones'][side][zone] = 'Partial'

            # Track for patient-level assessment - only update if not already complete
            if zone_paralysis[side][zone] == 'None':
                zone_paralysis[side][zone] = 'Partial'

            # Track contributing AUs and add to affected_aus
            for au in aus:
                if au in current_values:
                    # Check if we should use normalized value for this AU
                    use_normalized = False
                    au_base = au.split('_')[0] + '_r'
                    if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                        use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                    # Get the appropriate values
                    curr_val = current_values[au]
                    if use_normalized and au in current_values_normalized:
                        curr_val = current_values_normalized[au]

                    # Check if AU is below threshold but above complete threshold
                    if (curr_val < partial_thresholds['minimal_movement'] and
                            curr_val >= complete_thresholds['minimal_movement']):
                        affected_aus_by_zone_side[side][zone].add(au)
                        if au not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append(au)

                        # Track AU values and thresholds
                        info['paralysis']['contributing_aus'][side][zone]['minimal_movement'].append({
                            'au': au,
                            'value': curr_val,
                            'threshold': partial_thresholds['minimal_movement'],
                            'type': 'Partial',
                            'normalized': use_normalized
                        })

                    # Check if AU contributes to percent difference detection
                    elif au in other_values:
                        # Get the appropriate other value
                        other_val = other_values[au]
                        if use_normalized and au in other_values_normalized:
                            other_val = other_values_normalized[au]

                        au_percent_diff = calculate_percent_difference(curr_val, other_val)
                        if au_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and curr_val < other_val:
                            affected_aus_by_zone_side[side][zone].add(au)
                            if au not in info['paralysis']['affected_aus'][side]:
                                info['paralysis']['affected_aus'][side].append(au)

                            # Track AU values and thresholds
                            info['paralysis']['contributing_aus'][side][zone]['percent_diff'].append({
                                'au': au,
                                'current_value': curr_val,
                                'other_value': other_val,
                                'percent_diff': au_percent_diff,
                                'threshold': asymmetry_thresholds['partial']['percent_diff'],
                                'type': 'Partial',
                                'normalized': use_normalized
                            })

        # Two-tier detection as a potential override - check if primary AU has extreme asymmetry
        if primary_au_extreme and (other_au_moderate or zone == 'upper') and info['paralysis']['zones'][side][zone] == 'None':
            # Set as Complete - for upper face, we don't necessarily need other AU moderate
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Add to affected AUs
            for au in aus:
                au_base = au.split('_')[0] + '_r'
                if au_base == 'AU01_r':
                    affected_aus_by_zone_side[side][zone].add(au)
                    if au not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append(au)

        return True

    def process_lower_face_zone(self, info, zone, aus, side, 
                               current_values, other_values,
                               current_values_normalized, other_values_normalized,
                               confidence_score, zone_paralysis, affected_aus_by_zone_side,
                               criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None,
                               partial_thresholds=None, complete_thresholds=None):
        """Process lower face zone for paralysis detection.

        Args:
            info (dict): Current action info dictionary to update
            zone (str): Facial zone being analyzed ('lower')
            aus (list): Action units in this zone
            side (str): 'left' or 'right' - which side we're checking for paralysis
            current_values (dict): AU values for the side being checked
            other_values (dict): AU values for the opposite side
            current_values_normalized (dict): Normalized AU values for the side being checked
            other_values_normalized (dict): Normalized AU values for the opposite side
            confidence_score (float): Confidence score for detection
            zone_paralysis (dict): Tracking paralysis by zone
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side
            criteria_met (dict, optional): Dictionary of detection criteria that were met
            asymmetry_thresholds (dict, optional): Thresholds for asymmetry detection
            confidence_thresholds (dict, optional): Thresholds for confidence scores
            partial_thresholds (dict, optional): Thresholds for partial paralysis
            complete_thresholds (dict, optional): Thresholds for complete paralysis

        Returns:
            bool: True if successfully processed, False otherwise
        """
        if not criteria_met:
            criteria_met = {}
        if not asymmetry_thresholds:
            asymmetry_thresholds = self.asymmetry_thresholds[zone]
        if not confidence_thresholds:
            confidence_thresholds = self.confidence_thresholds[zone]
        if not partial_thresholds:
            partial_thresholds = self.paralysis_thresholds[zone]['partial']
        if not complete_thresholds:
            complete_thresholds = self.paralysis_thresholds[zone]['complete']
        
        # First try the combined score approach for lower face
        if hasattr(self, 'use_combined_score_for_lower_face') and self.use_combined_score_for_lower_face:
            processed = self.process_lower_face_combined_score(
                info, zone, aus, side, 
                current_values, other_values,
                current_values_normalized, other_values_normalized,
                confidence_score, zone_paralysis, affected_aus_by_zone_side,
                criteria_met, asymmetry_thresholds, confidence_thresholds
            )
            if processed:
                return True  # Successfully processed with combined score

        # Get specific AU normalized percent difference and ratio for key AUs
        # For lower face: AU12_r
        au12_percent_diff = None
        au12_ratio = None
        au25_percent_diff = None
        au25_ratio = None

        if 'AU12_r' in current_values and 'AU12_r' in other_values:
            # Use normalized values if available
            if 'AU12_r' in current_values_normalized and 'AU12_r' in other_values_normalized:
                current_au12 = current_values_normalized['AU12_r']
                other_au12 = other_values_normalized['AU12_r']

                if current_au12 > 0 or other_au12 > 0:
                    if current_au12 > 0 and other_au12 > 0:
                        au12_percent_diff = calculate_percent_difference(current_au12, other_au12)
                        au12_ratio = min(current_au12, other_au12) / max(current_au12, other_au12)
                    else:
                        au12_percent_diff = 100  # One side has zero movement
                        au12_ratio = 0
                    logger.debug(f"AU12_r normalized: percent_diff={au12_percent_diff:.1f}%, ratio={au12_ratio:.3f}")

        if 'AU25_r' in current_values and 'AU25_r' in other_values:
            # Use normalized values if available
            if 'AU25_r' in current_values_normalized and 'AU25_r' in other_values_normalized:
                current_au25 = current_values_normalized['AU25_r']
                other_au25 = other_values_normalized['AU25_r']

                if current_au25 > 0 or other_au25 > 0:
                    if current_au25 > 0 and other_au25 > 0:
                        au25_percent_diff = calculate_percent_difference(current_au25, other_au25)
                        au25_ratio = min(current_au25, other_au25) / max(current_au25, other_au25)
                    else:
                        au25_percent_diff = 100  # One side has zero movement
                        au25_ratio = 0
                    logger.debug(f"AU25_r normalized: percent_diff={au25_percent_diff:.1f}%, ratio={au25_ratio:.3f}")

        # Check for extreme asymmetry in any individual AU with normalized values if available
        has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, confidence_boost = self._check_for_extreme_asymmetry(
            current_values, other_values, current_values_normalized, other_values_normalized, zone
        )

        # Get weighted average activations
        current_avg = self._calculate_weighted_activation(side, zone, current_values, current_values_normalized)
        other_avg = self._calculate_weighted_activation('left' if side == 'right' else 'right', zone, other_values, other_values_normalized)
        
        # Calculate asymmetry ratio
        ratio = 0
        if current_avg > 0 and other_avg > 0:
            ratio = min(current_avg, other_avg) / max(current_avg, other_avg)

        # Calculate percent difference between sides for each AU
        au_percent_diffs = []
        for au in aus:
            if au in current_values and au in other_values:
                # Determine if we should use normalized values for this AU
                use_normalized = False
                au_base = au.split('_')[0] + '_r'
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                # Get the appropriate values
                current_val = current_values[au]
                other_val = other_values[au]

                if use_normalized and au in current_values_normalized and au in other_values_normalized:
                    current_val = current_values_normalized[au]
                    other_val = other_values_normalized[au]
                    logger.debug(f"Using normalized values for {au} percent diff: current={current_val}, other={other_val}")

                percent_diff = calculate_percent_difference(current_val, other_val)
                au_percent_diffs.append(percent_diff)

        # Get maximum percent difference (most asymmetric AU)
        max_percent_diff = max(au_percent_diffs) if au_percent_diffs else 0

        # Check each criterion separately and track which ones are met
        criteria_met = {
            'minimal_movement': current_avg < complete_thresholds['minimal_movement'],
            'ratio': (current_avg > 0 and other_avg > 0 and
                    ratio < asymmetry_thresholds['complete']['ratio'] and
                    current_avg < other_avg),
            'percent_diff': (max_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and
                           current_avg < other_avg),
            'extreme_asymmetry': (has_extreme_asymmetry and weaker_side == side)
        }

        # Special case for AU12_r normalized values in lower face
        if au12_ratio is not None and au12_percent_diff is not None:
            # If AU12_r normalized ratio is below threshold and this side is weaker
            if (au12_ratio < asymmetry_thresholds['complete']['ratio'] and 
                current_values_normalized['AU12_r'] < other_values_normalized['AU12_r']):
                criteria_met['au12_normalized_ratio'] = True
                logger.debug(
                    f"{side.upper()}: AU12_r normalized ratio is below threshold: {au12_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")

            # If AU12_r percent difference exceeds partial threshold and this side is weaker
            if (au12_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and 
                current_values_normalized['AU12_r'] < other_values_normalized['AU12_r']):
                criteria_met['au12_percent_diff'] = True
                logger.debug(
                    f"{side.upper()}: AU12_r normalized percent diff exceeds threshold: {au12_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")

        # Special case for AU25_r normalized values in lower face
        if au25_ratio is not None and au25_percent_diff is not None:
            # If AU25_r normalized ratio is below threshold and this side is weaker
            if (au25_ratio < asymmetry_thresholds['complete']['ratio'] and 
                current_values_normalized['AU25_r'] < other_values_normalized['AU25_r']):
                criteria_met['au25_normalized_ratio'] = True
                logger.debug(
                    f"{side.upper()}: AU25_r normalized ratio is below threshold: {au25_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")

            # If AU25_r percent difference exceeds partial threshold and this side is weaker
            if (au25_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and 
                current_values_normalized['AU25_r'] < other_values_normalized['AU25_r']):
                criteria_met['au25_percent_diff'] = True
                logger.debug(
                    f"{side.upper()}: AU25_r normalized percent diff exceeds threshold: {au25_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")

        # Special logic for lower face - individual AU analysis
        individual_au_criteria_met = False

        # Check AU12_r for clear asymmetry
        if au12_ratio is not None and au12_ratio < asymmetry_thresholds['complete']['ratio'] and current_values_normalized['AU12_r'] < other_values_normalized['AU12_r']:
            individual_au_criteria_met = True
            logger.debug(
                f"{side.upper()}: Individual AU12_r shows strong asymmetry, ratio: {au12_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")
            confidence_score += 0.1  # Boost confidence

            # Initialize the individual_au array if it doesn't exist
            if 'individual_au' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['individual_au'] = []

            # Track AU values for individual AU detection
            info['paralysis']['contributing_aus'][side][zone]['individual_au'].append({
                'au': 'AU12_r',
                'current_value': current_values_normalized['AU12_r'],
                'other_value': other_values_normalized['AU12_r'],
                'ratio': au12_ratio,
                'threshold': asymmetry_thresholds['complete']['ratio'],
                'type': 'Complete',
                'normalized': True
            })

        # Check AU25_r for clear asymmetry
        if au25_ratio is not None and au25_ratio < asymmetry_thresholds['complete']['ratio'] and current_values_normalized['AU25_r'] < other_values_normalized['AU25_r']:
            individual_au_criteria_met = True
            logger.debug(
                f"{side.upper()}: Individual AU25_r shows strong asymmetry, ratio: {au25_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")
            confidence_score += 0.1  # Boost confidence

            # Initialize the individual_au array if it doesn't exist
            if 'individual_au' not in info['paralysis']['contributing_aus'][side][zone]:
                info['paralysis']['contributing_aus'][side][zone]['individual_au'] = []

            # Track AU values for individual AU detection
            info['paralysis']['contributing_aus'][side][zone]['individual_au'].append({
                'au': 'AU25_r',
                'current_value': current_values_normalized['AU25_r'],
                'other_value': other_values_normalized['AU25_r'],
                'ratio': au25_ratio,
                'threshold': asymmetry_thresholds['complete']['ratio'],
                'type': 'Complete',
                'normalized': True
            })

        # Check for partial asymmetry based on percent difference
        if au12_percent_diff is not None and au12_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and current_values_normalized['AU12_r'] < other_values_normalized['AU12_r']:
            if not individual_au_criteria_met:  # Only count if not already counted for complete
                individual_au_criteria_met = True
                logger.debug(
                    f"{side.upper()}: Individual AU12_r shows moderate asymmetry, percent diff: {au12_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")
                confidence_score += 0.05  # Small boost for partial

                # Initialize the individual_au array if it doesn't exist
                if 'individual_au' not in info['paralysis']['contributing_aus'][side][zone]:
                    info['paralysis']['contributing_aus'][side][zone]['individual_au'] = []

                # Track AU values for individual AU detection
                info['paralysis']['contributing_aus'][side][zone]['individual_au'].append({
                    'au': 'AU12_r',
                    'current_value': current_values_normalized['AU12_r'],
                    'other_value': other_values_normalized['AU12_r'],
                    'percent_diff': au12_percent_diff,
                    'threshold': asymmetry_thresholds['partial']['percent_diff'],
                    'type': 'Partial',
                    'normalized': True
                })

        if au25_percent_diff is not None and au25_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and current_values_normalized['AU25_r'] < other_values_normalized['AU25_r']:
            if not individual_au_criteria_met:  # Only count if not already counted for complete
                individual_au_criteria_met = True
                logger.debug(
                    f"{side.upper()}: Individual AU25_r shows moderate asymmetry, percent diff: {au25_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")
                confidence_score += 0.05  # Small boost for partial

                # Initialize the individual_au array if it doesn't exist
                if 'individual_au' not in info['paralysis']['contributing_aus'][side][zone]:
                    info['paralysis']['contributing_aus'][side][zone]['individual_au'] = []

                # Track AU values for individual AU detection
                info['paralysis']['contributing_aus'][side][zone]['individual_au'].append({
                    'au': 'AU25_r',
                    'current_value': current_values_normalized['AU25_r'],
                    'other_value': other_values_normalized['AU25_r'],
                    'percent_diff': au25_percent_diff,
                    'threshold': asymmetry_thresholds['partial']['percent_diff'],
                    'type': 'Partial',
                    'normalized': True
                })

        # Add to criteria met
        criteria_met['individual_au'] = individual_au_criteria_met

        # Check for borderline cases
        is_borderline, adjusted_confidence = self._check_for_borderline_cases(
            zone, side, current_values, other_values, ratio, max_percent_diff, confidence_score
        )

        # Update confidence score if borderline case
        if is_borderline:
            confidence_score = adjusted_confidence
            criteria_met['borderline_case'] = True

        # Store confidence score
        info['paralysis']['confidence'][side][zone] = confidence_score

        # Two-tier asymmetry detection for lower face
        # Check if primary AU (AU12_r) has extreme asymmetry and any asymmetry in other AUs
        primary_au_extreme = False
        other_au_moderate = False

        for au in aus:
            au_base = au.split('_')[0] + '_r'
            if au in current_values and au in other_values:
                # Get the appropriate values (normalized if available)
                current_val = current_values[au]
                other_val = other_values[au]

                use_normalized = False
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                if use_normalized and au in current_values_normalized and au in other_values_normalized:
                    current_val = current_values_normalized[au]
                    other_val = other_values_normalized[au]

                percent_diff = calculate_percent_difference(current_val, other_val)

                # Reduced threshold for AU12_r to catch more cases
                au12_threshold = 120  # Reduced from 150

                # Check primary AU (AU12_r)
                if au_base == 'AU12_r' and percent_diff > au12_threshold and current_val < other_val:
                    primary_au_extreme = True
                    logger.debug(f"{side} primary AU12_r has extreme asymmetry: {percent_diff:.1f}% > {au12_threshold}%")
                # Check other AUs for any asymmetry
                elif au_base != 'AU12_r' and percent_diff > 25 and current_val < other_val:
                    other_au_moderate = True

        # Initialize tracking structures if they don't exist yet
        if 'extreme_asymmetry' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['extreme_asymmetry'] = []
        if 'normalized_ratio' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['normalized_ratio'] = []
        if 'borderline_case' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['borderline_case'] = []
        if 'individual_au' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['individual_au'] = []
        if 'minimal_movement' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['minimal_movement'] = []
        if 'asymmetry' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['asymmetry'] = []
        if 'percent_diff' not in info['paralysis']['contributing_aus'][side][zone]:
            info['paralysis']['contributing_aus'][side][zone]['percent_diff'] = []

        # Decision logic for lower face paralysis detection
        # Process through each detection path and update paralysis status

        # First priority: Extreme asymmetry
        if criteria_met['extreme_asymmetry']:
            # If zone has extreme asymmetry, set as Complete
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Track contributing AUs and add to affected_aus
            affected_aus_by_zone_side[side][zone].add(extreme_au)
            if extreme_au not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append(extreme_au)

            # Track AU values and thresholds for extreme asymmetry
            info['paralysis']['contributing_aus'][side][zone]['extreme_asymmetry'].append({
                'au': extreme_au,
                'current_value': current_values[extreme_au],
                'other_value': other_values[extreme_au],
                'percent_diff': extreme_percent_diff,
                'threshold': self.au12_extreme_asymmetry_threshold if extreme_au == 'AU12_r' else self.extreme_asymmetry_threshold,
                'type': 'Complete'
            })

        # Second priority: Individual AU detection
        elif criteria_met.get('individual_au', False) and confidence_score >= confidence_thresholds['complete']:
            logger.debug(f"{side.upper()}: Using individual AU detection for lower face, confidence: {confidence_score:.3f}")

            # Set as Complete
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

        # Third priority: AU12_r normalized ratio
        elif criteria_met.get('au12_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
            # AU12_r normalized ratio indicates complete paralysis
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Track contributing AUs
            affected_aus_by_zone_side[side][zone].add('AU12_r')
            if 'AU12_r' not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append('AU12_r')

            # Track values for this detection
            info['paralysis']['contributing_aus'][side][zone]['normalized_ratio'].append({
                'au': 'AU12_r',
                'current_value': current_values_normalized['AU12_r'],
                'other_value': other_values_normalized['AU12_r'],
                'ratio': au12_ratio,
                'threshold': asymmetry_thresholds['complete']['ratio'],
                'type': 'Complete',
                'normalized': True
            })

        # Fourth priority: AU25_r normalized ratio
        elif criteria_met.get('au25_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
            # AU25_r normalized ratio indicates complete paralysis
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Track contributing AUs
            affected_aus_by_zone_side[side][zone].add('AU25_r')
            if 'AU25_r' not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append('AU25_r')

            # Track values for this detection
            info['paralysis']['contributing_aus'][side][zone]['normalized_ratio'].append({
                'au': 'AU25_r',
                'current_value': current_values_normalized['AU25_r'],
                'other_value': other_values_normalized['AU25_r'],
                'ratio': au25_ratio,
                'threshold': asymmetry_thresholds['complete']['ratio'],
                'type': 'Complete',
                'normalized': True
            })

        # Fifth priority: Partial paralysis based on individual AU percent difference
        elif (criteria_met.get('au12_percent_diff', False) or criteria_met.get('au25_percent_diff', False)) and confidence_score >= confidence_thresholds['partial']:
            # Only assign if not already marked as complete
            if info['paralysis']['zones'][side][zone] == 'None':
                info['paralysis']['zones'][side][zone] = 'Partial'

            # Track for patient-level assessment - only update if not already complete
            if zone_paralysis[side][zone] == 'None':
                zone_paralysis[side][zone] = 'Partial'

            # Track contributing AUs
            if criteria_met.get('au12_percent_diff', False):
                affected_aus_by_zone_side[side][zone].add('AU12_r')
                if 'AU12_r' not in info['paralysis']['affected_aus'][side]:
                    info['paralysis']['affected_aus'][side].append('AU12_r')

            if criteria_met.get('au25_percent_diff', False):
                affected_aus_by_zone_side[side][zone].add('AU25_r')
                if 'AU25_r' not in info['paralysis']['affected_aus'][side]:
                    info['paralysis']['affected_aus'][side].append('AU25_r')

        # Sixth priority: Borderline cases
        elif criteria_met.get('borderline_case', False) and confidence_score >= confidence_thresholds['complete']:
            # Handle borderline cases with specific checks
            info['paralysis']['zones'][side][zone] = 'Complete'
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Track contributing AUs for borderline case
            for au in aus:
                if au in current_values and au in other_values:
                    percent_diff = calculate_percent_difference(current_values[au], other_values[au])
                    if percent_diff > 70.0 and current_values[au] < other_values[au]:
                        affected_aus_by_zone_side[side][zone].add(au)
                        if au not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append(au)

                        # Track AU values for borderline case
                        info['paralysis']['contributing_aus'][side][zone]['borderline_case'].append({
                            'au': au,
                            'current_value': current_values[au],
                            'other_value': other_values[au],
                            'percent_diff': percent_diff,
                            'ratio': current_values[au] / other_values[au] if other_values[au] > 0 else 0,
                            'type': 'Complete'
                        })

        # Seventh priority: Standard thresholds for complete paralysis
        elif ((criteria_met['minimal_movement'] or criteria_met['ratio']) and
              confidence_score >= confidence_thresholds['complete']):

            # Set full paralysis
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Track contributing AUs and add to affected_aus
            for au in aus:
                if au in current_values:
                    # Check if we should use normalized value for this AU
                    use_normalized = False
                    au_base = au.split('_')[0] + '_r'
                    if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                        use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                    # Get the appropriate value
                    curr_val = current_values[au]
                    if use_normalized and au in current_values_normalized:
                        curr_val = current_values_normalized[au]

                    # Check if AU is below threshold
                    if curr_val < complete_thresholds['minimal_movement']:
                        affected_aus_by_zone_side[side][zone].add(au)
                        if au not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append(au)

                        # Track AU values and thresholds
                        info['paralysis']['contributing_aus'][side][zone]['minimal_movement'].append({
                            'au': au,
                            'value': curr_val,
                            'threshold': complete_thresholds['minimal_movement'],
                            'type': 'Complete',
                            'normalized': use_normalized
                        })

                    # Check if AU contributes to ratio-based detection
                    elif au in other_values:
                        # Get the appropriate other value
                        other_val = other_values[au]
                        if use_normalized and au in other_values_normalized:
                            other_val = other_values_normalized[au]

                        if other_val > 0 and curr_val > 0:
                            au_ratio = min(curr_val, other_val) / max(curr_val, other_val)
                            if au_ratio < asymmetry_thresholds['complete']['ratio'] and curr_val < other_val:
                                affected_aus_by_zone_side[side][zone].add(au)
                                if au not in info['paralysis']['affected_aus'][side]:
                                    info['paralysis']['affected_aus'][side].append(au)

                                # Track AU values and thresholds
                                info['paralysis']['contributing_aus'][side][zone]['asymmetry'].append({
                                    'au': au,
                                    'current_value': curr_val,
                                    'other_value': other_val,
                                    'ratio': au_ratio,
                                    'threshold': asymmetry_thresholds['complete']['ratio'],
                                    'type': 'Complete',
                                    'normalized': use_normalized
                                })

        # Eighth priority: Partial paralysis detection
        elif ((criteria_met.get('minimal_movement', False) or criteria_met.get('percent_diff', False)) and
              confidence_score >= confidence_thresholds['partial']):
              
            # Only assign if not already marked as complete
            if info['paralysis']['zones'][side][zone] == 'None':
                info['paralysis']['zones'][side][zone] = 'Partial'

            # Track for patient-level assessment - only update if not already complete
            if zone_paralysis[side][zone] == 'None':
                zone_paralysis[side][zone] = 'Partial'

            # Track contributing AUs and add to affected_aus
            for au in aus:
                if au in current_values:
                    # Check if we should use normalized value for this AU
                    use_normalized = False
                    au_base = au.split('_')[0] + '_r'
                    if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                        use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                    # Get the appropriate values
                    curr_val = current_values[au]
                    if use_normalized and au in current_values_normalized:
                        curr_val = current_values_normalized[au]

                    # Check if AU is below threshold but above complete threshold
                    if (curr_val < partial_thresholds['minimal_movement'] and
                            curr_val >= complete_thresholds['minimal_movement']):
                        affected_aus_by_zone_side[side][zone].add(au)
                        if au not in info['paralysis']['affected_aus'][side]:
                            info['paralysis']['affected_aus'][side].append(au)

                        # Track AU values and thresholds
                        info['paralysis']['contributing_aus'][side][zone]['minimal_movement'].append({
                            'au': au,
                            'value': curr_val,
                            'threshold': partial_thresholds['minimal_movement'],
                            'type': 'Partial',
                            'normalized': use_normalized
                        })

                    # Check if AU contributes to percent difference detection
                    elif au in other_values:
                        # Get the appropriate other value
                        other_val = other_values[au]
                        if use_normalized and au in other_values_normalized:
                            other_val = other_values_normalized[au]

                        au_percent_diff = calculate_percent_difference(curr_val, other_val)
                        if au_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and curr_val < other_val:
                            affected_aus_by_zone_side[side][zone].add(au)
                            if au not in info['paralysis']['affected_aus'][side]:
                                info['paralysis']['affected_aus'][side].append(au)

                            # Track AU values and thresholds
                            info['paralysis']['contributing_aus'][side][zone]['percent_diff'].append({
                                'au': au,
                                'current_value': curr_val,
                                'other_value': other_val,
                                'percent_diff': au_percent_diff,
                                'threshold': asymmetry_thresholds['partial']['percent_diff'],
                                'type': 'Partial',
                                'normalized': use_normalized
                            })

        # Two-tier detection as a potential override - check if primary AU has extreme asymmetry
        if primary_au_extreme and other_au_moderate and info['paralysis']['zones'][side][zone] == 'None':
            # Set as Complete
            info['paralysis']['zones'][side][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis[side][zone] != 'Complete':
                zone_paralysis[side][zone] = 'Complete'

            # Add to affected AUs
            for au in aus:
                au_base = au.split('_')[0] + '_r'
                if au_base == 'AU12_r':
                    affected_aus_by_zone_side[side][zone].add(au)
                    if au not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append(au)

        return True