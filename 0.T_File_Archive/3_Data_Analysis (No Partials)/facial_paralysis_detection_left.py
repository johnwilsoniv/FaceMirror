"""
Implementation of left side detection logic for facial paralysis.
"""

import logging
from facial_au_constants import (
    EXTREME_ASYMMETRY_THRESHOLD, MID_FACE_FUNCTIONAL_THRESHOLD, 
    MID_FACE_FUNCTION_RATIO_OVERRIDE, UPPER_FACE_AU_AGREEMENT_REQUIRED
)
from facial_paralysis_helpers import calculate_percent_difference

logger = logging.getLogger(__name__)

def detect_left_side_paralysis(self, info, zone, aus, left_values, right_values, 
                               left_values_normalized, right_values_normalized,
                               left_avg, right_avg, zone_paralysis, affected_aus_by_zone_side,
                               partial_thresholds, complete_thresholds, 
                               asymmetry_thresholds, confidence_thresholds):
    """
    Detect paralysis on the left side of the face.
    
    Args:
        self: The FacialParalysisDetector instance
        info (dict): Current action info dictionary to update
        zone (str): Facial zone being analyzed
        aus (list): Action units in this zone
        left_values (dict): Left side AU values
        right_values (dict): Right side AU values
        left_values_normalized (dict): Normalized left side AU values
        right_values_normalized (dict): Normalized right side AU values
        left_avg (float): Weighted average for left side
        right_avg (float): Weighted average for right side
        zone_paralysis (dict): Tracking paralysis by zone
        affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side
        partial_thresholds (dict): Thresholds for partial paralysis
        complete_thresholds (dict): Thresholds for complete paralysis
        asymmetry_thresholds (dict): Thresholds for asymmetry detection
        confidence_thresholds (dict): Thresholds for confidence scores
    """

    # First, check if extreme asymmetry exists and left is the stronger side
    has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, _ = self._check_for_extreme_asymmetry(
        left_values, right_values, left_values_normalized, right_values_normalized, zone
    )

    # If there is extreme asymmetry and the weaker side is NOT left (i.e., right side is weaker),
    # then left side is the stronger side and should be protected from paralysis detection
    if has_extreme_asymmetry and weaker_side != 'left' and extreme_percent_diff > 120:
        # Log the protection
        logger.debug(
            f"Protected left side from paralysis detection in {zone} zone - extreme asymmetry: {extreme_percent_diff:.1f}% (weaker side: {weaker_side})")
        return

    # Calculated percent difference between sides for each AU
    au_percent_diffs = []
    for au in aus:
        if au in left_values and au in right_values:
            # Determine if we should use normalized values for this AU
            use_normalized = False
            au_base = au.split('_')[0] + '_r'
            if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

            # Get the appropriate values
            left_val = left_values[au]
            right_val = right_values[au]

            if use_normalized and au in left_values_normalized and au in right_values_normalized:
                left_val = left_values_normalized[au]
                right_val = right_values_normalized[au]
                logger.debug(f"Using normalized values for {au} percent diff: L={left_val}, R={right_val}")

            percent_diff = calculate_percent_difference(left_val, right_val)
            au_percent_diffs.append(percent_diff)

    # Get maximum percent difference (most asymmetric AU)
    max_percent_diff = max(au_percent_diffs) if au_percent_diffs else 0

    # Calculate asymmetry ratio for complete paralysis detection
    ratio = 0
    if left_avg > 0 and right_avg > 0:
        ratio = min(left_avg, right_avg) / max(left_avg, right_avg)

    # Get specific AU normalized percent difference and ratio for key AUs in each zone
    # For lower face: AU12_r
    au12_percent_diff = None
    au12_ratio = None
    if zone == 'lower' and 'AU12_r' in left_values and 'AU12_r' in right_values:
        # Use normalized values if available
        if 'AU12_r' in left_values_normalized and 'AU12_r' in right_values_normalized:
            left_au12 = left_values_normalized['AU12_r']
            right_au12 = right_values_normalized['AU12_r']

            if left_au12 > 0 or right_au12 > 0:
                if left_au12 > 0 and right_au12 > 0:
                    au12_percent_diff = calculate_percent_difference(left_au12, right_au12)
                    au12_ratio = min(left_au12, right_au12) / max(left_au12, right_au12)
                else:
                    au12_percent_diff = 100  # One side has zero movement
                    au12_ratio = 0
                logger.debug(f"AU12_r normalized: percent_diff={au12_percent_diff:.1f}%, ratio={au12_ratio:.3f}")

    # For lower face: AU25_r
    au25_percent_diff = None
    au25_ratio = None
    if zone == 'lower' and 'AU25_r' in left_values and 'AU25_r' in right_values:
        # Use normalized values if available
        if 'AU25_r' in left_values_normalized and 'AU25_r' in right_values_normalized:
            left_au25 = left_values_normalized['AU25_r']
            right_au25 = right_values_normalized['AU25_r']

            if left_au25 > 0 or right_au25 > 0:
                if left_au25 > 0 and right_au25 > 0:
                    au25_percent_diff = calculate_percent_difference(left_au25, right_au25)
                    au25_ratio = min(left_au25, right_au25) / max(left_au25, right_au25)
                else:
                    au25_percent_diff = 100  # One side has zero movement
                    au25_ratio = 0
                logger.debug(f"AU25_r normalized: percent_diff={au25_percent_diff:.1f}%, ratio={au25_ratio:.3f}")

    # For upper face: AU01_r
    au01_percent_diff = None
    au01_ratio = None
    if zone == 'upper' and 'AU01_r' in left_values and 'AU01_r' in right_values:
        # Use normalized values if available
        if 'AU01_r' in left_values_normalized and 'AU01_r' in right_values_normalized:
            left_au01 = left_values_normalized['AU01_r']
            right_au01 = right_values_normalized['AU01_r']

            if left_au01 > 0 or right_au01 > 0:
                if left_au01 > 0 and right_au01 > 0:
                    au01_percent_diff = calculate_percent_difference(left_au01, right_au01)
                    au01_ratio = min(left_au01, right_au01) / max(left_au01, right_au01)
                else:
                    au01_percent_diff = 100  # One side has zero movement
                    au01_ratio = 0
                logger.debug(f"AU01_r normalized: percent_diff={au01_percent_diff:.1f}%, ratio={au01_ratio:.3f}")

    # For mid face: AU45_r
    au45_percent_diff = None
    au45_ratio = None
    if zone == 'mid' and 'AU45_r' in left_values and 'AU45_r' in right_values:
        # Use normalized values if available
        if 'AU45_r' in left_values_normalized and 'AU45_r' in right_values_normalized:
            left_au45 = left_values_normalized['AU45_r']
            right_au45 = right_values_normalized['AU45_r']

            if left_au45 > 0 or right_au45 > 0:
                if left_au45 > 0 and right_au45 > 0:
                    au45_percent_diff = calculate_percent_difference(left_au45, right_au45)
                    au45_ratio = min(left_au45, right_au45) / max(left_au45, right_au45)
                else:
                    au45_percent_diff = 100  # One side has zero movement
                    au45_ratio = 0
                logger.debug(f"AU45_r normalized: percent_diff={au45_percent_diff:.1f}%, ratio={au45_ratio:.3f}")

    # Check for extreme asymmetry in any individual AU with normalized values if available
    has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, confidence_boost = self._check_for_extreme_asymmetry(
        left_values, right_values, left_values_normalized, right_values_normalized, zone
    )

    # Check each criterion separately and track which ones are met
    criteria_met = {
        'minimal_movement': left_avg < complete_thresholds['minimal_movement'],
        'ratio': (left_avg > 0 and right_avg > 0 and
                ratio < asymmetry_thresholds['complete']['ratio'] and
                left_avg < right_avg),
        'percent_diff': (max_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and
                        left_avg < right_avg),
        'extreme_asymmetry': (has_extreme_asymmetry and weaker_side == 'left')
    }

    # Special case for AU12_r normalized values in lower face
    if zone == 'lower' and au12_ratio is not None and au12_percent_diff is not None:
        # If AU12_r normalized ratio is below threshold and left is weaker
        if au12_ratio < asymmetry_thresholds['complete']['ratio'] and left_values_normalized['AU12_r'] < \
                right_values_normalized['AU12_r']:
            criteria_met['au12_normalized_ratio'] = True
            logger.debug(
                f"LEFT: AU12_r normalized ratio is below threshold: {au12_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")
        
        # If AU12_r percent difference exceeds partial threshold and left is weaker
        if au12_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and left_values_normalized['AU12_r'] < \
                right_values_normalized['AU12_r']:
            criteria_met['au12_percent_diff'] = True
            logger.debug(
                f"LEFT: AU12_r normalized percent diff exceeds threshold: {au12_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")

    # Special case for AU25_r normalized values in lower face
    if zone == 'lower' and au25_ratio is not None and au25_percent_diff is not None:
        # If AU25_r normalized ratio is below threshold and left is weaker
        if au25_ratio < asymmetry_thresholds['complete']['ratio'] and left_values_normalized['AU25_r'] < \
                right_values_normalized['AU25_r']:
            criteria_met['au25_normalized_ratio'] = True
            logger.debug(
                f"LEFT: AU25_r normalized ratio is below threshold: {au25_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")
        
        # If AU25_r percent difference exceeds partial threshold and left is weaker
        if au25_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and left_values_normalized['AU25_r'] < \
                right_values_normalized['AU25_r']:
            criteria_met['au25_percent_diff'] = True
            logger.debug(
                f"LEFT: AU25_r normalized percent diff exceeds threshold: {au25_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")

    # Special case for AU01_r normalized values in upper face
    if zone == 'upper' and au01_ratio is not None and au01_percent_diff is not None:
        # If AU01_r normalized ratio is below threshold and left is weaker
        if au01_ratio < asymmetry_thresholds['complete']['ratio'] and left_values_normalized['AU01_r'] < \
                right_values_normalized['AU01_r']:
            criteria_met['au01_normalized_ratio'] = True
            logger.debug(
                f"LEFT: AU01_r normalized ratio is below threshold: {au01_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")

    # Special case for AU45_r normalized values in mid face
    if zone == 'mid' and au45_ratio is not None and au45_percent_diff is not None:
        # If AU45_r normalized ratio is below threshold and left is weaker
        if au45_ratio < asymmetry_thresholds['complete']['ratio'] and left_values_normalized['AU45_r'] < \
                right_values_normalized['AU45_r']:
            criteria_met['au45_normalized_ratio'] = True
            logger.debug(
                f"LEFT: AU45_r normalized ratio is below threshold: {au45_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")

    # Calculate confidence score for these criteria
    confidence_score = self._calculate_confidence_score(
        'left', zone, left_values, right_values, criteria_met
    )

    # Apply any confidence boost from extreme asymmetry detection
    confidence_score += confidence_boost

    # Special logic for lower face
    if zone == 'lower':
        # If either AU12_r or AU25_r shows strong asymmetry, boost confidence
        individual_au_criteria_met = False
        
        # Check AU12_r for clear asymmetry
        if au12_ratio is not None and au12_ratio < asymmetry_thresholds['complete']['ratio'] and \
                left_values_normalized['AU12_r'] < right_values_normalized['AU12_r']:
            individual_au_criteria_met = True
            logger.debug(
                f"LEFT: Individual AU12_r shows strong asymmetry, ratio: {au12_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")
            confidence_score += 0.1  # Boost confidence
            
            # Track AU values for individual AU detection
            info['paralysis']['contributing_aus']['left'][zone]['individual_au'].append({
                'au': 'AU12_r',
                'left_value': left_values_normalized['AU12_r'],
                'right_value': right_values_normalized['AU12_r'],
                'ratio': au12_ratio,
                'threshold': asymmetry_thresholds['complete']['ratio'],
                'type': 'Complete',
                'normalized': True
            })
            
        # Check AU25_r for clear asymmetry
        if au25_ratio is not None and au25_ratio < asymmetry_thresholds['complete']['ratio'] and \
                left_values_normalized['AU25_r'] < right_values_normalized['AU25_r']:
            individual_au_criteria_met = True
            logger.debug(
                f"LEFT: Individual AU25_r shows strong asymmetry, ratio: {au25_ratio:.3f} < {asymmetry_thresholds['complete']['ratio']}")
            confidence_score += 0.1  # Boost confidence
            
            # Track AU values for individual AU detection
            info['paralysis']['contributing_aus']['left'][zone]['individual_au'].append({
                'au': 'AU25_r',
                'left_value': left_values_normalized['AU25_r'],
                'right_value': right_values_normalized['AU25_r'],
                'ratio': au25_ratio,
                'threshold': asymmetry_thresholds['complete']['ratio'],
                'type': 'Complete',
                'normalized': True
            })
        
        # Check for partial asymmetry based on percent difference
        if au12_percent_diff is not None and au12_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and \
                left_values_normalized['AU12_r'] < right_values_normalized['AU12_r']:
            if not individual_au_criteria_met:  # Only count if not already counted for complete
                individual_au_criteria_met = True
                logger.debug(
                    f"LEFT: Individual AU12_r shows moderate asymmetry, percent diff: {au12_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")
                confidence_score += 0.05  # Small boost for partial
                
                # Track AU values for individual AU detection
                info['paralysis']['contributing_aus']['left'][zone]['individual_au'].append({
                    'au': 'AU12_r',
                    'left_value': left_values_normalized['AU12_r'],
                    'right_value': right_values_normalized['AU12_r'],
                    'percent_diff': au12_percent_diff,
                    'threshold': asymmetry_thresholds['partial']['percent_diff'],
                    'type': 'Partial',
                    'normalized': True
                })
        
        if au25_percent_diff is not None and au25_percent_diff > asymmetry_thresholds['partial']['percent_diff'] and \
                left_values_normalized['AU25_r'] < right_values_normalized['AU25_r']:
            if not individual_au_criteria_met:  # Only count if not already counted for complete
                individual_au_criteria_met = True
                logger.debug(
                    f"LEFT: Individual AU25_r shows moderate asymmetry, percent diff: {au25_percent_diff:.1f}% > {asymmetry_thresholds['partial']['percent_diff']}")
                confidence_score += 0.05  # Small boost for partial
                
                # Track AU values for individual AU detection
                info['paralysis']['contributing_aus']['left'][zone]['individual_au'].append({
                    'au': 'AU25_r',
                    'left_value': left_values_normalized['AU25_r'],
                    'right_value': right_values_normalized['AU25_r'],
                    'percent_diff': au25_percent_diff,
                    'threshold': asymmetry_thresholds['partial']['percent_diff'],
                    'type': 'Partial',
                    'normalized': True
                })
        
        # Add to criteria met
        criteria_met['individual_au'] = individual_au_criteria_met

    # Check for borderline cases
    is_borderline, adjusted_confidence = self._check_for_borderline_cases(
        zone, 'left', left_values, right_values, ratio, max_percent_diff, confidence_score
    )

    # Update confidence score if borderline case
    if is_borderline:
        confidence_score = adjusted_confidence
        criteria_met['borderline_case'] = True

    # Store confidence score
    info['paralysis']['confidence']['left'][zone] = confidence_score

    # Two-tier asymmetry detection for lower face
    if zone == 'lower':
        # Check if primary AU (AU12_r) has extreme asymmetry and any asymmetry in other AUs
        primary_au_extreme = False
        other_au_moderate = False

        for au in aus:
            au_base = au.split('_')[0] + '_r'
            if au in left_values and au in right_values:
                # Get the appropriate values (normalized if available)
                left_val = left_values[au]
                right_val = right_values[au]

                use_normalized = False
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                if use_normalized and au in left_values_normalized and au in right_values_normalized:
                    left_val = left_values_normalized[au]
                    right_val = right_values_normalized[au]

                percent_diff = calculate_percent_difference(left_val, right_val)

                # Reduced threshold for AU12_r to catch more cases
                au12_threshold = 120  # Reduced from 150

                # Check primary AU (AU12_r)
                if au_base == 'AU12_r' and percent_diff > au12_threshold and left_val < right_val:
                    primary_au_extreme = True
                    logger.debug(f"Left primary AU12_r has extreme asymmetry: {percent_diff:.1f}% > {au12_threshold}%")
                # Check other AUs for any asymmetry
                elif au_base != 'AU12_r' and percent_diff > 25 and left_val < right_val:
                    other_au_moderate = True

        # If primary AU has extreme asymmetry and at least one other AU has moderate asymmetry
        if primary_au_extreme and other_au_moderate and info['paralysis']['zones']['left'][zone] == 'None':
            # Set as Complete
            info['paralysis']['zones']['left'][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis['left'][zone] != 'Complete':
                zone_paralysis['left'][zone] = 'Complete'

            # Add to affected AUs
            for au in aus:
                au_base = au.split('_')[0] + '_r'
                if au_base == 'AU12_r':
                    affected_aus_by_zone_side['left'][zone].add(au)
                    if au not in info['paralysis']['affected_aus']['left']:
                        info['paralysis']['affected_aus']['left'].append(au)

    # Two-tier asymmetry detection for upper face
    if zone == 'upper':
        # Check if primary AU (AU01_r) has extreme asymmetry and any asymmetry in other AUs
        primary_au_extreme = False
        other_au_moderate = False

        for au in aus:
            au_base = au.split('_')[0] + '_r'
            if au in left_values and au in right_values:
                # Get the appropriate values (normalized if available)
                left_val = left_values[au]
                right_val = right_values[au]

                use_normalized = False
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                if use_normalized and au in left_values_normalized and au in right_values_normalized:
                    left_val = left_values_normalized[au]
                    right_val = right_values_normalized[au]

                percent_diff = calculate_percent_difference(left_val, right_val)

                # Set threshold for AU01_r
                au01_threshold = 120

                # Check primary AU (AU01_r)
                if au_base == 'AU01_r' and percent_diff > au01_threshold and left_val < right_val:
                    primary_au_extreme = True
                    logger.debug(f"Left primary AU01_r has extreme asymmetry: {percent_diff:.1f}% > {au01_threshold}%")
                # Check other AUs for any asymmetry
                elif au_base != 'AU01_r' and percent_diff > 25 and left_val < right_val:
                    other_au_moderate = True

        # If primary AU has extreme asymmetry and at least one other AU has moderate asymmetry
        if primary_au_extreme and (other_au_moderate or zone == 'upper') and \
                info['paralysis']['zones']['left'][zone] == 'None':
            # Set as Complete - for upper face, we don't necessarily need other AU moderate
            info['paralysis']['zones']['left'][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis['left'][zone] != 'Complete':
                zone_paralysis['left'][zone] = 'Complete'

            # Add to affected AUs
            for au in aus:
                au_base = au.split('_')[0] + '_r'
                if au_base == 'AU01_r':
                    affected_aus_by_zone_side['left'][zone].add(au)
                    if au not in info['paralysis']['affected_aus']['left']:
                        info['paralysis']['affected_aus']['left'].append(au)

    # Combined score approach for midface zone
    if zone == 'mid' and self.use_combined_score_for_midface:
        # Get the appropriate values
        if 'AU45_r' in left_values and 'AU45_r' in right_values:
            # Calculate minimum value and ratio
            left_val = left_values['AU45_r']
            right_val = right_values['AU45_r']
            
            if left_values_normalized and right_values_normalized and 'AU45_r' in left_values_normalized and 'AU45_r' in right_values_normalized:
                left_val = left_values_normalized['AU45_r']
                right_val = right_values_normalized['AU45_r']
                
            min_val = left_val  # Since we're checking left side paralysis
            max_val = right_val
            ratio = min_val / max_val if max_val > 0 else 0
            
            # Calculate combined score
            combined_score = self._calculate_midface_combined_score(ratio, min_val)
            logger.debug(f"Left midface combined score: {combined_score:.3f} (min_val={min_val:.2f}, ratio={ratio:.3f})")
            
            # Determine classification based on combined score
            if combined_score < self.midface_combined_score_thresholds['complete'] and confidence_score >= confidence_thresholds['complete']:
                # Set as Complete paralysis
                info['paralysis']['zones']['left'][zone] = 'Complete'
                
                # Track for patient-level assessment
                if zone_paralysis['left'][zone] != 'Complete':
                    zone_paralysis['left'][zone] = 'Complete'
                    
                # Add to affected AUs
                affected_aus_by_zone_side['left'][zone].add('AU45_r')
                if 'AU45_r' not in info['paralysis']['affected_aus']['left']:
                    info['paralysis']['affected_aus']['left'].append('AU45_r')
                    
                # Track combined score calculation
                info['paralysis']['contributing_aus']['left'][zone]['combined_score'].append({
                    'au': 'AU45_r',
                    'left_value': left_val,
                    'right_value': right_val,
                    'ratio': ratio,
                    'min_value': min_val,
                    'combined_score': combined_score,
                    'threshold': self.midface_combined_score_thresholds['complete'],
                    'type': 'Complete'
                })
            elif combined_score < self.midface_combined_score_thresholds['partial'] and confidence_score >= confidence_thresholds['partial']:
                # Set as Partial paralysis if not already marked as Complete
                if info['paralysis']['zones']['left'][zone] == 'None':
                    info['paralysis']['zones']['left'][zone] = 'Partial'
                    
                # Track for patient-level assessment if not already Complete
                if zone_paralysis['left'][zone] == 'None':
                    zone_paralysis['left'][zone] = 'Partial'
                    
                # Add to affected AUs
                affected_aus_by_zone_side['left'][zone].add('AU45_r')
                if 'AU45_r' not in info['paralysis']['affected_aus']['left']:
                    info['paralysis']['affected_aus']['left'].append('AU45_r')
                    
                # Track combined score calculation
                info['paralysis']['contributing_aus']['left'][zone]['combined_score'].append({
                    'au': 'AU45_r',
                    'left_value': left_val,
                    'right_value': right_val,
                    'ratio': ratio,
                    'min_value': min_val,
                    'combined_score': combined_score,
                    'threshold': self.midface_combined_score_thresholds['partial'],
                    'type': 'Partial'
                })
    
    # If NOT using combined score for midface, proceed with original approach
    elif zone == 'mid' and not self.use_combined_score_for_midface:
        process_midface_original_approach(
            self, info, zone, aus, left_values, right_values, 
            left_values_normalized, right_values_normalized,
            confidence_score, zone_paralysis, affected_aus_by_zone_side,
            criteria_met, extreme_au, extreme_percent_diff, 
            au45_ratio, asymmetry_thresholds, confidence_thresholds, ratio
        )
    
    # For zones other than midface, apply the detection logic
    elif zone != 'mid':
        process_nonmidface_zones(
            self, info, zone, aus, left_values, right_values, 
            left_values_normalized, right_values_normalized,
            confidence_score, zone_paralysis, affected_aus_by_zone_side,
            criteria_met, extreme_au, extreme_percent_diff, 
            au12_ratio, au25_ratio, au01_ratio, asymmetry_thresholds, 
            confidence_thresholds, ratio, partial_thresholds, complete_thresholds
        )

def process_midface_original_approach(self, info, zone, aus, left_values, right_values, 
                                     left_values_normalized, right_values_normalized,
                                     confidence_score, zone_paralysis, affected_aus_by_zone_side,
                                     criteria_met, extreme_au, extreme_percent_diff, 
                                     au45_ratio, asymmetry_thresholds, confidence_thresholds, ratio):
    """Process midface zone using the original approach (not combined score)."""
    # Special case for extreme asymmetry in any zone
    if criteria_met['extreme_asymmetry']:
        # If zone has extreme asymmetry, set as Complete
        info['paralysis']['zones']['left'][zone] = 'Complete'

        # Track for patient-level assessment
        if zone_paralysis['left'][zone] != 'Complete':
            zone_paralysis['left'][zone] = 'Complete'

        # Track contributing AUs and add to affected_aus
        affected_aus_by_zone_side['left'][zone].add(extreme_au)
        if extreme_au not in info['paralysis']['affected_aus']['left']:
            info['paralysis']['affected_aus']['left'].append(extreme_au)

        # Track AU values and thresholds for extreme asymmetry
        info['paralysis']['contributing_aus']['left'][zone]['extreme_asymmetry'].append({
            'au': extreme_au,
            'left_value': left_values[extreme_au],
            'right_value': right_values[extreme_au],
            'percent_diff': extreme_percent_diff,
            'threshold': EXTREME_ASYMMETRY_THRESHOLD,
            'type': 'Complete'
        })
    # Special case for AU45_r normalized ratio (mid face)
    elif criteria_met.get('au45_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
        # Mid Face functional check - only mark as complete paralysis if below functional threshold
        if 'AU45_r' in left_values and left_values['AU45_r'] < MID_FACE_FUNCTIONAL_THRESHOLD:
            # AU45_r normalized ratio indicates complete paralysis
            info['paralysis']['zones']['left'][zone] = 'Complete'

            # Track for patient-level assessment
            if zone_paralysis['left'][zone] != 'Complete':
                zone_paralysis['left'][zone] = 'Complete'

            # Track contributing AUs
            affected_aus_by_zone_side['left'][zone].add('AU45_r')
            if 'AU45_r' not in info['paralysis']['affected_aus']['left']:
                info['paralysis']['affected_aus']['left'].append('AU45_r')

            # Track values for this detection
            info['paralysis']['contributing_aus']['left'][zone]['normalized_ratio'].append({
                'au': 'AU45_r',
                'left_value': left_values_normalized['AU45_r'],
                'right_value': right_values_normalized['AU45_r'],
                'ratio': au45_ratio,
                'threshold': asymmetry_thresholds['complete']['ratio'],
                'type': 'Complete',
                'normalized': True
            })
        else:
            # Don't mark as complete if it has functional eyelid closure
            if 'AU45_r' in left_values and left_values['AU45_r'] >= MID_FACE_FUNCTIONAL_THRESHOLD:
                logger.debug(
                    f"Skipping mid face detection due to functional closure: left AU45_r = {left_values['AU45_r']:.3f} >= {MID_FACE_FUNCTIONAL_THRESHOLD}")

                # If it has functional closure but still shows asymmetry, might be partial
                if ratio < MID_FACE_FUNCTION_RATIO_OVERRIDE:
                    info['paralysis']['zones']['left'][zone] = 'Partial'
                    if zone_paralysis['left'][zone] == 'None':
                        zone_paralysis['left'][zone] = 'Partial'
    # Special case for borderline cases
    elif criteria_met.get('borderline_case', False) and confidence_score >= confidence_thresholds['complete']:
        # For mid face, check functional threshold
        if 'AU45_r' in left_values and left_values['AU45_r'] < MID_FACE_FUNCTIONAL_THRESHOLD:
            info['paralysis']['zones']['left'][zone] = 'Complete'
            if zone_paralysis['left'][zone] != 'Complete':
                zone_paralysis['left'][zone] = 'Complete'
        else:
            logger.debug(
                f"Skipping mid face borderline case due to functional closure: left AU45_r = {left_values.get('AU45_r', 'N/A')}")

        # Track contributing AUs for borderline case (if we proceed with detection)
        if info['paralysis']['zones']['left'][zone] == 'Complete':
            for au in aus:
                if au in left_values and au in right_values:
                    percent_diff = calculate_percent_difference(left_values[au], right_values[au])
                    if percent_diff > 70.0 and left_values[au] < right_values[au]:
                        affected_aus_by_zone_side['left'][zone].add(au)
                        if au not in info['paralysis']['affected_aus']['left']:
                            info['paralysis']['affected_aus']['left'].append(au)

                        # Track AU values for borderline case
                        info['paralysis']['contributing_aus']['left'][zone]['borderline_case'].append({
                            'au': au,
                            'left_value': left_values[au],
                            'right_value': right_values[au],
                            'percent_diff': percent_diff,
                            'ratio': left_values[au] / right_values[au] if right_values[au] > 0 else 0,
                            'type': 'Complete'
                        })
    # Determine whether to detect complete, partial, or no paralysis
    elif ((criteria_met['minimal_movement'] or criteria_met['ratio']) and
          confidence_score >= confidence_thresholds['complete']):

        # Apply zone-specific checks before setting complete paralysis
        # For mid face, check if the eye can functionally close despite asymmetry
        if 'AU45_r' in left_values and left_values['AU45_r'] >= MID_FACE_FUNCTIONAL_THRESHOLD:
            # If it meets functional threshold, require more extreme asymmetry
            if ratio > MID_FACE_FUNCTION_RATIO_OVERRIDE:
                logger.debug(
                    f"Overriding mid face detection due to functional closure: {left_values['AU45_r']:.3f} >= {MID_FACE_FUNCTIONAL_THRESHOLD}")
                # Don't classify as Complete paralysis
                return  # Skip to next detection check

        # If we've passed all zone-specific checks, proceed with complete paralysis detection
        info['paralysis']['zones']['left'][zone] = 'Complete'

        # Track for patient-level assessment
        if zone_paralysis['left'][zone] != 'Complete':
            zone_paralysis['left'][zone] = 'Complete'

        # Track contributing AUs and add to affected_aus
        for au in aus:
            if au in left_values:
                # Check if we should use normalized value for this AU
                use_normalized = False
                au_base = au.split('_')[0] + '_r'
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                # Get the appropriate value
                left_val = left_values[au]
                if use_normalized and au in left_values_normalized:
                    left_val = left_values_normalized[au]

                # Check if AU is below threshold - handle minimal movement detection
                process_au_minimal_movement(
                    info, zone, au, left_val, self.paralysis_thresholds[zone]['complete']['minimal_movement'],
                    affected_aus_by_zone_side, 'left', use_normalized, 'Complete'
                )

                # Check if AU contributes to ratio-based detection
                if au in right_values:
                    # Get the appropriate right value
                    right_val = right_values[au]
                    if use_normalized and au in right_values_normalized:
                        right_val = right_values_normalized[au]

                    # Handle asymmetry detection
                    process_au_asymmetry(
                        info, zone, au, left_val, right_val, 
                        asymmetry_thresholds['complete']['ratio'],
                        affected_aus_by_zone_side, 'left', use_normalized, 'Complete'
                    )

    # Check for partial paralysis
    elif ((criteria_met.get('minimal_movement', False) and confidence_score >= confidence_thresholds['partial']) or
          (criteria_met.get('percent_diff', False) and confidence_score >= confidence_thresholds['partial'])):
        # Partial paralysis detected on left side
        # Only assign if not already marked as complete
        if info['paralysis']['zones']['left'][zone] == 'None':
            info['paralysis']['zones']['left'][zone] = 'Partial'

        # Track for patient-level assessment - only update if not already complete
        if zone_paralysis['left'][zone] == 'None':
            zone_paralysis['left'][zone] = 'Partial'

        # Track contributing AUs and add to affected_aus
        for au in aus:
            if au in left_values:
                # Check if we should use normalized value for this AU
                use_normalized = False
                au_base = au.split('_')[0] + '_r'
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                # Get the appropriate values
                left_val = left_values[au]
                if use_normalized and au in left_values_normalized:
                    left_val = left_values_normalized[au]

                # Check if AU is below threshold but above complete threshold - partial paralysis
                if (left_val < self.paralysis_thresholds[zone]['partial']['minimal_movement'] and
                        left_val >= self.paralysis_thresholds[zone]['complete']['minimal_movement']):
                    affected_aus_by_zone_side['left'][zone].add(au)
                    if au not in info['paralysis']['affected_aus']['left']:
                        info['paralysis']['affected_aus']['left'].append(au)

                    # Track AU values and thresholds
                    info['paralysis']['contributing_aus']['left'][zone]['minimal_movement'].append({
                        'au': au,
                        'value': left_val,
                        'threshold': self.paralysis_thresholds[zone]['partial']['minimal_movement'],
                        'type': 'Partial',
                        'normalized': use_normalized
                    })

                # Check if AU contributes to percent difference detection
                elif au in right_values:
                    # Get the appropriate right value
                    right_val = right_values[au]
                    if use_normalized and au in right_values_normalized:
                        right_val = right_values_normalized[au]

                    # Process AU percent difference for partial paralysis
                    process_au_percent_diff(
                        info, zone, au, left_val, right_val, 
                        asymmetry_thresholds['partial']['percent_diff'],
                        affected_aus_by_zone_side, 'left', use_normalized
                    )

def process_nonmidface_zones(self, info, zone, aus, left_values, right_values, 
                            left_values_normalized, right_values_normalized,
                            confidence_score, zone_paralysis, affected_aus_by_zone_side,
                            criteria_met, extreme_au, extreme_percent_diff, 
                            au12_ratio, au25_ratio, au01_ratio, asymmetry_thresholds, 
                            confidence_thresholds, ratio, partial_thresholds, complete_thresholds):
    """Process upper and lower facial zones for left side detection."""
    # Special case for extreme asymmetry in any zone
    if criteria_met['extreme_asymmetry']:
        # If zone has extreme asymmetry, set as Complete
        info['paralysis']['zones']['left'][zone] = 'Complete'

        # Track for patient-level assessment
        if zone_paralysis['left'][zone] != 'Complete':
            zone_paralysis['left'][zone] = 'Complete'

        # Track contributing AUs and add to affected_aus
        affected_aus_by_zone_side['left'][zone].add(extreme_au)
        if extreme_au not in info['paralysis']['affected_aus']['left']:
            info['paralysis']['affected_aus']['left'].append(extreme_au)

        # Track AU values and thresholds for extreme asymmetry
        info['paralysis']['contributing_aus']['left'][zone]['extreme_asymmetry'].append({
            'au': extreme_au,
            'left_value': left_values[extreme_au],
            'right_value': right_values[extreme_au],
            'percent_diff': extreme_percent_diff,
            'threshold': EXTREME_ASYMMETRY_THRESHOLD,
            'type': 'Complete'
        })
    # Special case for individual AU detection in lower face
    elif zone == 'lower' and criteria_met.get('individual_au', False) and confidence_score >= confidence_thresholds['complete']:
        logger.debug(f"LEFT: Using individual AU detection for lower face, confidence: {confidence_score:.3f}")
        
        # Set as Complete
        info['paralysis']['zones']['left'][zone] = 'Complete'

        # Track for patient-level assessment
        if zone_paralysis['left'][zone] != 'Complete':
            zone_paralysis['left'][zone] = 'Complete'
            
    # Special case for AU12_r normalized ratio
    elif zone == 'lower' and criteria_met.get('au12_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
        # AU12_r normalized ratio indicates complete paralysis
        info['paralysis']['zones']['left'][zone] = 'Complete'

        # Track for patient-level assessment
        if zone_paralysis['left'][zone] != 'Complete':
            zone_paralysis['left'][zone] = 'Complete'

        # Track contributing AUs
        affected_aus_by_zone_side['left'][zone].add('AU12_r')
        if 'AU12_r' not in info['paralysis']['affected_aus']['left']:
            info['paralysis']['affected_aus']['left'].append('AU12_r')

        # Track values for this detection
        info['paralysis']['contributing_aus']['left'][zone]['normalized_ratio'].append({
            'au': 'AU12_r',
            'left_value': left_values_normalized['AU12_r'],
            'right_value': right_values_normalized['AU12_r'],
            'ratio': au12_ratio,
            'threshold': asymmetry_thresholds['complete']['ratio'],
            'type': 'Complete',
            'normalized': True
        })
    # Special case for AU25_r normalized ratio
    elif zone == 'lower' and criteria_met.get('au25_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
        # AU25_r normalized ratio indicates complete paralysis
        info['paralysis']['zones']['left'][zone] = 'Complete'

        # Track for patient-level assessment
        if zone_paralysis['left'][zone] != 'Complete':
            zone_paralysis['left'][zone] = 'Complete'

        # Track contributing AUs
        affected_aus_by_zone_side['left'][zone].add('AU25_r')
        if 'AU25_r' not in info['paralysis']['affected_aus']['left']:
            info['paralysis']['affected_aus']['left'].append('AU25_r')

        # Track values for this detection
        info['paralysis']['contributing_aus']['left'][zone]['normalized_ratio'].append({
            'au': 'AU25_r',
            'left_value': left_values_normalized['AU25_r'],
            'right_value': right_values_normalized['AU25_r'],
            'ratio': au25_ratio,
            'threshold': asymmetry_thresholds['complete']['ratio'],
            'type': 'Complete',
            'normalized': True
        })
    # Check for partial paralysis based on individual AU percent difference
    elif zone == 'lower' and (criteria_met.get('au12_percent_diff', False) or criteria_met.get('au25_percent_diff', False)) and confidence_score >= confidence_thresholds['partial']:
        # Only assign if not already marked as complete
        if info['paralysis']['zones']['left'][zone] == 'None':
            info['paralysis']['zones']['left'][zone] = 'Partial'

        # Track for patient-level assessment - only update if not already complete
        if zone_paralysis['left'][zone] == 'None':
            zone_paralysis['left'][zone] = 'Partial'
            
        # Track contributing AUs
        if criteria_met.get('au12_percent_diff', False):
            affected_aus_by_zone_side['left'][zone].add('AU12_r')
            if 'AU12_r' not in info['paralysis']['affected_aus']['left']:
                info['paralysis']['affected_aus']['left'].append('AU12_r')
        
        if criteria_met.get('au25_percent_diff', False):
            affected_aus_by_zone_side['left'][zone].add('AU25_r')
            if 'AU25_r' not in info['paralysis']['affected_aus']['left']:
                info['paralysis']['affected_aus']['left'].append('AU25_r')
                
    # Special case for AU01_r normalized ratio (upper face)
    elif zone == 'upper' and criteria_met.get('au01_normalized_ratio', False) and confidence_score >= confidence_thresholds['complete']:
        # Upper Face verification - check if the right side actually has more movement
        # Only mark as paralyzed if the left side actually has less movement
        if 'AU01_r' in left_values and 'AU01_r' in right_values:
            if left_values['AU01_r'] < right_values['AU01_r']:
                # AU01_r normalized ratio indicates complete paralysis when left side is actually weaker
                info['paralysis']['zones']['left'][zone] = 'Complete'

                # Track for patient-level assessment
                if zone_paralysis['left'][zone] != 'Complete':
                    zone_paralysis['left'][zone] = 'Complete'

                # Track contributing AUs
                affected_aus_by_zone_side['left'][zone].add('AU01_r')
                if 'AU01_r' not in info['paralysis']['affected_aus']['left']:
                    info['paralysis']['affected_aus']['left'].append('AU01_r')

                # Track values for this detection
                info['paralysis']['contributing_aus']['left'][zone]['normalized_ratio'].append({
                    'au': 'AU01_r',
                    'left_value': left_values_normalized['AU01_r'],
                    'right_value': right_values_normalized['AU01_r'],
                    'ratio': au01_ratio,
                    'threshold': asymmetry_thresholds['complete']['ratio'],
                    'type': 'Complete',
                    'normalized': True
                })
            else:
                logger.debug(f"Skipping upper face detection - left AU01_r isn't weaker ({left_values['AU01_r']:.3f} vs {right_values['AU01_r']:.3f})")
    # Special case for borderline cases
    elif criteria_met.get('borderline_case', False) and confidence_score >= confidence_thresholds['complete']:
        # Handle borderline cases with specific checks for upper face
        handle_borderline_cases(
            info, zone, aus, left_values, right_values, 
            zone_paralysis, affected_aus_by_zone_side, 'left'
        )
    # Determine whether to detect complete, partial, or no paralysis
    elif ((criteria_met['minimal_movement'] or criteria_met['ratio']) and
          confidence_score >= confidence_thresholds['complete']):

        # Apply zone-specific checks before setting complete paralysis
        if zone == 'upper':
            # For upper face, verify we're detecting the correct side
            if 'AU01_r' in left_values and 'AU01_r' in right_values and 'AU02_r' in left_values and 'AU02_r' in right_values:
                au01_left_weaker = left_values['AU01_r'] < right_values['AU01_r']
                au02_left_weaker = left_values['AU02_r'] < right_values['AU02_r']

                # At least AU01_r (more important) must show left being weaker
                if not au01_left_weaker:
                    logger.debug(f"Skipping upper face detection - left AU01_r isn't weaker ({left_values['AU01_r']:.3f} vs {right_values['AU01_r']:.3f})")
                    # Don't mark left as paralyzed when it shows more movement
                    return

                # If UPPER_FACE_AU_AGREEMENT_REQUIRED is true, also check AU02_r
                if UPPER_FACE_AU_AGREEMENT_REQUIRED and not au02_left_weaker:
                    logger.debug(f"Discrepancy in upper face AUs - AU01_r suggests left paralysis but AU02_r doesn't")
                    # Don't classify as Complete unless confidence is very high
                    if confidence_score < 0.6:  # Require higher confidence
                        return

        # If we've passed all zone-specific checks, proceed with complete paralysis detection
        info['paralysis']['zones']['left'][zone] = 'Complete'

        # Track for patient-level assessment
        if zone_paralysis['left'][zone] != 'Complete':
            zone_paralysis['left'][zone] = 'Complete'

        # Track contributing AUs and add to affected_aus
        for au in aus:
            if au in left_values:
                # Check if we should use normalized value for this AU
                use_normalized = False
                au_base = au.split('_')[0] + '_r'
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                # Get the appropriate value
                left_val = left_values[au]
                if use_normalized and au in left_values_normalized:
                    left_val = left_values_normalized[au]

                # Check if AU is below threshold
                if left_val < complete_thresholds['minimal_movement']:
                    affected_aus_by_zone_side['left'][zone].add(au)
                    if au not in info['paralysis']['affected_aus']['left']:
                        info['paralysis']['affected_aus']['left'].append(au)

                    # Track AU values and thresholds
                    info['paralysis']['contributing_aus']['left'][zone]['minimal_movement'].append({
                        'au': au,
                        'value': left_val,
                        'threshold': complete_thresholds['minimal_movement'],
                        'type': 'Complete',
                        'normalized': use_normalized
                    })

                # Check if AU contributes to ratio-based detection
                elif au in right_values:
                    # Get the appropriate right value
                    right_val = right_values[au]
                    if use_normalized and au in right_values_normalized:
                        right_val = right_values_normalized[au]

                    process_au_asymmetry(
                        info, zone, au, left_val, right_val, 
                        asymmetry_thresholds['complete']['ratio'],
                        affected_aus_by_zone_side, 'left', use_normalized, 'Complete'
                    )

    # Check for partial paralysis
    elif ((criteria_met.get('minimal_movement', False) or criteria_met.get('percent_diff', False)) and
          confidence_score >= confidence_thresholds['partial']):
        # Partial paralysis detected on left side
        # Only assign if not already marked as complete
        if info['paralysis']['zones']['left'][zone] == 'None':
            info['paralysis']['zones']['left'][zone] = 'Partial'

        # Track for patient-level assessment - only update if not already complete
        if zone_paralysis['left'][zone] == 'None':
            zone_paralysis['left'][zone] = 'Partial'

        # Track contributing AUs and add to affected_aus
        for au in aus:
            if au in left_values:
                # Check if we should use normalized value for this AU
                use_normalized = False
                au_base = au.split('_')[0] + '_r'
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

                # Get the appropriate values
                left_val = left_values[au]
                if use_normalized and au in left_values_normalized:
                    left_val = left_values_normalized[au]

                # Check if AU is below threshold but above complete threshold
                if (left_val < partial_thresholds['minimal_movement'] and
                        left_val >= complete_thresholds['minimal_movement']):
                    affected_aus_by_zone_side['left'][zone].add(au)
                    if au not in info['paralysis']['affected_aus']['left']:
                        info['paralysis']['affected_aus']['left'].append(au)

                    # Track AU values and thresholds
                    info['paralysis']['contributing_aus']['left'][zone]['minimal_movement'].append({
                        'au': au,
                        'value': left_val,
                        'threshold': partial_thresholds['minimal_movement'],
                        'type': 'Partial',
                        'normalized': use_normalized
                    })

                # Check if AU contributes to percent difference detection
                elif au in right_values:
                    # Get the appropriate right value
                    right_val = right_values[au]
                    if use_normalized and au in right_values_normalized:
                        right_val = right_values_normalized[au]

                    process_au_percent_diff(
                        info, zone, au, left_val, right_val, 
                        asymmetry_thresholds['partial']['percent_diff'],
                        affected_aus_by_zone_side, 'left', use_normalized
                    )

def handle_borderline_cases(info, zone, aus, values, other_side_values, zone_paralysis, affected_aus_by_zone_side, side):
    """Handle borderline cases with specific checks for different zones."""
    # For upper face, verify side direction
    if zone == 'upper':
        # Only set as complete if side actually shows less movement
        if 'AU01_r' in values and 'AU01_r' in other_side_values:
            if ((side == 'left' and values['AU01_r'] < other_side_values['AU01_r']) or 
                (side == 'right' and values['AU01_r'] < other_side_values['AU01_r'])):
                info['paralysis']['zones'][side][zone] = 'Complete'
                if zone_paralysis[side][zone] != 'Complete':
                    zone_paralysis[side][zone] = 'Complete'
            else:
                logger.debug(f"Skipping upper face borderline case - {side} AU01_r isn't weaker")
                return
    else:
        # Lower face and other zones proceed normally
        info['paralysis']['zones'][side][zone] = 'Complete'
        if zone_paralysis[side][zone] != 'Complete':
            zone_paralysis[side][zone] = 'Complete'

    # Track contributing AUs for borderline case (if we proceed with detection)
    if info['paralysis']['zones'][side][zone] == 'Complete':
        for au in aus:
            if au in values and au in other_side_values:
                percent_diff = calculate_percent_difference(values[au], other_side_values[au])
                if percent_diff > 70.0 and values[au] < other_side_values[au]:
                    affected_aus_by_zone_side[side][zone].add(au)
                    if au not in info['paralysis']['affected_aus'][side]:
                        info['paralysis']['affected_aus'][side].append(au)

                    # Track AU values for borderline case
                    info['paralysis']['contributing_aus'][side][zone]['borderline_case'].append({
                        'au': au,
                        'left_value': values[au] if side == 'left' else other_side_values[au],
                        'right_value': other_side_values[au] if side == 'left' else values[au],
                        'percent_diff': percent_diff,
                        'ratio': values[au] / other_side_values[au] if other_side_values[au] > 0 else 0,
                        'type': 'Complete'
                    })

def process_au_minimal_movement(info, zone, au, value, threshold, affected_aus_by_zone_side, side, use_normalized, type_str):
    """Process AU for minimal movement detection."""
    if value < threshold:
        affected_aus_by_zone_side[side][zone].add(au)
        if au not in info['paralysis']['affected_aus'][side]:
            info['paralysis']['affected_aus'][side].append(au)

        # Track AU values and thresholds
        info['paralysis']['contributing_aus'][side][zone]['minimal_movement'].append({
            'au': au,
            'value': value,
            'threshold': threshold,
            'type': type_str,
            'normalized': use_normalized
        })

def process_au_asymmetry(info, zone, au, value, other_value, ratio_threshold, 
                        affected_aus_by_zone_side, side, use_normalized, type_str):
    """Process AU for asymmetry detection."""
    if other_value > 0 and value > 0:
        au_ratio = min(value, other_value) / max(value, other_value)
        if au_ratio < ratio_threshold and value < other_value:
            affected_aus_by_zone_side[side][zone].add(au)
            if au not in info['paralysis']['affected_aus'][side]:
                info['paralysis']['affected_aus'][side].append(au)

            # Track AU values and thresholds
            info['paralysis']['contributing_aus'][side][zone]['asymmetry'].append({
                'au': au,
                'left_value': value if side == 'left' else other_value,
                'right_value': other_value if side == 'left' else value,
                'ratio': au_ratio,
                'threshold': ratio_threshold,
                'type': type_str,
                'normalized': use_normalized
            })

def process_au_percent_diff(info, zone, au, value, other_value, threshold, 
                           affected_aus_by_zone_side, side, use_normalized):
    """Process AU for percent difference detection."""
    au_percent_diff = calculate_percent_difference(value, other_value)
    if au_percent_diff > threshold and value < other_value:
        affected_aus_by_zone_side[side][zone].add(au)
        if au not in info['paralysis']['affected_aus'][side]:
            info['paralysis']['affected_aus'][side].append(au)

        # Track AU values and thresholds
        info['paralysis']['contributing_aus'][side][zone]['percent_diff'].append({
            'au': au,
            'left_value': value if side == 'left' else other_value,
            'right_value': other_value if side == 'left' else value,
            'percent_diff': au_percent_diff,
            'threshold': threshold,
            'type': 'Partial',
            'normalized': use_normalized
        })
