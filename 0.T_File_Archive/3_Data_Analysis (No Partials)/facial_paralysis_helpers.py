"""
Helper calculation methods for the facial paralysis detector.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_weighted_activation(self, side, zone, values, values_normalized=None):
    """
    Calculate weighted average activation for key AUs in a zone.
    Gives higher weight to more diagnostically important AUs.

    Args:
        self: The FacialParalysisDetector instance
        side (str): 'left' or 'right'
        zone (str): 'upper', 'mid', or 'lower'
        values (dict): Dictionary of AU values
        values_normalized (dict, optional): Dictionary of normalized AU values

    Returns:
        float: Weighted average activation
    """
    # Use the updated weights from constants instead of hardcoded values
    weights = self.facial_zone_weights[zone]
    importance = self.paralysis_detection_au_importance.get(zone, {})

    # Calculate weighted sum and total weights
    weighted_sum = 0
    total_weight = 0
    
    # DEBUG: List of AUs where normalized values were used
    normalized_aus = []

    for au, value in values.items():
        au_base = au.split('_')[0] + '_r'  # Get base AU name (e.g., AU01_r)
        
        # Check if we should use normalized value for this AU
        should_use_normalized = False
        if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
            should_use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)
        
        # Get the appropriate value (normalized or raw)
        actual_value = value
        if should_use_normalized and values_normalized and au in values_normalized:
            prev_value = actual_value
            actual_value = values_normalized[au]
            normalized_aus.append(au)
            logger.debug(f"Using normalized value for {au} in {zone} zone: {actual_value} (was {prev_value})")
        
        if au_base in weights:
            # Get the weight and importance factor for this AU
            weight = weights[au_base]
            importance_factor = importance.get(au_base, 1.0)
            
            # Apply both weight and importance
            actual_weight = weight * importance_factor
            
            weighted_sum += actual_value * actual_weight
            total_weight += actual_weight

    # Log debug info when normalized values are used
    if normalized_aus:
        logger.debug(f"Used normalized values for {normalized_aus} in {zone} zone, {side} side")
        
    # Return weighted average or 0 if no weights
    return weighted_sum / total_weight if total_weight > 0 else 0

def calculate_percent_difference(value1, value2):
    """
    Calculate percent difference between two values.

    Args:
        value1 (float): First value
        value2 (float): Second value

    Returns:
        float: Percent difference
    """
    if value1 == 0 and value2 == 0:
        return 0
    if value1 == 0 or value2 == 0:
        return 100  # Maximum difference

    avg = (value1 + value2) / 2
    return (abs(value1 - value2) / avg) * 100

def calculate_confidence_score(self, side, zone, values, other_side_values, criteria_met):
    """
    Calculate a confidence score for paralysis detection based on
    multiple factors including consistency across AUs and severity of asymmetry.

    Args:
        self: The FacialParalysisDetector instance
        side (str): 'left' or 'right'
        zone (str): 'upper', 'mid', or 'lower'
        values (dict): Dictionary of AU values for this side
        other_side_values (dict): Dictionary of AU values for the other side
        criteria_met (dict): Dictionary of detection criteria that were met

    Returns:
        float: Confidence score (0-1)
    """
    # Define confidence factors
    confidence = 0.0

    # Factor 1: Proportion of AUs that show abnormality
    abnormal_aus = 0
    total_aus = len(values)

    # For lower face, give special treatment to AU12_r
    au12r_abnormal = False
    au25r_abnormal = False
    au_abnormality_weights = {}

    # Set default weights for different AUs
    for au in values:
        au_base = au.split('_')[0] + '_r'
        au_abnormality_weights[au] = 1.0  # Default weight

        # Increase weight for AU12_r in lower face
        if zone == 'lower' and au_base == 'AU12_r':
            au_abnormality_weights[au] = 1.5  # 50% more weight for AU12_r

    for au, value in values.items():
        # Check if AU value is below minimal movement threshold for this zone
        if criteria_met.get('minimal_movement', False) and au in values and value < \
                self.paralysis_thresholds[zone]['partial']['minimal_movement']:
            abnormal_aus += au_abnormality_weights.get(au, 1.0)

            # Track if this is AU12_r or AU25_r for lower face
            au_base = au.split('_')[0] + '_r'
            if zone == 'lower':
                if au_base == 'AU12_r':
                    au12r_abnormal = True
                elif au_base == 'AU25_r':
                    au25r_abnormal = True

        # Check if AU shows asymmetry
        if au in other_side_values and other_side_values[au] > 0 and value > 0:
            percent_diff = calculate_percent_difference(value, other_side_values[au])

            if criteria_met.get('percent_diff', False) and percent_diff > \
                    self.asymmetry_thresholds[zone]['partial']['percent_diff']:
                abnormal_aus += au_abnormality_weights.get(au, 1.0)

                # Track if this is AU12_r or AU25_r for lower face
                au_base = au.split('_')[0] + '_r'
                if zone == 'lower':
                    if au_base == 'AU12_r':
                        au12r_abnormal = True
                    elif au_base == 'AU25_r':
                        au25r_abnormal = True

            if criteria_met.get('ratio', False):
                ratio = min(value, other_side_values[au]) / max(value, other_side_values[au])
                if ratio < self.asymmetry_thresholds[zone]['complete']['ratio']:
                    abnormal_aus += au_abnormality_weights.get(au, 1.0)

                    # Track if this is AU12_r or AU25_r for lower face
                    au_base = au.split('_')[0] + '_r'
                    if zone == 'lower':
                        if au_base == 'AU12_r':
                            au12r_abnormal = True
                        elif au_base == 'AU25_r':
                            au25r_abnormal = True

        # NEW: Check for extreme asymmetry
        if criteria_met.get('extreme_asymmetry', False):
            abnormal_aus += 2.0  # Double weight for extreme asymmetry

    # Calculate total possible weighted abnormality score
    total_weight = sum(au_abnormality_weights.values())

    # Calculate proportion of abnormal AUs (capped at 1.0)
    au_proportion = min(1.0, abnormal_aus / (total_weight * 2)) if total_weight > 0 else 0

    # Special handling for lower face when only one of AU12_r or AU25_r is abnormal
    if zone == 'lower' and (au12r_abnormal or au25r_abnormal) and not (au12r_abnormal and au25r_abnormal):
        # If only one is abnormal, we still want a decent confidence
        # If it's AU12_r (more important for paralysis), give more confidence
        if au12r_abnormal:
            au_proportion = max(au_proportion, 0.7)  # At least 0.7 if AU12_r is abnormal
        else:  # Only AU25_r is abnormal
            au_proportion = max(au_proportion, 0.5)  # At least 0.5 if only AU25_r is abnormal

    # Factor 2: Number of criteria met
    criteria_count = sum(1 for met in criteria_met.values() if met)
    criteria_factor = min(1.0, criteria_count / 3)  # Cap at 1.0

    # Factor 3: Severity of the detection criteria
    severity_factor = 0.0

    if criteria_met.get('minimal_movement', False):
        avg_value = sum(values.values()) / len(values) if values else 0
        threshold = self.paralysis_thresholds[zone]['partial']['minimal_movement']
        # Higher confidence for values further below threshold
        if avg_value < threshold:
            severity_factor += 0.5 * (1 - (avg_value / threshold))

    if criteria_met.get('percent_diff', False):
        # Calculate average percent difference across all AUs
        percent_diffs = []

        for au, value in values.items():
            if au in other_side_values and other_side_values[au] > 0 and value > 0:
                pd = calculate_percent_difference(value, other_side_values[au])

                # Apply asymmetry weight if available for this AU in this zone
                au_base = au.split('_')[0] + '_r'
                weight = 1.0
                if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
                    weight = self.au_zone_detection_modifiers[zone][au_base].get('asymmetry_weight', 1.0)

                # NEW: Give extra weight to AU12_r for lower face
                if zone == 'lower' and au_base == 'AU12_r':
                    weight *= 1.5

                percent_diffs.append(pd * weight)

        if percent_diffs:
            avg_pd = sum(percent_diffs) / len(percent_diffs)
            threshold = self.asymmetry_thresholds[zone]['partial']['percent_diff']
            # FIXED: Higher confidence for higher percent differences with lower cap
            if avg_pd > threshold:
                severity_factor += 0.5 * min(1.0, (avg_pd - threshold) / 40)  # Lowered from 50 to 40

    # Special boost for individual AU detection
    if criteria_met.get('individual_au', False):
        severity_factor += 0.15

    # Factor 4: Consistency across AUs (standard deviation)
    consistency_factor = 0.0
    if len(values) > 1:
        values_list = list(values.values())
        std_dev = np.std(values_list)
        mean_val = np.mean(values_list)

        # Lower std dev relative to mean indicates higher consistency
        if mean_val > 0:
            cv = std_dev / mean_val  # Coefficient of variation
            # Higher consistency (lower CV) gives higher confidence
            consistency_factor = max(0, 1 - min(1.0, cv))

    # Lower face - if only one AU (AU12_r or AU25_r) shows clear asymmetry, don't penalize too much
    if zone == 'lower' and (au12r_abnormal or au25r_abnormal) and not (au12r_abnormal and au25r_abnormal):
        # Don't penalize as much for inconsistency if only one AU is abnormal
        consistency_factor = max(consistency_factor, 0.5)

    # Combine factors with weights
    confidence = (
            (0.4 * au_proportion) +
            (0.3 * criteria_factor) +
            (0.2 * severity_factor) +
            (0.1 * consistency_factor)
    )

    return min(1.0, confidence)  # Cap at 1.0

def calculate_midface_combined_score(ratio, minimum_value):
    """
    Calculate a combined score for midface paralysis detection.
    Combines both ratio and minimum function value.
    The formula accounts for both asymmetry and overall movement level.
    
    Args:
        ratio (float): Ratio between minimum and maximum sides
        minimum_value (float): Minimum value between sides
        
    Returns:
        float: Combined score (lower scores indicate more severe paralysis)
    """
    return ratio * (1 + minimum_value / 10)
