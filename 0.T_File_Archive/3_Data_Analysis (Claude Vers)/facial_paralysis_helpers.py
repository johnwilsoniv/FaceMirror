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

    for au, value in values.items():
        au_base = au.split('_')[0] + '_r'  # Get base AU name (e.g., AU01_r)

        # Check if we should use normalized value for this AU
        should_use_normalized = False
        if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:
            should_use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)

        # Get the appropriate value (normalized or raw)
        actual_value = value
        if should_use_normalized and values_normalized and au in values_normalized:
            actual_value = values_normalized[au]

        if au_base in weights:
            # Get the weight and importance factor for this AU
            weight = weights[au_base]
            importance_factor = importance.get(au_base, 1.0)

            # Apply both weight and importance
            actual_weight = weight * importance_factor

            weighted_sum += actual_value * actual_weight
            total_weight += actual_weight

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

    # Initialize logging context
    log_context = f"{side} {zone}"
    logger.debug(f"Calculating confidence score for {log_context}")

    # Define confidence factors
    confidence = 0.0

    # Factor 1: Proportion of AUs that show abnormality
    abnormal_aus = 0
    total_aus = len(values)

    logger.debug(f"{log_context} - Total AUs: {total_aus}")
    logger.debug(f"{log_context} - Criteria met: {criteria_met}")

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

        # Increase weight for AU45_r in midface
        if zone == 'mid' and au_base == 'AU45_r':
            au_abnormality_weights[au] = 1.5  # 50% more weight for AU45_r

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

        # Check for extreme asymmetry
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

    # NEW CODE: Boost confidence when AU25-specific criteria are met
    if criteria_met.get('au25_partial', False) or criteria_met.get('au25_complete', False):
        criteria_factor += 0.15  # Boost confidence when specific AU25 criteria are met
        criteria_factor = min(1.0, criteria_factor)  # Make sure it's still capped at 1.0

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

                # Give extra weight to AU12_r for lower face
                if zone == 'lower' and au_base == 'AU12_r':
                    weight *= 1.5

                # Give extra weight to AU45_r for midface
                if zone == 'mid' and au_base == 'AU45_r':
                    weight *= 1.5

                percent_diffs.append(pd * weight)

        if percent_diffs:
            avg_pd = sum(percent_diffs) / len(percent_diffs)
            threshold = self.asymmetry_thresholds[zone]['partial']['percent_diff']
            # Higher confidence for higher percent differences with lower cap
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

    logger.debug(f"{log_context} - Final confidence score: {confidence:.3f}")
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


def calculate_simplified_midface_score(au45_value, au45_ratio):
    """
    Calculate a simplified score for midface paralysis using only AU45_r.

    Args:
        au45_value (float): The AU45_r value of the side being evaluated
        au45_ratio (float): The ratio between sides for AU45_r

    Returns:
        float: Combined score (lower values indicate worse function)
    """
    # Calculate functional component for AU45_r (eyelid closure)
    if au45_value < 0.8:
        au45_functional = au45_value / 3.0  # Scale linearly for very low values
    else:
        # Higher values get proportionally higher scores (more normal)
        au45_functional = 0.267 + (au45_value - 0.8) * 0.5
        au45_functional = min(1.0, au45_functional)  # Cap at 1.0

    # Calculate asymmetry component with increased penalties for asymmetry
    au45_asymmetry = au45_ratio ** 0.7  # Exponential penalty for asymmetry

    # Combined score - weighted toward functional aspect
    combined_score = (0.7 * au45_functional) + (0.3 * au45_asymmetry)

    # Additional penalty for very low AU45 values (severely reduced function)
    if au45_value < 0.5:
        combined_score *= 0.8  # 20% penalty for very low AU45

    # Additional penalty for extreme asymmetry
    if au45_ratio < 0.2:
        combined_score *= 0.7  # 30% penalty for extreme asymmetry

    return combined_score


def calculate_lower_face_combined_score(au12_ratio, au25_ratio, au12_min_value, au25_min_value):
    """
    Calculate a combined score for lower face paralysis detection.
    Combines information from both AU12_r (Lip Corner Puller/Smile) and AU25_r (Lips Part).

    Args:
        au12_ratio (float): Ratio between minimum and maximum sides for AU12_r
        au25_ratio (float): Ratio between minimum and maximum sides for AU25_r
        au12_min_value (float): Minimum value between sides for AU12_r
        au25_min_value (float): Minimum value between sides for AU25_r

    Returns:
        float: Combined score (lower scores indicate more severe paralysis)
    """
    # Weight AU12_r and AU25_r - modified to give more importance to AU25_r
    au12_weight = 0.6  # Changed from 0.7
    au25_weight = 0.4  # Changed from 0.3

    # Combine ratios - if a ratio is None or 0, use 0.2 as a fallback (indicating significant asymmetry)
    au12_ratio = au12_ratio if au12_ratio and au12_ratio > 0 else 0.2
    au25_ratio = au25_ratio if au25_ratio and au25_ratio > 0 else 0.2

    # Combine ratios and minimum values
    combined_ratio = (au12_ratio * au12_weight) + (au25_ratio * au25_weight)

    # Ensure min values aren't None
    au12_min_value = au12_min_value if au12_min_value is not None else 0
    au25_min_value = au25_min_value if au25_min_value is not None else 0

    combined_min = (au12_min_value * au12_weight) + (au25_min_value * au25_weight)

    # Calculate the combined score
    # Using a lower factor than midface since lower face AUs typically have higher values
    return combined_ratio * (1 + combined_min / 5)


def calculate_functional_component(au45_value):
    """
    Calculate the functional component for midface paralysis detection.

    Args:
        au45_value (float): The AU45_r value (eyelid closure)

    Returns:
        float: Functional component score (0-1, lower values indicate worse function)
    """
    return min(1.0, au45_value / 3.0)


def calculate_asymmetry_component(ratio):
    """
    Calculate the asymmetry component for midface paralysis detection.

    Args:
        ratio (float): Ratio between the current side and other side AU45_r values

    Returns:
        float: Asymmetry component score (0-1, lower values indicate worse symmetry)
    """
    return min(1.0, ratio)


def get_cases_requiring_review(self, results):
    """
    Extract cases flagged for expert review due to borderline detection values.

    Args:
        results (dict): Detection results

    Returns:
        list: List of dictionaries containing borderline cases information
    """
    borderline_cases = []

    for patient_id, patient_data in results.items():
        for action, action_data in patient_data.items():
            if 'paralysis' in action_data and action_data['paralysis'].get('requires_review', False):
                # This is a borderline case - collect relevant info
                for side in ['left', 'right']:
                    if side in action_data['paralysis']['zones'] and action_data['paralysis']['zones'][side][
                        'mid'] != 'None':
                        case_info = {
                            'patient_id': patient_id,
                            'action': action,
                            'side': side,
                            'detection': action_data['paralysis']['zones'][side]['mid'],
                            'confidence': action_data['paralysis']['confidence'][side]['mid'],
                            'au_values': {
                                'AU45_r': {
                                    f"{side}": action_data[side]['normalized_au_values'].get('AU45_r', 0),
                                    f"{'right' if side == 'left' else 'left'}":
                                        action_data['right' if side == 'left' else 'left']['normalized_au_values'].get(
                                            'AU45_r', 0)
                                }
                            }
                        }
                        borderline_cases.append(case_info)

    return borderline_cases