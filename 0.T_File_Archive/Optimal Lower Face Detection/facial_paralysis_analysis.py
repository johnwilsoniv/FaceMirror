{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;}
{\colortbl;\red255\green255\blue255;\red31\green29\blue21;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c16078\c14902\c10588;\cssrgb\c100000\c100000\c100000;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 """\
Analysis helper methods for facial paralysis detection.\
"""\
\
import logging\
from facial_au_constants import (\
    EXTREME_ASYMMETRY_THRESHOLD,\
    MID_FACE_FUNCTIONAL_THRESHOLD,\
    MID_FACE_FUNCTION_RATIO_OVERRIDE,\
    AU07_EXTREME_ASYMMETRY_THRESHOLD  # Added the new constant\
)\
from facial_paralysis_helpers import calculate_percent_difference\
\
logger = logging.getLogger(__name__)\
\
def check_for_extreme_asymmetry(self, left_values, right_values, left_values_normalized=None, right_values_normalized=None, zone=None):\
    """\
    Check if any individual AU shows extreme asymmetry regardless of others.\
    \
    Args:\
        self: The FacialParalysisDetector instance\
        left_values (dict): Dictionary of left side AU values\
        right_values (dict): Dictionary of right side AU values\
        left_values_normalized (dict, optional): Dictionary of normalized left side AU values\
        right_values_normalized (dict, optional): Dictionary of normalized right side AU values\
        zone (str, optional): Current facial zone being analyzed\
        \
    Returns:\
        tuple: (has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, confidence_boost)\
    """\
    confidence_boost = 0.0\
    \
    for au in left_values:\
        if au in right_values:\
            # Skip if either value is too small for reliable comparison\
            if left_values[au] < 0.2 and right_values[au] < 0.2:\
                continue\
            \
            # Determine if we should use normalized values for this AU\
            if zone is None:\
                for z, aus in self.facial_zones.items():\
                    if au in aus:\
                        zone = z\
                        break\
            \
            use_normalized = False\
            au_base = au.split('_')[0] + '_r'\
            if zone and zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[zone]:\
                use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized', False)\
            \
            # Get the appropriate values\
            left_val = left_values[au]\
            right_val = right_values[au]\
            \
            if use_normalized and left_values_normalized and right_values_normalized:\
                if au in left_values_normalized and au in right_values_normalized:\
                    left_val = left_values_normalized[au]\
                    right_val = right_values_normalized[au]\
                    logger.debug(f"Using normalized values for extreme asymmetry check on \{au\}: L=\{left_val\}, R=\{right_val\}")\
                \
            # Calculate percent difference for this AU\
            avg = (left_val + right_val) / 2\
            if avg > 0:\
                percent_diff = abs(left_val - right_val) / avg * 100\
                \
                # Determine appropriate asymmetry threshold based on AU\
                asymmetry_threshold = EXTREME_ASYMMETRY_THRESHOLD  # Default\
                \
                # Special cases for different AUs\
                if zone == 'lower' and au_base == 'AU12_r':\
                    asymmetry_threshold = self.au12_extreme_asymmetry_threshold  # Lower threshold for AU12_r\
                    logger.debug(f"Using lower threshold (\{asymmetry_threshold\}) for AU12_r extreme asymmetry")\
                elif zone == 'upper' and au_base == 'AU01_r':\
                    asymmetry_threshold = self.au01_extreme_asymmetry_threshold  # Lower threshold for AU01_r\
                    logger.debug(f"Using lower threshold (\{asymmetry_threshold\}) for AU01_r extreme asymmetry")\
                elif zone == 'mid' and au_base == 'AU45_r':\
                    asymmetry_threshold = self.au45_extreme_asymmetry_threshold  # Lower threshold for AU45_r\
                    logger.debug(f"Using lower threshold (\{asymmetry_threshold\}) for AU45_r extreme asymmetry")\
                elif zone == 'mid' and au_base == 'AU07_r':\
                    asymmetry_threshold = self.au07_extreme_asymmetry_threshold  # Lower threshold for AU07_r\
                    logger.debug(f"Using lower threshold (\{asymmetry_threshold\}) for AU07_r extreme asymmetry")\
                \
                # Special case for extreme asymmetry in specific AUs\
                special_au_threshold = 180.0\
                if ((zone == 'lower' and au_base == 'AU12_r') or \
                    (zone == 'upper' and au_base == 'AU01_r') or\
                    (zone == 'mid' and au_base == 'AU45_r') or\
                    (zone == 'mid' and au_base == 'AU07_r')) and percent_diff > special_au_threshold:\
                    # Determine which side is weaker\
                    if left_val < right_val:\
                        weaker_side = 'left'\
                    else:\
                        weaker_side = 'right'\
                    # Return with higher confidence boost\
                    logger.debug(f"Detected VERY extreme asymmetry for \{au\}: \{percent_diff:.1f\}% (threshold: \{special_au_threshold\}), weaker side: \{weaker_side\}")\
                    return True, au, percent_diff, weaker_side, 0.2\
                \
                # If extremely asymmetric, return the AU, percent difference, and weaker side\
                if percent_diff > asymmetry_threshold:\
                    # Determine which side is weaker\
                    if left_val < right_val:\
                        weaker_side = 'left'\
                    else:\
                        weaker_side = 'right'\
                    logger.debug(f"Detected extreme asymmetry for \{au\}: \{percent_diff:.1f\}% (threshold: \{asymmetry_threshold\}), weaker side: \{weaker_side\}")\
                    return True, au, percent_diff, weaker_side, confidence_boost\
                    \
    return False, None, 0, None, 0.0\
\
def check_for_borderline_cases(self, zone, side, values, other_side_values, ratio, max_percent_diff, confidence_score):\
    """\
    Detect borderline cases that might be just above thresholds\
    \
    Args:\
        self: The FacialParalysisDetector instance\
        zone (str): 'upper', 'mid', or 'lower'\
        side (str): 'left' or 'right'\
        values (dict): Dictionary of AU values for this side\
        other_side_values (dict): Dictionary of AU values for the other side\
        ratio (float): Calculated ratio between sides\
        max_percent_diff (float): Maximum percent difference calculated\
        confidence_score (float): Current confidence score\
        \
    Returns:\
        bool: True if borderline case detected, False otherwise\
        float: Adjusted confidence score\
    """\
    # For mid face specific detection\
    if zone == 'mid' and 'AU45_r' in values:\
        # If ratio is close to threshold (within 0.05) and percent diff is high\
        if (ratio >= self.asymmetry_thresholds[zone]['complete']['ratio'] and \
            ratio <= self.asymmetry_thresholds[zone]['complete']['ratio'] + 0.05 and\
            max_percent_diff > 75.0):\
            \
            # Add confidence bonus if specified in modifiers\
            confidence_bonus = 0\
            if 'AU45_r' in self.au_zone_detection_modifiers.get(zone, \{\}):\
                confidence_bonus = self.au_zone_detection_modifiers[zone]['AU45_r'].get('confidence_bonus', 0)\
            \
            logger.debug(f"Detected borderline case for mid zone: ratio=\{ratio:.3f\}, percent_diff=\{max_percent_diff:.1f\}%, confidence boost: \{confidence_bonus\}")\
            # Return that this is a borderline case and the adjusted confidence\
            return True, confidence_score + confidence_bonus\
        \
        # Check for AU7_r in mid face\
        if 'AU07_r' in values and 'AU07_r' in other_side_values:\
            au7_value = values['AU07_r']\
            other_au7_value = other_side_values['AU07_r']\
            \
            if au7_value > 0 and other_au7_value > 0:\
                au7_ratio = min(au7_value, other_au7_value) / max(au7_value, other_au7_value)\
                au7_percent_diff = calculate_percent_difference(au7_value, other_au7_value)\
                \
                # If AU7_r shows significant asymmetry even though overall ratio is borderline\
                if (au7_ratio < 0.5 and au7_percent_diff > 65.0):\
                    # Add confidence bonus\
                    confidence_bonus = 0.1\
                    logger.debug(f"Detected AU07_r borderline case in midface: ratio=\{au7_ratio:.3f\}, percent_diff=\{au7_percent_diff:.1f\}%, confidence boost: \{confidence_bonus\}")\
                    return True, confidence_score + confidence_bonus\
            \
    # Special check for upper face with AU01_r - use normalized values for ratio check\
    if zone == 'upper' and 'AU01_r' in values and 'AU01_r' in other_side_values:\
        # Get normalized values if available\
        au01_value = values['AU01_r']\
        other_au01_value = other_side_values['AU01_r']\
        \
        # Try to get normalized values\
        if hasattr(self, 'current_normalized_values') and self.current_normalized_values:\
            curr_side = self.current_normalized_values.get(side, \{\})\
            other_side_norm = self.current_normalized_values.get('left' if side == 'right' else 'right', \{\})\
            \
            if 'AU01_r' in curr_side and 'AU01_r' in other_side_norm:\
                au01_value = curr_side['AU01_r']\
                other_au01_value = other_side_norm['AU01_r']\
                \
                if au01_value > 0 and other_au01_value > 0:\
                    norm_ratio = min(au01_value, other_au01_value) / max(au01_value, other_au01_value)\
                    norm_percent_diff = abs(au01_value - other_au01_value) / ((au01_value + other_au01_value) / 2) * 100\
                    \
                    # If normalized ratio is below threshold even though raw ratio is above\
                    if norm_ratio < self.asymmetry_thresholds[zone]['complete']['ratio'] and norm_percent_diff > 75.0:\
                        logger.debug(f"Detected AU01_r normalized ratio borderline case: ratio=\{norm_ratio:.3f\} (raw=\{ratio:.3f\}), percent_diff=\{norm_percent_diff:.1f\}%")\
                        return True, confidence_score + 0.1\
    \
    # Special check for lower face with AU12_r - use normalized values for ratio check\
    if zone == 'lower' and 'AU12_r' in values and 'AU12_r' in other_side_values:\
        # Get normalized values if available\
        au12_value = values['AU12_r']\
        other_au12_value = other_side_values['AU12_r']\
        \
        # Try to get normalized values\
        if hasattr(self, 'current_normalized_values') and self.current_normalized_values:\
            curr_side = self.current_normalized_values.get(side, \{\})\
            other_side_norm = self.current_normalized_values.get('left' if side == 'right' else 'right', \{\})\
            \
            if 'AU12_r' in curr_side and 'AU12_r' in other_side_norm:\
                au12_value = curr_side['AU12_r']\
                other_au12_value = other_side_norm['AU12_r']\
                \
                if au12_value > 0 and other_au12_value > 0:\
                    norm_ratio = min(au12_value, other_au12_value) / max(au12_value, other_au12_value)\
                    norm_percent_diff = abs(au12_value - other_au12_value) / ((au12_value + other_au12_value) / 2) * 100\
                    \
                    # If normalized ratio is below threshold even though raw ratio is above\
                    if norm_ratio < self.asymmetry_thresholds[zone]['complete']['ratio'] and norm_percent_diff > 75.0:\
                        logger.debug(f"Detected AU12_r normalized ratio borderline case: ratio=\{norm_ratio:.3f\} (raw=\{ratio:.3f\}), percent_diff=\{norm_percent_diff:.1f\}%")\
                        return True, confidence_score + 0.1\
        \
    # Check for lower face borderline cases\
    if zone == 'lower':\
        # If we have strong asymmetry in more than one AU\
        au_asymmetries = []\
        for au in values:\
            if au in other_side_values and values[au] > 0 and other_side_values[au] > 0:\
                percent_diff = calculate_percent_difference(values[au], other_side_values[au])\
                if percent_diff > 70.0:  # Significant asymmetry\
                    au_asymmetries.append((au, percent_diff))\
        \
        # If we have multiple AUs with high asymmetry\
        if len(au_asymmetries) >= 2:\
            # Calculate the average asymmetry\
            avg_asymmetry = sum(a[1] for a in au_asymmetries) / len(au_asymmetries)\
            if avg_asymmetry > 80.0:  # Very high average asymmetry\
                logger.debug(f"Detected lower face multi-AU asymmetry: \{au_asymmetries\}, avg=\{avg_asymmetry:.1f\}%")\
                return True, confidence_score + 0.05\
            \
        # Special case for if AU12_r has high asymmetry and ratio is close to threshold\
        for au, percent_diff in au_asymmetries:\
            if au == 'AU12_r' and percent_diff > 75.0 and ratio < 0.5:\
                logger.debug(f"Detected borderline AU12_r asymmetry: \{percent_diff:.1f\}%, ratio=\{ratio:.3f\}")\
                return True, confidence_score + 0.08\
                \
    # Check for upper face borderline cases\
    if zone == 'upper':\
        # If we have strong asymmetry in more than one AU\
        au_asymmetries = []\
        for au in values:\
            if au in other_side_values and values[au] > 0 and other_side_values[au] > 0:\
                percent_diff = calculate_percent_difference(values[au], other_side_values[au])\
                if percent_diff > 70.0:  # Significant asymmetry\
                    au_asymmetries.append((au, percent_diff))\
        \
        # If we have multiple AUs with high asymmetry\
        if len(au_asymmetries) >= 2:\
            # Calculate the average asymmetry\
            avg_asymmetry = sum(a[1] for a in au_asymmetries) / len(au_asymmetries)\
            if avg_asymmetry > 80.0:  # Very high average asymmetry\
                logger.debug(f"Detected upper face multi-AU asymmetry: \{au_asymmetries\}, avg=\{avg_asymmetry:.1f\}%")\
                return True, confidence_score + 0.05\
            \
        # Special case for if AU01_r has high asymmetry and ratio is close to threshold\
        for au, percent_diff in au_asymmetries:\
            if au == 'AU01_r' and percent_diff > 75.0 and ratio < 0.5:\
                logger.debug(f"Detected borderline AU01_r asymmetry: \{percent_diff:.1f\}%, ratio=\{ratio:.3f\}")\
                return True, confidence_score + 0.08\
    \
    return False, confidence_score}