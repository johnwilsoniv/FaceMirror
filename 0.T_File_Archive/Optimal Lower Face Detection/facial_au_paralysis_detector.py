{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;}
{\colortbl;\red255\green255\blue255;\red31\green29\blue21;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c16078\c14902\c10588;\cssrgb\c100000\c100000\c100000;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 """\
Facial paralysis detection module.\
Analyzes facial Action Units to detect possible facial paralysis.\
"""\
\
import numpy as np\
import logging\
from facial_au_constants import (\
    FACIAL_ZONES, ZONE_SPECIFIC_ACTIONS, PARALYSIS_THRESHOLDS,\
    ASYMMETRY_THRESHOLDS, CONFIDENCE_THRESHOLDS, FACIAL_ZONE_WEIGHTS,\
    EXTREME_ASYMMETRY_THRESHOLD, AU_ZONE_DETECTION_MODIFIERS,\
    PARALYSIS_DETECTION_AU_IMPORTANCE, BASELINE_AU_ACTIVATIONS,\
    AU12_EXTREME_ASYMMETRY_THRESHOLD, AU01_EXTREME_ASYMMETRY_THRESHOLD,\
    AU45_EXTREME_ASYMMETRY_THRESHOLD, AU07_EXTREME_ASYMMETRY_THRESHOLD,\
    MIDFACE_FUNCTIONAL_THRESHOLDS, USE_FUNCTIONAL_APPROACH_FOR_MIDFACE,\
    MIDFACE_COMPONENT_THRESHOLDS, USE_DUAL_CRITERIA_FOR_MIDFACE,\
    MIDFACE_COMBINED_SCORE_THRESHOLDS, USE_COMBINED_SCORE_FOR_MIDFACE,\
    # Lower face constants\
    LOWER_FACE_COMBINED_SCORE_THRESHOLDS, USE_COMBINED_SCORE_FOR_LOWER_FACE,\
    LOWER_FACE_PARALYSIS_THRESHOLDS, LOWER_FACE_ASYMMETRY_THRESHOLDS,\
    LOWER_FACE_CONFIDENCE_THRESHOLDS\
)\
\
# Import helper modules\
from facial_paralysis_helpers import (\
    calculate_weighted_activation, calculate_percent_difference,\
    calculate_confidence_score, calculate_midface_combined_score,\
    calculate_combined_midface_score,  # New helper function for combined midface score\
    calculate_lower_face_combined_score,\
    calculate_functional_component, calculate_asymmetry_component,\
    calculate_midface_functional_score\
)\
from facial_paralysis_analysis import check_for_extreme_asymmetry, check_for_borderline_cases\
from facial_paralysis_detection_left import detect_left_side_paralysis\
from facial_paralysis_detection_right import detect_right_side_paralysis\
\
# Configure logging\
logging.basicConfig(\
    level=logging.INFO,\
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'\
)\
logger = logging.getLogger(__name__)\
\
class FacialParalysisDetector:\
    """\
    Detects facial paralysis by analyzing AU asymmetry patterns by facial zone.\
    """\
\
    def __init__(self):\
        """Initialize the facial paralysis detector."""\
        self.facial_zones = FACIAL_ZONES\
        self.zone_specific_actions = ZONE_SPECIFIC_ACTIONS\
        self.paralysis_thresholds = PARALYSIS_THRESHOLDS\
        self.asymmetry_thresholds = ASYMMETRY_THRESHOLDS\
        self.confidence_thresholds = CONFIDENCE_THRESHOLDS\
        # Add reference to the zone weights\
        self.facial_zone_weights = FACIAL_ZONE_WEIGHTS\
        # Add reference to new constants\
        self.au_zone_detection_modifiers = AU_ZONE_DETECTION_MODIFIERS\
        self.paralysis_detection_au_importance = PARALYSIS_DETECTION_AU_IMPORTANCE\
        self.baseline_au_activations = BASELINE_AU_ACTIVATIONS\
        # For AU12_r in the lower face, extreme asymmetry threshold is lower\
        self.au12_extreme_asymmetry_threshold = AU12_EXTREME_ASYMMETRY_THRESHOLD\
        # For AU01_r in the upper face, extreme asymmetry threshold is also lower\
        self.au01_extreme_asymmetry_threshold = AU01_EXTREME_ASYMMETRY_THRESHOLD\
        # For AU45_r in the mid face, extreme asymmetry threshold is also lower\
        self.au45_extreme_asymmetry_threshold = AU45_EXTREME_ASYMMETRY_THRESHOLD\
        # For AU07_r in the mid face, extreme asymmetry threshold\
        self.au07_extreme_asymmetry_threshold = AU07_EXTREME_ASYMMETRY_THRESHOLD\
\
        # NEW: Add combined score related attributes for midface\
        self.midface_combined_score_thresholds = MIDFACE_COMBINED_SCORE_THRESHOLDS\
        self.use_combined_score_for_midface = USE_COMBINED_SCORE_FOR_MIDFACE\
        # Add reference to component thresholds for mid face\
        self.midface_component_thresholds = MIDFACE_COMPONENT_THRESHOLDS\
        self.use_dual_criteria_for_midface = USE_DUAL_CRITERIA_FOR_MIDFACE\
        # NEW: Add combined score related attributes for lower face\
        self.lower_face_combined_score_thresholds = LOWER_FACE_COMBINED_SCORE_THRESHOLDS\
        self.use_combined_score_for_lower_face = USE_COMBINED_SCORE_FOR_LOWER_FACE\
\
        # NEW: Add lower face specific thresholds\
        self.lower_face_paralysis_thresholds = LOWER_FACE_PARALYSIS_THRESHOLDS\
        self.lower_face_asymmetry_thresholds = LOWER_FACE_ASYMMETRY_THRESHOLDS\
        self.lower_face_confidence_thresholds = LOWER_FACE_CONFIDENCE_THRESHOLDS\
\
        self.midface_functional_thresholds = MIDFACE_FUNCTIONAL_THRESHOLDS\
        self.use_functional_approach_for_midface = USE_FUNCTIONAL_APPROACH_FOR_MIDFACE\
\
        # Make helper functions available as methods by binding them to self\
        from facial_paralysis_helpers import (\
            calculate_weighted_activation, calculate_percent_difference,\
            calculate_confidence_score, calculate_midface_combined_score,\
            calculate_combined_midface_score, calculate_lower_face_combined_score,\
            calculate_functional_component, calculate_asymmetry_component,\
            calculate_midface_functional_score\
        )\
\
        self._calculate_weighted_activation = lambda *args, **kwargs: calculate_weighted_activation(self, *args,\
                                                                                                    **kwargs)\
        self._calculate_percent_difference = calculate_percent_difference\
        self._calculate_confidence_score = lambda *args, **kwargs: calculate_confidence_score(self, *args, **kwargs)\
        self._calculate_midface_combined_score = calculate_midface_combined_score\
        self._calculate_combined_midface_score = calculate_combined_midface_score\
        self._calculate_lower_face_combined_score = calculate_lower_face_combined_score\
        self._calculate_midface_functional_score = calculate_midface_functional_score  # Fixed: Added binding for this function\
        self._calculate_functional_component = calculate_functional_component\
        self._calculate_asymmetry_component = calculate_asymmetry_component\
\
        from facial_paralysis_analysis import check_for_extreme_asymmetry, check_for_borderline_cases\
        self._check_for_extreme_asymmetry = lambda *args, **kwargs: check_for_extreme_asymmetry(self, *args, **kwargs)\
        self._check_for_borderline_cases = lambda *args, **kwargs: check_for_borderline_cases(self, *args, **kwargs)\
\
    # In facial_au_paralysis_detector.py, add this method to the FacialParalysisDetector class\
\
    def process_midface_combined_score(self, info, zone, aus, side,\
                                       current_side_values, other_side_values,\
                                       current_side_normalized, other_side_normalized,\
                                       confidence_score, zone_paralysis, affected_aus_by_zone_side,\
                                       criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None):\
        """Process midface zone using the combined score approach.\
\
        Args:\
            info (dict): Current action info dictionary to update\
            zone (str): Facial zone being analyzed ('mid')\
            aus (list): Action units in this zone\
            side (str): 'left' or 'right' - which side we're checking for paralysis\
            current_side_values (dict): AU values for the side being checked\
            other_side_values (dict): AU values for the opposite side\
            current_side_normalized (dict): Normalized AU values for the side being checked\
            other_side_normalized (dict): Normalized AU values for the opposite side\
            confidence_score (float): Confidence score for detection\
            zone_paralysis (dict): Tracking paralysis by zone\
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side\
            criteria_met (dict, optional): Dictionary of detection criteria that were met\
            asymmetry_thresholds (dict, optional): Thresholds for asymmetry detection\
            confidence_thresholds (dict, optional): Thresholds for confidence scores\
\
        Returns:\
            bool: True if successfully processed, False otherwise\
        """\
        # Check if we have the necessary AUs for combined score calculation\
        if 'AU45_r' in current_side_values and 'AU45_r' in other_side_values:\
            # Get the appropriate values for AU45_r (eyelid closure)\
            current_au45 = current_side_values['AU45_r']\
            other_au45 = other_side_values['AU45_r']\
\
            if current_side_normalized and other_side_normalized and 'AU45_r' in current_side_normalized and 'AU45_r' in other_side_normalized:\
                current_au45 = current_side_normalized['AU45_r']\
                other_au45 = other_side_normalized['AU45_r']\
\
            # Also check for AU07_r if available (lid tightener) from BS action\
            au7_available = False\
            current_au7 = 0\
            other_au7 = 0\
\
            # Try to get AU07_r values from the BS action if available\
            bs_data = None\
            for action_name, action_data in info.items():\
                if action_name == 'BS':\
                    bs_data = action_data\
                    break\
\
            # If BS data is available, extract AU7_r values\
            if bs_data and 'left' in bs_data and 'right' in bs_data:\
                bs_side_values = bs_data[side]['au_values'] if side in bs_data else None\
                bs_other_values = bs_data['left' if side == 'right' else 'right']['au_values']\
\
                bs_side_normalized = bs_data[side].get('normalized_au_values', \{\}) if side in bs_data else \{\}\
                bs_other_normalized = bs_data['left' if side == 'right' else 'right'].get('normalized_au_values', \{\})\
\
                if bs_side_normalized and bs_other_normalized and 'AU07_r' in bs_side_normalized and 'AU07_r' in bs_other_normalized:\
                    current_au7 = bs_side_normalized['AU07_r']\
                    other_au7 = bs_other_normalized['AU07_r']\
                    au7_available = True\
\
            # Calculate ratios for AU45_r and AU07_r (if available)\
            au45_ratio = min(current_au45, other_au45) / max(current_au45, other_au45) if max(current_au45,\
                                                                                              other_au45) > 0 else 0\
            au7_ratio = min(current_au7, other_au7) / max(current_au7, other_au7) if au7_available and max(current_au7,\
                                                                                                           other_au7) > 0 else 0\
\
            # Calculate combined score - lower values indicate more paralysis\
            combined_score = 0\
            if au7_available:\
                # If AU07_r is available, use weighted average of both AU scores\
                au45_weight = 0.7  # Higher weight for primary AU45_r\
                au7_weight = 0.3  # Lower weight for supporting AU07_r\
\
                # Calculate combined score with both AU45_r and AU07_r\
                au45_component = au45_ratio * (1 + current_au45 / 3.0)\
                au7_component = au7_ratio * (1 + current_au7 / 3.0)\
                combined_score = (au45_component * au45_weight) + (au7_component * au7_weight)\
            else:\
                # If only AU45_r is available, use only it for score\
                combined_score = au45_ratio * (1 + current_au45 / 3.0)\
\
            # Store confidence score\
            info['paralysis']['confidence'][side][zone] = confidence_score\
\
            # Initialize tracking structures if needed\
            if 'combined_score' not in info['paralysis']['contributing_aus'][side][zone]:\
                info['paralysis']['contributing_aus'][side][zone]['combined_score'] = []\
\
            # Use thresholds from constants - these determine severity levels\
            complete_threshold = self.midface_combined_score_thresholds['complete']\
            partial_threshold = self.midface_combined_score_thresholds['partial']\
\
            # Adjust confidence for midface - often needs a boost\
            adjusted_confidence = max(confidence_score, 0.3)\
\
            # Determine paralysis severity based on combined score and confidence\
            if combined_score < complete_threshold and adjusted_confidence >= self.confidence_thresholds[zone][\
                'complete']:\
                # Complete paralysis\
                info['paralysis']['zones'][side][zone] = 'Complete'  # String literal\
\
                # Track for patient-level assessment\
                if zone_paralysis[side][zone] != 'Complete':\
                    zone_paralysis[side][zone] = 'Complete'  # String literal\
\
                # Track contributing AUs\
                affected_aus_by_zone_side[side][zone].add('AU45_r')\
                if 'AU45_r' not in info['paralysis']['affected_aus'][side]:\
                    info['paralysis']['affected_aus'][side].append('AU45_r')\
\
                if au7_available:\
                    affected_aus_by_zone_side[side][zone].add('AU07_r')\
                    if 'AU07_r' not in info['paralysis']['affected_aus'][side]:\
                        info['paralysis']['affected_aus'][side].append('AU07_r')\
\
                # Track contribution details\
                info['paralysis']['contributing_aus'][side][zone]['combined_score'].append(\{\
                    'au': 'AU45_r + AU07_r' if au7_available else 'AU45_r',\
                    'current_au45': current_au45,\
                    'other_au45': other_au45,\
                    'au45_ratio': au45_ratio,\
                    'current_au7': current_au7 if au7_available else 'NA',\
                    'other_au7': other_au7 if au7_available else 'NA',\
                    'au7_ratio': au7_ratio if au7_available else 'NA',\
                    'combined_score': combined_score,\
                    'threshold': complete_threshold,\
                    'type': 'Complete'\
                \})\
\
                return True  # Successfully processed\
\
            # Check for partial paralysis\
            elif combined_score < partial_threshold and adjusted_confidence >= self.confidence_thresholds[zone][\
                'partial']:\
                # Partial paralysis - only if not already Complete\
                if info['paralysis']['zones'][side][zone] == 'None':\
                    info['paralysis']['zones'][side][zone] = 'Partial'  # String literal\
\
                # Track for patient-level assessment (if not already Complete)\
                if zone_paralysis[side][zone] == 'None':\
                    zone_paralysis[side][zone] = 'Partial'  # String literal\
\
                # Track contributing AUs\
                affected_aus_by_zone_side[side][zone].add('AU45_r')\
                if 'AU45_r' not in info['paralysis']['affected_aus'][side]:\
                    info['paralysis']['affected_aus'][side].append('AU45_r')\
\
                if au7_available:\
                    affected_aus_by_zone_side[side][zone].add('AU07_r')\
                    if 'AU07_r' not in info['paralysis']['affected_aus'][side]:\
                        info['paralysis']['affected_aus'][side].append('AU07_r')\
\
                # Track contribution details\
                info['paralysis']['contributing_aus'][side][zone]['combined_score'].append(\{\
                    'au': 'AU45_r + AU07_r' if au7_available else 'AU45_r',\
                    'current_au45': current_au45,\
                    'other_au45': other_au45,\
                    'au45_ratio': au45_ratio,\
                    'current_au7': current_au7 if au7_available else 'NA',\
                    'other_au7': other_au7 if au7_available else 'NA',\
                    'au7_ratio': au7_ratio if au7_available else 'NA',\
                    'combined_score': combined_score,\
                    'threshold': partial_threshold,\
                    'type': 'Partial'\
                \})\
\
                return True  # Successfully processed\
            else:\
                # No paralysis detected\
                info['paralysis']['zones'][side][zone] = 'None'  # String literal\
                return True  # Still successfully processed even if no paralysis\
\
        return False  # Couldn't process with combined score approach\
\
    def process_midface_combined_score_left(self, info, zone, aus, left_values, right_values,\
                                            left_values_normalized, right_values_normalized,\
                                            confidence_score, zone_paralysis, affected_aus_by_zone_side,\
                                            criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None):\
        """Wrapper method that calls the unified function for left side detection."""\
        return self.process_midface_combined_score(\
            info, zone, aus, 'left',\
            left_values, right_values,\
            left_values_normalized, right_values_normalized,\
            confidence_score, zone_paralysis, affected_aus_by_zone_side,\
            criteria_met, asymmetry_thresholds, confidence_thresholds\
        )\
\
    def process_midface_combined_score_right(self, info, zone, aus, right_values, left_values,\
                                             right_values_normalized, left_values_normalized,\
                                             confidence_score, zone_paralysis, affected_aus_by_zone_side,\
                                             criteria_met=None, asymmetry_thresholds=None, confidence_thresholds=None):\
        """Wrapper method that calls the unified function for right side detection."""\
        return self.process_midface_combined_score(\
            info, zone, aus, 'right',\
            right_values, left_values,\
            right_values_normalized, left_values_normalized,\
            confidence_score, zone_paralysis, affected_aus_by_zone_side,\
            criteria_met, asymmetry_thresholds, confidence_thresholds\
        )\
\
    def process_lower_face_combined_score(self, info, zone, aus, side,\
                                          current_side_values, other_side_values,\
                                          current_side_normalized, other_side_normalized,\
                                          confidence_score, zone_paralysis, affected_aus_by_zone_side,\
                                          criteria_met, asymmetry_thresholds, confidence_thresholds):\
        """Process lower face zone using the combined score approach.\
\
        Args:\
            info (dict): Current action info dictionary to update\
            zone (str): Facial zone being analyzed ('lower')\
            aus (list): Action units in this zone\
            side (str): 'left' or 'right' - which side we're checking for paralysis\
            current_side_values (dict): AU values for the side being checked\
            other_side_values (dict): AU values for the opposite side\
            current_side_normalized (dict): Normalized AU values for the side being checked\
            other_side_normalized (dict): Normalized AU values for the opposite side\
            confidence_score (float): Confidence score for detection\
            zone_paralysis (dict): Tracking paralysis by zone\
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side\
            criteria_met (dict): Dictionary of detection criteria that were met\
            asymmetry_thresholds (dict): Thresholds for asymmetry detection\
            confidence_thresholds (dict): Thresholds for confidence scores\
\
        Returns:\
            bool: True if successfully processed, False otherwise\
        """\
        # Check if we have the necessary AUs for combined score calculation\
        if 'AU12_r' in current_side_values and 'AU12_r' in other_side_values:\
            # Get the appropriate values for AU12_r\
            current_au12 = current_side_values['AU12_r']\
            other_au12 = other_side_values['AU12_r']\
\
            if current_side_normalized and other_side_normalized and 'AU12_r' in current_side_normalized and 'AU12_r' in other_side_normalized:\
                current_au12 = current_side_normalized['AU12_r']\
                other_au12 = other_side_normalized['AU12_r']\
\
            # SPECIAL HANDLING: Check if AU25_r is missing or zero on both sides\
            has_au25 = ('AU25_r' in current_side_values and 'AU25_r' in other_side_values and\
                        current_side_values['AU25_r'] > 0 and other_side_values['AU25_r'] > 0)\
\
            if not has_au25 and current_au12 > 0 and other_au12 > 0:\
                # Special case: Calculate score using only AU12_r\
                au12_ratio = min(current_au12, other_au12) / max(current_au12, other_au12)\
                au12_min = current_au12  # Since we're checking current side weakness\
\
                # Calculate simplified score for AU12-only\
                simplified_score = au12_ratio * (1 + au12_min / 5)\
\
                # Calculate percent difference for AU12\
                au12_percent_diff = abs(current_au12 - other_au12) / ((current_au12 + other_au12) / 2) * 100\
\
                # FIX for IMG_4923: Use a special threshold (0.80) for AU12-only detection\
                special_threshold = 0.80\
\
                # FIX for IMG_4923: Detect based on either simplified score OR strong percent difference\
                if ((simplified_score < special_threshold and current_au12 < other_au12) or\
                        (au12_percent_diff > 40 and current_au12 < other_au12)):\
\
                    # FIX for IMG_4923: Special confidence boost for missing AU25 cases\
                    adjusted_confidence = max(confidence_score, 0.4)  # Ensure minimum confidence of 0.4\
\
                    # Check confidence with adjusted value\
                    if adjusted_confidence >= self.lower_face_confidence_thresholds['partial']:\
                        # It's partial paralysis\
                        info['paralysis']['zones'][side][zone] = 'Partial'\
\
                        # Track for patient-level assessment\
                        if zone_paralysis[side][zone] == 'None':\
                            zone_paralysis[side][zone] = 'Partial'\
\
                        # Add affected AUs\
                        affected_aus_by_zone_side[side][zone].add('AU12_r')\
                        if 'AU12_r' not in info['paralysis']['affected_aus'][side]:\
                            info['paralysis']['affected_aus'][side].append('AU12_r')\
\
                        # Track special AU12-only detection with enhanced logging\
                        info['paralysis']['contributing_aus'][side][zone]['combined_score'].append(\{\
                            'au': 'AU12_r only (missing AU25)',\
                            'current_au12': current_au12,\
                            'other_au12': other_au12,\
                            'au12_ratio': au12_ratio,\
                            'au12_percent_diff': au12_percent_diff,\
                            'simplified_score': simplified_score,\
                            'threshold': special_threshold,\
                            'adjusted_confidence': adjusted_confidence,\
                            'original_confidence': confidence_score,\
                            'type': 'Partial'\
                        \})\
\
                        return True  # Successfully processed\
\
            # Continue with standard processing if AU25_r is present or if special case didn't detect paralysis\
            if 'AU25_r' in current_side_values and 'AU25_r' in other_side_values:\
                # Get the appropriate values for AU25_r\
                current_au25 = current_side_values['AU25_r']\
                other_au25 = other_side_values['AU25_r']\
\
                if current_side_normalized and other_side_normalized and 'AU25_r' in current_side_normalized and 'AU25_r' in other_side_normalized:\
                    current_au25 = current_side_normalized['AU25_r']\
                    other_au25 = other_side_normalized['AU25_r']\
\
                # Calculate ratios\
                au12_ratio = current_au12 / other_au12 if other_au12 > 0 else 0\
                au25_ratio = current_au25 / other_au25 if other_au25 > 0 else 0\
\
                # For detection purposes we want the smaller/larger ratio\
                au12_ratio = min(au12_ratio, 1.0) if au12_ratio > 0 else 0\
                au25_ratio = min(au25_ratio, 1.0) if au25_ratio > 0 else 0\
\
                # Minimum value (from current side since we're checking current side paralysis)\
                au12_min = current_au12\
                au25_min = current_au25\
\
                # Calculate combined score\
                combined_score = self._calculate_lower_face_combined_score(au12_ratio, au25_ratio, au12_min, au25_min)\
                logger.debug(\
                    f"\{side.capitalize()\} lower face combined score: \{combined_score:.3f\} (AU12_ratio=\{au12_ratio:.3f\}, AU25_ratio=\{au25_ratio:.3f\})")\
\
                # Use thresholds from constants\
                partial_threshold = self.lower_face_combined_score_thresholds['partial']\
                complete_threshold = self.lower_face_combined_score_thresholds['complete']\
\
                # Check if meets partial criteria\
                if combined_score < partial_threshold and confidence_score >= self.lower_face_confidence_thresholds[\
                    'partial']:\
                    # Track contribution for combined score calculation\
                    info['paralysis']['contributing_aus'][side][zone]['combined_score'].append(\{\
                        'au': 'Combined Lower Face',\
                        'current_au12': current_au12,\
                        'other_au12': other_au12,\
                        'current_au25': current_au25,\
                        'other_au25': other_au25,\
                        'au12_ratio': au12_ratio,\
                        'au25_ratio': au25_ratio,\
                        'combined_score': combined_score,\
                        'threshold': partial_threshold,\
                        'type': 'Partial'\
                    \})\
\
                    # See if it meets complete criteria\
                    if combined_score < complete_threshold and confidence_score >= \\\
                            self.lower_face_confidence_thresholds['complete']:\
                        # Complete paralysis\
                        info['paralysis']['zones'][side][zone] = 'Complete'\
\
                        # Track for patient-level assessment\
                        if zone_paralysis[side][zone] != 'Complete':\
                            zone_paralysis[side][zone] = 'Complete'\
\
                        # Update tracking information\
                        info['paralysis']['contributing_aus'][side][zone]['combined_score'][-1]['type'] = 'Complete'\
                        info['paralysis']['contributing_aus'][side][zone]['combined_score'][-1][\
                            'threshold'] = complete_threshold\
                    else:\
                        # Partial paralysis (if not already marked as Complete)\
                        if info['paralysis']['zones'][side][zone] == 'None':\
                            info['paralysis']['zones'][side][zone] = 'Partial'\
\
                        # Track for patient-level assessment\
                        if zone_paralysis[side][zone] == 'None':\
                            zone_paralysis[side][zone] = 'Partial'\
\
                    # Add affected AUs\
                    affected_aus_by_zone_side[side][zone].add('AU12_r')\
                    if 'AU12_r' not in info['paralysis']['affected_aus'][side]:\
                        info['paralysis']['affected_aus'][side].append('AU12_r')\
\
                    affected_aus_by_zone_side[side][zone].add('AU25_r')\
                    if 'AU25_r' not in info['paralysis']['affected_aus'][side]:\
                        info['paralysis']['affected_aus'][side].append('AU25_r')\
\
                    return True  # Successfully processed\
\
        return False  # Could not process with combined score\
\
    def process_lower_face_combined_score_left(self, info, zone, aus, left_values, right_values,\
                                               left_values_normalized, right_values_normalized,\
                                               confidence_score, zone_paralysis, affected_aus_by_zone_side,\
                                               criteria_met, asymmetry_thresholds, confidence_thresholds):\
        """Wrapper method that calls the unified function for left side detection."""\
        return self.process_lower_face_combined_score(\
            info, zone, aus, 'left',\
            left_values, right_values,\
            left_values_normalized, right_values_normalized,\
            confidence_score, zone_paralysis, affected_aus_by_zone_side,\
            criteria_met, asymmetry_thresholds, confidence_thresholds\
        )\
\
    def process_midface_functional_approach_left(self, info, zone, aus, left_values, right_values,\
                                                 left_values_normalized, right_values_normalized,\
                                                 confidence_score, zone_paralysis, affected_aus_by_zone_side):\
        """\
        Process midface zone using a functional approach for left side.\
\
        Args:\
            self: The FacialParalysisDetector instance\
            info (dict): Current action info dictionary to update\
            zone (str): Facial zone being analyzed ('mid')\
            aus (list): Action units in this zone\
            left_values (dict): Left side AU values\
            right_values (dict): Right side AU values\
            left_values_normalized (dict): Normalized left side AU values\
            right_values_normalized (dict): Normalized right side AU values\
            confidence_score (float): Confidence score for detection\
            zone_paralysis (dict): Tracking paralysis by zone\
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side\
\
        Returns:\
            bool: True if successfully processed, False otherwise\
        """\
        return self.process_midface_functional_approach(\
            info, zone, aus, 'left',\
            left_values, right_values,\
            left_values_normalized, right_values_normalized,\
            confidence_score, zone_paralysis, affected_aus_by_zone_side\
        )\
\
    def process_midface_functional_approach_right(self, info, zone, aus, right_values, left_values,\
                                                  right_values_normalized, left_values_normalized,\
                                                  confidence_score, zone_paralysis, affected_aus_by_zone_side):\
        """\
        Process midface zone using a functional approach for right side.\
\
        Args:\
            self: The FacialParalysisDetector instance\
            info (dict): Current action info dictionary to update\
            zone (str): Facial zone being analyzed ('mid')\
            aus (list): Action units in this zone\
            right_values (dict): Right side AU values\
            left_values (dict): Left side AU values\
            right_values_normalized (dict): Normalized right side AU values\
            left_values_normalized (dict): Normalized left side AU values\
            confidence_score (float): Confidence score for detection\
            zone_paralysis (dict): Tracking paralysis by zone\
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side\
\
        Returns:\
            bool: True if successfully processed, False otherwise\
        """\
        return self.process_midface_functional_approach(\
            info, zone, aus, 'right',\
            right_values, left_values,\
            right_values_normalized, left_values_normalized,\
            confidence_score, zone_paralysis, affected_aus_by_zone_side\
        )\
\
    def process_midface_functional_approach(self, info, zone, aus, side, current_values, other_values,\
                                            current_values_normalized, other_values_normalized,\
                                            confidence_score, zone_paralysis, affected_aus_by_zone_side):\
        """\
        Process midface zone using a dual-criteria approach based on both AU45_r and AU7_r.\
\
        Args:\
            self: The FacialParalysisDetector instance\
            info (dict): Current action info dictionary to update\
            zone (str): Facial zone being analyzed ('mid')\
            aus (list): Action units in this zone\
            side (str): 'left' or 'right' - which side we're checking for paralysis\
            current_values (dict): AU values for the side being evaluated\
            other_values (dict): AU values for the opposite side\
            current_values_normalized (dict): Normalized AU values for side being evaluated\
            other_values_normalized (dict): Normalized AU values for opposite side\
            confidence_score (float): Confidence score for detection\
            zone_paralysis (dict): Tracking paralysis by zone\
            affected_aus_by_zone_side (dict): Tracking affected AUs by zone and side\
\
        Returns:\
            bool: True if successfully processed, False otherwise\
        """\
        # Double check that we have the mid zone\
        if zone != 'mid':\
            return False\
\
        # Process the ES action data for AU45_r\
        au45_processed = False\
        au45_value = 0\
        au45_ratio = 0\
        other_au45_value = 0\
\
        if 'AU45_r' in current_values_normalized and 'AU45_r' in other_values_normalized:\
            au45_value = current_values_normalized['AU45_r']\
            other_au45_value = other_values_normalized['AU45_r']\
\
            # Calculate ratio (current side / other side)\
            au45_ratio = au45_value / other_au45_value if other_au45_value > 0 else 0\
            au45_processed = True\
\
        # Process the BS action data for AU7_r\
        au7_processed = False\
        au7_value = 0\
        au7_ratio = 0\
        other_au7_value = 0\
\
        # Get the BS action data if available\
        bs_data = None\
        for action_name, action_data in info.items():\
            if action_name == 'BS':\
                bs_data = action_data\
                break\
\
        # If BS data is available, extract AU7_r values\
        if bs_data and 'left' in bs_data and 'right' in bs_data:\
            bs_side_values = bs_data[side]['au_values'] if side in bs_data else None\
            bs_other_values = bs_data['left' if side == 'right' else 'right']['au_values']\
\
            bs_side_normalized = bs_data[side].get('normalized_au_values', \{\}) if side in bs_data else \{\}\
            bs_other_normalized = bs_data['left' if side == 'right' else 'right'].get('normalized_au_values', \{\})\
\
            if bs_side_normalized and bs_other_normalized and 'AU07_r' in bs_side_normalized and 'AU07_r' in bs_other_normalized:\
                au7_value = bs_side_normalized['AU07_r']\
                other_au7_value = bs_other_normalized['AU07_r']\
\
                # Calculate ratio (current side / other side)\
                au7_ratio = au7_value / other_au7_value if other_au7_value > 0 else 0\
                au7_processed = True\
\
        # For backward compatibility, also calculate functional score using AU45_r only\
        functional_score = self._calculate_midface_functional_score(au45_value, au45_ratio)\
\
        # If both AU45_r and AU7_r are available, calculate the combined score\
        combined_score = functional_score  # Default to AU45_r score\
        if au45_processed and au7_processed:\
            combined_score = self._calculate_combined_midface_score(au45_value, au45_ratio, au7_value, au7_ratio)\
\
        # Apply looser confidence threshold for midface detection\
        adjusted_confidence = max(confidence_score, 0.3)  # Ensure minimum confidence score of 0.3\
\
        # Store confidence score\
        info['paralysis']['confidence'][side][zone] = adjusted_confidence\
\
        # Initialize component tracking if not already present\
        if 'components' not in info['paralysis']['contributing_aus'][side][zone]:\
            info['paralysis']['contributing_aus'][side][zone]['components'] = []\
\
        # For backward compatibility\
        if 'functional_score' not in info['paralysis']['contributing_aus'][side][zone]:\
            info['paralysis']['contributing_aus'][side][zone]['functional_score'] = []\
\
        # Initialize AU7 tracking if not already present\
        if 'au7_components' not in info['paralysis']['contributing_aus'][side][zone]:\
            info['paralysis']['contributing_aus'][side][zone]['au7_components'] = []\
\
        # Debug logging to verify values\
        logger.debug(\
            f"\{side\} mid face assessment: AU45=\{au45_value:.2f\}, AU45_ratio=\{au45_ratio:.2f\}, " +\
            f"AU7=\{au7_value:.2f\}, AU7_ratio=\{au7_ratio:.2f\}, " +\
            f"combined_score=\{combined_score:.2f\}, confidence=\{adjusted_confidence:.2f\}"\
        )\
\
        # Thresholds for combined score\
        combined_complete_threshold = self.midface_combined_score_thresholds['complete']\
        combined_partial_threshold = self.midface_combined_score_thresholds['partial']\
\
        # Determine severity based on combined score - complete paralysis\
        if (combined_score < combined_complete_threshold and\
                adjusted_confidence >= self.confidence_thresholds[zone]['complete']):\
\
            # Complete paralysis - CRITICAL FIX: ALWAYS use string literal for consistency with other zones\
            info['paralysis']['zones'][side][zone] = 'Complete'  # Use string 'Complete', not numerical value\
\
            # Track for patient-level assessment - CRITICAL FIX: ALWAYS use string literal\
            if zone_paralysis[side][zone] != 'Complete':\
                zone_paralysis[side][zone] = 'Complete'  # Use string 'Complete', not numerical value\
\
            # Add to affected AUs\
            if au45_processed:\
                affected_aus_by_zone_side[side][zone].add('AU45_r')\
                if 'AU45_r' not in info['paralysis']['affected_aus'][side]:\
                    info['paralysis']['affected_aus'][side].append('AU45_r')\
\
            if au7_processed:\
                affected_aus_by_zone_side[side][zone].add('AU07_r')\
                if 'AU07_r' not in info['paralysis']['affected_aus'][side]:\
                    info['paralysis']['affected_aus'][side].append('AU07_r')\
\
            # Track contribution details for AU45\
            if au45_processed:\
                info['paralysis']['contributing_aus'][side][zone]['components'].append(\{\
                    'au': 'AU45_r',\
                    'value': au45_value,\
                    'other_value': other_au45_value,\
                    'ratio': au45_ratio,\
                    'threshold': self.midface_component_thresholds['functional']['complete'],\
                    'type': 'Complete'\
                \})\
\
            # Track contribution details for AU7\
            if au7_processed:\
                info['paralysis']['contributing_aus'][side][zone]['au7_components'].append(\{\
                    'au': 'AU07_r',\
                    'value': au7_value,\
                    'other_value': other_au7_value,\
                    'ratio': au7_ratio,\
                    'threshold': self.midface_component_thresholds['au7']['complete'],\
                    'type': 'Complete',\
                    'action': 'BS'\
                \})\
\
            # For backward compatibility\
            info['paralysis']['contributing_aus'][side][zone]['functional_score'].append(\{\
                'au': 'Combined AU45_r and AU07_r',\
                'value': combined_score,\
                'threshold': combined_complete_threshold,\
                'type': 'Complete'\
            \})\
\
            logger.debug(f"\{side\} mid face: COMPLETE paralysis detected with combined AU45_r and AU07_r approach")\
\
        # Determine severity based on component thresholds - partial paralysis\
        elif (combined_score < combined_partial_threshold and\
              adjusted_confidence >= self.confidence_thresholds[zone]['partial']):\
\
            # Partial paralysis (if not already marked as Complete) - CRITICAL FIX: ALWAYS use string literal\
            if info['paralysis']['zones'][side][zone] == 'None':\
                info['paralysis']['zones'][side][zone] = 'Partial'  # Use string 'Partial', not numerical value\
\
            # Track for patient-level assessment (if not already Complete) - CRITICAL FIX: ALWAYS use string literal\
            if zone_paralysis[side][zone] == 'None':\
                zone_paralysis[side][zone] = 'Partial'  # Use string 'Partial', not numerical value\
\
            # Add to affected AUs\
            if au45_processed:\
                affected_aus_by_zone_side[side][zone].add('AU45_r')\
                if 'AU45_r' not in info['paralysis']['affected_aus'][side]:\
                    info['paralysis']['affected_aus'][side].append('AU45_r')\
\
            if au7_processed:\
                affected_aus_by_zone_side[side][zone].add('AU07_r')\
                if 'AU07_r' not in info['paralysis']['affected_aus'][side]:\
                    info['paralysis']['affected_aus'][side].append('AU07_r')\
\
            # Track contribution details for AU45\
            if au45_processed:\
                info['paralysis']['contributing_aus'][side][zone]['components'].append(\{\
                    'au': 'AU45_r',\
                    'value': au45_value,\
                    'other_value': other_au45_value,\
                    'ratio': au45_ratio,\
                    'threshold': self.midface_component_thresholds['functional']['partial'],\
                    'type': 'Partial'\
                \})\
\
            # Track contribution details for AU7\
            if au7_processed:\
                info['paralysis']['contributing_aus'][side][zone]['au7_components'].append(\{\
                    'au': 'AU07_r',\
                    'value': au7_value,\
                    'other_value': other_au7_value,\
                    'ratio': au7_ratio,\
                    'threshold': self.midface_component_thresholds['au7']['partial'],\
                    'type': 'Partial',\
                    'action': 'BS'\
                \})\
\
            # For backward compatibility\
            info['paralysis']['contributing_aus'][side][zone]['functional_score'].append(\{\
                'au': 'Combined AU45_r and AU07_r',\
                'value': combined_score,\
                'threshold': combined_partial_threshold,\
                'type': 'Partial'\
            \})\
\
            logger.debug(f"\{side\} mid face: PARTIAL paralysis detected with combined AU45_r and AU07_r approach")\
        else:\
            # CRITICAL FIX: Make sure 'None' is explicitly set as a string\
            # If no paralysis is detected, ensure we set an explicit 'None' string, not a null value or number\
            info['paralysis']['zones'][side][zone] = 'None'\
            logger.debug(f"\{side\} mid face: NO paralysis detected with combined AU45_r and AU07_r approach")\
\
        # Always return True to indicate we processed this zone\
        return True\
\
    def process_lower_face_combined_score_right(self, info, zone, aus, right_values, left_values,\
                                                right_values_normalized, left_values_normalized,\
                                                confidence_score, zone_paralysis, affected_aus_by_zone_side,\
                                                criteria_met, asymmetry_thresholds, confidence_thresholds):\
        """Wrapper method that calls the unified function for right side detection."""\
        return self.process_lower_face_combined_score(\
            info, zone, aus, 'right',\
            right_values, left_values,\
            right_values_normalized, left_values_normalized,\
            confidence_score, zone_paralysis, affected_aus_by_zone_side,\
            criteria_met, asymmetry_thresholds, confidence_thresholds\
        )\
\
    def detect_paralysis(self, results):\
        """\
        Detect potential facial paralysis by analyzing asymmetry patterns by facial zone.\
        Independently assesses each zone on both left and right sides.\
        Uses zone-specific thresholds for more accurate detection.\
        Uses a dual-metric approach for improved detection of partial vs complete paralysis.\
        Includes confidence scoring to reduce false positives.\
\
        Args:\
            results (dict): Results dictionary with AU values for each action\
\
        Returns:\
            None: Updates results dictionary in place\
        """\
        if not results:\
            logger.warning("No results to analyze for paralysis detection")\
            return\
\
        # Track affected zones and severity for each side separately\
        zone_paralysis = \{\
            'left': \{'upper': 'None', 'mid': 'None', 'lower': 'None'\},\
            'right': \{'upper': 'None', 'mid': 'None', 'lower': 'None'\}\
        \}\
\
        # Track which AUs show asymmetry by zone and side\
        affected_aus_by_zone_side = \{\
            'left': \{'upper': set(), 'mid': set(), 'lower': set()\},\
            'right': \{'upper': set(), 'mid': set(), 'lower': set()\}\
        \}\
\
        # Analyze each action for paralysis by facial zone\
        for action, info in results.items():\
            # Reset the zone-specific paralysis data for this action\
            info['paralysis'] = \{\
                'detected': False,\
                'zones': \{\
                    'left': \{'upper': 'None', 'mid': 'None', 'lower': 'None'\},\
                    'right': \{'upper': 'None', 'mid': 'None', 'lower': 'None'\}\
                \},\
                'affected_aus': \{'left': [], 'right': []\},\
                'contributing_aus': \{'left': \{\}, 'right': \{\}\},  # Structure to track thresholds and values\
                'action_relevance': \{\},  # Track if this action is relevant for each zone\
                'confidence': \{'left': \{\}, 'right': \{\}\}  # Track confidence scores\
            \}\
\
            # Analyze each facial zone for both sides\
            for zone, aus in self.facial_zones.items():\
                # Determine if this action is relevant for this zone\
                is_relevant = action in self.zone_specific_actions[zone]\
                info['paralysis']['action_relevance'][zone] = is_relevant\
\
                # Only analyze zones if action is relevant for that zone\
                if not is_relevant:\
                    continue\
\
                # Get the zone-specific thresholds for both severity levels\
                partial_thresholds = self.paralysis_thresholds[zone]['partial']\
                complete_thresholds = self.paralysis_thresholds[zone]['complete']\
\
                # Get the asymmetry thresholds for this zone\
                asymmetry_thresholds = self.asymmetry_thresholds[zone]\
\
                # Get confidence thresholds for this zone\
                confidence_thresholds = self.confidence_thresholds[zone]\
\
                # Calculate AU values for left and right sides in this zone\
                left_values = \{\}\
                right_values = \{\}\
                left_values_normalized = \{\}\
                right_values_normalized = \{\}\
\
                for au in aus:\
                    if au in info['left']['au_values'] and au in info['right']['au_values']:\
                        # Use raw values for basic collection\
                        left_value = info['left']['au_values'][au]\
                        right_value = info['right']['au_values'][au]\
\
                        left_values[au] = left_value\
                        right_values[au] = right_value\
\
                        # Also collect normalized values if available\
                        if 'normalized_au_values' in info['left'] and 'normalized_au_values' in info['right']:\
                            if au in info['left']['normalized_au_values'] and au in info['right'][\
                                'normalized_au_values']:\
                                left_norm = info['left']['normalized_au_values'][au]\
                                right_norm = info['right']['normalized_au_values'][au]\
\
                                left_values_normalized[au] = left_norm\
                                right_values_normalized[au] = right_norm\
\
                if not left_values or not right_values:\
                    continue  # Skip if no data for this zone\
\
                # Store normalized values for access in other methods\
                self.current_normalized_values = \{\
                    'left': left_values_normalized,\
                    'right': right_values_normalized\
                \}\
\
                # Initialize contributing AU tracking for this zone if not already present\
                for side in ['left', 'right']:\
                    if zone not in info['paralysis']['contributing_aus'][side]:\
                        info['paralysis']['contributing_aus'][side][zone] = \{\
                            'minimal_movement': [],\
                            'asymmetry': [],\
                            'percent_diff': [],  # Category for percent difference\
                            'extreme_asymmetry': [],  # Category for extreme asymmetry detection\
                            'borderline_case': [],  # Category for borderline cases\
                            'normalized_ratio': [],  # Category for normalized ratio detection\
                            'combined_score': [],  # Category for combined score detection\
                            'individual_au': []  # Category for individual AU detection\
                        \}\
\
                    # Initialize confidence score for this zone\
                    if zone not in info['paralysis']['confidence'][side]:\
                        info['paralysis']['confidence'][side][zone] = 0.0\
\
                # Calculate weighted average activations with normalized values if available\
                left_avg = self._calculate_weighted_activation('left', zone, left_values, left_values_normalized)\
                right_avg = self._calculate_weighted_activation('right', zone, right_values, right_values_normalized)\
\
                # For midface detection, always use the functional approach first\
                if zone == 'mid':\
                    # Process left side\
                    left_processed = self.process_midface_functional_approach_left(\
                        info, zone, aus, left_values, right_values,\
                        left_values_normalized, right_values_normalized,\
                        # Calculate confidence score but don't let it be too restrictive\
                        max(self._calculate_confidence_score('left', zone, left_values, right_values, \{\}), 0.3),\
                        zone_paralysis, affected_aus_by_zone_side\
                    )\
\
                    # Process right side\
                    right_processed = self.process_midface_functional_approach_right(\
                        info, zone, aus, right_values, left_values,\
                        right_values_normalized, left_values_normalized,\
                        # Calculate confidence score but don't let it be too restrictive\
                        max(self._calculate_confidence_score('right', zone, right_values, left_values, \{\}), 0.3),\
                        zone_paralysis, affected_aus_by_zone_side\
                    )\
\
                    # CRITICAL FIX: DO NOT skip other detection methods for mid face\
                    # The original code had a condition like:\
                    # if left_processed and right_processed:\
                    #     continue\
                    # This was causing midface paralysis to not be properly detected\
\
                # If not midface (or if midface functional approach failed), continue with standard processing\
\
                # Calculated percent difference between sides for each AU\
                au_percent_diffs = []\
                for au in aus:\
                    if au in left_values and au in right_values:\
                        # Determine if we should use normalized values for this AU\
                        use_normalized = False\
                        au_base = au.split('_')[0] + '_r'\
                        if zone in self.au_zone_detection_modifiers and au_base in self.au_zone_detection_modifiers[\
                            zone]:\
                            use_normalized = self.au_zone_detection_modifiers[zone][au_base].get('use_normalized',\
                                                                                                 False)\
\
                        # Get the appropriate values\
                        left_val = left_values[au]\
                        right_val = right_values[au]\
\
                        if use_normalized and au in left_values_normalized and au in right_values_normalized:\
                            left_val = left_values_normalized[au]\
                            right_val = right_values_normalized[au]\
                            logger.debug(f"Using normalized values for \{au\} percent diff: L=\{left_val\}, R=\{right_val\}")\
\
                        percent_diff = calculate_percent_difference(left_val, right_val)\
                        au_percent_diffs.append(percent_diff)\
\
                # Get maximum percent difference (most asymmetric AU)\
                max_percent_diff = max(au_percent_diffs) if au_percent_diffs else 0\
\
                # Calculate asymmetry ratio for complete paralysis detection\
                ratio = 0\
                if left_avg > 0 and right_avg > 0:\
                    ratio = min(left_avg, right_avg) / max(left_avg, right_avg)\
\
                # First, check if extreme asymmetry exists and left is the stronger side\
                has_extreme_asymmetry, extreme_au, extreme_percent_diff, weaker_side, _ = self._check_for_extreme_asymmetry(\
                    left_values, right_values, left_values_normalized, right_values_normalized, zone\
                )\
\
                # Detect left side paralysis\
                detect_left_side_paralysis(\
                    self, info, zone, aus, left_values, right_values,\
                    left_values_normalized, right_values_normalized,\
                    left_avg, right_avg, zone_paralysis, affected_aus_by_zone_side,\
                    partial_thresholds, complete_thresholds, asymmetry_thresholds, confidence_thresholds\
                )\
\
                # Detect right side paralysis\
                detect_right_side_paralysis(\
                    self, info, zone, aus, left_values, right_values,\
                    left_values_normalized, right_values_normalized,\
                    left_avg, right_avg, zone_paralysis, affected_aus_by_zone_side,\
                    partial_thresholds, complete_thresholds, asymmetry_thresholds, confidence_thresholds\
                )\
\
                # Update the overall detection flag for this action if any zone shows paralysis\
                for side in ['left', 'right']:\
                    for z in ['upper', 'mid', 'lower']:\
                        if info['paralysis']['zones'][side][z] != 'None':\
                            info['paralysis']['detected'] = True\
\
            # Update patient-level paralysis information across all actions\
            for action, info in results.items():\
                # Copy the zone-level paralysis information determined across all actions\
                for side in ['left', 'right']:\
                    for zone in ['upper', 'mid', 'lower']:\
                        info['paralysis']['zones'][side][zone] = zone_paralysis[side][zone]\
\
                # Update the overall detection flag\
                for side in ['left', 'right']:\
                    for zone in ['upper', 'mid', 'lower']:\
                        if zone_paralysis[side][zone] != 'None':\
                            info['paralysis']['detected'] = True\
\
            logger.info("Completed paralysis detection")}