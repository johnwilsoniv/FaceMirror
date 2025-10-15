
# facial_au_constants.py

"""
Constants and definitions for facial AU analysis.
Contains definitions of Action Units (AUs) for different facial actions.
Focused on paralysis detection only.
V1.14 Update: Refined ACTION_TO_AUS for peak frame finding for all actions.
              Added descriptions for new actions.
              Removed INCLUDED_ACTIONS dependency (analyzer handles this).
V1.15 Update: Changed key AU for 'PL' from AU23 to AU17 based on analysis.
V1.19 Update: Removed all synkinesis and hypertonicity detection code.
"""
import logging
import pandas as pd
import re

logger = logging.getLogger(__name__)

def standardize_paralysis_label(val):
    """Standardizes paralysis labels to 'None', 'Partial', 'Complete'."""
    if val is None or pd.isna(val): return 'None'
    val_str = str(val).strip().lower()
    if val_str in ['none', 'no', 'n/a', '0', '0.0', 'normal', '', 'nan']: return 'None'
    if val_str in ['partial', 'mild', 'moderate', '1', '1.0', 'p', 'incomplete', 'i']: return 'Partial' # Added incomplete/i
    if val_str in ['complete', 'severe', '2', '2.0', 'c']: return 'Complete'
    logger.debug(f"Unexpected paralysis label: '{val}'. Defaulting to 'None'.")
    return 'None'

def standardize_binary_label(val):
    """Standardizes binary labels to 'Yes', 'No'."""
    if val is None or pd.isna(val): return 'No'
    if isinstance(val, bool): return 'Yes' if val else 'No'
    if isinstance(val, (int, float)):
        if val == 1: return 'Yes'
        if val == 0: return 'No'
    val_str = str(val).strip().lower()
    if val_str in ['yes', 'true', '1', '1.0', 'y']: return 'Yes'
    if val_str in ['no', 'none', 'false', '0', '0.0', 'n', '', 'nan']: return 'No'
    logger.debug(f"Unexpected binary label: '{val}'. Defaulting to 'No'.")
    return 'No'

ACTION_TO_AUS = {
    'RE': ['AU01_r', 'AU02_r'],  # Raise Eyebrows
    'ES': ['AU45_r'],            # Close Eyes Softly
    'ET': ['AU06_r', 'AU45_r'],  # Close Eyes Tightly (AU07→AU06 for OpenFace 3.0)
    'SS': ['AU12_r'],            # Soft Smile
    'BS': ['AU12_r', 'AU25_r', 'AU06_r'], # Big Smile (AU07→AU06 for OpenFace 3.0)
    'SO': ['AU25_r'],            # Say 'O' (AU26 not available in OpenFace 3.0)
    'SE': ['AU20_r', 'AU25_r'],  # Say 'E'
    'BL': [],                    # Baseline (Uses median frame logic)
    'FR': ['AU04_r'],            # Frown
    'BK': ['AU45_r'],            # Blink
    'WN': ['AU06_r'],            # Wrinkle Nose (AU09→AU06 for OpenFace 3.0)
    'PL': [],                    # Pucker Lips (AU17 not available - use median frame)
    'BC': [],                    # Blow Cheeks (AU23 not available - use median frame)
    'LT': ['AU15_r'],            # Lower Teeth (AU16→AU15 for OpenFace 3.0)
    'Unknown': []                # Default for unknown actions
}

ACTION_DESCRIPTIONS = {
    'RE': 'Raise Eyebrows',
    'ES': 'Close Eyes Softly',
    'ET': 'Close Eyes Tightly',
    'SS': 'Soft Smile',
    'BS': 'Big Smile',
    'SO': 'Say O',
    'SE': 'Say E',
    'BL': 'Baseline',
    'FR': 'Frown',             # New
    'BK': 'Blink',             # New
    'WN': 'Wrinkle Nose',
    'PL': 'Pucker Lips',
    'BC': 'Blow Cheeks',       # New
    'LT': 'Lower Teeth',       # New
    'Unknown': 'Unknown Action'
}

# Define Action Unit names for better readability (Unchanged)
AU_NAMES = {
    'AU01_r': 'Inner Brow Raiser', 'AU02_r': 'Outer Brow Raiser', 'AU04_r': 'Brow Lowerer',
    'AU05_r': 'Upper Lid Raiser', 'AU06_r': 'Cheek Raiser', 'AU07_r': 'Lid Tightener',
    'AU09_r': 'Nose Wrinkler', 'AU10_r': 'Upper Lip Raiser', 'AU12_r': 'Lip Corner Puller (Smile)',
    'AU14_r': 'Dimpler', 'AU15_r': 'Lip Corner Depressor', 'AU16_r': 'Lower Lip Depressor', # Added AU16 name
    'AU17_r': 'Chin Raiser',
    'AU20_r': 'Lip Stretcher', 'AU23_r': 'Lip Tightener', 'AU25_r': 'Lips Part',
    'AU26_r': 'Jaw Drop', 'AU45_r': 'Blink',
    'AU01_c': 'Inner Brow Raiser (binary)', 'AU02_c': 'Outer Brow Raiser (binary)', 'AU04_c': 'Brow Lowerer (binary)',
    'AU05_c': 'Upper Lid Raiser (binary)', 'AU06_c': 'Cheek Raiser (binary)', 'AU07_c': 'Lid Tightener (binary)',
    'AU09_c': 'Nose Wrinkler (binary)', 'AU10_c': 'Upper Lip Raiser (binary)', 'AU12_c': 'Lip Corner Puller (binary)',
    'AU14_c': 'Dimpler (binary)', 'AU15_c': 'Lip Corner Depressor (binary)', 'AU16_c': 'Lower Lip Depressor (binary)',
    'AU17_c': 'Chin Raiser (binary)',
    'AU20_c': 'Lip Stretcher (binary)', 'AU23_c': 'Lip Tightener (binary)', 'AU25_c': 'Lips Part (binary)',
    'AU26_c': 'Jaw Drop (binary)', 'AU28_c': 'Lip Suck (binary)', 'AU45_c': 'Blink (binary)'
}

# All AU columns to extract for analysis - ensure all potentially relevant AUs are here
ALL_AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU16_r', 'AU17_r', 'AU20_r', 'AU23_r',
    'AU25_r', 'AU26_r', 'AU45_r'
]

# Synkinesis and hypertonicity detection removed - paralysis detection only

PARALYSIS_SEVERITY_LEVELS = ['None', 'Partial', 'Complete', 'Error']
SEVERITY_ABBREVIATIONS = { 'None': 'N', 'Partial': 'I', 'Complete': 'C', 'Error': 'E' }
SEVERITY_ABBREVIATIONS_CONTRADICTION = { 'None': '(N)', 'Partial': '(I)', 'Complete': '(C)', 'Error': '(E)' }

USE_ML_FOR_UPPER_FACE = True
USE_ML_FOR_MIDFACE = True
USE_ML_FOR_LOWER_FACE = True

# Expert key mapping for paralysis detection only
EXPERT_KEY_MAPPING = {
    'Left Upper Face Paralysis': 'Paralysis - Left Upper Face',
    'Left Mid Face Paralysis': 'Paralysis - Left Mid Face',
    'Left Lower Face Paralysis': 'Paralysis - Left Lower Face',
    'Right Upper Face Paralysis': 'Paralysis - Right Upper Face',
    'Right Mid Face Paralysis': 'Paralysis - Right Mid Face',
    'Right Lower Face Paralysis': 'Paralysis - Right Lower Face',
}

PARALYSIS_FINDINGS_KEYS = [k for k in EXPERT_KEY_MAPPING if 'Paralysis' in k]
BOOL_FINDINGS_KEYS = [k for k in EXPERT_KEY_MAPPING if 'Paralysis' not in k]

FACIAL_ZONES = {
    'upper': ['AU01_r', 'AU02_r'],
    'mid': ['AU45_r', 'AU06_r'],
    'lower': ['AU12_r', 'AU25_r']
}

FACIAL_ZONE_WEIGHTS = {
    'upper': {'AU01_r': 0.7, 'AU02_r': 0.3},
    'mid': {'AU45_r': 0.7, 'AU06_r': 0.3},
    'lower': {'AU12_r': 0.56, 'AU25_r': 0.44}
}

AU_ZONE_DETECTION_MODIFIERS = {
    'upper': {
        'AU01_r': {'asymmetry_weight': 1.2, 'use_normalized': True, 'critical_detection': True},
        'AU02_r': {'asymmetry_weight': 1.0, 'use_normalized': True}
    },
    'mid': {
        'AU45_r': {'asymmetry_weight': 1.2, 'ignore_above_threshold': 2.0, 'confidence_bonus': 0.1, 'use_normalized': True},
        'AU06_r': {'asymmetry_weight': 1.0, 'use_normalized': True, 'confidence_bonus': 0.1}
    },
    'lower': {
        'AU12_r': {'asymmetry_weight': 1.3, 'use_normalized': True, 'critical_detection': True},
        'AU25_r': {'use_normalized': True}
    }
}

PARALYSIS_DETECTION_AU_IMPORTANCE = {
    'upper': {'AU01_r': 1.0, 'AU02_r': 0.8},
    'mid': {'AU45_r': 1.0, 'AU06_r': 0.8},
    'lower': {'AU12_r': 1.2, 'AU25_r': 0.9}
}

ZONE_SPECIFIC_ACTIONS = {
    'upper': ['RE'], #Remove FR
    'mid': ['ES', 'ET', 'BK'], #Remove WN
    'lower': ['BS', 'SS', 'SE'] #Remove SO, PL, BC, LT
}

PARALYSIS_THRESHOLDS = {
    'upper': {'partial': {'asymmetry_ratio': 0.7, 'minimal_movement': 0.6}, 'complete': {'asymmetry_ratio': 0.5, 'minimal_movement': 0.35}},
    'mid': {'partial': {'asymmetry_ratio': 0.65, 'minimal_movement': 0.8}, 'complete': {'asymmetry_ratio': 0.43, 'minimal_movement': 1.0}},
    'lower': {'partial': {'asymmetry_ratio': 0.65, 'minimal_movement': 2.0}, 'complete': {'asymmetry_ratio': 0.4, 'minimal_movement': 0.8}}
}

ASYMMETRY_THRESHOLDS = {
    'upper': {'partial': {'percent_diff': 55}, 'complete': {'ratio': 0.40}},
    'mid': {'partial': { 'percent_diff': 40 }, 'complete': { 'ratio': 0.43 }},
    'lower': {'partial': { 'percent_diff': 55 }, 'complete': { 'ratio': 0.5 }}
}

CONFIDENCE_THRESHOLDS = {
    'upper': {'partial': 0.5, 'complete': 0.37},
    'mid': {'partial': 0.45, 'complete': 0.35},
    'lower': {'partial': 0.45, 'complete': 0.35}
}

BASELINE_AU_ACTIVATIONS = {
    'AU01_r': 0.1, 'AU02_r': 0.1, 'AU04_r': 0.1, 'AU05_r': 0.2, 'AU06_r': 0.1,
    'AU07_r': 0.2, 'AU09_r': 0.1, 'AU10_r': 0.15, 'AU12_r': 0.2, 'AU14_r': 0.15,
    'AU15_r': 0.1, 'AU16_r': 0.1, 'AU17_r': 0.1, 'AU20_r': 0.1, 'AU23_r': 0.1, 'AU25_r': 0.2,
    'AU26_r': 0.1, 'AU45_r': 0.1
}

PATIENT_SUMMARY_COLUMNS = [
    'Patient ID',
    'Left Upper Face Paralysis', 'Left Mid Face Paralysis', 'Left Lower Face Paralysis',
    'Right Upper Face Paralysis', 'Right Mid Face Paralysis', 'Right Lower Face Paralysis',
    'Paralysis Detected',
]
