"""
Constants and definitions for facial AU analysis.
Contains definitions of Action Units (AUs) for different facial actions.
"""

# Define the key AUs for each action
ACTION_TO_AUS = {
    'RE': ['AU01_r', 'AU02_r'],  # Raise Eyebrows
    'ES': ['AU45_r'],            # Close Eyes Softly
    'ET': ['AU45_r'],            # Close Eyes Tightly
    'SS': ['AU12_r'],            # Soft Smile
    'BS': ['AU12_r', 'AU25_r', 'AU07_r'],  # Big Smile - Added AU07_r
    'SO': ['AU25_r', 'AU26_r'],  # Say 'O'
    'SE': ['AU12_r', 'AU25_r'],  # Say 'E'
    'BL': [],                    # Baseline
    'WN': ['AU09_r'],            # Wrinkle Nose (keep for reference, but it's not used in zone-specific actions)
    'PL': ['AU25_r'],            # Modified Pucker Lips (removed AU10_r)
    'Unknown': ['AU01_r', 'AU45_r']  # Default for unknown actions
}

# List of actions to include in analysis (excluding BL, WN, PL as requested)
INCLUDED_ACTIONS = ['RE', 'ES', 'ET', 'SS', 'BS', 'SO', 'SE']  # Removed PL and WN if they were included

# Action descriptions for labels
ACTION_DESCRIPTIONS = {
    'RE': 'Raise Eyebrows',
    'ES': 'Close Eyes Softly',
    'ET': 'Close Eyes Tightly',
    'SS': 'Soft Smile',
    'BS': 'Big Smile',
    'SO': 'Say O',
    'SE': 'Say E',
    'BL': 'Baseline',  # Changed from 'Blink' to 'Baseline'
    'WN': 'Wrinkle Nose',
    'PL': 'Pucker Lips',
    'Unknown': 'Unknown Action'
}

# Define Action Unit names for better readability
AU_NAMES = {
    'AU01_r': 'Inner Brow Raiser',
    'AU02_r': 'Outer Brow Raiser',
    'AU04_r': 'Brow Lowerer',
    'AU05_r': 'Upper Lid Raiser',
    'AU06_r': 'Cheek Raiser',
    'AU07_r': 'Lid Tightener',
    'AU09_r': 'Nose Wrinkler',
    'AU10_r': 'Upper Lip Raiser',
    'AU12_r': 'Lip Corner Puller (Smile)',
    'AU14_r': 'Dimpler',
    'AU15_r': 'Lip Corner Depressor',
    'AU17_r': 'Chin Raiser',
    'AU20_r': 'Lip Stretcher',
    'AU23_r': 'Lip Tightener',
    'AU25_r': 'Lips Part',
    'AU26_r': 'Jaw Drop',
    'AU45_r': 'Blink',
    'AU01_c': 'Inner Brow Raiser (binary)',
    'AU02_c': 'Outer Brow Raiser (binary)',
    'AU04_c': 'Brow Lowerer (binary)',
    'AU05_c': 'Upper Lid Raiser (binary)',
    'AU06_c': 'Cheek Raiser (binary)',
    'AU07_c': 'Lid Tightener (binary)',
    'AU09_c': 'Nose Wrinkler (binary)',
    'AU10_c': 'Upper Lip Raiser (binary)',
    'AU12_c': 'Lip Corner Puller (binary)',
    'AU14_c': 'Dimpler (binary)',
    'AU15_c': 'Lip Corner Depressor (binary)',
    'AU17_c': 'Chin Raiser (binary)',
    'AU20_c': 'Lip Stretcher (binary)',
    'AU23_c': 'Lip Tightener (binary)',
    'AU25_c': 'Lips Part (binary)',
    'AU26_c': 'Jaw Drop (binary)',
    'AU28_c': 'Lip Suck (binary)',
    'AU45_c': 'Blink (binary)'
}

# All AU columns to extract for analysis - only including AU_r values (ignoring AU_c)
ALL_AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
    'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
    'AU25_r', 'AU26_r', 'AU45_r'
]

# New columns for patient-level summary data
PATIENT_SUMMARY_COLUMNS = [
    'Patient ID',
    'Left Upper Face Paralysis', 'Left Mid Face Paralysis', 'Left Lower Face Paralysis',
    'Right Upper Face Paralysis', 'Right Mid Face Paralysis', 'Right Lower Face Paralysis',
    'Ocular-Oral Left', 'Ocular-Oral Right',
    'Oral-Ocular Left', 'Oral-Ocular Right',
    'Snarl-Smile Left', 'Snarl-Smile Right',
    'Paralysis Detected', 'Synkinesis Detected'
]

# Old output columns for backward compatibility
SUMMARY_COLUMNS = [
    'Patient ID', 'Action',
    'Max Side', 'Max Frame', 'Max Value',
    'Left AU01_r', 'Left AU02_r', 'Left AU04_r', 'Left AU05_r', 'Left AU06_r',
    'Left AU07_r', 'Left AU09_r', 'Left AU10_r', 'Left AU12_r', 'Left AU14_r',
    'Left AU15_r', 'Left AU17_r', 'Left AU20_r', 'Left AU23_r', 'Left AU25_r',
    'Left AU26_r', 'Left AU45_r',
    'Right AU01_r', 'Right AU02_r', 'Right AU04_r', 'Right AU05_r', 'Right AU06_r',
    'Right AU07_r', 'Right AU09_r', 'Right AU10_r', 'Right AU12_r', 'Right AU14_r',
    'Right AU15_r', 'Right AU17_r', 'Right AU20_r', 'Right AU23_r', 'Right AU25_r',
    'Right AU26_r', 'Right AU45_r',
    'Paralysis Detected', 'Paralyzed Side', 'Paralysis Severity',
    'Affected AUs', 'Synkinesis Detected', 'Synkinesis Types'
]

# Define facial zones with their corresponding AUs - UPDATED to include AU07_r in midface
FACIAL_ZONES = {
    'upper': ['AU01_r', 'AU02_r'],
    'mid': ['AU45_r', 'AU07_r'],  # Added AU07_r for midface detection
    'lower': ['AU12_r', 'AU25_r']
}

# Define weights for each AU in the zones for weighted calculation
# This allows finer control over how important each AU is for detecting paralysis
FACIAL_ZONE_WEIGHTS = {
    'upper': {'AU01_r': 0.7, 'AU02_r': 0.3},
    'mid': {'AU45_r': 0.7, 'AU07_r': 0.3},  # Updated weights to include AU07_r
    'lower': {'AU12_r': 0.56, 'AU25_r': 0.44}
}

# Define zone-specific AU detection modifiers - UPDATED with AU07_r settings
AU_ZONE_DETECTION_MODIFIERS = {
    'upper': {
        'AU01_r': {
            'asymmetry_weight': 1.2,         # Give more weight to asymmetry for eyebrow raising
            'use_normalized': True,          # ADDED: Use normalized values for upper face
            'critical_detection': True       # ADDED: Flag as critical AU for detection
        },
        'AU02_r': {
            'asymmetry_weight': 1.0,
            'use_normalized': True           # ADDED: Use normalized values for upper face
        }
    },
    'mid': {
        'AU45_r': {
            'asymmetry_weight': 1.2,         # Give more weight to asymmetry for blink
            'ignore_above_threshold': 2.0,    # If value exceeds this, don't consider it complete paralysis
            'confidence_bonus': 0.1,         # Add to confidence for AU45_r detection
            'use_normalized': True           # ADDED: Use normalized values for mid face
        },
        'AU07_r': {                          # Added settings for AU07_r
            'asymmetry_weight': 1.1,
            'ignore_above_threshold': 2.5,
            'confidence_bonus': 0.1,
            'use_normalized': True
        }
    },
    'lower': {
        'AU12_r': {
            'asymmetry_weight': 1.3,         # Increase weight on asymmetry for smiles
            'use_normalized': True,          # Always use normalized values for this AU
            'critical_detection': True       # Flag this as a critical AU for detection
        },
        'AU25_r': {
            'use_normalized': True           # Always use normalized values for this AU
        }
    }
}

# Define importance of each AU for paralysis detection by zone
PARALYSIS_DETECTION_AU_IMPORTANCE = {
    'upper': {
        'AU01_r': 1.0,  # Baseline importance
        'AU02_r': 0.8,  # Slightly less important
    },
    'mid': {
        'AU45_r': 1.0,  # Primary AU for mid face
        'AU07_r': 0.8,  # Added AU07_r with high importance
    },
    'lower': {
        'AU12_r': 1.2,
        'AU25_r': 0.9,
    }
}

# Define key actions for analyzing specific zones - EXPANDED for better coverage
ZONE_SPECIFIC_ACTIONS = {
    'upper': ['RE'],           # Raise Eyebrows
    'mid': ['ES', 'BS'],       # Close Eyes Softly, added Big Smile for AU07_r
    'lower': ['BS']            # Big Smile
}

# Threshold for extreme asymmetry in individual AUs
EXTREME_ASYMMETRY_THRESHOLD = 130.0  # Percent difference threshold to consider "extreme"

# NEW: Lower threshold specifically for AU12_r extreme asymmetry
AU12_EXTREME_ASYMMETRY_THRESHOLD = 120.0  # Reduced from 130.0 for AU12_r

# NEW: Lower threshold specifically for AU01_r extreme asymmetry
AU01_EXTREME_ASYMMETRY_THRESHOLD = 120.0  # Added for AU01_r upper face detection

# NEW: Lower threshold specifically for AU45_r extreme asymmetry
AU45_EXTREME_ASYMMETRY_THRESHOLD = 120.0  # Added for AU45_r mid face detection

# NEW: Lower threshold specifically for AU07_r extreme asymmetry
AU07_EXTREME_ASYMMETRY_THRESHOLD = 90.0   # Added for AU07_r mid face detection

# Upper face detection requires agreement between AUs
UPPER_FACE_AU_AGREEMENT_REQUIRED = True  # Require AU01_r and AU02_r to agree on which side is weaker

# Minimum threshold for functional eyelid closure
MID_FACE_FUNCTIONAL_THRESHOLD = 1.0  # Minimum AU45_r value considered functional
MID_FACE_FUNCTION_RATIO_OVERRIDE = 0.35  # More strict ratio threshold when function exists

# Component-specific thresholds for midface paralysis detection (UPDATED with AU7 thresholds)
MIDFACE_COMPONENT_THRESHOLDS = {
    'functional': {
        'complete': 0.35,  # Below this = complete paralysis
        'partial': 0.55    # Below this = partial paralysis
    },
    'asymmetry': {
        'complete': 0.30,  # Below this = complete paralysis
        'partial': 0.50    # Below this = partial paralysis
    },
    'au7': {  # New thresholds for AU7_r
        'complete': 0.40,  # Below this = complete paralysis
        'partial': 0.60    # Below this = partial paralysis
    }
}

# Legacy combined score thresholds for midface (keeping for backward compatibility)
MIDFACE_FUNCTIONAL_THRESHOLDS = {
    'complete': 0.45,  # Below this is complete paralysis
    'partial': 0.60    # Below this (but above complete) is partial paralysis
}

# New constants for combined score midface paralysis detection
MIDFACE_COMBINED_SCORE_THRESHOLDS = {
    'complete': 0.40,  # Updated for combined AU45_r and AU07_r approach
    'partial': 0.60    # Updated for combined AU45_r and AU07_r approach
}

# Flag to enable functional approach for midface
USE_FUNCTIONAL_APPROACH_FOR_MIDFACE = True

# Flag to enable dual-criteria approach for midface (NEW)
USE_DUAL_CRITERIA_FOR_MIDFACE = True

# Flag to enable combined score approach for midface
USE_COMBINED_SCORE_FOR_MIDFACE = True  # Set to True to use the new combined approach

# Updated zone-specific paralysis thresholds with REadjusted values for improved specificity
PARALYSIS_THRESHOLDS = {
       'upper': {
        'partial': {
            'asymmetry_ratio': 0.7,      # Slightly increased from 0.68
            'minimal_movement': 0.6      # Slightly reduced from 0.65
        },
        'complete': {
            'asymmetry_ratio': 0.5,      # Slightly increased from 0.48
            'minimal_movement': 0.35     # Slightly reduced from 0.38
        }
    },
    'mid': {
        'partial': {
            'asymmetry_ratio': 0.65,      # Adjusted from 0.67
            'minimal_movement': 0.8       # Increased from 0.48 to focus on significant blink reductions
        },
        'complete': {
            'asymmetry_ratio': 0.43,      # MODIFIED: Increased from 0.4 to capture border cases
            'minimal_movement': 1.0       # MODIFIED: Increased from 0.4 to better match expert labels
        }
    },
    'lower': {
        'partial': {
            'asymmetry_ratio': 0.65,      # Unchanged from previous adjustment
            'minimal_movement': 2.0       # Unchanged from previous adjustment
        },
        'complete': {
            'asymmetry_ratio': 0.4,       # MODIFIED: Decreased from 0.45 to improve detection
            'minimal_movement': 0.8       # MODIFIED: Decreased from 1.0 to catch cases with some movement
        }
    },
}

# Adjusted asymmetry thresholds for improved partial paralysis detection and specificity - UPDATED
ASYMMETRY_THRESHOLDS = {
    'upper': {
        'partial': {'percent_diff': 55},  # Adjusted from 58
        'complete': {'ratio': 0.40}       # MODIFIED: Reduced from 0.45 to improve upper face detection
    },
    'mid': {
        'partial': { 'percent_diff': 40 },      # Reduced from 50 to increase sensitivity
        'complete': { 'ratio': 0.43 }           # MODIFIED: Increased from 0.4 to capture border cases
    },
     'lower': {
        'partial': { 'percent_diff': 55 },  # Reduced from 60
        'complete': { 'ratio': 0.5 }        # Increased from 0.45
    },
}

# Confidence thresholds for improved detection reliability - UPDATED
CONFIDENCE_THRESHOLDS = {
     'upper': {
        'partial': 0.5,    # Unchanged
        'complete': 0.37   # MODIFIED: Reduced from 0.38 to catch more borderline cases in upper face
    },
     'mid': {
        'partial': 0.45,    # Unchanged
        'complete': 0.35    # Unchanged (already modified from 0.4)
    },
        'lower': {
        'partial': 0.45,    # Reduced from 0.55
        'complete': 0.35    # Reduced from 0.38
    },
}

# Old general thresholds (kept for backward compatibility)
PARALYSIS_THRESHOLD = 0.4  # Significant asymmetry suggesting paralysis
MINIMAL_MOVEMENT_THRESHOLD = 0.2  # Threshold below which movement is considered minimal/absent

LOWER_FACE_COMBINED_SCORE_THRESHOLDS = {
    'complete': 0.57,  # Below this is complete paralysis (unchanged)
    'partial': 0.90    # Increased from 0.86 to catch borderline cases like IMG_3137
}

# Flag to enable combined score approach for lower face
USE_COMBINED_SCORE_FOR_LOWER_FACE = True  # Set to True to enable the new approach

LOWER_FACE_PARALYSIS_THRESHOLDS = {
    'partial': {
        'asymmetry_ratio': 0.70,      # Increased from 0.65 to capture more partial cases
        'minimal_movement': 1.5       # Reduced from 2.0 to better detect partial cases
    },
    'complete': {
        'asymmetry_ratio': 0.45,      # Slightly increased from 0.4 based on data analysis
        'minimal_movement': 0.8       # Unchanged
    }
}

# Constants for midface combined score approach
MIDFACE_COMBINED_SCORE_THRESHOLDS = {
    'complete': 0.40,  # Below this is complete paralysis
    'partial': 0.60    # Below this (but above complete) is partial paralysis
}

# Flag to enable combined score approach for midface
USE_COMBINED_SCORE_FOR_MIDFACE = True

LOWER_FACE_ASYMMETRY_THRESHOLDS = {
    'partial': {'percent_diff': 40},  # Reduced from 55 to increase sensitivity
    'complete': {'ratio': 0.45}       # Unchanged
}

LOWER_FACE_CONFIDENCE_THRESHOLDS = {
    'partial': 0.35,    # Reduced from 0.45 to catch more partial cases
    'complete': 0.35    # Unchanged
}

# Updated synkinesis detection thresholds for better specificity - SIGNIFICANTLY INCREASED VALUES TO REDUCE FALSE POSITIVES
SYNKINESIS_THRESHOLDS = {
    'Ocular-Oral': {
        'trigger': 1.5,    # Increased more (was 1.2)
        'coupled': 1.0,    # Increased more (was 0.8)
        'ratio_lower': 0.4,
        'ratio_upper': 1.0,  # Reduced range (was 1.2)
        # New minimum co-occurrence requirement
        'min_coupled_aus': 2  # Require at least 2 coupled AUs to exceed threshold
    },
    'Oral-Ocular': {
        'trigger': 1.8,    # Significantly increased (was 1.5)
        'coupled': 1.2,    # Significantly increased (was 1.0)
        'ratio_lower': 0.5,
        'ratio_upper': 0.9,  # Narrower range (was 1.0)
        # New minimum co-occurrence requirement
        'min_coupled_aus': 2  # Require at least 2 coupled AUs to exceed threshold
    },
    'Snarl-Smile': {
        'trigger': 2.0,    # Significantly increased (was 1.5)
        # Weighted approach
        'AU09_r': 0.8,     # Significantly increased (was 0.5)
        'AU10_r': 0.8,     # Significantly increased (was 0.5)
        'AU14_r': 1.2,     # Significantly increased (was 0.9)
        # Weights for each AU in the weighted score calculation
        'AU09_weight': 0.4,  # Nose wrinkle importance
        'AU10_weight': 0.3,  # Upper lip raiser importance
        'AU14_weight': 0.3,  # Dimpler importance
        # Weighted score threshold
        'weighted_threshold': 0.5  # Minimum weighted score to detect snarl-smile
    }
}

# Define synkinesis patterns to detect - updated with improved definitions
# Removed Midface and Mentalis as requested
SYNKINESIS_PATTERNS = {
    'Ocular-Oral': {
        'trigger_aus': ['AU45_r', 'AU01_r', 'AU02_r', 'AU07_r'],  # Eye closure and eyebrow AUs
        'coupled_aus': ['AU12_r', 'AU25_r', 'AU14_r'],  # Mouth movement AUs
        'description': 'Eye actions cause unwanted mouth movement',
        'relevant_actions': ['ET', 'ES', 'RE', 'BL']  # Eye/brow actions
    },
    'Oral-Ocular': {
        'trigger_aus': ['AU12_r', 'AU25_r'],  # Mouth movement AUs
        'coupled_aus': ['AU45_r', 'AU07_r', 'AU06_r'],  # Added AU06_r (Cheek Raiser)
        'description': 'Mouth movement causes unwanted eye narrowing/closure',
        'relevant_actions': ['BS', 'SS', 'SO', 'SE']  # Mouth actions
    },
    'Snarl-Smile': {
        'trigger_aus': ['AU12_r', 'AU25_r'],  # Smile AUs
        'coupled_aus': ['AU09_r', 'AU10_r', 'AU14_r'],  # Nose wrinkler, upper lip raiser, dimpler
        'description': 'Smile causes unwanted nose wrinkling and upper lip raising (snarl)',
        'relevant_actions': ['BS', 'SS']  # Smile actions
    }
}

# Baseline AU activations for neutral expressions - used for normalization
# These are typical values seen in neutral expressions (to be subtracted from measured values)
BASELINE_AU_ACTIVATIONS = {
    'AU01_r': 0.1,  # Some people naturally have slightly raised inner brows
    'AU02_r': 0.1,  # Some people naturally have slightly raised outer brows
    'AU04_r': 0.1,  # Some natural brow lowering
    'AU05_r': 0.2,  # Natural upper lid raising
    'AU06_r': 0.1,  # Some cheek raising at rest
    'AU07_r': 0.2,  # Some lid tightening at rest
    'AU09_r': 0.1,  # Some natural nose wrinkling
    'AU10_r': 0.15, # Some upper lip raising at rest
    'AU12_r': 0.2,  # Natural smile at rest
    'AU14_r': 0.15, # Natural dimpler at rest
    'AU15_r': 0.1,  # Some lip corner depression
    'AU17_r': 0.1,  # Some chin raising
    'AU20_r': 0.1,  # Some lip stretching
    'AU23_r': 0.1,  # Some lip tightening
    'AU25_r': 0.2,  # Slight lips parting at rest
    'AU26_r': 0.1,  # Some jaw dropping
    'AU45_r': 0.1   # Slight blinking
}

# List of all synkinesis types for consistency
SYNKINESIS_TYPES = list(SYNKINESIS_PATTERNS.keys())

# Severity options for paralysis
PARALYSIS_SEVERITY_LEVELS = ['None', 'Partial', 'Complete']
