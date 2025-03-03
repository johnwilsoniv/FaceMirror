"""
Constants and definitions for facial AU analysis.
Contains definitions of Action Units (AUs) for different facial actions.
"""

# Define the key AUs for each action
ACTION_TO_AUS = {
    'RE': ['AU01_r', 'AU02_r'],  # Raise Eyebrows
    'ES': ['AU45_r'],            # Close Eyes Softly (removed AU07_r)
    'ET': ['AU45_r'],            # Close Eyes Tightly (removed AU07_r)
    'SS': ['AU12_r'],            # Soft Smile
    'BS': ['AU12_r', 'AU25_r'],  # Big Smile (added AU25_r)
    'SO': ['AU25_r', 'AU26_r'],  # Say 'O' (mouth opening)
    'SE': ['AU12_r', 'AU25_r'],  # Say 'E'
    'BL': ['AU45_r', 'AU07_r'],  # Blink
    'WN': ['AU09_r'],            # Wrinkle Nose
    'PL': ['AU25_r', 'AU10_r'],  # Pucker Lips (using AU10 upper lip raiser + AU25 lips part)
    'Unknown': ['AU01_r', 'AU45_r']  # Default for unknown actions
}

# Action descriptions for labels
ACTION_DESCRIPTIONS = {
    'RE': 'Raise Eyebrows',
    'ES': 'Close Eyes Softly',
    'ET': 'Close Eyes Tightly',
    'SS': 'Soft Smile',
    'BS': 'Big Smile',
    'SO': 'Say O',
    'SE': 'Say E',
    'BL': 'Blink',
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

# Output columns for summary CSV - removed Description, Key AUs, Symmetry Ratio, Asymmetry Detected, and Synkinesis Details
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
    'Paralysis Detected', 'Paralyzed Side',
    'Affected AUs', 'Synkinesis Detected', 'Synkinesis Types'
]

# Threshold values for paralysis detection
PARALYSIS_THRESHOLD = 0.4  # Significant asymmetry suggesting paralysis
MINIMAL_MOVEMENT_THRESHOLD = 0.2  # Threshold below which movement is considered minimal/absent

# Define synkinesis patterns to detect - updated with improved definitions
SYNKINESIS_PATTERNS = {
    'Ocular-Oral': {
        'trigger_aus': ['AU45_r', 'AU01_r', 'AU02_r', 'AU07_r'],  # Eye closure and eyebrow AUs
        'coupled_aus': ['AU12_r', 'AU25_r', 'AU14_r'],  # Mouth movement AUs
        'description': 'Eye actions cause unwanted mouth movement'
    },
    'Oral-Ocular': {
        'trigger_aus': ['AU12_r', 'AU25_r'],  # Mouth movement AUs
        'coupled_aus': ['AU45_r', 'AU07_r', 'AU06_r'],  # Added AU06_r (Cheek Raiser)
        'description': 'Mouth movement causes unwanted eye narrowing/closure'
    },
    'Midface': {
        'trigger_aus': ['AU45_r', 'AU01_r', 'AU02_r', 'AU07_r', 'AU12_r', 'AU25_r'],  # Eye and mouth AUs
        'coupled_aus': ['AU14_r'],  # Dimpler
        'description': 'Various facial actions cause unwanted dimpler activation'
    },
    'Mentalis': {
        'trigger_aus': ['AU45_r', 'AU01_r', 'AU02_r', 'AU07_r', 'AU12_r', 'AU25_r'],  # Eye and mouth AUs
        'coupled_aus': ['AU17_r'],  # Chin Raiser
        'description': 'Various facial actions cause unwanted chin raiser activation'
    },
    'Snarl-Smile': {
        'trigger_aus': ['AU12_r', 'AU25_r'],  # Smile AUs
        'coupled_aus': ['AU09_r', 'AU10_r', 'AU14_r'],  # Nose wrinkler, upper lip raiser, dimpler
        'description': 'Smile causes unwanted nose wrinkling and upper lip raising (snarl)'
    }
}
