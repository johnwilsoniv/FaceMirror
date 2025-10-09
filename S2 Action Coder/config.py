# --- START OF FILE config.py ---

# config.py - Configuration values and constants
from PyQt5.QtGui import QFont, QColor # Added QColor

# Standard GUI styling
SECTION_TITLE_FONT = ('Arial', 9); SECTION_TITLE_WEIGHT = QFont.Bold; STANDARD_MARGIN = 8; STANDARD_SPACING = 5
UI_COLORS = {
    'section_bg': '#f5f5f5',
    'section_border': '#cccccc',
    'highlight': '#4682B4', # Standard blue highlight for active actions (incl. BL)
    'text_normal': '#333333',
    'text_inactive': '#999999',
    'neutral_status_bg': '#e9ecef', # Light grey for "Uncoded" status display background
    'timeline_bl_color': '#dddddd', # Color for Baseline range on timeline
    'timeline_track_bg': '#ffffff', # Background of the timeline track itself (where no ranges are)
    # --- New Colors ---
    'timeline_tbc_color': '#fff3cd', # Light yellow for "To Be Coded"
    'timeline_nm_color': '#cfe2ff',  # Light blue for "Near Miss" placeholder
    'timeline_confirm_needed_border': '#dc3545', # Red border for confirmation needed
    'timeline_tbc_nm_text': '#6c757d', # Grey text for ??
}
STANDARD_BUTTON_STYLE = "QPushButton { background-color: #f0f0f0; color: #333333; border: 1px solid #cccccc; border-radius: 3px; padding: 4px 8px; min-height: 22px; } QPushButton:hover { background-color: #e0e0e0; } QPushButton:pressed { background-color: #d0d0d0; border: 1px solid #999999; }"
PRIMARY_BUTTON_STYLE = "QPushButton { background-color: #4682B4; color: white; border: 1px solid #3a6a94; border-radius: 3px; padding: 4px 10px; min-height: 30px; font-weight: bold; } QPushButton:hover { background-color: #5692c4; } QPushButton:pressed { background-color: #3a6a94; }"
# --- NEW PENDING STYLE ---
PENDING_BUTTON_STYLE = "QPushButton { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; border-radius: 3px; padding: 4px 8px; min-height: 22px; font-weight: bold; } QPushButton:hover { background-color: #bce0e8; } QPushButton:pressed { background-color: #a6d5dd; border: 1px solid #9acdd7; }"
# --- END NEW ---
GROUP_BOX_STYLE = "QGroupBox { font-family: Arial; font-weight: bold; border: 1px solid #cccccc; border-radius: 3px; margin-top: 12px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 3px; }"
CONFIRMATION_BUTTON_STYLE = STANDARD_BUTTON_STYLE
DISCARD_BUTTON_STYLE = "QPushButton { background-color: #d9534f; color: white; border: 1px solid #d43f3a; border-radius: 3px; padding: 4px 8px; min-height: 22px; font-weight: bold; } QPushButton:hover { background-color: #c9302c; } QPushButton:pressed { background-color: #ac2925; }"
DELETE_BUTTON_STYLE = DISCARD_BUTTON_STYLE # Use same style for delete
DISABLED_BUTTON_STYLE = "QPushButton { background-color: #f8f8f8; color: #bbbbbb; border: 1px solid #e0e0e0; border-radius: 3px; padding: 4px 8px; min-height: 22px; }"
DIMMED_UTILITY_BUTTON_STYLE = "QPushButton { background-color: #e8e8e8; color: #aaaaaa; border: 1px solid #d8d8d8; border-radius: 3px; padding: 4px 8px; min-height: 22px; }"
SLIDER_STYLE = "QSlider::groove:horizontal { border: 1px solid #bbb; background: white; height: 8px; border-radius: 4px; } QSlider::sub-page:horizontal { background: qlineargradient(x1:0, y1:0.2, x2:1, y2:1, stop:0 #4682B4, stop:1 #5692c4); border: 1px solid #444; height: 10px; border-radius: 4px; } QSlider::add-page:horizontal { background: #e0e0e0; border: 1px solid #777; height: 10px; border-radius: 4px; } QSlider::handle:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc); border: 1px solid #777; width: 15px; margin-top: -4px; margin-bottom: -4px; border-radius: 5px; }"


# Action mappings (BL is a normal action)
ACTION_MAPPINGS = {
    "RE": "Raise Eyebrows", "ES": "Close Eyes Softly", "ET": "Close Eyes Tightly", "SS": "Soft Smile", "BS": "Big Smile", "SO": "Say 'O'", "SE": "Say 'E'", "BL": "Baseline", # Baseline is now a standard action
    "FR": "Frown", "BK": "Blink", "WN": "Wrinkle Nose", "PL": "Pucker Lips", "BC": "Blow Cheeks", "LT": "Lower Teeth",
    "SO_SE": "Say 'O' then 'E' (Internal Code)" # Internal code for split logic
}
# --- Add internal codes (won't appear in buttons) ---
ACTION_MAPPINGS["TBC"] = "To Be Coded" # Internal placeholder
ACTION_MAPPINGS["NM"] = "Near Miss" # Internal placeholder


# Keyboard shortcuts (Ensure BL has '0')
MANUAL_KEY_SHORTCUTS = {
    "RE": "1", "ES": "2", "ET": "3", "SS": "4", "BS": "5", "SO": "6", "SE": "7", "BL": "0", # Verified BL shortcut
    "FR": "Q", "BK": "W", "WN": "E", "PL": "R", "BC": "T", "LT": "Y"
}

# Voice Command Mappings (No change needed for BL/Uncoded)
VOICE_COMMAND_MAPPINGS = {
    "raise your eyebrows": "RE", "eyebrows up": "RE", "raise eyebrows": "RE", "raise your eyes": "RE",
    "frown": "FR", "frown with your eyebrows": "FR", "make an angry face": "FR", "fran": "FR", "front": "FR", "round for me": "FR",
    "blink": "BK", "blink eyes": "BK", "blink a few times": "BK", "blake a few times": "BK",
    "close your eyes softly": "ES", "soft close": "ES", "gently close eyes": "ES", "close your eyes": "ES", "close eyes softly": "ES", "close your eyes shut": "ES", "close your eyes gently": "ES",
    "close your eyes tightly": "ET", "tight close": "ET", "squeeze eyes": "ET", "close eyes tight": "ET", "close your eyes tight tight tight tight tight": "ET", "close your eyes tight tight tight": "ET", "close your eyes tight tight": "ET",
    "take take take take": "ET", "take take take take take": "ET",
    "tight tight tight tight": "ET", "close eyes firmly": "ET",
    "wrinkle your nose": "WN", "scrunch nose": "WN", "wrinkle nose": "WN", "break all your nose": "WN",
    "pucker lips": "PL", "kiss": "PL", "purse lips": "PL", "purse your lips": "PL", "first lips": "PL", "press your lips": "PL", "push your lips": "PL", "pertial lips": "PL",
    "first your lips": "PL", "Brush your lips": "PL",
    "blow your cheeks": "BC", "puff cheeks": "BC", "blow your cheeks open like a blowfish": "BC", "blow your cheeks open": "BC", "blow your cheeks open like a bluefish": "BC", "pop your cheeks open": "BC",
    "blue cheeks open": "BC", "puff your cheeks": "BC",
    "soft smile": "SS", "slight smile": "SS", "gentle smile": "SS", "give me a gentle smile": "SS", "small smile":"SS","smile softly":"SS", "smile gently": "SS",
    "big smile": "BS", "show teeth smile": "BS", "give me a big smile": "BS", "give me a big smile show me your teeth": "BS", "show me your teeth": "BS", "give me a big smile show teeth": "BS", "show me all your teeth": "BS", "show you all your teeth": "BS",
    "smile with teeth": "BS",
    "say o": "SO", "oh": "SO",
    "say e": "SE", "eee": "SE", "e": "SE",
    "o e": "SO_SE",
    "say o e": "SO_SE",
    "say oee": "SO_SE",
    "oee": "SO_SE",
    "say o e e": "SO_SE",
    "o e e": "SO_SE",
    "oe": "SO_SE",
    "lower teeth": "LT", "show bottom teeth": "LT", "pirate teeth": "LT", "show me your bottom teeth": "LT", "show me your bottom teeth only": "LT",
    "and pirate": "LT", "like a pirate": "LT", "pirate": "LT",
    "open": "STOP",
    "open your eyes": "STOP",
    "relax": "STOP", "stop": "STOP", "okay": "STOP", "ok": "STOP", "great": "STOP", "good": "STOP", "neutral": "STOP", "done": "STOP", "back to normal": "STOP", "finished": "STOP",
}

# Command Identification Thresholds (No change)
FUZZY_SEQUENCE_THRESHOLD = 91
NEAR_MISS_THRESHOLD = 70
HIGH_CONFIDENCE_THRESHOLD = 95
STRICT_MATCH_THRESHOLD = 95 # Threshold for fuzz.ratio for SO_SE config phrases

# ----- SANITY CHECK ----- (No change)
if NEAR_MISS_THRESHOLD >= FUZZY_SEQUENCE_THRESHOLD:
     print(f"CONFIG WARNING: NEAR_MISS_THRESHOLD ({NEAR_MISS_THRESHOLD}) should be LESS than FUZZY_SEQUENCE_THRESHOLD ({FUZZY_SEQUENCE_THRESHOLD})!")

# UI Colors / Fonts (No change)
BUTTON_COLORS = { "normal": "#e0e0e0", "pressed": "#4a86e8", "text_normal": "#000000", "text_pressed": "#ffffff" }
OVERLAY_FONT = { "font": "Arial", "size": 24, "color": (255, 255, 255), "thickness": 2, "background_color": (0, 0, 0, 128) }

# Merge specific settings (No change)
MERGE_TIME_THRESHOLD_SECONDS = 0.25
MERGE_PAIRS = { ("SS", "BS"): "BS" }

# Timeline Interaction Settings
PAUSE_SETTLE_DELAY_MS = 150 # Milliseconds to wait after pause+seek before showing prompt UI

# --- END OF config.py ---