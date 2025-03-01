# config.py - Configuration values and constants

# Action mappings with codes and descriptions
ACTION_MAPPINGS = {
    "RE": "Raise Eyebrows",
    "ES": "Close Eyes Softly",
    "ET": "Close Eyes Tightly",
    "SS": "Soft Smile",
    "BS": "Big Smile",
    "SO": "Say 'O'",
    "SE": "Say 'E'",
    "BL": "Blink",
    "WN": "Wrinkle Nose",
    "PL": "Pucker Lips"
}

# Colors for the UI buttons in different states
BUTTON_COLORS = {
    "normal": "#e0e0e0",  # Darker grey for better contrast
    "pressed": "#4a86e8",  # Brighter blue
    "text_normal": "#000000",  # Black text on normal button
    "text_pressed": "#ffffff"  # White text on pressed button
}

# Font settings for video overlay
OVERLAY_FONT = {
    "font": "Arial",
    "size": 24,
    "color": (255, 255, 255),  # White
    "thickness": 2,
    "background_color": (0, 0, 0, 128),  # Semi-transparent black
    "position": (10, 30)  # Top-left corner (x, y)
}
