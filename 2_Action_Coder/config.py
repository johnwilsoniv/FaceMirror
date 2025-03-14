# config.py - Configuration values and constants
from PyQt5.QtGui import QFont  # Add this import at the top

# Standard GUI styling
SECTION_TITLE_FONT = ('Arial', 9)  # Font family and size for section headers
SECTION_TITLE_WEIGHT = QFont.Bold  # Weight constant for section headers
STANDARD_MARGIN = 8  # Standard margin value for consistent spacing
STANDARD_SPACING = 5  # Standard spacing between elements

# Standard colors for consistent UI
UI_COLORS = {
    'section_bg': '#f5f5f5',       # Light gray for section backgrounds
    'section_border': '#cccccc',   # Medium gray for borders
    'highlight': '#4682B4',        # Steel blue for highlights/focus
    'text_normal': '#333333',      # Dark gray for normal text
    'text_inactive': '#999999',    # Medium gray for inactive/placeholder text
}

# Standard button styling (non-action buttons)
STANDARD_BUTTON_STYLE = """
    QPushButton {
        background-color: #f0f0f0;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 3px;
        padding: 4px 8px;
        min-height: 22px;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
    }
    QPushButton:pressed {
        background-color: #d0d0d0;
        border: 1px solid #999999;
    }
"""

PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #4682B4;
        color: white;
        border: 1px solid #3a6a94;
        border-radius: 3px;
        padding: 4px 10px;
        min-height: 30px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #5692c4;
    }
    QPushButton:pressed {
        background-color: #3a6a94;
    }
"""

# Consistent GroupBox styling
GROUP_BOX_STYLE = """
    QGroupBox {
        font-family: Arial;
        font-weight: bold;
        border: 1px solid #cccccc;
        border-radius: 3px;
        margin-top: 12px;
        padding-top: 8px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 3px;
    }
"""
# Action mappings with codes and descriptions
ACTION_MAPPINGS = {
    "RE": "Raise Eyebrows",
    "ES": "Close Eyes Softly",
    "ET": "Close Eyes Tightly",
    "SS": "Soft Smile",
    "BS": "Big Smile",
    "SO": "Say 'O'",
    "SE": "Say 'E'",
    "BL": "Baseline"          # Kept Baseline with key 0
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
