# text_overlay.py - Common text overlay functionality for video output
import cv2
import numpy as np
import config

def add_text_overlay(frame, action_code, position="bottom-center"):
    """
    Add text overlay to a frame with semi-transparent background.
    Used for the output video processing.
    
    Args:
        frame: The video frame to modify
        action_code: The action code to display
        position: Where to place the overlay ("bottom-center", "top-center", etc.)
    
    Returns:
        The frame with overlay added
    """
    if not action_code:
        return frame
    
    # Make a copy to avoid modifying the original unexpectedly
    result = frame.copy()
    
    # Get action text from config
    action_text = config.ACTION_MAPPINGS.get(action_code, "")
    
    if action_text:
        text = f"{action_code}: {action_text}"
    else:
        text = f"{action_code}"
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Determine position
    if position == "bottom-center":
        x = (w - text_width) // 2
        y = h - baseline - 20  # 20px from bottom
    elif position == "top-center":
        x = (w - text_width) // 2
        y = text_height + 20  # 20px from top
    elif position == "top-left":
        x = 20
        y = text_height + 20
    else:  # Default to bottom-center
        x = (w - text_width) // 2
        y = h - baseline - 20
    
    # Create semi-transparent black rectangle for better text visibility
    overlay = result.copy()
    
    # Draw filled rectangle
    cv2.rectangle(
        overlay,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        (0, 0, 0),  # Black background
        -1  # Filled rectangle
    )
    
    # Apply the overlay with transparency
    alpha = 0.6  # 60% opacity
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    # Add the text
    cv2.putText(
        result,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),  # White text
        thickness
    )
    
    return result
