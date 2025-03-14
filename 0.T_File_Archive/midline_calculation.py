import numpy as np

def get_facial_midline(self, landmarks, frame_shape=None):
    """Calculate the anatomical midline points"""
    if landmarks is None:
        return None, None, {}  # Keep empty dict return for compatibility

    # Convert to float32 for calculations
    landmarks = landmarks.astype(np.float32)

    # Get medial eyebrow points
    left_medial_brow = landmarks[21]
    right_medial_brow = landmarks[22]

    # Calculate glabella and chin
    glabella = (left_medial_brow + right_medial_brow) / 2
    chin = landmarks[8]

    # Add to history
    self.glabella_history.append(glabella)
    self.chin_history.append(chin)

    if len(self.glabella_history) > self.history_size:
        self.glabella_history.pop(0)
    if len(self.chin_history) > self.history_size:
        self.chin_history.pop(0)

    # Calculate smooth midline points using simple mean
    smooth_glabella = np.mean(self.glabella_history, axis=0)
    smooth_chin = np.mean(self.chin_history, axis=0)

    # If frame shape is provided, extend the line to cover the entire frame
    if frame_shape is not None:
        height, width = frame_shape[:2]
        
        # Calculate direction vector
        direction = smooth_chin - smooth_glabella
        if np.any(direction):
            direction = direction / np.sqrt(np.sum(direction ** 2))
            
            # Parametric line equation: point = start + t * direction
            # Find t values for intersections with frame boundaries
            t_vals = []
            
            # Top boundary (y=0)
            if abs(direction[1]) > 1e-6:
                t = -smooth_glabella[1] / direction[1]
                x_intersect = smooth_glabella[0] + t * direction[0]
                if 0 <= x_intersect <= width:
                    t_vals.append(t)
            
            # Bottom boundary (y=height-1)
            if abs(direction[1]) > 1e-6:
                t = (height - 1 - smooth_glabella[1]) / direction[1]
                x_intersect = smooth_glabella[0] + t * direction[0]
                if 0 <= x_intersect <= width:
                    t_vals.append(t)
            
            # Left boundary (x=0)
            if abs(direction[0]) > 1e-6:
                t = -smooth_glabella[0] / direction[0]
                y_intersect = smooth_glabella[1] + t * direction[1]
                if 0 <= y_intersect <= height:
                    t_vals.append(t)
            
            # Right boundary (x=width-1)
            if abs(direction[0]) > 1e-6:
                t = (width - 1 - smooth_glabella[0]) / direction[0]
                y_intersect = smooth_glabella[1] + t * direction[1]
                if 0 <= y_intersect <= height:
                    t_vals.append(t)
            
            # Sort t values
            t_vals.sort()
            
            # Use the first and last t values to get the endpoints
            if len(t_vals) >= 2:
                smooth_glabella = smooth_glabella + t_vals[0] * direction
                smooth_chin = smooth_glabella + (t_vals[-1] - t_vals[0]) * direction
    
    # For backwards compatibility, still return an empty dict as third value
    empty_log_data = {}
    
    # Return the simple, original midline points
    return smooth_glabella, smooth_chin, empty_log_data
