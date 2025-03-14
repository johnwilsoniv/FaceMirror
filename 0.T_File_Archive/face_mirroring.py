import cv2
import numpy as np

def preprocess_tilted_frame(self, frame, landmarks):
    """Optionally preprocess frame to correct severe tilt before mirroring"""
    if landmarks is None or len(self.pose_history) == 0:
        return frame
        
    # Get average roll angle from pose history
    avg_roll = np.mean([p[2] for p in self.pose_history])
    
    # Only correct severe tilt
    if abs(avg_roll) < 0.2:  # Less than ~11 degrees
        return frame
        
    # Calculate face center
    face_center = tuple(map(int, np.mean(landmarks, axis=0)))
    
    # Get rotation matrix
    correction_angle = -avg_roll  # Full correction
    angle_degrees = np.degrees(correction_angle)
    rotation_matrix = cv2.getRotationMatrix2D(face_center, angle_degrees, 1.0)
    
    # Apply rotation
    height, width = frame.shape[:2]
    corrected_frame = cv2.warpAffine(frame, rotation_matrix, (width, height), 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_REPLICATE)
                                     
    # Note: we would need to transform landmarks too if we used this approach
    # For now, we're not using this function directly - instead applying smaller 
    # corrections throughout the pipeline
    
    return corrected_frame

def create_mirrored_faces(self, frame, landmarks):
    """Create mirrored faces by reflecting exactly along the anatomical midline with tilt compensation"""
    height, width = frame.shape[:2]

    # Get midline points with frame shape for full extension
    glabella, chin, _ = self.get_facial_midline(landmarks, frame.shape)

    if glabella is None or chin is None:
        return frame.copy(), frame.copy()

    # Calculate midline direction vector with tilt compensation
    direction = chin - glabella
    
    # Get head pose information if available
    if self.pose_history:
        avg_roll = np.mean([p[2] for p in self.pose_history])
        
        # Apply tilt correction to the direction vector
        if abs(avg_roll) > 0.1:  # Only correct significant tilt
            # Rotate the direction vector to compensate for head tilt
            correction_angle = -avg_roll * 0.3  # Partial correction (30%)
            
            # Create rotation matrix
            cos_angle = np.cos(correction_angle)
            sin_angle = np.sin(correction_angle)
            rotation_mat = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])
            
            # Apply rotation to direction vector
            direction = np.dot(rotation_mat, direction)
    
    # Normalize the direction vector
    direction = direction / np.sqrt(np.sum(direction ** 2))

    # Create meshgrid of coordinates
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

    # Reshape coordinates for vectorized operations
    coords_reshaped = coords.reshape(-1, 2)

    # Calculate perpendicular vector (pointing to anatomical right)
    # Note: OpenCV's coordinate system has y increasing downward
    perpendicular = np.array([-direction[1], direction[0]])

    # Calculate signed distances from midline
    # Positive distances are on anatomical right, negative on anatomical left
    diff = coords_reshaped - glabella
    distances = np.dot(diff, perpendicular)

    # Calculate reflection points
    reflection = coords_reshaped - 2 * np.outer(distances, perpendicular)
    reflection = reflection.reshape(height, width, 2)

    # Clip reflection coordinates to image bounds
    reflection[..., 0] = np.clip(reflection[..., 0], 0, width - 1)
    reflection[..., 1] = np.clip(reflection[..., 1], 0, height - 1)
    reflection = reflection.astype(np.int32)

    # Create output images
    anatomical_right_face = frame.copy()
    anatomical_left_face = frame.copy()

    # Create side masks based on signed distance
    distances = distances.reshape(height, width)
    anatomical_right_mask = distances >= 0  # Points on anatomical right
    anatomical_left_mask = distances < 0  # Points on anatomical left

    # Apply reflections
    for i in range(3):  # For each color channel
        # Anatomical right face: keep right side, reflect left side
        reflected_vals = frame[reflection[..., 1], reflection[..., 0], i]
        anatomical_right_face[..., i][anatomical_left_mask] = reflected_vals[anatomical_left_mask]

        # Anatomical left face: keep left side, reflect right side
        anatomical_left_face[..., i][anatomical_right_mask] = reflected_vals[anatomical_right_mask]

    # Apply adaptive gradient blending along the midline
    # Wider blending for more severe tilt
    blend_width = 20  # Base width in pixels
    
    # Adjust blend width based on head tilt if available
    if self.pose_history:
        avg_roll = np.mean([p[2] for p in self.pose_history])
        tilt_factor = abs(np.sin(avg_roll))
        blend_width = int(blend_width * (1 + tilt_factor))

    # Calculate points along the midline
    y_range = np.arange(height)
    if abs(direction[1]) > 1e-6:  # Avoid division by zero
        t = (y_range - glabella[1]) / direction[1]
        x_midline = (glabella[0] + t * direction[0]).astype(int)
        x_midline = np.clip(x_midline, blend_width // 2, width - blend_width // 2 - 1)

        # Create blending weights
        blend_weights = np.linspace(0, 1, blend_width)

        # Vectorized blending
        for i in range(blend_width):
            weight = blend_weights[i]
            x_offset = i - blend_width // 2
            x_coords = np.clip(x_midline + x_offset, 0, width - 1)

            # Apply blending using array indexing
            blend_weight = weight if x_offset >= 0 else (1 - weight)
            anatomical_right_face[y_range, x_coords] = (
                    anatomical_right_face[y_range, x_coords] * (1 - blend_weight) +
                    frame[y_range, x_coords] * blend_weight
            )
            anatomical_left_face[y_range, x_coords] = (
                    anatomical_left_face[y_range, x_coords] * (1 - blend_weight) +
                    frame[y_range, x_coords] * blend_weight
            )

    return anatomical_right_face, anatomical_left_face
