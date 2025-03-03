import numpy as np
import cv2

class FaceMirror:
    """Handles the creation of mirrored face images"""
    
    def __init__(self, landmark_detector):
        """Initialize with a reference to the landmark detector"""
        self.landmark_detector = landmark_detector
    
    def create_mirrored_faces(self, frame, landmarks):
        """Create mirrored faces by reflecting exactly along the anatomical midline

        Returns:
        - anatomical_right_face: Right side of face mirrored (patient's right)
        - anatomical_left_face: Left side of face mirrored (patient's left)
        """
        height, width = frame.shape[:2]

        # Get midline points
        glabella, chin = self.landmark_detector.get_facial_midline(landmarks)

        if glabella is None or chin is None:
            return frame.copy(), frame.copy()

        # Calculate midline direction vector
        direction = chin - glabella
        direction = direction / np.sqrt(np.sum(direction ** 2))  # normalize

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

        # Apply gradient blending along the midline
        blend_width = 20  # pixels

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
    
    def create_debug_frame(self, frame, landmarks):
        """Create debug visualization with all 68 landmarks, anatomical midline, and head yaw analysis"""
        debug_frame = frame.copy()

        if landmarks is not None:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Draw all 68 landmarks as dots
            for point in landmarks:
                x, y = point.astype(int)
                # Draw point
                cv2.circle(debug_frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dot

            # Highlight medial eyebrow points (21, 22)
            left_medial_brow = tuple(map(int, landmarks[21]))
            right_medial_brow = tuple(map(int, landmarks[22]))
            cv2.circle(debug_frame, left_medial_brow, 4, (255, 0, 0), -1)  # Blue dot
            cv2.circle(debug_frame, right_medial_brow, 4, (255, 0, 0), -1)  # Blue dot

            # Get midline points
            glabella, chin = self.landmark_detector.get_facial_midline(landmarks)

            if glabella is not None and chin is not None:
                # Convert points to integer coordinates for drawing
                glabella = tuple(map(int, glabella))
                chin = tuple(map(int, chin))

                # Draw midline points
                cv2.circle(debug_frame, glabella, 4, (0, 255, 0), -1)  # Green dot for glabella
                cv2.circle(debug_frame, chin, 4, (0, 255, 0), -1)  # Green dot for chin

                # Calculate and draw the extended midline
                direction = np.array([chin[0] - glabella[0], chin[1] - glabella[1]])

                if np.any(direction):
                    direction = direction / np.sqrt(np.sum(direction ** 2))
                    extension_length = np.sqrt(height ** 2 + width ** 2)

                    top_point = (
                        int(glabella[0] - direction[0] * extension_length),
                        int(glabella[1] - direction[1] * extension_length)
                    )
                    bottom_point = (
                        int(chin[0] + direction[0] * extension_length),
                        int(chin[1] + direction[1] * extension_length)
                    )

                    # Draw the extended midline
                    cv2.line(debug_frame, top_point, bottom_point, (0, 0, 255), 2)  # Red line

            # Calculate head yaw
            yaw = self.landmark_detector.calculate_head_pose(landmarks)
            
            # Create an information panel positioned below the top banner
            if yaw is not None:
                panel_width = 220
                panel_height = 85
                panel_x = width - panel_width - 10
                panel_y = 40  # Moved down from 10 to avoid the top banner
                
                # Draw semi-transparent panel background
                overlay = debug_frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), 
                            (panel_x + panel_width, panel_y + panel_height), 
                            (20, 20, 20), -1)
                # Apply transparency
                cv2.addWeighted(overlay, 0.7, debug_frame, 0.3, 0, debug_frame)
                
                # Panel title
                cv2.putText(debug_frame, "HEAD ROTATION ANALYSIS", 
                            (panel_x + 10, panel_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Yaw - color-coded based on stricter thresholds
                yaw_color = (0, 255, 0) if abs(yaw) <= 3.0 else (0, 165, 255) if abs(yaw) <= 5.0 else (0, 0, 255)
                cv2.putText(debug_frame, f"Rotation (Yaw): {yaw:.1f}°", 
                            (panel_x + 10, panel_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, yaw_color, 1)
                
                # Calculate frame quality
                quality = self.landmark_detector.calculate_frame_quality(landmarks)
                quality_percentage = int(quality * 100)
                
                # Quality indicator with color coding
                quality_color = (0, 255, 0) if quality >= 0.8 else (0, 165, 255) if quality >= 0.6 else (0, 0, 255)
                cv2.putText(debug_frame, f"Mirror Quality: {quality_percentage}%", 
                            (panel_x + 10, panel_y + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)

                # Validate head yaw
                pose_valid, pose_warning = self.landmark_detector.validate_head_pose(landmarks)
                
                # Add analysis banner at the top
                if not pose_valid:
                    # Add a warning banner at the top of the frame
                    cv2.rectangle(debug_frame, (0, 0), (width, 30), (0, 0, 255), -1)
                    
                    # Determine primary message
                    message = "EXCESSIVE HEAD ROTATION - MIRRORING AFFECTED"
                    cv2.putText(debug_frame, message, (width//2 - 190, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display specific warning
                    if pose_warning:
                        # Move warning message to bottom to avoid overlap
                        cv2.rectangle(debug_frame, (0, height-30), (width, height), (0, 0, 0), -1)
                        cv2.putText(debug_frame, pose_warning, (10, height-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Add a green banner when yaw is good
                    cv2.rectangle(debug_frame, (0, 0), (width, 30), (0, 150, 0), -1)
                    cv2.putText(debug_frame, "OPTIMAL HEAD ROTATION FOR MIRRORING", (width//2 - 180, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # If yaw is outside acceptable range, draw correction guide
                if not pose_valid:
                    # Calculate frame center and face center positions
                    frame_center_x = width // 2
                    center_landmarks = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]  # Midline landmarks
                    center_points = landmarks[center_landmarks]
                    face_center_x = int(np.mean(center_points[:, 0]))
                    
                    # Draw arrow indicating correction direction
                    arrow_y = height // 2
                    if face_center_x < frame_center_x:  # Face is turned left of center
                        cv2.arrowedLine(debug_frame, (face_center_x, arrow_y), (frame_center_x, arrow_y), 
                                        (0, 165, 255), 2, tipLength=0.03)
                    elif face_center_x > frame_center_x:  # Face is turned right of center
                        cv2.arrowedLine(debug_frame, (face_center_x, arrow_y), (frame_center_x, arrow_y), 
                                        (0, 165, 255), 2, tipLength=0.03)

            # Draw face boundary
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.polylines(debug_frame, [hull], True, (255, 255, 0), 2)

            # Add legend
            legend_y = 50
            cv2.putText(debug_frame, "Yellow: All landmarks", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(debug_frame, "Blue: Medial eyebrow points", (10, legend_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(debug_frame, "Green: Calculated midline points", (10, legend_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_frame, "Red: Extended midline", (10, legend_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return debug_frame
