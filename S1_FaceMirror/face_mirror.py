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

        # Apply reflections using vectorized operations
        reflected_frame = frame[reflection[..., 1], reflection[..., 0]]

        # Expand mask to 3D for color channels
        right_mask_3d = np.stack([anatomical_right_mask] * 3, axis=-1)
        left_mask_3d = np.stack([anatomical_left_mask] * 3, axis=-1)

        # Apply reflections in one vectorized operation
        anatomical_right_face = np.where(left_mask_3d, reflected_frame, frame)
        anatomical_left_face = np.where(right_mask_3d, reflected_frame, frame)

        # Apply gradient blending along the midline (6 pixels for faster processing)
        blend_width = 6

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
        """Create debug visualization with anatomical midline and head yaw analysis"""
        debug_frame = frame.copy()

        if landmarks is not None:
            # Get frame dimensions
            height, width = frame.shape[:2]

            # Get smoothed midline points from the landmark detector
            glabella, chin = self.landmark_detector.get_facial_midline(landmarks)

            # # --- START: Draw all 68 facial landmarks ---
            # # To disable landmarks, comment out this for-loop
            # for (x, y) in landmarks:
            #     cv2.circle(debug_frame, (x, y), 3, (0, 255, 255), -1) # Bright Yellow
            # # --- END: Draw all 68 facial landmarks ---

            if glabella is not None and chin is not None:
                # # --- START: Highlight key midline anchor points ---
                # # To disable, comment out this block.
                # # This draws over the yellow points to highlight them in green.
                # chin_point = tuple(chin.astype(int))
                # glabella_point = tuple(glabella.astype(int))
                #
                # # Draw the chin point (smoothed landmark 8) in green
                # cv2.circle(debug_frame, chin_point, 4, (0, 255, 0), -1)
                #
                # # Draw the calculated mid-glabellar point in green
                # cv2.circle(debug_frame, glabella_point, 4, (0, 255, 0), -1)
                # # --- END: Highlight key midline anchor points ---

                # Convert points to integer coordinates for drawing
                glabella_point = tuple(glabella.astype(int))
                chin_point = tuple(chin.astype(int))

                # Calculate and draw the extended midline using the same anchor points
                direction = np.array([chin_point[0] - glabella_point[0], chin_point[1] - glabella_point[1]])

                if np.any(direction):
                    direction = direction / np.sqrt(np.sum(direction ** 2))
                    extension_length = np.sqrt(height ** 2 + width ** 2)

                    top_point = (
                        int(glabella_point[0] - direction[0] * extension_length),
                        int(glabella_point[1] - direction[1] * extension_length)
                    )
                    bottom_point = (
                        int(chin_point[0] + direction[0] * extension_length),
                        int(chin_point[1] + direction[1] * extension_length)
                    )

                    # Draw the extended midline
                    cv2.line(debug_frame, top_point, bottom_point, (0, 0, 255), 2)  # Red line

            # Calculate head yaw
            yaw = self.landmark_detector.calculate_head_pose(landmarks)

            # Head rotation analysis panel removed
            if yaw is not None:

                # Update top banner based on yaw thresholds
                if abs(yaw) <= 3.0:
                    # Green banner for optimal
                    cv2.rectangle(debug_frame, (0, 0), (width, 30), (0, 150, 0), -1)
                    cv2.putText(debug_frame, "OPTIMAL HEAD ROTATION FOR MIRRORING", (width // 2 - 180, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                elif abs(yaw) <= 5.0:
                    # Yellow/orange banner for acceptable
                    cv2.rectangle(debug_frame, (0, 0), (width, 30), (0, 165, 255), -1)
                    cv2.putText(debug_frame, "ACCEPTABLE HEAD ROTATION", (width // 2 - 150, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Red banner for excessive
                    cv2.rectangle(debug_frame, (0, 0), (width, 30), (0, 0, 255), -1)
                    cv2.putText(debug_frame, "EXCESSIVE HEAD ROTATION - MIRRORING AFFECTED", (width // 2 - 190, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return debug_frame