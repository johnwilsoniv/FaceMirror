import numpy as np
import cv2


class FaceMirror:
    """Handles the creation of mirrored face images"""

    def __init__(self, landmark_detector):
        """Initialize with a reference to the landmark detector"""
        self.landmark_detector = landmark_detector
        # Cache for coordinate grid (reused across frames of same size)
        self._cached_coords = None
        self._cached_size = None

    def _get_coords(self, height, width):
        """Get or create cached coordinate grid."""
        if self._cached_size != (height, width):
            # Create coordinate grids once per video resolution
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            self._cached_coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
            self._cached_size = (height, width)
        return self._cached_coords

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
        norm = np.sqrt(direction[0]**2 + direction[1]**2)
        direction = direction / norm

        # Get cached coordinate grid
        coords = self._get_coords(height, width)

        # Calculate perpendicular vector (pointing to anatomical right)
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)

        # Calculate signed distances from midline (vectorized, in-place where possible)
        # diff = coords - glabella, then dot with perpendicular
        diff_x = coords[..., 0] - glabella[0]
        diff_y = coords[..., 1] - glabella[1]
        distances = diff_x * perpendicular[0] + diff_y * perpendicular[1]

        # Calculate reflection map for cv2.remap (MUCH faster than fancy indexing)
        # reflection = coords - 2 * distances * perpendicular
        map_x = coords[..., 0] - 2 * distances * perpendicular[0]
        map_y = coords[..., 1] - 2 * distances * perpendicular[1]

        # Clip to image bounds (in-place)
        np.clip(map_x, 0, width - 1, out=map_x)
        np.clip(map_y, 0, height - 1, out=map_y)

        # Use cv2.remap for fast bilinear interpolation (10-50x faster than fancy indexing)
        reflected_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # Create side masks
        right_mask = distances >= 0  # Points on anatomical right
        left_mask = ~right_mask  # Points on anatomical left

        # Use cv2.copyTo with masks (faster than np.where for images)
        anatomical_right_face = frame.copy()
        anatomical_left_face = frame.copy()

        # Convert masks to uint8 for OpenCV
        left_mask_u8 = left_mask.astype(np.uint8) * 255
        right_mask_u8 = right_mask.astype(np.uint8) * 255

        # Apply reflections using copyTo (respects mask)
        cv2.copyTo(reflected_frame, left_mask_u8, anatomical_right_face)
        cv2.copyTo(reflected_frame, right_mask_u8, anatomical_left_face)

        # Apply gradient blending along the midline (6 pixels for faster processing)
        blend_width = 6

        # Calculate points along the midline
        if abs(direction[1]) > 1e-6:  # Avoid division by zero
            y_range = np.arange(height)
            t = (y_range - glabella[1]) / direction[1]
            x_midline = (glabella[0] + t * direction[0]).astype(np.int32)
            np.clip(x_midline, blend_width // 2, width - blend_width // 2 - 1, out=x_midline)

            # Vectorized blending - process all blend positions at once
            blend_weights = np.linspace(0, 1, blend_width, dtype=np.float32)
            offsets = np.arange(blend_width) - blend_width // 2

            for i, (offset, weight) in enumerate(zip(offsets, blend_weights)):
                x_coords = np.clip(x_midline + offset, 0, width - 1)
                blend_w = weight if offset >= 0 else (1 - weight)
                inv_blend = 1.0 - blend_w

                # Vectorized blend for all y positions
                anatomical_right_face[y_range, x_coords] = (
                    anatomical_right_face[y_range, x_coords] * inv_blend +
                    frame[y_range, x_coords] * blend_w
                ).astype(np.uint8)
                anatomical_left_face[y_range, x_coords] = (
                    anatomical_left_face[y_range, x_coords] * inv_blend +
                    frame[y_range, x_coords] * blend_w
                ).astype(np.uint8)

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