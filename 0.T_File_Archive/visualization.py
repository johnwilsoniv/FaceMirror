import cv2
import numpy as np

def create_debug_frame(self, frame, landmarks):
    """Create debug visualization with all 68 landmarks, anatomical midline, and tilt angle"""
    debug_frame = frame.copy()

    if landmarks is not None:
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
        glabella, chin, _ = self.get_facial_midline(landmarks, frame.shape)

        if glabella is not None and chin is not None:
            # Convert points to integer coordinates for drawing
            glabella = tuple(map(int, glabella))
            chin = tuple(map(int, chin))

            # Draw midline points
            cv2.circle(debug_frame, glabella, 4, (0, 255, 0), -1)  # Green dot for glabella
            cv2.circle(debug_frame, chin, 4, (0, 255, 0), -1)  # Green dot for chin

            # Calculate and draw the extended midline
            height, width = frame.shape[:2]
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

        # Calculate and display tilt information
        left_eye_center = np.mean(landmarks[36:42], axis=0).astype(int)
        right_eye_center = np.mean(landmarks[42:48], axis=0).astype(int)
        
        # Draw eye centers and eye line
        cv2.circle(debug_frame, tuple(left_eye_center), 5, (255, 0, 255), -1)  # Magenta dot
        cv2.circle(debug_frame, tuple(right_eye_center), 5, (255, 0, 255), -1)  # Magenta dot
        cv2.line(debug_frame, tuple(left_eye_center), tuple(right_eye_center), (255, 0, 255), 2)  # Magenta line
        
        # Calculate tilt angle
        tilt_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], 
                               right_eye_center[0] - left_eye_center[0])
        tilt_degrees = np.degrees(tilt_angle)
        
        # Display tilt information
        cv2.putText(debug_frame, f"Tilt: {tilt_degrees:.1f} deg", (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
        # Display quality indicator based on tilt
        quality = "Good" if abs(tilt_degrees) < 10 else "Fair" if abs(tilt_degrees) < 25 else "Poor"
        quality_color = (0, 255, 0) if quality == "Good" else (0, 255, 255) if quality == "Fair" else (0, 0, 255)
        cv2.putText(debug_frame, f"Quality: {quality}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

        # Draw face boundary
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.polylines(debug_frame, [hull], True, (255, 255, 0), 2)

        # Add legend
        legend_y = 30
        cv2.putText(debug_frame, "Yellow: All landmarks", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_frame, "Blue: Medial eyebrow points", (10, legend_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_frame, "Green: Calculated midline points", (10, legend_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_frame, "Red: Extended midline", (10, legend_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_frame, "Magenta: Eye centers & tilt", (10, legend_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return debug_frame
