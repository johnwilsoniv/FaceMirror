import cv2
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import dlib
from scipy.spatial import Delaunay, cKDTree
from video_rotation import process_video_rotation
import time


class StableFaceSplitter:
    def __init__(self, debug_mode=False):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)

        # Face tracking parameters
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0
        self.detection_interval = 3  # Increased from 2 to 3 for efficiency

        # Initialize smoothing with exponential moving average
        self.smoothed_landmarks = None
        self.smoothed_glabella = None
        self.smoothed_nasal_tip = None
        self.smoothed_menton = None
        self.smoothing_factor = 0.7  # Higher values mean more smoothing

        # Reusable arrays for efficiency
        self.face_roi = None
        self.reflection_map = None
        self.distance_map = None

        # Control parameters
        self.blend_width = 20

        # Debug mode settings
        self.debug_mode = debug_mode
        if debug_mode:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        # Reset face tracking parameters
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0

        # Reset smoothed values
        self.smoothed_landmarks = None
        self.smoothed_glabella = None
        self.smoothed_nasal_tip = None
        self.smoothed_menton = None

        # Reset arrays
        self.face_roi = None
        self.reflection_map = None
        self.distance_map = None

    def preprocess_frame(self, frame):
        """Enhance frame for better face detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised, gray

    def get_face_roi(self, frame, landmarks):
        """Get region of interest around the face to avoid processing the entire image"""
        if landmarks is None:
            return None

        # Get face bounds with padding
        height, width = frame.shape[:2]
        x_min = max(0, int(np.min(landmarks[:, 0]) - width * 0.1))
        y_min = max(0, int(np.min(landmarks[:, 1]) - height * 0.1))
        x_max = min(width - 1, int(np.max(landmarks[:, 0]) + width * 0.1))
        y_max = min(height - 1, int(np.max(landmarks[:, 1]) + height * 0.1))

        # Ensure vertical lines extend to the top and bottom of the image
        y_min = 0
        y_max = height - 1

        return (x_min, y_min, x_max, y_max)

    def get_face_mesh(self, frame):
        """Get facial landmarks with temporal smoothing using EMA"""
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use ROI for face detection if available
        if self.face_roi is not None:
            x_min, y_min, x_max, y_max = self.face_roi
            detection_area = gray[y_min:y_max, x_min:x_max]
            detection_rect = dlib.rectangle(0, 0, x_max - x_min, y_max - y_min)
            faces = self.detector(detection_area)

            if faces:
                # Convert face rect back to original coordinates
                best_face = max(faces, key=lambda rect: rect.width() * rect.height())
                self.last_face = dlib.rectangle(
                    best_face.left() + x_min,
                    best_face.top() + y_min,
                    best_face.right() + x_min,
                    best_face.bottom() + y_min
                )
            else:
                # If no face in ROI, search in whole image, but less frequently
                if self.frame_count % self.detection_interval == 0:
                    faces = self.detector(gray)
                    if faces:
                        self.last_face = max(faces, key=lambda rect: rect.width() * rect.height())
                    else:
                        self.last_face = None
        else:
            # Initial detection or if ROI is lost
            if self.frame_count % self.detection_interval == 0 or self.last_face is None:
                faces = self.detector(gray)
                if faces:
                    self.last_face = max(faces, key=lambda rect: rect.width() * rect.height())
                else:
                    self.last_face = None

        if self.last_face is not None:
            # Extract landmarks
            landmarks = self.predictor(gray, self.last_face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)

            # Apply EMA smoothing for landmarks
            if self.smoothed_landmarks is None:
                self.smoothed_landmarks = points.copy()
            else:
                self.smoothed_landmarks = (
                        self.smoothing_factor * self.smoothed_landmarks +
                        (1 - self.smoothing_factor) * points
                )

            # Get and update face ROI for next frame
            self.face_roi = self.get_face_roi(frame, self.smoothed_landmarks)

            # Create triangulation (only needed for some applications)
            triangles = Delaunay(self.smoothed_landmarks)

            # Return integer coordinates for display
            return np.round(self.smoothed_landmarks).astype(np.int32), triangles

        return None, None

    def get_facial_midline(self, landmarks):
        """Calculate the anatomical midline points with EMA smoothing"""
        if landmarks is None:
            return None, None, None

        # Calculate midline points
        left_medial_brow = landmarks[21].astype(np.float32)
        right_medial_brow = landmarks[22].astype(np.float32)
        glabella = (left_medial_brow + right_medial_brow) / 2
        nasal_tip = landmarks[30].astype(np.float32)
        menton = landmarks[8].astype(np.float32)

        # Apply exponential moving average for smooth tracking
        if self.smoothed_glabella is None:
            self.smoothed_glabella = glabella
            self.smoothed_nasal_tip = nasal_tip
            self.smoothed_menton = menton
        else:
            self.smoothed_glabella = (
                    self.smoothing_factor * self.smoothed_glabella +
                    (1 - self.smoothing_factor) * glabella
            )
            self.smoothed_nasal_tip = (
                    self.smoothing_factor * self.smoothed_nasal_tip +
                    (1 - self.smoothing_factor) * nasal_tip
            )
            self.smoothed_menton = (
                    self.smoothing_factor * self.smoothed_menton +
                    (1 - self.smoothing_factor) * menton
            )

        return self.smoothed_glabella, self.smoothed_nasal_tip, self.smoothed_menton

    def create_mirrored_faces(self, frame, landmarks):
        """Efficiently create mirrored faces using ROI-based processing and vectorized operations"""
        height, width = frame.shape[:2]

        # Get midline points
        glabella, nasal_tip, menton = self.get_facial_midline(landmarks)
        if glabella is None:
            return frame.copy(), frame.copy()

        # Create output images
        anatomical_right_face = frame.copy()
        anatomical_left_face = frame.copy()

        # Get ROI around face to reduce computation
        if self.face_roi is None:
            x_min, y_min = 0, 0
            x_max, y_max = width - 1, height - 1
        else:
            x_min, y_min, x_max, y_max = self.face_roi
            # Ensure vertical line coverage
            y_min = 0
            y_max = height - 1

        # Create meshgrid only for ROI
        roi_height = y_max - y_min + 1
        roi_width = x_max - x_min + 1
        y_coords, x_coords = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
        coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
        coords_reshaped = coords.reshape(-1, 2)

        # Initialize or reuse arrays
        if self.reflection_map is None or self.reflection_map.shape[0] != coords_reshaped.shape[0]:
            self.reflection_map = np.zeros_like(coords_reshaped)
            self.distance_map = np.zeros(coords_reshaped.shape[0])
        else:
            self.reflection_map.fill(0)
            self.distance_map.fill(0)

        # Define the segments with extended endpoints for overlap
        # Format: (p1, p2, direction vector, normalized direction, perpendicular vector)
        segments = []

        # Precompute segment data
        # Vertical line above glabella
        p1 = np.array([glabella[0], 0])
        p2 = glabella
        direction = p2 - p1
        length = np.sqrt(np.sum(direction ** 2))
        if length > 1e-6:
            norm_dir = direction / length
            perp = np.array([-norm_dir[1], norm_dir[0]])
            segments.append((p1, p2, direction, norm_dir, perp, length))

        # Glabella to nasal tip
        p1 = glabella
        p2 = nasal_tip
        direction = p2 - p1
        length = np.sqrt(np.sum(direction ** 2))
        if length > 1e-6:
            norm_dir = direction / length
            perp = np.array([-norm_dir[1], norm_dir[0]])
            segments.append((p1, p2, direction, norm_dir, perp, length))

        # Nasal tip to menton
        p1 = nasal_tip
        p2 = menton
        direction = p2 - p1
        length = np.sqrt(np.sum(direction ** 2))
        if length > 1e-6:
            norm_dir = direction / length
            perp = np.array([-norm_dir[1], norm_dir[0]])
            segments.append((p1, p2, direction, norm_dir, perp, length))

        # Vertical line below menton
        p1 = menton
        p2 = np.array([menton[0], height - 1])
        direction = p2 - p1
        length = np.sqrt(np.sum(direction ** 2))
        if length > 1e-6:
            norm_dir = direction / length
            perp = np.array([-norm_dir[1], norm_dir[0]])
            segments.append((p1, p2, direction, norm_dir, perp, length))

        # Process each segment with vectorized operations
        total_weights = np.zeros(coords_reshaped.shape[0])

        for p1, p2, direction, norm_dir, perp, segment_length in segments:
            # Calculate vectors from p1 to each point
            v = coords_reshaped - p1

            # Calculate projections onto segment
            projections = np.dot(v, norm_dir)

            # Find points that project onto this segment
            on_segment = (projections >= 0) & (projections <= segment_length)

            if np.any(on_segment):
                # Calculate points on the line
                closest_points = p1 + np.outer(projections[on_segment], norm_dir)

                # Calculate vectors from closest points to original points
                to_line = coords_reshaped[on_segment] - closest_points

                # Calculate signed distances
                distances = np.dot(to_line, perp)

                # Calculate reflection points
                reflections = coords_reshaped[on_segment] - 2 * np.outer(distances, perp)

                # Calculate weights using parabolic function for smooth transitions
                # This gives more weight to the center of each segment and less at the ends
                normalized_projections = projections[on_segment] / segment_length
                weights = 4 * normalized_projections * (1 - normalized_projections)

                # Add weighted contribution to overall maps
                self.reflection_map[on_segment] += reflections * np.expand_dims(weights, axis=1)
                self.distance_map[on_segment] += distances * weights
                total_weights[on_segment] += weights

        # Normalize by total weights where weights > 0
        valid_points = total_weights > 0
        if np.any(valid_points):
            self.reflection_map[valid_points] /= np.expand_dims(total_weights[valid_points], axis=1)
            self.distance_map[valid_points] /= total_weights[valid_points]

        # Handle unassigned points efficiently using KD-Tree for nearest neighbor search
        if np.any(~valid_points):
            # Only build tree if we have valid points
            if np.sum(valid_points) > 0:
                # Use scipy's KDTree for efficient nearest neighbor search
                valid_indices = np.where(valid_points)[0]
                valid_coords = coords_reshaped[valid_points]

                # Build KD-Tree for efficient nearest neighbor search
                tree = cKDTree(valid_coords)

                # Find nearest valid point for each unassigned point
                unassigned_coords = coords_reshaped[~valid_points]
                _, indices = tree.query(unassigned_coords, k=1)

                # Copy reflection and distance from nearest valid point
                for i, (unassigned_idx, nearest_valid_idx) in enumerate(zip(np.where(~valid_points)[0], indices)):
                    self.reflection_map[unassigned_idx] = self.reflection_map[valid_indices[nearest_valid_idx]]
                    self.distance_map[unassigned_idx] = self.distance_map[valid_indices[nearest_valid_idx]]

        # Reshape maps back to image dimensions
        distances = self.distance_map.reshape(roi_height, roi_width)
        reflection = self.reflection_map.reshape(roi_height, roi_width, 2)

        # Clip reflections to image bounds
        reflection[..., 0] = np.clip(reflection[..., 0], 0, width - 1)
        reflection[..., 1] = np.clip(reflection[..., 1], 0, height - 1)
        reflection = reflection.astype(np.int32)

        # Create side masks
        anatomical_right_mask = np.zeros((height, width), dtype=bool)
        anatomical_left_mask = np.zeros((height, width), dtype=bool)

        # Set masks only in ROI area
        roi_right_mask = distances >= 0
        roi_left_mask = distances < 0

        anatomical_right_mask[y_min:y_max + 1, x_min:x_max + 1] = roi_right_mask
        anatomical_left_mask[y_min:y_max + 1, x_min:x_max + 1] = roi_left_mask

        # Apply reflections efficiently using vectorized operations
        for i in range(3):  # For each color channel
            # Get coordinates for lookup
            y_reflect = reflection[..., 1].flatten()
            x_reflect = reflection[..., 0].flatten()

            # Get reflected values
            reflected_vals = frame[y_reflect, x_reflect, i].reshape(roi_height, roi_width)

            # Apply reflections only in ROI
            y_roi, x_roi = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]

            # Update right face (keep right side, reflect left side)
            anatomical_right_face[y_roi[roi_left_mask], x_roi[roi_left_mask], i] = reflected_vals[roi_left_mask]

            # Update left face (keep left side, reflect right side)
            anatomical_left_face[y_roi[roi_right_mask], x_roi[roi_right_mask], i] = reflected_vals[roi_right_mask]

        # Apply gradient blending along the midline for smoother transition
        blend_width = self.blend_width

        # Calculate distance to midline for blending
        distance_to_midline = np.abs(distances)
        close_to_midline = distance_to_midline < blend_width / 2

        if np.any(close_to_midline):
            # Calculate blend weights
            blend_factors = np.clip(distance_to_midline[close_to_midline] / (blend_width / 2), 0, 1)

            # Get coordinates
            y_close = y_min + np.where(close_to_midline)[0] // roi_width
            x_close = x_min + np.where(close_to_midline)[0] % roi_width

            # Apply blending vectorized where possible
            for i, (y, x, factor) in enumerate(zip(y_close, x_close, blend_factors)):
                if 0 <= y < height and 0 <= x < width:  # Safety check
                    if distances.flat[i] >= 0:  # Anatomical right
                        anatomical_right_face[y, x] = (
                                anatomical_right_face[y, x] * factor +
                                frame[y, x] * (1 - factor)
                        )
                    else:  # Anatomical left
                        anatomical_left_face[y, x] = (
                                anatomical_left_face[y, x] * factor +
                                frame[y, x] * (1 - factor)
                        )

        return anatomical_right_face, anatomical_left_face

    def create_debug_frame(self, frame, landmarks):
        """Create debug visualization with all 68 landmarks and anatomical segmented midline"""
        if not self.debug_mode:
            return frame  # Skip debug frame creation if not in debug mode

        debug_frame = frame.copy()

        if landmarks is not None:
            # Draw all 68 landmarks as dots
            for point in landmarks:
                x, y = point.astype(int)
                # Draw point
                cv2.circle(debug_frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dot

            # Highlight medial eyebrow points (21, 22) and nasal tip (30)
            left_medial_brow = tuple(map(int, landmarks[21]))
            right_medial_brow = tuple(map(int, landmarks[22]))
            nasal_tip = tuple(map(int, landmarks[30]))

            cv2.circle(debug_frame, left_medial_brow, 4, (255, 0, 0), -1)  # Blue dot
            cv2.circle(debug_frame, right_medial_brow, 4, (255, 0, 0), -1)  # Blue dot
            cv2.circle(debug_frame, nasal_tip, 4, (0, 0, 255), -1)  # Red dot

            # Get midline points
            glabella, nasal_tip_smooth, menton = self.get_facial_midline(landmarks)

            if glabella is not None and nasal_tip_smooth is not None and menton is not None:
                # Convert points to integer coordinates for drawing
                glabella = tuple(map(int, glabella))
                nasal_tip_smooth = tuple(map(int, nasal_tip_smooth))
                menton = tuple(map(int, menton))

                # Draw midline points
                cv2.circle(debug_frame, glabella, 4, (0, 255, 0), -1)  # Green dot for glabella
                cv2.circle(debug_frame, nasal_tip_smooth, 4, (0, 255, 0), -1)  # Green dot for nasal tip
                cv2.circle(debug_frame, menton, 4, (0, 255, 0), -1)  # Green dot for menton

                # Calculate and draw the segmented midline
                height, width = frame.shape[:2]

                # Vertical line above glabella
                top_point = (glabella[0], 0)
                cv2.line(debug_frame, top_point, glabella, (0, 0, 255), 2)  # Red line

                # Glabella to nasal tip
                cv2.line(debug_frame, glabella, nasal_tip_smooth, (0, 0, 255), 2)  # Red line

                # Nasal tip to menton
                cv2.line(debug_frame, nasal_tip_smooth, menton, (0, 0, 255), 2)  # Red line

                # Vertical line below menton
                bottom_point = (menton[0], height - 1)
                cv2.line(debug_frame, menton, bottom_point, (0, 0, 255), 2)  # Red line

                # Add intersection points for clarity
                cv2.circle(debug_frame, (glabella[0], 0), 5, (255, 0, 255), -1)  # Magenta dot for top intersection
                cv2.circle(debug_frame, (menton[0], height - 1), 5, (255, 0, 255),
                           -1)  # Magenta dot for bottom intersection

                # Draw ROI if available
                if self.face_roi is not None:
                    x_min, y_min, x_max, y_max = self.face_roi
                    cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            # Draw face boundary
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.polylines(debug_frame, [hull], True, (255, 255, 0), 2)

            # Add legend
            legend_y = 30
            cv2.putText(debug_frame, "Yellow: All landmarks", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(debug_frame, "Blue: Medial eyebrow points", (10, legend_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(debug_frame, "Red: Nasal tip", (10, legend_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(debug_frame, "Green: Calculated midline points", (10, legend_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_frame, "Red line: Segmented midline", (10, legend_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(debug_frame, "Magenta: Extended midline points", (10, legend_y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(debug_frame, "Green rect: Processing ROI", (10, legend_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return debug_frame

    def process_video(self, input_path, output_dir):
        """Process video file with progress tracking and performance measurements"""
        # Reset tracking history at the start of each video
        self.reset_tracking_history()

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing video: {input_path.name}")

        # Process video rotation
        print("Checking video rotation...")
        rotated_input_path = output_dir / f"{input_path.stem}_rotated{input_path.suffix}"
        rotated_input_path = process_video_rotation(str(input_path), str(rotated_input_path))

        # Update output filenames to reflect anatomical sides
        anatomical_right_output = output_dir / f"{input_path.stem}_right_mirrored.mp4"
        anatomical_left_output = output_dir / f"{input_path.stem}_left_mirrored.mp4"
        debug_output = output_dir / f"{input_path.stem}_debug.mp4"

        # Open video and get properties
        cap = cv2.VideoCapture(str(rotated_input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {rotated_input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps

        print(f"\nVideo details:")
        print(f"- Resolution: {width}x{height}")
        print(f"- Frames: {total_frames}")
        print(f"- Duration: {duration:.1f} seconds")
        print(f"- FPS: {fps}")

        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        right_writer = cv2.VideoWriter(str(anatomical_right_output), fourcc, fps, (width, height))
        left_writer = cv2.VideoWriter(str(anatomical_left_output), fourcc, fps, (width, height))

        # Only create debug writer if in debug mode
        debug_writer = None
        if self.debug_mode:
            debug_writer = cv2.VideoWriter(str(debug_output), fourcc, fps, (width, height))

        # Performance tracking
        total_time = 0
        landmark_time = 0
        mirror_time = 0
        debug_time = 0
        write_time = 0

        # Process frames
        frame_count = 0
        last_progress = -1
        print("\nProcessing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Update progress every 1%
                progress = int((frame_count / total_frames) * 100)
                if progress != last_progress:
                    print(f"\rProgress: {progress}% ({frame_count}/{total_frames} frames)", end="")
                    if frame_count > 0:
                        avg_fps = frame_count / total_time if total_time > 0 else 0
                        print(f" | Avg speed: {avg_fps:.1f} fps", end="")
                    last_progress = progress

                start_time = time.time()

                # Process frame
                t1 = time.time()
                landmarks, _ = self.get_face_mesh(frame)
                t2 = time.time()
                landmark_time += t2 - t1

                if landmarks is not None:
                    t1 = time.time()
                    right_face, left_face = self.create_mirrored_faces(frame, landmarks)
                    t2 = time.time()
                    mirror_time += t2 - t1

                    if self.debug_mode:
                        t1 = time.time()
                        debug_frame = self.create_debug_frame(frame, landmarks)
                        t2 = time.time()
                        debug_time += t2 - t1
                    else:
                        debug_frame = None
                else:
                    right_face, left_face = frame.copy(), frame.copy()
                    debug_frame = frame.copy() if self.debug_mode else None

                t1 = time.time()
                right_writer.write(right_face.astype(np.uint8))
                left_writer.write(left_face.astype(np.uint8))
                if debug_writer:
                    debug_writer.write(debug_frame)
                t2 = time.time()
                write_time += t2 - t1

                frame_time = time.time() - start_time
                total_time += frame_time

            except Exception as e:
                if self.debug_mode:
                    self.logger.warning(f"\nError processing frame {frame_count}: {str(e)}")
                right_writer.write(frame)
                left_writer.write(frame)
                if debug_writer:
                    debug_writer.write(frame)

            frame_count += 1

        # Clean up
        print(f"\nProcessing complete: {frame_count} frames processed")
        if frame_count > 0:
            print(f"Performance statistics:")
            print(f"- Total processing time: {total_time:.1f} seconds")
            print(f"- Average speed: {frame_count / total_time:.1f} fps")
            print(f"- Time breakdown:")
            print(f"  - Landmark detection: {landmark_time:.1f}s ({landmark_time / total_time * 100:.1f}%)")
            print(f"  - Face mirroring: {mirror_time:.1f}s ({mirror_time / total_time * 100:.1f}%)")
            print(f"  - Debug visualization: {debug_time:.1f}s ({debug_time / total_time * 100:.1f}%)")
            print(f"  - Video writing: {write_time:.1f}s ({write_time / total_time * 100:.1f}%)")

        cap.release()
        right_writer.release()
        left_writer.release()
        if debug_writer:
            debug_writer.release()

        # Determine the list of output files
        output_files = [str(anatomical_right_output), str(anatomical_left_output)]
        if debug_writer:
            output_files.append(str(debug_output))
        if str(rotated_input_path) != str(input_path):
            output_files.append(str(rotated_input_path))

        print("\nOutput files:")
        for f in output_files:
            print(f"- {Path(f).name}")
        print("")

        return output_files


def main():
    """Simple command-line interface"""
    try:
        # Initialize root window for dialogs
        root = tk.Tk()
        root.withdraw()

        # First, select input files
        input_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not input_paths:
            return

        # Then immediately ask about debug mode before any processing
        #debug_mode = messagebox.askyesno("Debug Mode",
        #                                 "Would you like to generate debug visualizations?\n" +
        #                                 "(Turning off will make processing faster)")

        debug_mode = True
        
        root.destroy() # This would close the main window and any associated messageboxes

        # Create output directory
        output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the processor with selected debug mode
        print("\nInitializing face mirroring processor...")
        splitter = StableFaceSplitter(debug_mode=debug_mode)
        results = []

        # Now begin processing files
        for input_path in input_paths:
            try:
                outputs = splitter.process_video(input_path, output_dir)
                results.append({
                    'input': input_path,
                    'success': True,
                    'outputs': outputs
                })
            except Exception as e:
                results.append({
                    'input': input_path,
                    'success': False,
                    'error': str(e)
                })

        summary = "Processing Results:\n\n"
        for result in results:
            if result['success']:
                summary += f"✓ {Path(result['input']).name}\n"

                # Separate output types
                left_video = next((f for f in result['outputs'] if 'left_mirrored' in f), None)
                right_video = next((f for f in result['outputs'] if 'right_mirrored' in f), None)
                debug_video = next((f for f in result['outputs'] if 'debug' in f), None)
                rotated_video = next((f for f in result['outputs'] if 'rotated' in f), None)

                if left_video:
                    summary += f"  - Left: {Path(left_video).name}\n"
                if right_video:
                    summary += f"  - Right: {Path(right_video).name}\n"
                if debug_video:
                    summary += f"  - Debug: {Path(debug_video).name}\n"
                if rotated_video:
                    summary += f"  - Rotated: {Path(rotated_video).name}\n"
            else:
                summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

        messagebox.showinfo("Processing Complete", summary)

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()