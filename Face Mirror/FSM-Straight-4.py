import cv2
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import dlib
from scipy.spatial import Delaunay
from video_rotation import process_video_rotation


class StableFaceSplitter:
    def __init__(self, debug_mode=False):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)

        # Face tracking parameters
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0

        # Initialize smoothing history
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.history_size = 5

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

                # Reset smoothing history
                self.landmarks_history = []
                self.glabella_history = []
                self.chin_history = []

                # Reset face ROI if it exists
                if hasattr(self, 'face_roi'):
                    self.face_roi = None

    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        # Reset face tracking parameters
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0

        # Reset smoothing history
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []

        # Reset face ROI if it exists
        if hasattr(self, 'face_roi'):
            self.face_roi = None

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

    def update_face_roi(self, face_rect):
        """Update face ROI with temporal smoothing"""
        if face_rect is None:
            return

        current_roi = [face_rect.left(), face_rect.top(),
                       face_rect.right(), face_rect.bottom()]

        if self.face_roi is None:
            self.face_roi = current_roi
        else:
            # Smooth ROI transition
            alpha = 0.7  # Smoothing factor
            self.face_roi = [int(alpha * prev + (1 - alpha) * curr)
                             for prev, curr in zip(self.face_roi, current_roi)]

    def get_face_mesh(self, frame, detection_interval=2):
        """Get facial landmarks with temporal smoothing"""
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_count % detection_interval == 0 or self.last_face is None:
            faces = self.detector(gray)
            if not faces:
                self.last_face = None
                self.last_landmarks = None
                return None, None
            self.last_face = max(faces, key=lambda rect: rect.width() * rect.height())

        if self.last_face is not None:
            # Extract landmarks
            landmarks = self.predictor(gray, self.last_face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)  # Explicitly set dtype

            # Apply temporal smoothing
            self.landmarks_history.append(points)
            if len(self.landmarks_history) > self.history_size:
                self.landmarks_history.pop(0)

            # Calculate weighted average with explicit type handling
            weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
            weights = weights / np.sum(weights)  # Normalize weights

            # Initialize smoothed points with the correct type
            smoothed_points = np.zeros_like(points, dtype=np.float32)

            # Apply weighted average
            for pts, w in zip(self.landmarks_history, weights):
                smoothed_points += pts * w

            # Ensure integer coordinates for final output
            smoothed_points = np.round(smoothed_points).astype(np.int32)

            # Create triangulation
            triangles = Delaunay(smoothed_points)
            self.last_landmarks = smoothed_points

            return smoothed_points, triangles

        return None, None

    def validate_landmarks(self, landmarks):
        """Validate landmark detection quality"""
        if landmarks is None:
            return False

        # Check for outliers
        distances = np.linalg.norm(landmarks - landmarks.mean(axis=0), axis=1)
        z_scores = (distances - distances.mean()) / distances.std()
        if np.any(np.abs(z_scores) > 3):
            return False

        # Check for minimum face size
        face_width = landmarks[:, 0].max() - landmarks[:, 0].min()
        face_height = landmarks[:, 1].max() - landmarks[:, 1].min()
        if face_width < 60 or face_height < 60:
            return False

        return True

    def get_facial_midline(self, landmarks):
        """Calculate the anatomical midline points"""
        if landmarks is None:
            return None, None

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

        # Calculate smooth midline points
        smooth_glabella = np.mean(self.glabella_history, axis=0)
        smooth_chin = np.mean(self.chin_history, axis=0)

        return smooth_glabella, smooth_chin

    def create_mirrored_faces(self, frame, landmarks):
        """Create mirrored faces by reflecting exactly along the anatomical midline

        Returns:
        - anatomical_right_face: Right side of face mirrored (patient's right)
        - anatomical_left_face: Left side of face mirrored (patient's left)
        """
        height, width = frame.shape[:2]

        # Get midline points
        glabella, chin = self.get_facial_midline(landmarks)

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
        """Create debug visualization with all 68 landmarks and anatomical midline"""
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
            glabella, chin = self.get_facial_midline(landmarks)

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

        return debug_frame

    def process_video(self, input_path, output_dir):
        """Process video file with progress tracking"""
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
        debug_writer = cv2.VideoWriter(str(debug_output), fourcc, fps, (width, height))

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
                    last_progress = progress

                # Process frame
                landmarks, _ = self.get_face_mesh(frame)

                if landmarks is not None:
                    right_face, left_face = self.create_mirrored_faces(frame, landmarks)
                    debug_frame = self.create_debug_frame(frame, landmarks)
                else:
                    right_face, left_face = frame.copy(), frame.copy()
                    debug_frame = frame.copy()

                right_writer.write(right_face.astype(np.uint8))
                left_writer.write(left_face.astype(np.uint8))
                debug_writer.write(debug_frame)

            except Exception as e:
                if self.debug_mode:
                    self.logger.warning(f"\nError processing frame {frame_count}: {str(e)}")
                right_writer.write(frame)
                left_writer.write(frame)
                debug_writer.write(frame)

            frame_count += 1

        # Clean up
        print(f"\nProcessing complete: {frame_count} frames processed")
        cap.release()
        right_writer.release()
        left_writer.release()
        debug_writer.release()

        # Determine the list of output files
        output_files = [str(anatomical_right_output), str(anatomical_left_output), str(debug_output)]
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
        root = tk.Tk()
        root.withdraw()
        input_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not input_paths:
            return

        output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        splitter = StableFaceSplitter(debug_mode=True)
        results = []

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