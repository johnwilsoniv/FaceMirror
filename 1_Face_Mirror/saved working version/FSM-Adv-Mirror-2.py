import cv2
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import dlib
from scipy.spatial import Delaunay

class StableFaceSplitter:
    def __init__(self, debug_mode=False):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)

        # Cache for face detection results
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0

        # Smoothing parameters for midline stability
        self.midline_history = []
        self.history_size = 5

        self.debug_mode = debug_mode
        if debug_mode:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

    def get_face_mesh(self, frame, detection_interval=5):
        """Get facial landmarks with caching for performance"""
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
            landmarks = self.predictor(gray, self.last_face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            triangles = Delaunay(points)
            self.last_landmarks = points
            return points, triangles

        return None, None

    def get_stable_midline(self, landmarks, frame_width):
        """Calculate a stable vertical midline using nose bridge and chin"""
        if landmarks is None:
            return frame_width // 2

        # Use nose bridge (point 27) and chin (point 8)
        nose_bridge = landmarks[27][0]  # Top of nose bridge
        chin = landmarks[8][0]  # Chin point

        # Weighted average (60% nose bridge, 40% chin)
        current_mid = int(0.6 * nose_bridge + 0.4 * chin)

        # Add to history for temporal smoothing
        self.midline_history.append(current_mid)
        if len(self.midline_history) > self.history_size:
            self.midline_history.pop(0)

        # Calculate smooth midline
        smooth_mid = int(np.mean(self.midline_history))

        # Ensure midline stays within reasonable bounds
        margin = frame_width // 4
        return max(margin, min(frame_width - margin, smooth_mid))

    def create_mirrored_faces(self, frame, landmarks):
        """Create mirrored faces from left and right halves"""
        height, width = frame.shape[:2]

        # Get stable midline
        x_mid = self.get_stable_midline(landmarks, width)

        # Create left face (left half mirrored)
        left_face = np.zeros_like(frame)
        # Copy left half
        left_half = frame[:, :x_mid]
        # Create mirrored right half by flipping the left half
        left_half_flipped = cv2.flip(left_half, 1)
        # Calculate padding needed
        pad_width = width - (2 * x_mid)
        if pad_width > 0:
            # Pad the flipped half if needed
            left_half_flipped = np.pad(left_half_flipped, ((0, 0), (0, pad_width), (0, 0)), mode='edge')
        else:
            # Crop the flipped half if needed
            left_half_flipped = left_half_flipped[:, :width - x_mid]
        # Combine halves
        left_face[:, :x_mid] = left_half
        left_face[:, x_mid:] = left_half_flipped

        # Create right face (right half mirrored)
        right_face = np.zeros_like(frame)
        # Copy right half
        right_half = frame[:, x_mid:]
        # Create mirrored left half by flipping the right half
        right_half_flipped = cv2.flip(right_half, 1)
        # Calculate padding needed
        pad_width = x_mid - (width - x_mid)
        if pad_width > 0:
            # Pad the flipped half if needed
            right_half_flipped = np.pad(right_half_flipped, ((0, 0), (0, pad_width), (0, 0)), mode='edge')
        else:
            # Crop the flipped half if needed
            right_half_flipped = right_half_flipped[:, -x_mid:]
        # Combine halves
        right_face[:, x_mid:] = right_half
        right_face[:, :x_mid] = right_half_flipped

        # Apply gradient blending at the midline
        blend_width = 10  # Width of blending region in pixels

        # Create gradient masks
        gradient = np.linspace(0, 1, blend_width)
        gradient_mask = np.tile(gradient, (height, 1))

        # Ensure blend regions don't exceed image boundaries
        left_blend_start = max(0, x_mid - blend_width)
        left_blend_end = min(width, x_mid)
        right_blend_start = max(0, x_mid)
        right_blend_end = min(width, x_mid + blend_width)

        # Apply blending for left face
        if left_blend_start < left_blend_end:
            blend_region_width = left_blend_end - left_blend_start
            local_gradient = np.tile(np.linspace(0, 1, blend_region_width), (height, 1))
            left_face[:, left_blend_start:left_blend_end] = (
                    left_face[:, left_blend_start:left_blend_end] * (1 - local_gradient[..., np.newaxis]) +
                    frame[:, left_blend_start:left_blend_end] * local_gradient[..., np.newaxis]
            )

        # Apply blending for right face
        if right_blend_start < right_blend_end:
            blend_region_width = right_blend_end - right_blend_start
            local_gradient = np.tile(np.linspace(0, 1, blend_region_width), (height, 1))
            right_face[:, right_blend_start:right_blend_end] = (
                    right_face[:, right_blend_start:right_blend_end] * local_gradient[..., np.newaxis] +
                    frame[:, right_blend_start:right_blend_end] * (1 - local_gradient[..., np.newaxis])
            )

        return left_face, right_face

    def create_debug_frame(self, frame, landmarks):
        """Create debug visualization with landmarks and stable midline"""
        debug_frame = frame.copy()

        if landmarks is not None:
            # Draw only nose bridge and chin landmarks
            nose_bridge_point = tuple(landmarks[27].astype(int))
            chin_point = tuple(landmarks[8].astype(int))

            cv2.circle(debug_frame, nose_bridge_point, 4, (0, 255, 0), -1)
            cv2.circle(debug_frame, chin_point, 4, (0, 255, 0), -1)

            # Draw stable midline
            x_mid = self.get_stable_midline(landmarks, frame.shape[1])
            height = frame.shape[0]
            cv2.line(debug_frame, (x_mid, 0), (x_mid, height), (0, 0, 255), 2)

            # Draw face boundary
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.polylines(debug_frame, [hull], True, (255, 255, 0), 2)

        return debug_frame

    def process_video(self, input_path, output_dir):
        """Process video with stable face mirroring"""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        left_output = output_dir / f"{input_path.stem}_left_mirrored.mp4"
        right_output = output_dir / f"{input_path.stem}_right_mirrored.mp4"
        debug_output = output_dir / f"{input_path.stem}_debug.mp4"

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        left_writer = cv2.VideoWriter(str(left_output), fourcc, fps, (width, height))
        right_writer = cv2.VideoWriter(str(right_output), fourcc, fps, (width, height))
        debug_writer = cv2.VideoWriter(str(debug_output), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                landmarks, _ = self.get_face_mesh(frame)

                if landmarks is not None:
                    left_face, right_face = self.create_mirrored_faces(frame, landmarks)
                    debug_frame = self.create_debug_frame(frame, landmarks)
                else:
                    left_face, right_face = frame.copy(), frame.copy()
                    debug_frame = frame.copy()

                left_writer.write(left_face.astype(np.uint8))
                right_writer.write(right_face.astype(np.uint8))
                debug_writer.write(debug_frame)

            except Exception as e:
                if self.debug_mode:
                    self.logger.warning(f"Error processing frame: {str(e)}")
                left_writer.write(frame)
                right_writer.write(frame)
                debug_writer.write(frame)

        cap.release()
        left_writer.release()
        right_writer.release()
        debug_writer.release()

        return str(left_output), str(right_output), str(debug_output)


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
                left_video, right_video, debug_video = splitter.process_video(input_path, output_dir)
                results.append({
                    'input': input_path,
                    'success': True,
                    'outputs': (left_video, right_video, debug_video)
                })
            except Exception as e:
                results.append({
                    'input': input_path,
                    'success': False,
                    'error': str(e)
                })

        summary = f"Processing Results:\n\n"
        for result in results:
            if result['success']:
                summary += f"✓ {Path(result['input']).name}\n"
                summary += f"  - Left: {Path(result['outputs'][0]).name}\n"
                summary += f"  - Right: {Path(result['outputs'][1]).name}\n"
                summary += f"  - Debug: {Path(result['outputs'][2]).name}\n"
            else:
                summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

        messagebox.showinfo("Processing Complete", summary)

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()