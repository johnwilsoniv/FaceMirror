import cv2
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import dlib
from scipy.spatial import Delaunay
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import subprocess
import tempfile
import json

from progress_window import ProgressWindow
from video_utils import get_comprehensive_rotation, fix_rotation_with_progress


def process_videos_thread(input_paths, output_dir, progress_window, command_queue):
    """Process videos with optimized threading"""
    try:
        splitter = StableFaceSplitter(debug_mode=True, progress_window=progress_window)
        results = []

        progress_window.set_total_files(len(input_paths))

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

        # Show results
        summary = "\n".join([
            "Processing Results:\n",
            *[
                f"✓ {Path(r['input']).name}\n  - Left: {Path(r['outputs'][0]).name}\n  - Right: {Path(r['outputs'][1]).name}\n  - Debug: {Path(r['outputs'][2]).name}"
                if r['success'] else
                f"✗ {Path(r['input']).name} - Error: {r['error']}"
                for r in results]
        ])

        # Schedule window closure and show results
        progress_window.root.after(100, lambda: messagebox.showinfo("Processing Complete", summary))
        progress_window.close()

    except Exception as e:
        progress_window.close()
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


class StableFaceSplitter:
    def __init__(self, debug_mode=False, progress_window=None):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)

        # Pre-allocate buffers
        self.frame_buffer = None
        self.left_face_buffer = None
        self.right_face_buffer = None
        self.debug_frame_buffer = None

        # Cache and history
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0
        self.midline_history = []
        self.history_size = 5

        # Pre-compute blend gradients
        self.blend_width = 10
        self.gradient = np.linspace(0, 1, self.blend_width)

        self.debug_mode = debug_mode
        self.progress_window = progress_window

        if debug_mode:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

    def initialize_buffers(self, height, width):
        """Initialize reusable frame buffers"""
        self.frame_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.left_face_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.right_face_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.debug_frame_buffer = np.zeros((height, width, 3), dtype=np.uint8)

    def get_face_mesh(self, frame, detection_interval=5):
        """Optimized face mesh detection with caching"""
        self.frame_count += 1

        # Use cached landmarks when possible
        if self.frame_count % detection_interval != 0 and self.last_landmarks is not None:
            return self.last_landmarks, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            self.last_face = None
            self.last_landmarks = None
            return None, None

        self.last_face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self.predictor(gray, self.last_face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        self.last_landmarks = points

        return points, None

    def get_stable_midline(self, landmarks, frame_width):
        """Calculate stable midline using vectorized operations"""
        if landmarks is None:
            return frame_width // 2

        # Vectorized landmark calculations
        current_mid = int(0.6 * landmarks[27, 0] + 0.4 * landmarks[8, 0])

        self.midline_history.append(current_mid)
        if len(self.midline_history) > self.history_size:
            self.midline_history.pop(0)

        smooth_mid = int(np.mean(self.midline_history))
        margin = frame_width // 4

        return np.clip(smooth_mid, margin, frame_width - margin)

    def create_mirrored_faces(self, frame, landmarks):
        """Create mirrored faces using optimized numpy operations"""
        height, width = frame.shape[:2]
        x_mid = self.get_stable_midline(landmarks, width)

        # Reuse pre-allocated buffers
        np.copyto(self.left_face_buffer, 0)
        np.copyto(self.right_face_buffer, 0)

        # Extract and flip sections
        left_half = frame[:, :x_mid]
        right_half = frame[:, x_mid:]
        left_flipped = cv2.flip(left_half, 1)
        right_flipped = cv2.flip(right_half, 1)

        # Assign sections using direct indexing
        self.left_face_buffer[:, :x_mid] = left_half
        self.left_face_buffer[:, x_mid:] = left_flipped[:, :(width - x_mid)]
        self.right_face_buffer[:, x_mid:] = right_half
        self.right_face_buffer[:, :x_mid] = right_flipped[:, -x_mid:]

        # Vectorized blending
        blend_mask = np.tile(self.gradient, (height, 1))
        blend_mask_3d = blend_mask[..., np.newaxis]

        blend_region = slice(x_mid - self.blend_width, x_mid)
        self.left_face_buffer[:, blend_region] = (
                self.left_face_buffer[:, blend_region] * (1 - blend_mask_3d) +
                frame[:, blend_region] * blend_mask_3d
        )

        blend_region = slice(x_mid, x_mid + self.blend_width)
        self.right_face_buffer[:, blend_region] = (
            self.right_face_buffer[:, blend_region] * blend_mask_3d +
                frame[:, blend_region] * (1 - blend_mask_3d)
        )

        return self.left_face_buffer, self.right_face_buffer

    def create_debug_frame(self, frame, landmarks):
        """Create debug visualization with minimal operations"""
        np.copyto(self.debug_frame_buffer, frame)

        if landmarks is not None:
            # Draw key points
            nose_bridge_point = tuple(landmarks[27].astype(int))
            chin_point = tuple(landmarks[8].astype(int))
            cv2.circle(self.debug_frame_buffer, nose_bridge_point, 4, (0, 255, 0), -1)
            cv2.circle(self.debug_frame_buffer, chin_point, 4, (0, 255, 0), -1)

            # Draw midline
            x_mid = self.get_stable_midline(landmarks, frame.shape[1])
            cv2.line(self.debug_frame_buffer, (x_mid, 0), (x_mid, frame.shape[0]), (0, 0, 255), 2)

            # Draw face boundary
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.polylines(self.debug_frame_buffer, [hull], True, (255, 255, 0), 2)

        return self.debug_frame_buffer

    def process_video(self, input_path, output_dir):
        """Process video with enhanced rotation handling and progress updates"""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Using the new video_utils functions instead of class methods
            rotation_angle, needs_conversion = get_comprehensive_rotation(input_path, self.debug_mode)

            if needs_conversion:
                if self.debug_mode:
                    self.logger.info(f"Detected rotation {rotation_angle}° in {input_path.name}")

                # Update progress window for conversion if available
                if self.progress_window:
                    self.progress_window.label['text'] = f"Converting {input_path.name}"
                    self.progress_window.status['text'] = "Fixing rotation..."
                    self.progress_window.progress['value'] = 0

                # Using the new video_utils function
                rotated_path = fix_rotation_with_progress(
                    input_path,
                    output_dir,
                    rotation_angle,
                    debug_mode=self.debug_mode,
                    progress_callback=lambda p: setattr(self.progress_window.progress, 'value', p)
                    if self.progress_window else None
                )

                # Verify the rotated file exists and use it for further processing
                if Path(rotated_path).exists():
                    input_path = Path(rotated_path)
                    if self.debug_mode:
                        self.logger.info(f"Using rotated video: {input_path}")
                else:
                    if self.debug_mode:
                        self.logger.warning(f"Rotated video not found, using original: {input_path}")

            # Setup output files
            left_output = output_dir / f"{input_path.stem}_left_mirrored.mp4"
            right_output = output_dir / f"{input_path.stem}_right_mirrored.mp4"
            debug_output = output_dir / f"{input_path.stem}_debug.mp4"

            # Open the video file after rotation is complete
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"Error opening video file: {input_path}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize frame buffers
            self.initialize_buffers(height, width)

            # Update progress window
            if self.progress_window:
                self.progress_window.update_file_progress(str(input_path), total_frames)

            # Setup video writers
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writers = (
                cv2.VideoWriter(str(left_output), fourcc, fps, (width, height)),
                cv2.VideoWriter(str(right_output), fourcc, fps, (width, height)),
                cv2.VideoWriter(str(debug_output), fourcc, fps, (width, height))
            )

            try:
                # Read and process frames
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
                            left_face = frame.copy()
                            right_face = frame.copy()
                            debug_frame = frame.copy()

                        writers[0].write(left_face)
                        writers[1].write(right_face)
                        writers[2].write(debug_frame)

                        if self.progress_window:
                            self.progress_window.increment_progress()

                    except Exception as e:
                        if self.debug_mode:
                            self.logger.warning(f"Error processing frame: {str(e)}")
                        for writer in writers:
                            writer.write(frame)
                        if self.progress_window:
                            self.progress_window.increment_progress()

            finally:
                # Cleanup
                cap.release()
                for writer in writers:
                    writer.release()

            return str(left_output), str(right_output), str(debug_output)

        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"Error in process_video: {str(e)}")
            raise


def main():
    """Optimized command-line interface with safe thread handling"""
    try:
        # Suppress macOS warnings
        import os
        os.environ['TK_SILENCE_DEPRECATION'] = '1'

        root = tk.Tk()
        root.withdraw()

        if hasattr(root, 'createcommand'):
            root.createcommand('tk::mac::OpenDocument', lambda *args: None)
            root.createcommand('tk::mac::ReopenApplication', lambda *args: None)
            root.createcommand('tk::mac::ShowPreferences', lambda *args: None)

        input_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not input_paths:
            return

        output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create command queue for thread communication
        command_queue = queue.Queue()

        # Create progress window and start processing thread
        progress_window = ProgressWindow(command_queue=command_queue)
        processing_thread = threading.Thread(
            target=process_videos_thread,
            args=(input_paths, output_dir, progress_window, command_queue)
        )
        processing_thread.start()

        # Start progress window's main loop
        progress_window.root.mainloop()

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()