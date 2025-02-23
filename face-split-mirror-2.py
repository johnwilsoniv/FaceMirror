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


class ProgressWindow:
    def __init__(self, title="Processing Videos", command_queue=None):
        self.root = tk.Tk()
        self.root.title(title)
        self.command_queue = command_queue

        # Setup periodic queue check
        self.check_queue()

        # Set window size and position
        window_width, window_height = 400, 150
        x = (self.root.winfo_screenwidth() - window_width) // 2
        y = (self.root.winfo_screenheight() - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Create UI elements
        self.setup_ui()

        # Initialize progress variables
        self.total_frames = 0
        self.current_frame = 0
        self.current_file = ""
        self.file_number = 0
        self.total_files = 0

        # Configure window properties
        self.root.transient()
        self.root.lift()
        self.root.resizable(False, False)

    def setup_ui(self):
        """Create and layout UI elements"""
        self.label = ttk.Label(self.root, text="Initializing...", padding=(10, 5))
        self.progress = ttk.Progressbar(self.root, length=300, mode='determinate')
        self.status = ttk.Label(self.root, text="", padding=(10, 5))

        self.label.pack()
        self.progress.pack(pady=10)
        self.status.pack()

    def set_total_files(self, total):
        self.total_files = total
        self.file_number = 0

    def update_file_progress(self, filename, total_frames):
        self.current_file = filename
        self.total_frames = total_frames
        self.current_frame = 0
        self.file_number += 1
        self.update_display()

    def increment_progress(self):
        self.current_frame += 1
        if self.current_frame % 30 == 0:  # Reduced update frequency
            self.update_display()

    def update_display(self):
        if self.total_frames > 0:
            progress = (self.current_frame / self.total_frames) * 100
            self.progress['value'] = progress

            self.label['text'] = f"{Path(self.current_file).name}"
            self.status['text'] = (f"Processing file {self.file_number} of {self.total_files}\n"
                                   f"Frame {self.current_frame} of {self.total_frames} ({progress:.1f}%)")

        self.root.update()

    def check_queue(self):
        """Check for commands in the queue"""
        try:
            if self.command_queue:
                while True:
                    try:
                        command = self.command_queue.get_nowait()
                        if command == "close":
                            self.root.quit()
                            return
                    except queue.Empty:
                        break
        except tk.TclError:
            return

        # Schedule next check
        self.root.after(100, self.check_queue)

    def close(self):
        """Schedule window closure through the event loop"""
        if self.command_queue:
            self.command_queue.put("close")
        else:
            self.root.quit()


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

    def get_comprehensive_rotation(self, video_path):
        """
        Get comprehensive rotation information from video metadata using ffprobe.
        Returns rotation angle and whether conversion is needed.
        """
        try:
            # First check - direct rotation metadata
            cmd1 = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream_tags=rotate',
                '-of', 'json',
                str(video_path)
            ]

            # Second check - side data rotation
            cmd2 = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream_side_data_list',
                '-of', 'json',
                str(video_path)
            ]

            # Third check - container metadata
            cmd3 = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format_tags=rotate',
                '-of', 'json',
                str(video_path)
            ]

            rotation_angle = 0
            needs_conversion = False

            # Check 1: Stream Tags
            result = subprocess.run(cmd1, capture_output=True, text=True)
            if result.stdout:
                data = json.loads(result.stdout)
                if 'streams' in data and data['streams']:
                    tags = data['streams'][0].get('tags', {})
                    if 'rotate' in tags:
                        rotation_angle = float(tags['rotate'])
                        needs_conversion = True
                        if self.debug_mode:
                            self.logger.info(f"Found rotation in stream tags: {rotation_angle}")
                        return rotation_angle, needs_conversion

            # Check 2: Side Data
            result = subprocess.run(cmd2, capture_output=True, text=True)
            if result.stdout:
                data = json.loads(result.stdout)
                if 'streams' in data and data['streams']:
                    side_data_list = data['streams'][0].get('side_data_list', [])
                    for side_data in side_data_list:
                        if 'rotation' in side_data:
                            rotation_angle = float(side_data['rotation'])
                            needs_conversion = True
                            if self.debug_mode:
                                self.logger.info(f"Found rotation in side data: {rotation_angle}")
                            return rotation_angle, needs_conversion

            # Check 3: Format Tags
            result = subprocess.run(cmd3, capture_output=True, text=True)
            if result.stdout:
                data = json.loads(result.stdout)
                if 'format' in data and 'tags' in data['format']:
                    if 'rotate' in data['format']['tags']:
                        rotation_angle = float(data['format']['tags']['rotate'])
                        needs_conversion = True
                        if self.debug_mode:
                            self.logger.info(f"Found rotation in format tags: {rotation_angle}")
                        return rotation_angle, needs_conversion

            # If no rotation found, try to detect display matrix
            matrix_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=display_matrix',
                '-of', 'json',
                str(video_path)
            ]

            result = subprocess.run(matrix_cmd, capture_output=True, text=True)
            if result.stdout:
                data = json.loads(result.stdout)
                if 'streams' in data and data['streams']:
                    display_matrix = data['streams'][0].get('display_matrix')
                    if display_matrix:
                        # Display matrix values that indicate rotation
                        if display_matrix == '0 -1 1 1 0 0' or display_matrix == '[0, -1, 1, 1, 0, 0]':
                            rotation_angle = 90
                            needs_conversion = True
                        elif display_matrix == '-1 0 0 0 -1 1' or display_matrix == '[-1, 0, 0, 0, -1, 1]':
                            rotation_angle = 180
                            needs_conversion = True
                        elif display_matrix == '0 1 0 -1 0 1' or display_matrix == '[0, 1, 0, -1, 0, 1]':
                            rotation_angle = 270
                            needs_conversion = True

                        if needs_conversion and self.debug_mode:
                            self.logger.info(f"Found rotation in display matrix: {rotation_angle}")
                        return rotation_angle, needs_conversion

            if self.debug_mode:
                self.logger.info("No rotation metadata found")
            return 0, False

        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"Error checking rotation metadata: {str(e)}")
                self.logger.error(f"Command outputs:")
                for cmd in [cmd1, cmd2, cmd3, matrix_cmd]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        self.logger.error(f"Command {cmd}: {result.stdout}")
                        if result.stderr:
                            self.logger.error(f"Error: {result.stderr}")
                    except Exception as cmd_error:
                        self.logger.error(f"Error running command {cmd}: {str(cmd_error)}")
            return 0, False

    def fix_rotation_with_progress(self, input_path, output_dir, rotation_angle, progress_callback=None):
        """
        Fix video rotation using ffmpeg with progress updates.
        Returns the path to the rotated video.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir) / f"rotated_{input_path.name}"
            final_output = output_dir / f"rotated_{input_path.name}"

            # Get video duration for progress calculation
            duration_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(input_path)
            ]

            try:
                duration = float(subprocess.check_output(duration_cmd).decode().strip())
            except:
                duration = 0

            # Prepare ffmpeg command with transpose filter based on rotation angle
            transpose_filter = ""
            if rotation_angle == 90:
                transpose_filter = "transpose=1"  # 90 degrees clockwise
            elif rotation_angle == 180:
                transpose_filter = "transpose=2,transpose=2"  # 180 degrees
            elif rotation_angle == 270:
                transpose_filter = "transpose=2"  # 90 degrees counterclockwise

            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
            ]

            if transpose_filter:
                cmd.extend(['-vf', transpose_filter])

            cmd.extend([
                '-metadata:s:v:0', 'rotate=0',
                '-pix_fmt', 'yuv420p',
                '-progress', 'pipe:1',
                '-y',
                str(temp_output)
            ])

            if self.debug_mode:
                self.logger.info(f"Running FFmpeg command: {' '.join(cmd)}")

            # Run the FFmpeg process and wait for completion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Process progress updates while waiting for completion
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break

                if output and 'out_time_ms=' in output:
                    time_ms = int(output.split('out_time_ms=')[1].strip())
                    if duration > 0:
                        progress = (time_ms / 1000000) / duration * 100
                        if progress_callback:
                            progress_callback(min(progress, 100))

            # Wait for the process to complete
            process.wait()

            # Check if conversion was successful
            if process.returncode == 0:
                if self.debug_mode:
                    self.logger.info("Rotation correction successful")
                # Copy the temp file to the final output location
                import shutil
                shutil.copy2(temp_output, final_output)
                return str(final_output)  # Return the path as string for compatibility
            else:
                error_output = process.stderr.read()
                if self.debug_mode:
                    self.logger.error(f"FFmpeg error: {error_output}")
                return str(input_path)  # Return original path if rotation failed

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
            # Check rotation using enhanced detection
            rotation_angle, needs_conversion = self.get_comprehensive_rotation(input_path)

            if needs_conversion:
                if self.debug_mode:
                    self.logger.info(f"Detected rotation {rotation_angle}° in {input_path.name}")

                # Update progress window for conversion if available
                if self.progress_window:
                    self.progress_window.label['text'] = f"Converting {input_path.name}"
                    self.progress_window.status['text'] = "Fixing rotation..."
                    self.progress_window.progress['value'] = 0

                # Convert video and wait for completion
                rotated_path = self.fix_rotation_with_progress(
                    input_path,
                    output_dir,
                    rotation_angle,
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