import cv2
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox


class FaceSplitter:
    def __init__(self):
        cascade_file = 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_video(self, input_path):
        """Process a video file to create mirrored versions of each face half."""
        # Input validation and setup
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_dir = input_path.parent
        left_output = output_dir / f"{input_path.stem}_left_mirrored.mp4"
        right_output = output_dir / f"{input_path.stem}_right_mirrored.mp4"

        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writers with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        left_writer = cv2.VideoWriter(str(left_output), fourcc, fps, (width, height))
        right_writer = cv2.VideoWriter(str(right_output), fourcc, fps, (width, height))

        # Process frame by frame
        frame_count = 0
        prev_midpoint = None
        default_midpoint = width // 2  # Default to center of frame

        self.logger.info(f"Processing {input_path.name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                self.logger.info(f"Processing frame {frame_count}/{total_frames}")

            try:
                # Find face and midpoint with fallback to previous or default
                midpoint = self._find_face_midpoint(frame, prev_midpoint, default_midpoint)
                prev_midpoint = midpoint  # Update previous midpoint

                # Create and write mirrored frames
                left_frame, right_frame = self._create_mirrored_frames(frame, midpoint, width)

                left_writer.write(left_frame)
                right_writer.write(right_frame)

            except Exception as e:
                self.logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                # Use previous midpoint or default for this frame
                midpoint = prev_midpoint if prev_midpoint is not None else default_midpoint
                left_frame, right_frame = self._create_mirrored_frames(frame, midpoint, width)
                left_writer.write(left_frame)
                right_writer.write(right_frame)

        # Cleanup
        cap.release()
        left_writer.release()
        right_writer.release()

        self.logger.info(f"Completed processing {input_path.name}")
        return str(left_output), str(right_output)

    def _find_face_midpoint(self, frame, prev_midpoint, default_midpoint):
        """Find the midpoint of the face with fallback options."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(frame.shape[1] / 5), int(frame.shape[0] / 5))
        )

        if len(faces) > 0:
            # Use largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            current_midpoint = x + w // 2

            # Smooth transition if we have a previous midpoint
            if prev_midpoint is not None:
                return int(0.8 * prev_midpoint + 0.2 * current_midpoint)
            return current_midpoint
        else:
            # Fallback to previous midpoint or default
            return prev_midpoint if prev_midpoint is not None else default_midpoint

    def _create_mirrored_frames(self, frame, midpoint, width):
        """Create mirrored frames ensuring consistent width."""
        # Ensure midpoint is within frame bounds
        midpoint = max(0, min(midpoint, width))

        # Split frame
        left_half = frame[:, :midpoint]
        right_half = frame[:, midpoint:]

        # Calculate padding needed
        left_pad = width - left_half.shape[1] * 2
        right_pad = width - right_half.shape[1] * 2

        # Create mirrored versions with padding if needed
        if left_pad > 0:
            left_mirrored = np.hstack([
                left_half,
                cv2.flip(left_half, 1),
                np.zeros((frame.shape[0], left_pad, 3), dtype=frame.dtype)
            ])
        else:
            left_mirrored = np.hstack([
                left_half,
                cv2.flip(left_half, 1)[:, :width - left_half.shape[1]]
            ])

        if right_pad > 0:
            right_mirrored = np.hstack([
                cv2.flip(right_half, 1),
                right_half,
                np.zeros((frame.shape[0], right_pad, 3), dtype=frame.dtype)
            ])
        else:
            right_mirrored = np.hstack([
                cv2.flip(right_half, 1)[:, -(width - right_half.shape[1]):],
                right_half
            ])

        # Ensure both frames are exactly the right width
        left_mirrored = left_mirrored[:, :width]
        right_mirrored = right_mirrored[:, :width]

        return left_mirrored, right_mirrored


def select_video_files():
    """Open a file dialog to select multiple video files"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    return file_paths if file_paths else None


def main():
    """GUI-based usage for multiple files"""
    try:
        # Select input video files
        input_paths = select_video_files()
        if not input_paths:
            print("No video files selected. Exiting...")
            return

        # Process each video
        splitter = FaceSplitter()
        results = []

        for input_path in input_paths:
            try:
                left_video, right_video = splitter.process_video(input_path)
                results.append({
                    'input': input_path,
                    'success': True,
                    'outputs': (left_video, right_video)
                })
            except Exception as e:
                results.append({
                    'input': input_path,
                    'success': False,
                    'error': str(e)
                })

        # Show completion message with summary
        root = tk.Tk()
        root.withdraw()

        summary = "Processing Results:\n\n"
        for result in results:
            if result['success']:
                summary += f"✓ {Path(result['input']).name}\n"
            else:
                summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

        messagebox.showinfo("Processing Complete", summary)

    except Exception as e:
        # Show error message for overall process failure
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Error processing videos: {str(e)}")
        logging.error(f"Error processing videos: {str(e)}")


if __name__ == "__main__":
    main()