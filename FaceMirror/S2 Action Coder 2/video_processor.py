# video_processor.py - Creates output video with action annotations
import cv2
import os
import numpy as np
import config
from text_overlay import add_text_overlay  # Import shared overlay function

class VideoProcessor:
    def __init__(self, input_video, action_tracker):
        """Initialize the video processor."""
        self.input_video = input_video
        self.action_tracker = action_tracker
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0

    def init_video(self):
        """Initialize the video capture."""
        self.cap = cv2.VideoCapture(self.input_video)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video {self.input_video}")
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return True

    def process_video(self, output_video, progress_callback=None):
        """
        Process the video with action overlays.

        Args:
            output_video: Path to save the output video
            progress_callback: Optional callback function for progress updates
        """
        if not self.init_video():
            return False

        # Ensure directory exists
        output_dir = os.path.dirname(output_video)
        if output_dir:  # Only create if there's a directory path
            os.makedirs(output_dir, exist_ok=True)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))

        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Get action for current frame
            action_code = self.action_tracker.get_action_for_frame(frame_index)

            # Add overlay if an action is present
            if action_code:
                # Use the shared text overlay function
                frame = add_text_overlay(frame, action_code, position="bottom-center")

            out.write(frame)
            # Explicitly delete frame to release memory immediately
            del frame
            frame_index += 1

            # Update progress
            if progress_callback and frame_index % 10 == 0:
                progress = int((frame_index / self.total_frames) * 100)
                progress_callback(progress)

        self.cap.release()
        out.release()

        print(f"Video processing complete. Output saved to {output_video}")
        return True