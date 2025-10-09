import cv2
import numpy as np
from pathlib import Path
import logging
from video_rotation import process_video_rotation
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# Suppress PyTorch meshgrid warning
warnings.filterwarnings('ignore', message='torch.meshgrid: in an upcoming release')

class VideoProcessor:
    """Handles video file processing with face mirroring"""
    
    def __init__(self, landmark_detector, face_mirror, debug_mode=False, num_threads=6):
        """Initialize with references to landmark detector and face mirror

        Args:
            landmark_detector: Face landmark detector instance
            face_mirror: Face mirroring instance
            debug_mode: Enable debug logging
            num_threads: Number of threads for parallel frame processing (default: 6)
        """
        self.landmark_detector = landmark_detector
        self.face_mirror = face_mirror
        self.debug_mode = debug_mode
        self.num_threads = num_threads

        if debug_mode:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

    def _process_frame_batch(self, frame_data):
        """Process a single frame (worker function for threading)

        Args:
            frame_data: tuple of (frame_index, frame)

        Returns:
            tuple of (frame_index, right_face, left_face, debug_frame)
        """
        frame_index, frame = frame_data

        try:
            # Process frame
            landmarks, _ = self.landmark_detector.get_face_mesh(frame)

            if landmarks is not None:
                right_face, left_face = self.face_mirror.create_mirrored_faces(frame, landmarks)
                debug_frame = self.face_mirror.create_debug_frame(frame, landmarks)
            else:
                right_face, left_face = frame.copy(), frame.copy()
                debug_frame = frame.copy()

            return (frame_index, right_face, left_face, debug_frame)

        except Exception as e:
            if self.debug_mode:
                print(f"\nError processing frame {frame_index}: {str(e)}")
            return (frame_index, frame.copy(), frame.copy(), frame.copy())
    
    def process_video(self, input_path, output_dir):
        """Process video file with progress tracking"""
        # Reset tracking history at the start of each video
        self.landmark_detector.reset_tracking_history()

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

        # Read all frames into memory first
        print(f"\nReading {total_frames} frames into memory...")
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((frame_count, frame.copy()))
            frame_count += 1

        cap.release()
        print(f"Read {len(frames)} frames")

        # Process frames with threading and tqdm progress bar
        print(f"\nProcessing frames (using {self.num_threads} threads)...")
        frame_results = {}

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all frames and wrap futures with tqdm
            futures = {executor.submit(self._process_frame_batch, frame_data): frame_data[0]
                      for frame_data in frames}

            # Process results with tqdm progress bar
            with tqdm(total=len(frames), desc="Processing", unit="frame",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for future in as_completed(futures):
                    idx, right, left, debug = future.result()
                    frame_results[idx] = (right, left, debug)
                    pbar.update(1)

        # Write results in order
        print("\nWriting frames to output files...")
        for write_idx in tqdm(sorted(frame_results.keys()), desc="Writing", unit="frame"):
            right_face, left_face, debug_frame = frame_results[write_idx]
            right_writer.write(right_face.astype(np.uint8))
            left_writer.write(left_face.astype(np.uint8))
            debug_writer.write(debug_frame)

        # Clean up
        print(f"\nProcessing complete: {len(frame_results)} frames processed")
        right_writer.release()
        left_writer.release()
        debug_writer.release()

        # Print performance statistics
        self.landmark_detector.print_performance_summary()

        # Determine the list of output files
        output_files = [str(anatomical_right_output), str(anatomical_left_output), str(debug_output)]
        if str(rotated_input_path) != str(input_path):
            output_files.append(str(rotated_input_path))

        print("\nOutput files:")
        for f in output_files:
            print(f"- {Path(f).name}")
        print("")

        return output_files
