import cv2
import numpy as np
from pathlib import Path
import logging
from video_rotation import process_video_rotation
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import time
import sys

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

    def _process_frame_batch(self, frame_data, progress_queue=None):
        """Process a single frame (worker function for threading)

        Args:
            frame_data: tuple of (frame_index, frame)
            progress_queue: Queue for sending progress updates

        Returns:
            tuple of (frame_index, right_face, left_face, debug_frame)
        """
        frame_index, frame = frame_data

        try:
            # Notify that processing has started
            if progress_queue:
                progress_queue.put(('started', frame_index))

            # Process frame
            landmarks, _ = self.landmark_detector.get_face_mesh(frame)

            if landmarks is not None:
                right_face, left_face = self.face_mirror.create_mirrored_faces(frame, landmarks)
                debug_frame = self.face_mirror.create_debug_frame(frame, landmarks)
            else:
                right_face, left_face = frame.copy(), frame.copy()
                debug_frame = frame.copy()

            # Notify that processing has completed
            if progress_queue:
                progress_queue.put(('completed', frame_index))

            return (frame_index, right_face, left_face, debug_frame)

        except Exception as e:
            if self.debug_mode:
                print(f"\nError processing frame {frame_index}: {str(e)}")
            if progress_queue:
                progress_queue.put(('error', frame_index))
            return (frame_index, frame.copy(), frame.copy(), frame.copy())
    
    def _progress_monitor(self, progress_queue, total_frames, stop_event):
        """Monitor progress updates from workers and display them

        Args:
            progress_queue: Queue receiving progress updates
            total_frames: Total number of frames to process
            stop_event: Event to signal monitoring should stop
        """
        frames_completed = 0
        frames_started = 0
        frames_written = 0
        last_update_time = time.time()
        start_time = time.time()

        while not stop_event.is_set() or not progress_queue.empty():
            try:
                # Non-blocking get with timeout
                event_type, frame_index = progress_queue.get(timeout=0.1)

                if event_type == 'started':
                    frames_started += 1
                elif event_type == 'completed':
                    frames_completed += 1
                elif event_type == 'written':
                    frames_written = frame_index

                # Update every 10 seconds OR when all frames are completed
                current_time = time.time()
                time_since_last_update = current_time - last_update_time

                if time_since_last_update >= 10.0 or frames_completed == total_frames:
                    elapsed = current_time - start_time

                    # Use completed frames for progress if available, otherwise use started frames
                    active_frames = frames_completed if frames_completed > 0 else frames_started
                    fps = active_frames / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_frames - active_frames) / fps if fps > 0 else 0

                    # Format ETA as "45m 36s"
                    eta_minutes = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)
                    eta_formatted = f"{eta_minutes}m {eta_secs:02d}s"

                    # Calculate percentage
                    percent = (active_frames / total_frames) * 100

                    # Print with newline for log compatibility
                    status = "completed" if frames_completed > 0 else "processing"
                    print(f"Progress: {percent:.1f}% ({active_frames}/{total_frames} frames {status}) | "
                          f"{fps:.2f} fps | ETA: {eta_formatted}")
                    sys.stdout.flush()
                    last_update_time = current_time

            except Exception:
                # Queue empty or timeout - check if we should update
                current_time = time.time()
                time_since_last_update = current_time - last_update_time

                if time_since_last_update >= 10.0 and (frames_started > 0 or frames_completed > 0):
                    elapsed = current_time - start_time
                    active_frames = frames_completed if frames_completed > 0 else frames_started
                    fps = active_frames / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_frames - active_frames) / fps if fps > 0 else 0

                    eta_minutes = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)
                    eta_formatted = f"{eta_minutes}m {eta_secs:02d}s"

                    percent = (active_frames / total_frames) * 100
                    status = "completed" if frames_completed > 0 else "processing"

                    print(f"Progress: {percent:.1f}% ({active_frames}/{total_frames} frames {status}) | "
                          f"{fps:.2f} fps | ETA: {eta_formatted}")
                    sys.stdout.flush()
                    last_update_time = current_time

                if stop_event.is_set():
                    break
                continue

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

        # Process frames with threading for parallelization
        frame_count = 0
        frames_written = 0
        print(f"\nProcessing frames (using {self.num_threads} threads)...")
        print("Starting frame processing with live progress updates...")
        sys.stdout.flush()

        # Create progress queue and monitoring thread
        progress_queue = Queue(maxsize=200)  # Allow some buffering of progress events
        stop_event = threading.Event()

        monitor_thread = threading.Thread(
            target=self._progress_monitor,
            args=(progress_queue, total_frames, stop_event),
            daemon=False  # Changed from True - we need to wait for this thread to finish printing
        )
        monitor_thread.start()

        # Read all frames first (or use batching for very long videos)
        batch_size = 30  # Process frames in batches of 30
        frame_batch = []
        frame_results = {}  # Store results indexed by frame number

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in batch
                    if frame_batch:
                        futures = {executor.submit(self._process_frame_batch, (idx, frm), progress_queue): (idx, frm)
                                  for idx, frm in frame_batch}
                        for future in as_completed(futures):
                            idx, right, left, debug = future.result()
                            frame_results[idx] = (right, left, debug)
                    break

                frame_batch.append((frame_count, frame.copy()))

                # Process batch when it reaches batch_size
                if len(frame_batch) >= batch_size:
                    # Submit all frames in batch to thread pool with progress queue
                    futures = {executor.submit(self._process_frame_batch, (idx, frm), progress_queue): (idx, frm)
                              for idx, frm in frame_batch}

                    # Collect results as they complete
                    for future in as_completed(futures):
                        idx, right, left, debug = future.result()
                        frame_results[idx] = (right, left, debug)

                    # Write results in order
                    write_start = frame_count - batch_size + 1
                    for write_idx in range(write_start, frame_count + 1):
                        if write_idx in frame_results:
                            right_face, left_face, debug_frame = frame_results[write_idx]
                            right_writer.write(right_face.astype(np.uint8))
                            left_writer.write(left_face.astype(np.uint8))
                            debug_writer.write(debug_frame)
                            del frame_results[write_idx]  # Free memory
                            frames_written += 1
                            progress_queue.put(('written', frames_written))

                    frame_batch = []  # Clear batch

                frame_count += 1

        # Write any remaining results
        for write_idx in sorted(frame_results.keys()):
            right_face, left_face, debug_frame = frame_results[write_idx]
            right_writer.write(right_face.astype(np.uint8))
            left_writer.write(left_face.astype(np.uint8))
            debug_writer.write(debug_frame)
            frames_written += 1
            progress_queue.put(('written', frames_written))

        # Signal monitor thread to stop and wait for it
        stop_event.set()
        monitor_thread.join(timeout=10.0)  # Increased timeout to ensure all messages are printed

        # Clean up
        print(f"\nProcessing complete: {frame_count} frames processed")
        cap.release()
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
