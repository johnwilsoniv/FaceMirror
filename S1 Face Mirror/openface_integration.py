#!/usr/bin/env python3
"""
OpenFace 3.0 integration for AU extraction from mirrored videos.
"""

import os
import cv2
import csv
import shutil
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

# Set environment variables before importing OpenFace
os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
os.environ.setdefault('TMPDIR', os.path.expanduser('~/tmp'))

# Import direct RetinaFace model (like official demo2.py)
from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
from openface.Pytorch_Retinaface.detect import load_model
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from openface.Pytorch_Retinaface.utils.box_utils import decode
from openface.Pytorch_Retinaface.data import cfg_mnet

from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor
from openface3_to_18au_adapter import OpenFace3To18AUAdapter


class OpenFace3Processor:
    """Video processor for extracting action units using OpenFace 3.0"""

    def __init__(self, device=None, weights_dir=None, confidence_threshold=0.5, nms_threshold=0.4, calculate_landmarks=False, num_threads=6):
        """
        Initialize OpenFace 3.0 models

        Args:
            device: 'cpu', 'cuda', or 'mps' for GPU acceleration (None = auto-detect)
            weights_dir: Path to weights directory (defaults to ./weights)
            confidence_threshold: Minimum confidence for face detection (default: 0.5)
            nms_threshold: NMS threshold for face detection (default: 0.4)
            calculate_landmarks: Whether to calculate 98-point landmarks (for AU45, default: False)
            num_threads: Number of threads for parallel frame processing (default: 6, only used on CPU)
        """
        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()
        else:
            print(f"Using specified device: {device}")

        self.device = device
        self.calculate_landmarks = calculate_landmarks
        self.num_threads = num_threads

        # Cache for priorbox (optimization: avoid regenerating for same image size)
        self.priorbox_cache = {}
        self.priorbox_cache_lock = threading.Lock()

        # Determine weights directory
        if weights_dir is None:
            script_dir = Path(__file__).parent
            weights_dir = script_dir / 'weights'
        else:
            weights_dir = Path(weights_dir)

        print("Initializing OpenFace 3.0 models...")

        # Load RetinaFace model directly (like demo2.py and FaceDetector)
        self.cfg = cfg_mnet
        retinaface_model = RetinaFace(cfg=self.cfg, phase='test')
        retinaface_model = load_model(retinaface_model, str(weights_dir / 'Alignment_RetinaFace.pth'), device == 'cpu')
        retinaface_model.eval()
        self.retinaface_model = retinaface_model.to(device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        print("  ✓ RetinaFace model loaded (direct, no temp files)")

        # Conditionally initialize landmark detector
        if self.calculate_landmarks:
            self._init_landmark_detector(weights_dir, device)
        else:
            self.landmark_detector = None
            print("  ⊘ Landmark detector skipped (not required for basic AU extraction)")

        # Initialize multitask model for AU extraction
        self.multitask_model = MultitaskPredictor(
            model_path=str(weights_dir / 'MTL_backbone.pth'),
            device=device
        )
        print("  ✓ Multitask model loaded (AU extraction)")

        # Initialize AU adapter (converts 8 AUs -> 18 AUs)
        self.au_adapter = OpenFace3To18AUAdapter()
        print("  ✓ AU adapter initialized (8→18 conversion)")

        # Get AU availability report
        report = self.au_adapter.get_au_availability_report()
        print(f"\nAU Coverage: {report['total_available']}/{report['total_expected']} AUs available")
        print(f"  Available: {', '.join([au.replace('_r', '') for au in report['available_aus']])}")
        if self.calculate_landmarks:
            print(f"  Calculated: AU45 (from eye landmarks)")
        else:
            print(f"  AU45: Skipped (landmark detection disabled)")
        print(f"  Missing: {', '.join([au.replace('_r', '') for au in report['missing_aus']])}")
        print()

    def _auto_detect_device(self):
        """
        Auto-detect best available device for processing

        Note: MPS is not supported by OpenFace 3.0 models.
        Falls back to CPU on Apple Silicon.
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {device_name}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Apple MPS detected, but not supported by OpenFace 3.0 models")
            print("Falling back to CPU for compatibility")
            return 'cpu'
        else:
            print("Using CPU (no GPU detected)")
            return 'cpu'

    def _init_landmark_detector(self, weights_dir, device):
        """Initialize landmark detector (STAR) with suppressed output"""
        # Patch STAR config to use writable cache directory instead of /work
        # Also suppress verbose logging output
        import openface.STAR.conf.alignment as alignment_conf
        import logging
        import sys
        import io

        # Temporarily suppress logging and stdout during STAR initialization
        original_log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)  # Suppress INFO logs from STAR

        original_init = alignment_conf.Alignment.__init__

        def patched_init(self_inner, args):
            original_init(self_inner, args)
            # Replace hardcoded /work path with a writable location
            import os.path as osp
            home_dir = os.path.expanduser('~')
            self_inner.ckpt_dir = os.path.join(home_dir, '.cache', 'openface', 'STAR')
            self_inner.work_dir = osp.join(self_inner.ckpt_dir, self_inner.data_definition, self_inner.folder)
            self_inner.model_dir = osp.join(self_inner.work_dir, 'model')
            self_inner.log_dir = osp.join(self_inner.work_dir, 'log')
            # Create directories if they don't exist
            os.makedirs(self_inner.ckpt_dir, exist_ok=True)

            # Disable TensorBoard writer to prevent hanging
            self_inner.writer = None

        import os.path as osp
        alignment_conf.Alignment.__init__ = patched_init

        # Redirect stdout to suppress verbose initialization messages
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Initialize landmark detector for 98-point landmarks
            self.landmark_detector = LandmarkDetector(
                model_path=str(weights_dir / 'Landmark_98.pkl'),
                device=device
            )
        finally:
            # Restore stdout and logging
            sys.stdout = old_stdout
            logging.getLogger().setLevel(original_log_level)
            # Restore original __init__
            alignment_conf.Alignment.__init__ = original_init

        print("  ✓ Landmark detector loaded (98 points for AU45)")

    def preprocess_image(self, frame, resize=1.0):
        """
        Detect faces in a frame using RetinaFace (in-memory, no temp files)
        Based on official demo2.py implementation

        Args:
            frame: numpy array (BGR image from cv2.VideoCapture)
            resize: resize factor for input image (default: 1.0)

        Returns:
            List of detections, each as [x1, y1, x2, y2, confidence, ...]
        """
        img = frame.astype(np.float32)

        if resize != 1.0:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape

        # Preprocessing for RetinaFace
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        # Run detection
        with torch.no_grad():
            loc, conf, _ = self.retinaface_model(img)

        # Generate priors (CACHED - optimization to avoid regenerating for same image size)
        cache_key = (im_height, im_width)
        with self.priorbox_cache_lock:
            if cache_key not in self.priorbox_cache:
                priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(self.device)
                self.priorbox_cache[cache_key] = priors.data
            prior_data = self.priorbox_cache[cache_key]

        # Decode detections
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * torch.tensor([im_width, im_height, im_width, im_height]).to(self.device)
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # Filter by confidence threshold
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # Keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # Apply NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]

        # Scale back to original size if resized
        if resize != 1.0:
            dets[:, :4] = dets[:, :4] / resize

        return dets

    def _process_frame_worker(self, frame_data):
        """
        Process a single frame (worker function for multi-threading)

        Args:
            frame_data: tuple of (frame_index, frame, fps)

        Returns:
            tuple of (frame_index, csv_row_dict or None)
        """
        frame_index, frame, fps = frame_data
        timestamp = frame_index / fps

        try:
            # Detect faces using direct RetinaFace (in-memory, NO temp files!)
            dets = self.preprocess_image(frame)

            if dets is None or len(dets) == 0:
                # No face detected - return failed frame row
                csv_row = self._create_failed_frame_row(frame_index, timestamp)
                return (frame_index, csv_row)

            # Get detection confidence and coordinates for first (primary) face
            det = dets[0]
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            confidence = float(det[4]) if len(det) > 4 else 1.0

            # Extract face region from frame (in-memory)
            cropped_face = frame[y1:y2, x1:x2]

            # Extract 98-point landmarks if enabled (for AU45)
            landmarks_98 = None
            if self.calculate_landmarks and self.landmark_detector is not None:
                landmarks_98_list = self.landmark_detector.detect_landmarks(
                    frame,
                    dets,
                    confidence_threshold=0.5
                )
                landmarks_98 = landmarks_98_list[0] if landmarks_98_list is not None and len(landmarks_98_list) > 0 else None

                # Debug: Check if landmarks were extracted
                if frame_index % 100 == 0:  # Log every 100 frames
                    if landmarks_98 is not None:
                        print(f"  [Debug] Frame {frame_index}: Landmarks extracted, shape: {landmarks_98.shape}")
                    else:
                        print(f"  [Debug] Frame {frame_index}: No landmarks extracted")

            # Extract AUs using multitask model
            emotion_logits, gaze_output, au_output = self.multitask_model.predict(cropped_face)

            # Convert 8 AUs to 18 AUs using adapter
            csv_row = self.au_adapter.get_csv_row_dict(
                au_vector_8d=au_output,
                landmarks_98=landmarks_98,
                frame_num=frame_index,
                timestamp=timestamp,
                confidence=confidence,
                success=1
            )

            return (frame_index, csv_row)

        except Exception as e:
            # Return failed frame row on error
            csv_row = self._create_failed_frame_row(frame_index, timestamp)
            return (frame_index, csv_row)

    def process_video(self, video_path, output_csv_path, progress_callback=None):
        """
        Process a single video and extract AUs using multi-threaded processing
        (No video display - optimized for speed)

        Args:
            video_path: Path to input video file
            output_csv_path: Path to output CSV file
            progress_callback: Optional callback function(current, total, fps) for progress updates

        Returns:
            int: Number of frames successfully processed
        """
        video_path = Path(video_path)
        output_csv_path = Path(output_csv_path)

        # Ensure output directory exists
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing: {video_path.name}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        if self.device == 'cpu':
            print(f"  Using {self.num_threads} threads for parallel processing")

        # Reset AU adapter for new video
        self.au_adapter.reset()

        # Step 1: Read all frames into memory
        print("  Reading frames into memory...")
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((frame_count, frame.copy(), fps))
            frame_count += 1

        cap.release()
        print(f"  Read {len(frames)} frames")

        # Step 2: Process frames with multi-threading
        print(f"  Processing frames...")
        frame_results = {}
        processed_count = 0

        # Use ThreadPoolExecutor for CPU processing, or process sequentially for GPU
        if self.device == 'cpu':
            # Multi-threaded CPU processing
            from concurrent.futures import as_completed

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all frames
                futures = {executor.submit(self._process_frame_worker, frame_data): frame_data[0]
                          for frame_data in frames}

                # Process results with tqdm progress bar
                with tqdm(total=len(frames), desc="Extracting AUs", unit="frame",
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                    for future in as_completed(futures):
                        idx, csv_row = future.result()
                        frame_results[idx] = csv_row
                        pbar.update(1)
                        processed_count += 1

                        # Send progress updates with tqdm's rate
                        if progress_callback:
                            tqdm_rate = pbar.format_dict.get('rate', 0) or 0
                            progress_callback(processed_count, len(frames), tqdm_rate)

        else:
            # Sequential processing for GPU (to avoid GPU memory issues)
            with tqdm(total=len(frames), desc="Extracting AUs", unit="frame") as pbar:
                for frame_data in frames:
                    idx, csv_row = self._process_frame_worker(frame_data)
                    frame_results[idx] = csv_row
                    pbar.update(1)
                    processed_count += 1

                    # Send progress update with FPS calculation from tqdm
                    if progress_callback and processed_count % 10 == 0:
                        tqdm_rate = pbar.format_dict.get('rate', 0) or 0
                        progress_callback(processed_count, len(frames), tqdm_rate)

        # Step 3: Write CSV rows in correct order
        csv_rows = []
        failed_count = 0

        for idx in sorted(frame_results.keys()):
            csv_row = frame_results[idx]
            csv_rows.append(csv_row)
            if csv_row['success'] == 0:
                failed_count += 1

        # Write CSV file
        if csv_rows:
            self._write_csv(csv_rows, output_csv_path)
            print(f"\n  ✓ Processed {len(csv_rows) - failed_count} frames successfully")
            if failed_count > 0:
                print(f"  ⚠ {failed_count} frames failed (no face detected)")
            print(f"  Output: {output_csv_path}")
            success_count = len(csv_rows) - failed_count
        else:
            print(f"  ✗ No frames were processed")
            success_count = 0

        # Aggressive memory cleanup after each video
        import gc
        frames.clear()
        frame_results.clear()
        csv_rows.clear()
        del frames
        del frame_results
        del csv_rows

        # Force garbage collection
        gc.collect()

        # Clear PyTorch cache if using GPU
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return success_count

    def clear_cache(self):
        """
        Clear priorbox cache to free memory

        Call this if processing many videos with different resolutions
        and want to free up cache memory between batches.
        """
        with self.priorbox_cache_lock:
            self.priorbox_cache.clear()

    def _create_failed_frame_row(self, frame_num, timestamp):
        """
        Create a CSV row for a frame where face detection failed

        Args:
            frame_num: Frame number
            timestamp: Timestamp in seconds

        Returns:
            dict: CSV row with all AUs set to NaN
        """
        # Create dummy AU vector (all zeros, will become NaN)
        dummy_au_8d = np.zeros(8)

        return self.au_adapter.get_csv_row_dict(
            au_vector_8d=dummy_au_8d,
            landmarks_98=None,  # No landmarks available
            frame_num=frame_num,
            timestamp=timestamp,
            confidence=0.0,
            success=0
        )

    def _write_csv(self, csv_rows, output_path):
        """
        Write CSV file in OpenFace 2.0-compatible format

        Args:
            csv_rows: List of row dictionaries
            output_path: Path to output CSV file
        """
        if not csv_rows:
            return

        # Get column names from first row (maintains order)
        fieldnames = list(csv_rows[0].keys())

        # Write CSV
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def process_videos(directory_path, specific_files=None, output_dir=None):
    """
    Process video files in the given directory that end with 'mirrored',
    ignoring files that end with 'debug'.

    Args:
        directory_path (str): Path to the directory containing video files
        specific_files (list, optional): List of specific files to process.
                                         If None, all eligible files in the directory will be processed.
        output_dir (str, optional): Output directory for CSV files.
                                   Defaults to S1O Processed Files/Combined Data/

    Returns:
        int: Number of files that were processed
    """
    directory_path = Path(directory_path)

    # Check if directory exists
    if not directory_path.is_dir():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return 0

    # Determine output directory
    if output_dir is None:
        # Default: S1O Processed Files/Combined Data/
        s1o_base = directory_path.parent.parent / 'S1O Processed Files'
        output_dir = s1o_base / 'Combined Data'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize OpenFace 3.0 processor
    device = 'cpu'  # Use 'cuda' if GPU available
    processor = OpenFace3Processor(device=device)

    # Counter for processed files
    processed_count = 0

    # Define which files to process
    files_to_process = []

    if specific_files:
        # Process only the specific files
        files_to_process = [Path(f) for f in specific_files]
        print(f"Processing {len(files_to_process)} specific files from current session.")
    else:
        # Process all eligible files in the directory
        files_to_process = list(directory_path.iterdir())
        print(f"Processing all eligible files in {directory_path}")

    # Process each file
    for file_path in files_to_process:
        # Skip if not a file or doesn't exist
        if not file_path.is_file():
            print(f"Warning: {file_path} does not exist or is not a file. Skipping.")
            continue

        filename = file_path.name

        # Skip files with 'debug' in the filename
        if 'debug' in filename:
            print(f"Skipping debug file: {filename}")
            continue

        # Process file with 'mirrored' in the filename
        if 'mirrored' in filename:
            # Generate output CSV filename
            # Example: "video_left_mirrored.mp4" -> "video_left_mirrored.csv"
            csv_filename = file_path.stem + '.csv'
            output_csv_path = output_dir / csv_filename

            try:
                # Process video and extract AUs
                frame_count = processor.process_video(file_path, output_csv_path)

                if frame_count > 0:
                    processed_count += 1
                    print(f"✓ Successfully processed: {filename}\n")
                else:
                    print(f"✗ Failed to process: {filename}\n")

            except Exception as e:
                print(f"✗ Error processing {filename}: {e}\n")

    print(f"\nProcessing complete. {processed_count} files were processed.")
    return processed_count
