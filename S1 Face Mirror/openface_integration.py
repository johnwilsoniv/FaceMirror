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
import time
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager

# Set environment variables before importing OpenFace
os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
os.environ.setdefault('TMPDIR', os.path.expanduser('~/tmp'))

# ============================================================================
# OPTIMIZED THREADING: Balanced system-level parallelism
# ============================================================================
# With pre-loaded frames and parallel batch processing, we can safely allow
# more system threads for better CPU utilization.
#
# Settings:
# - OMP/OPENBLAS: 2 threads (balanced for multi-threaded application)
# - Application uses 6 worker threads for frame processing
# - ONNX Runtime uses 2 intra-op threads
#
# Total: 6 workers × (2 OMP + 2 ONNX) = ~24 threads max (good for 10-core systems)
# ============================================================================
os.environ.setdefault('OMP_NUM_THREADS', '2')  # Allow limited OpenMP parallelism
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')  # Allow limited BLAS parallelism

# ============================================================================
# GARBAGE COLLECTION OPTIMIZATION
# ============================================================================
# Increase GC threshold0 from default 700 to 10,000 to reduce GC overhead
# Research shows this can reduce GC utilization from ~3% to ~0.5% of runtime
# Thresholds: (gen0_threshold, gen1_threshold, gen2_threshold)
gc.set_threshold(10000, 10, 10)

# Batch processing configuration for memory efficiency
# Same rationale as video_processor.py - prevents loading entire video into memory
# BATCH_SIZE = 100 recommended for 16-32 GB systems (~1.2 GB per batch)
BATCH_SIZE = 100

# Import direct RetinaFace model (like official demo2.py)
from openface.Pytorch_Retinaface.models.retinaface import RetinaFace
from openface.Pytorch_Retinaface.detect import load_model
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from openface.Pytorch_Retinaface.utils.box_utils import decode
from openface.Pytorch_Retinaface.data import cfg_mnet

# Import optimized ONNX STAR detector with automatic fallback to PyTorch
try:
    from onnx_star_detector import OptimizedLandmarkDetector as LandmarkDetector
    USING_ONNX_LANDMARK_DETECTION = True
except ImportError:
    from openface.landmark_detection import LandmarkDetector
    USING_ONNX_LANDMARK_DETECTION = False

# Import optimized ONNX MTL predictor with automatic fallback to PyTorch
try:
    from onnx_mtl_detector import OptimizedMultitaskPredictor as MultitaskPredictor
    USING_ONNX_MTL = True
except ImportError:
    from openface.multitask_model import MultitaskPredictor
    USING_ONNX_MTL = False

from openface3_to_18au_adapter import OpenFace3To18AUAdapter


class OpenFace3Processor:
    """Video processor for extracting action units using OpenFace 3.0"""

    def __init__(self, device=None, weights_dir=None, confidence_threshold=0.5, nms_threshold=0.4, calculate_landmarks=False, num_threads=6, debug_mode=False):
        """
        Initialize OpenFace 3.0 models

        Args:
            device: 'cpu', 'cuda', or 'mps' for GPU acceleration (None = auto-detect)
            weights_dir: Path to weights directory (defaults to ./weights)
            confidence_threshold: Minimum confidence for face detection (default: 0.5)
            nms_threshold: NMS threshold for face detection (default: 0.4)
            calculate_landmarks: Whether to calculate 98-point landmarks (for AU45, default: False)
            num_threads: Number of threads for parallel frame processing (default: 6, only used on CPU)
            debug_mode: Enable debug logging and performance summaries (default: False)
        """
        # Flag to track if we're processing pre-cropped mirrored videos
        self.skip_face_detection = False
        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()
        else:
            print(f"Using specified device: {device}")

        # Force CPU if MPS is specified (OpenFace 3.0 doesn't support MPS)
        # ONNX models run on CPU via ONNX Runtime + CoreML anyway
        if device == 'mps':
            print("Note: MPS not supported by OpenFace 3.0 models, using CPU instead")
            print("(ONNX models will still use CoreML Neural Engine acceleration)")
            device = 'cpu'

        self.device = device
        self.calculate_landmarks = calculate_landmarks
        self.num_threads = num_threads
        self.debug_mode = debug_mode

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
        print("  RetinaFace model loaded (direct, no temp files)")

        # Conditionally initialize landmark detector
        if self.calculate_landmarks:
            self._init_landmark_detector(weights_dir, device)
        else:
            self.landmark_detector = None
            print("  ⊘ Landmark detector skipped (not required for basic AU extraction)")

        # Initialize multitask model for AU extraction
        # OptimizedMultitaskPredictor auto-selects ONNX (fast) or PyTorch (slow)
        self.multitask_model = MultitaskPredictor(
            model_path=str(weights_dir / 'MTL_backbone.pth'),
            onnx_model_path=str(weights_dir / 'mtl_efficientnet_b0_coreml.onnx'),
            device=device
        )

        # Report backend
        if hasattr(self.multitask_model, 'backend'):
            backend = self.multitask_model.backend
            if backend == 'onnx':
                print("  Multitask model loaded (ONNX-accelerated AU extraction)")
                if hasattr(self.multitask_model.predictor, 'backend'):
                    onnx_backend = self.multitask_model.predictor.backend
                    if onnx_backend == 'coreml':
                        print("    Using CoreML Neural Engine acceleration")
                    else:
                        print("    Using ONNX CPU (optimized)")
            else:
                print("  Multitask model loaded (PyTorch - slower)")
                print("    To enable acceleration, run: ./run_mtl_conversion.sh")
        else:
            print("  Multitask model loaded (AU extraction)")

        # Initialize AU adapter (converts 8 AUs -> 18 AUs)
        self.au_adapter = OpenFace3To18AUAdapter()
        print("  AU adapter initialized (8→18 conversion)")

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

        # Warm up models and freeze for GC optimization
        self._warmup_models()

    def _warmup_models(self):
        """
        Warm up models with realistic inference to trigger CoreML/Neural Engine compilation.
        Also freeze models from garbage collection to reduce GC overhead.

        This eliminates first-batch latency spikes (500ms-2s reduction).
        """
        print("Warming up AU extraction models...")

        # Create realistic dummy frame (480p resolution with realistic pixel values)
        # Use gray values around 128 to simulate a real face
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        # Add some variation to trigger proper model compilation
        dummy_frame[100:380, 200:440] = 160  # Brighter "face" region

        try:
            # Warm up face detection with multiple passes to ensure CoreML compilation
            for _ in range(3):  # Multiple passes ensure CoreML is fully initialized
                dets = self.preprocess_image(dummy_frame)

            # Warm up multitask model with realistic face crop
            # Create a synthetic 128x128 face crop (typical size)
            synthetic_face = np.full((128, 128, 3), 140, dtype=np.uint8)
            synthetic_face[20:108, 20:108] = 160  # Inner face region

            # Warm up with multiple passes
            for _ in range(3):
                _ = self.multitask_model.predict(synthetic_face)

            # Warm up landmark detector if enabled
            if self.calculate_landmarks and self.landmark_detector is not None:
                # Create fake detections for landmark warm-up
                fake_dets = np.array([[200, 100, 440, 380, 0.99]])
                for _ in range(2):
                    _ = self.landmark_detector.detect_landmarks(
                        dummy_frame, fake_dets, confidence_threshold=0.5
                    )

            print("  AU extraction models warmed up (CoreML/Neural Engine ready)")
        except Exception as e:
            print(f"  Warm-up completed with warnings (models will still work)")

        # Run garbage collection once
        # NOTE: We do NOT call gc.freeze() here because it can prevent proper cleanup of
        # ONNX Runtime / CoreML sessions, leading to process hanging on exit
        gc.collect()

        print("  Models initialized and ready")

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

            # CRITICAL: Close and disable TensorBoard writer to prevent process hanging on exit
            # TensorBoard writers have background threads that must be properly terminated
            if hasattr(self_inner, 'writer') and self_inner.writer is not None:
                try:
                    self_inner.writer.close()
                except:
                    pass
            self_inner.writer = None

        import os.path as osp
        alignment_conf.Alignment.__init__ = patched_init

        # Redirect stdout to suppress verbose initialization messages
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Initialize landmark detector for 98-point landmarks
            # Use ONNX-optimized STAR detector for 5-7x speedup (same as mirroring)
            self.landmark_detector = LandmarkDetector(
                model_path=str(weights_dir / 'Landmark_98.pkl'),
                onnx_model_path=str(weights_dir / 'star_landmark_98_coreml.onnx'),
                device=device
            )
        finally:
            # Restore stdout and logging
            sys.stdout = old_stdout
            logging.getLogger().setLevel(original_log_level)
            # Restore original __init__
            alignment_conf.Alignment.__init__ = original_init

        # Report backend
        if hasattr(self.landmark_detector, 'backend'):
            backend = self.landmark_detector.backend
            if backend == 'onnx':
                print("  Landmark detector loaded (ONNX-accelerated for AU45)")
                print("    Expected: 10-20x speedup (~90-180ms per frame)")
            else:
                print("  Landmark detector loaded (PyTorch - slower)")
                print("    To enable acceleration, run: ./run_onnx_conversion.sh")
        elif USING_ONNX_LANDMARK_DETECTION:
            print("  Landmark detector loaded (ONNX acceleration module loaded)")
        else:
            print("  Landmark detector loaded (98 points for AU45)")

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
            # ========================================================================
            # OPTIMIZATION: Skip face detection for pre-cropped mirrored videos
            # ========================================================================
            # Mirrored videos are already face-cropped and aligned from the mirror
            # step. We can use the entire frame as the face crop, saving ~167ms per
            # frame (RetinaFace detection time).
            # ========================================================================
            if self.skip_face_detection:
                # Mirrored video: entire frame is the face crop
                cropped_face = frame
                confidence = 1.0  # High confidence since we know it's a face

                # ================================================================
                # OPTIMIZATION: Skip RetinaFace but run STAR for AU45 landmarks
                # ================================================================
                # Create full-frame bounding box and pass to STAR
                # This skips ~160ms RetinaFace detection but still gets AU45
                # ================================================================
                landmarks_98 = None

                # DIAGNOSTIC: Log AU45 calculation decision (only once per video)
                if frame_index == 0:
                    detector_type = type(self.landmark_detector).__name__ if self.landmark_detector is not None else 'None'
                    detector_backend = getattr(self.landmark_detector, 'backend', 'unknown') if self.landmark_detector is not None else 'N/A'
                    print(f"  [AU45 Diagnostic] calculate_landmarks={self.calculate_landmarks}")
                    print(f"  [AU45 Diagnostic] landmark_detector type: {detector_type}")
                    print(f"  [AU45 Diagnostic] detector backend: {detector_backend}")

                if self.calculate_landmarks and self.landmark_detector is not None:
                    # Create full-frame bounding box (entire frame is the face)
                    h, w = frame.shape[:2]
                    full_frame_det = np.array([[0, 0, w, h, 1.0]])  # x1, y1, x2, y2, confidence

                    # DIAGNOSTIC: Log STAR call on first frame
                    if frame_index == 0:
                        print(f"  [AU45 Diagnostic] About to call detect_landmarks() for AU45...")

                    # Run STAR for 98-point landmarks (for AU45)
                    landmarks_98_list = self.landmark_detector.detect_landmarks(
                        frame,
                        full_frame_det,
                        confidence_threshold=0.5
                    )

                    # DIAGNOSTIC: Log result
                    if frame_index == 0:
                        result_status = "SUCCESS" if (landmarks_98_list is not None and len(landmarks_98_list) > 0) else "FAILED"
                        print(f"  [AU45 Diagnostic] detect_landmarks() returned: {result_status}")

                    landmarks_98 = landmarks_98_list[0] if landmarks_98_list is not None and len(landmarks_98_list) > 0 else None
            else:
                # Original pipeline: detect faces using RetinaFace
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

        # ========================================================================
        # OPTIMIZATION: Detect if this is a pre-cropped mirrored video
        # ========================================================================
        # Mirrored videos (from the mirroring step) are already face-cropped and
        # aligned. We can skip RetinaFace detection entirely, saving ~167ms per
        # frame (3-4x speedup for AU extraction!).
        # ========================================================================
        if 'mirrored' in video_path.name.lower() and 'debug' not in video_path.name.lower():
            self.skip_face_detection = True
            print(f"Processing: {video_path.name}")
            print(f"  Detected pre-cropped mirrored video")
            print(f"  Skipping face detection (already aligned)")
            print(f"  Expected speedup: 3-4x faster AU extraction")
        else:
            self.skip_face_detection = False
            print(f"Processing: {video_path.name}")

        # Ensure output directory exists
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # OPTIMIZATION: Set minimal buffer size to reduce latency between batches
        # Default buffer can hold many frames, causing delays when transitioning
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        if self.device == 'cpu':
            print(f"  Using {self.num_threads} threads for parallel processing")

        # Reset AU adapter for new video
        self.au_adapter.reset()

        # ============================================================================
        # BATCH PROCESSING: Process video in batches to prevent memory exhaustion
        # ============================================================================
        # Same strategy as video_processor.py for memory efficiency
        # ============================================================================

        print(f"  Processing video in batches of {BATCH_SIZE} frames...")

        # Send initial progress update to GUI
        if progress_callback:
            progress_callback(0, total_frames, 0.0)

        global_frame_count = 0  # Track progress across all batches
        total_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE
        csv_rows = []  # Accumulate CSV rows across all batches
        failed_count = 0

        # Timing accumulators for performance analysis
        total_read_time = 0.0
        total_process_time = 0.0
        total_store_time = 0.0
        total_cleanup_time = 0.0

        # ============================================================================
        # BATCH READING OPTIMIZATION: Pre-read frames into memory
        # ============================================================================
        # NEW APPROACH: Read entire batch into memory FIRST, then process in parallel.
        # This eliminates cv2.VideoCapture threading conflicts because the VideoCapture
        # object is only used in the main thread, not in worker threads.
        #
        # Memory usage: BATCH_SIZE * frame_size_bytes
        # For 1080p video: 100 frames * 1920 * 1080 * 3 = ~600MB per batch
        # ============================================================================

        def read_batch_preload(cap_obj, start_frame_idx, max_frames, video_fps):
            """
            Pre-read a batch of frames into memory for parallel processing.

            This function reads ALL frames in the batch into memory first,
            then returns them as a list. This allows worker threads to process
            frames without touching the cv2.VideoCapture object.
            """
            batch = []
            for _ in range(max_frames):
                ret, frame = cap_obj.read()
                if not ret:
                    break
                # Store complete frame data (index, frame copy, fps)
                batch.append((start_frame_idx + len(batch), frame.copy(), video_fps))
            return batch

        # Main processing loop with tqdm for terminal output (matches mirroring)
        try:
            # Create red tqdm progress bar (matches mirroring style)
            pbar = tqdm(total=total_frames, desc="Extracting AUs", unit="frame",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                       colour='red')  # Red progress bar to distinguish from mirroring (blue)
            batch_num = 0
            prev_batch_end = time.time()  # Track time between batches

            # Read first batch (PRE-LOAD into memory)
            read_start = time.time()
            current_batch = read_batch_preload(cap, global_frame_count, BATCH_SIZE, fps)
            global_frame_count += len(current_batch)
            read_elapsed = time.time() - read_start
            total_read_time += read_elapsed

            while current_batch:
                batch_num += 1
                batch_start_time = time.time()

                # ============ STEP 2: Process current batch IN PARALLEL ============
                process_start = time.time()
                batch_results = {}

                # ============================================================================
                # PARALLEL PROCESSING ENABLED - Frames are pre-loaded in memory!
                # ============================================================================
                # Now that frames are pre-read into memory, we can safely use parallel
                # processing without cv2.VideoCapture threading conflicts.
                #
                # Worker threads process frame data from memory (not from VideoCapture).
                # This gives us 6x parallelization speedup!
                # ============================================================================
                if True:  # ENABLED: Parallel processing with pre-loaded frames
                    # Multi-threaded CPU processing
                    from concurrent.futures import as_completed

                    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                        # Submit all frames in current batch
                        futures = {executor.submit(self._process_frame_worker, frame_data): frame_data[0]
                                  for frame_data in current_batch}

                        # Collect results as they complete
                        for future in as_completed(futures):
                            idx, csv_row = future.result()
                            batch_results[idx] = csv_row

                            # UPDATE PROGRESS BAR IMMEDIATELY as each frame completes
                            if pbar is not None:
                                pbar.update(1)

                            # Send progress updates every 10 frames (matches mirroring code)
                            if (idx + 1) % 10 == 0:
                                # Get accurate FPS from tqdm's rolling average
                                tqdm_rate = pbar.format_dict.get('rate', 0) or 0 if pbar is not None else 0

                                # Send to GUI callback (reduced from every frame to every 10 frames)
                                if progress_callback and tqdm_rate > 0:
                                    try:
                                        progress_callback(idx + 1, total_frames, tqdm_rate)
                                    except Exception:
                                        # GUI callback failed (window might be closing), continue processing
                                        pass
                else:
                    # Sequential processing (avoid enumerate to prevent iterator deadlock with ONNX/CoreML threads)
                    batch_size = len(current_batch)
                    for i in range(batch_size):
                        frame_data = current_batch[i]
                        idx, csv_row = self._process_frame_worker(frame_data)
                        batch_results[idx] = csv_row

                        # UPDATE PROGRESS BAR IMMEDIATELY as each frame completes
                        if pbar is not None:
                            pbar.update(1)

                        # Send progress updates every 10 frames (matches mirroring code)
                        if (idx + 1) % 10 == 0:
                            # Get accurate FPS from tqdm's rolling average
                            tqdm_rate = pbar.format_dict.get('rate', 0) or 0 if pbar is not None else 0

                            # Send to GUI callback (matches mirroring pattern)
                            if progress_callback and tqdm_rate > 0:
                                try:
                                    progress_callback(idx + 1, total_frames, tqdm_rate)
                                except Exception:
                                    # GUI callback failed (window might be closing), continue processing
                                    pass

                process_elapsed = time.time() - process_start
                total_process_time += process_elapsed

                # ============ STEP 3: Store CSV rows in order ============
                # Accumulate CSV rows for final write (they're small, unlike video frames)
                store_start = time.time()

                for idx in sorted(batch_results.keys()):
                    csv_row = batch_results[idx]
                    csv_rows.append(csv_row)
                    if csv_row['success'] == 0:
                        failed_count += 1

                store_elapsed = time.time() - store_start
                total_store_time += store_elapsed

                # ============ STEP 4: Clear batch and read next batch ============
                cleanup_start = time.time()

                # Read next batch (PRE-LOAD into memory for next iteration)
                if global_frame_count < total_frames:
                    read_start = time.time()
                    next_batch = read_batch_preload(cap, global_frame_count, BATCH_SIZE, fps)
                    global_frame_count += len(next_batch)
                    read_elapsed = time.time() - read_start
                    total_read_time += read_elapsed
                else:
                    next_batch = []

                # Clear current batch
                current_batch.clear()
                batch_results.clear()
                del current_batch
                del batch_results

                # Move next batch to current
                current_batch = next_batch

                # OPTIMIZED GC: Only run full collection every 20 batches (reduced from 5)
                # This reduces GC overhead from ~3% to ~0.5% while maintaining memory efficiency
                gc_start = time.time()
                if batch_num % 20 == 0:
                    gc.collect()
                gc_elapsed = time.time() - gc_start

                # DIAGNOSTIC: Check if GC is the bottleneck
                if gc_elapsed > 0.5:
                    print(f"\n  WARNING: Batch {batch_num} GC took {gc_elapsed:.2f}s!")

                cleanup_elapsed = time.time() - cleanup_start
                total_cleanup_time += cleanup_elapsed

                prev_batch_end = time.time()

                # Log performance summary every 10 batches (reduced from 5 to minimize overhead)
                # Only show during debug mode to avoid console spam
                if self.debug_mode and batch_num % 10 == 0:
                    print(f"\n  [OpenFace Performance Summary - Batch {batch_num}/{total_batches}]")
                    print(f"    Read:    {total_read_time:.2f}s ({total_read_time/batch_num:.3f}s/batch)")
                    print(f"    Process: {total_process_time:.2f}s ({total_process_time/batch_num:.3f}s/batch)")
                    print(f"    Store:   {total_store_time:.2f}s ({total_store_time/batch_num:.3f}s/batch)")
                    print(f"    Cleanup: {total_cleanup_time:.2f}s ({total_cleanup_time/batch_num:.3f}s/batch)")
                    total_time = total_read_time + total_process_time + total_store_time + total_cleanup_time
                    print(f"    Total:   {total_time:.2f}s")
                    if total_time > 0:
                        print(f"    Breakdown: Read {total_read_time/total_time*100:.1f}% | "
                              f"Process {total_process_time/total_time*100:.1f}% | "
                              f"Store {total_store_time/total_time*100:.1f}% | "
                              f"Cleanup {total_cleanup_time/total_time*100:.1f}%")

        finally:
            # Properly close tqdm progress bar to ensure clean stdout state
            try:
                if 'pbar' in locals() and pbar is not None:
                    pbar.close()  # Properly close to restore stdout
                    # Force flush both stdout and stderr to ensure TQDM fully releases them
                    sys.stdout.flush()
                    sys.stderr.flush()
            except Exception:
                pass  # Ignore errors

        # Clean up video capture
        cap.release()

        # Print detailed timing breakdown (only if no GUI callback)
        if not progress_callback:
            print(f"\n{'='*60}")
            print("OPENFACE AU EXTRACTION PERFORMANCE BREAKDOWN")
            print(f"{'='*60}")
            total_time = total_read_time + total_process_time + total_store_time + total_cleanup_time
            print(f"Total processing time: {total_time:.2f}s")
            print(f"  Read frames:    {total_read_time:>8.2f}s ({total_read_time/total_time*100:>5.1f}%)")
            print(f"  Process frames: {total_process_time:>8.2f}s ({total_process_time/total_time*100:>5.1f}%)")
            print(f"  Store results:  {total_store_time:>8.2f}s ({total_store_time/total_time*100:>5.1f}%)")
            print(f"  Cleanup:        {total_cleanup_time:>8.2f}s ({total_cleanup_time/total_time*100:>5.1f}%)")
            print(f"\nAverage FPS: {global_frame_count/total_time:.1f} frames/sec")
            print(f"  Read:    {global_frame_count/total_read_time:.1f} fps")
            print(f"  Process: {global_frame_count/total_process_time:.1f} fps")
            print(f"{'='*60}\n")

        # Print completion message (clear the progress line first)
        print()  # Newline to clear the \r progress line

        # Write final CSV file
        if csv_rows:
            self._write_csv(csv_rows, output_csv_path)

            # Always print completion (even with GUI)
            print(f"  Processed {len(csv_rows) - failed_count} frames successfully")
            if failed_count > 0:
                print(f"  {failed_count} frames failed (no face detected)")
            print(f"  Output: {output_csv_path}")
            success_count = len(csv_rows) - failed_count
        else:
            print(f"  No frames were processed")
            success_count = 0

        # Final memory cleanup
        csv_rows.clear()
        del csv_rows
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
                    print(f"Successfully processed: {filename}\n")
                else:
                    print(f"Failed to process: {filename}\n")

            except Exception as e:
                print(f"Error processing {filename}: {e}\n")

    print(f"\nProcessing complete. {processed_count} files were processed.")
    return processed_count
