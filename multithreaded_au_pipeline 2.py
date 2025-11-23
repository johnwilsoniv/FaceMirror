#!/usr/bin/env python3
"""
Multi-threaded AU Pipeline for Parallel Processing
Implements concurrent processing of multiple pipeline stages to improve throughput.

Key optimizations:
1. Parallel frame processing with thread pool
2. Asynchronous I/O for video reading
3. Pipeline stage parallelization
4. Batch processing for multiple frames

Expected speedup: 1.5-2x on multi-core systems
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import hashlib

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))


@dataclass
class FrameData:
    """Container for frame data passing through pipeline stages."""
    frame_id: int
    frame: np.ndarray
    timestamp: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    landmarks: Optional[np.ndarray] = None
    aus: Optional[Dict] = None
    processing_time: Dict = None


class PipelineStage:
    """Base class for pipeline stages."""

    def __init__(self, name: str):
        self.name = name
        self.processing_times = deque(maxlen=100)

    def process(self, data: FrameData) -> FrameData:
        """Process frame data through this stage."""
        raise NotImplementedError

    def get_avg_time(self) -> float:
        """Get average processing time for this stage."""
        return np.mean(self.processing_times) if self.processing_times else 0


class DetectionStage(PipelineStage):
    """Face detection stage."""

    def __init__(self):
        super().__init__("Detection")
        from pymtcnn import MTCNN
        self.detector = MTCNN()
        self.skip_interval = 3
        self.last_bbox = None
        self.frames_since_detection = 0

    def process(self, data: FrameData) -> FrameData:
        start = time.perf_counter()

        # Skip detection if we have a recent bbox
        if self.last_bbox and self.frames_since_detection < self.skip_interval:
            data.bbox = self.last_bbox
            self.frames_since_detection += 1
        else:
            # Run detection
            detection = self.detector.detect(data.frame)
            if detection and isinstance(detection, tuple) and len(detection) == 2:
                bboxes, _ = detection
                if len(bboxes) > 0:
                    bbox = bboxes[0]
                    x, y, w, h = [int(v) for v in bbox]
                    data.bbox = (x, y, w, h)
                    self.last_bbox = data.bbox
                    self.frames_since_detection = 0

        elapsed = (time.perf_counter() - start) * 1000
        self.processing_times.append(elapsed)
        return data


class LandmarkStage(PipelineStage):
    """Landmark detection stage."""

    def __init__(self):
        super().__init__("Landmarks")
        from pyclnf import CLNF
        self.clnf = CLNF(
            model_dir="pyclnf/models",
            max_iterations=5,
            convergence_threshold=0.5,
            debug_mode=False
        )

    def process(self, data: FrameData) -> FrameData:
        if data.bbox is None:
            return data

        start = time.perf_counter()
        landmarks, _ = self.clnf.fit(data.frame, data.bbox)
        data.landmarks = landmarks

        elapsed = (time.perf_counter() - start) * 1000
        self.processing_times.append(elapsed)
        return data


class AUPredictionStage(PipelineStage):
    """AU prediction stage with caching."""

    def __init__(self):
        super().__init__("AU Prediction")
        from pyfaceau import FullPythonAUPipeline
        self.au_pipeline = FullPythonAUPipeline(
            pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
            au_models_dir="pyfaceau/weights/AU_predictors",
            triangulation_file="pyfaceau/weights/tris_68_full.txt",
            patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
            verbose=False
        )
        self.cache = {}
        self.cache_size = 32

    def _get_frame_hash(self, frame: np.ndarray, bbox: Tuple) -> str:
        """Compute hash for frame region."""
        x, y, w, h = bbox
        region = frame[y:y+h, x:x+w]
        small = cv2.resize(region, (8, 8))
        return hashlib.md5(small.tobytes()).hexdigest()

    def process(self, data: FrameData) -> FrameData:
        if data.bbox is None or data.landmarks is None:
            return data

        start = time.perf_counter()

        # Check cache
        frame_hash = self._get_frame_hash(data.frame, data.bbox)
        if frame_hash in self.cache:
            data.aus = self.cache[frame_hash]
        else:
            # Compute AUs
            au_result = self.au_pipeline._process_frame(
                data.frame,
                frame_idx=data.frame_id,
                timestamp=data.timestamp
            )
            data.aus = au_result.get('aus', {}) if au_result else {}

            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove oldest
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[frame_hash] = data.aus

        elapsed = (time.perf_counter() - start) * 1000
        self.processing_times.append(elapsed)
        return data


class MultithreadedAUPipeline:
    """
    Multi-threaded AU pipeline with parallel processing.
    """

    def __init__(self,
                 n_workers: int = None,
                 batch_size: int = 4,
                 verbose: bool = True):
        """
        Initialize multi-threaded pipeline.

        Args:
            n_workers: Number of worker threads (None = CPU count)
            batch_size: Number of frames to process in parallel
            verbose: Print performance info
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.batch_size = batch_size
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing Multi-threaded Pipeline")
            print(f"  Workers: {self.n_workers}")
            print(f"  Batch size: {batch_size}")
            print(f"  CPU cores: {mp.cpu_count()}")

        # Initialize pipeline stages
        import warnings
        warnings.filterwarnings('ignore')

        # Redirect output during initialization
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            self.detection_stage = DetectionStage()
            self.landmark_stage = LandmarkStage()
            self.au_stage = AUPredictionStage()
        finally:
            sys.stdout = old_stdout

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers)

        # Queues for pipeline stages
        self.input_queue = queue.Queue(maxsize=batch_size * 2)
        self.detection_queue = queue.Queue(maxsize=batch_size * 2)
        self.landmark_queue = queue.Queue(maxsize=batch_size * 2)
        self.output_queue = queue.Queue(maxsize=batch_size * 2)

        # Performance tracking
        self.frame_count = 0
        self.total_times = deque(maxlen=100)

    def process_frame_sequential(self, frame_data: FrameData) -> FrameData:
        """Process single frame through all stages sequentially."""
        frame_data = self.detection_stage.process(frame_data)
        frame_data = self.landmark_stage.process(frame_data)
        frame_data = self.au_stage.process(frame_data)
        return frame_data

    def process_batch_parallel(self, frames: List[np.ndarray]) -> List[FrameData]:
        """
        Process batch of frames in parallel.

        Different stages can process different frames simultaneously.
        """
        frame_data_list = []
        for i, frame in enumerate(frames):
            frame_data = FrameData(
                frame_id=self.frame_count + i,
                frame=frame,
                timestamp=(self.frame_count + i) / 30.0,
                processing_time={}
            )
            frame_data_list.append(frame_data)

        # Submit all frames to thread pool
        futures = []
        for frame_data in frame_data_list:
            future = self.executor.submit(self.process_frame_sequential, frame_data)
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            results.append(future.result())

        self.frame_count += len(frames)
        return results

    def process_video_pipeline(self, video_path: str, max_frames: int = None):
        """
        Process video using pipelined architecture.

        Stages run in parallel threads processing different frames.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"\nProcessing {total_frames} frames with pipeline architecture...")

        # Worker threads for each stage
        def detection_worker():
            while True:
                item = self.input_queue.get()
                if item is None:
                    self.detection_queue.put(None)
                    break
                result = self.detection_stage.process(item)
                self.detection_queue.put(result)

        def landmark_worker():
            while True:
                item = self.detection_queue.get()
                if item is None:
                    self.landmark_queue.put(None)
                    break
                result = self.landmark_stage.process(item)
                self.landmark_queue.put(result)

        def au_worker():
            while True:
                item = self.landmark_queue.get()
                if item is None:
                    self.output_queue.put(None)
                    break
                result = self.au_stage.process(item)
                self.output_queue.put(result)

        # Start worker threads
        threads = [
            threading.Thread(target=detection_worker),
            threading.Thread(target=landmark_worker),
            threading.Thread(target=au_worker)
        ]

        for t in threads:
            t.start()

        # Feed frames to pipeline
        start_time = time.perf_counter()
        frames_processed = 0

        # Producer thread to read frames
        def frame_producer():
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_data = FrameData(
                    frame_id=i,
                    frame=frame,
                    timestamp=i / 30.0
                )
                self.input_queue.put(frame_data)

            # Signal end
            self.input_queue.put(None)

        producer = threading.Thread(target=frame_producer)
        producer.start()

        # Collect results
        results = []
        while True:
            result = self.output_queue.get()
            if result is None:
                break
            results.append(result)
            frames_processed += 1

            if frames_processed % 10 == 0:
                elapsed = time.perf_counter() - start_time
                fps = frames_processed / elapsed
                print(f"Processed {frames_processed}/{total_frames} frames | "
                      f"FPS: {fps:.1f}")

        # Wait for threads to complete
        producer.join()
        for t in threads:
            t.join()

        cap.release()

        # Final statistics
        total_time = time.perf_counter() - start_time
        avg_fps = frames_processed / total_time

        print(f"\n{'='*60}")
        print(f"PIPELINE PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total frames: {frames_processed}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"\nStage performance:")
        print(f"  Detection: {self.detection_stage.get_avg_time():.1f}ms")
        print(f"  Landmarks: {self.landmark_stage.get_avg_time():.1f}ms")
        print(f"  AU Prediction: {self.au_stage.get_avg_time():.1f}ms")

        return results

    def benchmark(self, video_path: str):
        """Benchmark different processing modes."""

        if not Path(video_path).exists():
            print(f"Error: Video not found at {video_path}")
            return

        print("="*60)
        print("MULTI-THREADED PIPELINE BENCHMARK")
        print("="*60)

        # Test 1: Sequential processing (baseline)
        print("\n1. SEQUENTIAL PROCESSING (baseline):")
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(20):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        start = time.perf_counter()
        for i, frame in enumerate(frames):
            frame_data = FrameData(
                frame_id=i,
                frame=frame,
                timestamp=i/30.0
            )
            _ = self.process_frame_sequential(frame_data)
        seq_time = time.perf_counter() - start
        seq_fps = len(frames) / seq_time
        print(f"  Time: {seq_time:.2f}s")
        print(f"  FPS: {seq_fps:.1f}")

        # Test 2: Batch parallel processing
        print("\n2. BATCH PARALLEL PROCESSING:")
        self.frame_count = 0
        start = time.perf_counter()
        batch_results = self.process_batch_parallel(frames)
        batch_time = time.perf_counter() - start
        batch_fps = len(frames) / batch_time
        print(f"  Time: {batch_time:.2f}s")
        print(f"  FPS: {batch_fps:.1f}")
        print(f"  Speedup: {batch_fps/seq_fps:.2f}x")

        # Test 3: Pipeline architecture
        print("\n3. PIPELINE ARCHITECTURE:")
        results = self.process_video_pipeline(video_path, max_frames=20)

        print("\n" + "="*60)
        print("EXPECTED IMPROVEMENTS")
        print("="*60)
        print("- Batch parallel: 1.3-1.5x speedup")
        print("- Pipeline architecture: 1.5-2x speedup")
        print("- With GPU acceleration: 3-5x speedup")


def main():
    """Run multi-threaded pipeline benchmark."""

    pipeline = MultithreadedAUPipeline(
        n_workers=4,
        batch_size=4,
        verbose=True
    )

    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    pipeline.benchmark(video_path)


if __name__ == "__main__":
    main()