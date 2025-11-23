#!/usr/bin/env python3
"""
Production-Ready AU Pipeline with Debug Operations Disabled
Removes all unnecessary logging, print statements, and debug operations
for maximum performance.

Expected improvements:
- Remove print statements in hot loops
- Disable verbose logging during initialization
- Remove debug mode checks
- Streamline initialization messages
"""

import cv2
import numpy as np
import time
import hashlib
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, List
from collections import deque
from functools import lru_cache
import os

# Disable all print statements globally for production
PRODUCTION_MODE = True
SILENT_MODE = os.environ.get('AU_PIPELINE_SILENT', 'false').lower() == 'true'

# Add local packages to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))
sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pyfaceau'))

# Monkey-patch print to be silent in production
_original_print = print
def production_print(*args, **kwargs):
    if not PRODUCTION_MODE or not SILENT_MODE:
        return _original_print(*args, **kwargs)

# Apply monkey patch
if PRODUCTION_MODE:
    print = production_print


class ProductionAUPipeline:
    """
    Production-ready AU pipeline with all optimizations and debug disabled.
    """

    def __init__(self,
                 cache_size: int = 32,
                 temporal_window: int = 5,
                 skip_detection_interval: int = 3,
                 verbose: bool = False,  # Default to False for production
                 production_mode: bool = True):
        """
        Initialize production pipeline.

        Args:
            cache_size: Number of frames to cache features for
            temporal_window: Window size for temporal smoothing
            skip_detection_interval: Skip face detection every N frames
            verbose: Print initialization info (disabled by default)
            production_mode: Run in production mode (no debug output)
        """
        self.verbose = verbose and not production_mode
        self.production_mode = production_mode
        self.cache_size = cache_size
        self.skip_detection_interval = skip_detection_interval

        if self.verbose:
            print("Initializing Production AU Pipeline...")

        # Suppress all initialization messages
        import warnings
        warnings.filterwarnings('ignore')

        # Redirect stdout temporarily during initialization
        if self.production_mode:
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            # Initialize components
            from pymtcnn import MTCNN
            from pyclnf import CLNF
            from pyfaceau import FullPythonAUPipeline

            init_start = time.perf_counter()

            # Face detector - disable verbose output
            self.detector = MTCNN()

            # CLNF landmark detector - disable debug mode
            self.clnf = CLNF(
                model_dir="pyclnf/models",
                max_iterations=5,
                convergence_threshold=0.5,
                debug_mode=False  # Explicitly disable debug
            )

            # AU pipeline - disable verbose output
            self.au_pipeline = FullPythonAUPipeline(
                pdm_file="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
                au_models_dir="pyfaceau/weights/AU_predictors",
                triangulation_file="pyfaceau/weights/tris_68_full.txt",
                patch_expert_file="pyfaceau/weights/svr_patches_0.25_general.txt",
                verbose=False  # Disable verbose output
            )

            init_time = (time.perf_counter() - init_start) * 1000

        finally:
            # Restore stdout
            if self.production_mode:
                sys.stdout = old_stdout

        # Feature cache
        self.feature_cache = {}
        self.au_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Temporal coherence tracking
        self.bbox_history = deque(maxlen=temporal_window)
        self.landmark_history = deque(maxlen=temporal_window)
        self.au_history = deque(maxlen=temporal_window)
        self.frame_count = 0
        self.last_detection_frame = -999

        # Performance tracking (minimal in production)
        self.timing_history = {
            'total': deque(maxlen=100)
        } if not self.production_mode else None

        if self.verbose:
            print(f"Pipeline initialized in {init_time:.1f}ms")

    def compute_frame_hash(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Compute perceptual hash of face region for cache lookup.
        """
        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]

        # Resize to 16x16 for hash
        small = cv2.resize(face_region, (16, 16))

        # Convert to grayscale
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Compute average hash
        avg = np.mean(small)
        hash_bits = (small > avg).flatten()
        return hashlib.md5(hash_bits.tobytes()).hexdigest()

    def should_skip_detection(self) -> bool:
        """
        Determine if we should skip face detection based on temporal stability.
        """
        # Always detect if we don't have history
        if len(self.bbox_history) < 2:
            return False

        # Check if enough frames since last detection
        frames_since_detection = self.frame_count - self.last_detection_frame
        if frames_since_detection < self.skip_detection_interval:
            # Check stability of recent bboxes
            if len(self.bbox_history) >= 3:
                recent_bboxes = list(self.bbox_history)[-3:]
                x_vals = [b[0] for b in recent_bboxes]
                y_vals = [b[1] for b in recent_bboxes]

                x_var = np.var(x_vals)
                y_var = np.var(y_vals)

                # Skip if stable
                return x_var < 25 and y_var < 25

        return False

    def predict_bbox_from_history(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Predict next bbox from history using simple linear motion.
        """
        if len(self.bbox_history) < 2:
            return None

        prev_bbox = self.bbox_history[-2]
        curr_bbox = self.bbox_history[-1]

        # Simple motion prediction
        dx = curr_bbox[0] - prev_bbox[0]
        dy = curr_bbox[1] - prev_bbox[1]

        predicted_bbox = (
            curr_bbox[0] + dx,
            curr_bbox[1] + dy,
            curr_bbox[2],
            curr_bbox[3]
        )

        return predicted_bbox

    def smooth_aus(self, current_aus: Dict, alpha: float = 0.7) -> Dict:
        """
        Apply temporal smoothing to AU predictions.
        """
        if len(self.au_history) == 0:
            return current_aus

        prev_aus = self.au_history[-1]
        smoothed = {}

        for au_name in current_aus:
            if au_name in prev_aus:
                smoothed[au_name] = alpha * current_aus[au_name] + (1 - alpha) * prev_aus[au_name]
            else:
                smoothed[au_name] = current_aus[au_name]

        return smoothed

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame with all optimizations and no debug output.

        Returns:
            Dictionary containing:
            - 'bbox': Face bounding box
            - 'landmarks': 68 facial landmarks
            - 'aus': Action unit predictions
            - 'timing': Component timings (only if not in production mode)
        """
        self.frame_count += 1
        result = {
            'bbox': None,
            'landmarks': None,
            'aus': None
        }

        if not self.production_mode:
            result['timing'] = {}
            result['cache_hit'] = False

        total_start = time.perf_counter()

        # 1. Face Detection (with temporal optimization)
        if self.should_skip_detection():
            # Use predicted bbox
            bbox = self.predict_bbox_from_history()
        else:
            # Run actual detection
            detection = self.detector.detect(frame)
            self.last_detection_frame = self.frame_count

            if detection is not None and isinstance(detection, tuple) and len(detection) == 2:
                bboxes, confidences = detection
                if len(bboxes) > 0:
                    bbox = bboxes[0]
                    x, y, w, h = [int(v) for v in bbox]
                    bbox = (x, y, w, h)
            else:
                bbox = None

        if bbox is None:
            return result

        result['bbox'] = bbox
        self.bbox_history.append(bbox)

        # 2. Landmark Detection
        landmarks, info = self.clnf.fit(frame, bbox)

        if landmarks is None or len(landmarks) != 68:
            return result

        result['landmarks'] = landmarks
        self.landmark_history.append(landmarks)

        # 3. AU Prediction (with caching)
        # Check cache
        frame_hash = self.compute_frame_hash(frame, bbox)

        if frame_hash in self.au_cache:
            # Cache hit!
            aus = self.au_cache[frame_hash]
            self.cache_hits += 1
            if not self.production_mode:
                result['cache_hit'] = True
        else:
            # Cache miss - compute AUs
            self.cache_misses += 1

            # Use the existing AU pipeline
            au_result = self.au_pipeline._process_frame(
                frame,
                frame_idx=self.frame_count,
                timestamp=self.frame_count/30.0
            )

            aus = au_result.get('aus', {}) if au_result else {}

            # Update cache
            if len(self.au_cache) >= self.cache_size:
                # Remove oldest
                oldest = next(iter(self.au_cache))
                del self.au_cache[oldest]
            self.au_cache[frame_hash] = aus

        # Apply temporal smoothing
        if len(self.au_history) > 0:
            aus = self.smooth_aus(aus)

        result['aus'] = aus
        self.au_history.append(aus)

        # Timing (only if not in production mode)
        if not self.production_mode:
            total_time = (time.perf_counter() - total_start) * 1000
            result['timing']['total'] = total_time
            self.timing_history['total'].append(total_time)

        return result

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics (minimal in production mode).
        """
        stats = {}

        # Cache stats
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            stats['cache_hit_rate'] = self.cache_hits / total_requests * 100
        else:
            stats['cache_hit_rate'] = 0

        # FPS (only if tracking timing)
        if self.timing_history and len(self.timing_history['total']) > 0:
            mean_time = np.mean(self.timing_history['total'])
            stats['fps'] = 1000.0 / mean_time
            stats['mean_frame_time'] = mean_time

        return stats


def benchmark_production_pipeline():
    """
    Benchmark the production pipeline with debug disabled.
    """
    print("=" * 60)
    print("PRODUCTION PIPELINE BENCHMARK")
    print("=" * 60)

    # Test with debug enabled (baseline)
    print("\n1. WITH DEBUG OUTPUT (baseline):")
    pipeline_debug = ProductionAUPipeline(
        cache_size=32,
        temporal_window=5,
        skip_detection_interval=3,
        verbose=True,
        production_mode=False
    )

    # Test with debug disabled (production)
    print("\n2. PRODUCTION MODE (silent):")

    # Set environment variable for complete silence
    os.environ['AU_PIPELINE_SILENT'] = 'true'

    pipeline_prod = ProductionAUPipeline(
        cache_size=32,
        temporal_window=5,
        skip_detection_interval=3,
        verbose=False,
        production_mode=True
    )

    # Load test video
    video_path = "Patient Data/Normal Cohort/Shorty.mov"
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Test both pipelines
    for name, pipeline in [("Debug Mode", pipeline_debug), ("Production Mode", pipeline_prod)]:
        print(f"\n{name} Test:")
        print("-" * 40)

        cap = cv2.VideoCapture(video_path)
        num_frames = 30
        frame_times = []

        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            result = pipeline.process_frame(frame)
            elapsed = (time.perf_counter() - start) * 1000
            frame_times.append(elapsed)

            # Progress update only for debug mode
            if not pipeline.production_mode and (i + 1) % 10 == 0:
                avg_time = np.mean(frame_times)
                fps = 1000.0 / avg_time
                print(f"Frame {i+1:3d}: {elapsed:6.1f}ms | Avg: {avg_time:6.1f}ms | FPS: {fps:.1f}")

        cap.release()

        # Summary
        avg_time = np.mean(frame_times)
        fps = 1000.0 / avg_time
        print(f"\nResults:")
        print(f"  Average frame time: {avg_time:.1f}ms")
        print(f"  Average FPS: {fps:.1f}")

        # Get stats
        stats = pipeline.get_performance_stats()
        if 'cache_hit_rate' in stats:
            print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")

    print("\n" + "=" * 60)
    print("EXPECTED IMPROVEMENTS")
    print("=" * 60)
    print("Production mode improvements:")
    print("  - No print statements in loops: ~5-10ms saved")
    print("  - No debug checks: ~2-5ms saved")
    print("  - No verbose logging: ~1-2ms saved")
    print("  - Total expected: 8-17ms improvement (1-2% speedup)")
    print("\nAdditional optimization opportunities:")
    print("  - Disable assertion checks: ~1-2ms")
    print("  - Use -O flag for Python: ~5% speedup")
    print("  - Compile with Cython: ~20-30% speedup")


if __name__ == "__main__":
    benchmark_production_pipeline()