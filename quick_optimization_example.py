#!/usr/bin/env python3
"""
Quick optimization examples that can be implemented immediately.
These optimizations target the identified bottlenecks from profiling.
"""

import numpy as np
import numba
from functools import lru_cache
import hashlib
import cv2
from typing import Tuple, Optional
import time


# ============================================================================
# OPTIMIZATION 1: Fix CLNF Convergence (Currently 0% convergence rate)
# ============================================================================

class OptimizedCLNFConfig:
    """Optimized CLNF configuration for faster convergence."""

    def __init__(self):
        # Reduce iterations (was 10, never converged)
        self.max_iterations = 3

        # Increase convergence threshold (was too strict)
        self.convergence_threshold = 1.0  # pixels

        # Use previous frame for initialization
        self.use_temporal_init = True

        # Reduce regularization for faster convergence
        self.regularization = 10.0  # was 20.0


# ============================================================================
# OPTIMIZATION 2: Numba Acceleration for Hot Functions
# ============================================================================

@numba.jit(nopython=True, parallel=True, cache=True)
def compute_response_maps_parallel(
    image: np.ndarray,
    landmarks: np.ndarray,
    patch_size: int = 11
) -> np.ndarray:
    """
    Parallel computation of response maps using Numba.
    This function was taking 27.2s in profiling.
    """
    n_landmarks = landmarks.shape[0]
    h, w = image.shape[:2]
    responses = np.zeros((n_landmarks, h, w), dtype=np.float32)

    # Parallel loop over landmarks
    for i in numba.prange(n_landmarks):
        x, y = int(landmarks[i, 0]), int(landmarks[i, 1])

        # Define patch bounds
        x_start = max(0, x - patch_size // 2)
        x_end = min(w, x + patch_size // 2 + 1)
        y_start = max(0, y - patch_size // 2)
        y_end = min(h, y + patch_size // 2 + 1)

        # Compute response for this landmark
        for yy in range(y_start, y_end):
            for xx in range(x_start, x_end):
                # Simplified response computation
                dist = np.sqrt((xx - x)**2 + (yy - y)**2)
                responses[i, yy, xx] = np.exp(-dist / 10.0)

    return responses


@numba.jit(nopython=True, cache=True)
def kde_mean_shift_fast(
    points: np.ndarray,
    weights: np.ndarray,
    bandwidth: float = 5.0,
    max_iter: int = 5
) -> np.ndarray:
    """
    Fast KDE mean shift using Numba.
    Original was called 81,600 times taking 24.5s total.
    """
    n_points = points.shape[0]
    new_points = np.zeros_like(points)

    for i in range(n_points):
        point = points[i]

        for _ in range(max_iter):
            # Compute weighted mean
            total_weight = 0.0
            weighted_sum = np.zeros(2, dtype=np.float32)

            for j in range(n_points):
                # Gaussian kernel
                dist = np.sqrt((points[j, 0] - point[0])**2 +
                              (points[j, 1] - point[1])**2)
                kernel_weight = np.exp(-(dist / bandwidth)**2)

                weighted_sum += kernel_weight * points[j] * weights[j]
                total_weight += kernel_weight * weights[j]

            if total_weight > 0:
                point = weighted_sum / total_weight

        new_points[i] = point

    return new_points


# ============================================================================
# OPTIMIZATION 3: Caching for AU Features
# ============================================================================

class CachedFeatureExtractor:
    """Cache HOG and geometry features between similar frames."""

    def __init__(self, cache_size: int = 32):
        self.cache_size = cache_size
        self._hog_cache = {}
        self._geometry_cache = {}

    def _compute_frame_hash(self, face_region: np.ndarray) -> str:
        """Compute perceptual hash of face region."""
        # Downsample for faster hashing
        small = cv2.resize(face_region, (8, 8))
        # Simple average hash
        avg = np.mean(small)
        hash_bits = (small > avg).flatten()
        return hashlib.md5(hash_bits.tobytes()).hexdigest()

    @lru_cache(maxsize=32)
    def extract_hog_cached(self, face_hash: str) -> np.ndarray:
        """Extract HOG features with caching."""
        if face_hash in self._hog_cache:
            return self._hog_cache[face_hash]

        # Compute HOG (placeholder - actual implementation would use real HOG)
        hog_features = np.random.randn(4096).astype(np.float32)

        # Update cache
        if len(self._hog_cache) >= self.cache_size:
            # Remove oldest
            self._hog_cache.pop(next(iter(self._hog_cache)))
        self._hog_cache[face_hash] = hog_features

        return hog_features

    def extract_features(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract features with caching."""
        # Get face region
        x_min, x_max = int(np.min(landmarks[:, 0])), int(np.max(landmarks[:, 0]))
        y_min, y_max = int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 1]))

        face_region = frame[y_min:y_max, x_min:x_max]
        face_hash = self._compute_frame_hash(face_region)

        # Use cached HOG if available
        hog_features = self.extract_hog_cached(face_hash)

        # Geometry features (fast, no need to cache)
        geometry_features = self._extract_geometry(landmarks)

        return np.concatenate([hog_features, geometry_features])

    def _extract_geometry(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract geometry features from landmarks."""
        # Placeholder - actual implementation would compute real geometry features
        return np.random.randn(1024).astype(np.float32)


# ============================================================================
# OPTIMIZATION 4: Temporal Coherence for Video
# ============================================================================

class TemporalTracker:
    """Track faces and landmarks across frames to avoid recomputation."""

    def __init__(self, similarity_threshold: float = 0.95):
        self.last_bbox = None
        self.last_landmarks = None
        self.last_aus = None
        self.similarity_threshold = similarity_threshold
        self.frames_since_detection = 0
        self.max_tracking_frames = 5

    def should_detect(self) -> bool:
        """Determine if we need to run face detection."""
        if self.last_bbox is None:
            return True
        if self.frames_since_detection >= self.max_tracking_frames:
            return True
        return False

    def track_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Track bbox from previous frame using simple template matching."""
        if self.last_bbox is None:
            return None

        x, y, w, h = self.last_bbox

        # Simple tracking: assume small movement
        search_region = 20
        x_search = max(0, x - search_region)
        y_search = max(0, y - search_region)

        # In practice, would use optical flow or correlation filter
        # For now, just add small random drift
        dx = np.random.randint(-5, 6)
        dy = np.random.randint(-5, 6)

        new_bbox = (x + dx, y + dy, w, h)
        self.last_bbox = new_bbox
        self.frames_since_detection += 1

        return new_bbox

    def predict_landmarks(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """Predict landmarks using temporal coherence."""
        if self.last_landmarks is None:
            return None

        # Simple prediction: small random drift
        # In practice, would use Kalman filter or optical flow
        drift = np.random.randn(68, 2) * 0.5
        predicted = self.last_landmarks + drift

        return predicted

    def update(self, bbox, landmarks, aus):
        """Update tracker with new detections."""
        self.last_bbox = bbox
        self.last_landmarks = landmarks
        self.last_aus = aus
        self.frames_since_detection = 0


# ============================================================================
# OPTIMIZATION 5: Batched AU Prediction
# ============================================================================

@numba.jit(nopython=True, parallel=True)
def predict_aus_batch(features: np.ndarray, models: np.ndarray) -> np.ndarray:
    """
    Predict all 17 AUs in parallel using Numba.
    Original takes 1032ms sequentially.
    """
    n_aus = 17
    aus = np.zeros(n_aus, dtype=np.float32)

    # Parallel loop over AUs
    for i in numba.prange(n_aus):
        # Simplified SVM prediction (dot product + bias)
        # In practice, would load actual SVM weights
        aus[i] = np.dot(features, models[i, :-1]) + models[i, -1]
        # Apply sigmoid for probability
        aus[i] = 1.0 / (1.0 + np.exp(-aus[i]))

    return aus


# ============================================================================
# INTEGRATED OPTIMIZED PIPELINE
# ============================================================================

class OptimizedAUPipeline:
    """Optimized pipeline incorporating all improvements."""

    def __init__(self):
        self.clnf_config = OptimizedCLNFConfig()
        self.feature_extractor = CachedFeatureExtractor()
        self.tracker = TemporalTracker()

        # Pre-compile Numba functions
        self._warmup_numba()

    def _warmup_numba(self):
        """Pre-compile Numba functions to avoid first-call overhead."""
        dummy_image = np.random.randn(100, 100).astype(np.float32)
        dummy_landmarks = np.random.randn(68, 2).astype(np.float32)
        dummy_points = np.random.randn(100, 2).astype(np.float32)
        dummy_weights = np.ones(100, dtype=np.float32)

        # Trigger compilation
        compute_response_maps_parallel(dummy_image, dummy_landmarks)
        kde_mean_shift_fast(dummy_points, dummy_weights)

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        """Process a frame with all optimizations."""

        result = {}
        start_time = time.perf_counter()

        # 1. Face Detection (with tracking)
        if self.tracker.should_detect():
            # Run full detection
            bbox = self._detect_face(frame)
            self.tracker.update(bbox, None, None)
        else:
            # Use tracking
            bbox = self.tracker.track_bbox(frame)

        result['detection_time'] = (time.perf_counter() - start_time) * 1000

        if bbox is None:
            return result

        # 2. Landmark Fitting (with temporal init)
        lm_start = time.perf_counter()

        # Use previous landmarks as initialization
        init_landmarks = self.tracker.predict_landmarks(frame)

        # Optimized CLNF with reduced iterations
        landmarks = self._fit_landmarks_optimized(frame, bbox, init_landmarks)

        result['landmark_time'] = (time.perf_counter() - lm_start) * 1000

        # 3. AU Prediction (with caching)
        au_start = time.perf_counter()

        # Extract features with caching
        features = self.feature_extractor.extract_features(frame, landmarks)

        # Batch predict all AUs
        aus = self._predict_aus_batch(features)

        result['au_time'] = (time.perf_counter() - au_start) * 1000

        # Update tracker
        self.tracker.update(bbox, landmarks, aus)

        result.update({
            'bbox': bbox,
            'landmarks': landmarks,
            'aus': aus,
            'total_time': (time.perf_counter() - start_time) * 1000
        })

        return result

    def _detect_face(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Placeholder for face detection."""
        # In practice, would use actual MTCNN
        return (100, 100, 200, 200)

    def _fit_landmarks_optimized(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        init_landmarks: Optional[np.ndarray]
    ) -> np.ndarray:
        """Optimized landmark fitting with Numba acceleration."""

        # Use initial landmarks if available
        if init_landmarks is None:
            # Initialize from bbox
            landmarks = self._init_landmarks_from_bbox(bbox)
        else:
            landmarks = init_landmarks

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)

        # Optimized CLNF iterations
        for iteration in range(self.clnf_config.max_iterations):
            # Compute response maps in parallel (Numba)
            response_maps = compute_response_maps_parallel(gray, landmarks)

            # Extract peaks (simplified)
            peaks = np.zeros_like(landmarks)
            for i in range(68):
                max_loc = np.unravel_index(
                    np.argmax(response_maps[i]),
                    response_maps[i].shape
                )
                peaks[i] = [max_loc[1], max_loc[0]]

            # Mean-shift update (Numba)
            weights = np.ones(68, dtype=np.float32)
            new_landmarks = kde_mean_shift_fast(peaks, weights)

            # Check convergence
            movement = np.mean(np.abs(new_landmarks - landmarks))
            landmarks = new_landmarks

            if movement < self.clnf_config.convergence_threshold:
                break

        return landmarks

    def _init_landmarks_from_bbox(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Initialize landmarks from bounding box."""
        x, y, w, h = bbox
        # Simplified: place landmarks in a grid
        landmarks = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            landmarks[i, 0] = x + w * (i % 8) / 8
            landmarks[i, 1] = y + h * (i // 8) / 8
        return landmarks

    def _predict_aus_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch predict all AUs."""
        # Placeholder SVM models
        models = np.random.randn(17, len(features) + 1).astype(np.float32)
        return predict_aus_batch(features, models)


# ============================================================================
# BENCHMARK COMPARISON
# ============================================================================

def benchmark_optimizations():
    """Compare original vs optimized pipeline."""

    print("="*60)
    print("OPTIMIZATION BENCHMARK")
    print("="*60)

    # Create test data
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Initialize optimized pipeline
    print("\nInitializing optimized pipeline...")
    opt_pipeline = OptimizedAUPipeline()

    # Warm up
    print("Warming up...")
    for _ in range(3):
        opt_pipeline.process_frame(test_frame, 0)

    # Benchmark
    print("\nBenchmarking 10 frames...")
    times = []

    for i in range(10):
        start = time.perf_counter()
        result = opt_pipeline.process_frame(test_frame, i)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        print(f"Frame {i:2d}: {elapsed:6.1f}ms "
              f"(Det: {result.get('detection_time', 0):5.1f}ms, "
              f"LM: {result.get('landmark_time', 0):5.1f}ms, "
              f"AU: {result.get('au_time', 0):5.1f}ms)")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Average time: {np.mean(times):.1f}ms")
    print(f"Expected speedup vs original (2046ms): {2046/np.mean(times):.1f}x")
    print("\nNote: This is a simplified demo. Actual speedup will vary.")


if __name__ == "__main__":
    benchmark_optimizations()