#!/usr/bin/env python3
"""
Optimized STAR landmark detector using ONNX Runtime with CoreML acceleration.

This module provides a drop-in replacement for the PyTorch-based STAR landmark detector,
optimized for Apple Silicon using the Neural Engine via CoreML execution provider.

Expected performance: 10-20x speedup (1800ms → 90-180ms per frame)
"""

import numpy as np
import cv2
import math
from typing import Optional, Tuple
import onnxruntime as ort

# Import performance profiler
from performance_profiler import get_profiler


class ONNXStarDetector:
    """
    ONNX-accelerated STAR landmark detector for Apple Silicon

    This class provides the same interface as the OpenFace 3.0 LandmarkDetector,
    but uses ONNX Runtime with CoreML execution provider for massive speedup.
    """

    def __init__(self, onnx_model_path: str, use_coreml: bool = True):
        """
        Initialize ONNX STAR detector

        Args:
            onnx_model_path: Path to converted ONNX model
            use_coreml: Whether to attempt CoreML execution provider (default: True)
        """
        self.input_size = 256
        self.target_face_scale = 1.0

        # Configure execution providers
        # NOTE: CoreML may fail on complex models like STAR due to unsupported operations
        # We try CoreML first, but gracefully fall back to optimized CPU execution
        if use_coreml:
            providers = [
                ('CoreMLExecutionProvider', {
                    'MLComputeUnits': 'ALL',  # Use Neural Engine + GPU + CPU
                    'ModelFormat': 'MLProgram',  # Use latest CoreML format
                }),
                'CPUExecutionProvider'  # Fallback
            ]
        else:
            providers = ['CPUExecutionProvider']

        # Load ONNX model
        print(f"Loading ONNX STAR model from: {onnx_model_path}")

        # Configure session options to prevent thread conflicts
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # Single thread per operator
        sess_options.inter_op_num_threads = 1  # Sequential operator execution
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Suppress CoreML compilation warnings (they're expected for complex models)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=providers)

        # Check which providers are actually active
        active_providers = self.session.get_providers()

        if 'CoreMLExecutionProvider' in active_providers:
            print("✓ Using CoreML Neural Engine acceleration")
            print("  Expected: 10-20x speedup")
            self.backend = 'coreml'
        else:
            print("✓ Using ONNX Runtime with optimized CPU execution")
            print("  Expected: 3-5x speedup over PyTorch")
            print("  (CoreML not available for this model - some operations unsupported)")
            self.backend = 'onnx_cpu'

    def _compose_rotate_and_scale(self, angle: float, scale: float, shift_xy: Tuple[float, float],
                                   from_center: Tuple[float, float], to_center: Tuple[float, float]) -> np.ndarray:
        """
        Compose rotation and scaling transformation matrix

        Args:
            angle: Rotation angle in radians
            scale: Scale factor
            shift_xy: Translation (x, y)
            from_center: Source center point (x, y)
            to_center: Target center point (x, y)

        Returns:
            3x3 transformation matrix
        """
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)

        return rot_scale_m

    def _get_crop_matrix(self, scale: float, center_w: float, center_h: float) -> np.ndarray:
        """
        Get crop transformation matrix for face alignment

        Args:
            scale: Face scale factor
            center_w: Face center X coordinate
            center_h: Face center Y coordinate

        Returns:
            3x3 transformation matrix
        """
        # align_corners=True (as per STAR demo.py)
        to_w = self.input_size - 1
        to_h = self.input_size - 1

        rot_mu = 0
        scale_mu = self.input_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)

        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0]
        )

        return matrix

    def _preprocess(self, image: np.ndarray, scale: float, center_w: float, center_h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for STAR model inference

        Args:
            image: Input BGR image
            scale: Face scale factor
            center_w: Face center X coordinate
            center_h: Face center Y coordinate

        Returns:
            Tuple of (input_tensor, transformation_matrix)
        """
        # Get transformation matrix
        matrix = self._get_crop_matrix(scale, center_w, center_h)

        # Apply perspective transformation
        input_crop = cv2.warpPerspective(
            image, matrix,
            dsize=(self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderValue=0
        )

        # Convert to tensor format (1, 3, 256, 256)
        input_tensor = input_crop[np.newaxis, :]  # Add batch dimension
        input_tensor = input_tensor.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        # Normalize to [-1, 1] range (as per STAR preprocessing)
        input_tensor = input_tensor.astype(np.float32)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0

        return input_tensor, matrix

    def _denorm_points(self, points: np.ndarray) -> np.ndarray:
        """
        Denormalize landmark points from [-1, 1] to [0, input_size-1]

        Args:
            points: Normalized points in range [-1, 1]

        Returns:
            Denormalized points in pixel coordinates
        """
        # align_corners=True: [-1, +1] -> [0, SIZE-1]
        return (points + 1) / 2 * (self.input_size - 1)

    def _postprocess(self, landmarks_normalized: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Postprocess normalized landmarks to original image coordinates

        Args:
            landmarks_normalized: Landmarks in normalized coordinates [-1, 1]
            matrix: Forward transformation matrix

        Returns:
            Landmarks in original image coordinates
        """
        # Denormalize from [-1, 1] to [0, 255]
        landmarks = self._denorm_points(landmarks_normalized)

        # Apply inverse transformation
        inv_matrix = np.linalg.inv(matrix)

        dstPoints = np.zeros(landmarks.shape, dtype=np.float32)
        for i in range(landmarks.shape[0]):
            dstPoints[i][0] = inv_matrix[0][0] * landmarks[i][0] + inv_matrix[0][1] * landmarks[i][1] + inv_matrix[0][2]
            dstPoints[i][1] = inv_matrix[1][0] * landmarks[i][0] + inv_matrix[1][1] * landmarks[i][1] + inv_matrix[1][2]

        return dstPoints

    def detect_landmarks(self, image: np.ndarray, dets: np.ndarray, confidence_threshold: float = 0.5):
        """
        Detect 98-point facial landmarks for detected faces

        This method matches the interface of OpenFace 3.0 LandmarkDetector for drop-in compatibility.

        Args:
            image: Input BGR image (numpy array)
            dets: Face detections array with format [x1, y1, x2, y2, confidence, ...]
            confidence_threshold: Minimum confidence for processing (default: 0.5)

        Returns:
            List of 98-point landmark arrays (one per face)
        """
        profiler = get_profiler()
        results = []

        for det in dets:
            x1, y1, x2, y2 = det[:4].astype(int)
            conf = det[4]

            # Skip low-confidence detections
            if conf < confidence_threshold:
                continue

            # Calculate face center and scale (same as OpenFace 3.0)
            center_w = (x2 + x1) / 2
            center_h = (y2 + y1) / 2
            scale = min(x2 - x1, y2 - y1) / 200 * 1.05

            # Preprocess image
            with profiler.time_block("preprocessing", f"STAR_preprocess"):
                input_tensor, matrix = self._preprocess(image, float(scale), float(center_w), float(center_h))

            # Run ONNX inference on Neural Engine
            with profiler.time_block("model_inference", f"STAR_{self.backend}"):
                outputs = self.session.run(None, {'input_image': input_tensor})

            # Extract landmarks from output
            # STAR model returns (output, heatmap, landmarks)
            # We want the last output (landmarks)
            with profiler.time_block("postprocessing", f"STAR_postprocess"):
                landmarks_normalized = outputs[-1][0]  # Shape: (98, 2)

                # Postprocess to original image coordinates
                landmarks = self._postprocess(landmarks_normalized, matrix)

                results.append(landmarks)

        return results


class OptimizedLandmarkDetector:
    """
    Wrapper class that automatically selects ONNX or PyTorch implementation

    This class provides seamless fallback from ONNX (fast) to PyTorch (slow)
    based on model availability.
    """

    def __init__(self, model_path: str, onnx_model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize landmark detector with automatic ONNX/PyTorch selection

        Args:
            model_path: Path to PyTorch model (.pkl)
            onnx_model_path: Path to ONNX model (.onnx), defaults to same directory as model_path
            device: Device for PyTorch fallback ('cpu' or 'cuda')
        """
        from pathlib import Path

        # Determine ONNX model path
        if onnx_model_path is None:
            model_dir = Path(model_path).parent
            onnx_model_path = model_dir / 'star_landmark_98_coreml.onnx'

        # Try ONNX first (fast path)
        if Path(onnx_model_path).exists():
            try:
                print(f"Using ONNX-accelerated STAR detector (CoreML/Neural Engine)")
                self.detector = ONNXStarDetector(str(onnx_model_path), use_coreml=True)
                self.backend = 'onnx'
                return
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                print(f"Falling back to PyTorch implementation...")

        # Fallback to PyTorch (slow path)
        print(f"Using PyTorch STAR detector (CPU - slower)")
        print(f"To enable acceleration, run: python convert_star_to_onnx.py")

        from openface.landmark_detection import LandmarkDetector
        self.detector = LandmarkDetector(model_path=model_path, device=device)
        self.backend = 'pytorch'

    def detect_landmarks(self, image: np.ndarray, dets: np.ndarray, confidence_threshold: float = 0.5):
        """
        Detect landmarks using the selected backend

        Args:
            image: Input BGR image
            dets: Face detections
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of landmark arrays
        """
        return self.detector.detect_landmarks(image, dets, confidence_threshold)


# Convenience function for benchmarking
def benchmark_detector(image_path: str, onnx_model_path: str, num_iterations: int = 10):
    """
    Benchmark ONNX detector performance

    Args:
        image_path: Path to test image with face
        onnx_model_path: Path to ONNX model
        num_iterations: Number of iterations for timing
    """
    import time

    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create fake detection (center of image)
    h, w = image.shape[:2]
    fake_det = np.array([[w*0.25, h*0.25, w*0.75, h*0.75, 0.99]])

    # Initialize detector
    detector = ONNXStarDetector(onnx_model_path)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = detector.detect_landmarks(image, fake_det)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    for i in range(num_iterations):
        start = time.time()
        landmarks = detector.detect_landmarks(image, fake_det)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.1f} ms")

    # Statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)

    print(f"\nBenchmark Results:")
    print(f"  Average: {avg_time:.1f} ms")
    print(f"  Min:     {min_time:.1f} ms")
    print(f"  Max:     {max_time:.1f} ms")
    print(f"  Std:     {std_time:.1f} ms")
    print(f"  FPS:     {1000/avg_time:.1f}")

    return avg_time


if __name__ == '__main__':
    print("ONNX STAR Detector Module")
    print("=" * 60)
    print("This module provides CoreML-accelerated STAR landmark detection.")
    print("")
    print("Usage:")
    print("  from onnx_star_detector import OptimizedLandmarkDetector")
    print("  detector = OptimizedLandmarkDetector('weights/Landmark_98.pkl')")
    print("  landmarks = detector.detect_landmarks(image, detections)")
    print("=" * 60)
