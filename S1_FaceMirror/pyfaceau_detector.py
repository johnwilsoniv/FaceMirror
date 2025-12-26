#!/usr/bin/env python3

def safe_print(*args, **kwargs):
    """Print wrapper that handles BrokenPipeError in GUI subprocess contexts."""
    import builtins
    try:
        builtins.print(*args, **kwargs)
    except (BrokenPipeError, IOError):
        pass  # Stdout disconnected

"""
68-point facial landmark detector using GPU-accelerated PyPI packages.

Uses:
- pymtcnn (PyPI) for face detection
- pyclnf (PyPI) for GPU-accelerated CLNF landmark detection
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import threading

# PyPI packages
from pymtcnn import MTCNN
from pyclnf import CLNF


class PyFaceAU68LandmarkDetector:
    """
    Face detection and 68-point landmark tracking using GPU-accelerated CLNF.

    This class provides:
    - PyMTCNN face detection (CoreML/CUDA/CPU auto-selection)
    - CLNF 68-point landmarks (GPU-accelerated, ~15 fps)
    - Temporal smoothing (5-frame history)
    - Head pose estimation
    """

    def __init__(self, debug_mode=False, device='auto', model_dir=None,
                 skip_redetection=False, skip_face_detection=False,
                 use_clnf_refinement=True):
        """
        Initialize GPU-accelerated detector.

        Args:
            debug_mode: Enable debug output
            device: GPU device ('auto', 'mps', 'cuda', 'cpu')
            model_dir: Directory containing model weights (defaults to ./weights)
            skip_redetection: Skip face detection after first frame (tracking mode)
            skip_face_detection: Skip face detection entirely, use default bbox
            use_clnf_refinement: Ignored - always uses full CLNF
        """
        self.skip_face_detection = skip_face_detection
        self.skip_redetection = skip_redetection
        self.debug_mode = debug_mode

        if debug_mode:
            safe_print("\n" + "="*60)
            safe_print("GPU-ACCELERATED LANDMARK DETECTOR")
            safe_print("="*60)

        # Initialize PyMTCNN for face detection
        if skip_face_detection:
            self.face_detector = None
            if debug_mode:
                safe_print("  PyMTCNN: Skipped (using default bbox)")
        else:
            self.face_detector = MTCNN(backend='auto')  # Auto-selects: coreml/cuda/cpu
            backend_info = self.face_detector.get_backend_info()
            if debug_mode:
                safe_print(f"  PyMTCNN: {backend_info.get('backend', 'unknown')}")

        # Initialize CLNF for landmark detection (GPU-accelerated)
        self.clnf = CLNF(
            convergence_profile='video',
            use_gpu=True,
            use_validator=False,  # Faster without validation
            use_eye_refinement=True
        )
        if debug_mode:
            gpu_status = "GPU" if self.clnf.use_gpu else "CPU"
            safe_print(f"  CLNF: Loaded ({gpu_status} accelerated)")

        # Bbox calibration coefficients (MTCNN → CLNF)
        # These map PyMTCNN output to CLNF-expected input
        self.bbox_coeffs = (-0.0075, 0.2459, 1.0323, 0.7751)

        if debug_mode:
            safe_print("="*60 + "\n")

        # Tracking state
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0
        self.cached_bbox = None
        self._frame_idx = 0

        # Thread lock for CoreML/CLNF access (these are NOT thread-safe)
        self._detector_lock = threading.Lock()

        # Temporal smoothing history (5-frame, matches open2GR)
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.yaw_history = []
        self.frame_quality_history = []
        self.history_size = 5

        # Warmup models
        self._warmup_models()

    def _warmup_models(self):
        """Warm up models with dummy inference"""
        if self.debug_mode:
            safe_print("Warming up models...")

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            if self.face_detector:
                _ = self.face_detector.detect(dummy_frame)

            # Warm up CLNF
            dummy_bbox = (100, 100, 200, 200)
            _ = self.clnf.fit(dummy_frame, dummy_bbox)

            # CRITICAL: Reset CLNF temporal state after warmup!
            # Warmup sets _prev_frame_params which video mode would reuse,
            # causing landmarks to be placed at the warmup bbox location
            # instead of the actual detected face bbox.
            self.clnf.reset_temporal_state()

            if self.debug_mode:
                safe_print("  Models warmed up\n")
        except Exception as e:
            if self.debug_mode:
                safe_print(f"  Warmup warning (non-critical): {e}\n")

    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        self.last_face = None
        self.last_landmarks = None
        self.cached_bbox = None
        self.frame_count = 0
        self._frame_idx = 0
        self.landmarks_history.clear()
        self.glabella_history.clear()
        self.chin_history.clear()
        self.yaw_history.clear()
        self.frame_quality_history.clear()

    def cleanup_memory(self):
        """Cleanup memory after processing"""
        self.reset_tracking_history()
        import gc
        gc.collect()

    def get_face_mesh(self, frame, detection_interval=30):
        """
        Get 68-point facial landmarks using GPU-accelerated CLNF.

        Args:
            frame: BGR image (numpy array)
            detection_interval: Re-detect face every N frames (0 = never)

        Returns:
            smoothed_points: (68, 2) array of smoothed landmark coordinates
            info: Validation info dict or None
        """
        # Thread lock required: CoreML/Metal models are NOT thread-safe
        # Without this lock, concurrent access from ThreadPoolExecutor causes segfaults
        with self._detector_lock:
            self.frame_count += 1
            self._frame_idx += 1

            # Handle skip_face_detection mode
            if self.skip_face_detection:
                if self.cached_bbox is None:
                    h, w = frame.shape[:2]
                    margin = 0.1
                    x1 = int(w * margin)
                    y1 = int(h * margin)
                    x2 = int(w * (1 - margin))
                    y2 = int(h * (1 - margin))
                    # Convert to CLNF format (x, y, w, h)
                    self.cached_bbox = (x1, y1, x2 - x1, y2 - y1)

            # Face detection (first frame or periodic refresh)
            should_detect = (
                self.cached_bbox is None and
                not self.skip_face_detection and
                (self._frame_idx == 1 or not self.skip_redetection)
            )

            # Also re-detect periodically if detection_interval > 0
            if not should_detect and detection_interval > 0:
                if self._frame_idx % detection_interval == 0 and not self.skip_face_detection:
                    should_detect = True

            if should_detect and self.face_detector is not None:
                try:
                    boxes, _ = self.face_detector.detect(frame)

                    if boxes is not None and len(boxes) > 0:
                        # PyMTCNN returns [x, y, w, h, conf]
                        x, y, w, h = boxes[0][:4]

                        # Apply calibration coefficients
                        cx, cy, cw, ch = self.bbox_coeffs
                        self.cached_bbox = (
                            x + w * cx,
                            y + h * cy,
                            w * cw,
                            h * ch
                        )
                    else:
                        if self.cached_bbox is None:
                            return None, {'valid': False, 'reason': 'No face detected'}

                except Exception as e:
                    if self.debug_mode:
                        safe_print(f"Face detection error: {e}")
                    if self.cached_bbox is None:
                        return None, {'valid': False, 'reason': str(e)}

            # CLNF landmark detection
            if self.cached_bbox is None:
                return None, {'valid': False, 'reason': 'No bbox available'}

            try:
                landmarks, info = self.clnf.fit(frame, self.cached_bbox, return_params=True)

                if landmarks is None:
                    # Try reusing previous landmarks
                    if self.last_landmarks is not None:
                        return self.last_landmarks.copy(), {'valid': True, 'reused': True}
                    return None, {'valid': False, 'reason': 'CLNF failed'}

                # Ensure float32 for calculations
                landmarks = landmarks.astype(np.float32)

                # Temporal smoothing (5-frame weighted average)
                self.landmarks_history.append(landmarks.copy())
                if len(self.landmarks_history) > self.history_size:
                    self.landmarks_history.pop(0)

                # Weighted average (more weight to recent frames)
                weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
                weights = weights / np.sum(weights)

                smoothed_points = np.zeros_like(landmarks, dtype=np.float32)
                for pts, w in zip(self.landmarks_history, weights):
                    smoothed_points += pts * w

                # Update tracking
                self.last_landmarks = smoothed_points.copy()

                # Update yaw history
                yaw = self.calculate_head_pose(smoothed_points)
                self.yaw_history.append(yaw)
                if len(self.yaw_history) > self.history_size:
                    self.yaw_history.pop(0)

                # Calculate frame quality
                quality = self.calculate_frame_quality(smoothed_points)
                self.frame_quality_history.append(quality)
                if len(self.frame_quality_history) > self.history_size:
                    self.frame_quality_history.pop(0)

                return smoothed_points, {
                    'valid': True,
                    'params': info.get('params') if info else None,
                    'confidence': info.get('confidence', 1.0) if info else 1.0
                }

            except Exception as e:
                if self.debug_mode:
                    safe_print(f"Landmark detection error: {e}")

                # Try reusing previous landmarks
                if self.last_landmarks is not None:
                    return self.last_landmarks.copy(), {'valid': True, 'reused': True}

                return None, {'valid': False, 'reason': str(e)}

    def get_facial_midline(self, landmarks):
        """
        Calculate anatomical midline points using 68-point landmarks.

        68-point landmark indices (dlib/PFLD standard):
        - Left medial eyebrow: index 21
        - Right medial eyebrow: index 22
        - Chin center: index 8

        Args:
            landmarks: (68, 2) array of landmarks

        Returns:
            glabella: Midpoint between medial eyebrows
            chin: Chin center point
        """
        if landmarks is None or len(landmarks) != 68:
            return None, None

        landmarks = landmarks.astype(np.float32)

        # Get medial eyebrow points
        left_medial_brow = landmarks[21]
        right_medial_brow = landmarks[22]

        # Calculate glabella and chin
        glabella = (left_medial_brow + right_medial_brow) / 2
        chin = landmarks[8]

        # Add to history for temporal smoothing
        self.glabella_history.append(glabella)
        self.chin_history.append(chin)

        if len(self.glabella_history) > self.history_size:
            self.glabella_history.pop(0)
        if len(self.chin_history) > self.history_size:
            self.chin_history.pop(0)

        # Calculate smooth midline points
        smooth_glabella = np.mean(self.glabella_history, axis=0)
        smooth_chin = np.mean(self.chin_history, axis=0)

        return smooth_glabella, smooth_chin

    def calculate_head_pose(self, landmarks):
        """
        Calculate head yaw using symmetric landmark pairs.

        Args:
            landmarks: (68, 2) array of landmarks

        Returns:
            yaw: Head rotation angle in degrees
        """
        if landmarks is None or len(landmarks) != 68:
            return None

        landmarks = landmarks.astype(np.float32)

        # Get midline for reference
        glabella, chin = self.get_facial_midline(landmarks)
        if glabella is None or chin is None:
            return None

        # Calculate face center line
        center_landmarks = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]
        center_points = landmarks[center_landmarks]
        center_x = np.mean(center_points[:, 0])

        # Symmetric landmark pairs for yaw estimation
        landmark_pairs = [
            (36, 45),  # Eyes outer corners
            (39, 42),  # Eyes inner corners
            (17, 26),  # Eyebrows outer
            (48, 54),  # Mouth corners
            (1, 15),   # Jaw
            (4, 12)    # Cheeks
        ]

        yaw_estimates = []
        weights = []

        for left_idx, right_idx in landmark_pairs:
            left_point = landmarks[left_idx]
            right_point = landmarks[right_idx]

            left_dist = center_x - left_point[0]
            right_dist = right_point[0] - center_x

            if left_dist > 0 and right_dist > 0:
                avg_dist = (left_dist + right_dist) / 2
                ratio_diff = (right_dist - left_dist) / avg_dist
                yaw_estimate = ratio_diff * 45.0

                weight = 2.0 if (left_idx, right_idx) in [(36, 45), (39, 42)] else 1.0
                yaw_estimates.append(yaw_estimate)
                weights.append(weight)

        if yaw_estimates:
            weights = np.array(weights) / np.sum(weights)
            return np.average(yaw_estimates, weights=weights)

        # Fallback to nose offset
        eyes_center = (landmarks[39] + landmarks[42]) / 2
        nose_tip = landmarks[30]
        nose_offset = nose_tip[0] - eyes_center[0]
        face_width = np.linalg.norm(landmarks[36] - landmarks[45])
        if face_width > 0:
            normalized_offset = nose_offset / (face_width / 2)
            return normalized_offset * 45.0

        return None

    def calculate_face_stability(self):
        """
        Calculate face stability based on yaw history.

        Returns:
            stability: Float between 0.0 and 1.0
            is_stable: Boolean indicating stability
        """
        if len(self.yaw_history) < 3:
            return 0.0, False

        valid_yaw = [y for y in self.yaw_history if y is not None]
        if len(valid_yaw) < 3:
            return 0.0, False

        yaw_std = np.std(valid_yaw)
        max_std = 5.0
        stability = max(0.0, 1.0 - (yaw_std / max_std))

        is_stable = stability >= 0.7
        return stability, is_stable

    def calculate_frame_quality(self, landmarks):
        """
        Calculate frame quality score based on head yaw.

        Returns:
            quality: Float between 0.0 and 1.0
        """
        if landmarks is None:
            return 0.0

        yaw = self.calculate_head_pose(landmarks)
        if yaw is None:
            return 0.0

        ideal_range = 3.0
        yaw_quality = max(0.0, 1.0 - (abs(yaw) - ideal_range) / 7.0) if abs(yaw) > ideal_range else 1.0

        stability, _ = self.calculate_face_stability() if len(self.yaw_history) >= 3 else (0.5, False)

        overall_quality = (yaw_quality * 0.9) + (stability * 0.1)
        return overall_quality

    def print_performance_summary(self):
        """Print performance summary (compatibility method)."""
        pass

    def cleanup_memory(self):
        """Cleanup GPU memory (compatibility method)."""
        self.reset_tracking_history()
        import gc
        gc.collect()
