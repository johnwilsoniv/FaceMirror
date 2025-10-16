import numpy as np
import cv2
import os
import torch
from typing import Optional, Tuple, List

# Set environment variables to prevent read-only filesystem errors
# OpenFace/PyTorch may try to write to /work or other restricted locations
os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
os.environ.setdefault('TMPDIR', os.path.expanduser('~/tmp'))
os.environ.setdefault('TEMP', os.path.expanduser('~/tmp'))
os.environ.setdefault('TMP', os.path.expanduser('~/tmp'))

# Create temp directory if it doesn't exist
temp_dir = os.path.expanduser('~/tmp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm


class OpenFace3LandmarkDetector:
    """
    Face detection and landmark tracking using OpenFace 3.0

    This class provides face detection and landmark tracking using OpenFace 3.0's
    efficient RetinaFace and STAR models.
    """

    def __init__(self, debug_mode=False, device='cpu', model_dir=None):
        """
        Initialize OpenFace 3.0 face detector and landmark predictor

        Args:
            debug_mode: Enable debug output
            device: 'cpu' or 'cuda' for GPU acceleration
            model_dir: Directory containing OpenFace 3.0 model weights (defaults to script directory/weights)
        """
        import os

        # Determine model directory - use config_paths for cross-platform support
        if model_dir is None:
            import config_paths
            model_dir = str(config_paths.get_weights_dir())

        # OpenFace 3.0 expects weights at "./weights/mobilenetV1X0.25_pretrain.tar"
        # We need to patch the hardcoded path in OpenFace's internal code
        weights_file_path = os.path.join(model_dir, 'mobilenetV1X0.25_pretrain.tar')

        # Save original working directory to restore later
        original_cwd = os.getcwd()
        self.original_cwd = original_cwd

        # Patch OpenFace's hardcoded weights path
        # The FaceDetector initialization tries to load './weights/mobilenetV1X0.25_pretrain.tar'
        # We'll monkey-patch the path to use the absolute path instead
        import openface.face_detection

        # Store the absolute path to weights file for the monkey patch
        self._weights_abs_path = weights_file_path

        # Check if weights file exists
        if not os.path.exists(weights_file_path):
            raise FileNotFoundError(f"Weights file not found at {weights_file_path}")

        # Monkey patch: Replace the hardcoded relative path with absolute path
        # This is done by changing directory to the parent of weights, so './weights/...' resolves correctly
        weights_parent = os.path.dirname(model_dir)
        os.chdir(weights_parent)
        if debug_mode:
            print(f"Changed working directory to: {weights_parent}")
            print(f"Weights accessible at: ./weights/mobilenetV1X0.25_pretrain.tar")

        # Initialize OpenFace 3.0 models with correct parameters
        self.face_detector = FaceDetector(
            model_path=f'{model_dir}/Alignment_RetinaFace.pth',
            device=device,
            confidence_threshold=0.9
        )

        # Patch STAR config to use writable cache directory instead of /work
        import openface.STAR.conf.alignment as alignment_conf
        original_init = alignment_conf.Alignment.__init__

        def patched_init(self_inner, args):
            original_init(self_inner, args)
            # Replace hardcoded /work path with a writable location
            home_dir = os.path.expanduser('~')
            self_inner.ckpt_dir = os.path.join(home_dir, '.cache', 'openface', 'STAR')
            self_inner.work_dir = osp.join(self_inner.ckpt_dir, self_inner.data_definition, self_inner.folder)
            self_inner.model_dir = osp.join(self_inner.work_dir, 'model')
            self_inner.log_dir = osp.join(self_inner.work_dir, 'log')
            # Create directories if they don't exist
            os.makedirs(self_inner.ckpt_dir, exist_ok=True)

        import os.path as osp
        alignment_conf.Alignment.__init__ = patched_init

        self.landmark_detector = LandmarkDetector(
            model_path=f'{model_dir}/Landmark_98.pkl',
            device=device
        )

        # Restore original __init__
        alignment_conf.Alignment.__init__ = original_init

        # DO NOT restore working directory yet - keep it at weights_parent
        # OpenFace may need to access ./weights/ during runtime, not just initialization
        # We'll store the weights_parent path for later use
        self._weights_parent = weights_parent

        if debug_mode:
            print(f"Keeping working directory at: {os.getcwd()}")
            print(f"Original working directory saved as: {original_cwd}")

        # Suppress verbose "Processing face..." output from landmark detector
        import openface.landmark_detection as lm_module
        original_detect = lm_module.LandmarkDetector.detect_landmarks

        def silent_detect(self_inner, image, dets, confidence_threshold=0.5):
            # Temporarily redirect stdout to suppress prints
            import io
            import contextlib
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                return original_detect(self_inner, image, dets, confidence_threshold)

        lm_module.LandmarkDetector.detect_landmarks = silent_detect

        # Tracking history
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0

        # Initialize smoothing history
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.history_size = 3

        # Face ROI tracking
        self.face_roi = None
        self.roi_expansion_factor = 1.5  # Expand ROI by 50% for safety

        # Head pose history for stability calculation
        self.yaw_history = []
        self.pose_history_size = 5

        # Frame quality history
        self.frame_quality_history = []

        # Adaptive detection interval
        self.adaptive_interval = True
        self.min_detection_interval = 1
        self.max_detection_interval = 8
        self.current_detection_interval = 2

        # Performance tracking
        self.perf_detection_time = []
        self.perf_landmark_time = []
        self.perf_total_time = []

        # Debug mode
        self.debug_mode = debug_mode

        # Device setting (store for use in detection methods)
        self.device = torch.device(device)

    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.yaw_history = []
        self.frame_quality_history = []
        self.face_roi = None
        self.current_detection_interval = 2
        self.perf_detection_time = []
        self.perf_landmark_time = []
        self.perf_total_time = []

    def cleanup_memory(self):
        """
        Aggressive memory cleanup after processing a video.
        Clears all accumulated data and performance tracking.
        """
        # Clear performance tracking arrays (these grow unbounded)
        self.perf_detection_time.clear()
        self.perf_landmark_time.clear()
        self.perf_total_time.clear()

        # Clear all history buffers
        self.landmarks_history.clear()
        self.glabella_history.clear()
        self.chin_history.clear()
        self.yaw_history.clear()
        self.frame_quality_history.clear()

        # Clear PyTorch cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reset all tracking state
        self.reset_tracking_history()

    def _convert_68_landmarks(self, landmarks_98):
        """
        Convert OpenFace 3.0's 98 WFLW landmarks - but actually we DON'T convert!

        The WFLW 98-point format is different from dlib's 68-point format.
        Instead of trying to convert, we'll just return all 98 landmarks and
        update the methods that use them (get_facial_midline, calculate_head_pose)
        to use the correct WFLW indices directly.

        Args:
            landmarks_98: (98, 2) array of OpenFace 3.0 WFLW landmarks

        Returns:
            landmarks_98: Same 98 landmarks (no conversion)
        """
        if landmarks_98 is None or len(landmarks_98) < 98:
            return None

        # Return all 98 landmarks - we'll use WFLW indices directly
        return landmarks_98

    def update_face_roi(self, face_bbox, frame_shape):
        """
        Update face ROI with temporal smoothing and expansion

        Args:
            face_bbox: Face bounding box [x1, y1, x2, y2, confidence]
            frame_shape: (height, width) of frame
        """
        if face_bbox is None:
            return

        height, width = frame_shape[:2]
        x1, y1, x2, y2 = face_bbox[:4]

        # Expand ROI for safety
        face_width = x2 - x1
        face_height = y2 - y1
        expand_w = int(face_width * (self.roi_expansion_factor - 1.0) / 2)
        expand_h = int(face_height * (self.roi_expansion_factor - 1.0) / 2)

        expanded_roi = [
            max(0, int(x1 - expand_w)),
            max(0, int(y1 - expand_h)),
            min(width, int(x2 + expand_w)),
            min(height, int(y2 + expand_h))
        ]

        if self.face_roi is None:
            self.face_roi = expanded_roi
        else:
            # Smooth ROI transition
            alpha = 0.6  # Smoothing factor
            self.face_roi = [int(alpha * prev + (1 - alpha) * curr)
                            for prev, curr in zip(self.face_roi, expanded_roi)]

    def get_detection_roi(self, frame_shape):
        """
        Get ROI for face detection (returns None for full-frame detection)

        Args:
            frame_shape: (height, width, channels) of frame

        Returns:
            roi: [x1, y1, x2, y2] or None for full-frame
        """
        if self.face_roi is None:
            return None

        # Ensure ROI is valid
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = self.face_roi

        # Sanity check
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return None

        return self.face_roi

    def update_detection_interval(self):
        """
        Dynamically adjust detection interval based on face stability

        Stable face = longer interval (detect less often)
        Unstable face = shorter interval (detect more often)
        """
        if not self.adaptive_interval:
            return

        # Need at least 3 frames of history
        if len(self.yaw_history) < 3:
            self.current_detection_interval = self.min_detection_interval
            return

        # Calculate stability
        stability, is_stable = self.calculate_face_stability()

        if is_stable:
            # Very stable - increase interval up to max
            self.current_detection_interval = min(
                self.current_detection_interval + 1,
                self.max_detection_interval
            )
        else:
            # Unstable - decrease interval down to min
            self.current_detection_interval = max(
                self.current_detection_interval - 1,
                self.min_detection_interval
            )

    def _detect_faces_from_array(self, img_array):
        """
        Detect faces in a numpy array (wrapper for OpenFace 3.0's file-based API)

        Args:
            img_array: BGR numpy array (H, W, 3)

        Returns:
            detections: Array of face detections with format [x1, y1, x2, y2, conf, landmarks...]
            img_raw: Original image
        """
        # Preprocess image (adapted from FaceDetector.preprocess_image)
        img = np.float32(img_array)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            loc, conf, landms = self.face_detector.model(img)

        # Decode predictions (adapted from FaceDetector.detect_faces)
        im_height, im_width, _ = img_array.shape
        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]]).to(self.device)

        priorbox = PriorBox(self.face_detector.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.face_detector.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.face_detector.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # Filter by confidence threshold
        inds = np.where(scores > self.face_detector.confidence_threshold)[0]
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

        # Apply NMS
        from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.face_detector.nms_threshold)
        dets = dets[keep]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        return dets, img_array

    def get_face_mesh(self, frame, detection_interval=None):
        """
        Get facial landmarks with temporal smoothing using OpenFace 3.0

        Args:
            frame: Input BGR image (numpy array)
            detection_interval: Run face detection every N frames (None = use adaptive interval)

        Returns:
            smoothed_points: (98, 2) array of smoothed landmark coordinates
            triangles: Delaunay triangulation (not used in current implementation)
        """
        import time

        frame_start = time.time()
        self.frame_count += 1

        # Use adaptive interval if not specified
        if detection_interval is None:
            detection_interval = self.current_detection_interval

        # Run face detection at intervals or if no face is tracked
        should_detect = (self.frame_count % detection_interval == 0) or (self.last_face is None)

        if should_detect:
            detect_start = time.time()
            try:
                # Get ROI for detection (None = full frame)
                roi = self.get_detection_roi(frame.shape)

                # Crop frame to ROI if available
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    frame_crop = frame[y1:y2, x1:x2].copy()
                else:
                    frame_crop = frame
                    x1, y1 = 0, 0

                # OpenFace 3.0 face detection from numpy array
                detections, img_raw = self._detect_faces_from_array(frame_crop)

                if detections is None or len(detections) == 0:
                    # Lost face - reset ROI and try full frame next time
                    self.face_roi = None
                    self.last_face = None
                    self.last_landmarks = None
                    return None, None

                # Get the largest face (most confident detection)
                # detections format: [x1, y1, x2, y2, confidence, landmarks...]
                face_det = detections[0][:5].copy()

                # Adjust coordinates back to full frame if we used ROI
                if roi is not None:
                    face_det[0] += x1  # x1
                    face_det[1] += y1  # y1
                    face_det[2] += x1  # x2
                    face_det[3] += y1  # y2

                # Store bbox + confidence (first 5 elements) for landmark detection
                self.last_face = face_det

                # Update ROI for next frame
                self.update_face_roi(face_det, frame.shape)

                # Track detection time
                detect_time = time.time() - detect_start
                self.perf_detection_time.append(detect_time)

            except Exception as e:
                if self.debug_mode:
                    print(f"Face detection error: {e}")
                self.face_roi = None
                self.last_face = None
                self.last_landmarks = None
                return None, None

        if self.last_face is not None:
            landmark_start = time.time()
            try:
                # Detect landmarks using OpenFace 3.0
                # The detector expects image and detections array with [x1, y1, x2, y2, confidence]
                landmarks_98 = self.landmark_detector.detect_landmarks(
                    frame,
                    np.array([self.last_face])
                )

                if landmarks_98 is None or len(landmarks_98) == 0:
                    return None, None

                # Get first face's landmarks
                landmarks_98 = landmarks_98[0]

                # Convert to 98-point format (no conversion needed now)
                points = self._convert_68_landmarks(landmarks_98)

                if points is None:
                    return None, None

                # Track landmark detection time
                landmark_time = time.time() - landmark_start
                self.perf_landmark_time.append(landmark_time)

                # Ensure float32 for calculations
                points = points.astype(np.float32)

                # Apply temporal smoothing
                self.landmarks_history.append(points)
                if len(self.landmarks_history) > self.history_size:
                    self.landmarks_history.pop(0)

                # Calculate weighted average (more weight to recent frames)
                weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
                weights = weights / np.sum(weights)

                smoothed_points = np.zeros_like(points, dtype=np.float32)
                for pts, w in zip(self.landmarks_history, weights):
                    smoothed_points += pts * w

                # Ensure integer coordinates for final output
                smoothed_points = np.round(smoothed_points).astype(np.int32)

                # Update yaw history
                yaw = self.calculate_head_pose(smoothed_points)
                self.yaw_history.append(yaw)
                if len(self.yaw_history) > self.pose_history_size:
                    self.yaw_history.pop(0)

                # Calculate frame quality
                quality = self.calculate_frame_quality(smoothed_points)
                self.frame_quality_history.append(quality)
                if len(self.frame_quality_history) > self.pose_history_size:
                    self.frame_quality_history.pop(0)

                # Update adaptive detection interval based on stability
                self.update_detection_interval()

                # Update last landmarks
                self.last_landmarks = smoothed_points

                # Track total frame processing time
                total_time = time.time() - frame_start
                self.perf_total_time.append(total_time)

                # Return landmarks and None for triangles (not needed)
                return smoothed_points, None

            except Exception as e:
                if self.debug_mode:
                    print(f"Landmark detection error: {e}")
                return None, None

        return None, None

    def get_facial_midline(self, landmarks):
        """
        Calculate the anatomical midline points

        WFLW 98-point landmark structure:
        - 0-32: Face contour (chin is around index 16)
        - 33-41: Right eyebrow
        - 42-50: Left eyebrow
        - 51-95: Eyes, mouth, etc.

        Based on diagnose_landmarks.py analysis, the correct medial brow points are:
        - Left medial brow: index 38
        - Right medial brow: index 50
        """
        if landmarks is None:
            return None, None

        # Convert to float32 for calculations
        landmarks = landmarks.astype(np.float32)

        # WFLW 98 landmark indices:
        # - Chin center: index 16
        # - Left eyebrow medial: index 38
        # - Right eyebrow medial: index 50
        left_medial_brow = landmarks[38]   # Left eyebrow inner corner (patient's left)
        right_medial_brow = landmarks[50]  # Right eyebrow inner corner (patient's right)

        # Calculate glabella (midpoint between the medial eyebrow points) and chin
        glabella = (left_medial_brow + right_medial_brow) / 2
        chin = landmarks[16]  # Chin center in WFLW

        # Add to history
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
        Calculate head yaw using multiple facial reference points

        Returns:
            yaw: Head rotation angle in degrees
        """
        if landmarks is None:
            return None

        # Convert to float32 for calculations
        landmarks = landmarks.astype(np.float32)

        # Use the existing midline calculation for efficiency
        glabella, chin = self.get_facial_midline(landmarks)

        if glabella is None or chin is None:
            return None

        # Calculate midline direction vector
        direction = chin - glabella
        if np.linalg.norm(direction) < 1e-6:
            return None

        direction = direction / np.linalg.norm(direction)

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

        # Fallback to simple nose offset
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
        Calculate face stability based on yaw history

        Returns:
            stability: Float between 0.0 and 1.0 (1.0 = perfectly stable)
            is_stable: Boolean indicating if face is considered stable
        """
        if len(self.yaw_history) < 3:
            return 0.0, False

        valid_yaw = [y for y in self.yaw_history if y is not None]

        if len(valid_yaw) < 3:
            return 0.0, False

        yaw_std = np.std(valid_yaw)
        max_std = 5.0
        stability = max(0.0, 1.0 - (yaw_std / max_std))

        stability_threshold = 0.7
        is_stable = stability >= stability_threshold

        return stability, is_stable

    def calculate_frame_quality(self, landmarks):
        """
        Calculate frame quality score based on head yaw

        Returns:
            quality: Float between 0.0 and 1.0 (1.0 = perfect quality)
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

    def get_performance_stats(self):
        """
        Get performance statistics for the current video processing session

        Returns:
            dict: Performance statistics including timing and adaptive interval info
        """
        stats = {
            'total_frames': self.frame_count,
            'detection_count': len(self.perf_detection_time),
            'avg_detection_time': np.mean(self.perf_detection_time) if self.perf_detection_time else 0,
            'avg_landmark_time': np.mean(self.perf_landmark_time) if self.perf_landmark_time else 0,
            'avg_total_time': np.mean(self.perf_total_time) if self.perf_total_time else 0,
            'current_interval': self.current_detection_interval,
            'fps_estimate': 1.0 / np.mean(self.perf_total_time) if self.perf_total_time else 0
        }
        return stats

    def print_performance_summary(self):
        """Print a summary of performance statistics"""
        stats = self.get_performance_stats()

        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total frames processed:     {stats['total_frames']}")
        print(f"Face detections run:        {stats['detection_count']}")
        print(f"Detection frequency:        Every {stats['total_frames'] / max(stats['detection_count'], 1):.1f} frames")
        print(f"Current adaptive interval:  {stats['current_interval']}")
        print(f"\nTiming (per frame):")
        print(f"  Detection time:           {stats['avg_detection_time']*1000:.1f} ms")
        print(f"  Landmark time:            {stats['avg_landmark_time']*1000:.1f} ms")
        print(f"  Total processing:         {stats['avg_total_time']*1000:.1f} ms")
        print(f"\nEstimated FPS:              {stats['fps_estimate']:.1f}")
        print("="*60)
