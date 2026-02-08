#!/usr/bin/env python3
"""
SPIGA/FaceNet Landmark Detector

Uses SPIGA for 98-point WFLW landmark detection and facenet-pytorch MTCNN
for face detection. Converts 98-point output to 68-point dlib format
for compatibility with pyfaceau AU prediction pipeline.

Architecture:
    FaceNet MTCNN → SPIGA (98 landmarks) → Map 98→68 → PyFaceAU

Dependencies:
    - spiga: SPIGA landmark detection model
    - facenet-pytorch: MTCNN face detection

Usage:
    detector = SPIGALandmarkDetector()
    landmarks_68, info = detector.get_face_mesh(frame)
"""

import numpy as np
import cv2
import threading
from typing import Optional, Tuple
from pathlib import Path

from landmark_mapper import wflw_to_dlib_68, validate_mapping


def safe_print(*args, **kwargs):
    """Print wrapper that handles BrokenPipeError in GUI subprocess contexts."""
    import builtins
    try:
        builtins.print(*args, **kwargs)
    except (BrokenPipeError, IOError):
        pass


class SPIGALandmarkDetector:
    """
    Face detection and 98-to-68 landmark mapping using SPIGA/FaceNet.

    This class provides:
    - FaceNet MTCNN face detection
    - SPIGA 98-point landmarks (WFLW format)
    - 98→68 landmark mapping for pyfaceau compatibility
    - Temporal smoothing (5-frame history)
    - Head pose estimation
    """

    def __init__(self, debug_mode=False, device='auto'):
        """
        Initialize SPIGA detector with 68-point output.

        Args:
            debug_mode: Enable debug output
            device: Device selection ('auto', 'mps', 'cuda', 'cpu')
        """
        self.debug_mode = debug_mode
        self._spiga_available = False
        self._mtcnn_available = False

        if debug_mode:
            safe_print("\n" + "="*60)
            safe_print("SPIGA LANDMARK DETECTOR (98→68 mapping)")
            safe_print("="*60)

        # Determine device
        import torch
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if debug_mode:
            safe_print(f"  Device: {self.device}")

        # Initialize facenet-pytorch MTCNN for face detection
        try:
            from facenet_pytorch import MTCNN

            # Use CPU for MTCNN as it's more stable
            mtcnn_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
            self.mtcnn = MTCNN(
                keep_all=False,  # Only return largest face
                device=mtcnn_device,
                select_largest=True,
                post_process=False,
            )
            self._mtcnn_available = True

            if debug_mode:
                safe_print(f"  FaceNet MTCNN: Loaded (device: {mtcnn_device})")

        except ImportError as e:
            if debug_mode:
                safe_print(f"  FaceNet MTCNN: Not available ({e})")
            raise ImportError(
                "facenet-pytorch is not installed. Install with: pip install facenet-pytorch"
            ) from e

        # Initialize SPIGA for landmark detection
        try:
            from spiga.inference.config import ModelConfig
            import spiga.inference.pretreatment as pretreat
            from spiga.models.spiga import SPIGA
            import pkg_resources

            # SPIGA uses WFLW dataset configuration for 98 landmarks
            self.spiga_config = ModelConfig('wflw')

            # Custom SPIGA initialization that supports CPU/MPS (not just CUDA)
            # The original SPIGAFramework hardcodes CUDA, so we re-implement the loading
            weights_path = pkg_resources.resource_filename('spiga', 'models/weights')
            self._spiga_transforms = pretreat.get_transformers(self.spiga_config)

            # Create model
            self._spiga_model = SPIGA(
                num_landmarks=self.spiga_config.dataset.num_landmarks,
                num_edges=self.spiga_config.dataset.num_edges
            )

            # Load weights
            weights_file = f"{weights_path}/{self.spiga_config.model_weights}"
            if not Path(weights_file).exists():
                # Download if not present
                model_state_dict = torch.hub.load_state_dict_from_url(
                    self.spiga_config.model_weights_url,
                    model_dir=weights_path,
                    file_name=self.spiga_config.model_weights
                )
            else:
                model_state_dict = torch.load(weights_file, map_location='cpu')

            self._spiga_model.load_state_dict(model_state_dict)

            # Force CPU for SPIGA - MPS has im2col fallback overhead
            self._spiga_device = torch.device('cpu')
            self._spiga_model = self._spiga_model.to(self._spiga_device)
            self._spiga_model.eval()

            # Load 3D model for pose estimation
            loader_3DM = pretreat.AddModel3D(
                self.spiga_config.dataset.ldm_ids,
                ftmap_size=self.spiga_config.ftmap_size,
                focal_ratio=self.spiga_config.focal_ratio,
                totensor=True
            )
            params_3DM = loader_3DM()
            self._spiga_model3d = params_3DM['model3d'].to(self._spiga_device)
            self._spiga_cam_matrix = params_3DM['cam_matrix'].to(self._spiga_device)
            # Pre-expand for single face (batch size 1) to avoid per-frame allocation
            self._spiga_model3d_batch = self._spiga_model3d.unsqueeze(0)
            self._spiga_cam_matrix_batch = self._spiga_cam_matrix.unsqueeze(0)

            self._spiga_available = True

            if debug_mode:
                safe_print(f"  SPIGA: Loaded (WFLW 98-point model, device: {self._spiga_device})")
                safe_print(f"  Output: 68-point dlib format (mapped)")

        except ImportError as e:
            if debug_mode:
                safe_print(f"  SPIGA: Not available ({e})")
            raise ImportError(
                "SPIGA is not installed. Install with: pip install spiga"
            ) from e

        if debug_mode:
            safe_print("="*60 + "\n")

        # Thread lock for model access
        self._detector_lock = threading.Lock()

        # Tracking state
        self.last_landmarks = None
        self.last_landmarks_98 = None  # Store original 98-point for debugging
        self.frame_count = 0
        self._frame_idx = 0
        self.cached_bbox = None

        # Temporal smoothing history (5-frame, matches pyfaceau)
        self.landmarks_history = []
        self.glabella_history = []
        self.chin_history = []
        self.yaw_history = []
        self.frame_quality_history = []
        self.history_size = 5

        # Warmup models
        self._warmup_models()

    @property
    def landmark_count(self):
        """Return the number of output landmarks (68 for dlib compatibility)."""
        return 68

    @property
    def native_landmark_count(self):
        """Return the native number of landmarks from SPIGA (98)."""
        return 98

    def _warmup_models(self):
        """Warm up models with dummy inference."""
        if self.debug_mode:
            safe_print("Warming up SPIGA/MTCNN models...")

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            # Run detection on dummy frame
            self._detect_face(dummy_frame)

            if self.debug_mode:
                safe_print("  Models warmed up\n")
        except Exception as e:
            if self.debug_mode:
                safe_print(f"  Warmup warning (non-critical): {e}\n")

    def reset_tracking_history(self):
        """Reset all tracking history between videos."""
        self.last_landmarks = None
        self.last_landmarks_98 = None
        self.frame_count = 0
        self._frame_idx = 0
        self.cached_bbox = None
        self.landmarks_history.clear()
        self.glabella_history.clear()
        self.chin_history.clear()
        self.yaw_history.clear()
        self.frame_quality_history.clear()

    def cleanup_memory(self):
        """Cleanup memory after processing."""
        self.reset_tracking_history()
        import gc
        gc.collect()

    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect face using FaceNet MTCNN.

        Args:
            frame: BGR image (numpy array)

        Returns:
            (x1, y1, x2, y2) bounding box or None if no face detected
        """
        # MTCNN expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            boxes, probs = self.mtcnn.detect(frame_rgb)

            if boxes is not None and len(boxes) > 0:
                # Get first (largest) face
                box = boxes[0]
                x1, y1, x2, y2 = box
                return (float(x1), float(y1), float(x2), float(y2))

        except Exception as e:
            if self.debug_mode:
                safe_print(f"MTCNN detection error: {e}")

        return None

    def _spiga_inference(self, image: np.ndarray, bboxes: list) -> dict:
        """
        Run SPIGA inference on an image with bounding boxes.

        This is a CPU/MPS-compatible reimplementation of SPIGAFramework.inference()

        Args:
            image: RGB image (numpy array)
            bboxes: List of bounding boxes [[x, y, w, h], ...]

        Returns:
            Dictionary with 'landmarks' key containing detected landmarks
        """
        import torch

        # Pretreatment: crop and transform faces
        crop_bboxes = []
        crop_images = []
        for bbox in bboxes:
            sample = {'image': image, 'bbox': list(bbox)}
            sample_crop = self._spiga_transforms(sample)
            crop_bboxes.append(sample_crop['bbox'])
            crop_images.append(sample_crop['image'])

        # Images to tensor and move to device
        batch_images = torch.from_numpy(np.array(crop_images)).float().to(self._spiga_device)

        # Use pre-expanded 3D model for single face (common case)
        if len(bboxes) == 1:
            batch_model3D = self._spiga_model3d_batch
            batch_cam_matrix = self._spiga_cam_matrix_batch
        else:
            batch_model3D = self._spiga_model3d.unsqueeze(0).expand(len(bboxes), -1, -1)
            batch_cam_matrix = self._spiga_cam_matrix.unsqueeze(0).expand(len(bboxes), -1, -1)

        # Run inference with inference_mode (faster than no_grad)
        model_inputs = [batch_images, batch_model3D, batch_cam_matrix]
        with torch.inference_mode():
            outputs = self._spiga_model(model_inputs)

        # Post-treatment: convert landmarks back to image coordinates
        # SPIGA's TargetCropAug expands the bbox by target_dist before scaling to 256x256
        # We need to compute the correct inverse affine transformation
        features = {}
        target_dist = self.spiga_config.target_dist  # 1.6 by default
        img_size_x, img_size_y = self.spiga_config.image_size  # (256, 256)

        if 'Landmarks' in outputs.keys():
            # Raw output shape: (batch, num_landmarks, 2)
            landmarks = outputs['Landmarks'][-1].cpu().numpy()
            # Scale from [0,1] to [0,256]
            landmarks = landmarks * np.array([img_size_x, img_size_y])

            # Apply correct inverse transformation for each bbox
            landmarks_out = []
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                # SPIGA expands the bbox to a square with side = max(w,h) * target_dist
                side = max(w, h) * target_dist
                # Scale factor: 256 / side (assuming square output)
                scale = img_size_x / side
                # Offset: where the expanded region starts in original image coords
                x_offset = x - (side - w) / 2
                y_offset = y - (side - h) / 2

                # Inverse transformation: from 256x256 space to original image
                lm = landmarks[i]  # (num_landmarks, 2)
                lm_orig = lm / scale + np.array([x_offset, y_offset])
                landmarks_out.append(lm_orig.tolist())

            # Output: list of faces, each face is list of [x,y] landmarks
            features['landmarks'] = landmarks_out

        # Extract head pose and compute 2D face center from 3D projection
        if 'Pose' in outputs.keys():
            pose = outputs['Pose'].cpu().numpy()  # (batch, 6): [pitch, yaw, roll, tx, ty, tz]

            # Get camera matrix and 3D model for projection
            cam = self._spiga_cam_matrix.cpu().numpy()  # (3, 3)
            model3d = self._spiga_model3d.cpu().numpy()  # (98, 3)

            pose_out = []
            face_centers = []
            for i, bbox in enumerate(bboxes):
                p = pose[i]  # [pitch, yaw, roll, tx, ty, tz]
                t = p[3:6]

                # Build rotation matrix matching pose_proj.py exactly:
                #   euler[0] = -(pitch - 90), euler[1] = -yaw, euler[2] = -(roll + 90)
                #   R = Ry @ Rp @ Rr  (yaw-pitch-roll order)
                a0 = np.radians(-(p[0] - 90))  # modified yaw
                a1 = np.radians(-p[1])          # modified pitch
                a2 = np.radians(-(p[2] + 90))   # modified roll
                cy_, sy_ = np.cos(a0), np.sin(a0)
                cp_, sp_ = np.cos(a1), np.sin(a1)
                cr_, sr_ = np.cos(a2), np.sin(a2)
                Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
                Rp = np.array([[cp_, -sp_, 0], [sp_, cp_, 0], [0, 0, 1]])
                Rr = np.array([[1, 0, 0], [0, cr_, -sr_], [0, sr_, cr_]])
                R = Ry @ Rp @ Rr

                # Project ALL 98 3D model points to 2D (not just centroid)
                # Under perspective projection, centroid(projected) != projected(centroid)
                # Camera matrix is for ftmap_size (64x64), scale to crop size (256x256)
                ftmap_to_crop = 4.0  # 256 / 64
                pts_cam = (R @ model3d.T).T + t  # (98, 3) in camera coords
                pts_proj = (cam @ pts_cam.T).T  # (98, 3) projected
                pts_2d = pts_proj[:, :2] / pts_proj[:, 2:3]  # (98, 2) perspective division
                center_ftmap = np.mean(pts_2d, axis=0)  # centroid of projected points
                u_crop = center_ftmap[0] * ftmap_to_crop
                v_crop = center_ftmap[1] * ftmap_to_crop

                # Transform from crop space to original image space
                x, y, w, h = bbox
                side = max(w, h) * target_dist
                scale = img_size_x / side
                x_offset = x - (side - w) / 2
                y_offset = y - (side - h) / 2

                u_image = u_crop / scale + x_offset
                v_image = v_crop / scale + y_offset
                face_centers.append([float(u_image), float(v_image)])
                pose_out.append(p.tolist())

            features['pose'] = pose_out
            features['face_center'] = face_centers

        return features

    def _detect_98_landmarks(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect 98-point WFLW landmarks using SPIGA.

        Args:
            frame: BGR image (numpy array)
            bbox: (x1, y1, x2, y2) face bounding box

        Returns:
            Tuple of:
                landmarks: (98, 2) array of landmarks or None if detection failed
                face_center: (2,) array [x, y] of 3D-projected face center, or None
        """
        # SPIGA expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = bbox

        # SPIGA expects bounding box as [x, y, w, h]
        spiga_bbox = [x1, y1, x2 - x1, y2 - y1]

        try:
            # SPIGA inference (using our CPU-compatible implementation)
            features = self._spiga_inference(frame_rgb, [spiga_bbox])

            if features and 'landmarks' in features:
                landmarks = features['landmarks'][0]  # First face
                face_center = None
                if 'face_center' in features:
                    face_center = np.array(features['face_center'][0], dtype=np.float32)
                return np.array(landmarks, dtype=np.float32), face_center

        except Exception as e:
            if self.debug_mode:
                safe_print(f"SPIGA detection error: {e}")

        return None, None

    def get_face_mesh(self, frame: np.ndarray, detection_interval: int = 30) -> Tuple[Optional[np.ndarray], dict]:
        """
        Get 68-point facial landmarks (mapped from 98-point SPIGA).

        Args:
            frame: BGR image (numpy array)
            detection_interval: Re-detect face every N frames (0 = every frame)

        Returns:
            smoothed_points: (68, 2) array of smoothed landmark coordinates
            info: Validation info dict
        """
        with self._detector_lock:
            self.frame_count += 1
            self._frame_idx += 1

            # Detect face (first frame or periodic refresh)
            # Note: If cached_bbox is already set (e.g., by SPIGALandmarkWrapper),
            # skip detection to use the provided bbox from pyfaceau pipeline
            should_detect = (
                self.cached_bbox is None or
                (detection_interval > 0 and self._frame_idx % detection_interval == 0)
            )

            if should_detect:
                bbox = self._detect_face(frame)
                if bbox is not None:
                    self.cached_bbox = bbox
                elif self.cached_bbox is None:
                    # No face found and no cached bbox
                    if self.last_landmarks is not None:
                        return self.last_landmarks.copy(), {'valid': True, 'reused': True}
                    return None, {'valid': False, 'reason': 'MTCNN failed to detect face'}

            # Detect 98-point landmarks using cached bbox
            landmarks_98, face_center = self._detect_98_landmarks(frame, self.cached_bbox)

            if landmarks_98 is None:
                # Try reusing previous landmarks
                if self.last_landmarks is not None:
                    return self.last_landmarks.copy(), {'valid': True, 'reused': True}
                return None, {'valid': False, 'reason': 'SPIGA failed to detect landmarks'}

            # Store 98-point landmarks for debugging
            self.last_landmarks_98 = landmarks_98.copy()

            # Map 98 → 68 landmarks
            landmarks_68 = wflw_to_dlib_68(landmarks_98)

            if landmarks_68 is None:
                if self.last_landmarks is not None:
                    return self.last_landmarks.copy(), {'valid': True, 'reused': True}
                return None, {'valid': False, 'reason': 'Landmark mapping failed'}

            # Validate mapping
            validation = validate_mapping(landmarks_98, landmarks_68)

            # Temporal smoothing (5-frame weighted average)
            self.landmarks_history.append(landmarks_68.copy())
            if len(self.landmarks_history) > self.history_size:
                self.landmarks_history.pop(0)

            # Weighted average (more weight to recent frames)
            weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
            weights = weights / np.sum(weights)

            smoothed_points = np.zeros_like(landmarks_68, dtype=np.float32)
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

            info = {
                'valid': validation['valid'],
                'mapping_warnings': validation.get('warnings', []),
                'mapping_stats': validation.get('stats', {}),
                'native_landmarks': 98,
                'output_landmarks': 68,
                'confidence': 1.0
            }
            if face_center is not None:
                info['face_center'] = face_center

            return smoothed_points, info

    def get_facial_midline(self, landmarks: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculate anatomical midline points using 68-point landmarks.

        68-point landmark indices (dlib standard):
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

    def calculate_head_pose(self, landmarks: np.ndarray) -> Optional[float]:
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
            return float(np.average(yaw_estimates, weights=weights))

        # Fallback to nose offset
        eyes_center = (landmarks[39] + landmarks[42]) / 2
        nose_tip = landmarks[30]
        nose_offset = nose_tip[0] - eyes_center[0]
        face_width = np.linalg.norm(landmarks[36] - landmarks[45])
        if face_width > 0:
            normalized_offset = nose_offset / (face_width / 2)
            return float(normalized_offset * 45.0)

        return None

    def calculate_face_stability(self) -> Tuple[float, bool]:
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

    def calculate_frame_quality(self, landmarks: np.ndarray) -> float:
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


if __name__ == "__main__":
    # Test the detector
    print("Testing SPIGA/FaceNet Landmark Detector...")

    try:
        detector = SPIGALandmarkDetector(debug_mode=True)
        print(f"\nDetector initialized:")
        print(f"  Output landmarks: {detector.landmark_count}")
        print(f"  Native landmarks: {detector.native_landmark_count}")

        # Test with dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks, info = detector.get_face_mesh(dummy_frame)

        print(f"\nTest detection result:")
        print(f"  Landmarks: {landmarks.shape if landmarks is not None else None}")
        print(f"  Info: {info}")

    except ImportError as e:
        print(f"\nDependencies not available: {e}")
        print("Install with: pip install spiga facenet-pytorch")
