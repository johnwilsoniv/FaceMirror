#!/usr/bin/env python3
"""
68-point facial landmark detector using PyFaceAU components.

This replaces the OpenFace 3.0 STAR 98-point system with PyFaceAU's
68-point PFLD detector + CLNF refinement for improved accuracy.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
from pyfaceau.detectors.pfld import CunjianPFLDDetector
from pyfaceau.refinement.targeted_refiner import TargetedCLNFRefiner
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN


class PyFaceAU68LandmarkDetector:
    """
    Face detection and 68-point landmark tracking using PyFaceAU components.

    This class provides:
    - RetinaFace face detection (CoreML accelerated)
    - PFLD 68-point landmarks (4.37% NME accuracy)
    - Optional CLNF refinement (improves AU accuracy to r=0.92)
    - Temporal smoothing (5-frame history)
    - Head pose estimation
    """

    def __init__(self, debug_mode=False, device='cpu', model_dir=None,
                 skip_redetection=False, skip_face_detection=False,
                 use_clnf_refinement=True):
        """
        Initialize PyFaceAU 68-point detector.

        Args:
            debug_mode: Enable debug output
            device: Ignored - PyFaceAU auto-detects best backend
            model_dir: Directory containing model weights (defaults to ./weights)
            skip_redetection: Skip RetinaFace after first frame (tracking mode)
            skip_face_detection: Skip RetinaFace entirely, use default bbox
            use_clnf_refinement: Enable CLNF landmark refinement (default: True)
        """
        self.skip_face_detection = skip_face_detection
        self.skip_redetection = skip_redetection
        self.use_clnf_refinement = use_clnf_refinement
        self.debug_mode = debug_mode

        # Determine weights directory
        if model_dir is None:
            import config_paths
            model_dir = str(config_paths.get_weights_dir())
        model_dir = Path(model_dir)

        if debug_mode:
            print("\n" + "="*60)
            print("PYFACEAU 68-POINT LANDMARK DETECTOR")
            print("="*60)

        # Initialize RetinaFace detector (unless skipped)
        if skip_face_detection:
            self.face_detector = None
            if debug_mode:
                print("  RetinaFace: Skipped (using default bbox)")
        else:
            self.face_detector = ONNXRetinaFaceDetector(
                str(model_dir / 'retinaface_mobilenet025_coreml.onnx'),
                use_coreml=True,
                confidence_threshold=0.5,
                nms_threshold=0.4
            )
            if debug_mode:
                print("  RetinaFace: Loaded (CoreML accelerated)")

        # Initialize PFLD 68-point landmark detector
        self.landmark_detector = CunjianPFLDDetector(
            str(model_dir / 'pfld_cunjian.onnx'),
            use_coreml=True
        )
        if debug_mode:
            print("  PFLD: Loaded (68-point, 4.37% NME accuracy)")

        # Initialize CLNF refiner (optional)
        if use_clnf_refinement:
            patch_expert_file = str(model_dir / 'svr_patches_0.25_general.txt')
            self.clnf_refiner = TargetedCLNFRefiner(
                patch_expert_file,
                search_window=3,
                pdm=None,  # No PDM enforcement for mirroring (only for AU extraction)
                enforce_pdm=False
            )
            if debug_mode:
                print(f"  CLNF Refiner: Loaded ({len(self.clnf_refiner.patch_experts)} patch experts)")
        else:
            self.clnf_refiner = None
            if debug_mode:
                print("  CLNF Refiner: Disabled")

        # Initialize MTCNN fallback detector
        try:
            # MTCNN weights are in pyfaceau package
            pyfaceau_dir = Path(__file__).parent.parent / "pyfaceau" / "pyfaceau" / "detectors"
            mtcnn_weights = pyfaceau_dir / "openface_mtcnn_weights.pth"

            if not mtcnn_weights.exists():
                # Try alternative location
                mtcnn_weights = model_dir / 'openface_mtcnn_weights.pth'

            # Use relaxed thresholds for challenging videos (paralysis patients)
            # Analysis showed [0.3, 0.4, 0.4] works for both IMG_8401 and IMG_9330
            self.mtcnn_detector = OpenFaceMTCNN(
                weights_path=str(mtcnn_weights),
                min_face_size=60,
                thresholds=[0.3, 0.4, 0.4]  # Relaxed for paralysis patients
            )
            if debug_mode:
                print("  MTCNN Fallback: Loaded (for failure recovery)")
        except Exception as e:
            self.mtcnn_detector = None
            if debug_mode:
                print(f"  MTCNN Fallback: Not available ({e})")

        if debug_mode:
            print("="*60 + "\n")

        # Tracking state
        self.last_face = None
        self.last_landmarks = None
        self.frame_count = 0
        self.cached_bbox = None

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
            print("Warming up models...")

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            if self.face_detector:
                _ = self.face_detector.detect_faces(dummy_frame)

            # Warm up landmark detector
            dummy_bbox = np.array([100, 100, 200, 200])
            _ = self.landmark_detector.detect_landmarks(dummy_frame, dummy_bbox)

            if self.debug_mode:
                print("  Models warmed up\n")
        except Exception as e:
            if self.debug_mode:
                print(f"  Warmup warning (non-critical): {e}\n")

    def reset_tracking_history(self):
        """Reset all tracking history between videos"""
        self.last_face = None
        self.last_landmarks = None
        self.cached_bbox = None
        self.frame_count = 0
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

    def _validate_landmarks(self, landmarks, bbox, frame_shape):
        """
        Validate if landmarks are properly placed on the face.

        Primary failure indicator: Landmarks should cover a reasonable portion of frame width.
        When RetinaFace detects a partial face (e.g., only one eye), landmark spread is small
        relative to frame size.

        Args:
            landmarks: (68, 2) array of landmarks
            bbox: [x1, y1, x2, y2] (for reference)
            frame_shape: (height, width, channels)

        Returns:
            is_valid: bool
            reason: str (why it failed or "Valid")
            confidence: float 0-1
        """
        if landmarks is None or bbox is None or len(landmarks) != 68:
            return False, "No detection", 0.0

        frame_h, frame_w = frame_shape[:2]

        # Get landmark bounds
        x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

        # Calculate spread as percentage of frame
        spread_x = x_max - x_min
        spread_y = y_max - y_min

        spread_x_pct = spread_x / frame_w * 100
        spread_y_pct = spread_y / frame_h * 100

        # Analysis from test videos (frame: 1080x1920):
        # IMG_8401 w/MTCNN (PASS): 78.8% x 52.7%
        # IMG_9330 w/RetinaFace (SHOULD FAIL): 47.7% x 25.6% - bbox cuts off chin!
        # IMG_0434 (PASS): 34.4% x 19.1%
        # IMG_0942 (PASS): 34.8% x 21.3%

        # Key insight: IMG_9330's abnormally wide spread (47.7%) with low vertical (25.6%)
        # indicates bbox cutting off lower face. Use ratio check to catch this.

        # Check minimum spreads (catch total failures)
        if spread_x_pct < 30:
            return False, f"Low horizontal spread ({spread_x_pct:.0f}% of frame)", spread_x_pct / 100

        if spread_y_pct < 15:
            return False, f"Low vertical spread ({spread_y_pct:.0f}% of frame)", spread_y_pct / 100

        # Check if spread is too wide relative to height (indicates cut-off face)
        # IMG_9330: 47.7 / 25.6 = 1.86 (too wide!)
        # IMG_0434: 34.4 / 19.1 = 1.80 (normal)
        # IMG_0942: 34.8 / 21.3 = 1.63 (normal)
        # Threshold: ratio > 1.85 indicates bbox cut off bottom
        spread_ratio = spread_x_pct / spread_y_pct if spread_y_pct > 0 else 0
        if spread_ratio > 1.85:
            return False, f"Abnormal spread ratio ({spread_ratio:.2f})", 1.0 / spread_ratio

        # All checks passed
        confidence = min(spread_x_pct / 100, spread_y_pct / 100, 1.0)
        return True, "Valid", confidence

    def _select_best_face_bbox(self, detections, frame_shape):
        """
        Select the best face bbox from multiple detections.

        Problem: RetinaFace sometimes gives highest confidence to partial faces
        (e.g., just one eye) especially with surgical markings or paralysis.

        Solution: Score each bbox by size, position, and aspect ratio.

        Args:
            detections: Array of detections, each [x1, y1, x2, y2, conf]
            frame_shape: (height, width, channels) of frame

        Returns:
            Best detection [x1, y1, x2, y2, conf]
        """
        if len(detections) == 1:
            return detections[0]

        frame_h, frame_w = frame_shape[:2]
        frame_center_y = frame_h / 2

        scores = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            conf = det[4]

            # Calculate bbox properties
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_area = bbox_w * bbox_h
            bbox_center_y = (y1 + y2) / 2

            # Size score: prefer larger bboxes (normalized to frame)
            size_ratio = bbox_area / (frame_w * frame_h)
            size_score = min(size_ratio / 0.3, 1.0)  # Target ~30% of frame

            # Vertical position score: prefer vertically centered faces
            y_offset = abs(bbox_center_y - frame_center_y)
            y_offset_ratio = y_offset / frame_h
            position_score = max(0, 1.0 - (y_offset_ratio / 0.3))

            # Aspect ratio score: faces should be roughly square (0.75 to 1.3)
            aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 0
            if 0.75 <= aspect_ratio <= 1.3:
                aspect_score = 1.0
            elif 0.5 <= aspect_ratio < 0.75 or 1.3 < aspect_ratio <= 1.5:
                aspect_score = 0.7
            else:
                aspect_score = 0.3

            # Confidence score (normalized)
            conf_score = min(conf, 1.0)

            # Weighted total score
            # Size and aspect ratio are most important (they distinguish full vs partial faces)
            # Position helps when multiple full faces detected
            # Confidence is least important (can be misleading)
            total_score = (
                size_score * 0.4 +
                aspect_score * 0.3 +
                position_score * 0.2 +
                conf_score * 0.1
            )

            scores.append(total_score)

            if self.debug_mode:
                print(f"  Det bbox=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}] "
                      f"conf={conf:.3f} size={size_score:.2f} pos={position_score:.2f} "
                      f"aspect={aspect_score:.2f} → score={total_score:.3f}")

        # Select detection with highest score
        best_idx = np.argmax(scores)

        if self.debug_mode:
            print(f"  Selected detection {best_idx} (score={scores[best_idx]:.3f})")

        return detections[best_idx]

    def get_face_mesh(self, frame, detection_interval=2):
        """
        Get 68-point facial landmarks with temporal smoothing.

        Args:
            frame: BGR image (numpy array)
            detection_interval: Ignored - uses skip_redetection flag

        Returns:
            smoothed_points: (68, 2) array of smoothed landmark coordinates
            triangles: None (not used)
        """
        self.frame_count += 1

        # Handle skip_face_detection mode
        if self.skip_face_detection:
            if self.cached_bbox is None:
                # Create default centered bbox
                h, w = frame.shape[:2]
                margin = 0.1
                x1 = int(w * margin)
                y1 = int(h * margin)
                x2 = int(w * (1 - margin))
                y2 = int(h * (1 - margin))
                self.cached_bbox = np.array([x1, y1, x2, y2])

                if self.debug_mode:
                    print(f"  Using default centered bbox: {x1},{y1},{x2},{y2}")

        # Detect face (only if needed)
        should_detect = (
            (self.cached_bbox is None) and
            not self.skip_face_detection and
            (self.frame_count == 1 or not self.skip_redetection)
        )

        if should_detect:
            try:
                detections, _ = self.face_detector.detect_faces(frame)

                if detections is None or len(detections) == 0:
                    self.cached_bbox = None
                    self.last_face = None
                    self.last_landmarks = None
                    return None, None

                # Select best face bbox (not just highest confidence)
                det = self._select_best_face_bbox(detections, frame.shape)
                bbox = det[:4].astype(int)  # [x1, y1, x2, y2]
                self.cached_bbox = bbox
                self.last_face = det

            except Exception as e:
                if self.debug_mode:
                    print(f"Face detection error: {e}")
                return None, None

        # Detect landmarks
        if self.cached_bbox is not None:
            try:
                landmarks_68, _ = self.landmark_detector.detect_landmarks(
                    frame,
                    self.cached_bbox
                )

                # Apply CLNF refinement if enabled
                if self.use_clnf_refinement and self.clnf_refiner is not None:
                    landmarks_68 = self.clnf_refiner.refine_landmarks(frame, landmarks_68)

                # Validate landmark placement with ORIGINAL bbox and frame size
                is_valid, reason, confidence = self._validate_landmarks(landmarks_68, self.cached_bbox, frame.shape)

                # Track validation results for return
                validation_info = {
                    'validation_passed': is_valid,
                    'reason': reason,
                    'confidence': confidence,
                    'used_fallback': False
                }

                # After validation, expand bbox to contain all landmarks (with 10% padding)
                # This ensures final bbox properly contains all detected landmarks
                lm_x_min, lm_x_max = np.min(landmarks_68[:, 0]), np.max(landmarks_68[:, 0])
                lm_y_min, lm_y_max = np.min(landmarks_68[:, 1]), np.max(landmarks_68[:, 1])

                lm_w = lm_x_max - lm_x_min
                lm_h = lm_y_max - lm_y_min

                # Add 10% padding on each side
                padding_x = lm_w * 0.10
                padding_y = lm_h * 0.10

                expanded_bbox = np.array([
                    max(0, lm_x_min - padding_x),
                    max(0, lm_y_min - padding_y),
                    min(frame.shape[1], lm_x_max + padding_x),
                    min(frame.shape[0], lm_y_max + padding_y)
                ]).astype(int)

                # Update cached bbox to expanded version for final output
                self.cached_bbox = expanded_bbox

                # If validation failed, try MTCNN fallback
                if not is_valid and self.mtcnn_detector is not None:
                    if self.debug_mode:
                        print(f"  ⚠️  RetinaFace validation failed: {reason} (conf={confidence:.2f})")
                        print(f"  🔄 Retrying with MTCNN fallback...")

                    try:
                        # Detect with MTCNN
                        mtcnn_bboxes, _ = self.mtcnn_detector.detect(frame, return_landmarks=False)

                        if mtcnn_bboxes is not None and len(mtcnn_bboxes) > 0:
                            # Select best MTCNN bbox
                            # MTCNN bboxes are already in [x1, y1, x2, y2] format
                            mtcnn_dets = np.column_stack([mtcnn_bboxes, np.ones(len(mtcnn_bboxes))])
                            mtcnn_det = self._select_best_face_bbox(mtcnn_dets, frame.shape)
                            mtcnn_bbox = mtcnn_det[:4].astype(int)

                            # Clip bbox to frame boundaries (MTCNN can return negative coords)
                            h, w = frame.shape[:2]
                            mtcnn_bbox[0] = max(0, mtcnn_bbox[0])
                            mtcnn_bbox[1] = max(0, mtcnn_bbox[1])
                            mtcnn_bbox[2] = min(w, mtcnn_bbox[2])
                            mtcnn_bbox[3] = min(h, mtcnn_bbox[3])

                            if self.debug_mode:
                                print(f"  MTCNN bbox (clipped): {mtcnn_bbox}")

                            # Re-detect landmarks with MTCNN bbox
                            landmarks_68, _ = self.landmark_detector.detect_landmarks(frame, mtcnn_bbox)

                            # Re-apply CLNF refinement
                            if self.use_clnf_refinement and self.clnf_refiner is not None:
                                landmarks_68 = self.clnf_refiner.refine_landmarks(frame, landmarks_68)

                            # Re-validate with MTCNN bbox
                            is_valid_mtcnn, reason_mtcnn, confidence_mtcnn = self._validate_landmarks(
                                landmarks_68, mtcnn_bbox, frame.shape
                            )

                            if is_valid_mtcnn:
                                # Success! Expand bbox to contain all landmarks
                                lm_x_min, lm_x_max = np.min(landmarks_68[:, 0]), np.max(landmarks_68[:, 0])
                                lm_y_min, lm_y_max = np.min(landmarks_68[:, 1]), np.max(landmarks_68[:, 1])

                                lm_w = lm_x_max - lm_x_min
                                lm_h = lm_y_max - lm_y_min

                                padding_x = lm_w * 0.10
                                padding_y = lm_h * 0.10

                                mtcnn_bbox_expanded = np.array([
                                    max(0, lm_x_min - padding_x),
                                    max(0, lm_y_min - padding_y),
                                    min(frame.shape[1], lm_x_max + padding_x),
                                    min(frame.shape[0], lm_y_max + padding_y)
                                ]).astype(int)

                                # Use expanded bbox for final output
                                self.cached_bbox = mtcnn_bbox_expanded

                                # Update validation info for successful MTCNN fallback
                                validation_info = {
                                    'validation_passed': True,
                                    'reason': reason_mtcnn,
                                    'confidence': confidence_mtcnn,
                                    'used_fallback': True
                                }

                                if self.debug_mode:
                                    print(f"  ✅ MTCNN fallback succeeded: {reason_mtcnn} (conf={confidence_mtcnn:.2f})")
                            else:
                                # Update validation info for failed MTCNN fallback
                                validation_info = {
                                    'validation_passed': False,
                                    'reason': f"Both detectors failed: RetinaFace ({reason}), MTCNN ({reason_mtcnn})",
                                    'confidence': max(confidence, confidence_mtcnn),
                                    'used_fallback': True
                                }

                                if self.debug_mode:
                                    print(f"  ❌ MTCNN fallback also failed: {reason_mtcnn} (conf={confidence_mtcnn:.2f})")
                        else:
                            if self.debug_mode:
                                print(f"  ❌ MTCNN found no faces")

                    except Exception as e:
                        if self.debug_mode:
                            print(f"  ❌ MTCNN fallback error: {e}")

                elif not is_valid and self.debug_mode:
                    print(f"  ⚠️  Validation failed: {reason} (conf={confidence:.2f}, no MTCNN fallback)")

                # Ensure float32 for calculations
                landmarks_68 = landmarks_68.astype(np.float32)

                # Apply temporal smoothing (5-frame weighted average)
                self.landmarks_history.append(landmarks_68.copy())
                if len(self.landmarks_history) > self.history_size:
                    self.landmarks_history.pop(0)

                # Weighted average (more weight to recent frames)
                weights = np.linspace(0.5, 1.0, len(self.landmarks_history))
                weights = weights / np.sum(weights)

                smoothed_points = np.zeros_like(landmarks_68, dtype=np.float32)
                for pts, w in zip(self.landmarks_history, weights):
                    smoothed_points += pts * w

                # Convert to integer coordinates
                smoothed_points = np.round(smoothed_points).astype(np.int32)

                # Update tracking
                self.last_landmarks = smoothed_points

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

                # Print warning if validation failed
                if not validation_info['validation_passed']:
                    if self.debug_mode:
                        print(f"  ⚠️  WARNING: Landmark detection may be unreliable - {validation_info['reason']}")

                return smoothed_points, validation_info

            except Exception as e:
                # Landmark detection failed - ALWAYS try to reuse previous landmarks first
                if self.last_landmarks is not None:
                    if self.debug_mode:
                        print(f"  Landmark/CLNF failed (frame {self.frame_count}), reusing previous: {type(e).__name__}")
                    return self.last_landmarks.copy(), None

                # No previous landmarks available
                if self.debug_mode:
                    print(f"Landmark detection error (frame {self.frame_count}): {type(e).__name__}: {e}")

                # Try re-detecting face if not in skip mode
                if not self.skip_face_detection and not self.skip_redetection:
                    self.cached_bbox = None

                return None, None

        return None, None

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

        # Convert to float32 for calculations
        landmarks = landmarks.astype(np.float32)

        # Get medial eyebrow points (same as open2GR)
        left_medial_brow = landmarks[21]   # Left eyebrow inner corner
        right_medial_brow = landmarks[22]  # Right eyebrow inner corner

        # Calculate glabella and chin
        glabella = (left_medial_brow + right_medial_brow) / 2
        chin = landmarks[8]  # Chin center

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

        Uses the same algorithm as open2GR for consistency.

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

                # Higher weight for eye landmarks (more reliable)
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
        """
        Print performance summary (compatibility method).

        PyFaceAU uses internal profiling. This method is kept for API
        compatibility with the video processor.
        """
        # No-op: PyFaceAU reports performance during processing
        # This is kept to prevent AttributeError when called by video_processor
        pass

    def cleanup_memory(self):
        """Cleanup GPU memory (compatibility method for old interface)"""
        # PyFaceAU doesn't require explicit GPU memory cleanup
        # This is kept for API compatibility with OpenFace3
        pass
