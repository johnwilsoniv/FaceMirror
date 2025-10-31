#!/usr/bin/env python3
"""
FAN2 68-Point Landmark Detector (Component 3)

Face Alignment Network (FAN2) trained on 300W dataset.
More accurate than PFLD, especially for challenging poses.

Model: fan2_68_landmark.onnx (50.9MB)
Input: 256×256 RGB, normalized [0-1]
Output: 68 landmarks with (x, y, confidence)
"""

import numpy as np
import cv2
import onnxruntime as ort


class FAN2LandmarkDetector:
    """
    FAN2 68-point facial landmark detector using ONNX runtime.

    Face Alignment Network trained on 300W-LP and iBUG datasets.
    Known for high accuracy, especially on challenging poses and occlusions.

    Model details:
    - Input: 256×256 RGB image (normalized 0-1)
    - Output: 68 × 3 values (x, y, confidence for each landmark)
    - Architecture: Hourglass network (coordinate-based variant)
    - Size: 50.9MB
    """

    def __init__(self, onnx_path='weights/fan2_68_landmark.onnx', use_gpu=False):
        """
        Initialize FAN2 landmark detector.

        Args:
            onnx_path: Path to FAN2 ONNX model
            use_gpu: Use CUDA if available (default: CPU only for stability)
        """
        self.input_size = 256

        # Setup ONNX session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_face(self, face_crop):
        """
        Preprocess face crop for FAN2 model.

        Args:
            face_crop: BGR face crop from detection

        Returns:
            Preprocessed tensor ready for inference [1, 3, 256, 256]
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Resize to 256×256
        face_resized = cv2.resize(face_rgb, (self.input_size, self.input_size),
                                  interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0

        # HWC → CHW
        face_chw = np.transpose(face_normalized, (2, 0, 1))

        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)

        return face_batch

    def postprocess_landmarks(self, output, bbox, original_crop_size):
        """
        Convert model output to absolute landmark coordinates.

        Args:
            output: Raw model output [68, 3] (x, y, confidence)
            bbox: Face bounding box [x1, y1, x2, y2] in original image
            original_crop_size: (width, height) of face crop before resize

        Returns:
            landmarks: (68, 2) array of absolute (x, y) coordinates
            confidences: (68,) array of confidence scores
        """
        # Extract coordinates and confidences
        # Output is in 64×64 heatmap coordinate space (not 256×256)
        landmarks_normalized = output[:, :2]  # (68, 2)
        confidences = output[:, 2]  # (68,)

        # Scale from 64×64 heatmap space to original crop size
        crop_width, crop_height = original_crop_size
        scale_x = crop_width / 64.0
        scale_y = crop_height / 64.0

        landmarks_scaled = landmarks_normalized.copy()
        landmarks_scaled[:, 0] *= scale_x
        landmarks_scaled[:, 1] *= scale_y

        # Translate to absolute image coordinates
        x1, y1, x2, y2 = bbox
        landmarks_abs = landmarks_scaled.copy()
        landmarks_abs[:, 0] += x1
        landmarks_abs[:, 1] += y1

        return landmarks_abs, confidences

    def detect_landmarks(self, frame, bbox):
        """
        Detect 68 facial landmarks from face bounding box.

        Args:
            frame: Full frame (BGR)
            bbox: Face bounding box [x1, y1, x2, y2]

        Returns:
            landmarks: (68, 2) array of (x, y) coordinates, or None if failed
            confidences: (68,) array of confidence scores, or None if failed
        """
        # Extract face crop
        x1, y1, x2, y2 = map(int, bbox)

        # Clip to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, None

        # Store original crop size for postprocessing
        original_crop_size = (face_crop.shape[1], face_crop.shape[0])

        # Preprocess
        input_tensor = self.preprocess_face(face_crop)

        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        landmarks_raw = outputs[0][0]  # Remove batch dimension: (68, 3)

        # Postprocess (scale to absolute coordinates)
        landmarks, confidences = self.postprocess_landmarks(
            landmarks_raw,
            [x1, y1, x2, y2],
            original_crop_size
        )

        return landmarks, confidences

    def detect_landmarks_batch(self, frame, bboxes):
        """
        Detect landmarks for multiple faces in batch.

        Args:
            frame: Full frame (BGR)
            bboxes: List of face bounding boxes [[x1, y1, x2, y2], ...]

        Returns:
            List of (landmarks, confidences) tuples
        """
        results = []

        for bbox in bboxes:
            landmarks, confidences = self.detect_landmarks(frame, bbox)
            results.append((landmarks, confidences))

        return results


def visualize_landmarks(frame, landmarks, confidences=None, color=(0, 255, 0), radius=2):
    """
    Visualize 68 landmarks on frame.

    Args:
        frame: BGR image
        landmarks: (68, 2) array of (x, y) coordinates
        confidences: Optional (68,) array of confidence scores
        color: BGR color tuple (or can color by confidence)
        radius: Point radius

    Returns:
        Annotated frame
    """
    frame_vis = frame.copy()

    for i, (x, y) in enumerate(landmarks):
        # Color by confidence if provided
        if confidences is not None:
            conf = confidences[i]
            # Red (low confidence) to green (high confidence)
            r = int(255 * (1 - conf))
            g = int(255 * conf)
            b = 0
            point_color = (b, g, r)
        else:
            point_color = color

        cv2.circle(frame_vis, (int(x), int(y)), radius, point_color, -1)

    return frame_vis
