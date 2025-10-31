#!/usr/bin/env python3
"""
PFLD 68-Point Landmark Detector (Component 3)

Lightweight ONNX-based landmark detector using PFLD architecture.
Designed to replace dlib for better PyInstaller compatibility.

Model: pytorch_face_landmark PFLD (112×112 input)
Input: Face bounding box from RetinaFace
Output: 68 facial landmarks in iBUG format
"""

import numpy as np
import cv2
import onnxruntime as ort


class PFLDLandmarkDetector:
    """
    PFLD 68-point facial landmark detector using ONNX runtime.

    This detector takes face crops from RetinaFace and returns 68 landmarks
    in the standard iBUG format expected by OpenFace AU extraction.

    Model details:
    - Input: 112×112 RGB image (normalized 0-1)
    - Output: 136 values (68 × 2 coordinates)
    - Speed: ~100 FPS on CPU
    - Size: 2.8MB (vs 95MB for dlib)
    """

    def __init__(self, onnx_path='weights/pfld_68_landmarks.onnx', use_gpu=False):
        """
        Initialize PFLD landmark detector.

        Args:
            onnx_path: Path to PFLD ONNX model
            use_gpu: Use CUDA if available (default: CPU only for stability)
        """
        self.input_size = 112

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
        Preprocess face crop for PFLD model.

        Args:
            face_crop: BGR face crop from RetinaFace detection

        Returns:
            Preprocessed tensor ready for inference [1, 3, 112, 112]
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Resize to 112×112
        face_resized = cv2.resize(face_rgb, (self.input_size, self.input_size),
                                  interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0

        # HWC → CHW
        face_chw = np.transpose(face_normalized, (2, 0, 1))

        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)

        return face_batch

    def postprocess_landmarks(self, output, bbox):
        """
        Convert model output to absolute landmark coordinates.

        Args:
            output: Raw model output [136] (normalized coordinates)
            bbox: Face bounding box [x1, y1, x2, y2]

        Returns:
            landmarks: (68, 2) array of absolute (x, y) coordinates
        """
        # Reshape to (68, 2)
        landmarks_normalized = output.reshape(68, 2)

        # Scale to face bounding box
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        landmarks_abs = landmarks_normalized.copy()
        landmarks_abs[:, 0] = landmarks_normalized[:, 0] * width + x1
        landmarks_abs[:, 1] = landmarks_normalized[:, 1] * height + y1

        return landmarks_abs

    def detect_landmarks(self, frame, bbox):
        """
        Detect 68 facial landmarks from face bounding box.

        Args:
            frame: Full frame (BGR)
            bbox: Face bounding box [x1, y1, x2, y2] from RetinaFace

        Returns:
            landmarks: (68, 2) array of (x, y) coordinates, or None if failed
        """
        # Extract face crop with padding
        x1, y1, x2, y2 = map(int, bbox)

        # Add 10% padding for better landmark detection
        width = x2 - x1
        height = y2 - y1
        pad_w = int(width * 0.1)
        pad_h = int(height * 0.1)

        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(frame.shape[1], x2 + pad_w)
        y2_pad = min(frame.shape[0], y2 + pad_h)

        face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if face_crop.size == 0:
            return None

        # Preprocess
        input_tensor = self.preprocess_face(face_crop)

        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        landmarks_raw = outputs[0][0]  # Remove batch dimension

        # Postprocess (landmarks are relative to padded bbox)
        landmarks = self.postprocess_landmarks(landmarks_raw, [x1_pad, y1_pad, x2_pad, y2_pad])

        return landmarks

    def detect_landmarks_batch(self, frame, bboxes):
        """
        Detect landmarks for multiple faces in batch.

        Args:
            frame: Full frame (BGR)
            bboxes: List of face bounding boxes [[x1, y1, x2, y2], ...]

        Returns:
            List of landmark arrays [(68, 2), ...]
        """
        landmarks_list = []

        for bbox in bboxes:
            landmarks = self.detect_landmarks(frame, bbox)
            landmarks_list.append(landmarks)

        return landmarks_list


def visualize_landmarks(frame, landmarks, color=(0, 255, 0), radius=2):
    """
    Visualize 68 landmarks on frame for debugging.

    Args:
        frame: BGR image
        landmarks: (68, 2) array of (x, y) coordinates
        color: BGR color tuple
        radius: Point radius

    Returns:
        Annotated frame
    """
    frame_vis = frame.copy()

    for i, (x, y) in enumerate(landmarks):
        cv2.circle(frame_vis, (int(x), int(y)), radius, color, -1)

        # Different colors for different facial regions (optional)
        if i < 17:  # Jaw
            cv2.circle(frame_vis, (int(x), int(y)), radius, (255, 0, 0), -1)
        elif i < 27:  # Eyebrows
            cv2.circle(frame_vis, (int(x), int(y)), radius, (0, 255, 0), -1)
        elif i < 36:  # Nose
            cv2.circle(frame_vis, (int(x), int(y)), radius, (0, 0, 255), -1)
        elif i < 48:  # Eyes
            cv2.circle(frame_vis, (int(x), int(y)), radius, (255, 255, 0), -1)
        else:  # Mouth
            cv2.circle(frame_vis, (int(x), int(y)), radius, (255, 0, 255), -1)

    return frame_vis
