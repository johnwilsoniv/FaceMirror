#!/usr/bin/env python3
"""
C++ OpenFace MTCNN Detector (ONNX Implementation)

Python implementation of MTCNN using the exact weights extracted from
C++ OpenFace, running on onnxruntime with CoreML acceleration support.

This detector should produce IDENTICAL bboxes to C++ OpenFace MTCNN.
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple


class CPPMTCNNDetector:
    """
    MTCNN face detector using C++ OpenFace's exact weights.

    This implementation replicates C++ OpenFace's MTCNN detection pipeline
    using the extracted ONNX models.
    """

    def __init__(self, model_dir: str = "cpp_mtcnn_onnx", use_coreml: bool = False):
        """
        Initialize MTCNN detector with extracted C++ weights.

        Args:
            model_dir: Directory containing pnet.onnx, rnet.onnx, onet.onnx
            use_coreml: Enable CoreML acceleration (ARM Mac only)
        """
        model_dir = Path(model_dir)

        # Setup ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Providers
        if use_coreml:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load models
        print(f"Loading C++ MTCNN models from {model_dir}...")
        self.pnet = ort.InferenceSession(str(model_dir / 'pnet.onnx'),
                                         sess_options=sess_options,
                                         providers=providers)
        self.rnet = ort.InferenceSession(str(model_dir / 'rnet.onnx'),
                                         sess_options=sess_options,
                                         providers=providers)
        self.onet = ort.InferenceSession(str(model_dir / 'onet.onnx'),
                                         sess_options=sess_options,
                                         providers=providers)
        print(f"  ✓ Models loaded (provider: {self.pnet.get_providers()[0]})")

        # Detection thresholds (from C++ OpenFace)
        self.thresholds = [0.6, 0.7, 0.7]  # PNet, RNet, ONet

        # Minimum face size
        self.min_face_size = 40

        # Scale factor for image pyramid
        self.factor = 0.709

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for MTCNN (matches C++ OpenFace exactly).

        Args:
            img: BGR image (H, W, 3)

        Returns:
            Preprocessed image (1, 3, H, W) normalized
        """
        # CRITICAL: C++ OpenFace uses BGR channel ordering (OpenCV default)
        # The weights were trained with BGR input, so we must NOT convert to RGB!
        # When C++ does cv::split(), it creates [B, G, R] channel order.
        # Our ONNX models preserve this BGR channel ordering.

        # Normalize exactly as C++ does: (x - 127.5) * 0.0078125
        # Note: 0.0078125 = 1/128, not 1/127.5!
        img = (img.astype(np.float32) - 127.5) * 0.0078125

        # Transpose to (C, H, W) and add batch dimension
        # This keeps BGR order: (H, W, BGR) → (BGR, H, W)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        return img

    def _generate_bboxes(self, score_map, reg_map, scale, threshold):
        """
        Generate bounding boxes from PNet output.

        Args:
            score_map: (H, W, 2) - classification scores
            reg_map: (H, W, 4) - bbox regression
            scale: Current scale factor
            threshold: Classification threshold

        Returns:
            bboxes: (N, 5) - [x1, y1, x2, y2, score]
        """
        stride = 2
        cellsize = 12

        # Get indices where score > threshold
        mask = score_map[:, :, 1] > threshold

        if not mask.any():
            return np.empty((0, 5))

        # Get bbox coordinates
        indices = np.where(mask)
        score = score_map[indices][:, 1]
        reg = reg_map[indices]

        # Calculate bbox positions (C++ uses int() truncation, not rounding)
        # FaceDetectorMTCNN.cpp:571-576
        y, x = indices
        x1 = ((stride * x + 1) / scale).astype(np.int32)
        y1 = ((stride * y + 1) / scale).astype(np.int32)
        x2 = ((stride * x + cellsize) / scale).astype(np.int32)
        y2 = ((stride * y + cellsize) / scale).astype(np.int32)

        # Apply regression
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        x1 = x1 + reg[:, 0] * w
        y1 = y1 + reg[:, 1] * h
        x2 = x2 + reg[:, 2] * w
        y2 = y2 + reg[:, 3] * h

        # CRITICAL: Ensure x2 >= x1 and y2 >= y1 (some regression can cause negative widths/heights)
        # Swap coordinates if needed
        x1_new = np.minimum(x1, x2)
        x2_new = np.maximum(x1, x2)
        y1_new = np.minimum(y1, y2)
        y2_new = np.maximum(y1, y2)

        # Combine to bboxes
        bboxes = np.column_stack([x1_new, y1_new, x2_new, y2_new, score])

        return bboxes

    def _rectify(self, total_boxes):
        """
        Convert bounding boxes to squares by expanding to max(width, height).
        Matches C++ FaceDetectorMTCNN.cpp rectify() function (lines 698-718).

        Args:
            total_boxes: (N, 5) - [x1, y1, x2, y2, score]

        Returns:
            total_boxes: (N, 5) - [x1, y1, x2, y2, score] with square bboxes
        """
        # C++ rectify():
        # float max_side = std::max(width, height);
        # float new_min_x = total_bboxes[i].x + 0.5 * (width - max_side);
        # float new_min_y = total_bboxes[i].y + 0.5 * (height - max_side);
        # total_bboxes[i].width = max_side;
        # total_bboxes[i].height = max_side;

        widths = total_boxes[:, 2] - total_boxes[:, 0]
        heights = total_boxes[:, 3] - total_boxes[:, 1]
        max_side = np.maximum(widths, heights)

        # Center the bbox when expanding to square
        total_boxes[:, 0] = total_boxes[:, 0] + 0.5 * (widths - max_side)
        total_boxes[:, 1] = total_boxes[:, 1] + 0.5 * (heights - max_side)
        total_boxes[:, 2] = total_boxes[:, 0] + max_side
        total_boxes[:, 3] = total_boxes[:, 1] + max_side

        return total_boxes

    def _nms(self, boxes, threshold, method='Union'):
        """
        Non-maximum suppression.

        Args:
            boxes: (N, 5) - [x1, y1, x2, y2, score]
            threshold: IoU threshold
            method: 'Union' or 'Min'

        Returns:
            keep: Indices of boxes to keep
        """
        if boxes.shape[0] == 0:
            return np.empty((0,), dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            if method == 'Union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif method == 'Min':
                ovr = inter / np.minimum(areas[i], areas[order[1:]])

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    def _square_bbox(self, bboxes):
        """Convert bboxes to square (matching C++ rectify function)."""
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        l = np.maximum(w, h)

        # Calculate new positions
        new_x = bboxes[:, 0] + w * 0.5 - l * 0.5
        new_y = bboxes[:, 1] + h * 0.5 - l * 0.5

        # C++ casts to int here (FaceDetectorMTCNN.cpp:607-610)
        bboxes[:, 0] = np.floor(new_x).astype(np.int32)
        bboxes[:, 1] = np.floor(new_y).astype(np.int32)
        bboxes[:, 2] = bboxes[:, 0] + np.floor(l).astype(np.int32)
        bboxes[:, 3] = bboxes[:, 1] + np.floor(l).astype(np.int32)

        return bboxes

    def _pad_bbox(self, bboxes, img_h, img_w):
        """Pad bboxes to fit in image."""
        x1 = bboxes[:, 0].copy()
        y1 = bboxes[:, 1].copy()
        x2 = bboxes[:, 2].copy()
        y2 = bboxes[:, 3].copy()

        # Calculate padding
        dx1 = np.maximum(0, -x1).astype(np.int32)
        dy1 = np.maximum(0, -y1).astype(np.int32)
        dx2 = np.maximum(0, x2 - img_w).astype(np.int32)
        dy2 = np.maximum(0, y2 - img_h).astype(np.int32)

        # Clip to image bounds
        x1 = np.maximum(0, x1).astype(np.int32)
        y1 = np.maximum(0, y1).astype(np.int32)
        x2 = np.minimum(img_w, x2).astype(np.int32)
        y2 = np.minimum(img_h, y2).astype(np.int32)

        return x1, y1, x2, y2, dx1, dy1, dx2, dy2

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces using MTCNN.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            bboxes: (N, 4) - [x, y, width, height]
            landmarks: (N, 5, 2) - 5-point landmarks (eyes, nose, mouth corners)
        """
        img_h, img_w = img.shape[:2]

        # C++ converts to float32 early (FaceDetectorMTCNN.cpp:673)
        # This affects resizing behavior in RNet/ONet stages
        img_float = img.astype(np.float32)

        # Build image pyramid
        min_size = self.min_face_size
        m = 12.0 / min_size
        min_l = min(img_h, img_w) * m

        scales = []
        scale = m
        while min_l >= 12:
            scales.append(scale)
            scale *= self.factor
            min_l *= self.factor

        # ===== Stage 1: PNet =====
        total_boxes = []
        debug_pnet = True  # Enable for RNet debugging

        for i, scale in enumerate(scales):
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))

            # Resize image (from float32 like C++ does at FaceDetectorMTCNN.cpp:694)
            img_scaled = cv2.resize(img_float, (ws, hs))
            img_data = self._preprocess(img_scaled)

            # Debug: Log PNet input for first scale
            if debug_pnet and i == 0:
                with open('/tmp/python_pnet_debug.txt', 'w') as f:
                    f.write(f"Python PNet Debug - Scale 0\n")
                    f.write(f"{'='*37}\n")
                    f.write(f"Scaled image size: {ws}x{hs}\n")
                    f.write(f"Preprocessed shape: {img_data.shape}\n")
                    f.write(f"Sample pixels before normalization (from img_scaled):\n")
                    f.write(f"  [0,0]: B={img_scaled[0, 0, 0]} G={img_scaled[0, 0, 1]} R={img_scaled[0, 0, 2]}\n")
                    f.write(f"  [0,1]: B={img_scaled[0, 1, 0]} G={img_scaled[0, 1, 1]} R={img_scaled[0, 1, 2]}\n")
                    f.write(f"After normalization:\n")
                    f.write(f"  [0,0]: B={img_data[0, 0, 0, 0]:.6f} G={img_data[0, 1, 0, 0]:.6f} R={img_data[0, 2, 0, 0]:.6f}\n")
                    f.write(f"  [0,1]: B={img_data[0, 0, 0, 1]:.6f} G={img_data[0, 1, 0, 1]:.6f} R={img_data[0, 2, 0, 1]:.6f}\n")

            # Run PNet
            output = self.pnet.run(None, {'input': img_data})[0]

            # PNet output: (1, 6, H, W)
            # [:, 0:2] = classification (logit[0]=not_face, logit[1]=face)
            # [:, 2:6] = bbox regression (dx1, dy1, dx2, dy2)
            output = output[0].transpose(1, 2, 0)  # (H, W, 6)

            # C++ calculation (FaceDetectorMTCNN.cpp:706-708):
            # prob = 1.0 / (1.0 + exp(logit[0] - logit[1]))
            logit_not_face = output[:, :, 0]
            logit_face = output[:, :, 1]
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            # Create score_map in softmax-like format for compatibility
            score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
            reg_map = output[:, :, 2:6]

            if debug_pnet:
                print(f"  [DEBUG] Scale {i}: {scale:.3f}, Image size: {ws}x{hs}")
                print(f"          Output shape: {output.shape}")
                print(f"          Score range: [{score_map[:,:,1].min():.4f}, {score_map[:,:,1].max():.4f}]")
                print(f"          Scores > threshold: {(score_map[:,:,1] > self.thresholds[0]).sum()}")

            # Generate bboxes
            boxes = self._generate_bboxes(score_map, reg_map, scale, self.thresholds[0])

            if debug_pnet:
                print(f"          Boxes before NMS: {boxes.shape[0]}")

            if boxes.shape[0] > 0:
                # NMS within scale
                keep = self._nms(boxes, 0.5, 'Union')
                boxes = boxes[keep]
                if debug_pnet:
                    print(f"          Boxes after NMS: {boxes.shape[0]}")
                total_boxes.append(boxes)

        if len(total_boxes) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.vstack(total_boxes)

        if debug_pnet:
            print(f"\n[DEBUG] After PNet Stage 1:")
            print(f"  Total boxes from all scales: {total_boxes.shape[0]}")

            # Show top 10 scoring boxes BEFORE NMS to see if face box exists
            sorted_indices = np.argsort(total_boxes[:, 4])[::-1][:10]
            print(f"\n[BBOX TRACE] Top 10 scoring PNet boxes BEFORE cross-scale NMS:")
            for idx, i in enumerate(sorted_indices):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  #{idx+1}: x1={total_boxes[i,0]:.0f}, y1={total_boxes[i,1]:.0f}, "
                      f"w={w_i:.0f}, h={h_i:.0f}, score={total_boxes[i,4]:.6f}")

            # Show boxes in face region (y=300-900, w>200) where C++ found the face
            face_region = (total_boxes[:, 1] >= 300) & (total_boxes[:, 1] <= 900) & ((total_boxes[:, 2] - total_boxes[:, 0]) > 200)
            face_boxes = total_boxes[face_region]
            if face_boxes.shape[0] > 0:
                sorted_face = face_boxes[np.argsort(face_boxes[:, 4])[::-1][:10]]
                print(f"\n[BBOX TRACE] Top boxes in FACE REGION (y=300-900, w>200):")
                for idx, box in enumerate(sorted_face):
                    w_i = box[2] - box[0]
                    h_i = box[3] - box[1]
                    print(f"  #{idx+1}: x1={box[0]:.0f}, y1={box[1]:.0f}, "
                          f"w={w_i:.0f}, h={h_i:.0f}, score={box[4]:.6f}")

        # NMS across scales
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]

        if debug_pnet:
            print(f"  After cross-scale NMS (threshold=0.7): {total_boxes.shape[0]}")
            print(f"\n[BBOX TRACE] First 5 boxes after PNet NMS (BEFORE square_bbox):")
            for i in range(min(5, total_boxes.shape[0])):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  Box {i}: x1={total_boxes[i,0]:.0f}, y1={total_boxes[i,1]:.0f}, "
                      f"x2={total_boxes[i,2]:.0f}, y2={total_boxes[i,3]:.0f}, "
                      f"w={w_i:.0f}, h={h_i:.0f}, score={total_boxes[i,4]:.6f}")

        if total_boxes.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # ===== Stage 2: RNet =====
        total_boxes = self._square_bbox(total_boxes)

        if debug_pnet:
            print(f"\n[BBOX TRACE] First 5 boxes AFTER square_bbox (entering RNet loop):")
            for i in range(min(5, total_boxes.shape[0])):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  Box {i}: x1={total_boxes[i,0]:.0f}, y1={total_boxes[i,1]:.0f}, "
                      f"x2={total_boxes[i,2]:.0f}, y2={total_boxes[i,3]:.0f}, "
                      f"w={w_i:.0f}, h={h_i:.0f}, square={abs(w_i-h_i)<1.0}")

        x1, y1, x2, y2, dx1, dy1, dx2, dy2 = self._pad_bbox(total_boxes, img_h, img_w)

        rnet_input = []
        valid_indices = []
        for i in range(total_boxes.shape[0]):
            # C++ FaceDetectorMTCNN.cpp:752-774
            # Calculate target dimensions (width/height + 1 to match C++ inclusive coordinates)
            width_target = int(total_boxes[i, 2] - total_boxes[i, 0] + 1)
            height_target = int(total_boxes[i, 3] - total_boxes[i, 1] + 1)

            # Calculate bounds in input image (x-1 to match C++ adjustment)
            start_x_in = max(int(total_boxes[i, 0] - 1), 0)
            start_y_in = max(int(total_boxes[i, 1] - 1), 0)
            end_x_in = min(int(total_boxes[i, 0] + width_target - 1), img_w)
            end_y_in = min(int(total_boxes[i, 1] + height_target - 1), img_h)

            # Calculate bounds in output image
            start_x_out = max(int(-total_boxes[i, 0] + 1), 0)
            start_y_out = max(int(-total_boxes[i, 1] + 1), 0)
            end_x_out = min(int(width_target - (total_boxes[i, 0] + (total_boxes[i, 2] - total_boxes[i, 0]) - img_w)), width_target)
            end_y_out = min(int(height_target - (total_boxes[i, 1] + (total_boxes[i, 3] - total_boxes[i, 1]) - img_h)), height_target)

            # Skip if invalid
            if end_x_in <= start_x_in or end_y_in <= start_y_in:
                continue

            # Create padded face region (extract from float32 image like C++)
            face = np.zeros((height_target, width_target, 3), dtype=np.float32)
            face[start_y_out:end_y_out, start_x_out:end_x_out] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            # Resize to 24x24
            face = cv2.resize(face, (24, 24))

            # DEBUG: Log first few RNet inputs
            if debug_pnet and i < 3:
                with open('/tmp/python_rnet_debug.txt', 'a' if i > 0 else 'w') as f:
                    f.write(f"Python RNet Debug - Detection {i}\n")
                    f.write(f"={'='*37}\n")
                    f.write(f"Input bbox: x={total_boxes[i, 0]:.0f} y={total_boxes[i, 1]:.0f} ")
                    f.write(f"w={total_boxes[i, 2]-total_boxes[i, 0]:.0f} h={total_boxes[i, 3]-total_boxes[i, 1]:.0f}\n")
                    # Preprocess for logging
                    face_prep = self._preprocess(face)
                    # Sample pixels from preprocessed (already in CHW format)
                    f.write(f"After resize (24x24), sample preprocessed pixels:\n")
                    for r in range(min(2, 24)):
                        for c in range(min(2, 24)):
                            # face_prep is (1, 3, 24, 24), extract BGR
                            b_val = face_prep[0, 0, r, c]
                            g_val = face_prep[0, 1, r, c]
                            r_val = face_prep[0, 2, r, c]
                            f.write(f"  [{r},{c}]: B={b_val:.6f} G={g_val:.6f} R={r_val:.6f}\n")

            rnet_input.append(self._preprocess(face))
            valid_indices.append(i)

        if len(rnet_input) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        rnet_input = np.vstack(rnet_input)

        # Run RNet
        output = self.rnet.run(None, {'input': rnet_input})[0]

        # RNet output: (N, 6)
        # [:, 0:2] = classification (logit[0]=not_face, logit[1]=face)
        # [:, 2:6] = bbox regression
        # C++ calculation (FaceDetectorMTCNN.cpp:781):
        # prob = 1.0 / (1.0 + exp(logit[0] - logit[1]))
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        # DEBUG: Log first few RNet outputs
        if debug_pnet:
            with open('/tmp/python_rnet_debug.txt', 'a') as f:
                for i in range(min(3, len(output))):
                    f.write(f"RNet output: logit[0]={output[i, 0]:.6f} logit[1]={output[i, 1]:.6f} prob={scores[i]:.6f}\n\n")

        # Filter by threshold (C++ FaceDetectorMTCNN.cpp:943-965)
        keep = scores > self.thresholds[1]

        if debug_pnet:
            print(f"\n[DEBUG] After RNet Stage 2:")
            print(f"  Boxes input to RNet: {len(rnet_input)}")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Threshold: {self.thresholds[1]}")
            print(f"  Boxes passing threshold: {keep.sum()}")

        if not keep.any():
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]

        if total_boxes.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # NMS BEFORE regression (C++ FaceDetectorMTCNN.cpp:968)
        # CRITICAL: C++ does NMS on pre-regression boxes!
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = reg[keep]

        if debug_pnet:
            print(f"\n[DEBUG] After RNet NMS:")
            print(f"  Boxes after NMS: {total_boxes.shape[0]}")

        if total_boxes.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply regression (C++ FaceDetectorMTCNN.cpp:971)
        w = total_boxes[:, 2] - total_boxes[:, 0]
        h = total_boxes[:, 3] - total_boxes[:, 1]
        total_boxes[:, 0] += reg[:, 0] * w
        total_boxes[:, 1] += reg[:, 1] * h
        total_boxes[:, 2] += reg[:, 2] * w
        total_boxes[:, 3] += reg[:, 3] * h
        total_boxes[:, 4] = scores

        # LOGGING: After RNet regression
        if debug_pnet and total_boxes.shape[0] > 0:
            print(f"\n[BBOX TRACE] After RNet regression (before rectify):")
            for i in range(min(5, total_boxes.shape[0])):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  Box {i}: x1={total_boxes[i,0]:.2f}, y1={total_boxes[i,1]:.2f}, "
                      f"x2={total_boxes[i,2]:.2f}, y2={total_boxes[i,3]:.2f}, "
                      f"w={w_i:.2f}, h={h_i:.2f}, square={abs(w_i-h_i)<1.0}")

        # Convert to squares (C++ FaceDetectorMTCNN.cpp:974)
        total_boxes = self._square_bbox(total_boxes)

        # LOGGING: After rectify
        if debug_pnet and total_boxes.shape[0] > 0:
            print(f"\n[BBOX TRACE] After RNet rectify (_square_bbox):")
            for i in range(min(5, total_boxes.shape[0])):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  Box {i}: x1={total_boxes[i,0]:.2f}, y1={total_boxes[i,1]:.2f}, "
                      f"x2={total_boxes[i,2]:.2f}, y2={total_boxes[i,3]:.2f}, "
                      f"w={w_i:.2f}, h={h_i:.2f}, square={abs(w_i-h_i)<1.0}")

        if total_boxes.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # ===== Stage 3: ONet =====
        total_boxes = self._square_bbox(total_boxes)
        x1, y1, x2, y2, dx1, dy1, dx2, dy2 = self._pad_bbox(total_boxes, img_h, img_w)

        onet_input = []
        valid_indices = []
        for i in range(total_boxes.shape[0]):
            # C++ FaceDetectorMTCNN.cpp:825-847 (same pattern as RNet but for 48x48)
            # Calculate target dimensions (width/height + 1 to match C++ inclusive coordinates)
            width_target = int(total_boxes[i, 2] - total_boxes[i, 0] + 1)
            height_target = int(total_boxes[i, 3] - total_boxes[i, 1] + 1)

            if debug_pnet and i == 0:
                print(f"\n[DEBUG] ONet Face Extraction for bbox 0:")
                print(f"  Input bbox: x1={total_boxes[i, 0]:.1f}, y1={total_boxes[i, 1]:.1f}, x2={total_boxes[i, 2]:.1f}, y2={total_boxes[i, 3]:.1f}")
                print(f"  Target size: {width_target}x{height_target}")

            # Calculate bounds in input image (x-1 to match C++ adjustment)
            start_x_in = max(int(total_boxes[i, 0] - 1), 0)
            start_y_in = max(int(total_boxes[i, 1] - 1), 0)
            end_x_in = min(int(total_boxes[i, 0] + width_target - 1), img_w)
            end_y_in = min(int(total_boxes[i, 1] + height_target - 1), img_h)

            # Calculate bounds in output image
            start_x_out = max(int(-total_boxes[i, 0] + 1), 0)
            start_y_out = max(int(-total_boxes[i, 1] + 1), 0)
            end_x_out = min(int(width_target - (total_boxes[i, 0] + (total_boxes[i, 2] - total_boxes[i, 0]) - img_w)), width_target)
            end_y_out = min(int(height_target - (total_boxes[i, 1] + (total_boxes[i, 3] - total_boxes[i, 1]) - img_h)), height_target)

            if debug_pnet and i == 0:
                print(f"  Input region: [{start_y_in}:{end_y_in}, {start_x_in}:{end_x_in}] (size: {end_y_in-start_y_in}x{end_x_in-start_x_in})")
                print(f"  Output region: [{start_y_out}:{end_y_out}, {start_x_out}:{end_x_out}] (size: {end_y_out-start_y_out}x{end_x_out-start_x_out})")

            # Skip if invalid
            if end_x_in <= start_x_in or end_y_in <= start_y_in:
                continue

            # Create padded face region (extract from float32 image like C++)
            face = np.zeros((height_target, width_target, 3), dtype=np.float32)
            face[start_y_out:end_y_out, start_x_out:end_x_out] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            if debug_pnet and i == 0:
                # Save the pre-resize face for inspection
                face_debug = face.copy().astype(np.uint8)
                import cv2 as cv2_debug
                cv2_debug.imwrite('/tmp/onet_debug_face0_pre_resize.jpg', face_debug)
                print(f"  Saved pre-resize face to /tmp/onet_debug_face0_pre_resize.jpg")

            # Resize to 48x48
            face = cv2.resize(face, (48, 48))
            onet_input.append(self._preprocess(face))
            valid_indices.append(i)

        if len(onet_input) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        onet_input = np.vstack(onet_input)

        # Run ONet
        output = self.onet.run(None, {'input': onet_input})[0]

        # ONet output: (N, 16)
        # [:, 0:2] = classification (logit[0]=not_face, logit[1]=face)
        # [:, 2:6] = bbox regression
        # [:, 6:16] = 5 landmarks (x1,y1,x2,y2,...,x5,y5)
        # C++ calculation (FaceDetectorMTCNN.cpp:854):
        # prob = 1.0 / (1.0 + exp(logit[0] - logit[1]))
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        if debug_pnet:
            print(f"\n[DEBUG] ONet Raw Output:")
            for i in range(min(2, len(output))):
                print(f"  Face {i}: logit[0]={output[i, 0]:.4f}, logit[1]={output[i, 1]:.4f}, score={scores[i]:.4f}")
            # Save first face for inspection
            if len(onet_input) > 0:
                # Denormalize: reverse (x-127.5)*0.0078125
                face_denorm = (onet_input[0].transpose(1, 2, 0) / 0.0078125) + 127.5
                face_denorm = np.clip(face_denorm, 0, 255).astype(np.uint8)
                # cv2 is already imported at module level
                import cv2 as cv2_local
                cv2_local.imwrite('/tmp/onet_debug_face0.jpg', face_denorm)
                print(f"  Saved ONet input face 0 to /tmp/onet_debug_face0.jpg")

        # Filter by threshold
        keep = scores > self.thresholds[2]

        if debug_pnet:
            print(f"\n[DEBUG] After ONet Stage 3:")
            print(f"  Boxes input to ONet: {len(onet_input)}")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Threshold: {self.thresholds[2]}")
            print(f"  Boxes passing threshold: {keep.sum()}")

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]
        landmarks = output[keep, 6:16]

        if total_boxes.shape[0] == 0:
            if debug_pnet:
                print(f"  Final result: 0 detections (filtered by ONet)")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply bbox regression
        # CRITICAL: C++ FaceDetectorMTCNN.cpp:729-730 adds +1 to width/height for ONet (add1=true)
        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        total_boxes[:, 0] += reg[:, 0] * w
        total_boxes[:, 1] += reg[:, 1] * h
        total_boxes[:, 2] += reg[:, 2] * w
        total_boxes[:, 3] += reg[:, 3] * h
        total_boxes[:, 4] = scores

        # LOGGING: After ONet bbox regression
        if debug_pnet and total_boxes.shape[0] > 0:
            print(f"\n[BBOX TRACE] After ONet bbox regression (with +1):")
            for i in range(min(5, total_boxes.shape[0])):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  Box {i}: x1={total_boxes[i,0]:.2f}, y1={total_boxes[i,1]:.2f}, "
                      f"x2={total_boxes[i,2]:.2f}, y2={total_boxes[i,3]:.2f}, "
                      f"w={w_i:.2f}, h={h_i:.2f}, square={abs(w_i-h_i)<1.0}")

        # Denormalize landmarks (from [-1,1] to image coordinates)
        for i in range(5):
            landmarks[:, 2*i] = total_boxes[:, 0] + landmarks[:, 2*i] * w
            landmarks[:, 2*i+1] = total_boxes[:, 1] + landmarks[:, 2*i+1] * h

        landmarks = landmarks.reshape(-1, 5, 2)

        # Final NMS
        keep = self._nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[keep]
        landmarks = landmarks[keep]

        # LOGGING: After final NMS
        if debug_pnet and total_boxes.shape[0] > 0:
            print(f"\n[BBOX TRACE] After ONet final NMS:")
            for i in range(min(5, total_boxes.shape[0])):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  Box {i}: x1={total_boxes[i,0]:.2f}, y1={total_boxes[i,1]:.2f}, "
                      f"x2={total_boxes[i,2]:.2f}, y2={total_boxes[i,3]:.2f}, "
                      f"w={w_i:.2f}, h={h_i:.2f}, square={abs(w_i-h_i)<1.0}")

        # Apply final bbox correction to make it tight around facial landmarks
        # (C++ FaceDetectorMTCNN.cpp:1110-1113)
        # This is a learned correction specific to C++ OpenFace
        # C++: x = x + width * -0.0075
        #      y = y + height * 0.2459
        #      width = width * 1.0323
        #      height = height * 0.7751
        for k in range(total_boxes.shape[0]):
            w = total_boxes[k, 2] - total_boxes[k, 0]
            h = total_boxes[k, 3] - total_boxes[k, 1]
            # Calculate new values BEFORE modifying total_boxes
            new_x1 = total_boxes[k, 0] + w * -0.0075
            new_y1 = total_boxes[k, 1] + h * 0.2459
            new_width = w * 1.0323
            new_height = h * 0.7751
            # Apply all corrections atomically
            total_boxes[k, 0] = new_x1
            total_boxes[k, 1] = new_y1
            total_boxes[k, 2] = new_x1 + new_width
            total_boxes[k, 3] = new_y1 + new_height

        # LOGGING: After final bbox correction
        if debug_pnet and total_boxes.shape[0] > 0:
            print(f"\n[BBOX TRACE] After final bbox correction (1.0323 * w, 0.7751 * h):")
            for i in range(min(5, total_boxes.shape[0])):
                w_i = total_boxes[i, 2] - total_boxes[i, 0]
                h_i = total_boxes[i, 3] - total_boxes[i, 1]
                print(f"  Box {i}: x1={total_boxes[i,0]:.2f}, y1={total_boxes[i,1]:.2f}, "
                      f"x2={total_boxes[i,2]:.2f}, y2={total_boxes[i,3]:.2f}, "
                      f"w={w_i:.2f}, h={h_i:.2f}, square={abs(w_i-h_i)<1.0}")

        # Convert to (x, y, width, height) format
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        return bboxes, landmarks


def test_detector():
    """Quick test of C++ MTCNN detector."""
    import sys

    # Test on a calibration frame
    test_image = Path("calibration_frames/patient1_frame1.jpg")

    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        sys.exit(1)

    print("Testing C++ MTCNN Detector")
    print("="*80)

    # Load image
    img = cv2.imread(str(test_image))
    print(f"Image: {test_image.name} ({img.shape[1]}x{img.shape[0]})")

    # Detect
    detector = CPPMTCNNDetector()
    bboxes, landmarks = detector.detect(img)

    print(f"\nDetected {len(bboxes)} face(s)")

    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        print(f"  Face {i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
        print(f"           scale={np.sqrt(w*h):.1f}")

    # Visualize
    img_vis = img.copy()
    for bbox, lms in zip(bboxes, landmarks):
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for lm in lms:
            cv2.circle(img_vis, tuple(lm.astype(int)), 2, (0, 0, 255), -1)

    cv2.imwrite('cpp_mtcnn_test.jpg', img_vis)
    print(f"\n✓ Saved visualization: cpp_mtcnn_test.jpg")


if __name__ == "__main__":
    test_detector()
