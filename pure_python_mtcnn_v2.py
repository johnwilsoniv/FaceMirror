#!/usr/bin/env python3
"""
Pure Python CNN MTCNN - Version 2
Standalone implementation using Pure Python CNN.
"""

import numpy as np
import cv2
from cpp_cnn_loader import CPPCNN
import os


class PurePythonMTCNN_V2:
    """
    Pure Python CNN MTCNN - standalone implementation.
    """

    def __init__(self, model_dir=None):
        """
        Initialize using Pure Python CNN instead of ONNX.

        Args:
            model_dir: Directory containing PNet.dat, RNet.dat, ONet.dat
        """
        # Don't call parent __init__ (we don't need ONNX models)
        if model_dir is None:
            model_dir = os.path.expanduser(
                "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
                "face_detection/mtcnn/convert_to_cpp/"
            )

        print(f"Loading Pure Python CNN MTCNN V2...")
        print(f"Model directory: {model_dir}")

        # Load Pure Python CNNs instead of ONNX
        self.pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))
        self.rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))
        self.onet = CPPCNN(os.path.join(model_dir, "ONet.dat"))

        # TEMPORARY: Lowered ONet threshold while debugging
        # Official: [0.6, 0.7, 0.7]
        self.thresholds = [0.6, 0.7, 0.5]  # PNet, RNet, ONet (ONet lowered for testing)
        self.min_face_size = 40
        self.factor = 0.709

        print(f"✓ Pure Python CNN MTCNN V2 loaded!")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess for Pure Python CNN (no batch dimension).

        Args:
            img: BGR image (H, W, 3)

        Returns:
            Normalized image (3, H, W) in RGB order (matching C++)
        """
        # Same normalization as ONNX version
        img_norm = (img.astype(np.float32) - 127.5) * 0.0078125

        # Transpose to (C, H, W) AND flip BGR to RGB (matching C++ line 151-155)
        # C++ does: input_maps.push_back(channels[2,1,0])
        # NO batch dimension for Pure Python CNN
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # Flip BGR to RGB (critical fix!)
        img_rgb = img_chw[[2, 1, 0], :, :]

        return img_rgb

    # Override the network inference to use Pure Python CNN
    # The rest of the detect() method is inherited from CPPMTCNNDetector

    def _run_pnet(self, img_data):
        """Run PNet using Pure Python CNN."""
        # Pure Python CNN returns a list of outputs
        outputs = self.pnet(img_data)
        output = outputs[-1]  # Extract final output: (6, H, W)

        # Add batch dimension to match ONNX format: (1, 6, H, W)
        output = output[np.newaxis, :, :, :]

        return output

    def _run_rnet(self, img_data):
        """Run RNet using Pure Python CNN."""
        outputs = self.rnet(img_data)
        output = outputs[-1]  # Extract final output: (6,)
        return output

    def _run_onet(self, img_data):
        """Run ONet using Pure Python CNN."""
        outputs = self.onet(img_data)
        output = outputs[-1]  # Extract final output: (16,)
        return output

    # Now patch the detect() method to use our _run_* methods
    # We need to override the part that calls self.pnet.run()

    def detect(self, img: np.ndarray, debug=False):
        """
        Detect faces using Pure Python CNN MTCNN.
        Uses the exact pipeline from CPPMTCNNDetector but with Pure Python CNN inference.

        Args:
            img: Input image (H, W, 3) in BGR format
            debug: If True, print detailed logging
        """
        # Import here to avoid circular dependency
        from typing import Tuple
        import cv2

        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        if debug:
            print("\n" + "=" * 80)
            print("PURE PYTHON MTCNN V2 DEBUG")
            print("=" * 80)
            print(f"Input image: {img.shape}")
            print(f"Thresholds: {self.thresholds}")

        # Build image pyramid (same as ONNX version)
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
        if debug:
            print(f"\n--- STAGE 1: PNet ---")
            print(f"Image pyramid: {len(scales)} scales")

        total_boxes = []

        for i, scale in enumerate(scales):
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))

            # Resize image
            img_scaled = cv2.resize(img_float, (ws, hs))
            img_data = self._preprocess(img_scaled)

            # Run PNet using Pure Python CNN
            output = self._run_pnet(img_data)

            # Rest is same as ONNX version
            output = output[0].transpose(1, 2, 0)  # (H, W, 6)

            logit_not_face = output[:, :, 0]
            logit_face = output[:, :, 1]
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
            reg_map = output[:, :, 2:6]

            # Generate bboxes
            boxes = self._generate_bboxes(score_map, reg_map, scale, self.thresholds[0])

            if boxes.shape[0] > 0:
                keep = self._nms(boxes, 0.5, 'Union')
                boxes = boxes[keep]
                total_boxes.append(boxes)
                if debug and boxes.shape[0] > 0:
                    print(f"  Scale {i+1}/{len(scales)}: {boxes.shape[0]} boxes after NMS")

        if len(total_boxes) == 0:
            if debug:
                print(f"✗ PNet: No boxes detected")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.vstack(total_boxes)
        if debug:
            print(f"\nPNet: {total_boxes.shape[0]} boxes before cross-scale NMS")

        # NMS across scales
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]

        if debug:
            print(f"PNet: {total_boxes.shape[0]} boxes after cross-scale NMS")

        if total_boxes.shape[0] == 0:
            if debug:
                print(f"✗ PNet: No boxes after NMS")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply PNet bbox regression (matching C++ line 1009)
        # This is the CRITICAL missing step!
        if debug:
            print(f"\nPNet: Before bbox regression:")
            for i in range(min(5, total_boxes.shape[0])):
                x1, y1, x2, y2 = total_boxes[i, 0:4]
                w = x2 - x1
                h = y2 - y1
                dx1, dy1, dx2, dy2 = total_boxes[i, 5:9]
                print(f"  Box {i+1}: ({x1:.1f}, {y1:.1f}) {w:.1f}×{h:.1f}, reg=[{dx1:.3f}, {dy1:.3f}, {dx2:.3f}, {dy2:.3f}]")

        total_boxes = self._apply_bbox_regression(total_boxes)

        if debug:
            print(f"\nPNet: After bbox regression:")
            for i in range(min(5, total_boxes.shape[0])):
                x1, y1, x2, y2 = total_boxes[i, 0:4]
                w = x2 - x1
                h = y2 - y1
                print(f"  Box {i+1}: ({x1:.1f}, {y1:.1f}) {w:.1f}×{h:.1f}")

        # ===== Stage 2: RNet =====
        if debug:
            print(f"\n--- STAGE 2: RNet ---")
            print(f"RNet input: {total_boxes.shape[0]} boxes from PNet")

        total_boxes = self._square_bbox(total_boxes)
        x1, y1, x2, y2, dx1, dy1, dx2, dy2 = self._pad_bbox(total_boxes, img_h, img_w)

        rnet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            # Simplified cropping (no zero-padding)
            x1 = int(max(0, total_boxes[i, 0]))
            y1 = int(max(0, total_boxes[i, 1]))
            x2 = int(min(img_w, total_boxes[i, 2]))
            y2 = int(min(img_h, total_boxes[i, 3]))

            if x2 <= x1 or y2 <= y1:
                continue

            face = img_float[y1:y2, x1:x2]
            face = cv2.resize(face, (24, 24))
            rnet_input.append(self._preprocess(face))
            valid_indices.append(i)

        if len(rnet_input) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        # Run RNet
        rnet_outputs = []
        for face_data in rnet_input:
            output = self._run_rnet(face_data)
            rnet_outputs.append(output)

        output = np.vstack(rnet_outputs)

        # Calculate scores
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        if debug:
            print(f"RNet: Tested {len(scores)} faces")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Scores > {self.thresholds[1]}: {(scores > self.thresholds[1]).sum()}")
            if len(scores) > 0:
                top_5_idx = np.argsort(scores)[-5:][::-1]
                print(f"  Top 5 scores: {scores[top_5_idx]}")

        # Filter by threshold
        keep = scores > self.thresholds[1]

        if not keep.any():
            if debug:
                print(f"✗ RNet: No faces passed threshold {self.thresholds[1]}")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]

        # NMS
        if debug:
            print(f"  Before RNet NMS: {total_boxes.shape[0]} boxes")

        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = reg[keep]

        if debug:
            print(f"  After RNet NMS: {total_boxes.shape[0]} boxes")

        if total_boxes.shape[0] == 0:
            if debug:
                print(f"✗ RNet: No boxes after NMS")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply regression
        w = total_boxes[:, 2] - total_boxes[:, 0]
        h = total_boxes[:, 3] - total_boxes[:, 1]
        total_boxes[:, 0] += reg[:, 0] * w
        total_boxes[:, 1] += reg[:, 1] * h
        total_boxes[:, 2] += reg[:, 2] * w
        total_boxes[:, 3] += reg[:, 3] * h
        total_boxes[:, 4] = scores

        # ===== Stage 3: ONet =====
        if debug:
            print(f"\n--- STAGE 3: ONet ---")
            print(f"ONet input: {total_boxes.shape[0]} boxes from RNet")

        total_boxes = self._square_bbox(total_boxes)
        x1, y1, x2, y2, dx1, dy1, dx2, dy2 = self._pad_bbox(total_boxes, img_h, img_w)

        onet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            # Simplified cropping (no zero-padding)
            x1 = int(max(0, total_boxes[i, 0]))
            y1 = int(max(0, total_boxes[i, 1]))
            x2 = int(min(img_w, total_boxes[i, 2]))
            y2 = int(min(img_h, total_boxes[i, 3]))

            if x2 <= x1 or y2 <= y1:
                continue

            face = img_float[y1:y2, x1:x2]
            face = cv2.resize(face, (48, 48))
            onet_input.append(self._preprocess(face))
            valid_indices.append(i)

        if len(onet_input) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        # Run ONet
        onet_outputs = []
        for face_data in onet_input:
            output = self._run_onet(face_data)
            onet_outputs.append(output)

        output = np.vstack(onet_outputs)

        # Calculate scores
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        if debug:
            print(f"ONet: Tested {len(scores)} faces")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Scores > {self.thresholds[2]}: {(scores > self.thresholds[2]).sum()}")
            if len(scores) > 0:
                print(f"  All scores: {scores}")

        # Filter by threshold
        keep = scores > self.thresholds[2]

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]
        landmarks = output[keep, 6:16]

        if debug:
            print(f"  After ONet threshold filter: {total_boxes.shape[0]} boxes")

        if total_boxes.shape[0] == 0:
            if debug:
                print(f"✗ ONet: No faces passed threshold {self.thresholds[2]}")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply bbox regression (with +1 for ONet)
        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        total_boxes[:, 0] += reg[:, 0] * w
        total_boxes[:, 1] += reg[:, 1] * h
        total_boxes[:, 2] += reg[:, 2] * w
        total_boxes[:, 3] += reg[:, 3] * h
        total_boxes[:, 4] = scores

        # Denormalize landmarks
        for i in range(5):
            landmarks[:, 2*i] = total_boxes[:, 0] + landmarks[:, 2*i] * w
            landmarks[:, 2*i+1] = total_boxes[:, 1] + landmarks[:, 2*i+1] * h

        landmarks = landmarks.reshape(-1, 5, 2)

        # Final NMS
        keep = self._nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[keep]
        landmarks = landmarks[keep]

        # Apply C++ bbox correction
        for k in range(total_boxes.shape[0]):
            w = total_boxes[k, 2] - total_boxes[k, 0]
            h = total_boxes[k, 3] - total_boxes[k, 1]
            new_x1 = total_boxes[k, 0] + w * -0.0075
            new_y1 = total_boxes[k, 1] + h * 0.2459
            new_width = w * 1.0323
            new_height = h * 0.7751
            total_boxes[k, 0] = new_x1
            total_boxes[k, 1] = new_y1
            total_boxes[k, 2] = new_x1 + new_width
            total_boxes[k, 3] = new_y1 + new_height

        # Convert to (x, y, width, height) format
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        return bboxes, landmarks

    # Helper methods

    def _generate_bboxes(self, score_map, reg_map, scale, threshold):
        """Generate bounding boxes from PNet output."""
        stride = 2
        cellsize = 12

        t_index = np.where(score_map[:, :, 1] > threshold)

        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg_map[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = score_map[t_index[0], t_index[1], 1]
        boundingbox = np.vstack([
            np.round((stride * t_index[1] + 1) / scale),
            np.round((stride * t_index[0] + 1) / scale),
            np.round((stride * t_index[1] + 1 + cellsize) / scale),
            np.round((stride * t_index[0] + 1 + cellsize) / scale),
            score,
            reg
        ])

        return boundingbox.T

    def _nms(self, boxes, threshold, method):
        """Non-Maximum Suppression."""
        if boxes.shape[0] == 0:
            return np.array([])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = []
        while sorted_s.shape[0] > 0:
            i = sorted_s[-1]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[sorted_s[:-1]])
            yy1 = np.maximum(y1[i], y1[sorted_s[:-1]])
            xx2 = np.minimum(x2[i], x2[sorted_s[:-1]])
            yy2 = np.minimum(y2[i], y2[sorted_s[:-1]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            if method == 'Min':
                o = inter / np.minimum(area[i], area[sorted_s[:-1]])
            else:
                o = inter / (area[i] + area[sorted_s[:-1]] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)[0]]

        return pick

    def _apply_bbox_regression(self, bboxes):
        """
        Apply bbox regression corrections to PNet boxes.

        Matches C++ apply_correction() function (FaceDetectorMTCNN.cpp:815-837).
        Called with add1=False for PNet (line 1009 in C++).

        Args:
            bboxes: Array with shape (N, 9) where:
                - Columns 0-3: x1, y1, x2, y2 (current bbox coordinates)
                - Column 4: score
                - Columns 5-8: dx1, dy1, dx2, dy2 (regression offsets)

        Returns:
            Array with shape (N, 9) with updated bbox coordinates
        """
        result = bboxes.copy()

        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, 0:4]
            dx1, dy1, dx2, dy2 = bboxes[i, 5:9]

            # Current box dimensions
            w = x2 - x1
            h = y2 - y1

            # Apply regression (matching C++ lines 828-832)
            # new_min_x = curr_box.x + corrections[i].x * curr_box.width
            # new_min_y = curr_box.y + corrections[i].y * curr_box.height
            # new_max_x = curr_box.x + curr_box.width + curr_box.width * corrections[i].width
            # new_max_y = curr_box.y + curr_box.height + curr_box.height * corrections[i].height

            new_x1 = x1 + dx1 * w
            new_y1 = y1 + dy1 * h
            new_x2 = x2 + dx2 * w
            new_y2 = y2 + dy2 * h

            result[i, 0] = new_x1
            result[i, 1] = new_y1
            result[i, 2] = new_x2
            result[i, 3] = new_y2

        return result

    def _square_bbox(self, bboxes):
        """Convert bounding boxes to squares."""
        square_bboxes = bboxes.copy()
        h = bboxes[:, 3] - bboxes[:, 1]
        w = bboxes[:, 2] - bboxes[:, 0]
        max_side = np.maximum(h, w)
        square_bboxes[:, 0] = bboxes[:, 0] + w * 0.5 - max_side * 0.5
        square_bboxes[:, 1] = bboxes[:, 1] + h * 0.5 - max_side * 0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side
        return square_bboxes

    def _pad_bbox(self, bboxes, img_h, img_w):
        """Calculate padding for bboxes (not used in simplified version but kept for compatibility)."""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        dx1 = np.zeros_like(x1)
        dy1 = np.zeros_like(y1)
        dx2 = np.zeros_like(x2)
        dy2 = np.zeros_like(y2)

        return x1, y1, x2, y2, dx1, dy1, dx2, dy2
