#!/usr/bin/env python3
"""
Debug RNet/ONet processing in V2.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

# Patch to add debug output
class DebugPurePythonMTCNN_V2(PurePythonMTCNN_V2):
    def _run_rnet(self, img_data):
        """Debug RNet."""
        outputs = self.rnet(img_data)
        output = outputs[-1]

        print(f"    RNet output: {output}")
        print(f"    RNet output shape: {output.shape}")

        # Calculate score
        score = 1.0 / (1.0 + np.exp(output[0] - output[1]))
        print(f"    RNet score: {score:.4f} (threshold={self.thresholds[1]})")

        return output

    def detect(self, img: np.ndarray):
        """Override to add RNet debugging"""
        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        # Build image pyramid (copy from parent)
        min_size = self.min_face_size
        m = 12.0 / min_size
        min_l = min(img_h, img_w) * m

        scales = []
        scale = m
        while min_l >= 12:
            scales.append(scale)
            scale *= self.factor
            min_l *= self.factor

        print(f"\n{'='*80}")
        print(f"RNet/ONet DEBUG")
        print(f"{'='*80}")

        # ===== Stage 1: PNet ===== (simplified, just get boxes)
        total_boxes = []

        for i, scale in enumerate(scales):
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))

            img_scaled = cv2.resize(img_float, (ws, hs))
            img_data = self._preprocess(img_scaled)

            output = self._run_pnet(img_data)
            output = output[0].transpose(1, 2, 0)

            logit_not_face = output[:, :, 0]
            logit_face = output[:, :, 1]
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
            reg_map = output[:, :, 2:6]

            boxes = self._generate_bboxes(score_map, reg_map, scale, self.thresholds[0])

            if boxes.shape[0] > 0:
                keep = self._nms(boxes, 0.5, 'Union')
                boxes = boxes[keep]
                total_boxes.append(boxes)

        if len(total_boxes) == 0:
            print("No boxes from PNet")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.vstack(total_boxes)
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]

        print(f"\nPNet: {total_boxes.shape[0]} boxes after NMS")

        # Show top 5 boxes
        print(f"Top 5 boxes:")
        for i in range(min(5, total_boxes.shape[0])):
            x1, y1, x2, y2, score = total_boxes[i]
            print(f"  Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score={score:.3f}")

        # ===== Stage 2: RNet =====
        print(f"\n{'='*80}")
        print(f"STAGE 2: RNet")
        print(f"{'='*80}")

        total_boxes = self._square_bbox(total_boxes)
        x1, y1, x2, y2, dx1, dy1, dx2, dy2 = self._pad_bbox(total_boxes, img_h, img_w)

        rnet_input = []
        valid_indices = []

        for i in range(min(5, total_boxes.shape[0])):  # Just first 5 for debugging
            width_target = int(total_boxes[i, 2] - total_boxes[i, 0] + 1)
            height_target = int(total_boxes[i, 3] - total_boxes[i, 1] + 1)

            start_x_in = max(int(total_boxes[i, 0] - 1), 0)
            start_y_in = max(int(total_boxes[i, 1] - 1), 0)
            end_x_in = min(int(total_boxes[i, 0] + width_target - 1), img_w)
            end_y_in = min(int(total_boxes[i, 1] + height_target - 1), img_h)

            start_x_out = max(int(-total_boxes[i, 0] + 1), 0)
            start_y_out = max(int(-total_boxes[i, 1] + 1), 0)
            end_x_out = min(int(width_target - (total_boxes[i, 0] + (total_boxes[i, 2] - total_boxes[i, 0]) - img_w)), width_target)
            end_y_out = min(int(height_target - (total_boxes[i, 1] + (total_boxes[i, 3] - total_boxes[i, 1]) - img_h)), height_target)

            if end_x_in <= start_x_in or end_y_in <= start_y_in:
                continue

            face = np.zeros((height_target, width_target, 3), dtype=np.float32)
            face[start_y_out:end_y_out, start_x_out:end_x_out] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            face = cv2.resize(face, (24, 24))

            print(f"\n  Box {i}:")
            print(f"    Input bbox: [{total_boxes[i, 0]:.1f}, {total_boxes[i, 1]:.1f}, {total_boxes[i, 2]:.1f}, {total_boxes[i, 3]:.1f}]")
            print(f"    Face crop: {face.shape}, range=[{face.min():.1f}, {face.max():.1f}]")

            face_data = self._preprocess(face)
            print(f"    Preprocessed: {face_data.shape}, range=[{face_data.min():.3f}, {face_data.max():.3f}]")

            rnet_input.append(face_data)
            valid_indices.append(i)

        if len(rnet_input) == 0:
            print("\nNo valid RNet inputs")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Run RNet
        print(f"\nRunning RNet on {len(rnet_input)} faces...")
        rnet_outputs = []
        for i, face_data in enumerate(rnet_input):
            print(f"\n  Face {i}:")
            output = self._run_rnet(face_data)
            rnet_outputs.append(output)

        output = np.vstack(rnet_outputs)

        # Calculate scores
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        print(f"\nRNet scores: {scores}")
        print(f"RNet threshold: {self.thresholds[1]}")
        print(f"Passing RNet: {(scores > self.thresholds[1]).sum()} / {len(scores)}")

        return np.empty((0, 4)), np.empty((0, 5, 2))  # Stop here for debugging


# Load image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')

# Create debug detector
detector = DebugPurePythonMTCNN_V2()

# Run detection
bboxes, landmarks = detector.detect(img)
