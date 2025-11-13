#!/usr/bin/env python3
"""
Debug Pure Python CNN MTCNN V2 to see where detection fails.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

# Patch to add debug output
class DebugPurePythonMTCNN_V2(PurePythonMTCNN_V2):
    def detect(self, img: np.ndarray):
        """Debug version with verbose output"""
        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        print(f"\n{'='*80}")
        print(f"DEBUG: Pure Python CNN MTCNN V2")
        print(f"{'='*80}")
        print(f"Image: {img_w}x{img_h}")

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

        print(f"Min face size: {self.min_face_size}")
        print(f"Pyramid scales: {len(scales)}")
        print(f"Thresholds: {self.thresholds}")

        # ===== Stage 1: PNet =====
        print(f"\n{'='*80}")
        print(f"STAGE 1: PNet")
        print(f"{'='*80}")

        total_boxes = []

        for i, scale in enumerate(scales[:3]):  # Just first 3 scales for debugging
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))

            print(f"\nScale {i}: {scale:.4f} -> {ws}x{hs}")

            # Resize image
            img_scaled = cv2.resize(img_float, (ws, hs))
            img_data = self._preprocess(img_scaled)

            print(f"  Input shape: {img_data.shape}")
            print(f"  Input range: [{img_data.min():.3f}, {img_data.max():.3f}]")

            # Run PNet
            output = self._run_pnet(img_data)

            print(f"  PNet output shape: {output.shape}")
            print(f"  PNet output range: [{output.min():.3f}, {output.max():.3f}]")

            # Process output
            output = output[0].transpose(1, 2, 0)  # (H, W, 6)

            logit_not_face = output[:, :, 0]
            logit_face = output[:, :, 1]
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            print(f"  Prob face range: [{prob_face.min():.4f}, {prob_face.max():.4f}]")
            print(f"  Scores > {self.thresholds[0]}: {(prob_face > self.thresholds[0]).sum()}")

            score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
            reg_map = output[:, :, 2:6]

            # Generate bboxes
            boxes = self._generate_bboxes(score_map, reg_map, scale, self.thresholds[0])

            print(f"  Boxes before NMS: {boxes.shape[0]}")

            if boxes.shape[0] > 0:
                # Show first few boxes
                for j in range(min(5, boxes.shape[0])):
                    x1, y1, x2, y2, score = boxes[j]
                    print(f"    Box {j}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score={score:.3f}")

                keep = self._nms(boxes, 0.5, 'Union')
                boxes = boxes[keep]
                print(f"  Boxes after NMS: {boxes.shape[0]}")
                total_boxes.append(boxes)

        if len(total_boxes) == 0:
            print("\n❌ No boxes from PNet")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        print(f"\nTotal boxes from PNet: {sum(len(b) for b in total_boxes)}")

        # Call parent for rest
        return super().detect(img)


# Load image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')

# Create debug detector
detector = DebugPurePythonMTCNN_V2()

# Run detection
bboxes, landmarks = detector.detect(img)

print(f"\n{'='*80}")
print(f"FINAL RESULT")
print(f"{'='*80}")
print(f"Detected {len(bboxes)} faces")
