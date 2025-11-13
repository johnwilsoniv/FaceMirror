#!/usr/bin/env python3
"""
Debug bbox scaling issue in Pure Python CNN MTCNN.
Track bbox through entire pipeline to find where scaling breaks.
"""

import cv2
import numpy as np
from pure_python_mtcnn_detector import PurePythonMTCNNDetector

# Patch the detector to print intermediate results
class DebugMTCNNDetector(PurePythonMTCNNDetector):
    def detect(self, img: np.ndarray, debug=False):
        """Patched detect with detailed bbox tracking"""
        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        # Build image pyramid
        min_size = self.min_face_size
        m = 12.0 / min_size
        min_l = min(img_h, img_w) * m

        scales = []
        scale = m
        while min_l >= 12:
            scales.append(scale)
            scale *= self.scale_factor
            min_l *= self.scale_factor

        print(f"\n{'='*80}")
        print(f"BBOX SCALING DEBUG")
        print(f"{'='*80}")
        print(f"Image: {img_w}x{img_h}")
        print(f"Min face size: {self.min_face_size}")
        print(f"Scales: {len(scales)} scales")
        print(f"Scale values: {scales[:5]}..." if len(scales) > 5 else f"Scale values: {scales}")

        # ===== Stage 1: PNet =====
        print(f"\n{'='*80}")
        print(f"STAGE 1: PNet")
        print(f"{'='*80}")

        total_boxes = []

        for i, scale in enumerate(scales):
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))

            # Resize image
            img_scaled = cv2.resize(img_float, (ws, hs))
            img_data = self._preprocess(img_scaled)

            # Run PNet
            output = self.pnet(img_data)
            output = output[-1]
            output = output.transpose(1, 2, 0)

            # Calculate probabilities
            logit_not_face = output[:, :, 0]
            logit_face = output[:, :, 1]
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
            reg_map = output[:, :, 2:6]

            # Generate bboxes
            boxes = self._generate_bboxes(score_map, reg_map, scale, self.pnet_threshold)

            if boxes.shape[0] > 0:
                if i < 3 or boxes.shape[0] < 10:  # Print first few scales or sparse detections
                    print(f"\n  Scale {i}: {scale:.4f} (resized to {ws}x{hs})")
                    print(f"    Detected {boxes.shape[0]} boxes before NMS")
                    if boxes.shape[0] > 0 and boxes.shape[0] <= 5:
                        for j, box in enumerate(boxes[:5]):
                            x1, y1, x2, y2, score = box
                            print(f"      Box {j}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score={score:.3f} size={(x2-x1):.1f}x{(y2-y1):.1f}")

                keep = self._nms(boxes, 0.5, 'Union')
                boxes = boxes[keep]

                if i < 3 or boxes.shape[0] < 10:
                    print(f"    After NMS: {boxes.shape[0]} boxes")

                total_boxes.append(boxes)

        if len(total_boxes) == 0:
            print("\n❌ No boxes from PNet")
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.vstack(total_boxes)
        print(f"\n  Total boxes from all scales: {total_boxes.shape[0]}")

        # NMS across scales
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        print(f"  After inter-scale NMS: {total_boxes.shape[0]} boxes")

        # Show top boxes
        print(f"\n  Top 5 boxes after PNet NMS:")
        for i, box in enumerate(total_boxes[:5]):
            x1, y1, x2, y2, score = box
            print(f"    Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score={score:.3f} size={(x2-x1):.1f}x{(y2-y1):.1f}")

        # Continue with RNet/ONet...
        # Call parent implementation for the rest
        return super().detect(img, debug=True)


# Load image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')

# Create debug detector
detector = DebugMTCNNDetector()
detector.pnet_threshold = 0.6
detector.rnet_threshold = 0.6
detector.onet_threshold = 0.6

# Run detection
bboxes, landmarks = detector.detect(img)

print(f"\n{'='*80}")
print(f"FINAL RESULT")
print(f"{'='*80}")
print(f"Detected {len(bboxes)} faces")
if len(bboxes) > 0:
    x, y, w, h = bboxes[0]
    print(f"  Position: x={x:.1f}, y={y:.1f}")
    print(f"  Size: w={w:.1f}, h={h:.1f}")

    # Compare to C++ gold standard
    cpp_bbox = np.array([331.6, 753.5, 367.9, 422.8])
    print(f"\nC++ Gold Standard:")
    print(f"  Position: x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}")
    print(f"  Size: w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")

    print(f"\nRatio to C++:")
    print(f"  Size: {w/cpp_bbox[2]:.3f}x width, {h/cpp_bbox[3]:.3f}x height")
