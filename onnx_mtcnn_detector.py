#!/usr/bin/env python3
"""
ONNX MTCNN Detector - Full Cascade Implementation

Implements the complete MTCNN face detection pipeline using ONNX models
with weights extracted from C++ OpenFace.
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path


class ONNXMTCNNDetector:
    """MTCNN face detector using ONNX runtime."""

    def __init__(self, model_dir='cpp_mtcnn_onnx'):
        """Initialize MTCNN with ONNX models."""
        self.model_dir = Path(model_dir)

        # Load ONNX models
        print("Loading ONNX MTCNN models...")
        self.pnet = ort.InferenceSession(str(self.model_dir / 'pnet.onnx'))
        self.rnet = ort.InferenceSession(str(self.model_dir / 'rnet.onnx'))
        self.onet = ort.InferenceSession(str(self.model_dir / 'onet.onnx'))
        print("✓ Models loaded")

        # Detection parameters (match C++ defaults)
        self.min_face_size = 40
        self.scale_factor = 0.709
        self.pnet_threshold = 0.6
        self.rnet_threshold = 0.7
        self.onet_threshold = 0.7
        self.nms_threshold = 0.7

    def preprocess_image(self, img):
        """Preprocess image: BGR -> RGB, normalize to [-1, 1]."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0

        return img_normalized

    def compute_scales(self, height, width):
        """Compute image pyramid scales."""
        min_dimension = min(height, width)
        min_net_size = 12

        # Calculate scale to make smallest face equal to min_net_size
        scale = min_net_size / self.min_face_size

        scales = []
        scaled_dim = min_dimension * scale

        while scaled_dim >= min_net_size:
            scales.append(scale)
            scale *= self.scale_factor
            scaled_dim = min_dimension * scale

        return scales

    def run_pnet(self, img):
        """Stage 1: Proposal Network (PNet)."""
        height, width = img.shape[:2]
        scales = self.compute_scales(height, width)

        all_boxes = []

        for scale in scales:
            # Resize image
            scaled_h = int(height * scale)
            scaled_w = int(width * scale)
            scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            # Prepare input: HWC -> CHW -> BCHW
            input_data = scaled_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            # Run PNet
            output = self.pnet.run(None, {'input': input_data})[0][0]

            # Output shape: (6, H, W) = [cls_not_face, cls_face, bbox_dx, bbox_dy, bbox_dw, bbox_dh]
            cls_map = output[1, :, :]  # Face probability
            bbox_map = output[2:6, :, :]  # Bbox regression

            # Find faces (threshold on probability)
            face_indices = np.where(cls_map > self.pnet_threshold)

            if len(face_indices[0]) == 0:
                continue

            # Convert to bounding boxes
            for y, x in zip(face_indices[0], face_indices[1]):
                score = cls_map[y, x]

                # Map back to original image coordinates
                # PNet uses 12x12 receptive field with stride 2
                bbox_x = int((x * 2) / scale)
                bbox_y = int((y * 2) / scale)
                bbox_w = int(12 / scale)
                bbox_h = int(12 / scale)

                # Apply bbox regression
                dx = bbox_map[0, y, x]
                dy = bbox_map[1, y, x]
                dw = bbox_map[2, y, x]
                dh = bbox_map[3, y, x]

                bbox_x = int(bbox_x + dx * bbox_w)
                bbox_y = int(bbox_y + dy * bbox_h)
                bbox_w = int(bbox_w * np.exp(dw))
                bbox_h = int(bbox_h * np.exp(dh))

                all_boxes.append([bbox_x, bbox_y, bbox_w, bbox_h, score])

        if len(all_boxes) == 0:
            return []

        # Non-maximum suppression
        boxes = np.array(all_boxes)
        keep = self.nms(boxes, self.nms_threshold)

        return boxes[keep]

    def run_rnet(self, img, boxes):
        """Stage 2: Refinement Network (RNet)."""
        if len(boxes) == 0:
            return []

        refined_boxes = []

        for box in boxes:
            x, y, w, h, _ = box

            # Extract and resize face patch to 24x24
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            face_patch = img[y1:y2, x1:x2]
            face_patch = cv2.resize(face_patch, (24, 24), interpolation=cv2.INTER_LINEAR)

            # Prepare input: HWC -> CHW -> BCHW
            input_data = face_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            # Run RNet
            output = self.rnet.run(None, {'input': input_data})[0][0]

            # Output: [cls_not_face, cls_face, bbox_dx, bbox_dy, bbox_dw, bbox_dh]
            score = output[1]

            if score > self.rnet_threshold:
                # Apply bbox regression
                dx, dy, dw, dh = output[2:6]

                bbox_x = int(x + dx * w)
                bbox_y = int(y + dy * h)
                bbox_w = int(w * np.exp(dw))
                bbox_h = int(h * np.exp(dh))

                refined_boxes.append([bbox_x, bbox_y, bbox_w, bbox_h, score])

        if len(refined_boxes) == 0:
            return []

        # Non-maximum suppression
        boxes = np.array(refined_boxes)
        keep = self.nms(boxes, self.nms_threshold)

        return boxes[keep]

    def run_onet(self, img, boxes):
        """Stage 3: Output Network (ONet)."""
        if len(boxes) == 0:
            return [], []

        final_boxes = []
        landmarks = []

        for box in boxes:
            x, y, w, h, _ = box

            # Extract and resize face patch to 48x48
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            face_patch = img[y1:y2, x1:x2]
            face_patch = cv2.resize(face_patch, (48, 48), interpolation=cv2.INTER_LINEAR)

            # Prepare input: HWC -> CHW -> BCHW
            input_data = face_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            # Run ONet
            output = self.onet.run(None, {'input': input_data})[0][0]

            # Output: [cls_not_face, cls_face, bbox_dx, bbox_dy, bbox_dw, bbox_dh,
            #          lm1_x, lm1_y, lm2_x, lm2_y, lm3_x, lm3_y, lm4_x, lm4_y, lm5_x, lm5_y]
            score = output[1]

            if score > self.onet_threshold:
                # Apply bbox regression
                dx, dy, dw, dh = output[2:6]

                bbox_x = int(x + dx * w)
                bbox_y = int(y + dy * h)
                bbox_w = int(w * np.exp(dw))
                bbox_h = int(h * np.exp(dh))

                final_boxes.append([bbox_x, bbox_y, bbox_w, bbox_h, score])

                # Extract landmarks (5 points)
                lm = output[6:16].reshape(5, 2)
                lm[:, 0] = x + lm[:, 0] * w  # x coordinates
                lm[:, 1] = y + lm[:, 1] * h  # y coordinates
                landmarks.append(lm)

        if len(final_boxes) == 0:
            return [], []

        # Final NMS
        boxes = np.array(final_boxes)
        keep = self.nms(boxes, self.nms_threshold)

        final_boxes = boxes[keep]
        landmarks = [landmarks[i] for i in keep]

        return final_boxes, landmarks

    def nms(self, boxes, threshold):
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img):
        """Run full MTCNN cascade."""
        # Preprocess
        img_normalized = self.preprocess_image(img)

        # Stage 1: PNet
        pnet_boxes = self.run_pnet(img_normalized)
        print(f"  PNet: {len(pnet_boxes)} proposals")

        # Stage 2: RNet
        rnet_boxes = self.run_rnet(img_normalized, pnet_boxes)
        print(f"  RNet: {len(rnet_boxes)} refined boxes")

        # Stage 3: ONet
        onet_boxes, landmarks = self.run_onet(img_normalized, rnet_boxes)
        print(f"  ONet: {len(onet_boxes)} final detections")

        return onet_boxes, landmarks


def main():
    """Test ONNX MTCNN detector."""
    print("="*80)
    print("ONNX MTCNN FULL CASCADE TEST")
    print("="*80)

    # Initialize detector
    detector = ONNXMTCNNDetector()

    # Load test image
    test_image = 'calibration_frames/patient1_frame1.jpg'
    print(f"\nLoading test image: {test_image}")
    img = cv2.imread(test_image)
    print(f"Image shape: {img.shape}")

    # Run detection
    print("\nRunning ONNX MTCNN cascade:")
    boxes, landmarks = detector.detect(img)

    # Display results
    print("\n" + "="*80)
    print("DETECTION RESULTS")
    print("="*80)

    if len(boxes) > 0:
        for i, (box, lm) in enumerate(zip(boxes, landmarks)):
            x, y, w, h, score = box
            print(f"\nFace {i+1}:")
            print(f"  Bounding box: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
            print(f"  Confidence: {score:.4f}")
            print(f"  Landmarks:")
            for j, pt in enumerate(lm):
                print(f"    Point {j+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    else:
        print("No faces detected")

    # Visualize
    print("\n" + "="*80)
    print("Saving visualization...")
    vis_img = img.copy()

    for i, (box, lm) in enumerate(zip(boxes, landmarks)):
        x, y, w, h, score = box

        # Draw bounding box
        cv2.rectangle(vis_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

        # Draw landmarks
        for pt in lm:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

        # Draw confidence
        cv2.putText(vis_img, f"{score:.3f}", (int(x), int(y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    output_path = 'onnx_mtcnn_detection.jpg'
    cv2.imwrite(output_path, vis_img)
    print(f"✓ Saved to: {output_path}")


if __name__ == '__main__':
    main()
