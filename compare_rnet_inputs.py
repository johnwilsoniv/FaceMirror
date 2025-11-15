"""
Compare RNet Inputs - Check if Cropping/Padding Produces Different Patches

CoreML and ONNX produce slightly different PNet boxes. Let's see if the complex
RNet cropping logic amplifies these differences into significantly different inputs.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
from pymtcnn.backends.onnx_backend import ONNXMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

class DebugCoreMLInputs(CoreMLMTCNN):
    """CoreML with RNet input capture"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnet_inputs = []
        self.rnet_bboxes = []

    def detect(self, img):
        """Override detect to capture RNet inputs"""
        self.rnet_inputs = []
        self.rnet_bboxes = []

        # Get PNet boxes first
        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        # Run PNet stage (simplified from base.py)
        scales = []
        m = 12.0 / self.min_face_size
        min_len = min(img_h, img_w)
        scale = m
        while scale * min_len >= 12:
            scales.append(scale)
            scale *= self.factor

        all_boxes = []
        for scale in scales:
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))
            img_scaled = cv2.resize(img_float, (ws, hs), interpolation=cv2.INTER_LINEAR)
            img_data = self._preprocess(img_scaled, flip_bgr_to_rgb=True)

            pnet_out = self._run_pnet(img_data)
            pnet_chw = pnet_out[0].transpose(1, 2, 0)

            # Extract face probabilities
            prob_face = 1.0 / (1.0 + np.exp(pnet_chw[:, :, 0] - pnet_chw[:, :, 1]))

            # Find boxes above threshold
            inds = np.where(prob_face > self.thresholds[0])
            if len(inds[0]) == 0:
                continue

            # Generate boxes (simplified)
            for y, x in zip(inds[0], inds[1]):
                score = prob_face[y, x]
                reg = pnet_chw[y, x, 2:6]

                bbox_x1 = (2 * x) / scale
                bbox_y1 = (2 * y) / scale
                bbox_x2 = (2 * x + 11) / scale
                bbox_y2 = (2 * y + 11) / scale

                # Apply regression
                w = bbox_x2 - bbox_x1
                h = bbox_y2 - bbox_y1
                bbox_x1 = bbox_x1 + reg[0] * w
                bbox_y1 = bbox_y1 + reg[1] * h
                bbox_x2 = bbox_x2 + reg[2] * w
                bbox_y2 = bbox_y2 + reg[3] * h

                all_boxes.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2, score])

        if len(all_boxes) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.array(all_boxes)

        # NMS
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]

        # Square boxes for RNet
        total_boxes_squared = self._square_bbox(total_boxes)

        # Crop RNet inputs (matching base.py logic EXACTLY)
        for i in range(total_boxes_squared.shape[0]):
            bbox = total_boxes_squared[i]
            self.rnet_bboxes.append(bbox[:4])

            bbox_x = bbox[0]
            bbox_y = bbox[1]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            # C++ RNet cropping with +1 padding
            width_target = int(bbox_w + 1)
            height_target = int(bbox_h + 1)

            start_x_in = max(int(bbox_x - 1), 0)
            start_y_in = max(int(bbox_y - 1), 0)
            end_x_in = min(int(bbox_x + width_target - 1), img_w)
            end_y_in = min(int(bbox_y + height_target - 1), img_h)

            start_x_out = max(int(-bbox_x + 1), 0)
            start_y_out = max(int(-bbox_y + 1), 0)
            end_x_out = min(int(width_target - (bbox_x + bbox_w - img_w)), width_target)
            end_y_out = min(int(height_target - (bbox_y + bbox_h - img_h)), height_target)

            tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)
            tmp[start_y_out:end_y_out, start_x_out:end_x_out] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            face = cv2.resize(tmp, (24, 24))
            preprocessed = self._preprocess(face, flip_bgr_to_rgb=True)
            self.rnet_inputs.append(preprocessed)

        # Continue with normal detection
        return super().detect(img)

class DebugONNXInputs(ONNXMTCNN):
    """ONNX with RNet input capture"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnet_inputs = []
        self.rnet_bboxes = []

    def detect(self, img):
        """Override detect to capture RNet inputs"""
        self.rnet_inputs = []
        self.rnet_bboxes = []

        # Same logic as CoreML but using ONNX models
        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        # Run PNet stage
        scales = []
        m = 12.0 / self.min_face_size
        min_len = min(img_h, img_w)
        scale = m
        while scale * min_len >= 12:
            scales.append(scale)
            scale *= self.factor

        all_boxes = []
        for scale in scales:
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))
            img_scaled = cv2.resize(img_float, (ws, hs), interpolation=cv2.INTER_LINEAR)
            img_data = self._preprocess(img_scaled, flip_bgr_to_rgb=True)

            pnet_out = self._run_pnet(img_data)
            pnet_chw = pnet_out[0].transpose(1, 2, 0)

            prob_face = 1.0 / (1.0 + np.exp(pnet_chw[:, :, 0] - pnet_chw[:, :, 1]))

            inds = np.where(prob_face > self.thresholds[0])
            if len(inds[0]) == 0:
                continue

            for y, x in zip(inds[0], inds[1]):
                score = prob_face[y, x]
                reg = pnet_chw[y, x, 2:6]

                bbox_x1 = (2 * x) / scale
                bbox_y1 = (2 * y) / scale
                bbox_x2 = (2 * x + 11) / scale
                bbox_y2 = (2 * y + 11) / scale

                w = bbox_x2 - bbox_x1
                h = bbox_y2 - bbox_y1
                bbox_x1 = bbox_x1 + reg[0] * w
                bbox_y1 = bbox_y1 + reg[1] * h
                bbox_x2 = bbox_x2 + reg[2] * w
                bbox_y2 = bbox_y2 + reg[3] * h

                all_boxes.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2, score])

        if len(all_boxes) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.array(all_boxes)
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        total_boxes_squared = self._square_bbox(total_boxes)

        for i in range(total_boxes_squared.shape[0]):
            bbox = total_boxes_squared[i]
            self.rnet_bboxes.append(bbox[:4])

            bbox_x = bbox[0]
            bbox_y = bbox[1]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            width_target = int(bbox_w + 1)
            height_target = int(bbox_h + 1)

            start_x_in = max(int(bbox_x - 1), 0)
            start_y_in = max(int(bbox_y - 1), 0)
            end_x_in = min(int(bbox_x + width_target - 1), img_w)
            end_y_in = min(int(bbox_y + height_target - 1), img_h)

            start_x_out = max(int(-bbox_x + 1), 0)
            start_y_out = max(int(-bbox_y + 1), 0)
            end_x_out = min(int(width_target - (bbox_x + bbox_w - img_w)), width_target)
            end_y_out = min(int(height_target - (bbox_y + bbox_h - img_h)), height_target)

            tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)
            tmp[start_y_out:end_y_out, start_x_out:end_x_out] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            face = cv2.resize(tmp, (24, 24))
            preprocessed = self._preprocess(face, flip_bgr_to_rgb=True)
            self.rnet_inputs.append(preprocessed)

        return super().detect(img)

print("="*80)
print("RNet Input Comparison - CoreML vs ONNX")
print("="*80)

# Load image
img = cv2.imread(TEST_IMAGE)

# Initialize debug detectors
print("\nInitializing detectors...")
coreml_detector = DebugCoreMLInputs(verbose=False)
onnx_detector = DebugONNXInputs(verbose=False)

# Run detection to capture inputs
print("Running CoreML...")
coreml_detector.detect(img)

print("Running ONNX...")
onnx_detector.detect(img)

print("\n" + "="*80)
print("RNet Input Statistics")
print("="*80)

print(f"\nCoreML: {len(coreml_detector.rnet_inputs)} RNet inputs captured")
print(f"ONNX:   {len(onnx_detector.rnet_inputs)} RNet inputs captured")

if len(coreml_detector.rnet_inputs) > 0 and len(onnx_detector.rnet_inputs) > 0:
    # Compare first few inputs
    num_compare = min(5, len(coreml_detector.rnet_inputs), len(onnx_detector.rnet_inputs))

    print(f"\n" + "="*80)
    print(f"Comparing First {num_compare} RNet Inputs")
    print("="*80)

    for i in range(num_compare):
        coreml_input = coreml_detector.rnet_inputs[i]
        onnx_input = onnx_detector.rnet_inputs[i]
        coreml_bbox = coreml_detector.rnet_bboxes[i]
        onnx_bbox = onnx_detector.rnet_bboxes[i]

        diff = np.abs(coreml_input - onnx_input)

        print(f"\nInput {i}:")
        print(f"  CoreML bbox: [{coreml_bbox[0]:.2f}, {coreml_bbox[1]:.2f}, {coreml_bbox[2]:.2f}, {coreml_bbox[3]:.2f}]")
        print(f"  ONNX bbox:   [{onnx_bbox[0]:.2f}, {onnx_bbox[1]:.2f}, {onnx_bbox[2]:.2f}, {onnx_bbox[3]:.2f}]")
        print(f"  Bbox diff:   [{abs(coreml_bbox[0]-onnx_bbox[0]):.2f}, {abs(coreml_bbox[1]-onnx_bbox[1]):.2f}, {abs(coreml_bbox[2]-onnx_bbox[2]):.2f}, {abs(coreml_bbox[3]-onnx_bbox[3]):.2f}]")
        print(f"  Input shape: {coreml_input.shape}")
        print(f"  Max pixel diff: {diff.max():.6f}")
        print(f"  Mean pixel diff: {diff.mean():.6f}")

        if diff.max() > 0.1:
            print(f"  ⚠ SIGNIFICANT DIFFERENCE - cropping produced different patches!")
        elif diff.max() > 0.01:
            print(f"  ⚠ MODERATE DIFFERENCE - small bbox changes amplified by cropping")
        else:
            print(f"  ✓ Similar inputs")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print(f"""
If RNet inputs are very different despite small bbox differences:
  → The complex cropping/padding logic amplifies small PNet box variations
  → Different crops lead to different RNet scores
  → This is why ONNX gets different filtering results (6 vs 9 boxes)

The cascade effect:
  1. PNet model differences: ~1% score variation
  2. Borderline boxes flip (0.59 vs 0.61 around 0.6 threshold)
  3. CoreML: 99 boxes, ONNX: 100 boxes (1 box difference)
  4. Different boxes → different cropping coordinates
  5. Complex padding amplifies small coordinate differences
  6. RNet sees different input patches
  7. RNet produces different scores
  8. Different filtering: CoreML=9, ONNX=6
  9. Different boxes reach ONet
  10. ONet produces 2 boxes for same face that don't overlap enough
  11. Final NMS doesn't merge them (IoU=0.36 < 0.7)
""")
