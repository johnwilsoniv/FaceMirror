"""
Check NMS Face Merging - Why Doesn't ONNX Merge 2 Overlapping Boxes?

Both CoreML and ONNX use the same Base class NMS logic, so if ONNX
detects 2 faces, they must not overlap enough (IoU < 0.7 for final NMS).
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
from pymtcnn.backends.onnx_backend import ONNXMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]"""
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

print("="*80)
print("NMS Face Merging Analysis")
print("="*80)

# Load image
img = cv2.imread(TEST_IMAGE)

# Initialize detectors
print("\nInitializing detectors...")
coreml_detector = CoreMLMTCNN(verbose=False)
onnx_detector = ONNXMTCNN(verbose=False)

# Run detection
print("\nRunning detections...")
coreml_bboxes, coreml_landmarks, coreml_debug = coreml_detector.detect_with_debug(img)
onnx_bboxes, onnx_landmarks, onnx_debug = onnx_detector.detect_with_debug(img)

print("\n" + "="*80)
print("ONNX Final Detections")
print("="*80)

if len(onnx_bboxes) >= 2:
    print(f"\nONNX detected {len(onnx_bboxes)} faces:")
    for i, bbox in enumerate(onnx_bboxes):
        print(f"  Box {i}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")

    # Check IoU between ONNX's 2 boxes
    iou = compute_iou(onnx_bboxes[0], onnx_bboxes[1])
    print(f"\nIoU between ONNX box 0 and box 1: {iou:.4f}")

    # Final NMS uses 0.7 threshold with 'Min' mode
    # Min mode: IoU = intersection / min(area1, area2)
    box1_area = onnx_bboxes[0][2] * onnx_bboxes[0][3]
    box2_area = onnx_bboxes[1][2] * onnx_bboxes[1][3]

    x1_min, y1_min = onnx_bboxes[0][0], onnx_bboxes[0][1]
    x1_max, y1_max = onnx_bboxes[0][0] + onnx_bboxes[0][2], onnx_bboxes[0][1] + onnx_bboxes[0][3]

    x2_min, y2_min = onnx_bboxes[1][0], onnx_bboxes[1][1]
    x2_max, y2_max = onnx_bboxes[1][0] + onnx_bboxes[1][2], onnx_bboxes[1][1] + onnx_bboxes[1][3]

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max > x_inter_min and y_inter_max > y_inter_min:
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        iou_min_mode = inter_area / min(box1_area, box2_area)

        print(f"IoU with 'Min' mode (used in final NMS): {iou_min_mode:.4f}")
        print(f"Final NMS threshold: 0.7")

        if iou_min_mode <= 0.7:
            print(f"\n✗ FOUND THE BUG: IoU ({iou_min_mode:.4f}) ≤ 0.7")
            print("  → NMS doesn't merge these boxes (threshold not exceeded)")
            print("  → But they SHOULD be merged - they're the same face!")
        else:
            print(f"\n⚠ IoU ({iou_min_mode:.4f}) > 0.7 - boxes SHOULD be merged by NMS")
            print("  → This indicates a bug in NMS logic")
    else:
        print("\n✗ Boxes don't overlap at all!")
        print("  → These are truly separate detections (not overlap issue)")

print("\n" + "="*80)
print("CoreML Final Detections")
print("="*80)

if len(coreml_bboxes) >= 1:
    print(f"\nCoreML detected {len(coreml_bboxes)} face:")
    for i, bbox in enumerate(coreml_bboxes):
        print(f"  Box {i}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("""
If IoU between ONNX's 2 boxes is < 0.7:
  → NMS threshold too low (should be higher to merge similar faces)
  → Or boxes are actually at different locations (model issue)

If IoU between ONNX's 2 boxes is >= 0.7:
  → NMS implementation bug (should have merged but didn't)
  → Check if scores are being considered correctly
""")

print("\n" + "="*80)
print("ONet Scores Before NMS")
print("="*80)

print("""
We need to check the ONet scores BEFORE final NMS to see if one box
has a much lower score and should be filtered. The issue might be that
both boxes have similar high scores, making NMS think they're both valid.
""")
