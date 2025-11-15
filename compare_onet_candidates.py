"""
Compare ONet Candidates - CoreML vs ONNX

Find out why ONNX passes 2 faces through ONet while CoreML passes 1
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN
from pymtcnn.backends.onnx_backend import ONNXMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("ONet Candidates Comparison - CoreML vs ONNX")
print("="*80)

# Load image
img = cv2.imread(TEST_IMAGE)

# Initialize detectors
coreml_detector = CoreMLMTCNN(verbose=False)
onnx_detector = ONNXMTCNN(verbose=False)

# Run with debug
print("\nRunning CoreML...")
coreml_bboxes, coreml_landmarks, coreml_debug = coreml_detector.detect_with_debug(img)

print("Running ONNX...")
onnx_bboxes, onnx_landmarks, onnx_debug = onnx_detector.detect_with_debug(img)

# Compare stage counts
print("\n" + "="*80)
print("Stage Counts")
print("="*80)
print(f"\nPNet:  CoreML={coreml_debug['pnet']['num_boxes']}, ONNX={onnx_debug['pnet']['num_boxes']}")
print(f"RNet:  CoreML={coreml_debug['rnet']['num_boxes']}, ONNX={onnx_debug['rnet']['num_boxes']}")
print(f"ONet:  CoreML={coreml_debug['onet']['num_boxes']}, ONNX={onnx_debug['onet']['num_boxes']}")
print(f"Final: CoreML={coreml_debug['final']['num_boxes']}, ONNX={onnx_debug['final']['num_boxes']}")

# Compare RNet outputs
print("\n" + "="*80)
print("RNet Output Boxes")
print("="*80)

print(f"\nCoreML RNet boxes ({coreml_debug['rnet']['num_boxes']}):")
for i, box in enumerate(coreml_debug['rnet']['boxes']):
    print(f"  Box {i}: x={box[0]:.1f}, y={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}")

print(f"\nONNX RNet boxes ({onnx_debug['rnet']['num_boxes']}):")
for i, box in enumerate(onnx_debug['rnet']['boxes']):
    print(f"  Box {i}: x={box[0]:.1f}, y={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}")

# Compare ONet outputs
print("\n" + "="*80)
print("ONet Output Boxes (After Final Calibration)")
print("="*80)

print(f"\nCoreML ONet boxes ({coreml_debug['onet']['num_boxes']}):")
for i, box in enumerate(coreml_debug['onet']['boxes']):
    print(f"  Box {i}: x={box[0]:.1f}, y={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}")

print(f"\nONNX ONet boxes ({onnx_debug['onet']['num_boxes']}):")
for i, box in enumerate(onnx_debug['onet']['boxes']):
    print(f"  Box {i}: x={box[0]:.1f}, y={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}")

# IoU between ONNX's 2 boxes
if onnx_debug['onet']['num_boxes'] == 2:
    box1 = onnx_debug['onet']['boxes'][0]
    box2 = onnx_debug['onet']['boxes'][1]

    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max > x_inter_min and y_inter_max > y_inter_min:
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area

        print(f"\nIoU between ONNX's 2 boxes: {iou:.4f}")
        print(f"  → These boxes {'overlap significantly' if iou > 0.5 else 'are mostly separate'}")
    else:
        print(f"\nONNX's 2 boxes do not overlap")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("\nPossible causes for ONNX detecting 2 faces:")
print("1. ONet scores: One box is borderline (near 0.7 threshold)")
print("2. Final NMS: The 2 boxes don't overlap enough (IoU < 0.7) to be merged")
print("3. Model numerical differences: ~2% PNet difference propagates through pipeline")
print("\nNext step: Check ONet scores before final NMS to see if one is borderline")
