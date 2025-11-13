#!/usr/bin/env python3
"""Test the fully-fixed CPPMTCNNDetector on the previously-failed frame."""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

# Frame that previously failed (0 detections)
test_frame = "bbox_dataset/patient_061_20250408_191322000_iOS/frames/frame_00312.jpg"
gt_bbox = (184.57, 618.861, 766.321, 799.469)
gt_scale = np.sqrt(gt_bbox[2] * gt_bbox[3])

print("="*80)
print("TESTING FIXED C++ MTCNN DETECTOR")
print("="*80)

# Load image
print(f"\nLoading test frame: {test_frame}")
img = cv2.imread(test_frame)
if img is None:
    print("ERROR: Could not load image")
    exit(1)

print(f"Image shape: {img.shape}")
print(f"Ground truth bbox: ({gt_bbox[0]:.2f}, {gt_bbox[1]:.2f}, {gt_bbox[2]:.2f}, {gt_bbox[3]:.2f})")
print(f"Ground truth scale: {gt_scale:.2f}")

# Initialize detector
print("\nInitializing detector...")
detector = CPPMTCNNDetector()
detector.min_face_size = 40

# Run detection
print("\nRunning detection...")
bboxes, landmarks = detector.detect(img)

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")

if len(bboxes) == 0:
    print("\n❌ FAILED: No faces detected")
    print("\nThis frame previously failed with 0 detections.")
    print("If still failing, there may be additional bugs to fix.")
else:
    print(f"\n✅ SUCCESS: Detected {len(bboxes)} face(s)")

    for i, bbox in enumerate(bboxes):
        pred_scale = np.sqrt(bbox[2] * bbox[3])
        init_scale_error = abs(pred_scale - gt_scale) / gt_scale * 100

        print(f"\nFace {i+1}:")
        print(f"  Predicted bbox: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})")
        print(f"  Predicted scale: {pred_scale:.2f}")
        print(f"  Init scale error: {init_scale_error:.2f}%")

        if init_scale_error < 5.0:
            print(f"  ✅ Excellent match! (< 5% error)")
        elif init_scale_error < 10.0:
            print(f"  ✓ Good match (< 10% error)")
        else:
            print(f"  ⚠ Large error (> 10%)")

print("\n" + "="*80)
