#!/usr/bin/env python3
"""
Debug script to find where optimized version diverges from original.
"""

import numpy as np
import cv2
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2 as OriginalMTCNN
from pure_python_mtcnn_optimized import PurePythonMTCNN_Optimized as OptimizedMTCNN

# Load a test frame
cap = cv2.VideoCapture("Patient Data/Normal Cohort/IMG_0422.MOV")
cap.set(cv2.CAP_PROP_POS_FRAMES, 222)
ret, frame = cap.read()
cap.release()

print("Testing layer outputs...")

# Initialize both
orig = OriginalMTCNN()
opt = OptimizedMTCNN()

# Run detection
print("\nRunning original...")
orig_bboxes, orig_landmarks = orig.detect(frame)
print(f"Original detected {len(orig_bboxes)} faces")
if len(orig_bboxes) > 0:
    print(f"  Box 0: {orig_bboxes[0]}")

print("\nRunning optimized...")
opt_bboxes, opt_landmarks = opt.detect(frame)
print(f"Optimized detected {len(opt_bboxes)} faces")
if len(opt_bboxes) > 0:
    print(f"  Box 0: {opt_bboxes[0]}")

if len(orig_bboxes) > 0 and len(opt_bboxes) > 0:
    diff = np.abs(orig_bboxes[0] - opt_bboxes[0])
    print(f"\nDifferences: {diff}")
    print(f"Max diff: {diff.max():.2f} pixels")

    # Calculate IoU
    x1 = max(orig_bboxes[0][0], opt_bboxes[0][0])
    y1 = max(orig_bboxes[0][1], opt_bboxes[0][1])
    x2 = min(orig_bboxes[0][0] + orig_bboxes[0][2], opt_bboxes[0][0] + opt_bboxes[0][2])
    y2 = min(orig_bboxes[0][1] + orig_bboxes[0][3], opt_bboxes[0][1] + opt_bboxes[0][3])

    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
        union = (orig_bboxes[0][2] * orig_bboxes[0][3] +
                 opt_bboxes[0][2] * opt_bboxes[0][3] - intersection)
        iou = intersection / union
        print(f"IoU: {iou:.4f}")
