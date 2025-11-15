#!/usr/bin/env python3
"""Debug landmark coordinate conversion"""

import numpy as np
import cv2
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

from pymtcnn.backends.coreml_backend import CoreMLMTCNN

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

# Load image
img = cv2.imread(TEST_IMAGE)
print(f"Test image: {TEST_IMAGE} ({img.shape[1]}x{img.shape[0]})")

# Get PyMTCNN detection
print("\n" + "="*80)
print("PyMTCNN CoreML Detection")
print("="*80)
detector = CoreMLMTCNN(verbose=False)
bboxes, landmarks = detector.detect(img)

if len(bboxes) > 0:
    bbox = bboxes[0]
    lms = landmarks[0]
    
    print(f"\nBBox: x={bbox[0]:.2f}, y={bbox[1]:.2f}, w={bbox[2]-bbox[0]:.2f}, h={bbox[3]-bbox[1]:.2f}")
    print(f"      [x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, x2={bbox[2]:.2f}, y2={bbox[3]:.2f}]")
    print(f"\nLandmarks (absolute coordinates):")
    for i, (x, y) in enumerate(lms):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

# Read C++ debug CSV
print("\n" + "="*80)
print("C++ MTCNN Detection")
print("="*80)

df = pd.read_csv("/tmp/mtcnn_debug.csv")
row = df.iloc[0]

bbox_x = row['bbox_x']
bbox_y = row['bbox_y']
bbox_w = row['bbox_w']
bbox_h = row['bbox_h']

print(f"\nBBox: x={bbox_x:.2f}, y={bbox_y:.2f}, w={bbox_w:.2f}, h={bbox_h:.2f}")
print(f"      [x1={bbox_x:.2f}, y1={bbox_y:.2f}, x2={bbox_x+bbox_w:.2f}, y2={bbox_y+bbox_h:.2f}]")

print(f"\nLandmarks (normalized):")
for i in range(1, 6):
    lm_x = row[f'lm{i}_x']
    lm_y = row[f'lm{i}_y']
    print(f"  Point {i-1}: ({lm_x:.6f}, {lm_y:.6f})")

print(f"\nLandmarks (converted to absolute):")
for i in range(1, 6):
    lm_x_norm = row[f'lm{i}_x']
    lm_y_norm = row[f'lm{i}_y']
    lm_x_abs = bbox_x + (lm_x_norm * bbox_w)
    lm_y_abs = bbox_y + (lm_y_norm * bbox_h)
    print(f"  Point {i-1}: ({lm_x_abs:.2f}, {lm_y_abs:.2f})")

# Check if bboxes are similar
print("\n" + "="*80)
print("BBox Comparison")
print("="*80)
if len(bboxes) > 0:
    pymtcnn_bbox_w = bboxes[0][2] - bboxes[0][0]
    pymtcnn_bbox_h = bboxes[0][3] - bboxes[0][1]
    
    print(f"C++ BBox:     [{bbox_x:.1f}, {bbox_y:.1f}] -> [{bbox_x+bbox_w:.1f}, {bbox_y+bbox_h:.1f}]  (w={bbox_w:.1f}, h={bbox_h:.1f})")
    print(f"PyMTCNN BBox: [{bboxes[0][0]:.1f}, {bboxes[0][1]:.1f}] -> [{bboxes[0][2]:.1f}, {bboxes[0][3]:.1f}]  (w={pymtcnn_bbox_w:.1f}, h={pymtcnn_bbox_h:.1f})")
    
    # Compute IoU
    x1 = max(bbox_x, bboxes[0][0])
    y1 = max(bbox_y, bboxes[0][1])
    x2 = min(bbox_x + bbox_w, bboxes[0][2])
    y2 = min(bbox_y + bbox_h, bboxes[0][3])
    
    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
        area_cpp = bbox_w * bbox_h
        area_pymtcnn = pymtcnn_bbox_w * pymtcnn_bbox_h
        union = area_cpp + area_pymtcnn - intersection
        iou = intersection / union
        print(f"\nIoU: {iou:.4f}")
