#!/usr/bin/env python3
"""
Test Pure Python MTCNN with VERY LOW thresholds to see all detections.
"""

import cv2
import numpy as np
from pure_python_mtcnn_detector import PurePythonMTCNNDetector

# Load image
img_path = 'calibration_frames/patient1_frame1.jpg'
img = cv2.imread(img_path)

print("="*80)
print("TESTING PURE PYTHON MTCNN - LOW THRESHOLDS")
print("="*80)
print(f"\nImage: {img_path}")
print(f"Size: {img.shape[1]}x{img.shape[0]}")

# Create detector with VERY LOW thresholds
detector = PurePythonMTCNNDetector()
print(f"\nDefault thresholds:")
print(f"  PNet: {detector.pnet_threshold}")
print(f"  RNet: {detector.rnet_threshold}")
print(f"  ONet: {detector.onet_threshold}")

# Lower all thresholds
detector.pnet_threshold = 0.6
detector.rnet_threshold = 0.6
detector.onet_threshold = 0.6

print(f"\nLowered thresholds:")
print(f"  PNet: {detector.pnet_threshold}")
print(f"  RNet: {detector.rnet_threshold}")
print(f"  ONet: {detector.onet_threshold}")

# Run detection
bboxes, landmarks = detector.detect(img, debug=True)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nDetected {len(bboxes)} faces")

for i, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
    x, y, w, h = bbox
    print(f"\nFace {i+1}:")
    print(f"  BBox: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
    print(f"  Landmarks: {lm.shape}")

# Visualize
img_vis = img.copy()
for bbox, lm in zip(bboxes, landmarks):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Draw bbox (GREEN)
    cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(img_vis, "Pure Python CNN", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw landmarks (RED dots)
    for point in lm:
        px, py = int(point[0]), int(point[1])
        cv2.circle(img_vis, (px, py), 3, (0, 0, 255), -1)

output_path = "pure_python_mtcnn_low_threshold.jpg"
cv2.imwrite(output_path, img_vis)
print(f"\n✓ Saved: {output_path}")
