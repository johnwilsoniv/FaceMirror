"""
Debug MTCNN pipeline step-by-step to find where Python diverges from C++.
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')

# Run Python MTCNN with detailed debugging
print("=" * 80)
print("MTCNN Pipeline Debugging")
print("=" * 80)

detector = CPPMTCNNDetector()

# Manually step through the detection to inspect intermediate values
print("\n[Running full detection...]")
bboxes, landmarks = detector.detect(img)

print(f"\n[Final Python Results]")
print(f"  Total faces: {len(bboxes)}")
for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox
    print(f"  Face {i+1}: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")

# Load C++ results
import csv
from pathlib import Path

cpp_file = Path("/tmp/mtcnn_debug.csv")
if cpp_file.exists():
    print(f"\n[C++ Results]")
    with open(cpp_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['bbox_x'])
            y = float(row['bbox_y'])
            w = float(row['bbox_w'])
            h = float(row['bbox_h'])
            print(f"  C++: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")

# Calculate difference
if len(bboxes) > 0 and cpp_file.exists():
    # Get largest Python face
    widths = [bbox[2] for bbox in bboxes]
    largest_idx = np.argmax(widths)
    py_bbox = bboxes[largest_idx]

    print(f"\n[Comparison]")
    print(f"  Python (Face {largest_idx+1}):")
    print(f"    x={py_bbox[0]:.4f}, y={py_bbox[1]:.4f}, w={py_bbox[2]:.4f}, h={py_bbox[3]:.4f}")
    print(f"  C++:")
    print(f"    x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")
    print(f"  Differences:")
    print(f"    Δx = {py_bbox[0] - x:.4f} ({abs(py_bbox[0] - x):.1f} pixels)")
    print(f"    Δy = {py_bbox[1] - y:.4f} ({abs(py_bbox[1] - y):.1f} pixels)")
    print(f"    Δw = {py_bbox[2] - w:.4f} ({abs(py_bbox[2] - w):.1f} pixels)")
    print(f"    Δh = {py_bbox[3] - h:.4f} ({abs(py_bbox[3] - h):.1f} pixels)")

print("=" * 80)
