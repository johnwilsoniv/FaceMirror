#!/usr/bin/env python3
"""
Compare C++ OpenFace MTCNN vs PyMTCNN (BGR-Fixed)
- Green boxes: C++ OpenFace MTCNN
- Blue boxes: PyMTCNN (BGR-Fixed)
- Red dots: PyMTCNN landmarks
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

print("="*80)
print("C++ MTCNN vs PyMTCNN COMPARISON")
print("="*80)

# Load test image
img = cv2.imread('cpp_mtcnn_test.jpg')
print(f"\nTest image shape: {img.shape}")

# 1. Load C++ bbox
cpp_bboxes = []
try:
    with open('/tmp/cpp_mtcnn_final_bbox.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y, w, h = map(float, parts[:4])
                cpp_bboxes.append([x, y, w, h])
    cpp_bboxes = np.array(cpp_bboxes)
    print(f"\nC++ MTCNN Results:")
    print(f"  Detected {len(cpp_bboxes)} face(s)")
    for i, bbox in enumerate(cpp_bboxes):
        print(f"  Face {i+1}: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}x{bbox[3]:.1f})")
except:
    print("\n⚠ No C++ bbox found")
    cpp_bboxes = np.array([])

# 2. Run PyMTCNN
print(f"\nPyMTCNN (BGR-Fixed) Results:")
detector = CPPMTCNNDetector()
py_bboxes, py_landmarks = detector.detect(img)
print(f"  Detected {len(py_bboxes)} face(s)")
for i, bbox in enumerate(py_bboxes):
    print(f"  Face {i+1}: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}x{bbox[3]:.1f})")

# 3. Draw comparison
vis = img.copy()

# Draw C++ bboxes in GREEN (thicker)
if len(cpp_bboxes) > 0:
    for i, bbox in enumerate(cpp_bboxes):
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(vis, 'C++ MTCNN', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

# Draw PyMTCNN bboxes in BLUE (thinner)
if len(py_bboxes) > 0:
    for i, bbox in enumerate(py_bboxes):
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(vis, 'PyMTCNN (BGR-Fixed)', (x, y+h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

# Draw PyMTCNN landmarks in RED
if len(py_landmarks) > 0:
    for lm in py_landmarks:
        if lm is not None and len(lm) >= 10:
            for j in range(5):
                try:
                    lm_x = int(lm[j*2])
                    lm_y = int(lm[j*2 + 1])
                    cv2.circle(vis, (lm_x, lm_y), 5, (0, 0, 255), -1)
                except:
                    pass

# Save
output_path = 'cpp_vs_pymtcnn_comparison.jpg'
cv2.imwrite(output_path, vis)

print(f"\n{'='*80}")
print(f"COMPARISON SUMMARY:")
print(f"{'='*80}")

if len(cpp_bboxes) > 0 and len(py_bboxes) > 0:
    # Compare bboxes
    cpp_bbox = cpp_bboxes[0]
    py_bbox = py_bboxes[0]

    diff = np.abs(cpp_bbox - py_bbox)

    print(f"\nBBox Comparison:")
    print(f"  C++ MTCNN:  ({cpp_bbox[0]:.1f}, {cpp_bbox[1]:.1f}, {cpp_bbox[2]:.1f}x{cpp_bbox[3]:.1f})")
    print(f"  PyMTCNN:    ({py_bbox[0]:.1f}, {py_bbox[1]:.1f}, {py_bbox[2]:.1f}x{py_bbox[3]:.1f})")
    print(f"  Difference: ({diff[0]:.1f}, {diff[1]:.1f}, {diff[2]:.1f}x{diff[3]:.1f})")
    print(f"  Max diff:   {diff.max():.1f} pixels")

print(f"\n✅ Saved visualization to: {output_path}")
print(f"\nVisualization legend:")
print(f"  🟢 GREEN (thick) = C++ OpenFace MTCNN")
print(f"  🔵 BLUE (thin)   = PyMTCNN (BGR-Fixed)")
print(f"  🔴 RED (dots)    = PyMTCNN Landmarks")
