#!/usr/bin/env python3
"""
Test Pure Python MTCNN after adding PNet bbox regression fix.
This should now match C++ behavior!
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("TESTING PNET BBOX REGRESSION FIX")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]

print(f"\nTest image: {img_w}×{img_h}")
print(f"C++ Gold Standard: x=331.6, y=753.5, w=367.9, h=422.8")

# Create detector
detector = PurePythonMTCNN_V2()

# Run detection with debug output
print("\n" + "=" * 80)
print("RUNNING DETECTION WITH DEBUG OUTPUT")
print("=" * 80)

bboxes, landmarks = detector.detect(img, debug=True)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

if len(bboxes) > 0:
    print(f"\n✓ Detected {len(bboxes)} face(s)")

    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        area = w * h

        # Calculate IoU with C++ gold standard
        cpp_x, cpp_y, cpp_w, cpp_h = 331.6, 753.5, 367.9, 422.8
        cpp_x2 = cpp_x + cpp_w
        cpp_y2 = cpp_y + cpp_h

        py_x2 = x + w
        py_y2 = y + h

        # Intersection
        inter_x1 = max(x, cpp_x)
        inter_y1 = max(y, cpp_y)
        inter_x2 = min(py_x2, cpp_x2)
        inter_y2 = min(py_y2, cpp_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        cpp_area = cpp_w * cpp_h
        union_area = area + cpp_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        print(f"\n  Face {i+1}:")
        print(f"    Position: ({x:.1f}, {y:.1f})")
        print(f"    Size: {w:.1f} × {h:.1f} px")
        print(f"    Area: {area:.0f} px²")
        print(f"    IoU with C++ gold standard: {iou:.4f} ({iou*100:.1f}%)")

        if iou > 0.7:
            print(f"    ✓ EXCELLENT match! (IoU > 70%)")
        elif iou > 0.5:
            print(f"    ✓ Good match (IoU > 50%)")
        elif iou > 0.3:
            print(f"    ⚠ Moderate match (IoU > 30%)")
        else:
            print(f"    ✗ Poor match (IoU < 30%)")
else:
    print("\n✗ No faces detected")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("""
Key Fix Applied:
- Added _apply_bbox_regression() method (matching C++ apply_correction)
- Applied PNet bbox regression AFTER cross-scale NMS (matching C++ line 1009)
- This expands the initial 12/scale boxes to full face size using PNet's
  learned regression offsets

Expected Behavior:
- PNet should now generate properly sized boxes (~368×423 for test image)
- RNet should accept these boxes (well-framed faces)
- Final detection should match C++ gold standard

Previous Behavior:
- PNet generated 12/scale boxes WITHOUT applying regression
- Small boxes (40-80px) passed RNet (tightly cropped features)
- Large boxes would work but regression was never applied
- Result: Tiny 30px detection instead of 368px
""")
