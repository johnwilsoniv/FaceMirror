#!/usr/bin/env python3
"""
Test Pure Python CNN MTCNN V2 (hybrid with ONNX pipeline).
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

# Load image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')

print("="*80)
print("PURE PYTHON CNN MTCNN V2 TEST")
print("="*80)
print(f"Image: {img.shape[1]}x{img.shape[0]}")

# Create detector
detector = PurePythonMTCNN_V2()

# Run detection
print("\nRunning detection...")
bboxes, landmarks = detector.detect(img)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Detected {len(bboxes)} faces")

# C++ gold standard
cpp_bbox = np.array([331.6, 753.5, 367.9, 422.8])

if len(bboxes) > 0:
    pp_bbox = bboxes[0]
    pp_landmarks = landmarks[0]

    print(f"\nPure Python CNN V2:")
    print(f"  Position: x={pp_bbox[0]:.1f}, y={pp_bbox[1]:.1f}")
    print(f"  Size: w={pp_bbox[2]:.1f}, h={pp_bbox[3]:.1f}")

    print(f"\nC++ Gold Standard:")
    print(f"  Position: x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}")
    print(f"  Size: w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")

    # Calculate differences
    dx = abs(pp_bbox[0] - cpp_bbox[0])
    dy = abs(pp_bbox[1] - cpp_bbox[1])
    dw = abs(pp_bbox[2] - cpp_bbox[2])
    dh = abs(pp_bbox[3] - cpp_bbox[3])

    print(f"\nDifferences:")
    print(f"  Position: dx={dx:.1f}px ({dx/cpp_bbox[0]*100:.1f}%), dy={dy:.1f}px ({dy/cpp_bbox[1]*100:.1f}%)")
    print(f"  Size: dw={dw:.1f}px ({dw/cpp_bbox[2]*100:.1f}%), dh={dh:.1f}px ({dh/cpp_bbox[3]*100:.1f}%)")

    if dx < 10 and dy < 10 and dw < 10 and dh < 10:
        print("\n🎉 EXCELLENT! Within 10px on all dimensions!")
    elif dx < 50 and dy < 50 and dw < 50 and dh < 50:
        print("\n✓ Good! Within 50px")
    else:
        print("\n⚠️  Differences still significant")

    # Visualize
    img_vis = img.copy()

    # Draw C++ bbox (GREEN)
    x, y, w, h = cpp_bbox.astype(int)
    cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 6)
    cv2.putText(img_vis, "C++ MTCNN", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Draw Pure Python V2 bbox (RED)
    x, y, w, h = pp_bbox.astype(int)
    cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.putText(img_vis, "Pure Python V2", (x, y+h+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Draw landmarks
    for point in pp_landmarks:
        px, py = int(point[0]), int(point[1])
        cv2.circle(img_vis, (px, py), 5, (255, 0, 0), -1)

    output_path = "pure_python_v2_comparison.jpg"
    cv2.imwrite(output_path, img_vis)
    print(f"\n✓ Saved: {output_path}")
else:
    print("\n❌ No faces detected")
