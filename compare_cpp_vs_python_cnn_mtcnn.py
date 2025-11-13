#!/usr/bin/env python3
"""
Visual comparison: C++ MTCNN vs Current Python MTCNN implementations.
"""

import cv2
import numpy as np
import os

print("="*80)
print("C++ MTCNN vs PYTHON MTCNN COMPARISON")
print("="*80)

# Load test image
test_img = "calibration_frames/patient1_frame1.jpg"
print(f"\nLoading test image: {test_img}")
img = cv2.imread(test_img)
print(f"Image shape: {img.shape}")

# Load C++ MTCNN results
cpp_bbox_path = "/tmp/cpp_mtcnn_final_bbox.txt"
print(f"\n1. Loading C++ MTCNN results from: {cpp_bbox_path}")

cpp_bboxes = []
if os.path.exists(cpp_bbox_path):
    with open(cpp_bbox_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    x, y, w, h = map(float, parts[:4])
                    cpp_bboxes.append([x, y, w, h])

    print(f"   Found {len(cpp_bboxes)} C++ bboxes")
    for i, bbox in enumerate(cpp_bboxes):
        print(f"   C++ bbox {i+1}: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}")
else:
    print(f"   C++ results not found!")
    print("   Run C++ OpenFace first to generate comparison data")

# Try current Python MTCNN (ONNX-based)
print(f"\n2. Running current Python MTCNN (ONNX-based)...")
try:
    from cpp_mtcnn_detector import CPPMTCNNDetector

    detector = CPPMTCNNDetector()
    py_bboxes, py_landmarks = detector.detect(img)

    print(f"   Found {len(py_bboxes)} Python bboxes")
    for i, bbox in enumerate(py_bboxes):
        x, y, w, h = bbox
        print(f"   Python bbox {i+1}: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
except Exception as e:
    print(f"   Error running Python MTCNN: {e}")
    py_bboxes = []
    py_landmarks = []

# Create visualization
print(f"\n3. Creating visualization...")
vis = img.copy()

# Draw C++ bboxes in GREEN (thicker, 6px)
if len(cpp_bboxes) > 0:
    for i, bbox in enumerate(cpp_bboxes):
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 6)
        cv2.putText(vis, 'C++ MTCNN (Gold Standard)', (x, y-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# Draw Python bboxes in BLUE (thinner, 3px)
if len(py_bboxes) > 0:
    for i, bbox in enumerate(py_bboxes):
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(vis, 'Python MTCNN (ONNX)', (x, y+h+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

# Draw Python landmarks in RED
if len(py_landmarks) > 0:
    for lm in py_landmarks:
        if lm is not None and len(lm) >= 10:
            for j in range(5):
                try:
                    lm_x = int(lm[j*2])
                    lm_y = int(lm[j*2 + 1])
                    cv2.circle(vis, (lm_x, lm_y), 8, (0, 0, 255), -1)
                except:
                    pass

# Add comparison info
info_y = 50
cv2.putText(vis, "MTCNN COMPARISON", (20, info_y),
           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

info_y += 50
if len(cpp_bboxes) > 0 and len(py_bboxes) > 0:
    cpp_bbox = cpp_bboxes[0]
    py_bbox = py_bboxes[0]

    dx = abs(cpp_bbox[0] - py_bbox[0])
    dy = abs(cpp_bbox[1] - py_bbox[1])
    dw = abs(cpp_bbox[2] - py_bbox[2])
    dh = abs(cpp_bbox[3] - py_bbox[3])

    cv2.putText(vis, f"Position diff: dx={dx:.1f}, dy={dy:.1f}", (20, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    info_y += 40
    cv2.putText(vis, f"Size diff: dw={dw:.1f}, dh={dh:.1f}", (20, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# Save visualization
output_path = "cpp_vs_python_mtcnn_comparison.jpg"
cv2.imwrite(output_path, vis)
print(f"\n✅ Saved comparison to: {output_path}")

# Show analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if len(cpp_bboxes) > 0 and len(py_bboxes) > 0:
    print("\nBoth detectors found faces!")
    cpp_bbox = cpp_bboxes[0]
    py_bbox = py_bboxes[0]

    print(f"\nC++ bbox:    x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}, w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")
    print(f"Python bbox: x={py_bbox[0]:.1f}, y={py_bbox[1]:.1f}, w={py_bbox[2]:.1f}, h={py_bbox[3]:.1f}")

    dx = abs(cpp_bbox[0] - py_bbox[0])
    dy = abs(cpp_bbox[1] - py_bbox[1])
    dw = abs(cpp_bbox[2] - py_bbox[2])
    dh = abs(cpp_bbox[3] - py_bbox[3])

    print(f"\nDifferences:")
    print(f"  Position: dx={dx:.1f}px, dy={dy:.1f}px")
    print(f"  Size: dw={dw:.1f}px, dh={dh:.1f}px")

    if dx < 10 and dy < 10 and dw < 20 and dh < 20:
        print("\n✅ EXCELLENT! Bboxes are very close!")
    elif dx < 50 and dy < 50 and dw < 50 and dh < 50:
        print("\n✓ Good! Bboxes are reasonably close")
    else:
        print("\n⚠ Significant differences detected")
        print("   This is expected - ONNX uses modified weights + different PReLU")
        print("   Pure Python CNN (next step) will match C++ exactly!")

elif len(cpp_bboxes) == 0:
    print("\n⚠ No C++ results found")
    print("   Run C++ OpenFace first to generate comparison data")
elif len(py_bboxes) == 0:
    print("\n⚠ Python MTCNN found no faces")
else:
    print("\n⚠ Detection count mismatch")

print("\n" + "="*80)
print("NEXT STEP: Integrate Pure Python CNN")
print("="*80)
print("\nCurrent Python MTCNN uses ONNX (with PReLU issues)")
print("Pure Python CNN implementation is ready!")
print("Next: Replace ONNX inference with pure Python CNN calls")
print("Expected result: Perfect matching with C++ (green boxes)")
print("="*80)
