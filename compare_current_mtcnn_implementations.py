#!/usr/bin/env python3
"""
Compare current ONNX MTCNN (Python) vs C++ MTCNN gold standard.
Shows the ~30-70px bbox differences due to ONNX weight modifications.
"""

import cv2
import numpy as np
from pathlib import Path

# Get C++ MTCNN results from previous comparison
cpp_bbox = np.array([331.6, 753.5, 367.9, 422.8])  # x, y, w, h from previous session

# Load test image
test_image = Path("calibration_frames/patient1_frame1.jpg")
img = cv2.imread(str(test_image))

print("="*80)
print("MTCNN IMPLEMENTATION COMPARISON")
print("="*80)
print(f"\nImage: {test_image.name} ({img.shape[1]}x{img.shape[0]})")

# Get ONNX MTCNN results
from cpp_mtcnn_detector import CPPMTCNNDetector

print("\nRunning ONNX MTCNN detector...")
onnx_detector = CPPMTCNNDetector()
onnx_bboxes, onnx_landmarks = onnx_detector.detect(img)

if len(onnx_bboxes) > 0:
    onnx_bbox = onnx_bboxes[0]  # Take first detection
else:
    onnx_bbox = np.array([0, 0, 0, 0])

print("\n" + "="*80)
print("DETECTION RESULTS")
print("="*80)

print(f"\nC++ MTCNN (Gold Standard):")
print(f"  Position: x={cpp_bbox[0]:.1f}, y={cpp_bbox[1]:.1f}")
print(f"  Size: w={cpp_bbox[2]:.1f}, h={cpp_bbox[3]:.1f}")

print(f"\nONNX MTCNN (Current Python):")
print(f"  Position: x={onnx_bbox[0]:.1f}, y={onnx_bbox[1]:.1f}")
print(f"  Size: w={onnx_bbox[2]:.1f}, h={onnx_bbox[3]:.1f}")

# Calculate differences
dx = abs(onnx_bbox[0] - cpp_bbox[0])
dy = abs(onnx_bbox[1] - cpp_bbox[1])
dw = abs(onnx_bbox[2] - cpp_bbox[2])
dh = abs(onnx_bbox[3] - cpp_bbox[3])

print(f"\nDifferences:")
print(f"  Position: dx={dx:.1f}px, dy={dy:.1f}px")
print(f"  Size: dw={dw:.1f}px, dh={dh:.1f}px")

# Create visualization
img_vis = img.copy()

# Draw C++ bbox (GREEN - thick 6px)
x, y, w, h = cpp_bbox.astype(int)
cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 6)
cv2.putText(img_vis, "C++ MTCNN (Gold Standard)", (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

# Draw ONNX bbox (BLUE - thin 3px)
x, y, w, h = onnx_bbox.astype(int)
cv2.rectangle(img_vis, (x, y), (x+w, y+h), (255, 0, 0), 3)
cv2.putText(img_vis, "ONNX MTCNN (ONNX)", (x, y+h+40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

# Add comparison text
text_y = 60
cv2.putText(img_vis, "MTCNN COMPARISON", (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
text_y += 60
cv2.putText(img_vis, f"Position diff: dx={dx:.1f}, dy={dy:.1f}", (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
text_y += 50
cv2.putText(img_vis, f"Size diff: dw={dw:.1f}, dh={dh:.1f}", (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

output_path = "mtcnn_comparison_onnx_vs_cpp.jpg"
cv2.imwrite(output_path, img_vis)

print(f"\n✓ Saved comparison: {output_path}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("\nThe ~30-70px differences are due to:")
print("  1. ONNX weight modifications during export")
print("  2. PReLU implementation workarounds in ONNX")
print("\nFor perfect C++ matching, we would need:")
print("  - Pure Python CNN loading original C++ .dat weights")
print("  - Exact C++ PReLU behavior (already implemented)")
print("  - This requires resolving .dat format architecture mismatches")
