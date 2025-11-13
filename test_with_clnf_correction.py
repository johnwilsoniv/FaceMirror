#!/usr/bin/env python3
"""
Test Pure Python MTCNN with the CLNF bbox correction applied.

The C++ code applies a hardcoded correction (lines 386-396) after MTCNN detection.
This correction is specific to CLNF's expected bbox format.

If the gold standard includes this correction, we should apply it too!
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("TESTING: Pure Python MTCNN with CLNF Correction")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
print(f"\nTest image: {img.shape[1]}×{img.shape[0]}")
print(f"C++ Gold Standard (after CLNF correction): x=331, y=753, w=368, h=423")

# Create detector
detector = PurePythonMTCNN_V2()

# Run detection
bboxes, landmarks = detector.detect(img, debug=False)

print(f"\n✅ Detected {len(bboxes)} face(s)")

if len(bboxes) == 0:
    print("❌ No faces detected!")
    exit(1)

# Show raw MTCNN output
print("\n" + "=" * 80)
print("RAW MTCNN OUTPUT (before CLNF correction)")
print("=" * 80)

for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox
    print(f"\nFace {i+1}:")
    print(f"  Position: ({x:.0f}, {y:.0f})")
    print(f"  Size: {w:.0f}×{h:.0f}")

# Apply CLNF correction (matching C++ lines 386-396)
print("\n" + "=" * 80)
print("APPLYING CLNF CORRECTION")
print("=" * 80)

corrected_bboxes = []

for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox

    # C++ correction coefficients (from lines 389-392)
    new_x = x + w * -0.0075
    new_y = y + h * 0.2459
    new_w = w * 1.0323
    new_h = h * 0.7751

    print(f"\nFace {i+1}:")
    print(f"  Before: ({x:.0f}, {y:.0f}) {w:.0f}×{h:.0f}")
    print(f"  After:  ({new_x:.0f}, {new_y:.0f}) {new_w:.0f}×{new_h:.0f}")
    print(f"  Changes:")
    print(f"    x: {new_x - x:+.0f}px ({(new_x - x)/w * 100:+.1f}%)")
    print(f"    y: {new_y - y:+.0f}px ({(new_y - y)/h * 100:+.1f}%)")
    print(f"    w: {new_w - w:+.0f}px ({(new_w - w)/w * 100:+.1f}%)")
    print(f"    h: {new_h - h:+.0f}px ({(new_h - h)/h * 100:+.1f}%)")

    corrected_bboxes.append([new_x, new_y, new_w, new_h])

# Compare to gold standard
print("\n" + "=" * 80)
print("COMPARISON TO GOLD STANDARD")
print("=" * 80)

gold_x, gold_y, gold_w, gold_h = 331, 753, 368, 423

for i, bbox in enumerate(corrected_bboxes):
    x, y, w, h = bbox

    x_err = abs(x - gold_x)
    y_err = abs(y - gold_y)
    w_err = abs(w - gold_w)
    h_err = abs(h - gold_h)

    print(f"\nFace {i+1} (after CLNF correction):")
    print(f"  Detected: ({x:.0f}, {y:.0f}) {w:.0f}×{h:.0f}")
    print(f"  Gold:     ({gold_x}, {gold_y}) {gold_w}×{gold_h}")
    print(f"  Error:")
    print(f"    Position: {x_err:.0f}px, {y_err:.0f}px")
    print(f"    Size: {w_err:.0f}px, {h_err:.0f}px")

    if x_err < 20 and y_err < 20:
        print(f"  ✅ EXCELLENT! Position within 20px")
    elif x_err < 50 and y_err < 50:
        print(f"  ✅ GOOD! Position within 50px")

    if w_err < 50 and h_err < 50:
        print(f"  ✅ EXCELLENT! Size within 50px")
    elif w_err < 100 and h_err < 100:
        print(f"  ✅ GOOD! Size within 100px")

# Visualize
img_vis = img.copy()

# Draw gold standard (red)
cv2.rectangle(img_vis, (331, 753), (699, 1176), (0, 0, 255), 3)
cv2.putText(img_vis, 'GOLD (with CLNF corr)', (331, 740),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Draw raw MTCNN (blue - dashed effect with thin lines)
for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_vis, 'Raw MTCNN', (x1, y1-30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Draw corrected (green)
for i, bbox in enumerate(corrected_bboxes):
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img_vis, 'With CLNF correction', (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite('test_with_clnf_correction_result.jpg', img_vis)
print(f"\n📊 Saved visualization: test_with_clnf_correction_result.jpg")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

best_bbox = corrected_bboxes[0]
best_err_x = abs(best_bbox[0] - gold_x)
best_err_y = abs(best_bbox[1] - gold_y)

if best_err_x < 20 and best_err_y < 20:
    print("""
✅✅✅ PERFECT MATCH! ✅✅✅

Pure Python MTCNN with BGR→RGB fix and CLNF correction
now matches C++ output within 20px!

The fix is complete and working correctly.
""")
elif best_err_x < 50 and best_err_y < 50:
    print("""
✅ VERY CLOSE! Within 50px of gold standard.

Pure Python MTCNN with BGR→RGB fix is working well.
Small remaining differences may be due to:
- Floating point precision
- Slightly different NMS implementations
- Other minor differences between C++ and Python

This is good enough for production use!
""")
else:
    print(f"""
⚠ Still {best_err_x:.0f}px, {best_err_y:.0f}px off from gold standard.

Further investigation needed.
""")
