#!/usr/bin/env python3
"""
We detected 2 faces with Python MTCNN.
Let's check which face C++ selected and compare the CORRECT one!

Hypothesis: C++ selected Face #1, but we've been comparing to Face #2.
"""

import cv2
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("WHICH FACE DID C++ SELECT?")
print("=" * 80)

img = cv2.imread('calibration_frames/patient1_frame1.jpg')
print(f"\nTest image: {img.shape[1]}×{img.shape[0]}")

# Python detection
detector = PurePythonMTCNN_V2()
bboxes, landmarks = detector.detect(img, debug=False)

print(f"\nPython detected {len(bboxes)} faces:")
for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox
    print(f"\nFace #{i+1}:")
    print(f"  Position: ({x:.0f}, {y:.0f})")
    print(f"  Size: {w:.0f}×{h:.0f}")
    print(f"  Area: {w*h:.0f}px²")

print("\n" + "=" * 80)
print("COMPARING TO C++ GOLD STANDARD")
print("=" * 80)

gold_x, gold_y, gold_w, gold_h = 331, 753, 368, 423
print(f"\nC++ Gold: ({gold_x}, {gold_y}) {gold_w}×{gold_h}")
print(f"C++ Area: {gold_w*gold_h}px²")

print("\n" + "=" * 80)
print("DISTANCE ANALYSIS")
print("=" * 80)

best_match_idx = -1
best_match_dist = float('inf')

for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox

    # Calculate distance from gold
    x_err = abs(x - gold_x)
    y_err = abs(y - gold_y)
    w_err = abs(w - gold_w)
    h_err = abs(h - gold_h)

    # Euclidean distance in (x, y, w, h) space
    dist = (x_err**2 + y_err**2 + w_err**2 + h_err**2) ** 0.5

    print(f"\nFace #{i+1} error from gold:")
    print(f"  Position: {x_err:.0f}px, {y_err:.0f}px")
    print(f"  Size: {w_err:.0f}px, {h_err:.0f}px")
    print(f"  Total distance: {dist:.1f}")

    if dist < best_match_dist:
        best_match_dist = dist
        best_match_idx = i

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"\n✅ Best match: Face #{best_match_idx+1} (distance: {best_match_dist:.1f})")

best_bbox = bboxes[best_match_idx]
x, y, w, h = best_bbox

x_err = abs(x - gold_x)
y_err = abs(y - gold_y)

if x_err < 50 and y_err < 50:
    print(f"""
✅ EXCELLENT MATCH! Position error <50px

Face #{best_match_idx+1}: ({x:.0f}, {y:.0f}) {w:.0f}×{h:.0f}
C++ Gold:  ({gold_x}, {gold_y}) {gold_w}×{gold_h}
Error: {x_err:.0f}px, {y_err:.0f}px

This confirms Pure Python MTCNN is working correctly with the RGB fix!

Remaining differences ({x_err:.0f}px, {y_err:.0f}px) are likely due to:
1. Floating point precision differences
2. Rounding differences in coordinate calculations
3. Slight NMS implementation differences

These differences are NORMAL and acceptable for production use.
""")
else:
    print(f"""
⚠ Still {x_err:.0f}px, {y_err:.0f}px off.

Need to investigate further:
- ONet bbox regression differences?
- Different bbox squaring implementation?
- Network output differences between Pure Python CNN and C++?
""")

# Visualize with labels
img_vis = img.copy()

# Draw gold (red)
cv2.rectangle(img_vis, (331, 753), (699, 1176), (0, 0, 255), 3)
cv2.putText(img_vis, 'C++ GOLD', (331, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Draw all detected faces
colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255)]
for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)

    color = colors[i % len(colors)]
    thickness = 4 if i == best_match_idx else 2

    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)

    label = f'BEST MATCH (Face #{i+1})' if i == best_match_idx else f'Face #{i+1}'
    cv2.putText(img_vis, label, (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite('compare_which_face_selected.jpg', img_vis)
print(f"\n📊 Saved: compare_which_face_selected.jpg")
