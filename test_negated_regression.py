#!/usr/bin/env python3
"""
Quick test: What if we NEGATE the PNet regression values?

If dy1=-0.0696 moves the box UP when it should move DOWN,
maybe the sign is simply flipped in Pure Python CNN output!

This test will negate all regression values and run detection again.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("TESTING: Negated PNet Regression Values")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print(f"\nTest image: {img_w}×{img_h}")
print(f"C++ Gold Standard: x=331, y=753, w=368, h=423")

# Create detector with modified regression
detector = PurePythonMTCNN_V2()

# Monkey-patch the _generate_bboxes method to NEGATE regression values
original_generate_bboxes = detector._generate_bboxes

def generate_bboxes_negated(score_map, reg_map, scale, threshold):
    """Modified to negate regression values."""
    # Call original
    boxes = original_generate_bboxes(score_map, reg_map, scale, threshold)

    if boxes.shape[0] > 0:
        # NEGATE regression values (columns 5-8)
        boxes[:, 5:9] = -boxes[:, 5:9]

    return boxes

detector._generate_bboxes = generate_bboxes_negated

print("\n🔧 Patched _generate_bboxes to NEGATE regression values")
print("   Original: dy1 = -0.0696 (moves box UP)")
print("   Modified: dy1 = +0.0696 (moves box DOWN)")

# Run detection
bboxes, landmarks = detector.detect(img, debug=True)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if len(bboxes) > 0:
    print(f"\n✅ Detected {len(bboxes)} face(s)!")

    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        print(f"\nFace {i+1}:")
        print(f"  Position: ({x:.0f}, {y:.0f})")
        print(f"  Size: {w:.0f}×{h:.0f}")

        # Compare to gold standard
        gold_x, gold_y, gold_w, gold_h = 331, 753, 368, 423
        x_err = abs(x - gold_x)
        y_err = abs(y - gold_y)
        w_err = abs(w - gold_w)
        h_err = abs(h - gold_h)

        print(f"  Error from gold:")
        print(f"    x: {x_err:.0f}px")
        print(f"    y: {y_err:.0f}px")
        print(f"    w: {w_err:.0f}px")
        print(f"    h: {h_err:.0f}px")

        if x_err < 50 and y_err < 50:
            print(f"  ✅ GOOD! Position is within 50px of gold standard")
        else:
            print(f"  ⚠ Position error is large")

        if w_err < 100 and h_err < 100:
            print(f"  ✅ GOOD! Size is within 100px of gold standard")
        else:
            print(f"  ⚠ Size error is large")

    # Visualize
    img_vis = img.copy()

    # Draw gold standard (red)
    cv2.rectangle(img_vis, (331, 753), (699, 1176), (0, 0, 255), 3)
    cv2.putText(img_vis, 'GOLD', (331, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw detected boxes (green)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_vis, f'DETECTED (negated regression)', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite('test_negated_regression_result.jpg', img_vis)
    print(f"\n📊 Saved visualization: test_negated_regression_result.jpg")

else:
    print("\n❌ No faces detected!")
    print("   Negating regression didn't fix the issue.")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if len(bboxes) > 0 and abs(bboxes[0][1] - 753) < 50:
    print("""
✅ SUCCESS! Negating the regression values FIXED the detection!

This proves that Pure Python CNN outputs regression values with THE WRONG SIGN.

Root cause: Pure Python CNN has a sign error somewhere in:
- Weight loading
- Layer implementation
- Output processing

Next step: Find WHERE in Pure Python CNN the sign gets flipped.
""")
else:
    print("""
⚠ Negating didn't fix it. The problem is more complex.

Need to compare raw PNet outputs between Python and C++ to find the difference.
""")
