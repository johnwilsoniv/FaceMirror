#!/usr/bin/env python3
"""
Test: What if we just DON'T APPLY PNet regression at all?

Our trace showed the raw PNet box at (333, 703) was only 50px from gold y=753.
After regression it moved to (364, 672) which is 81px away - WORSE!

Maybe for Pure Python CNN, we should just skip PNet regression entirely?
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("TESTING: PNet WITHOUT Bbox Regression")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
print(f"\nTest image: {img.shape[1]}×{img.shape[0]}")
print(f"C++ Gold Standard: x=331, y=753, w=368, h=423")

# Create detector
detector = PurePythonMTCNN_V2()

# Monkey-patch to SKIP regression
original_apply_bbox_regression = detector._apply_bbox_regression

def skip_regression(bboxes):
    """Just return boxes unchanged."""
    return bboxes

detector._apply_bbox_regression = skip_regression

print("\n🔧 Patched _apply_bbox_regression to SKIP regression")
print("   Raw PNet boxes will go directly to RNet without adjustment")

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
        print(f"    Position: {x_err:.0f}px, {y_err:.0f}px")
        print(f"    Size: {w_err:.0f}px, {h_err:.0f}px")

        if x_err < 50 and y_err < 50:
            print(f"  ✅ Position is within 50px!")
        if w_err < 100 and h_err < 100:
            print(f"  ✅ Size is within 100px!")

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
        cv2.putText(img_vis, f'NO REGRESSION', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite('test_without_pnet_regression_result.jpg', img_vis)
    print(f"\n📊 Saved visualization: test_without_pnet_regression_result.jpg")

else:
    print("\n❌ No faces detected!")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if len(bboxes) > 0 and abs(bboxes[0][1] - 753) < 100:
    print("""
✅ SUCCESS! Skipping PNet regression WORKS BETTER!

This confirms that Pure Python CNN PNet regression makes things WORSE, not better.

Next step: Either:
1. Fix Pure Python CNN to output correct regression values
2. Or just disable PNet regression for Pure Python CNN

For now, option 2 might be the pragmatic choice.
""")
else:
    print("""
⚠ Still didn't work. The problem runs deeper than just regression.

Need to investigate further - maybe compare raw PNet outputs between C++ and Python.
""")
