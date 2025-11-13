#!/usr/bin/env python3
"""
Test the BGR→RGB fix for Pure Python CNN preprocessing.

C++ flips BGR to RGB before feeding to network (line 151-155).
We were keeping BGR order - this is THE BUG that caused wrong detections!
"""

import cv2
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("TESTING: BGR→RGB Fix")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
print(f"\nTest image: {img.shape[1]}×{img.shape[0]}")
print(f"C++ Gold Standard: x=331, y=753, w=368, h=423")

# Create detector (now with RGB preprocessing!)
detector = PurePythonMTCNN_V2()

print("\n✅ Now using BGR→RGB conversion to match C++ (line 151-155)")

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
            print(f"  ✅ EXCELLENT! Position within 50px")
        if w_err < 100 and h_err < 100:
            print(f"  ✅ EXCELLENT! Size within 100px")

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
        cv2.putText(img_vis, f'DETECTED (RGB fix)', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite('test_rgb_fix_result.jpg', img_vis)
    print(f"\n📊 Saved visualization: test_rgb_fix_result.jpg")

else:
    print("\n❌ No faces detected")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if len(bboxes) > 0 and abs(bboxes[0][0] - 331) < 50 and abs(bboxes[0][1] - 753) < 50:
    print("""
✅✅✅ SUCCESS! The BGR→RGB fix WORKS! ✅✅✅

Root cause identified and fixed:
- Python was feeding BGR to Pure Python CNN
- C++ flips BGR to RGB before inference (line 151-155)
- This caused completely different network outputs

The fix: img_rgb = img_chw[[2, 1, 0], :, :]

Detection is now working correctly!
""")
else:
    print("""
⚠ Still having issues. Need further investigation.
""")
