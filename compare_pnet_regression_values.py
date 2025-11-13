#!/usr/bin/env python3
"""
Compare PNet regression values between Python and C++ for the problematic box.

The box at (333, 703) with regression dy1=-0.0696 moves the box UP when it should move DOWN.
Let's check if C++ gets different regression values from the same PNet output location.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2
from cpp_mtcnn_detector import CPPMTCNNDetector

print("=" * 80)
print("COMPARING PNET REGRESSION VALUES: Python vs C++")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

print(f"\nTest image: {img_w}×{img_h}")
print(f"C++ Gold Standard: x=331, y=753, w=368, h=423")

# Run Python PNet on first scale
py_detector = PurePythonMTCNN_V2()

min_face_size = 40
factor = 0.709
m = 12.0 / min_face_size
scale = m  # First scale

hs = int(np.ceil(img_h * scale))
ws = int(np.ceil(img_w * scale))
img_scaled = cv2.resize(img_float, (ws, hs))
img_data = py_detector._preprocess(img_scaled)

output = py_detector._run_pnet(img_data)
output = output[0].transpose(1, 2, 0)  # (H, W, 6)

print(f"\nPython PNet output shape: {output.shape}")
print(f"Channels: {output.shape[2]}")

# The problematic box came from around (333, 703) in original image
# At scale=0.3, this maps to approximately (333*0.3, 703*0.3) = (100, 211) in scaled image
# Then with stride=2 and cellsize=12, the heatmap position is approximately:
# x_heatmap = (100 * scale - 1) / 2 ≈ 50
# y_heatmap = (211 * scale - 1) / 2 ≈ 105

# Actually, let's reverse the formula:
# The box at (333, 703) came from PNet at this scale
# Original formula: x1 = (stride * heatmap_x + 1) / scale
# So: heatmap_x = (x1 * scale - 1) / stride
heatmap_x = int((333 * scale - 1) / 2)
heatmap_y = int((703 * scale - 1) / 2)

print(f"\nEstimated heatmap position for box at (333, 703): ({heatmap_x}, {heatmap_y})")
print(f"Heatmap size: {output.shape[1]}×{output.shape[0]}")

# Extract regression values at this location
if 0 <= heatmap_y < output.shape[0] and 0 <= heatmap_x < output.shape[1]:
    dx1_py = output[heatmap_y, heatmap_x, 2]
    dy1_py = output[heatmap_y, heatmap_x, 3]
    dx2_py = output[heatmap_y, heatmap_x, 4]
    dy2_py = output[heatmap_y, heatmap_x, 5]

    prob_py = 1.0 / (1.0 + np.exp(output[heatmap_y, heatmap_x, 0] - output[heatmap_y, heatmap_x, 1]))

    print(f"\nPython PNet regression values at ({heatmap_x}, {heatmap_y}):")
    print(f"  dx1: {dx1_py:.6f}")
    print(f"  dy1: {dy1_py:.6f}")
    print(f"  dx2: {dx2_py:.6f}")
    print(f"  dy2: {dy2_py:.6f}")
    print(f"  prob: {prob_py:.6f}")

    # Compute what this would do to the box
    raw_box_x = int((2 * heatmap_x + 1) / scale)
    raw_box_y = int((2 * heatmap_y + 1) / scale)
    raw_box_w = int(12 / scale)
    raw_box_h = int(12 / scale)

    print(f"\nRaw box from this heatmap position:")
    print(f"  Position: ({raw_box_x}, {raw_box_y})")
    print(f"  Size: {raw_box_w}×{raw_box_h}")

    # Apply regression
    new_y1 = raw_box_y + dy1_py * raw_box_h
    new_y2 = raw_box_y + raw_box_h + dy2_py * raw_box_h

    print(f"\nAfter regression:")
    print(f"  y1: {raw_box_y} → {new_y1:.1f} (Δ{new_y1-raw_box_y:+.1f})")
    print(f"  y2: {raw_box_y+raw_box_h} → {new_y2:.1f} (Δ{new_y2-(raw_box_y+raw_box_h):+.1f})")

    if dy1_py < 0:
        print(f"\n⚠ PROBLEM: dy1={dy1_py:.4f} is NEGATIVE, moves box UP")
        print(f"           But box needs to move DOWN to reach y=753!")
else:
    print(f"\n⚠ Heatmap position ({heatmap_x}, {heatmap_y}) is out of bounds!")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("""
The raw box at y=703 needs to move DOWN (+50px) to reach gold standard y=753.
But dy1 is NEGATIVE, which moves the box UP.

Possible causes:
1. PNet regression values are correct but FORMULA is wrong?
2. PNet regression values have opposite sign from what C++ expects?
3. There's a coordinate system flip somewhere?

Next step: Check if C++ has the same regression formula or if there's a sign flip.
""")
