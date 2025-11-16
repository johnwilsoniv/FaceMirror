#!/usr/bin/env python3
"""
Compare C++ and Python initialization to find divergence point.
"""

import sys
import csv
import numpy as np

# Add pyclnf to path
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

from pathlib import Path
from pyclnf.core.pdm import PDM

# Load PDM
pdm_dir = Path('pyclnf/models/exported_pdm')
pdm = PDM(str(pdm_dir))

# Test bbox (same as test script)
bbox = [310, 780, 370, 370]  # [x, y, width, height]

# Initialize from bbox (matching CLNF.fit())
x, y, w, h = bbox
cx = x + w / 2.0
cy = y + h / 2.0

# Compute scale from bbox size
# OpenFace uses: scale = bbox_size / (reference_frame_width * 0.7)
# For 68-point model, reference frame is typically ~100 units wide
reference_width = 100.0  # Approximate
scale = (w + h) / 2.0 / (reference_width * 0.7)

print("Python initialization from bbox:")
print(f"  Bbox: {bbox}")
print(f"  Center: ({cx:.2f}, {cy:.2f})")
print(f"  Size: {w}x{h}")
print(f"  Computed scale: {scale:.6f}")

# Create initial parameters [scale, rot_x, rot_y, rot_z, tx, ty, q...]
initial_params = pdm.init_params(tuple(bbox))

print(f"\nInitial parameters:")
print(f"  Scale: {initial_params[0]:.6f}")
print(f"  Rotation: ({initial_params[1]:.6f}, {initial_params[2]:.6f}, {initial_params[3]:.6f})")
print(f"  Translation: ({initial_params[4]:.6f}, {initial_params[5]:.6f})")
print(f"  Shape params (first 5): {initial_params[6:11]}")

# Compute initial landmarks
landmarks = pdm.params_to_landmarks_2d(initial_params)

print(f"\nInitial landmarks (tracked):")
for lm_idx in [36, 48, 30, 8]:
    x, y = landmarks[lm_idx]
    print(f"  Landmark_{lm_idx}: ({x:.4f}, {y:.4f})")

# Load C++ CSV to compare final landmarks
print("\n" + "="*60)
print("C++ FINAL LANDMARKS (for comparison)")
print("="*60)

with open('/tmp/cpp_baseline/patient1_frame1.csv', 'r') as f:
    reader = csv.DictReader(f)
    row = next(reader)

    print("C++ final landmarks (tracked):")
    for lm_idx in [36, 48, 30, 8]:
        cpp_x = float(row[f'x_{lm_idx}'])
        cpp_y = float(row[f'y_{lm_idx}'])
        py_x, py_y = landmarks[lm_idx]

        diff_x = py_x - cpp_x
        diff_y = py_y - cpp_y
        dist = np.sqrt(diff_x**2 + diff_y**2)

        print(f"  Landmark_{lm_idx}: C++({cpp_x:.2f}, {cpp_y:.2f}) vs PY_INIT({py_x:.2f}, {py_y:.2f}) diff=({diff_x:+.2f}, {diff_y:+.2f}) dist={dist:.2f}px")

print("\nConclusion:")
print("If Python INIT landmarks match C++ FINAL landmarks closely,")
print("then C++ is also initializing to the same position and the")
print("divergence happens during optimization.")
