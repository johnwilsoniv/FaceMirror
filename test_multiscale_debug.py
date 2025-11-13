"""
Test multi-scale implementation with debug output.
"""

import numpy as np
import cv2
from pyclnf import CLNF

# Load test image
video_path = "Patient Data/Normal Cohort/IMG_0433.MOV"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

# OpenFace C++ bbox from PYCLNF_SCALE_PROBLEM.md
bbox = (286.5, 686.2, 425.8, 412.6)

print("=" * 60)
print("Multi-Scale CLNF Test")
print("=" * 60)
print(f"Input bbox: {bbox}")
print(f"Expected scale: 2.799 (from OpenFace C++)")
print()

# Initialize CLNF
clnf = CLNF(model_dir="pyclnf/models")

# Check the scale mapping
print("Window to scale mapping:")
for window_size, scale_idx in clnf.window_to_scale.items():
    patch_scale = clnf.patch_scaling[scale_idx]
    print(f"  Window {window_size} -> scale_idx {scale_idx} -> patch_scale {patch_scale}")
print()

# Manually add debug output by modifying fit temporarily
# Get initial params
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
initial_params = clnf.pdm.init_params(bbox)

print(f"Initial params: scale={initial_params[0]:.3f}, tx={initial_params[4]:.1f}, ty={initial_params[5]:.1f}")
print()

# Now run fit
landmarks, info = clnf.fit(frame, bbox, return_params=True)

print("=" * 60)
print("Final Results")
print("=" * 60)
print(f"Converged: {info['converged']}")
print(f"Iterations: {info['iterations']}")
print(f"Final scale: {info['params'][0]:.3f}")
print(f"Expected scale: 2.799")
print(f"Difference: {info['params'][0] - 2.799:.3f} ({(info['params'][0] / 2.799 - 1) * 100:.1f}%)")
print()
print(f"Landmark Y range: [{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")
print(f"Expected Y range: [691.4, 1087.6]")
