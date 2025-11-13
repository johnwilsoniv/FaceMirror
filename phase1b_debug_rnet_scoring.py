#!/usr/bin/env python3
"""
Phase 1B: Debug why RNet scores the face crop as 0.63 instead of >0.7

We found that crop #6 (452×451px at 330,672) is a perfect face but gets 0.6314.
Let's investigate why!
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2

print("=" * 80)
print("PHASE 1B: DEBUG RNET SCORING")
print("=" * 80)

# Load test image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_float = img.astype(np.float32)

# Extract the exact problematic crop
x1, y1, x2, y2 = 330, 672, 782, 1123  # 452×451px
face_crop = img_float[y1:y2, x1:x2]

print(f"\nProblematic crop:")
print(f"  Location: ({x1}, {y1}) to ({x2}, {y2})")
print(f"  Size: {x2-x1}×{y2-y1}px")
print(f"  Expected: Should be the main face, score >0.7")
print(f"  Actual: Scores 0.6314 - gets rejected!")

# Save original crop
cv2.imwrite('debug_rnet/original_crop_452x451.jpg', face_crop.astype(np.uint8))
print(f"\nSaved: debug_rnet/original_crop_452x451.jpg")

# Resize to 24×24 (what RNet sees)
face_24x24 = cv2.resize(face_crop, (24, 24))
cv2.imwrite('debug_rnet/resized_24x24.jpg', face_24x24.astype(np.uint8))
print(f"Saved: debug_rnet/resized_24x24.jpg")

# Create detector
detector = PurePythonMTCNN_V2()

# Preprocess and run RNet
print("\n" + "-" * 80)
print("Running Python RNet...")
print("-" * 80)

face_data = detector._preprocess(face_24x24)
print(f"After preprocessing: {face_data.shape}, range=[{face_data.min():.3f}, {face_data.max():.3f}]")

# Save preprocessed version for inspection
preprocessed_vis = ((face_data.transpose(1, 2, 0) / 0.0078125) + 127.5).astype(np.uint8)
cv2.imwrite('debug_rnet/preprocessed_visualization.jpg', preprocessed_vis)
print(f"Saved: debug_rnet/preprocessed_visualization.jpg")

output = detector._run_rnet(face_data)
print(f"\nRNet output: {output}")
print(f"  Shape: {output.shape}")
print(f"  Values: {output}")

logit_not_face = output[0]
logit_face = output[1]
score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

print(f"\nScore calculation:")
print(f"  logit_not_face: {logit_not_face:.4f}")
print(f"  logit_face: {logit_face:.4f}")
print(f"  difference: {logit_not_face - logit_face:.4f}")
print(f"  exp(diff): {np.exp(logit_not_face - logit_face):.4f}")
print(f"  FINAL SCORE: {score:.4f}")

# Also get bbox regression from RNet
dx1, dy1, dx2, dy2 = output[2:6]
print(f"\nRNet bbox regression: [{dx1:.3f}, {dy1:.3f}, {dx2:.3f}, {dy2:.3f}]")

# Now test the HIGH-scoring small crops
print("\n" + "=" * 80)
print("Compare with HIGH-SCORING small crops")
print("=" * 80)

# Test crop #1: 40×41px at (888, 783) - score 0.8654
print("\nCrop #1 (should score 0.8654):")
x1_small, y1_small = 888, 783
x2_small, y2_small = x1_small + 41, y1_small + 41
small_crop = img_float[y1_small:y2_small, x1_small:x2_small]
small_24x24 = cv2.resize(small_crop, (24, 24))

cv2.imwrite('debug_rnet/small_crop1_original.jpg', small_crop.astype(np.uint8))
cv2.imwrite('debug_rnet/small_crop1_24x24.jpg', small_24x24.astype(np.uint8))

small_data = detector._preprocess(small_24x24)
small_output = detector._run_rnet(small_data)
small_score = 1.0 / (1.0 + np.exp(small_output[0] - small_output[1]))

print(f"  Location: ({x1_small}, {y1_small})")
print(f"  Size: 40×41px")
print(f"  RNet score: {small_score:.4f}")
print(f"  logit_not_face: {small_output[0]:.4f}")
print(f"  logit_face: {small_output[1]:.4f}")

# Visual comparison
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"""
Key Findings:

1. LARGE FACE CROP (452×451px):
   - Visual: Clear, well-framed face
   - RNet score: {score:.4f} < 0.7 threshold
   - Result: REJECTED by RNet

2. SMALL FEATURE CROP (40×41px):
   - Visual: Tight crop (eye/feature)
   - RNet score: {small_score:.4f} > 0.7 threshold
   - Result: ACCEPTED by RNet

Why is RNet scoring backwards?

Possible causes:
A. Resize artifacts: 452→24 creates different patterns than 40→24
B. Preprocessing difference: C++ and Python differ somehow
C. RNet weights: Python loaded weights incorrectly
D. Aspect ratio: Large crop is 452×451 (nearly square after squaring)
E. Image content: Large crop includes background, small crop is tight

Check these files:
- debug_rnet/original_crop_452x451.jpg (the actual face)
- debug_rnet/resized_24x24.jpg (what RNet sees)
- debug_rnet/small_crop1_24x24.jpg (high-scoring feature)

Next steps:
1. Run same crops through C++ RNet for comparison
2. Check if resize method matters (INTER_LINEAR vs others)
3. Verify RNet weights checksum
""")

import os
os.makedirs('debug_rnet', exist_ok=True)
