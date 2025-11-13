#!/usr/bin/env python3
"""
Compare Pure Python CNN RNet vs ONNX RNet on the same input.
"""

import cv2
import numpy as np
from pure_python_mtcnn_v2 import PurePythonMTCNN_V2
from cpp_mtcnn_detector import CPPMTCNNDetector

# Load image
img = cv2.imread('calibration_frames/patient1_frame1.jpg')
img_h, img_w = img.shape[:2]
img_float = img.astype(np.float32)

# Test bbox from PNet (one that got score 0.6943 in Pure Python RNet)
test_bbox = np.array([405.0, 1610.0, 462.0, 1667.0, 0.99])

# Extract face crop using same logic
width_target = int(test_bbox[2] - test_bbox[0] + 1)
height_target = int(test_bbox[3] - test_bbox[1] + 1)

start_x_in = max(int(test_bbox[0] - 1), 0)
start_y_in = max(int(test_bbox[1] - 1), 0)
end_x_in = min(int(test_bbox[0] + width_target - 1), img_w)
end_y_in = min(int(test_bbox[1] + height_target - 1), img_h)

start_x_out = max(int(-test_bbox[0] + 1), 0)
start_y_out = max(int(-test_bbox[1] + 1), 0)
end_x_out = min(int(width_target - (test_bbox[0] + (test_bbox[2] - test_bbox[0]) - img_w)), width_target)
end_y_out = min(int(height_target - (test_bbox[1] + (test_bbox[3] - test_bbox[1]) - img_h)), height_target)

face = np.zeros((height_target, width_target, 3), dtype=np.float32)
face[start_y_out:end_y_out, start_x_out:end_x_out] = \
    img_float[start_y_in:end_y_in, start_x_in:end_x_in]

face_24 = cv2.resize(face, (24, 24))

print(f"Test face crop:")
print(f"  Size: {face_24.shape}")
print(f"  Range: [{face_24.min():.1f}, {face_24.max():.1f}]")

# Preprocess for Pure Python CNN (no batch)
face_pp = (face_24 - 127.5) * 0.0078125
face_pp = np.transpose(face_pp, (2, 0, 1))  # (3, 24, 24)

print(f"\nPreprocessed (Pure Python):")
print(f"  Shape: {face_pp.shape}")
print(f"  Range: [{face_pp.min():.3f}, {face_pp.max():.3f}]")

# Preprocess for ONNX (with batch)
face_onnx = (face_24 - 127.5) * 0.0078125
face_onnx = np.transpose(face_onnx, (2, 0, 1))  # (3, 24, 24)
face_onnx = face_onnx[np.newaxis, :, :, :]  # (1, 3, 24, 24)

print(f"\nPreprocessed (ONNX):")
print(f"  Shape: {face_onnx.shape}")
print(f"  Range: [{face_onnx.min():.3f}, {face_onnx.max():.3f}]")

# Run Pure Python RNet
print(f"\n{'='*80}")
print(f"PURE PYTHON CNN RNet")
print(f"{'='*80}")

pp_detector = PurePythonMTCNN_V2()
pp_outputs = pp_detector.rnet(face_pp)
pp_output = pp_outputs[-1]

print(f"Output: {pp_output}")
print(f"Shape: {pp_output.shape}")

# Calculate score
pp_score = 1.0 / (1.0 + np.exp(pp_output[0] - pp_output[1]))
print(f"Score: {pp_score:.4f}")
print(f"Regression: {pp_output[2:6]}")

# Run ONNX RNet
print(f"\n{'='*80}")
print(f"ONNX RNet")
print(f"{'='*80}")

onnx_detector = CPPMTCNNDetector()
onnx_output_list = onnx_detector.rnet.run(None, {'input': face_onnx})
onnx_output = onnx_output_list[0][0]  # Remove batch dimension

print(f"Output: {onnx_output}")
print(f"Shape: {onnx_output.shape}")

# Calculate score
onnx_score = 1.0 / (1.0 + np.exp(onnx_output[0] - onnx_output[1]))
print(f"Score: {onnx_score:.4f}")
print(f"Regression: {onnx_output[2:6]}")

# Compare
print(f"\n{'='*80}")
print(f"COMPARISON")
print(f"{'='*80}")
print(f"Score difference: {abs(pp_score - onnx_score):.6f}")
print(f"Output difference (L2): {np.linalg.norm(pp_output - onnx_output):.6f}")
print(f"Output difference (max): {np.abs(pp_output - onnx_output).max():.6f}")

if abs(pp_score - onnx_score) < 0.01:
    print(f"\n✓ Scores match!")
else:
    print(f"\n❌ Scores differ significantly")
    print(f"\nElement-wise differences:")
    for i in range(6):
        print(f"  Element {i}: PP={pp_output[i]:.6f}, ONNX={onnx_output[i]:.6f}, diff={pp_output[i] - onnx_output[i]:.6f}")
