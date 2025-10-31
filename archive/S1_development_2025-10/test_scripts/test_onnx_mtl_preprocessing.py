#!/usr/bin/env python3
"""
Test ONNX MTL preprocessing using the actual detector class
"""

import numpy as np
import cv2
import torch
from torchvision import transforms
from onnx_mtl_detector import ONNXMultitaskPredictor

# Create test face
test_face = np.random.randint(50, 200, (250, 250, 3), dtype=np.uint8)

print("="*80)
print("TESTING ACTUAL ONNX MTL DETECTOR PREPROCESSING")
print("="*80)

# Initialize ONNX detector
try:
    detector = ONNXMultitaskPredictor('weights/mtl_efficientnet_b0_coreml.onnx')
    print("✓ ONNX detector loaded\n")
except Exception as e:
    print(f"Error loading detector: {e}")
    exit(1)

# ============================================================================
# PYTORCH PREPROCESSING
# ============================================================================
print("1. PYTORCH PREPROCESSING")
print("-"*80)

face_rgb_pt = cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB)
transform_pt = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
face_tensor_pt = transform_pt(face_rgb_pt)

print(f"Output shape: {face_tensor_pt.shape}")
print(f"Value range: [{face_tensor_pt.min():.6f}, {face_tensor_pt.max():.6f}]")
print(f"Sample pixels: {face_tensor_pt[:,0,0].numpy()}")

# ============================================================================
# ONNX PREPROCESSING (actual detector)
# ============================================================================
print("\n2. ONNX PREPROCESSING (from detector.preprocess)")
print("-"*80)

face_tensor_onnx = detector.preprocess(test_face)

print(f"Output shape: {face_tensor_onnx.shape}")
print(f"Value range: [{face_tensor_onnx.min():.6f}, {face_tensor_onnx.max():.6f}]")
print(f"Sample pixels: {face_tensor_onnx[0,:,0,0]}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n3. COMPARISON")
print("-"*80)

# Remove batch dimension from ONNX output for comparison
onnx_no_batch = face_tensor_onnx.squeeze(0)

diff = np.abs(face_tensor_pt.numpy() - onnx_no_batch)
print(f"Absolute differences:")
print(f"  Mean:   {diff.mean():.8f}")
print(f"  Max:    {diff.max():.8f}")

if diff.max() < 1e-5:
    print("\n✓✓✓ PERFECT MATCH - Preprocessing is identical!")
elif diff.max() < 1e-3:
    print("\n✓✓ EXCELLENT - Only minor numerical precision differences")
else:
    print(f"\n✗ POOR - Still has differences (max: {diff.max():.6f})")
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\nMax diff at {max_idx}:")
    print(f"  PyTorch: {face_tensor_pt.numpy()[max_idx]:.6f}")
    print(f"  ONNX:    {onnx_no_batch[max_idx]:.6f}")

print("\n" + "="*80)
