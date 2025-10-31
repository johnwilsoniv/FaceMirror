#!/usr/bin/env python3
"""
Final test: Does ONNX preprocessing match PyTorch EXACTLY with the fix applied?
"""

import numpy as np
import cv2
import torch
from torchvision import transforms

# Create realistic test face crop (simulating actual face detection output)
test_face = np.random.randint(50, 200, (250, 250, 3), dtype=np.uint8)  # Realistic face values

print("="*80)
print("FINAL PREPROCESSING TEST")
print("="*80)
print(f"Test input shape: {test_face.shape}, dtype: {test_face.dtype}")
print(f"Value range: [{test_face.min()}, {test_face.max()}]\n")

# ============================================================================
# PYTORCH METHOD (from openface/multitask_model.py)
# ============================================================================
print("1. PYTORCH METHOD")
print("-"*80)

# Exact code from MultitaskPredictor.preprocess
face_rgb_pt = cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB)
transform_pt = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
face_tensor_pt = transform_pt(face_rgb_pt)

print(f"Output shape: {face_tensor_pt.shape}")
print(f"Output dtype: {face_tensor_pt.dtype}")
print(f"Value range: [{face_tensor_pt.min():.6f}, {face_tensor_pt.max():.6f}]")
print(f"Sample pixels: {face_tensor_pt[:,0,0].numpy()}")

# ============================================================================
# ONNX METHOD (fixed code from onnx_mtl_detector.py)
# ============================================================================
print("\n2. ONNX METHOD (FIXED)")
print("-"*80)

# Exact fixed code from ONNXMultitaskPredictor.preprocess
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
input_size = 224

face_rgb_onnx = cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB)
face_float = face_rgb_onnx.astype(np.float32) / 255.0  # Convert BEFORE resize
face_resized = cv2.resize(face_float, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
face_normalized = (face_resized - mean) / std
face_tensor_onnx = face_normalized.transpose(2, 0, 1)  # HWC -> CHW

print(f"Output shape: {face_tensor_onnx.shape}")
print(f"Output dtype: {face_tensor_onnx.dtype}")
print(f"Value range: [{face_tensor_onnx.min():.6f}, {face_tensor_onnx.max():.6f}]")
print(f"Sample pixels: {face_tensor_onnx[:,0,0]}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n3. COMPARISON")
print("-"*80)

diff = np.abs(face_tensor_pt.numpy() - face_tensor_onnx)
print(f"Absolute differences:")
print(f"  Mean:   {diff.mean():.8f}")
print(f"  Median: {np.median(diff):.8f}")
print(f"  Max:    {diff.max():.8f}")

if diff.max() < 1e-5:
    print("\n✓✓✓ PERFECT - Preprocessing is identical!")
    print("The fix is correct. User needs to regenerate ONNXv2 files with fixed code.")
elif diff.max() < 1e-3:
    print("\n✓✓ EXCELLENT - Only minor numerical precision differences")
    print("The fix is correct. User needs to regenerate ONNXv2 files with fixed code.")
else:
    print("\n✗✗✗ ERROR - Still has significant differences")
    print("The fix is NOT working correctly. Further investigation needed.")
    print(f"\nMax diff location: {np.unravel_index(np.argmax(diff), diff.shape)}")
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"  PyTorch: {face_tensor_pt.numpy()[max_idx]:.6f}")
    print(f"  ONNX:    {face_tensor_onnx[max_idx]:.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if diff.max() < 1e-3:
    print("\nThe ONNX preprocessing fix is CORRECT.")
    print("\nThe ONNXv2 test files were likely generated BEFORE the fix.")
    print("User should:")
    print("  1. Verify the fixed code is in onnx_mtl_detector.py")
    print("  2. Clear Python cache")
    print("  3. Re-run S1 Face Mirror to generate new ONNXv3 files")
    print("  4. Compare ONNXv3 vs PyTorch (should show >0.99 correlation)")
else:
    print("\nThe fix needs more work.")

print("="*80)
