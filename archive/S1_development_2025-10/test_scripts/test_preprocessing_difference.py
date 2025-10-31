#!/usr/bin/env python3
"""
Test preprocessing differences between ONNX and PyTorch implementations
"""

import numpy as np
import cv2
import torch
from torchvision import transforms

# Create a test image
test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

print("="*80)
print("PREPROCESSING COMPARISON: ONNX vs PyTorch")
print("="*80)

# ============================================================================
# PYTORCH MTL PREPROCESSING
# ============================================================================
print("\n1. PYTORCH MTL PREPROCESSING")
print("-" * 80)

# Convert BGR to RGB (as OpenFace does)
face_rgb_pytorch = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
print(f"After BGR->RGB: shape={face_rgb_pytorch.shape}, dtype={face_rgb_pytorch.dtype}")
print(f"  Value range: [{face_rgb_pytorch.min()}, {face_rgb_pytorch.max()}]")

# PyTorch transform
pytorch_transform = transforms.Compose([
    transforms.ToTensor(),  # HWC -> CHW and [0, 255] -> [0, 1]
    transforms.Resize((224, 224)),  # Resize as tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_tensor_pytorch = pytorch_transform(face_rgb_pytorch)
print(f"After ToTensor: shape={face_tensor_pytorch.shape}, dtype={face_tensor_pytorch.dtype}")
print(f"  Value range (before normalize): [0, 1] (implicit)")
print(f"After Resize: shape={face_tensor_pytorch.shape}")
print(f"After Normalize: shape={face_tensor_pytorch.shape}")
print(f"  Value range: [{face_tensor_pytorch.min():.4f}, {face_tensor_pytorch.max():.4f}]")

pytorch_output = face_tensor_pytorch.numpy()

# ============================================================================
# ONNX MTL PREPROCESSING
# ============================================================================
print("\n2. ONNX MTL PREPROCESSING")
print("-" * 80)

# ONNX preprocessing (from onnx_mtl_detector.py)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Convert BGR to RGB
face_rgb_onnx = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
print(f"After BGR->RGB: shape={face_rgb_onnx.shape}, dtype={face_rgb_onnx.dtype}")

# Resize to 224x224 (BEFORE converting to float - this is different!)
face_resized = cv2.resize(face_rgb_onnx, (224, 224), interpolation=cv2.INTER_LINEAR)
print(f"After Resize: shape={face_resized.shape}, dtype={face_resized.dtype}")
print(f"  Value range: [{face_resized.min()}, {face_resized.max()}]")

# Convert to float32 and normalize to [0, 1]
face_float = face_resized.astype(np.float32) / 255.0
print(f"After /255.0: shape={face_float.shape}, dtype={face_float.dtype}")
print(f"  Value range: [{face_float.min():.4f}, {face_float.max():.4f}]")

# Apply ImageNet normalization
face_normalized = (face_float - mean) / std
print(f"After Normalize: shape={face_normalized.shape}")
print(f"  Value range: [{face_normalized.min():.4f}, {face_normalized.max():.4f}]")

# Convert to NCHW format
face_tensor_onnx = face_normalized.transpose(2, 0, 1)
print(f"After transpose: shape={face_tensor_onnx.shape}")

onnx_output = face_tensor_onnx

# ============================================================================
# COMPARISON
# ============================================================================
print("\n3. COMPARISON")
print("-" * 80)

print(f"PyTorch output shape: {pytorch_output.shape}")
print(f"ONNX output shape: {onnx_output.shape}")

# Calculate differences
abs_diff = np.abs(pytorch_output - onnx_output)
rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-10)

print(f"\nAbsolute differences:")
print(f"  Mean: {abs_diff.mean():.6f}")
print(f"  Median: {np.median(abs_diff):.6f}")
print(f"  Max: {abs_diff.max():.6f}")

print(f"\nRelative differences (%):")
print(f"  Mean: {rel_diff.mean() * 100:.2f}%")
print(f"  Median: {np.median(rel_diff) * 100:.2f}%")

# Check if they're identical
if np.allclose(pytorch_output, onnx_output, rtol=1e-5, atol=1e-7):
    print("\n✓ Preprocessing is IDENTICAL (within numerical precision)")
else:
    print("\n✗ Preprocessing has DIFFERENCES")
    print("\nKEY DIFFERENCE IDENTIFIED:")
    print("  PyTorch: ToTensor -> Resize (on float tensor with bilinear)")
    print("  ONNX:    Resize (on uint8 with cv2.INTER_LINEAR) -> ToFloat")
    print("\nThis order difference causes interpolation artifacts!")

# ============================================================================
# TEST WITH ACTUAL TRANSFORM ORDER FIX
# ============================================================================
print("\n4. TESTING ORDER FIX")
print("-" * 80)

# Try ONNX preprocessing with correct order
print("Trying: BGR->RGB -> ToFloat -> Resize -> Normalize")

face_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
face_float = face_rgb.astype(np.float32) / 255.0

# Resize AFTER converting to float (matching PyTorch)
face_resized_fixed = cv2.resize(face_float, (224, 224), interpolation=cv2.INTER_LINEAR)
face_normalized_fixed = (face_resized_fixed - mean) / std
face_tensor_fixed = face_normalized_fixed.transpose(2, 0, 1)

abs_diff_fixed = np.abs(pytorch_output - face_tensor_fixed)
print(f"\nWith order fix:")
print(f"  Absolute difference mean: {abs_diff_fixed.mean():.6f}")
print(f"  Absolute difference max: {abs_diff_fixed.max():.6f}")

if np.allclose(pytorch_output, face_tensor_fixed, rtol=1e-5, atol=1e-7):
    print("  ✓ FIXED! Preprocessing now matches PyTorch")
else:
    print("  ⚠ Still has small differences (may be numerical precision)")

print("\n" + "="*80)
