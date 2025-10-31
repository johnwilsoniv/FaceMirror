#!/usr/bin/env python3
"""
Test if PyTorch and ONNX preprocessing produce identical outputs
"""

import numpy as np
import cv2
import torch
from torchvision import transforms

# Create test image
test_face = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

print("="*80)
print("PREPROCESSING COMPARISON: PyTorch vs ONNX")
print("="*80)

# ============================================================================
# PYTORCH PREPROCESSING (as used in openface/multitask_model.py)
# ============================================================================
print("\n1. PYTORCH PREPROCESSING (actual OpenFace code)")
print("-"*80)

# Convert BGR to RGB
face_rgb_pt = cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB)
print(f"After BGR->RGB: shape={face_rgb_pt.shape}, dtype={face_rgb_pt.dtype}")

# Apply transforms (as in MultitaskPredictor)
transform_pt = transforms.Compose([
    transforms.ToTensor(),  # Converts HWC uint8 [0,255] -> CHW float [0,1]
    transforms.Resize((224, 224)),  # Resizes the CHW tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_tensor_pt = transform_pt(face_rgb_pt)
print(f"After transforms: shape={face_tensor_pt.shape}, dtype={face_tensor_pt.dtype}")
print(f"  Value range: [{face_tensor_pt.min():.4f}, {face_tensor_pt.max():.4f}]")

pytorch_output = face_tensor_pt.numpy()

# ============================================================================
# ONNX PREPROCESSING (current implementation)
# ============================================================================
print("\n2. ONNX PREPROCESSING (current fix)")
print("-"*80)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Convert BGR to RGB
face_rgb_onnx = cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB)

# Convert to float BEFORE resize (current fix)
face_float = face_rgb_onnx.astype(np.float32) / 255.0

# Resize on float32
face_resized = cv2.resize(face_float, (224, 224), interpolation=cv2.INTER_LINEAR)

# Normalize
face_normalized = (face_resized - mean) / std

# Convert to CHW
face_tensor_onnx = face_normalized.transpose(2, 0, 1)

print(f"After preprocessing: shape={face_tensor_onnx.shape}, dtype={face_tensor_onnx.dtype}")
print(f"  Value range: [{face_tensor_onnx.min():.4f}, {face_tensor_onnx.max():.4f}]")

onnx_output = face_tensor_onnx

# ============================================================================
# COMPARISON
# ============================================================================
print("\n3. COMPARISON")
print("-"*80)

abs_diff = np.abs(pytorch_output - onnx_output)
print(f"Absolute differences:")
print(f"  Mean: {abs_diff.mean():.8f}")
print(f"  Median: {np.median(abs_diff):.8f}")
print(f"  Max: {abs_diff.max():.8f}")

if abs_diff.max() < 1e-5:
    print("\n✓ PERFECT MATCH - Preprocessing is identical")
elif abs_diff.max() < 1e-3:
    print("\n✓ EXCELLENT - Minor numerical precision differences only")
elif abs_diff.max() < 0.01:
    print("\n⚠ GOOD - Small differences (should not affect outputs much)")
else:
    print("\n✗ POOR - Significant preprocessing differences detected")
    print("\nInvestigating the difference...")

    # Show where the biggest differences are
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(f"  Largest diff at position {max_idx}:")
    print(f"    PyTorch: {pytorch_output[max_idx]:.6f}")
    print(f"    ONNX: {onnx_output[max_idx]:.6f}")
    print(f"    Difference: {abs_diff[max_idx]:.6f}")

print("\n" + "="*80)
