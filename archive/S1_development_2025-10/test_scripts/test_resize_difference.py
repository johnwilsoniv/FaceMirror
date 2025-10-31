#!/usr/bin/env python3
"""
Deep dive into Resize operation differences
"""

import numpy as np
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F

# Create a simple test image with clear patterns
test_image = np.array([
    [[100, 150, 200], [110, 160, 210]],
    [[120, 170, 220], [130, 180, 230]]
], dtype=np.uint8)

print("="*80)
print("RESIZE OPERATION COMPARISON")
print("="*80)
print(f"Input image shape: {test_image.shape}")
print(f"Input values:\n{test_image}")

# ============================================================================
# METHOD 1: PyTorch transforms.Resize (what the library uses)
# ============================================================================
print("\n1. PYTORCH TRANSFORMS.RESIZE")
print("-" * 80)

# Convert to tensor first (as PyTorch does)
tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # CHW, [0, 1]
print(f"After ToTensor: shape={tensor.shape}, range=[{tensor.min():.4f}, {tensor.max():.4f}]")

# Use transforms.Resize
resize_transform = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
resized_pytorch_transforms = resize_transform(tensor)
print(f"After transforms.Resize: shape={resized_pytorch_transforms.shape}")
print(f"  First pixel values: {resized_pytorch_transforms[:, 0, 0]}")

# ============================================================================
# METHOD 2: PyTorch F.interpolate (lower level)
# ============================================================================
print("\n2. PYTORCH F.INTERPOLATE")
print("-" * 80)

tensor_batch = tensor.unsqueeze(0)  # Add batch dimension
resized_pytorch_interpolate = F.interpolate(tensor_batch, size=(224, 224), mode='bilinear', align_corners=False)
resized_pytorch_interpolate = resized_pytorch_interpolate.squeeze(0)
print(f"After F.interpolate (align_corners=False): shape={resized_pytorch_interpolate.shape}")
print(f"  First pixel values: {resized_pytorch_interpolate[:, 0, 0]}")

# Try with align_corners=True
resized_pytorch_align = F.interpolate(tensor_batch, size=(224, 224), mode='bilinear', align_corners=True)
resized_pytorch_align = resized_pytorch_align.squeeze(0)
print(f"After F.interpolate (align_corners=True): shape={resized_pytorch_align.shape}")
print(f"  First pixel values: {resized_pytorch_align[:, 0, 0]}")

# ============================================================================
# METHOD 3: OpenCV resize on uint8 (current ONNX method)
# ============================================================================
print("\n3. OPENCV RESIZE ON UINT8 (current ONNX)")
print("-" * 80)

resized_cv2_uint8 = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_LINEAR)
resized_cv2_uint8_float = resized_cv2_uint8.astype(np.float32) / 255.0
resized_cv2_uint8_chw = resized_cv2_uint8_float.transpose(2, 0, 1)
print(f"After cv2.resize(uint8): shape={resized_cv2_uint8_chw.shape}")
print(f"  First pixel values: {resized_cv2_uint8_chw[:, 0, 0]}")

# ============================================================================
# METHOD 4: OpenCV resize on float32 (potential fix)
# ============================================================================
print("\n4. OPENCV RESIZE ON FLOAT32")
print("-" * 80)

test_image_float = test_image.astype(np.float32) / 255.0
resized_cv2_float = cv2.resize(test_image_float, (224, 224), interpolation=cv2.INTER_LINEAR)
resized_cv2_float_chw = resized_cv2_float.transpose(2, 0, 1)
print(f"After cv2.resize(float32): shape={resized_cv2_float_chw.shape}")
print(f"  First pixel values: {resized_cv2_float_chw[:, 0, 0]}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n5. DIFFERENCES FROM PYTORCH TRANSFORMS.RESIZE")
print("-" * 80)

pytorch_ref = resized_pytorch_transforms.numpy()

methods = [
    ("F.interpolate(align_corners=False)", resized_pytorch_interpolate.numpy()),
    ("F.interpolate(align_corners=True)", resized_pytorch_align.numpy()),
    ("cv2.resize on uint8", resized_cv2_uint8_chw),
    ("cv2.resize on float32", resized_cv2_float_chw),
]

for name, result in methods:
    diff = np.abs(pytorch_ref - result)
    print(f"\n{name}:")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff: {diff.max():.6f}")
    print(f"  First pixel diff: {np.abs(pytorch_ref[:, 0, 0] - result[:, 0, 0])}")

    if np.allclose(pytorch_ref, result, rtol=1e-5, atol=1e-6):
        print(f"  ✓ MATCHES PyTorch transforms.Resize")
    else:
        print(f"  ✗ Does NOT match")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nThe issue is that PyTorch's transforms.Resize uses different")
print("interpolation logic than cv2.resize, even when both use 'bilinear'.")
print("\nPyTorch uses align_corners=False and antialias=True by default,")
print("which gives different results than OpenCV's INTER_LINEAR.")
