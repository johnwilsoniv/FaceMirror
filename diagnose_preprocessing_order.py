"""
Diagnose ONNX Preprocessing Order Bug

Test if preprocessing before vs after resize causes the observed discrepancy.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add pymtcnn to path
sys.path.insert(0, str(Path(__file__).parent / "pymtcnn"))

TEST_IMAGE = "calibration_frames/patient1_frame1.jpg"

print("="*80)
print("Preprocessing Order Diagnostic")
print("="*80)

# Load test image
img = cv2.imread(TEST_IMAGE)
print(f"\nOriginal image shape: {img.shape}")
print(f"Original image dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")

# Test resize at scale 0.5
scale = 0.5
target_h = int(img.shape[0] * scale)
target_w = int(img.shape[1] * scale)

print(f"\nTarget size: {target_h} x {target_w}")

# Method 1: CORRECT - Resize THEN preprocess (CoreML/Base/OldONNX)
print("\n" + "="*80)
print("Method 1: Resize THEN Preprocess (CORRECT - CoreML/Base/Old ONNX)")
print("="*80)

img_resized_first = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
print(f"After resize: shape={img_resized_first.shape}, dtype={img_resized_first.dtype}, range=[{img_resized_first.min()}, {img_resized_first.max()}]")

# Normalize
img_norm_after = (img_resized_first.astype(np.float32) - 127.5) * 0.0078125
print(f"After normalize: shape={img_norm_after.shape}, dtype={img_norm_after.dtype}, range=[{img_norm_after.min():.4f}, {img_norm_after.max():.4f}]")

# BGR -> RGB
img_rgb_after = cv2.cvtColor(img_norm_after, cv2.COLOR_BGR2RGB)
print(f"After BGR→RGB: shape={img_rgb_after.shape}, dtype={img_rgb_after.dtype}, range=[{img_rgb_after.min():.4f}, {img_rgb_after.max():.4f}]")

# Transpose HWC -> CHW
img_final_correct = img_rgb_after.transpose(2, 0, 1)
print(f"After transpose: shape={img_final_correct.shape}")

# Method 2: BROKEN - Preprocess THEN resize (Current ONNX)
print("\n" + "="*80)
print("Method 2: Preprocess THEN Resize (BROKEN - Current ONNX)")
print("="*80)

# BGR -> RGB first
img_rgb_first = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"After BGR→RGB: shape={img_rgb_first.shape}, dtype={img_rgb_first.dtype}, range=[{img_rgb_first.min()}, {img_rgb_first.max()}]")

# Normalize
img_norm_before = (img_rgb_first.astype(np.float32) - 127.5) / 128.0
print(f"After normalize: shape={img_norm_before.shape}, dtype={img_norm_before.dtype}, range=[{img_norm_before.min():.4f}, {img_norm_before.max():.4f}]")

# Resize normalized image
img_resized_after = cv2.resize(img_norm_before, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
print(f"After resize: shape={img_resized_after.shape}, dtype={img_resized_after.dtype}, range=[{img_resized_after.min():.4f}, {img_resized_after.max():.4f}]")

# Transpose HWC -> CHW
img_final_broken = img_resized_after.transpose(2, 0, 1)
print(f"After transpose: shape={img_final_broken.shape}")

# Compare the two methods
print("\n" + "="*80)
print("Comparison")
print("="*80)

diff = np.abs(img_final_correct - img_final_broken)
print(f"\nAbsolute difference:")
print(f"  Mean: {diff.mean():.6f}")
print(f"  Max:  {diff.max():.6f}")
print(f"  Median: {np.median(diff):.6f}")
print(f"  Std:  {diff.std():.6f}")

# Check for overshoot
print(f"\nMethod 1 (correct) range: [{img_final_correct.min():.4f}, {img_final_correct.max():.4f}]")
print(f"Method 2 (broken) range:  [{img_final_broken.min():.4f}, {img_final_broken.max():.4f}]")

if img_final_broken.min() < -1.01 or img_final_broken.max() > 1.01:
    print("\n⚠ WARNING: Method 2 has values outside [-1, 1] range due to interpolation!")
    overshoot_pixels = ((img_final_broken < -1.01) | (img_final_broken > 1.01)).sum()
    total_pixels = img_final_broken.size
    print(f"   Overshoot pixels: {overshoot_pixels} / {total_pixels} ({100*overshoot_pixels/total_pixels:.2f}%)")

# Sample pixel comparison
print("\n" + "="*80)
print("Sample Pixel Values (center region)")
print("="*80)

cy, cx = target_h // 2, target_w // 2
for i in range(3):  # RGB channels
    print(f"\nChannel {i} (center pixel):")
    print(f"  Method 1 (correct): {img_final_correct[i, cy, cx]:.6f}")
    print(f"  Method 2 (broken):  {img_final_broken[i, cy, cx]:.6f}")
    print(f"  Difference:         {abs(img_final_correct[i, cy, cx] - img_final_broken[i, cy, cx]):.6f}")

print("\n" + "="*80)
print("Conclusion")
print("="*80)

if diff.max() > 0.01:
    print("\n✗ SIGNIFICANT DIFFERENCE DETECTED!")
    print("  The preprocessing order matters significantly.")
    print("  Current ONNX backend (Method 2) produces different results than CoreML/Base.")
    print("  This explains the 42.86% IoU failure!")
else:
    print("\n✓ Methods produce similar results")
    print("  Preprocessing order is not the root cause.")
