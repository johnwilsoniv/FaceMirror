#!/usr/bin/env python3
"""Debug pyfhog vs C++ - check for BGR/RGB issues."""

import numpy as np
import cv2
import os

print("=" * 60)
print("PYFHOG DEBUG TEST")
print("=" * 60)

# Load C++ exported HOG features
cpp_hog_path = '/tmp/cpp_hog_features_frame1.txt'
cpp_hog = []
with open(cpp_hog_path, 'r') as f:
    for line in f:
        if line.startswith('hog_'):
            val = float(line.split('=')[1])
            cpp_hog.append(val)
cpp_hog = np.array(cpp_hog)
print(f"C++ HOG: {len(cpp_hog)} features")

# Load aligned face
aligned_path = '/tmp/cpp_aligned_frame1.png'
aligned_face = cv2.imread(aligned_path)
print(f"Aligned face shape: {aligned_face.shape}")
print(f"Aligned face dtype: {aligned_face.dtype}")
print(f"Aligned face range: [{aligned_face.min()}, {aligned_face.max()}]")
print(f"Channel means: B={aligned_face[:,:,0].mean():.1f}, G={aligned_face[:,:,1].mean():.1f}, R={aligned_face[:,:,2].mean():.1f}")

import pyfhog

# Test 1: Standard BGR (OpenCV default)
print("\n--- Test 1: BGR (standard) ---")
py_hog_bgr = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
corr_bgr = np.corrcoef(cpp_hog, py_hog_bgr)[0, 1]
print(f"Correlation: {corr_bgr:.4f}")

# Test 2: RGB (swap channels)
print("\n--- Test 2: RGB (swapped) ---")
aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
py_hog_rgb = pyfhog.extract_fhog_features(aligned_rgb, cell_size=8)
corr_rgb = np.corrcoef(cpp_hog, py_hog_rgb)[0, 1]
print(f"Correlation: {corr_rgb:.4f}")

# Test 3: Grayscale (but expanded to 3 channels)
print("\n--- Test 3: Grayscale->BGR ---")
gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
py_hog_gray = pyfhog.extract_fhog_features(gray_bgr, cell_size=8)
corr_gray = np.corrcoef(cpp_hog, py_hog_gray)[0, 1]
print(f"Correlation: {corr_gray:.4f}")

# Test 4: Check if the aligned face was saved correctly
# Reload as different image formats
print("\n--- Test 4: Image format check ---")
# Check if png was saved with correct color
test_bmp = '/tmp/cpp_0422/IMG_0422_aligned/frame_det_00_000001.bmp'
if os.path.exists(test_bmp):
    aligned_bmp = cv2.imread(test_bmp)
    print(f"BMP shape: {aligned_bmp.shape}")
    print(f"BMP channel means: B={aligned_bmp[:,:,0].mean():.1f}, G={aligned_bmp[:,:,1].mean():.1f}, R={aligned_bmp[:,:,2].mean():.1f}")

    py_hog_bmp = pyfhog.extract_fhog_features(aligned_bmp, cell_size=8)
    corr_bmp = np.corrcoef(cpp_hog, py_hog_bmp)[0, 1]
    print(f"BMP correlation: {corr_bmp:.4f}")

    # Check if images are the same
    diff_imgs = np.abs(aligned_face.astype(float) - aligned_bmp.astype(float))
    print(f"PNG vs BMP diff: mean={diff_imgs.mean():.4f}, max={diff_imgs.max():.4f}")
else:
    print(f"BMP file not found: {test_bmp}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"BGR correlation:  {corr_bgr:.4f}")
print(f"RGB correlation:  {corr_rgb:.4f}")
print(f"Gray correlation: {corr_gray:.4f}")

best = max(corr_bgr, corr_rgb, corr_gray)
if best > 0.99:
    print(f"\n✅ Best correlation {best:.4f} - pyfhog works correctly")
else:
    print(f"\n❌ Best correlation only {best:.4f} - issue remains")
