#!/usr/bin/env python3
"""Test if pyfhog produces identical output to C++ OpenFace HOG."""

import numpy as np
import cv2
import os

print("=" * 60)
print("PYFHOG vs C++ OPENFACE HOG TEST")
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

# Load the EXACT aligned face that C++ used for HOG
aligned_path = '/tmp/cpp_aligned_face_for_au.png'
if not os.path.exists(aligned_path):
    print(f"ERROR: {aligned_path} not found")
    print("Run OpenFace first to generate it")
    exit(1)

aligned_face = cv2.imread(aligned_path)
print(f"Aligned face: {aligned_path}")
print(f"  Shape: {aligned_face.shape}")
print(f"  dtype: {aligned_face.dtype}")
print(f"  Range: [{aligned_face.min()}, {aligned_face.max()}]")

import pyfhog
py_hog = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
print(f"Python HOG: {len(py_hog)} features")

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\nC++ HOG: min={cpp_hog.min():.4f}, max={cpp_hog.max():.4f}, mean={cpp_hog.mean():.4f}")
print(f"Py  HOG: min={py_hog.min():.4f}, max={py_hog.max():.4f}, mean={py_hog.mean():.4f}")

# Correlation
corr = np.corrcoef(cpp_hog, py_hog)[0, 1]
print(f"\nCorrelation: {corr:.6f}")

# Element-wise difference
diff = np.abs(cpp_hog - py_hog)
print(f"Mean abs diff: {diff.mean():.6f}")
print(f"Max abs diff: {diff.max():.6f}")

# Check if they match
if corr > 0.999:
    print("\n✅ PERFECT MATCH: pyfhog == C++ dlib FHOG")
elif corr > 0.99:
    print("\n✅ MATCH: pyfhog produces same output as C++ (minor float differences)")
elif corr > 0.90:
    print("\n⚠️ CLOSE: Some differences detected")
else:
    print(f"\n❌ MISMATCH: Correlation only {corr:.4f}")

# Debug: check first few values
print("\nFirst 10 values:")
print(f"  C++: {cpp_hog[:10]}")
print(f"  Py:  {py_hog[:10]}")
