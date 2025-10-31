#!/usr/bin/env python3
"""
Quick test: Verify PyFHOG works and masking doesn't black out eyes/mouth
"""

import sys
sys.path.insert(0, '../pyfhog/src')  # Add pyfhog to path

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from openface22_face_aligner import OpenFace22FaceAligner
from triangulation_parser import TriangulationParser
import pyfhog

print("=" * 80)
print("Testing PyFHOG and Masking")
print("=" * 80)

# Load components
print("\n1. Loading components...")
aligner = OpenFace22FaceAligner('In-the-wild_aligned_PDM_68.txt')
triangulation = TriangulationParser('tris_68_full.txt')

# Load test data
print("\n2. Loading test frame...")
df = pd.read_csv('of22_validation/IMG_0942_left_mirrored.csv')
video_path = '/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4'
cap = cv2.VideoCapture(video_path)

# Test frame 493 (neutral, eyes open)
frame_num = 493
row = df[df['frame'] == frame_num].iloc[0]

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
ret, frame = cap.read()

x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
x = row[x_cols].values.astype(np.float32)
y = row[y_cols].values.astype(np.float32)
landmarks = np.stack([x, y], axis=1)

# Test masking
print("\n3. Testing masking...")
aligned_with_mask = aligner.align_face(
    frame, landmarks,
    row['p_tx'], row['p_ty'], row['p_rz'],
    apply_mask=True, triangulation=triangulation
)

aligned_without_mask = aligner.align_face(
    frame, landmarks,
    row['p_tx'], row['p_ty'], row['p_rz'],
    apply_mask=False
)

# Save comparison
comparison = np.hstack([aligned_without_mask, aligned_with_mask])
cv2.imwrite('test_masking_eyes_mouth.png', comparison)

print(f"  ✓ Saved: test_masking_eyes_mouth.png")
print(f"    Left: No mask | Right: With mask")
print(f"    Check that eyes and mouth are VISIBLE in masked version")

# Test PyFHOG
print("\n4. Testing PyFHOG...")
try:
    # Convert to grayscale (HOG expects grayscale)
    aligned_gray = cv2.cvtColor(aligned_with_mask, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    hog_features = pyfhog.extract_fhog_features(aligned_gray)

    print(f"  ✓ PyFHOG extraction successful!")
    print(f"    HOG features shape: {hog_features.shape}")
    print(f"    Expected: ~5000-6000 features")
    print(f"    Feature range: [{hog_features.min():.4f}, {hog_features.max():.4f}]")

    # Check if dimensionality is reasonable
    if 5000 <= len(hog_features) <= 6000:
        print(f"  ✓ Feature dimensionality looks correct!")
    else:
        print(f"  ⚠ Warning: Feature count unexpected")

except Exception as e:
    print(f"  ✗ PyFHOG failed: {e}")
    import traceback
    traceback.print_exc()

cap.release()

print("\n" + "=" * 80)
print("Results:")
print("=" * 80)
print("1. Check test_masking_eyes_mouth.png:")
print("   - Eyes should be VISIBLE (not blacked out)")
print("   - Mouth should be VISIBLE (not blacked out)")
print("   - Neck/ears/background should be BLACK")
print("")
print("2. PyFHOG status: Check output above")
print("=" * 80)
