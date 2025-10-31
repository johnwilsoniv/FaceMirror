#!/usr/bin/env python3
"""
Diagnose why frame 2 has r~0.97 while frame 1 has r=1.0
"""

import numpy as np
import cv2
import pyfhog
from pathlib import Path
from openface22_hog_parser import OF22HOGParser
import struct

# Load OpenFace HOG file
hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/pyfhog_validation_output/IMG_0942_left_mirrored.hog"
aligned_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

print("="*80)
print("FRAME 2 DIAGNOSTIC")
print("="*80)

# Parse OpenFace HOG - load first 2 frames
print("\n1. Loading OpenFace HOG (frames 1-2)...")
with open(hog_file, 'rb') as f:
    header_bytes = f.read(12)
    num_cols, num_rows, num_chan = struct.unpack('<iii', header_bytes)
    print(f"   OpenFace .hog header: cols={num_cols}, rows={num_rows}, chan={num_chan}")

    n_features = num_rows * num_cols * num_chan

    # Read frame 1
    n_values = 1 + n_features
    data_bytes = f.read(n_values * 4)
    frame1_data = np.frombuffer(data_bytes, dtype=np.float32)
    frame1_idx = frame1_data[0]
    openface_hog1 = frame1_data[1:]
    print(f"   Frame 1 index: {frame1_idx}, HOG shape: {openface_hog1.shape}")

    # Read frame 2
    data_bytes = f.read(n_values * 4)
    frame2_data = np.frombuffer(data_bytes, dtype=np.float32)
    frame2_idx = frame2_data[0]
    openface_hog2 = frame2_data[1:]
    print(f"   Frame 2 index: {frame2_idx}, HOG shape: {openface_hog2.shape}")

# Load aligned faces and extract with pyfhog
print("\n2. Extracting HOG with pyfhog (same frames)...")

# Frame 1
img1_path = aligned_dir / "frame_det_00_000001.bmp"
img1 = cv2.imread(str(img1_path))
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
pyfhog_hog1 = pyfhog.extract_fhog_features(img1_rgb, cell_size=8)
print(f"   Frame 1 pyfhog shape: {pyfhog_hog1.shape}")

# Frame 2
img2_path = aligned_dir / "frame_det_00_000001.bmp"  # Note: This is checking if frame naming is wrong
if not img2_path.exists():
    # Try frame 2
    img2_path = aligned_dir / "frame_det_00_000002.bmp"

img2 = cv2.imread(str(img2_path))
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
pyfhog_hog2 = pyfhog.extract_fhog_features(img2_rgb, cell_size=8)
print(f"   Frame 2 pyfhog shape: {pyfhog_hog2.shape}")
print(f"   Frame 2 image path: {img2_path}")

# Compare correlations
print("\n3. Comparing correlations...")
corr1 = np.corrcoef(openface_hog1.flatten(), pyfhog_hog1.flatten())[0, 1]
corr2 = np.corrcoef(openface_hog2.flatten(), pyfhog_hog2.flatten())[0, 1]
print(f"   Frame 1 correlation: r = {corr1:.6f}")
print(f"   Frame 2 correlation: r = {corr2:.6f}")

# Compare images
print("\n4. Comparing aligned face images...")
print(f"   Frame 1 image: mean={img1_rgb.mean():.2f}, std={img1_rgb.std():.2f}, shape={img1_rgb.shape}")
print(f"   Frame 2 image: mean={img2_rgb.mean():.2f}, std={img2_rgb.std():.2f}, shape={img2_rgb.shape}")

# Check if images are identical (debugging - are we loading the same image twice?)
images_identical = np.array_equal(img1_rgb, img2_rgb)
print(f"   Images identical: {images_identical}")

# Compare HOG features
print("\n5. Comparing OpenFace HOG features...")
hog_diff = np.abs(openface_hog1 - openface_hog2)
print(f"   Mean difference between frames: {hog_diff.mean():.6f}")
print(f"   Max difference between frames: {hog_diff.max():.6f}")
print(f"   Frames are identical: {np.array_equal(openface_hog1, openface_hog2)}")

# Statistical analysis
print("\n6. Feature statistics...")
print(f"   Frame 1 OpenFace: mean={openface_hog1.mean():.6f}, std={openface_hog1.std():.6f}")
print(f"   Frame 1 pyfhog:   mean={pyfhog_hog1.mean():.6f}, std={pyfhog_hog1.std():.6f}")
print(f"   Frame 2 OpenFace: mean={openface_hog2.mean():.6f}, std={openface_hog2.std():.6f}")
print(f"   Frame 2 pyfhog:   mean={pyfhog_hog2.mean():.6f}, std={pyfhog_hog2.std():.6f}")

print("\n" + "="*80)
print("ANALYSIS:")
if corr1 > 0.9999 and corr2 < 0.99:
    print("Frame 1 is perfect but frame 2 is degraded.")
    print("Possible causes:")
    print("  1. Frame indexing issue (pyfhog loading wrong frame)")
    print("  2. OpenFace using different alignment for frame 2")
    print("  3. Subtle difference in preprocessing for subsequent frames")
elif images_identical:
    print("WARNING: Frame 1 and frame 2 images are IDENTICAL!")
    print("This suggests the aligned face extraction is not working correctly.")
    print("OpenFace may have reused the same face for both frames.")
else:
    print("Both frames are different as expected.")
    print(f"Frame 1: r={corr1:.6f}, Frame 2: r={corr2:.6f}")
print("="*80)
