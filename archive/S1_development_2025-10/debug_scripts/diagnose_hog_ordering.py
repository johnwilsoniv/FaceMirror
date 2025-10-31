#!/usr/bin/env python3
"""
Diagnose HOG feature ordering between OpenFace and pyfhog

Check if there's a row/column transpose or other ordering issue
"""

import numpy as np
import cv2
import pyfhog
from pathlib import Path
from openface22_hog_parser import OF22HOGParser
import struct

# Load one frame from OpenFace
hog_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/pyfhog_validation_output/IMG_0942_left_mirrored.hog"
aligned_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/pyfhog_validation_output/IMG_0942_left_mirrored_aligned")

print("="*80)
print("HOG ORDERING DIAGNOSTIC")
print("="*80)

# Parse OpenFace HOG - frame 1
print("\n1. Loading OpenFace HOG (frame 1)...")
with open(hog_file, 'rb') as f:
    header_bytes = f.read(12)
    num_cols, num_rows, num_chan = struct.unpack('<iii', header_bytes)
    print(f"   OpenFace .hog header: cols={num_cols}, rows={num_rows}, chan={num_chan}")
    print(f"   Total features: {num_rows * num_cols * num_chan}")

    n_features = num_rows * num_cols * num_chan
    n_values = 1 + n_features
    data_bytes = f.read(n_values * 4)
    frame_data = np.frombuffer(data_bytes, dtype=np.float32)

    frame_idx = frame_data[0]
    openface_hog = frame_data[1:]

    print(f"   Frame index: {frame_idx}")
    print(f"   HOG shape: {openface_hog.shape}")

# Load corresponding aligned face and extract with pyfhog
print("\n2. Extracting HOG with pyfhog (same frame)...")
img_path = aligned_dir / "frame_det_00_000001.bmp"
img = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"   Image shape: {img_rgb.shape}")

pyfhog_hog = pyfhog.extract_fhog_features(img_rgb, cell_size=8)
print(f"   pyfhog output shape: {pyfhog_hog.shape}")

# Try reshaping to see structure
print("\n3. Reshaping features to (rows, cols, channels)...")
openface_reshaped = openface_hog.reshape((num_rows, num_cols, num_chan))
pyfhog_reshaped = pyfhog_hog.reshape((num_rows, num_cols, num_chan))

print(f"   OpenFace reshaped: {openface_reshaped.shape}")
print(f"   pyfhog reshaped: {pyfhog_reshaped.shape}")

# Check if transpose helps
print("\n4. Testing different orderings...")

# Original
corr_original = np.corrcoef(openface_hog.flatten(), pyfhog_hog.flatten())[0, 1]
print(f"   Original ordering: r = {corr_original:.6f}")

# Try transposing rows/cols
pyfhog_transposed = np.transpose(pyfhog_reshaped, (1, 0, 2)).flatten()
corr_transposed = np.corrcoef(openface_hog.flatten(), pyfhog_transposed)[0, 1]
print(f"   Transposed (swap rows↔cols): r = {corr_transposed:.6f}")

# Try different channel ordering
pyfhog_chan_last = np.transpose(pyfhog_reshaped, (0, 1, 2)).flatten()  # Same as original
pyfhog_chan_first = np.transpose(pyfhog_reshaped, (2, 0, 1)).flatten()
corr_chan_first = np.corrcoef(openface_hog.flatten(), pyfhog_chan_first)[0, 1]
print(f"   Channels-first: r = {corr_chan_first:.6f}")

# Try both transpose + channel reorder
pyfhog_both = np.transpose(pyfhog_reshaped, (2, 1, 0)).flatten()
corr_both = np.corrcoef(openface_hog.flatten(), pyfhog_both)[0, 1]
print(f"   Both (transpose + chan): r = {corr_both:.6f}")

# Find best correlation
orderings = [
    ("Original (r,c,ch)", corr_original, pyfhog_hog),
    ("Transpose (c,r,ch)", corr_transposed, pyfhog_transposed),
    ("Channels-first (ch,r,c)", corr_chan_first, pyfhog_chan_first),
    ("Both (ch,c,r)", corr_both, pyfhog_both),
]

best_name, best_corr, best_features = max(orderings, key=lambda x: x[1])

print("\n5. RESULT:")
print(f"   Best ordering: {best_name}")
print(f"   Correlation: r = {best_corr:.6f}")

if best_corr > 0.9999:
    print(f"\n✅ PERFECT! Found the correct ordering!")
    print(f"   The issue was feature ordering, not the FHOG algorithm itself.")
elif best_corr > 0.999:
    print(f"\n✅ EXCELLENT! Very close match.")
else:
    print(f"\n⚠️  Still not perfect. May be other issues.")

# Show first 10 features for manual inspection
print("\n6. First 10 features comparison:")
print(f"   OpenFace: {openface_hog[:10]}")
print(f"   pyfhog (best): {best_features[:10]}")
print(f"   Difference: {np.abs(openface_hog[:10] - best_features[:10])}")
