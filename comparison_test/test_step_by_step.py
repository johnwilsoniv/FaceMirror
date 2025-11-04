#!/usr/bin/env python3
"""Step-by-step test to isolate issues."""

import sys
import os
import subprocess
import json
import numpy as np

print("="*80)
print("STEP-BY-STEP DIAGNOSTIC TEST")
print("="*80)

# Configuration
OPENFACE_BINARY = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"
MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
OUTPUT_DIR = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_openface/IMG_8401"

print(f"\nTest image: {TEST_IMAGE}")
print(f"Model dir: {MODEL_DIR}")
print(f"OpenFace binary: {OPENFACE_BINARY}")

# Step 1: Run C++ OpenFace
print(f"\n{'='*80}")
print("STEP 1: Running C++ OpenFace")
print(f"{'='*80}")

cmd = [
    OPENFACE_BINARY,
    "-f", TEST_IMAGE,
    "-out_dir", OUTPUT_DIR,
    "-2Dfp"
]

print(f"Command: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"ERROR: OpenFace failed with return code {result.returncode}")
    print(f"STDERR: {result.stderr}")
    sys.exit(1)

print(f"SUCCESS: C++ OpenFace completed")

# Step 2: Parse CSV
print(f"\n{'='*80}")
print("STEP 2: Parsing CSV output")
print(f"{'='*80}")

csv_file = f"{OUTPUT_DIR}/IMG_8401.csv"
print(f"Reading: {csv_file}")

with open(csv_file, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    values = lines[1].strip().split(',')

print(f"CSV has {len(header)} columns")
print(f"First 10 columns: {header[:10]}")

# Parse landmarks
landmarks = []
for i in range(68):
    try:
        x_idx = header.index(f'x_{i}')
        y_idx = header.index(f'y_{i}')
    except ValueError:
        x_idx = header.index(f' x_{i}')
        y_idx = header.index(f' y_{i}')

    x = float(values[x_idx])
    y = float(values[y_idx])
    landmarks.append([x, y])

landmarks = np.array(landmarks)
print(f"SUCCESS: Extracted {len(landmarks)} landmarks")
print(f"Landmark 0 (jaw left): ({landmarks[0][0]:.1f}, {landmarks[0][1]:.1f})")
print(f"Landmark 30 (nose tip): ({landmarks[30][0]:.1f}, {landmarks[30][1]:.1f})")

# Save landmarks
lm_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"
np.save(lm_file, landmarks)
print(f"\nSaved landmarks to: {lm_file}")

# Get confidence
try:
    conf_idx = header.index('confidence')
except ValueError:
    conf_idx = header.index(' confidence')
confidence = float(values[conf_idx])
print(f"Confidence: {confidence:.3f}")

# Step 3: Test loading image WITHOUT cv2 (use PIL instead to avoid segfault)
print(f"\n{'='*80}")
print("STEP 3: Loading image with PIL (avoiding cv2 segfault)")
print(f"{'='*80}")

from PIL import Image
img = Image.open(TEST_IMAGE)
print(f"SUCCESS: Loaded image {img.size[0]}x{img.size[1]}")

print(f"\n{'='*80}")
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print(f"{'='*80}")
print(f"\nC++ OpenFace landmarks saved to: {lm_file}")
print(f"Use these landmarks to test PyfaceLM and Python pipeline separately.")
