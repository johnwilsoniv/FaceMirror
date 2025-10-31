#!/usr/bin/env python3
"""Quick single-frame test of CalcParams fixes"""

import cv2
import pandas as pd
import numpy as np
import warnings

from pdm_parser import PDMParser
from calc_params import CalcParams

# Capture warnings
warnings.simplefilter("always")
warning_count = 0

def warning_handler(message, category, filename, lineno, file=None, line=None):
    global warning_count
    if "Ill-conditioned matrix" in str(message):
        warning_count += 1

warnings.showwarning = warning_handler

print("="*80)
print("QUICK CalcParams Fix Validation - Single Frame")
print("="*80)

# Load components
VIDEO_PATH = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4"
CSV_PATH = "of22_validation/IMG_0942_left_mirrored.csv"
PDM_PATH = "In-the-wild_aligned_PDM_68.txt"

print("\n1. Loading PDM and CalcParams...")
pdm = PDMParser(PDM_PATH)
print(f"  PDM loaded: mean_shape={pdm.mean_shape.shape}, princ_comp={pdm.princ_comp.shape}")

calc_params = CalcParams(pdm)
print(f"  CalcParams initialized")

# Load first frame
print("\n2. Loading test frame...")
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
df = pd.read_csv(CSV_PATH)
cap.release()

# Get landmarks
landmarks_2d = np.zeros((68, 2), dtype=np.float32)
for i in range(68):
    landmarks_2d[i, 0] = df.iloc[0][f'x_{i}']
    landmarks_2d[i, 1] = df.iloc[0][f'y_{i}']

print(f"  Frame loaded: {frame.shape}")
print(f"  Landmarks loaded: {landmarks_2d.shape}")

# Test CalcParams with warning counting
print("\n3. Running CalcParams with fixes...")
warning_count = 0

pose_params, shape_params = calc_params.calc_params(landmarks_2d)

print(f"\n4. Results:")
print(f"  Pose params: {pose_params}")
print(f"  Shape params: {shape_params[:5]}... (showing first 5)")
print(f"  Ill-conditioned matrix warnings: {warning_count}")

# Verify PDM state is not corrupted
print(f"\n5. PDM state check:")
print(f"  mean_shape shape: {pdm.mean_shape.shape} (should be (204, 1))")
print(f"  princ_comp shape: {pdm.princ_comp.shape} (should be (204, 34))")

if pdm.mean_shape.shape != (204, 1):
    print(f"  ❌ PDM STATE CORRUPTED!")
else:
    print(f"  ✅ PDM state intact")

print(f"\n6. Summary:")
if warning_count == 0:
    print(f"  ✅ SUCCESS: No ill-conditioned matrix warnings!")
    print(f"  Fixes are working!")
elif warning_count < 10:
    print(f"  ⚠️  PARTIAL: Only {warning_count} warnings (better than ~1100 before)")
else:
    print(f"  ❌ FAILED: Still {warning_count} warnings")

print("\n" + "="*80)
