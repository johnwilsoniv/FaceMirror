#!/usr/bin/env python3
"""
Test PDM round-trip conversion: landmarks -> params -> landmarks
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')

print("="*80)
print("PDM ROUND-TRIP TEST")
print("="*80)

MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
CPP_LANDMARKS = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

# Load C++ landmarks
cpp_landmarks = np.load(CPP_LANDMARKS)
print(f"\nOriginal C++ landmarks (first 5):")
for i in range(5):
    print(f"  Landmark {i}: ({cpp_landmarks[i, 0]:.2f}, {cpp_landmarks[i, 1]:.2f})")

# Load PDM
from pyfacelm.clnf.pdm import PointDistributionModel

pdm_path = Path(MODEL_DIR) / "pdms" / "In-the-wild_aligned_PDM_68.txt"
pdm = PointDistributionModel(pdm_path)

# Test round-trip
print(f"\n{'='*80}")
print("ROUND-TRIP TEST")
print(f"{'='*80}")

# Convert to params
params, scale, translation = pdm.landmarks_to_params_2d(cpp_landmarks)
print(f"\nStep 1: landmarks -> params")
print(f"  Scale: {scale:.6f}")
print(f"  Translation: ({translation[0]:.2f}, {translation[1]:.2f})")
print(f"  Params (first 5): {params[:5]}")

# Clamp params
params_clamped = pdm.clamp_params(params, n_std=3.0)
print(f"\nStep 2: Clamp params")
print(f"  Params changed: {not np.allclose(params, params_clamped)}")
if not np.allclose(params, params_clamped):
    diff = np.abs(params - params_clamped)
    print(f"  Max param change: {diff.max():.6f}")
    print(f"  Num params clamped: {np.sum(diff > 0)}")

# Convert back to landmarks
reconstructed = pdm.params_to_landmarks_2d(params_clamped, scale, translation)

print(f"\nStep 3: params -> landmarks")
print(f"Reconstructed landmarks (first 5):")
for i in range(5):
    print(f"  Landmark {i}: ({reconstructed[i, 0]:.2f}, {reconstructed[i, 1]:.2f})")

# Compare
errors = np.linalg.norm(reconstructed - cpp_landmarks, axis=1)
mean_error = np.mean(errors)
max_error = np.max(errors)
max_idx = np.argmax(errors)

print(f"\n{'='*80}")
print("ROUND-TRIP ERROR ANALYSIS")
print(f"{'='*80}")
print(f"\nMean error: {mean_error:.2f} pixels")
print(f"Max error: {max_error:.2f} pixels at landmark {max_idx}")
print(f"Median error: {np.median(errors):.2f} pixels")

print(f"\nError distribution:")
print(f"  < 1px:  {np.sum(errors < 1)} landmarks")
print(f"  < 5px:  {np.sum(errors < 5)} landmarks")
print(f"  < 10px: {np.sum(errors < 10)} landmarks")
print(f"  > 10px: {np.sum(errors >= 10)} landmarks")

if mean_error > 5:
    print(f"\n⚠️  WARNING: Round-trip error > 5px!")
    print(f"   PDM conversion is introducing significant error")
    print(f"\n   Worst 10 landmarks:")
    worst = np.argsort(errors)[-10:][::-1]
    for idx in worst:
        print(f"     Landmark {idx:2d}: {errors[idx]:6.2f}px")
elif mean_error > 1:
    print(f"\n⚠️  CAUTION: Round-trip error > 1px")
    print(f"   PDM conversion has minor error")
else:
    print(f"\n✓ Round-trip error < 1px - PDM conversion is accurate")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
