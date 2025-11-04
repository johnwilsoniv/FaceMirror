#!/usr/bin/env python3
"""
Test if param clamping is causing the error.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')

print("="*80)
print("PARAM CLAMPING TEST")
print("="*80)

MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
CPP_LANDMARKS = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

# Load C++ landmarks
cpp_landmarks = np.load(CPP_LANDMARKS)

# Load PDM
from pyfacelm.clnf.pdm import PointDistributionModel

pdm_path = Path(MODEL_DIR) / "pdms" / "In-the-wild_aligned_PDM_68.txt"
pdm = PointDistributionModel(pdm_path)

print(f"\n{'='*80}")
print("WITHOUT CLAMPING")
print(f"{'='*80}")

# Convert to params (old norm-based method)
params, scale, translation = pdm.landmarks_to_params_2d(cpp_landmarks)
print(f"\nParams (first 10): {params[:10]}")
print(f"Scale: {scale:.6f}")
print(f"Translation: ({translation[0]:.2f}, {translation[1]:.2f})")

# Reconstruct WITHOUT clamping
reconstructed_unclamped = pdm.params_to_landmarks_2d(params, scale, translation)
error_unclamped = np.linalg.norm(reconstructed_unclamped - cpp_landmarks, axis=1)
print(f"\nMean error (no clamping): {np.mean(error_unclamped):.2f} pixels")
print(f"Max error (no clamping): {np.max(error_unclamped):.2f} pixels")

print(f"\n{'='*80}")
print("WITH CLAMPING (3 std)")
print(f"{'='*80}")

# Clamp and reconstruct
params_clamped = pdm.clamp_params(params, n_std=3.0)
print(f"\nClamped params (first 10): {params_clamped[:10]}")
print(f"Params changed: {not np.allclose(params, params_clamped)}")
diff = np.abs(params - params_clamped)
print(f"Max param change: {diff.max():.6f}")
print(f"Num params clamped: {np.sum(diff > 0)}")

reconstructed_clamped = pdm.params_to_landmarks_2d(params_clamped, scale, translation)
error_clamped = np.linalg.norm(reconstructed_clamped - cpp_landmarks, axis=1)
print(f"\nMean error (with clamping): {np.mean(error_clamped):.2f} pixels")
print(f"Max error (with clamping): {np.max(error_clamped):.2f} pixels")

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

print(f"\nError difference: {np.mean(error_clamped) - np.mean(error_unclamped):.2f} pixels")
print(f"\nIs clamping the problem? {np.mean(error_clamped) > np.mean(error_unclamped) + 10}")

if np.mean(error_unclamped) < 10:
    print(f"\n✓ Round-trip works without clamping!")
    print(f"  The bug is in param clamping or initial estimation.")
elif np.mean(error_unclamped) < 100:
    print(f"\n⚠️  Round-trip has moderate error even without clamping.")
    print(f"   The scale/translation estimation needs improvement.")
else:
    print(f"\n✗ Round-trip has large error even without clamping.")
    print(f"  The landmarks_to_params_2d conversion is fundamentally broken.")

# Show per-landmark errors
print(f"\n{'='*80}")
print("PER-LANDMARK ERRORS (first 10)")
print(f"{'='*80}")
print(f"\n  Idx   Unclamped   Clamped     Difference")
for i in range(10):
    print(f"  {i:2d}     {error_unclamped[i]:7.2f}   {error_clamped[i]:7.2f}   {error_clamped[i] - error_unclamped[i]:7.2f}")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
