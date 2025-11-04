#!/usr/bin/env python3
"""
Test joint optimization convergence (params + scale + translation).
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')

print("="*80)
print("JOINT OPTIMIZATION CONVERGENCE TEST")
print("="*80)

MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
CPP_LANDMARKS = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

# Load C++ landmarks
cpp_landmarks = np.load(CPP_LANDMARKS)

# Load PDM
from pyfacelm.clnf.pdm import PointDistributionModel

pdm_path = Path(MODEL_DIR) / "pdms" / "In-the-wild_aligned_PDM_68.txt"
pdm = PointDistributionModel(pdm_path)

# Just call the function and see what params it returns
params, scale, translation = pdm.landmarks_to_params_2d(cpp_landmarks, max_iterations=10)

print(f"\nFinal params (first 10): {params[:10]}")
print(f"Param magnitudes: min={params.min():.2f}, max={params.max():.2f}, mean={np.abs(params).mean():.2f}")
print(f"Final scale: {scale:.6f}")
print(f"Final translation: ({translation[0]:.2f}, {translation[1]:.2f})")

# Reconstruct
reconstructed = pdm.params_to_landmarks_2d(params, scale, translation)
error = np.linalg.norm(reconstructed - cpp_landmarks, axis=1)

print(f"\nReconstruction error: {np.mean(error):.2f} pixels")

# Clamp and check
params_clamped = pdm.clamp_params(params, n_std=3.0)
num_clamped = np.sum(np.abs(params - params_clamped) > 0)
print(f"\nParams clamped: {num_clamped}/{len(params)}")

if num_clamped == len(params):
    print(f"⚠️  ALL params are being clamped! Optimization went wrong.")
    print(f"\nEigenvalue ranges (sqrt for std dev limits):")
    for i in range(min(10, len(pdm.eigenvalues))):
        limit = 3.0 * np.sqrt(pdm.eigenvalues[i])
        print(f"  Param {i}: limit = ±{limit:.2f}, actual = {params[i]:.2f}, clamped = {params_clamped[i]:.2f}")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
