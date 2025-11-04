#!/usr/bin/env python3
"""Test pyfaceau CLNF only (same test as PyfaceLM for comparison)."""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

print("="*80)
print("TESTING PYFACEAU CLNF ONLY")
print("="*80)

# Configuration
MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
CPP_LANDMARKS_FILE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

print(f"\nTest image: {TEST_IMAGE}")
print(f"Model dir: {MODEL_DIR}")

# Load C++ landmarks
cpp_landmarks = np.load(CPP_LANDMARKS_FILE)
print(f"\nLoaded C++ ground truth: {len(cpp_landmarks)} landmarks")

# Import cv2
print("\n" + "="*80)
print("Importing cv2...")
print("="*80)
import cv2
print("✓ cv2 imported")

# Load image
print("\n" + "="*80)
print("Loading image...")
print("="*80)
image = cv2.imread(TEST_IMAGE)
print(f"✓ Image loaded: shape={image.shape}")

# Import pyfaceau CLNF
print("\n" + "="*80)
print("Importing pyfaceau CLNF...")
print("="*80)
from pyfaceau.clnf.clnf_detector import CLNFDetector
print("✓ pyfaceau CLNF imported")

# Initialize detector
print("\n" + "="*80)
print("Initializing pyfaceau CLNFDetector...")
print("="*80)
detector = CLNFDetector(
    model_dir=MODEL_DIR,
    max_iterations=10,
    convergence_threshold=0.01
)
print("✓ Detector initialized")

# Run CLNF refinement
print("\n" + "="*80)
print("Running CLNF refinement...")
print("="*80)
refined_landmarks, converged, num_iterations = detector.refine_landmarks(
    image,
    cpp_landmarks,
    scale_idx=2,  # 0.50 scale
    regularization=0.5,
    multi_scale=True
)

print(f"✓ CLNF refinement completed")
print(f"  Converged: {converged}")
print(f"  Iterations: {num_iterations}")

# Calculate metrics
print("\n" + "="*80)
print("Calculating accuracy metrics...")
print("="*80)

errors = np.linalg.norm(refined_landmarks - cpp_landmarks, axis=1)
mean_error = np.mean(errors)
median_error = np.median(errors)
max_error = np.max(errors)

print(f"\npyfaceau CLNF vs C++ OpenFace:")
print(f"  Mean error:   {mean_error:.2f} pixels")
print(f"  Median error: {median_error:.2f} pixels")
print(f"  Max error:    {max_error:.2f} pixels")

# Interpretation
if mean_error < 2:
    quality = "EXCELLENT (perfect match)"
elif mean_error < 5:
    quality = "GOOD (clinically equivalent)"
elif mean_error < 10:
    quality = "ACCEPTABLE (useful)"
else:
    quality = "POOR (investigate issues)"

print(f"\n  Quality: {quality}")

# Save results
results_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/pyfaceau_clnf_test_results.npz"
np.savez(results_file,
         cpp_landmarks=cpp_landmarks,
         pyfaceau_landmarks=refined_landmarks,
         errors=errors,
         mean_error=mean_error,
         median_error=median_error,
         max_error=max_error,
         converged=converged,
         iterations=num_iterations)

print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*80)
print("PYFACEAU CLNF TEST COMPLETED SUCCESSFULLY!")
print("="*80)
