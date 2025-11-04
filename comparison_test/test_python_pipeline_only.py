#!/usr/bin/env python3
"""Test full Python pipeline only (MTCNN + CLNF)."""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau')

print("="*80)
print("TESTING FULL PYTHON PIPELINE (MTCNN + CLNF)")
print("="*80)

# Configuration
MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
CPP_LANDMARKS_FILE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

print(f"\nTest image: {TEST_IMAGE}")
print(f"Model dir: {MODEL_DIR}")

# Load C++ landmarks (ground truth)
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

# Step 1: MTCNN face detection
print("\n" + "="*80)
print("Step 1: MTCNN face detection...")
print("="*80)

from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN

mtcnn = OpenFaceMTCNN(device='cpu')
print("✓ MTCNN initialized")

bboxes, landmarks_5pt = mtcnn.detect(image)
print(f"✓ Detected {len(bboxes)} face(s)")

if len(bboxes) == 0:
    print("ERROR: No faces detected")
    sys.exit(1)

bbox = bboxes[0]
print(f"  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
print(f"  Confidence: {bbox[4]:.3f}")

# Step 2: Initialize 68 landmarks from bbox
print("\n" + "="*80)
print("Step 2: Initializing 68-point landmarks from bbox...")
print("="*80)

def bbox_to_68_landmarks(bbox):
    """Convert bounding box to rough 68-point landmark initialization."""
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Create a rough face template
    template = np.array([
        # Jaw (0-16)
        [-0.3, 0.3], [-0.28, 0.35], [-0.25, 0.4], [-0.2, 0.45], [-0.15, 0.48],
        [-0.1, 0.5], [-0.05, 0.51], [0, 0.52], [0.05, 0.51], [0.1, 0.5],
        [0.15, 0.48], [0.2, 0.45], [0.25, 0.4], [0.28, 0.35], [0.3, 0.3],
        [0.32, 0.25], [0.33, 0.2],
        # Eyebrows (17-26)
        [-0.25, -0.1], [-0.2, -0.12], [-0.15, -0.13], [-0.1, -0.12], [-0.05, -0.1],
        [0.05, -0.1], [0.1, -0.12], [0.15, -0.13], [0.2, -0.12], [0.25, -0.1],
        # Nose (27-35)
        [0, -0.05], [0, 0.0], [0, 0.05], [0, 0.1],
        [-0.08, 0.12], [-0.04, 0.13], [0, 0.14], [0.04, 0.13], [0.08, 0.12],
        # Eyes (36-47)
        [-0.2, -0.05], [-0.17, -0.07], [-0.14, -0.07], [-0.11, -0.05],
        [-0.14, -0.04], [-0.17, -0.04],
        [0.11, -0.05], [0.14, -0.07], [0.17, -0.07], [0.2, -0.05],
        [0.17, -0.04], [0.14, -0.04],
        # Mouth (48-67)
        [-0.12, 0.25], [-0.09, 0.27], [-0.05, 0.28], [0, 0.285], [0.05, 0.28],
        [0.09, 0.27], [0.12, 0.25], [0.1, 0.28], [0.05, 0.3], [0, 0.305],
        [-0.05, 0.3], [-0.1, 0.28], [-0.08, 0.26], [-0.05, 0.27], [0, 0.275],
        [0.05, 0.27], [0.08, 0.26], [0.05, 0.27], [0, 0.275], [-0.05, 0.27],
    ])

    # Scale and translate
    landmarks = template.copy()
    landmarks[:, 0] = cx + landmarks[:, 0] * w
    landmarks[:, 1] = cy + landmarks[:, 1] * h

    return landmarks

initial_68pt = bbox_to_68_landmarks(bbox)
print(f"✓ Initialized 68 landmarks from bbox")

# Calculate error of MTCNN initialization vs C++
init_errors = np.linalg.norm(initial_68pt - cpp_landmarks, axis=1)
init_mean_error = np.mean(init_errors)
print(f"  MTCNN initialization mean error: {init_mean_error:.2f} pixels")

# Step 3: CLNF refinement
print("\n" + "="*80)
print("Step 3: CLNF refinement...")
print("="*80)

from pyfaceau.clnf.clnf_detector import CLNFDetector

clnf = CLNFDetector(
    model_dir=MODEL_DIR,
    max_iterations=10,
    convergence_threshold=0.01
)
print("✓ CLNF detector initialized")

refined_landmarks, converged, num_iterations = clnf.refine_landmarks(
    image,
    initial_68pt,
    scale_idx=2,
    regularization=0.5,
    multi_scale=True
)

print(f"✓ CLNF refinement completed")
print(f"  Converged: {converged}")
print(f"  Iterations: {num_iterations}")

# Calculate final metrics
print("\n" + "="*80)
print("Final Results...")
print("="*80)

final_errors = np.linalg.norm(refined_landmarks - cpp_landmarks, axis=1)
final_mean_error = np.mean(final_errors)
final_median_error = np.median(final_errors)
final_max_error = np.max(final_errors)

print(f"\nPython Pipeline vs C++ OpenFace:")
print(f"  Mean error:   {final_mean_error:.2f} pixels")
print(f"  Median error: {final_median_error:.2f} pixels")
print(f"  Max error:    {final_max_error:.2f} pixels")

# Interpretation
if final_mean_error < 2:
    quality = "EXCELLENT (perfect match)"
elif final_mean_error < 5:
    quality = "GOOD (clinically equivalent)"
elif final_mean_error < 10:
    quality = "ACCEPTABLE (useful)"
else:
    quality = "POOR (investigate issues)"

print(f"\n  Quality: {quality}")

# Save results
results_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/python_pipeline_test_results.npz"
np.savez(results_file,
         cpp_landmarks=cpp_landmarks,
         initial_landmarks=initial_68pt,
         python_pipeline_landmarks=refined_landmarks,
         initial_errors=init_errors,
         final_errors=final_errors,
         initial_mean_error=init_mean_error,
         final_mean_error=final_mean_error,
         final_median_error=final_median_error,
         final_max_error=final_max_error,
         converged=converged,
         iterations=num_iterations)

print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*80)
print("PYTHON PIPELINE TEST COMPLETED SUCCESSFULLY!")
print("="*80)
