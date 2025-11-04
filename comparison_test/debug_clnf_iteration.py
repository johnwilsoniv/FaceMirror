#!/usr/bin/env python3
"""
Debug CLNF to see what's happening at each step.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/PyfaceLM')

print("="*80)
print("CLNF DEBUG - ONE ITERATION")
print("="*80)

MODEL_DIR = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model"
TEST_IMAGE = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/frames/IMG_8401.jpg"
CPP_LANDMARKS = "/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_test/results/cpp_landmarks.npy"

# Load data
cpp_landmarks = np.load(CPP_LANDMARKS)
image = cv2.imread(TEST_IMAGE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

print(f"\nImage shape: {image.shape}")
print(f"Initial landmarks: {cpp_landmarks.shape}")
print(f"Landmark 0 (jaw left): ({cpp_landmarks[0, 0]:.1f}, {cpp_landmarks[0, 1]:.1f})")
print(f"Landmark 30 (nose): ({cpp_landmarks[30, 0]:.1f}, {cpp_landmarks[30, 1]:.1f})")

# Initialize CLNF components manually
from pyfacelm.clnf.pdm import PointDistributionModel
from pyfacelm.clnf.cen_patch_experts import CENPatchExperts

print(f"\n{'='*80}")
print("Loading models...")
print(f"{'='*80}")

pdm_path = Path(MODEL_DIR) / "pdms" / "In-the-wild_aligned_PDM_68.txt"
pdm = PointDistributionModel(pdm_path)

patch_experts = CENPatchExperts(MODEL_DIR)

# Single iteration debug
print(f"\n{'='*80}")
print("ITERATION 0")
print(f"{'='*80}")

landmarks = cpp_landmarks.copy()

# Step 1: Convert to params
print("\n[Step 1] Convert landmarks to PDM parameters...")
params, scale, translation = pdm.landmarks_to_params_2d(landmarks)
print(f"  Scale: {scale:.6f}")
print(f"  Translation: ({translation[0]:.2f}, {translation[1]:.2f})")
print(f"  Params (first 5): {params[:5]}")

# Step 2: Clamp params
params_clamped = pdm.clamp_params(params, n_std=3.0)
print(f"\n[Step 2] Clamped params (first 5): {params_clamped[:5]}")
print(f"  Params changed: {not np.allclose(params, params_clamped)}")

# Step 3: Compute CEN responses for ONE landmark
print(f"\n[Step 3] Computing CEN response for landmark 30 (nose)...")
scale_idx = 2  # 0.50 scale
expert = patch_experts.patch_experts[scale_idx][30]

lm_x, lm_y = landmarks[30]
search_radius = int(max(expert.width_support, expert.height_support) * 2.0)

x1 = max(0, int(lm_x - search_radius))
y1 = max(0, int(lm_y - search_radius))
x2 = min(gray.shape[1], int(lm_x + search_radius))
y2 = min(gray.shape[0], int(lm_y + search_radius))

print(f"  Landmark position: ({lm_x:.1f}, {lm_y:.1f})")
print(f"  Search radius: {search_radius}")
print(f"  Extraction bounds: x=[{x1}, {x2}), y=[{y1}, {y2})")

patch = gray[y1:y2, x1:x2]
print(f"  Patch shape: {patch.shape}")

response = expert.response(patch)
print(f"  Response shape: {response.shape}")
print(f"  Response min/max: {response.min():.6f} / {response.max():.6f}")

# Step 4: Find peak in response
from scipy.ndimage import gaussian_filter

response_smooth = gaussian_filter(response, sigma=1.0)
response_norm = (response_smooth - response_smooth.min())
if response_norm.max() > 0:
    response_norm = response_norm / response_norm.max()

h, w = response.shape
y_grid, x_grid = np.mgrid[0:h, 0:w]

total_weight = np.sum(response_norm)
if total_weight > 0:
    mean_x = np.sum(x_grid * response_norm) / total_weight
    mean_y = np.sum(y_grid * response_norm) / total_weight

    target_x = x1 + mean_x
    target_y = y1 + mean_y

    print(f"\n[Step 4] Mean-shift target finding...")
    print(f"  Response peak (in patch): ({mean_x:.2f}, {mean_y:.2f})")
    print(f"  Target (in image): ({target_x:.2f}, {target_y:.2f})")
    print(f"  Current landmark: ({lm_x:.2f}, {lm_y:.2f})")
    print(f"  Movement: dx={target_x - lm_x:.2f}, dy={target_y - lm_y:.2f}")
    print(f"  Distance: {np.sqrt((target_x - lm_x)**2 + (target_y - lm_y)**2):.2f} pixels")

    # Check if this makes sense
    expected_center = (patch.shape[1] // 2, patch.shape[0] // 2)
    print(f"\n  Sanity check:")
    print(f"    Expected peak near patch center: {expected_center}")
    print(f"    Actual peak in patch: ({mean_x:.2f}, {mean_y:.2f})")
    print(f"    Offset from center: ({mean_x - expected_center[0]:.2f}, {mean_y - expected_center[1]:.2f})")

    # Visualize response map
    print(f"\n  Response map (8x8 corner):")
    corner = response[:min(8, h), :min(8, w)]
    for row in corner:
        print("    " + " ".join(f"{v:6.3f}" for v in row))

print(f"\n{'='*80}")
print("DEBUG COMPLETE")
print(f"{'='*80}")
