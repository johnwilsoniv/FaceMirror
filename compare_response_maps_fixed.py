#!/usr/bin/env python3
"""
Compare C++ and Python response maps after fixing scale mapping bug.
Now that initialization is confirmed identical, response maps should match.
"""
import cv2
import numpy as np
from pyclnf.clnf import CLNF

# Test image
image_path = "calibration_frames/patient1_frame1.jpg"
cpp_bbox = (301.938, 782.149, 400.586, 400.585)

# Initialize CLNF (now with corrected scale mapping)
clnf = CLNF(
    model_dir="pyclnf/models",
    regularization=35,
    max_iterations=10,
    convergence_threshold=0.005,
    sigma=1.5,
    weight_multiplier=0.0,
    window_sizes=[11, 9, 7, 5],  # Corrected to match C++
    detector=None,
    debug_mode=True,
    tracked_landmarks=[36]
)

# Load image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Testing response map comparison after scale mapping fix:")
print(f"  Image: {image_path}")
print(f"  Bbox: {cpp_bbox}")
print(f"  Window sizes: {clnf.window_sizes}")
print(f"  Scale mapping: {clnf._map_windows_to_scales()}")
print()

# Get initial params
params = clnf.pdm.init_params(cpp_bbox)
init_landmarks = clnf.pdm.params_to_landmarks_2d(params)

print(f"Initial params:")
print(f"  scale: {params[0]:.6f}")
print(f"  rotation: ({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
print(f"  translation: ({params[4]:.6f}, {params[5]:.6f})")
print(f"  Landmark_36: ({init_landmarks[36, 0]:.6f}, {init_landmarks[36, 1]:.6f})")
print()

# First iteration: window_size=11, scale_idx=0 (0.25)
window_size = 11
scale_idx = 0
patch_scale = clnf.patch_scaling[scale_idx]

print(f"Iteration 1: window_size={window_size}, scale_idx={scale_idx}, patch_scale={patch_scale}")

# Get patch expert for landmark 36
# Access via CENModel.scale_models[scale]['views'][view_idx]['patches'][landmark_idx]
patch_expert = clnf.ccnf.scale_models[patch_scale]['views'][0]['patches'][36]

# Extract patch location for landmark 36
lm_idx = 36
cx, cy = init_landmarks[lm_idx]

# Get area of interest (same as optimizer does)
patch_dim = patch_expert.width_support
aoi_size = window_size + patch_dim - 1

# Calculate top-left corner of area of interest
cx_scaled = int(round(cx / patch_scale))
cy_scaled = int(round(cy / patch_scale))
half_aoi = aoi_size // 2
x0 = cx_scaled - half_aoi
y0 = cy_scaled - half_aoi

# Downsample image (simple approach - just resize)
# In the real optimizer, this is done via warpAffine with similarity transform
scale_factor = 1.0 / patch_scale
new_width = int(gray.shape[1] * scale_factor)
new_height = int(gray.shape[0] * scale_factor)
warped_img = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

print(f"  Landmark center: ({cx:.2f}, {cy:.2f})")
print(f"  Scaled center: ({cx_scaled}, {cy_scaled})")
print(f"  patch_dim: {patch_dim}")
print(f"  aoi_size: {aoi_size}")
print(f"  AOI top-left: ({x0}, {y0})")
print(f"  Warped image size: {warped_img.shape}")

area_of_interest = warped_img[y0:y0+aoi_size, x0:x0+aoi_size].astype(np.float32)
print(f"  AOI shape: {area_of_interest.shape}")
print(f"  AOI stats: min={area_of_interest.min()}, max={area_of_interest.max()}, mean={area_of_interest.mean():.1f}")

# Save AOI for comparison
np.save('/tmp/python_aoi_lm36.npy', area_of_interest)
print(f"  Saved AOI to /tmp/python_aoi_lm36.npy")

# Get response map
response_map = patch_expert.response(area_of_interest)
print(f"  Response map shape: {response_map.shape}")
print(f"  Response map stats: min={response_map.min():.6f}, max={response_map.max():.6f}, mean={response_map.mean():.6f}")

# Find peak
peak_y, peak_x = np.unravel_index(response_map.argmax(), response_map.shape)
print(f"  Peak location: ({peak_x}, {peak_y}) = {response_map[peak_y, peak_x]:.6f}")

# Save for comparison
np.save('/tmp/python_response_lm36_iter1.npy', response_map)
print(f"\nSaved Python response map to /tmp/python_response_lm36_iter1.npy")

# Load C++ response map if available
try:
    cpp_response = np.load('/tmp/cpp_response_lm36_iter1.npy')
    print(f"\nComparison with C++ response map:")
    print(f"  C++ shape: {cpp_response.shape}")
    print(f"  C++ stats: min={cpp_response.min():.6f}, max={cpp_response.max():.6f}, mean={cpp_response.mean():.6f}")
    cpp_peak_y, cpp_peak_x = np.unravel_index(cpp_response.argmax(), cpp_response.shape)
    print(f"  C++ peak: ({cpp_peak_x}, {cpp_peak_y}) = {cpp_response[cpp_peak_y, cpp_peak_x]:.6f}")

    # Calculate correlation
    correlation = np.corrcoef(response_map.flatten(), cpp_response.flatten())[0, 1]
    print(f"  Correlation: {correlation:.6f}")

    # Calculate difference
    diff = np.abs(response_map - cpp_response)
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Max absolute difference: {diff.max():.6f}")

    if correlation > 0.99:
        print("\n✓ Response maps MATCH!")
    else:
        print(f"\n✗ Response maps differ (correlation={correlation:.6f})")

except FileNotFoundError:
    print("\nC++ response map not found. Run C++ comparison first.")
