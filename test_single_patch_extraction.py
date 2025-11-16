#!/usr/bin/env python3
"""
Test single patch extraction to debug response map divergence.
Extract landmark 36 patch at Python's ITER0 position and compute response.
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')

from pathlib import Path
from pyclnf.core.pdm import PDM
from pyclnf.core.cen_patch_expert import CENModel

# Load image
image_path = 'calibration_frames/patient1_frame1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print(f"Image shape: {image.shape}")

# Load models
pdm_dir = Path('pyclnf/models/exported_pdm')
pdm = PDM(str(pdm_dir))

cen = CENModel('pyclnf/models', scales=[0.25])

# Get landmark 36 CEN patch expert
patch_expert = cen.scale_models[0.25]['views'][0]['patches'][36]

print(f"\nPatch expert loaded:")
print(f"  Patch width: {patch_expert.width}")
print(f"  Patch height: {patch_expert.height}")

# Use Python's ITER0 landmark 36 position (from debug output)
lm_x = 377.862007
lm_y = 848.967994

print(f"\nLandmark 36 position: ({lm_x:.6f}, {lm_y:.6f})")

# Extract 11x11 patch directly (no warping first)
window_size = 11
half_window = window_size // 2

# Get integer center
cx_int = int(lm_x)
cy_int = int(lm_y)

print(f"\nDirect patch extraction (no warping):")
print(f"  Window center (int): ({cx_int}, {cy_int})")
print(f"  Window bounds: x=[{cx_int-half_window}, {cx_int+half_window+1}], y=[{cy_int-half_window}, {cy_int+half_window+1}]")

# Extract window
window = image[cy_int-half_window:cy_int+half_window+1,
               cx_int-half_window:cx_int+half_window+1]

print(f"  Extracted window shape: {window.shape}")
print(f"  Window stats: min={window.min()}, max={window.max()}, mean={window.mean():.1f}")

# Now compute CEN response on each position in the window
print(f"\nComputing CEN responses at {window_size}x{window_size} positions...")

response_map = np.zeros((window_size, window_size), dtype=np.float32)

pw = patch_expert.width
ph = patch_expert.height

for i in range(window_size):
    for j in range(window_size):
        # Extract patch at this position
        # i is row (y), j is col (x)
        y = cy_int - half_window + i
        x = cx_int - half_window + j

        # Extract patch centered at (x, y)
        patch_y0 = y - ph // 2
        patch_x0 = x - pw // 2

        # Check bounds
        if (patch_y0 >= 0 and patch_y0 + ph < image.shape[0] and
            patch_x0 >= 0 and patch_x0 + pw < image.shape[1]):
            patch = image[patch_y0:patch_y0+ph, patch_x0:patch_x0+pw]

            # Compute response
            resp = patch_expert.response(patch)
            response_map[i, j] = resp
        else:
            response_map[i, j] = 0.0

print(f"\nResponse map computed:")
print(f"  Shape: {response_map.shape}")
print(f"  Min: {response_map.min():.6f}, Max: {response_map.max():.6f}, Mean: {response_map.mean():.6f}")
peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
print(f"  Peak at: {peak_idx} = {response_map.max():.6f}")
print(f"  Center [5,5]: {response_map[5, 5]:.6f}")

# Save for comparison
np.save('/tmp/python_response_map_direct.npy', response_map)
print(f"\nSaved to /tmp/python_response_map_direct.npy")

# Compare with the saved CLNF response map
clnf_resp = np.load('/tmp/python_response_map_lm36_iter0_ws11.npy')

print(f"\nComparison with CLNF response map:")
print(f"  CLNF peak: {np.unravel_index(np.argmax(clnf_resp), clnf_resp.shape)} = {clnf_resp.max():.6f}")
print(f"  Correlation: {np.corrcoef(response_map.flatten(), clnf_resp.flatten())[0, 1]:.6f}")

# Show both side by side
print(f"\nDirect extraction response map:")
for i in range(11):
    row_str = "  "
    for j in range(11):
        row_str += f"{response_map[i,j]:.3f} "
    print(row_str)

print(f"\nCLNF response map (with warping):")
for i in range(11):
    row_str = "  "
    for j in range(11):
        row_str += f"{clnf_resp[i,j]:.3f} "
    print(row_str)
