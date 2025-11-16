#!/usr/bin/env python3
"""
Verify CEN response computation on the saved area_of_interest.
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

from pyclnf.core.cen_patch_expert import CENModel

# Load the saved area_of_interest
area_of_interest = cv2.imread('/tmp/area_of_interest_lm36.png', cv2.IMREAD_GRAYSCALE)

print(f"Loaded area_of_interest:")
print(f"  Shape: {area_of_interest.shape}")
print(f"  Stats: min={area_of_interest.min()}, max={area_of_interest.max()}, mean={area_of_interest.mean():.1f}")

# Load CEN model
cen = CENModel('pyclnf/models', scales=[0.25])
patch_expert = cen.scale_models[0.25]['views'][0]['patches'][36]

print(f"\nPatch expert:")
print(f"  Width: {patch_expert.width}, Height: {patch_expert.height}")

# Compute response
response_map = patch_expert.response(area_of_interest)

print(f"\nComputed response map:")
print(f"  Shape: {response_map.shape}")
print(f"  Min: {response_map.min():.6f}, Max: {response_map.max():.6f}, Mean: {response_map.mean():.6f}")
peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
print(f"  Peak at: {peak_idx} = {response_map.max():.6f}")

# Load the saved CLNF response map for comparison
clnf_resp = np.load('/tmp/python_response_map_lm36_iter0_ws11.npy')

print(f"\nCLNF saved response map:")
print(f"  Shape: {clnf_resp.shape}")
print(f"  Min: {clnf_resp.min():.6f}, Max: {clnf_resp.max():.6f}, Mean: {clnf_resp.mean():.6f}")
peak_idx_clnf = np.unravel_index(np.argmax(clnf_resp), clnf_resp.shape)
print(f"  Peak at: {peak_idx_clnf} = {clnf_resp.max():.6f}")

# Compare
print(f"\nComparison:")
print(f"  Correlation: {np.corrcoef(response_map.flatten(), clnf_resp.flatten())[0, 1]:.6f}")
print(f"  Max difference: {np.abs(response_map - clnf_resp).max():.6f}")

# If they match, then CEN is working correctly
# The issue must be in the warping transformation or bbox initialization

if np.allclose(response_map, clnf_resp, atol=1e-5):
    print("\n✓ Response maps MATCH! CEN computation is correct.")
    print("  The issue is likely in:")
    print("  1. Different bbox/initialization between Python and C++")
    print("  2. Different warping transformation")
    print("  3. Different reference shape calculation")
else:
    print("\n✗ Response maps DIFFER! CEN computation may have a bug.")

# Now let's compare with the C++ response map
cpp_data = np.fromfile('/tmp/cpp_response_map_lm36_iter0.bin', dtype=np.float32)
cpp_resp = cpp_data[2:123].reshape(11, 11, order='F')

print(f"\nC++ response map:")
print(f"  Shape: {cpp_resp.shape}")
print(f"  Min: {cpp_resp.min():.6f}, Max: {cpp_resp.max():.6f}, Mean: {cpp_resp.mean():.6f}")
peak_idx_cpp = np.unravel_index(np.argmax(cpp_resp), cpp_resp.shape)
print(f"  Peak at: {peak_idx_cpp} = {cpp_resp.max():.6f}")

print(f"\nPython vs C++ comparison:")
print(f"  Correlation: {np.corrcoef(response_map.flatten(), cpp_resp.flatten())[0, 1]:.6f}")
print(f"\nConclusion:")
print(f"  Python peak at {peak_idx}, C++ peak at {peak_idx_cpp}")
print(f"  Peaks are {abs(peak_idx[0]-peak_idx_cpp[0])} rows and {abs(peak_idx[1]-peak_idx_cpp[1])} cols apart")
print(f"  This suggests different landmark positions or warping transformations")
