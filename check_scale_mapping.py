#!/usr/bin/env python3
"""
Check which scale is being used for window_size=11
"""

import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

from pyclnf import CLNF

# Initialize CLNF with window_sizes=[11]
clnf = CLNF(
    model_dir='pyclnf/models',
    detector=None,
    window_sizes=[11]
)

print(f"CLNF configuration:")
print(f"  window_sizes: {clnf.window_sizes}")
print(f"  patch_scaling: {clnf.patch_scaling}")
print(f"  window_to_scale mapping: {clnf.window_to_scale}")

# Check which scale is used for window_size=11
scale_idx = clnf.window_to_scale[11]
patch_scale = clnf.patch_scaling[scale_idx]

print(f"\nFor window_size=11:")
print(f"  scale_idx: {scale_idx}")
print(f"  patch_scale: {patch_scale}")

print(f"\n→ CLNF is using scale {patch_scale}, NOT 0.25!")
print(f"\nThis explains the CEN response difference:")
print(f"  My test loaded scale 0.25 model")
print(f"  CLNF is actually using scale {patch_scale} model")
print(f"\nLet me verify by loading the correct scale...")

# Load the correct scale
import numpy as np
from pyclnf.core.cen_patch_expert import CENModel

cen = CENModel('pyclnf/models', scales=[patch_scale])
patch_expert = cen.scale_models[patch_scale]['views'][0]['patches'][36]

# Load the saved area_of_interest
area_of_interest = np.load('/tmp/area_of_interest_lm36_before_response.npy')

# Compute response
resp = patch_expert.response(area_of_interest)

print(f"\nCEN response with scale {patch_scale}:")
print(f"  Max: {resp.max():.6f} at {np.unravel_index(np.argmax(resp), resp.shape)}")
print(f"  Mean: {resp.mean():.6f}")

# Compare with BEFORE_SIGMA
before_sigma = np.load('/tmp/python_response_map_lm36_iter0_ws11_BEFORE_SIGMA.npy')
print(f"\nBEFORE_SIGMA from CLNF:")
print(f"  Max: {before_sigma.max():.6f} at {np.unravel_index(np.argmax(before_sigma), before_sigma.shape)}")

if np.allclose(resp, before_sigma, atol=1e-5):
    print(f"\n✓ MATCH! This is the correct scale.")
else:
    print(f"\n✗ Still doesn't match. Correlation: {np.corrcoef(resp.flatten(), before_sigma.flatten())[0, 1]:.6f}")
