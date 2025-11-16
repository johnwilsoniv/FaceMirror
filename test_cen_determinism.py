#!/usr/bin/env python3
"""
Test CEN determinism by calling response() multiple times with the same input.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

from pyclnf.core.cen_patch_expert import CENModel, NUMBA_AVAILABLE

print(f"NUMBA_AVAILABLE: {NUMBA_AVAILABLE}")

# Load the saved area_of_interest
area_of_interest = np.load('/tmp/area_of_interest_lm36_before_response.npy')

print(f"\nArea of interest:")
print(f"  Shape: {area_of_interest.shape}, dtype: {area_of_interest.dtype}")
print(f"  Min: {area_of_interest.min()}, Max: {area_of_interest.max()}, Mean: {area_of_interest.mean():.1f}")
print(f"  Sum: {area_of_interest.sum()}")

# Load CEN model
print(f"\nLoading CEN model...")
cen = CENModel('pyclnf/models', scales=[0.25])
patch_expert = cen.scale_models[0.25]['views'][0]['patches'][36]

print(f"Patch expert:")
print(f"  Width: {patch_expert.width}, Height: {patch_expert.height}")
print(f"  Num layers: {len(patch_expert.weights)}")
print(f"  Is empty: {patch_expert.is_empty}")

# Call response() multiple times and check if results are consistent
print(f"\nCalling response() 5 times with the same input...")

results = []
for i in range(5):
    resp = patch_expert.response(area_of_interest)
    peak_idx = np.unravel_index(np.argmax(resp), resp.shape)
    peak_val = resp.max()

    print(f"  Call {i+1}: max={peak_val:.6f} at {peak_idx}, mean={resp.mean():.6f}")
    results.append((peak_val, peak_idx, resp.copy()))

# Check if all results are identical
all_same = True
for i in range(1, len(results)):
    if not np.array_equal(results[0][2], results[i][2]):
        all_same = False
        print(f"\n✗ Result {i+1} differs from result 1!")
        print(f"  Max difference: {np.abs(results[0][2] - results[i][2]).max():.6f}")

if all_same:
    print(f"\n✓ All 5 calls produced identical results!")
    print(f"  CEN.response() is deterministic within this process.")
else:
    print(f"\n✗ CEN.response() is NON-DETERMINISTIC!")

# Now compare with the saved BEFORE_SIGMA response map from CLNF
before_sigma = np.load('/tmp/python_response_map_lm36_iter0_ws11_BEFORE_SIGMA.npy')

print(f"\nComparing with CLNF BEFORE_SIGMA:")
print(f"  CLNF max: {before_sigma.max():.6f} at {np.unravel_index(np.argmax(before_sigma), before_sigma.shape)}")
print(f"  Now max:  {results[0][0]:.6f} at {results[0][1]}")
print(f"  Correlation: {np.corrcoef(results[0][2].flatten(), before_sigma.flatten())[0, 1]:.6f}")

if np.allclose(results[0][2], before_sigma, atol=1e-5):
    print(f"\n✓ Matches CLNF output!")
else:
    print(f"\n✗ Does NOT match CLNF output!")
    print(f"  Max difference: {np.abs(results[0][2] - before_sigma).max():.6f}")
    print(f"\nPossible causes:")
    print(f"  1. Different CEN model loaded (different scales/views)")
    print(f"  2. Model weights changed between runs")
    print(f"  3. Input data type handling differs")
