#!/usr/bin/env python3
"""
Compare CEN model hashes between Python and C++ to verify they're using the same model.
"""

import sys
import hashlib
import numpy as np

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

from pyclnf.core.cen_patch_expert import CENModel

# Load Python CEN model
print("Loading Python CEN model...")
cen = CENModel('pyclnf/models', scales=[0.25])
patch_expert = cen.scale_models[0.25]['views'][0]['patches'][36]

print(f"\nPython CEN Patch Expert (landmark 36, scale 0.25, view 0):")
print(f"  Width: {patch_expert.width}, Height: {patch_expert.height}")
print(f"  Num layers: {len(patch_expert.weights)}")
print(f"  Confidence: {patch_expert.confidence}")

# Compute hash of all model parameters
def hash_patch_expert(expert):
    """Compute SHA256 hash of all model parameters"""
    h = hashlib.sha256()

    # Hash dimensions
    h.update(np.array([expert.width_support, expert.height_support], dtype=np.int32).tobytes())

    # Hash number of layers
    h.update(np.array([len(expert.weights)], dtype=np.int32).tobytes())

    # Hash each layer
    for i in range(len(expert.weights)):
        # Hash activation function type
        h.update(np.array([expert.activation_function[i]], dtype=np.int32).tobytes())

        # Hash bias
        h.update(expert.biases[i].tobytes())

        # Hash weights
        h.update(expert.weights[i].tobytes())

    # Hash confidence
    h.update(np.array([expert.confidence], dtype=np.float64).tobytes())

    return h.hexdigest()

py_hash = hash_patch_expert(patch_expert)
print(f"\nPython model hash: {py_hash}")

# Print layer details for debugging
print(f"\nLayer details:")
for i in range(len(patch_expert.weights)):
    print(f"  Layer {i}:")
    print(f"    Activation: {patch_expert.activation_function[i]}")
    print(f"    Bias shape: {patch_expert.biases[i].shape}, mean: {patch_expert.biases[i].mean():.6f}")
    print(f"    Weight shape: {patch_expert.weights[i].shape}, mean: {patch_expert.weights[i].mean():.6f}")

    # Print a few sample values
    print(f"    Bias first 5: {patch_expert.biases[i].flatten()[:5]}")
    print(f"    Weight first 5: {patch_expert.weights[i].flatten()[:5]}")

# Now we need to get the C++ model hash
# Let's check which model file Python is using
import os
model_path = 'pyclnf/models/patch_experts/cen_patches_0.25_of.dat'
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    print(f"\nPython model file: {model_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    # Hash the entire file
    with open(model_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  File SHA256: {file_hash}")
else:
    print(f"\nWARNING: Model file not found: {model_path}")

# Check what C++ is using
print(f"\n" + "="*70)
print("C++ MODEL INFORMATION")
print("="*70)
print("\nTo compare with C++, we need to:")
print("1. Find which model file C++ OpenFace is using")
print("2. Either:")
print("   a) Extract and hash the weights from C++ binary, or")
print("   b) Verify C++ is using the same .dat file")
print("\nLet's check the OpenFace model directory...")

# Check OpenFace model directory
openface_model_paths = [
    '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/model/patch_experts/cen_patches_0.25_of.dat',
    '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/patch_experts/cen_patches_0.25_of.dat',
]

for cpp_path in openface_model_paths:
    if os.path.exists(cpp_path):
        cpp_size = os.path.getsize(cpp_path)
        print(f"\nFound C++ model: {cpp_path}")
        print(f"  File size: {cpp_size:,} bytes ({cpp_size / 1024 / 1024:.2f} MB)")

        with open(cpp_path, 'rb') as f:
            cpp_file_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"  File SHA256: {cpp_file_hash}")

        if cpp_file_hash == file_hash:
            print(f"  ✓ MATCHES Python model file!")
        else:
            print(f"  ✗ DIFFERS from Python model file!")
            print(f"    Python: {file_hash}")
            print(f"    C++:    {cpp_file_hash}")
            print(f"\n    → The models are DIFFERENT! This explains the divergence.")
