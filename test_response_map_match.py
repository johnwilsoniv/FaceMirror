#!/usr/bin/env python3
"""
Compare Python vs C++ response maps at the SAME position.
Uses C++ initial position directly to extract AOI and compute response.
"""

import sys
sys.path.insert(0, 'pyclnf')

import numpy as np
import cv2
from pyclnf.core.eye_patch_expert import HierarchicalEyeModel

def main():
    # Load image
    image = cv2.imread('comparison_frame_0000.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # C++ values from debug output
    cpp_eye8_pos = np.array([399.954468, 824.986633])
    cpp_a1 = 3.252852
    cpp_b1 = -0.352717
    
    # Compute AOI extraction parameters (matching C++)
    window_size = 3
    patch_size = 11  # Standard eye patch size
    aoi_size = window_size + patch_size - 1  # = 13
    half_aoi = (aoi_size - 1) / 2.0  # = 6.0
    
    # C++ transform: tx = x - a1*half + b1*half, ty = y - a1*half - b1*half
    tx = cpp_eye8_pos[0] - cpp_a1 * half_aoi + cpp_b1 * half_aoi
    ty = cpp_eye8_pos[1] - cpp_a1 * half_aoi - cpp_b1 * half_aoi
    
    print(f"=== Using C++ Position for AOI Extraction ===")
    print(f"Eye_8 position: ({cpp_eye8_pos[0]:.6f}, {cpp_eye8_pos[1]:.6f})")
    print(f"a1={cpp_a1:.6f}, b1={cpp_b1:.6f}")
    print(f"aoi_size={aoi_size}, half_aoi={half_aoi}")
    print(f"tx={tx:.6f}, ty={ty:.6f}")
    
    # Extract AOI using same transform as C++
    sim = np.array([[cpp_a1, -cpp_b1, tx],
                    [cpp_b1, cpp_a1, ty]], dtype=np.float32)
    
    aoi = cv2.warpAffine(
        gray, sim, (aoi_size, aoi_size),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
    )
    
    print(f"\nPython AOI (13x13) using C++ position:")
    for row in range(aoi_size):
        print("  " + " ".join(f"{aoi[row, col]:.1f}" for col in range(aoi_size)))
    
    # Load eye model and get patch expert for landmark 8
    eye_model = HierarchicalEyeModel('pyclnf/models')
    
    # Access patch experts through ccnf model (scale 1.0)
    ccnf_model = eye_model.ccnf['left']
    patch_experts = ccnf_model.get_all_patch_experts(1.0)  # scale = 1.0
    
    if 8 not in patch_experts:
        print(f"Error: No patch expert for landmark 8")
        print(f"Available: {list(patch_experts.keys())}")
        return
    
    patch_expert = patch_experts[8]
    print(f"\nPatch expert: {patch_expert.width}x{patch_expert.height}")
    
    # Compute response map (3x3)
    resp_h = aoi_size - patch_expert.height + 1
    resp_w = aoi_size - patch_expert.width + 1
    
    response_map = np.zeros((resp_h, resp_w), dtype=np.float32)
    
    for j in range(resp_w):
        for i in range(resp_h):
            patch = aoi[i:i+patch_expert.height, j:j+patch_expert.width]
            response = patch_expert.compute_response(patch.astype(np.float32))
            response_map[i, j] = response
    
    # Shift to non-negative (like C++)
    min_val = response_map.min()
    if min_val < 0:
        response_map = response_map - min_val
    
    print(f"\nPython Response map ({resp_h}x{resp_w}):")
    for row in range(resp_h):
        print("  " + " ".join(f"{response_map[row, col]:.4f}" for col in range(resp_w)))
    
    # Compare with C++ response map
    print("\n=== C++ Response map (from binary) ===")
    import struct
    with open('/tmp/cpp_eye_lm8_response.bin', 'rb') as f:
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        cpp_response = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
    
    for row in range(rows):
        print("  " + " ".join(f"{cpp_response[row, col]:.4f}" for col in range(cols)))
    
    # Compute difference
    print("\n=== Difference (Python - C++) ===")
    diff = response_map - cpp_response
    for row in range(resp_h):
        print("  " + " ".join(f"{diff[row, col]:+.4f}" for col in range(resp_w)))
    
    print(f"\nMax abs difference: {np.abs(diff).max():.6f}")
    print(f"Mean abs difference: {np.abs(diff).mean():.6f}")

if __name__ == '__main__':
    main()
