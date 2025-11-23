#!/usr/bin/env python3
"""Test to identify left eye bias root cause."""

import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'pyclnf')

from pyclnf.core.eye_pdm import EyePDM
from pyclnf.core.eye_patch_expert import align_shapes_with_scale

def main():
    # Load left eye PDM with proper path
    left_pdm = EyePDM('pyclnf/models/exported_eye_pdm_left')
    
    # Get mean shape
    mean_2d = left_pdm.params_to_landmarks_2d(np.zeros(left_pdm.n_params))
    
    print("=== Left Eye PDM Mean Shape Analysis ===")
    print(f"\nMean shape (28 landmarks):")
    for i in [0, 8, 14, 20]:
        print(f"  Landmark {i}: ({mean_2d[i, 0]:.4f}, {mean_2d[i, 1]:.4f})")
    
    print("\n--- Outer ring landmarks 8-19 ---")
    for i in range(8, 20):
        print(f"  {i}: X={mean_2d[i, 0]:8.4f}, Y={mean_2d[i, 1]:8.4f}")
    
    # Test alignment transform
    # Create a slightly scaled/rotated version of mean shape
    test_scale = 3.0
    test_rot = 0.1  # radians
    
    c, s = np.cos(test_rot), np.sin(test_rot)
    R = np.array([[c, -s], [s, c]])
    
    image_shape = (test_scale * (mean_2d @ R.T)).flatten()
    reference_shape = mean_2d.flatten()
    
    sim_img_to_ref = align_shapes_with_scale(image_shape, reference_shape)
    sim_ref_to_img = np.linalg.inv(sim_img_to_ref)
    
    print("\n=== Similarity Transform Test ===")
    print(f"Applied scale: {test_scale}, rotation: {test_rot:.3f} rad")
    print(f"\nsim_img_to_ref:\n{sim_img_to_ref}")
    print(f"\nsim_ref_to_img:\n{sim_ref_to_img}")
    
    # Expected sim_ref_to_img should recover the scale and rotation
    print(f"\nExpected scale from transform: sqrt({sim_ref_to_img[0,0]**2 + sim_ref_to_img[0,1]**2:.4f}) = {np.sqrt(sim_ref_to_img[0,0]**2 + sim_ref_to_img[0,1]**2):.4f}")
    print(f"Expected rotation from transform: atan2({sim_ref_to_img[1,0]:.4f}, {sim_ref_to_img[0,0]:.4f}) = {np.arctan2(sim_ref_to_img[1,0], sim_ref_to_img[0,0]):.4f} rad")
    
    # Test mean-shift transformation
    # A mean-shift of (+1, 0) in reference space should become what in image space?
    test_ms_ref = np.array([[1.0, 0.0]])  # Move right in ref space
    test_ms_img = test_ms_ref @ sim_ref_to_img.T
    print(f"\nMean-shift test:")
    print(f"  In reference space: ({test_ms_ref[0, 0]:.4f}, {test_ms_ref[0, 1]:.4f})")
    print(f"  In image space: ({test_ms_img[0, 0]:.4f}, {test_ms_img[0, 1]:.4f})")
    
    # Check: what does "positive X" mean in reference space?
    print("\n=== Reference Space Convention ===")
    print(f"Landmark 8 (outer corner) X: {mean_2d[8, 0]:.4f}")
    print(f"Landmark 14 (inner corner) X: {mean_2d[14, 0]:.4f}")
    print(f"Eye center X: {np.mean(mean_2d[:, 0]):.4f}")
    
    if mean_2d[8, 0] < mean_2d[14, 0]:
        print("\nOuter corner has LOWER X than inner corner")
        print("So +X direction points toward inner corner (RIGHT in image for left eye)")
    else:
        print("\nOuter corner has HIGHER X than inner corner")
        print("So +X direction points toward outer corner (LEFT in image for left eye)")

if __name__ == '__main__':
    main()
