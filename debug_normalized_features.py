#!/usr/bin/env python3
"""
Debug normalized features comparison between Python and C++.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pathlib import Path

from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, LEFT_EYE_MAPPING

def main():
    print("=" * 70)
    print("NORMALIZED FEATURES DEBUG")
    print("=" * 70)

    # Load video frame
    video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
    ret, frame = video.read()
    video.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load eye model
    model_dir = 'pyclnf/models'
    eye_model = HierarchicalEyeModel(model_dir)
    pdm = eye_model.pdm['left']

    # C++ input landmarks
    CPP_LEFT_EYE_INPUT = {
        36: (392.1590, 847.6613), 37: (410.0039, 828.3166),
        38: (436.9223, 826.1841), 39: (461.9583, 842.8420),
        40: (438.4380, 850.4288), 41: (411.4089, 853.9998)
    }

    target_points = np.array([CPP_LEFT_EYE_INPUT[i] for i in [36, 37, 38, 39, 40, 41]])
    params = eye_model._fit_eye_shape(target_points, LEFT_EYE_MAPPING, 'left', main_rotation=None)
    eye_landmarks = pdm.params_to_landmarks_2d(params)

    # Get patch expert for Eye_8
    ccnf = eye_model.ccnf['left']
    patch_experts = ccnf.get_all_patch_experts(eye_model.patch_scale)
    pe = patch_experts[8]

    print(f"\nPatch expert Eye_8:")
    print(f"  Size: {pe.height}x{pe.width}")

    # Extract center patch with window size 3
    ws = 3
    x, y = eye_landmarks[8]
    patch_size = pe.height
    aoi_size = ws + patch_size - 1  # 3 + 11 - 1 = 13
    half_aoi = (aoi_size - 1) / 2.0

    print(f"\nLandmark 8 position: ({x:.2f}, {y:.2f})")
    print(f"Window size: {ws}, AOI size: {aoi_size}")

    # Create transformation matrix
    a1 = 1.0
    b1 = 0.0
    tx = x - a1 * half_aoi + b1 * half_aoi
    ty = y - a1 * half_aoi - b1 * half_aoi

    print(f"Affine transform: tx={tx:.2f}, ty={ty:.2f}")

    sim = np.array([[a1, -b1, tx], [b1, a1, ty]], dtype=np.float32)

    # Extract area of interest
    area_of_interest = cv2.warpAffine(
        gray.astype(np.float32),
        sim,
        (aoi_size, aoi_size),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
    )

    # Extract center patch (for 3x3 response, center is at (1,1))
    center_i, center_j = 1, 1  # Center of 3x3
    patch = area_of_interest[center_i:center_i+patch_size, center_j:center_j+patch_size]

    print(f"\nCenter patch shape: {patch.shape}")
    print(f"  min: {patch.min():.2f}, max: {patch.max():.2f}, mean: {patch.mean():.2f}")

    # Normalize like C++ ResponseOpenBlas
    features = patch.astype(np.float32)
    features_flat = features.flatten('F')  # Column-major like C++
    feature_mean = np.mean(features_flat)
    features_centered = features_flat - feature_mean
    feature_norm = np.linalg.norm(features_centered)

    if feature_norm > 1e-10:
        normalized_features = features_centered / feature_norm
    else:
        normalized_features = features_centered

    print(f"\nPython normalized features (column-major):")
    print(f"  feature_mean: {feature_mean:.6f}")
    print(f"  feature_norm: {feature_norm:.6f}")
    print(f"  First 10 values:")
    for i in range(10):
        print(f"    [{i}]: {normalized_features[i]:.6f}")

    print(f"\nC++ normalized input (from debug):")
    print("  [0]: 1.000000 (bias term)")
    cpp_normalized = [-0.019852, -0.022680, -0.025950, -0.026463, -0.019941,
                     -0.043181, -0.059916, -0.077854, 0.011430]
    for i, val in enumerate(cpp_normalized):
        print(f"  [{i+1}]: {val:.6f}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Note: C++ normalized_input[0]=1 is for bias, so compare Python[i] with C++[i+1]
    print("\nPython vs C++ (offset by 1 for bias):")
    for i in range(len(cpp_normalized)):
        py_val = normalized_features[i]
        cpp_val = cpp_normalized[i]
        diff = py_val - cpp_val
        print(f"  [{i}]: Python={py_val:.6f}, C++={cpp_val:.6f}, diff={diff:.6f}")

    # Compute neuron response with Python values
    print("\n" + "=" * 70)
    print("NEURON RESPONSE COMPUTATION")
    print("=" * 70)

    neuron = pe.neurons[0]
    weights = neuron['weights']
    weights_flat = weights.flatten('F') * neuron['norm_weights']

    sigmoid_input = neuron['bias'] + np.dot(weights_flat, normalized_features)
    sigmoid_output = (2.0 * neuron['alpha']) / (1.0 + np.exp(-np.clip(sigmoid_input, -500, 500)))

    print(f"\nPython neuron 0:")
    print(f"  alpha: {neuron['alpha']:.6f}")
    print(f"  bias: {neuron['bias']:.6f}")
    print(f"  sigmoid_input: {sigmoid_input:.6f}")
    print(f"  sigmoid_output: {sigmoid_output:.6f}")

    print(f"\nC++ neuron 0 (from debug):")
    print(f"  alpha: 3.439889 (this is RIGHT eye!)")
    print(f"  sigmoid_input: 0.795442")
    print(f"  sigmoid_output: 4.740157")

    # But wait - the C++ debug was from RIGHT eye, not left!
    # Let me compute what the Python result should be
    sum_alphas = sum(n['alpha'] for n in pe.neurons)
    total_response = sigmoid_output
    for i, n in enumerate(pe.neurons[1:], 1):
        w_flat = n['weights'].flatten('F') * n['norm_weights']
        si = n['bias'] + np.dot(w_flat, normalized_features)
        so = (2.0 * n['alpha']) / (1.0 + np.exp(-np.clip(si, -500, 500)))
        total_response += so

    normalized_response = total_response / (2.0 * sum_alphas)

    print(f"\nPython total LEFT eye response:")
    print(f"  Total response: {total_response:.6f}")
    print(f"  Sum alphas: {sum_alphas:.6f}")
    print(f"  Normalized: {normalized_response:.6f}")

if __name__ == '__main__':
    main()
