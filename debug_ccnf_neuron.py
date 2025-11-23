#!/usr/bin/env python3
"""
Debug CCNF neuron computation to compare with C++.
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
    print("CCNF NEURON DEBUG - Comparing Python vs C++")
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
    print(f"  Num neurons: {pe.num_neurons}")
    print(f"  Betas: {pe.betas}")

    # Extract center patch for Eye_8 with window size 3 (to match C++ debug)
    ws = 3
    x, y = eye_landmarks[8]
    patch_size = pe.height
    aoi_size = ws + patch_size - 1  # 3 + 11 - 1 = 13
    half_aoi = (aoi_size - 1) / 2.0

    # Create transformation matrix
    a1 = 1.0
    b1 = 0.0
    tx = x - a1 * half_aoi + b1 * half_aoi
    ty = y - a1 * half_aoi - b1 * half_aoi

    sim = np.array([[a1, -b1, tx], [b1, a1, ty]], dtype=np.float32)

    # Extract area of interest
    area_of_interest = cv2.warpAffine(
        gray.astype(np.float32),
        sim,
        (aoi_size, aoi_size),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
    )

    print(f"\nArea of interest: {area_of_interest.shape}")
    print(f"  min: {area_of_interest.min():.2f}, max: {area_of_interest.max():.2f}")

    # Extract center patch (for 3x3 response, center is at (1,1))
    # Response size = aoi_size - patch_size + 1 = 13 - 11 + 1 = 3
    center_i, center_j = 1, 1  # Center of 3x3
    patch = area_of_interest[center_i:center_i+patch_size, center_j:center_j+patch_size]

    print(f"\nCenter patch shape: {patch.shape}")
    print(f"  min: {patch.min():.2f}, max: {patch.max():.2f}")

    # Now compute CCNF response with detailed debug
    features = patch.astype(np.float32)

    print("\n" + "=" * 70)
    print("NEURON-BY-NEURON COMPUTATION")
    print("=" * 70)

    total_response = 0.0
    sum_alphas = 0.0

    for i, neuron in enumerate(pe.neurons):
        alpha = neuron['alpha']
        if abs(alpha) < 1e-4:
            continue
        sum_alphas += alpha

        weights = neuron['weights']

        # Flatten in column-major (Fortran) order
        features_flat = features.flatten('F')
        feature_mean = np.mean(features_flat)
        features_centered = features_flat - feature_mean
        feature_norm = np.linalg.norm(features_centered)

        if feature_norm > 1e-10:
            normalized_features = features_centered / feature_norm
        else:
            normalized_features = features_centered

        weights_flat = weights.flatten('F') * neuron['norm_weights']

        sigmoid_input = neuron['bias'] + np.dot(weights_flat, normalized_features)
        sigmoid_output = (2.0 * alpha) / (1.0 + np.exp(-np.clip(sigmoid_input, -500, 500)))
        total_response += sigmoid_output

        print(f"\nNeuron {i}:")
        print(f"  alpha: {alpha:.6f}")
        print(f"  bias: {neuron['bias']:.6f}")
        print(f"  norm_weights: {neuron['norm_weights']:.6f}")
        print(f"  weights shape: {weights.shape}")
        print(f"  sigmoid_input: {sigmoid_input:.6f}")
        print(f"  sigmoid_output: {sigmoid_output:.6f}")

        if i == 0:
            print(f"  First 10 normalized_features: {normalized_features[:10]}")
            print(f"  First 10 weights_flat: {weights_flat[:10]}")

    normalized_response = total_response / (2.0 * sum_alphas) if sum_alphas > 1e-10 else total_response

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total response: {total_response:.6f}")
    print(f"Sum alphas: {sum_alphas:.6f}")
    print(f"Normalized response: {normalized_response:.6f}")

    print("\n" + "=" * 70)
    print("C++ COMPARISON VALUES")
    print("=" * 70)
    print("C++ (from /tmp/cpp_ccnf_neuron_debug.txt):")
    print("  Total response: 20.978935")
    print("  Sum alphas: 47.135536")
    print("  Normalized response: 0.222538")
    print("")
    print("  Neuron 0: alpha=3.439889, sigmoid_input=0.795442, sigmoid_output=4.740157")
    print("  Neuron 6: alpha=11.792998, sigmoid_input=0.787645, sigmoid_output=16.211258")

    if abs(normalized_response - 0.222538) < 0.01:
        print("\n*** RESPONSES MATCH! ***")
    else:
        print(f"\n*** MISMATCH: Python={normalized_response:.6f}, C++={0.222538:.6f} ***")

if __name__ == '__main__':
    main()
