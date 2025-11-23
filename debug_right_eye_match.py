#!/usr/bin/env python3
"""
Debug right eye landmark 0 to match C++ debug output exactly.
"""

import sys
sys.path.insert(0, 'pyclnf')
sys.path.insert(0, 'pymtcnn')

import numpy as np
import cv2
from pathlib import Path

def load_right_eye_ccnf():
    """Load right eye CCNF patch experts from exported models."""
    base_dir = Path('pyclnf/models/exported_eye_ccnf_right/scale_1.00')

    # Load patch 0
    patch_dir = base_dir / 'view_00' / 'patch_00'
    meta = np.load(patch_dir / 'metadata.npz')

    neurons = []
    neuron_files = sorted(patch_dir.glob('neuron_*.npz'))
    for nf in neuron_files:
        data = np.load(nf)
        neurons.append({
            'alpha': float(data['alpha']),
            'bias': float(data['bias']),
            'norm_weights': float(data['norm_weights']),
            'weights': data['weights']
        })

    return {
        'height': int(meta['height']),
        'width': int(meta['width']),
        'neurons': neurons
    }

def main():
    print("=" * 70)
    print("RIGHT EYE LANDMARK 0 - Match C++ Debug")
    print("=" * 70)

    # Load video frame
    video = cv2.VideoCapture('Patient Data/Normal Cohort/Shorty.mov')
    ret, frame = video.read()
    video.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load right eye patch expert
    pe = load_right_eye_ccnf()
    print(f"Patch size: {pe['height']}x{pe['width']}")
    print(f"Num neurons: {len(pe['neurons'])}")

    # C++ right eye landmarks from debug
    # From cpp_eye_init_debug.txt for right_eye_28:
    # We need the initial right eye landmarks
    # Main landmarks: 42-47 map to eye landmarks 8,10,12,14,16,18

    # The C++ debug shows RIGHT eye as model 1
    # Let's use the C++ input landmarks for right eye
    CPP_RIGHT_EYE = {
        42: (560.8530, 833.9062),
        43: (583.5312, 812.9091),
        44: (611.5075, 810.8795),
        45: (635.8052, 825.4882),
        46: (615.4611, 834.7896),
        47: (587.8980, 837.3765)
    }

    # Need to fit PDM to get eye landmark 0 position
    # For now, let's just estimate based on the eye model structure
    # Landmark 0 in 28-point eye model is at the pupil/center area

    # Actually, we can use the fitted eye landmarks from the debug
    # Looking at cpp_eye_init_debug.txt for right eye (model 1)
    # We need to find or compute right eye landmark 0 position

    # Let me try loading the right eye PDM and fitting
    from pyclnf.core.eye_patch_expert import HierarchicalEyeModel, RIGHT_EYE_MAPPING

    eye_model = HierarchicalEyeModel('pyclnf/models')
    pdm = eye_model.pdm['right']

    target_points = np.array([CPP_RIGHT_EYE[i] for i in [42, 43, 44, 45, 46, 47]])
    params = eye_model._fit_eye_shape(target_points, RIGHT_EYE_MAPPING, 'right', main_rotation=None)
    eye_landmarks = pdm.params_to_landmarks_2d(params)

    print(f"\nPython right eye landmark 0: ({eye_landmarks[0, 0]:.2f}, {eye_landmarks[0, 1]:.2f})")
    print(f"C++ right eye landmark 0 (initial): (579.76, 809.13)")
    print(f"*** Position difference: dx={eye_landmarks[0, 0]-579.76:.2f}, dy={eye_landmarks[0, 1]-809.13:.2f} ***")

    # Extract patch for landmark 0 with window size 3
    ws = 3
    # Use C++ position instead of our computed one
    x, y = 579.76, 809.13  # C++ initial position
    # x, y = eye_landmarks[0]  # Python computed position
    patch_size = pe['height']
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

    # Extract center patch
    center_i, center_j = 1, 1
    patch = area_of_interest[center_i:center_i+patch_size, center_j:center_j+patch_size]

    print(f"\nPatch shape: {patch.shape}")
    print(f"Patch range: [{patch.min():.2f}, {patch.max():.2f}], mean={patch.mean():.2f}")

    # Don't flip - using exact C++ position instead
    print("\n*** Using C++ landmark position ***")

    # Normalize like C++
    features = patch.astype(np.float32)
    features_flat = features.flatten('F')
    feature_mean = np.mean(features_flat)
    features_centered = features_flat - feature_mean
    feature_norm = np.linalg.norm(features_centered)

    if feature_norm > 1e-10:
        normalized_features = features_centered / feature_norm
    else:
        normalized_features = features_centered

    print(f"\nNormalization:")
    print(f"  mean: {feature_mean:.6f}")
    print(f"  norm: {feature_norm:.6f}")

    print(f"\nPython normalized features (first 10):")
    for i in range(10):
        print(f"  [{i+1}]: {normalized_features[i]:.6f}")

    print(f"\nC++ normalized input (from debug):")
    cpp_features = [-0.019852, -0.022680, -0.025950, -0.026463, -0.019941,
                   -0.043181, -0.059916, -0.077854, 0.011430]
    for i, val in enumerate(cpp_features):
        print(f"  [{i+1}]: {val:.6f}")

    # Compute neuron responses
    print("\n" + "=" * 70)
    print("NEURON RESPONSES")
    print("=" * 70)

    total_response = 0.0
    sum_alphas = 0.0

    for i, neuron in enumerate(pe['neurons']):
        alpha = neuron['alpha']
        sum_alphas += alpha

        weights_flat = neuron['weights'].flatten('F') * neuron['norm_weights']
        sigmoid_input = neuron['bias'] + np.dot(weights_flat, normalized_features)
        sigmoid_output = (2.0 * alpha) / (1.0 + np.exp(-np.clip(sigmoid_input, -500, 500)))
        total_response += sigmoid_output

        print(f"Neuron {i}: alpha={alpha:.6f}, sigmoid_in={sigmoid_input:.6f}, sigmoid_out={sigmoid_output:.6f}")

    normalized_response = total_response / (2.0 * sum_alphas)

    print(f"\nPython results:")
    print(f"  Total response: {total_response:.6f}")
    print(f"  Sum alphas: {sum_alphas:.6f}")
    print(f"  Normalized: {normalized_response:.6f}")

    print(f"\nC++ results (from debug):")
    print(f"  Total response: 20.978935")
    print(f"  Sum alphas: 47.135536")
    print(f"  Normalized: 0.222538")

    if abs(normalized_response - 0.222538) < 0.01:
        print("\n*** MATCH! ***")
    else:
        print(f"\n*** MISMATCH: Python={normalized_response:.6f}, C++={0.222538:.6f} ***")
        print(f"*** Ratio: {0.222538 / normalized_response:.1f}x ***")

if __name__ == '__main__':
    main()
