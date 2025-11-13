#!/usr/bin/env python3
"""
Find the canary: Minimal test to identify where PyCLNF diverges from OpenFace C++.

This script extracts a SINGLE patch at a SINGLE position and computes its response,
comparing every step with what OpenFace C++ should produce.
"""

import cv2
import numpy as np
from pyclnf import CLNF
from pyclnf.core.optimizer import NURLMSOptimizer

# Load test frame
video_path = 'Patient Data/Normal Cohort/IMG_0433.MOV'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_bbox = (241, 555, 532, 532)

# Initialize CLNF (just to get initial landmarks)
clnf = CLNF(model_dir='pyclnf/models', max_iterations=1)
params = clnf.pdm.init_params(face_bbox)
landmarks = clnf.pdm.params_to_landmarks_2d(params)

# Pick landmark 48 (mouth corner - showed 3.6px offset)
landmark_idx = 48
lm_x, lm_y = landmarks[landmark_idx]

# Get patch expert
patch_experts = clnf._get_patch_experts(view_idx=0, scale=0.25)
patch_expert = patch_experts[landmark_idx]

print("=" * 80)
print("CANARY TEST: Single Patch Response Computation")
print("=" * 80)
print(f"Landmark {landmark_idx} at ({lm_x:.1f}, {lm_y:.1f})")
print(f"Patch expert: {patch_expert.width}x{patch_expert.height}")
print(f"Number of neurons: {len(patch_expert.neurons)}")
print()

# Extract patch at landmark center
patch_half_w = patch_expert.width // 2
patch_half_h = patch_expert.height // 2

x1 = int(lm_x) - patch_half_w
y1 = int(lm_y) - patch_half_h
x2 = x1 + patch_expert.width
y2 = y1 + patch_expert.height

patch = gray[y1:y2, x1:x2]

print(f"Extracted patch from image:")
print(f"  Bounds: ({x1}, {y1}) to ({x2}, {y2})")
print(f"  Patch shape: {patch.shape}")
print(f"  Patch dtype: {patch.dtype}")
print(f"  Patch value range: [{patch.min()}, {patch.max()}]")
print(f"  Patch mean: {patch.mean():.2f}")
print(f"  Patch std: {patch.std():.2f}")
print()

# Convert to float [0, 1] (as done in _extract_features)
patch_float = patch.astype(np.float32) / 255.0

print(f"Normalized patch (float32 [0,1]):")
print(f"  Value range: [{patch_float.min():.6f}, {patch_float.max():.6f}]")
print(f"  Mean: {patch_float.mean():.6f}")
print(f"  Std: {patch_float.std():.6f}")
print()

# Compute response using patch expert
response = patch_expert.compute_response(patch)

print(f"Total response from patch expert: {response:.6f}")
print()

# Now break down neuron-by-neuron
print("=" * 80)
print("NEURON-BY-NEURON BREAKDOWN")
print("=" * 80)

total_response_manual = 0.0

for i, neuron in enumerate(patch_expert.neurons):
    if abs(neuron['alpha']) < 1e-4:
        continue

    weights = neuron['weights']
    bias = neuron['bias']
    alpha = neuron['alpha']
    norm_weights = neuron['norm_weights']

    # Resize features to match weights if needed
    features = patch_float
    if features.shape != weights.shape:
        features = cv2.resize(features, (weights.shape[1], weights.shape[0]))

    # Compute means
    weight_mean = np.mean(weights)
    feature_mean = np.mean(features)

    # Center the data (matches TM_CCOEFF_NORMED preprocessing)
    weights_centered = weights - weight_mean
    features_centered = features - feature_mean

    # Compute norms
    weight_norm = np.linalg.norm(weights_centered)
    feature_norm = np.linalg.norm(features_centered)

    # Compute normalized cross-correlation
    if weight_norm > 1e-10 and feature_norm > 1e-10:
        correlation = np.sum(weights_centered * features_centered) / (weight_norm * feature_norm)
    else:
        correlation = 0.0

    # Apply OpenFace formula
    sigmoid_input = correlation * norm_weights + bias

    # Sigmoid
    if sigmoid_input >= 0:
        sigmoid_val = 1 / (1 + np.exp(-sigmoid_input))
    else:
        sigmoid_val = np.exp(sigmoid_input) / (1 + np.exp(sigmoid_input))

    neuron_response = (2.0 * alpha) * sigmoid_val
    total_response_manual += neuron_response

    if i < 5:  # Print first 5 neurons
        print(f"\nNeuron {i}:")
        print(f"  Alpha: {alpha:.6f}")
        print(f"  Bias: {bias:.6f}")
        print(f"  Norm weights: {norm_weights:.6f}")
        print(f"  Weight mean: {weight_mean:.6f}, Feature mean: {feature_mean:.6f}")
        print(f"  Weight norm: {weight_norm:.6f}, Feature norm: {feature_norm:.6f}")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Sigmoid input: {sigmoid_input:.6f}")
        print(f"  Sigmoid output: {sigmoid_val:.6f}")
        print(f"  Neuron response: {neuron_response:.6f}")

print()
print("=" * 80)
print(f"Total response (manual sum): {total_response_manual:.6f}")
print(f"Total response (patch_expert.compute_response): {response:.6f}")
print(f"Difference: {abs(response - total_response_manual):.10f}")
print("=" * 80)

# Now compute a response map for an 11x11 window
print()
print("=" * 80)
print("RESPONSE MAP COMPUTATION (11x11 window)")
print("=" * 80)

window_size = 11
half_window = window_size // 2

response_map = np.zeros((window_size, window_size))

start_x = int(lm_x) - half_window
start_y = int(lm_y) - half_window

for i in range(window_size):
    for j in range(window_size):
        patch_x = start_x + j
        patch_y = start_y + i

        # Extract patch
        y1 = patch_y - patch_half_h
        y2 = y1 + patch_expert.height
        x1 = patch_x - patch_half_w
        x2 = x1 + patch_expert.width

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > gray.shape[1] or y2 > gray.shape[0]:
            response_map[i, j] = -1e10
            continue

        patch = gray[y1:y2, x1:x2]
        response_map[i, j] = patch_expert.compute_response(patch)

# Find peak
peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
peak_y, peak_x = peak_idx
center = (window_size - 1) / 2.0
offset_x = peak_x - center
offset_y = peak_y - center
offset_dist = np.sqrt(offset_x**2 + offset_y**2)

print(f"Response map shape: {response_map.shape}")
print(f"Response map range: [{response_map.min():.6f}, {response_map.max():.6f}]")
print(f"Peak location: ({peak_x}, {peak_y})")
print(f"Center (landmark position): ({center:.1f}, {center:.1f})")
print(f"Peak offset: ({offset_x:+.1f}, {offset_y:+.1f}) - distance={offset_dist:.2f} pixels")
print()

# Print the actual response map values
print("Response map values (center 5x5):")
for i in range(window_size//2 - 2, window_size//2 + 3):
    row_str = ""
    for j in range(window_size//2 - 2, window_size//2 + 3):
        val = response_map[i, j]
        if i == int(center) and j == int(center):
            row_str += f"[{val:6.3f}] "  # Center position
        elif i == peak_y and j == peak_x:
            row_str += f"*{val:6.3f}* "  # Peak position
        else:
            row_str += f" {val:6.3f}  "
    print(row_str)

print()
print("Legend: [center], *peak*")
print("=" * 80)
