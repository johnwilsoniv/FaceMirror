#!/usr/bin/env python3
"""
Debug script to verify normalization computation matches C++ exactly.

We'll use the EXACT same raw patch values from C++ debug and verify
Python produces the same normalized output and sigmoid input.
"""

import numpy as np
import sys
sys.path.insert(0, 'pyclnf')

def main():
    # Use C++ raw patch values from cpp_ccnf_neuron_debug.txt (second model section)
    # This is the model that has same params as Python eye model
    # We need to recreate the exact input to compare

    # From the earlier Python eye debug output, let's use those exact values
    # Raw area_of_interest values from /tmp/python_eye_lm8_debug.txt
    raw_patch = np.array([
        [49, 49, 54, 59, 62, 67, 73, 76, 80, 84, 87],
        [53, 54, 58, 61, 63, 69, 73, 78, 81, 81, 81],
        [56, 56, 60, 63, 66, 75, 82, 80, 79, 79, 81],
        [60, 58, 59, 64, 69, 80, 80, 81, 79, 79, 81],
        [65, 62, 63, 72, 81, 90, 91, 81, 81, 84, 79],
        [70, 66, 67, 75, 87, 99, 100, 84, 84, 90, 84],
        [75, 73, 76, 90, 96, 101, 102, 93, 90, 84, 79],
        [81, 79, 91, 104, 102, 101, 103, 101, 96, 95, 81],
        [89, 92, 103, 111, 109, 108, 111, 109, 104, 100, 91],
        [101, 109, 117, 118, 113, 117, 117, 108, 101, 95, 86],
        [113, 120, 126, 126, 127, 137, 135, 118, 107, 98, 87]
    ], dtype=np.float32)

    print("Testing Python normalization vs C++ expected values")
    print("="*60)

    # Python normalization (column-major flattening)
    features_flat = raw_patch.flatten('F')  # Column-major like C++
    feature_mean = np.mean(features_flat)
    features_centered = features_flat - feature_mean
    feature_norm = np.linalg.norm(features_centered)
    normalized_features = features_centered / feature_norm

    print(f"\nFeature mean: {feature_mean:.6f}")
    print(f"Feature norm: {feature_norm:.6f}")

    print(f"\nNormalized input (first 12 values):")
    print(f"  [0]: 1.00000000  (bias term)")
    for i in range(11):
        print(f"  [{i+1}]: {normalized_features[i]:.8f}")

    # Now test with model parameters from Python eye model debug
    # Neuron 0: alpha=6.818644, bias=-16.442822, norm_w=67.509662
    alpha = 6.818644
    bias = -16.442822
    norm_w = 67.509662

    # Load actual weights from the exported model
    import os
    patch_dir = 'pyclnf/models/exported_eye_ccnf_left/scale_1.00/view_00/patch_08/'
    neuron_data = np.load(patch_dir + 'neuron_00.npz')
    weights = neuron_data['weights']

    print(f"\nLoaded weights shape: {weights.shape}")
    print(f"weights[0,0]: {weights[0,0]:.8f}")

    # Compute sigmoid input
    weights_flat = weights.flatten('F') * norm_w
    sigmoid_input = bias + np.dot(weights_flat, normalized_features)

    print(f"\nSigmoid input: {sigmoid_input:.6f}")
    print(f"Expected from Python debug: -21.607064")

    # Check if weights match what's in the debug
    # From /tmp/python_eye_lm8_debug.txt:
    # Neuron 0 raw weights (first 10, before norm_weights scaling):
    #   [0]: 0.13774830
    print(f"\nExpected raw weight[0]: 0.13774830")
    print(f"Actual raw weight[0]: {weights.flatten('F')[0]:.8f}")

    # Now let's compare with C++ second section
    print("\n" + "="*60)
    print("COMPARING WITH C++ VALUES")
    print("="*60)

    # C++ second section had:
    # Neuron 0: sigmoid_input=3.946698
    # With the SAME model params (alpha=7.045101, bias=-20.294249, norm_w=61.948643)

    # Wait - those are DIFFERENT params than the Python eye model!
    # Python eye model has: alpha=6.818644, bias=-16.442822, norm_w=67.509662
    # C++ second section has: alpha=7.045101, bias=-20.294249, norm_w=61.948643

    # Let me check the C++ params more carefully...
    # Actually from python_ccnf_neuron_debug_call0.txt (the first file in summary):
    # Neuron 0: alpha=7.045101, bias=-20.294249, norm_w=61.948643, sigmoid_input=-3.267370

    # That's a DIFFERENT model entirely! The C++ debug in call0 was using main CCNF model,
    # not the eye model.

    print("\nIMPORTANT FINDING:")
    print("The C++ debug files show DIFFERENT model parameters than Python eye model!")
    print("This suggests C++ eye model uses different exported weights than Python.")
    print("\nPython eye patch_08 Neuron 0:")
    print(f"  alpha: {neuron_data['alpha']}")
    print(f"  bias: {neuron_data['bias']}")
    print(f"  norm_weights: {neuron_data['norm_weights']}")

    # Let's check if there might be a view issue - are we loading correct view?
    # Or a scale issue?
    print("\n" + "="*60)
    print("CHECKING MODEL EXPORT")
    print("="*60)

    # Check what C++ uses for eye model
    # The C++ eye model might be using a different scale or view

    # Also check: is the Python response computation different than C++ in some way?
    # Let's verify the full dot product manually

    dot_product = np.dot(weights_flat, normalized_features)
    print(f"\nDot product (weights_flat · normalized): {dot_product:.6f}")
    print(f"Bias: {bias:.6f}")
    print(f"Sigmoid input = bias + dot = {sigmoid_input:.6f}")

    # Sigmoid output
    sigmoid_output = (2.0 * alpha) / (1.0 + np.exp(-sigmoid_input))
    print(f"Sigmoid output: {sigmoid_output:.6f}")

if __name__ == '__main__':
    main()
