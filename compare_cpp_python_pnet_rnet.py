#!/usr/bin/env python3
"""
Compare C++ and Python PNet/RNet outputs numerically to verify they match.
"""

import cv2
import numpy as np
from cpp_mtcnn_detector import CPPMTCNNDetector

def compare_pnet_outputs():
    """
    Compare PNet outputs between C++ and Python at scale 0.2.
    """
    print(f"\n{'='*80}")
    print(f"PNet Comparison: C++ vs Python")
    print(f"{'='*80}\n")

    # Load C++ outputs
    cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
    cpp_logit0 = np.fromfile('/tmp/cpp_pnet_logit0_scale0.bin', dtype=np.float32)
    cpp_logit1 = np.fromfile('/tmp/cpp_pnet_logit1_scale0.bin', dtype=np.float32)

    # C++ output shape from debug: 187x103
    cpp_logit0 = cpp_logit0.reshape(187, 103)
    cpp_logit1 = cpp_logit1.reshape(187, 103)

    # C++ input shape: 216x384x3 (HWC)
    cpp_input = cpp_input.reshape(384, 216, 3)  # OpenCV format is HWC

    print(f"C++ PNet Outputs:")
    print(f"  Input shape: {cpp_input.shape}")
    print(f"  Logit0 (non-face) shape: {cpp_logit0.shape}")
    print(f"  Logit1 (face) shape: {cpp_logit1.shape}")
    print(f"  Sample logits at (0,0): non-face={cpp_logit0[0,0]:.6f}, face={cpp_logit1[0,0]:.6f}")
    print(f"  Sample logits at (1,1): non-face={cpp_logit0[1,1]:.6f}, face={cpp_logit1[1,1]:.6f}")

    # Compute probability from C++ logits
    cpp_prob = 1.0 / (1.0 + np.exp(cpp_logit0 - cpp_logit1))
    print(f"  Probability range: [{cpp_prob.min():.6f}, {cpp_prob.max():.6f}]\n")

    # Run Python PNet at same scale
    img = cv2.imread("calibration_frames/patient1_frame1.jpg")
    h, w = img.shape[:2]

    # C++ uses scale 0.2 for first scale
    scale = 0.2
    h_scaled = int(np.ceil(h * scale))
    w_scaled = int(np.ceil(w * scale))

    print(f"Running Python PNet at scale {scale}:")
    print(f"  Original image: {w}x{h}")
    print(f"  Scaled image: {w_scaled}x{h_scaled}")

    # Resize and preprocess
    img_scaled = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_LINEAR)
    img_preprocessed = img_scaled.astype(np.float32)
    img_preprocessed = (img_preprocessed - 127.5) * 0.0078125

    # Save Python input for comparison
    np.save('/tmp/python_pnet_input_scale0.npy', img_preprocessed)

    # Convert to CHW for ONNX
    img_chw = np.transpose(img_preprocessed, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, 0)

    # Run PNet
    detector = CPPMTCNNDetector()
    output = detector.pnet.run(None, {detector.pnet.get_inputs()[0].name: img_batch})[0]

    # Extract logits
    py_logit0 = output[0, 0, :, :]  # Non-face
    py_logit1 = output[0, 1, :, :]  # Face

    print(f"\nPython PNet Outputs:")
    print(f"  Input shape: {img_preprocessed.shape}")
    print(f"  Logit0 (non-face) shape: {py_logit0.shape}")
    print(f"  Logit1 (face) shape: {py_logit1.shape}")
    print(f"  Sample logits at (0,0): non-face={py_logit0[0,0]:.6f}, face={py_logit1[0,0]:.6f}")
    print(f"  Sample logits at (1,1): non-face={py_logit0[1,1]:.6f}, face={py_logit1[1,1]:.6f}")

    # Compute probability from Python logits
    py_prob = 1.0 / (1.0 + np.exp(py_logit0 - py_logit1))
    print(f"  Probability range: [{py_prob.min():.6f}, {py_prob.max():.6f}]\n")

    # Compare shapes
    print(f"{'='*80}")
    print(f"Shape Comparison:")
    print(f"{'='*80}")
    if cpp_logit0.shape != py_logit0.shape:
        print(f"  ❌ SHAPE MISMATCH!")
        print(f"     C++:    {cpp_logit0.shape}")
        print(f"     Python: {py_logit0.shape}")
        return False
    else:
        print(f"  ✓ Shapes match: {cpp_logit0.shape}\n")

    # Compare logits numerically
    print(f"{'='*80}")
    print(f"Numerical Comparison:")
    print(f"{'='*80}")

    # Logit0 (non-face)
    diff0 = np.abs(cpp_logit0 - py_logit0)
    print(f"\nLogit0 (non-face) differences:")
    print(f"  Mean abs diff: {diff0.mean():.6f}")
    print(f"  Max abs diff:  {diff0.max():.6f}")
    print(f"  Median abs diff: {np.median(diff0):.6f}")
    print(f"  % within 0.001: {(diff0 < 0.001).sum() / diff0.size * 100:.1f}%")
    print(f"  % within 0.01:  {(diff0 < 0.01).sum() / diff0.size * 100:.1f}%")
    print(f"  % within 0.1:   {(diff0 < 0.1).sum() / diff0.size * 100:.1f}%")

    # Logit1 (face)
    diff1 = np.abs(cpp_logit1 - py_logit1)
    print(f"\nLogit1 (face) differences:")
    print(f"  Mean abs diff: {diff1.mean():.6f}")
    print(f"  Max abs diff:  {diff1.max():.6f}")
    print(f"  Median abs diff: {np.median(diff1):.6f}")
    print(f"  % within 0.001: {(diff1 < 0.001).sum() / diff1.size * 100:.1f}%")
    print(f"  % within 0.01:  {(diff1 < 0.01).sum() / diff1.size * 100:.1f}%")
    print(f"  % within 0.1:   {(diff1 < 0.1).sum() / diff1.size * 100:.1f}%")

    # Probability comparison
    diff_prob = np.abs(cpp_prob - py_prob)
    print(f"\nProbability (after softmax) differences:")
    print(f"  Mean abs diff: {diff_prob.mean():.6f}")
    print(f"  Max abs diff:  {diff_prob.max():.6f}")
    print(f"  Median abs diff: {np.median(diff_prob):.6f}")
    print(f"  % within 0.001: {(diff_prob < 0.001).sum() / diff_prob.size * 100:.1f}%")
    print(f"  % within 0.01:  {(diff_prob < 0.01).sum() / diff_prob.size * 100:.1f}%")

    # Show worst mismatches
    print(f"\nWorst 5 mismatches (by probability):")
    flat_diff = diff_prob.flatten()
    worst_indices = np.argsort(flat_diff)[-5:][::-1]
    for rank, idx in enumerate(worst_indices):
        y = idx // diff_prob.shape[1]
        x = idx % diff_prob.shape[1]
        cpp_val = cpp_prob[y, x]
        py_val = py_prob[y, x]
        diff_val = flat_diff[idx]
        print(f"  #{rank+1}: ({x:3d},{y:3d}) C++={cpp_val:.6f}, Py={py_val:.6f}, diff={diff_val:.6f}")

    # Overall assessment
    print(f"\n{'='*80}")
    print(f"Assessment:")
    print(f"{'='*80}")

    # Check if outputs match within tolerance
    logit_match = (diff0.max() < 0.01) and (diff1.max() < 0.01)
    prob_match = diff_prob.max() < 0.01

    if logit_match and prob_match:
        print(f"  ✓ PNet outputs MATCH (differences < 0.01)")
        return True
    elif diff0.max() < 0.1 and diff1.max() < 0.1:
        print(f"  ⚠ PNet outputs MOSTLY match (differences < 0.1)")
        print(f"     Some numerical precision issues, but should be functionally equivalent")
        return True
    else:
        print(f"  ❌ PNet outputs DO NOT MATCH!")
        print(f"     Large differences detected - there's a bug!")
        return False

def compare_rnet_outputs():
    """
    Compare RNet preprocessing from debug files.
    """
    print(f"\n{'='*80}")
    print(f"RNet Comparison: C++ vs Python")
    print(f"{'='*80}\n")

    # Read C++ RNet debug
    with open('/tmp/cpp_rnet_debug.txt', 'r') as f:
        cpp_output = f.read()

    print("C++ RNet Debug Output:")
    print(cpp_output)

    # Read Python RNet debug
    try:
        with open('/tmp/python_rnet_debug.txt', 'r') as f:
            py_output = f.read()

        print("\nPython RNet Debug Output:")
        print(py_output)
    except FileNotFoundError:
        print("\n⚠ Python RNet debug file not found - need to run Python detector with debug enabled")

if __name__ == "__main__":
    # Compare PNet
    pnet_match = compare_pnet_outputs()

    # Compare RNet (basic comparison of debug text)
    compare_rnet_outputs()

    print(f"\n{'='*80}")
    print(f"FINAL VERDICT:")
    print(f"{'='*80}")

    if pnet_match:
        print(f"✓ PNet outputs match between C++ and Python")
        print(f"  Networks are computing the same thing!")
    else:
        print(f"❌ PNet outputs DO NOT match")
        print(f"   Investigation needed to find the bug!")
