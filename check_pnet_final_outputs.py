#!/usr/bin/env python3
"""
Check if PNet FINAL outputs (classification logits) match between C++ and Python.
This is what actually matters for detection!
"""

import numpy as np
from convert_mtcnn_to_onnx import PNet, load_weights_to_model
import torch

# Load C++ PNet input
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)  # HWC

# Load C++ PNet FINAL outputs (after all 8 layers)
cpp_logit0 = np.fromfile('/tmp/cpp_pnet_logit0_scale0.bin', dtype=np.float32)
cpp_logit1 = np.fromfile('/tmp/cpp_pnet_logit1_scale0.bin', dtype=np.float32)
cpp_logit0 = cpp_logit0.reshape(187, 103)
cpp_logit1 = cpp_logit1.reshape(187, 103)

print("="*80)
print("PNET FINAL OUTPUT COMPARISON")
print("="*80)

print(f"\nC++ FINAL outputs (layer 7 - after all processing):")
print(f"  Logit0 (non-face) shape: {cpp_logit0.shape}")
print(f"  Logit1 (face) shape: {cpp_logit1.shape}")
print(f"  Sample at [0,0]:")
print(f"    Logit0 (non-face): {cpp_logit0[0,0]:.6f}")
print(f"    Logit1 (face): {cpp_logit1[0,0]:.6f}")

# Run Python PNet (full forward pass through all 8 layers)
print(f"\nRunning Python PNet (all layers)...")
pnet = PNet()
load_weights_to_model(pnet, 'cpp_mtcnn_weights/pnet', 'pnet')
pnet.eval()

py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
with torch.no_grad():
    py_output = pnet(py_input)

# Extract final classification logits (channels 0 and 1 of final output)
py_logit0 = py_output[0, 0, :, :].numpy()
py_logit1 = py_output[0, 1, :, :].numpy()

print(f"\nPython FINAL outputs:")
print(f"  Logit0 (non-face) shape: {py_logit0.shape}")
print(f"  Logit1 (face) shape: {py_logit1.shape}")
print(f"  Sample at [0,0]:")
print(f"    Logit0 (non-face): {py_logit0[0,0]:.6f}")
print(f"    Logit1 (face): {py_logit1[0,0]:.6f}")

# Compare
diff0 = np.abs(cpp_logit0 - py_logit0)
diff1 = np.abs(cpp_logit1 - py_logit1)

print(f"\n{'='*80}")
print(f"FINAL OUTPUT DIFFERENCES:")
print(f"{'='*80}")
print(f"Logit0 (non-face):")
print(f"  Mean: {diff0.mean():.6f}")
print(f"  Max:  {diff0.max():.6f}")
print(f"  At [0,0]: {abs(cpp_logit0[0,0] - py_logit0[0,0]):.6f}")

print(f"\nLogit1 (face):")
print(f"  Mean: {diff1.mean():.6f}")
print(f"  Max:  {diff1.max():.6f}")
print(f"  At [0,0]: {abs(cpp_logit1[0,0] - py_logit1[0,0]):.6f}")

# Compute probabilities (softmax)
cpp_prob = 1.0 / (1.0 + np.exp(cpp_logit0 - cpp_logit1))
py_prob = 1.0 / (1.0 + np.exp(py_logit0 - py_logit1))

prob_diff = np.abs(cpp_prob - py_prob)

print(f"\nProbability (after softmax):")
print(f"  Mean: {prob_diff.mean():.6f}")
print(f"  Max:  {prob_diff.max():.6f}")
print(f"  % within 0.01: {100*(prob_diff < 0.01).sum()/prob_diff.size:.1f}%")
print(f"  % within 0.001: {100*(prob_diff < 0.001).sum()/prob_diff.size:.1f}%")

print(f"\n{'='*80}")
print(f"CONCLUSION:")
print(f"{'='*80}")
if diff0.max() < 1.0 and diff1.max() < 1.0:
    print(f"✅ Final outputs are CLOSE (max diff < 1.0)")
    print(f"   The intermediate layer 0 divergence doesn't propagate fully!")
else:
    print(f"❌ Final outputs DIVERGE significantly")
    print(f"   The layer 0 bug propagates through the network!")
