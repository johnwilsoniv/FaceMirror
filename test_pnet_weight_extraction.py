#!/usr/bin/env python3
"""
Test if PNet weight extraction is correct by comparing C++ and Python layer 0 outputs.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Load the C++ PNet input saved during detection
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)  # HWC format from C++

print(f"C++ PNet Input shape: {cpp_input.shape}")
print(f"  Sample pixels at (0,0): {cpp_input[0,0,:]}")

# Load C++ PNet layer 0 output
cpp_logit0 = np.fromfile('/tmp/cpp_pnet_logit0_scale0.bin', dtype=np.float32)
cpp_logit1 = np.fromfile('/tmp/cpp_pnet_logit1_scale0.bin', dtype=np.float32)
cpp_logit0 = cpp_logit0.reshape(187, 103)
cpp_logit1 = cpp_logit1.reshape(187, 103)

print(f"\nC++ PNet Final Output:")
print(f"  Logit0 shape: {cpp_logit0.shape}")
print(f"  Logit0 at (0,0): {cpp_logit0[0,0]}")
print(f"  Logit1 at (0,0): {cpp_logit1[0,0]}")

# Load extracted PNet weights
pnet_conv1_weights = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_weights.npy')
pnet_conv1_biases = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_biases.npy')

print(f"\nExtracted PNet Layer 0 Weights:")
print(f"  Weight shape: {pnet_conv1_weights.shape}")
print(f"  Bias shape: {pnet_conv1_biases.shape}")
print(f"  Weight stats: min={pnet_conv1_weights.min():.6f}, max={pnet_conv1_weights.max():.6f}")

# Convert C++ input to PyTorch format (HWC -> CHW)
py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 384, 216)
print(f"\nPyTorch input shape: {py_input.shape}")

# Run PyTorch conv with extracted weights
conv_weights = torch.from_numpy(pnet_conv1_weights).float()
conv_biases = torch.from_numpy(pnet_conv1_biases).float()

output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)
print(f"\nPyTorch Conv Output shape: {output.shape}")  # Should be (1, 10, 382, 214)

# Extract just the first 2 channels (classification logits)
# PNet layer 0 outputs 10 channels, but we need to check if they match the C++ final logits
# Actually, C++ final logits come from the LAST layer (layer 7), not layer 0!
# Let me check what layer 0 should output...

# Actually, the C++ debug outputs we have are the FINAL PNet outputs (after all layers),
# not layer 0 outputs. So we can't directly compare here.
# We need to run the FULL PNet forward pass.

# Load the full PNet model
from convert_mtcnn_to_onnx import PNet, load_weights_to_model

pnet = PNet()
load_weights_to_model(pnet, 'cpp_mtcnn_weights/pnet', 'pnet')
pnet.eval()

# Run Python PNet
with torch.no_grad():
    py_output = pnet(py_input)

print(f"\nPython PNet Full Forward Pass:")
print(f"  Output shape: {py_output.shape}")  # Should be (1, 6, H, W)

# Extract classification logits (channels 0 and 1)
py_logit0 = py_output[0, 0, :, :].numpy()  # Non-face
py_logit1 = py_output[0, 1, :, :].numpy()  # Face

print(f"  Logit0 shape: {py_logit0.shape}")
print(f"  Logit0 at (0,0): {py_logit0[0,0]}")
print(f"  Logit1 at (0,0): {py_logit1[0,0]}")

# Compare with C++
print(f"\n{'='*80}")
print(f"COMPARISON:")
print(f"{'='*80}")
print(f"C++ logit0[0,0]:  {cpp_logit0[0,0]:.6f}")
print(f"Python logit0[0,0]: {py_logit0[0,0]:.6f}")
print(f"Difference: {abs(cpp_logit0[0,0] - py_logit0[0,0]):.6f}")

print(f"\nC++ logit1[0,0]:  {cpp_logit1[0,0]:.6f}")
print(f"Python logit1[0,0]: {py_logit1[0,0]:.6f}")
print(f"Difference: {abs(cpp_logit1[0,0] - py_logit1[0,0]):.6f}")

# Overall statistics
diff0 = np.abs(cpp_logit0 - py_logit0)
diff1 = np.abs(cpp_logit1 - py_logit1)

print(f"\nOverall Logit0 Differences:")
print(f"  Mean: {diff0.mean():.6f}")
print(f"  Max:  {diff0.max():.6f}")

print(f"\nOverall Logit1 Differences:")
print(f"  Mean: {diff1.mean():.6f}")
print(f"  Max:  {diff1.max():.6f}")

if diff0.max() < 0.01 and diff1.max() < 0.01:
    print(f"\n✅ WEIGHTS ARE CORRECT! (differences < 0.01)")
elif diff0.max() < 0.1 and diff1.max() < 0.1:
    print(f"\n⚠️  MOSTLY CORRECT (differences < 0.1)")
else:
    print(f"\n❌ WEIGHTS ARE WRONG! Large differences detected")
