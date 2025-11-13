#!/usr/bin/env python3
"""
Compare C++ and ONNX ONet layer-by-layer to find divergence point.
"""

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# Load C++ ONet input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))

print("="*80)
print("LAYER-BY-LAYER COMPARISON: C++ vs ONNX")
print("="*80)

# Load ONNX model and extract intermediate outputs
sess = ort.InferenceSession('cpp_mtcnn_onnx/onet.onnx', providers=['CPUExecutionProvider'])

# Prepare input in CHW format for ONNX
onnx_input = np.transpose(cpp_input, (2, 0, 1))  # HWC -> CHW
onnx_input = np.expand_dims(onnx_input, 0)  # (1, 3, 48, 48)

print(f"\nInput tensor:")
print(f"  Shape (ONNX format): {onnx_input.shape}")
print(f"  Sample [0,0,0,:5]: {onnx_input[0, 0, 0, :5]}")
print(f"  Sample [0,1,0,:5]: {onnx_input[0, 1, 0, :5]}")
print(f"  Sample [0,2,0,:5]: {onnx_input[0, 2, 0, :5]}")
print(f"  Range: [{onnx_input.min():.6f}, {onnx_input.max():.6f}]")

# Get ONNX output
output = sess.run(None, {'input': onnx_input})[0]

print(f"\nONNX Final Output (logits):")
print(f"  logit[0]: {output[0, 0]:.6f}")
print(f"  logit[1]: {output[0, 1]:.6f}")

print(f"\nC++ Final Output (from /tmp/cpp_onet_debug.txt):")
print(f"  logit[0]: -3.41372")
print(f"  logit[1]:  3.41272")

print(f"\nDifference:")
print(f"  Δlogit[0]: {-3.41372 - output[0, 0]:.6f}")
print(f"  Δlogit[1]: {3.41272 - output[0, 1]:.6f}")

# Now let's manually run through the first layer to check
print(f"\n{'='*80}")
print(f"TESTING FIRST CONV LAYER MANUALLY")
print(f"{'='*80}")

# Load first conv layer weights from extracted data
conv1_weights = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy')
conv1_biases = np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy')

print(f"\nConv1 Layer Weights:")
print(f"  Shape: {conv1_weights.shape}")  # Should be (32, 3, 3, 3) for ONNX
print(f"  Range: [{conv1_weights.min():.6f}, {conv1_weights.max():.6f}]")
print(f"  Sample weight [0,0,0,0]: {conv1_weights[0,0,0,0]:.6f}")

print(f"\nConv1 Layer Biases:")
print(f"  Shape: {conv1_biases.shape}")  # Should be (32,)
print(f"  Range: [{conv1_biases.min():.6f}, {conv1_biases.max():.6f}]")
print(f"  Sample bias [0]: {conv1_biases[0]:.6f}")

# Manual convolution using PyTorch
conv1_manual = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv1_manual.weight.data = torch.from_numpy(conv1_weights)
conv1_manual.bias.data = torch.from_numpy(conv1_biases)

input_tensor = torch.from_numpy(onnx_input)
conv1_output = conv1_manual(input_tensor)

print(f"\nManual Conv1 Output:")
print(f"  Shape: {conv1_output.shape}")  # Should be (1, 32, 46, 46)
print(f"  Range: [{conv1_output.min().item():.6f}, {conv1_output.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {conv1_output[0,0,0,0].item():.6f}")

# Load PReLU weights
prelu1_weights = np.load('cpp_mtcnn_weights/onet/onet_layer01_prelu_weights.npy')
print(f"\nPReLU1 Weights:")
print(f"  Shape: {prelu1_weights.shape}")
print(f"  Range: [{prelu1_weights.min():.6f}, {prelu1_weights.max():.6f}]")

# Apply PReLU
prelu1_weights_reshaped = prelu1_weights.reshape(1, -1, 1, 1)
prelu1_output = torch.max(conv1_output, torch.zeros_like(conv1_output)) + \
                torch.from_numpy(prelu1_weights_reshaped) * torch.min(conv1_output, torch.zeros_like(conv1_output))

print(f"\nAfter PReLU1:")
print(f"  Shape: {prelu1_output.shape}")
print(f"  Range: [{prelu1_output.min().item():.6f}, {prelu1_output.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {prelu1_output[0,0,0,0].item():.6f}")

print(f"\n{'='*80}")
print(f"CONCLUSION")
print(f"{'='*80}")
print(f"If manual computation matches ONNX, then ONNX conversion is correct.")
print(f"If it doesn't match, there's a weight loading or format issue.")
