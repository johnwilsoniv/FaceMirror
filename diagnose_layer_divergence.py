#!/usr/bin/env python3
"""
Find the exact layer where C++ and ONNX outputs diverge.
Run ONNX model layer-by-layer with extracted weights and compare to manual computation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load C++ ONet input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))

print("="*80)
print("LAYER-BY-LAYER MANUAL COMPUTATION")
print("="*80)

# Convert to CHW format for PyTorch
input_chw = np.transpose(cpp_input, (2, 0, 1))  # HWC -> CHW
x = torch.from_numpy(input_chw).unsqueeze(0)  # (1, 3, 48, 48)

print(f"\nInput shape: {x.shape}")
print(f"Input [0,0,0,:3]: {x[0,0,0,:3]}")

# ============================================================================
# LAYER 0: Conv1
# ============================================================================
conv1_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_weights.npy'))
conv1_biases = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer00_conv_biases.npy'))

conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=True)
conv1.weight.data = conv1_weights
conv1.bias.data = conv1_biases

x = conv1(x)
print(f"\n[Layer 0] Conv1 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 1: PReLU1
# ============================================================================
prelu1_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer01_prelu_weights.npy'))
prelu1_weights_reshaped = prelu1_weights.reshape(1, -1, 1, 1)

# Apply PReLU: output = max(x, 0) + weight * min(x, 0)
x = torch.max(x, torch.zeros_like(x)) + prelu1_weights_reshaped * torch.min(x, torch.zeros_like(x))

print(f"\n[Layer 1] PReLU1 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 2: MaxPool1
# ============================================================================
x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

print(f"\n[Layer 2] MaxPool1 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 3: Conv2
# ============================================================================
conv2_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer03_conv_weights.npy'))
conv2_biases = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer03_conv_biases.npy'))

conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=True)
conv2.weight.data = conv2_weights
conv2.bias.data = conv2_biases

x = conv2(x)
print(f"\n[Layer 3] Conv2 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 4: PReLU2
# ============================================================================
prelu2_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer04_prelu_weights.npy'))
prelu2_weights_reshaped = prelu2_weights.reshape(1, -1, 1, 1)

x = torch.max(x, torch.zeros_like(x)) + prelu2_weights_reshaped * torch.min(x, torch.zeros_like(x))

print(f"\n[Layer 4] PReLU2 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 5: MaxPool2
# ============================================================================
x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

print(f"\n[Layer 5] MaxPool2 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 6: Conv3
# ============================================================================
conv3_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer06_conv_weights.npy'))
conv3_biases = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer06_conv_biases.npy'))

conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True)
conv3.weight.data = conv3_weights
conv3.bias.data = conv3_biases

x = conv3(x)
print(f"\n[Layer 6] Conv3 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 7: PReLU3
# ============================================================================
prelu3_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer07_prelu_weights.npy'))
prelu3_weights_reshaped = prelu3_weights.reshape(1, -1, 1, 1)

x = torch.max(x, torch.zeros_like(x)) + prelu3_weights_reshaped * torch.min(x, torch.zeros_like(x))

print(f"\n[Layer 7] PReLU3 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 8: MaxPool3
# ============================================================================
x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

print(f"\n[Layer 8] MaxPool3 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 9: Conv4
# ============================================================================
conv4_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer09_conv_weights.npy'))
conv4_biases = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer09_conv_biases.npy'))

conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0, bias=True)
conv4.weight.data = conv4_weights
conv4.bias.data = conv4_biases

x = conv4(x)
print(f"\n[Layer 9] Conv4 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# LAYER 10: PReLU4
# ============================================================================
prelu4_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer10_prelu_weights.npy'))
prelu4_weights_reshaped = prelu4_weights.reshape(1, -1, 1, 1)

x = torch.max(x, torch.zeros_like(x)) + prelu4_weights_reshaped * torch.min(x, torch.zeros_like(x))

print(f"\n[Layer 10] PReLU4 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,0,0,0]: {x[0,0,0,0].item():.6f}")

# ============================================================================
# Flatten for FC layers
# ============================================================================
x = x.view(x.size(0), -1)
print(f"\nFlattened shape: {x.shape}")  # Should be (1, 1152)
print(f"  Sample [0,:5]: {x[0,:5]}")

# ============================================================================
# LAYER 11: FC1
# ============================================================================
fc1_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer11_fc_weights.npy'))
fc1_bias = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer11_fc_bias.npy'))

fc1 = nn.Linear(1152, 256, bias=True)
fc1.weight.data = fc1_weights
fc1.bias.data = fc1_bias

x = fc1(x)
print(f"\n[Layer 11] FC1 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,:5]: {x[0,:5]}")

# ============================================================================
# LAYER 12: PReLU5
# ============================================================================
prelu5_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer12_prelu_weights.npy'))

# For FC layer, PReLU weights are 1D (256,)
x = torch.max(x, torch.zeros_like(x)) + prelu5_weights * torch.min(x, torch.zeros_like(x))

print(f"\n[Layer 12] PReLU5 output:")
print(f"  Shape: {x.shape}")
print(f"  Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
print(f"  Sample [0,:5]: {x[0,:5]}")

# ============================================================================
# LAYER 13: FC2 (Final Output)
# ============================================================================
fc2_weights = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer13_fc_weights.npy'))
fc2_bias = torch.from_numpy(np.load('cpp_mtcnn_weights/onet/onet_layer13_fc_bias.npy'))

fc2 = nn.Linear(256, 16, bias=True)
fc2.weight.data = fc2_weights
fc2.bias.data = fc2_bias

x = fc2(x)
print(f"\n[Layer 13] FC2 (Final) output:")
print(f"  Shape: {x.shape}")
print(f"  logit[0]: {x[0,0].item():.6f}")
print(f"  logit[1]: {x[0,1].item():.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print(f"\n" + "="*80)
print(f"FINAL COMPARISON")
print(f"="*80)
print(f"Manual computation:")
print(f"  logit[0]: {x[0,0].item():.6f}")
print(f"  logit[1]: {x[0,1].item():.6f}")

print(f"\nC++ .dat model (from /tmp/cpp_onet_debug.txt):")
print(f"  logit[0]: -3.41372")
print(f"  logit[1]:  3.41272")

print(f"\nDifference:")
print(f"  Δlogit[0]: {-3.41372 - x[0,0].item():.6f}")
print(f"  Δlogit[1]: {3.41272 - x[0,1].item():.6f}")

if abs(-3.41372 - x[0,0].item()) < 0.01:
    print(f"\n✓ MANUAL COMPUTATION MATCHES C++!")
    print(f"This means weight extraction is correct.")
else:
    print(f"\n❌ MANUAL COMPUTATION DIFFERS FROM C++")
    print(f"Check layer outputs above to find where divergence starts.")
