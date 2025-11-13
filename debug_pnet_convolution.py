#!/usr/bin/env python3
"""
Manual convolution computation to debug PNet layer 0.
Computes the first output value manually to verify correctness.
"""

import numpy as np

# Load C++ PNet input
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)  # HWC

# Load extracted PNet layer 0 weights
weights = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_weights.npy')  # (10, 3, 3, 3)
biases = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_biases.npy')   # (10,)

print("="*80)
print("MANUAL CONVOLUTION COMPUTATION")
print("="*80)

print(f"\nInput shape (HWC): {cpp_input.shape}")
print(f"Weight shape (OCHW): {weights.shape}")
print(f"Bias shape: {biases.shape}")

# Manually compute output at position [0, 0] for output channel 0
# Conv formula: output[oc, oh, ow] = sum over (ic, kh, kw) of input[ic, oh+kh, ow+kw] * weight[oc, ic, kh, kw] + bias[oc]

# For output position (oh=0, ow=0), output channel 0:
# We need input[0:3, 0:3, 0:3] (3 channels, 3x3 patch)

output_manual = 0.0
for ic in range(3):  # Input channels
    for kh in range(3):  # Kernel height
        for kw in range(3):  # Kernel width
            input_val = cpp_input[kh, kw, ic]  # Input is HWC
            weight_val = weights[0, ic, kh, kw]  # Weights are OCHW
            output_manual += input_val * weight_val

output_manual += biases[0]

print(f"\n{'='*80}")
print(f"MANUAL COMPUTATION AT OUTPUT[0,0,0]:")
print(f"{'='*80}")
print(f"Result: {output_manual:.10f}")

# Load C++ layer 0 output
with open('/tmp/cpp_pnet_layer0_after_conv_output.bin', 'rb') as f:
    num_channels = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_output = np.fromfile(f, dtype=np.float32).reshape(num_channels, height, width)

print(f"\nC++ output at [0,0,0]: {cpp_output[0, 0, 0]:.10f}")

# Python computation
import torch
import torch.nn.functional as F

py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
conv_weights = torch.from_numpy(weights).float()
conv_biases = torch.from_numpy(biases).float()

with torch.no_grad():
    py_output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)

print(f"Python output at [0,0,0]: {py_output[0, 0, 0, 0].item():.10f}")

# Differences
diff_manual_cpp = abs(output_manual - cpp_output[0, 0, 0])
diff_manual_py = abs(output_manual - py_output[0, 0, 0, 0].item())
diff_cpp_py = abs(cpp_output[0, 0, 0] - py_output[0, 0, 0, 0].item())

print(f"\n{'='*80}")
print(f"DIFFERENCES:")
print(f"{'='*80}")
print(f"Manual vs C++:    {diff_manual_cpp:.10f}")
print(f"Manual vs Python: {diff_manual_py:.10f}")
print(f"C++ vs Python:    {diff_cpp_py:.10f}")

# Show a few weight and input values to verify loading
print(f"\n{'='*80}")
print(f"SAMPLE WEIGHT VALUES:")
print(f"{'='*80}")
print(f"Weight[0,0,0,0]: {weights[0, 0, 0, 0]:.6f}")
print(f"Weight[0,1,1,1]: {weights[0, 1, 1, 1]:.6f}")
print(f"Weight[0,2,2,2]: {weights[0, 2, 2, 2]:.6f}")
print(f"Bias[0]: {biases[0]:.6f}")

print(f"\n{'='*80}")
print(f"SAMPLE INPUT VALUES:")
print(f"{'='*80}")
print(f"Input[0,0,0]: {cpp_input[0, 0, 0]:.6f}")
print(f"Input[1,1,1]: {cpp_input[1, 1, 1]:.6f}")
print(f"Input[2,2,2]: {cpp_input[2, 2, 2]:.6f}")

# Verification
if diff_manual_cpp < 1e-5 and diff_manual_py < 1e-5:
    print(f"\n✅ Both C++ and Python match manual computation!")
    print(f"   Weight extraction is CORRECT.")
elif diff_manual_cpp < 1e-5:
    print(f"\n❌ C++ matches manual, but Python does NOT!")
    print(f"   Python convolution has a bug.")
elif diff_manual_py < 1e-5:
    print(f"\n❌ Python matches manual, but C++ does NOT!")
    print(f"   C++ convolution or weight loading has a bug.")
else:
    print(f"\n❌ Neither C++ nor Python matches manual computation!")
    print(f"   Weight extraction is likely WRONG.")
