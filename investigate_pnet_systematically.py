#!/usr/bin/env python3
"""
Systematic investigation of PNet divergence.
Check for systematic issues like channel swap, sign errors, etc.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Load C++ PNet input
cpp_input = np.fromfile('/tmp/cpp_pnet_input_scale0.bin', dtype=np.float32)
cpp_input = cpp_input.reshape(384, 216, 3)  # HWC

# Load extracted PNet layer 0 weights
weights = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_weights.npy')
biases = np.load('cpp_mtcnn_weights/pnet/pnet_layer00_conv_biases.npy')

print("="*80)
print("SYSTEMATIC PNET INVESTIGATION")
print("="*80)

# 1. Check if inputs are identical by comparing with what we saved
print("\n1. INPUT VERIFICATION:")
print(f"   Input shape: {cpp_input.shape}")
print(f"   Input range: [{cpp_input.min():.6f}, {cpp_input.max():.6f}]")
print(f"   Input mean: {cpp_input.mean():.6f}")
print(f"   Sample values:")
print(f"     [0,0,0]: {cpp_input[0,0,0]:.6f}")
print(f"     [0,0,1]: {cpp_input[0,0,1]:.6f}")
print(f"     [0,0,2]: {cpp_input[0,0,2]:.6f}")

# 2. Check weights for each output channel
print("\n2. WEIGHT ANALYSIS PER CHANNEL:")
for ch in range(10):
    w_mean = weights[ch].mean()
    w_std = weights[ch].std()
    w_min = weights[ch].min()
    w_max = weights[ch].max()
    b = biases[ch]
    print(f"   Channel {ch}: bias={b:+.6f}, weight mean={w_mean:+.6f}, std={w_std:.6f}, range=[{w_min:+.6f}, {w_max:+.6f}]")

# 3. Run Python convolution
py_input = torch.from_numpy(cpp_input).permute(2, 0, 1).unsqueeze(0).float()
conv_weights = torch.from_numpy(weights).float()
conv_biases = torch.from_numpy(biases).float()

with torch.no_grad():
    py_output = F.conv2d(py_input, conv_weights, conv_biases, stride=1, padding=0)

py_output_np = py_output[0].numpy()

# 4. Load C++ output
with open('/tmp/cpp_pnet_layer0_after_conv_output.bin', 'rb') as f:
    num_channels = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    cpp_output = np.fromfile(f, dtype=np.float32).reshape(num_channels, height, width)

print(f"\n3. OUTPUT SHAPES:")
print(f"   C++ output: {cpp_output.shape}")
print(f"   Python output: {py_output_np.shape}")

# 5. Check for systematic patterns
print(f"\n4. SYSTEMATIC PATTERN CHECKS:")

# Check if channels are swapped
print("\n   A. Channel swap test:")
for swap_ch in range(10):
    diff_with_swap = np.abs(cpp_output[1] - py_output_np[swap_ch]).mean()
    if swap_ch == 1:
        print(f"      Channel 1 vs {swap_ch} (self): mean diff = {diff_with_swap:.6f}")
    else:
        print(f"      Channel 1 vs {swap_ch}: mean diff = {diff_with_swap:.6f}")

# Check if there's a sign flip
print("\n   B. Sign flip test on channel 1:")
diff_normal = np.abs(cpp_output[1] - py_output_np[1]).mean()
diff_negated = np.abs(cpp_output[1] - (-py_output_np[1])).mean()
print(f"      Normal: mean diff = {diff_normal:.6f}")
print(f"      Negated: mean diff = {diff_negated:.6f}")
if diff_negated < diff_normal:
    print(f"      ⚠️  NEGATION IMPROVES MATCH!")

# Check correlation
correlation = np.corrcoef(cpp_output[1].flatten(), py_output_np[1].flatten())[0, 1]
print(f"\n   C. Correlation between C++ and Python channel 1: {correlation:.6f}")
if correlation < -0.5:
    print(f"      ⚠️  NEGATIVE CORRELATION - likely sign flip or inversion!")
elif correlation > 0.5:
    print(f"      ✓ POSITIVE CORRELATION - systematic relationship exists")

# 6. Sample specific values at different positions
print(f"\n5. SAMPLE VALUES AT DIFFERENT POSITIONS:")
positions = [(0, 0), (10, 10), (100, 50), (200, 100), (307, 121)]
for h, w in positions:
    cpp_val = cpp_output[1, h, w]
    py_val = py_output_np[1, h, w]
    diff = cpp_val - py_val
    ratio = cpp_val / py_val if py_val != 0 else float('inf')
    print(f"   Position [{h:3d},{w:3d}]: C++={cpp_val:+.6f}, Py={py_val:+.6f}, diff={diff:+.6f}, ratio={ratio:+.6f}")

# 7. Check if it's a constant offset
print(f"\n6. OFFSET ANALYSIS:")
diff_ch1 = cpp_output[1] - py_output_np[1]
print(f"   Difference stats for channel 1:")
print(f"     Mean: {diff_ch1.mean():.6f}")
print(f"     Std: {diff_ch1.std():.6f}")
print(f"     If std << mean, it's a constant offset")
print(f"     Std/Mean ratio: {abs(diff_ch1.std() / diff_ch1.mean()) if diff_ch1.mean() != 0 else float('inf'):.6f}")
