#!/usr/bin/env python3
"""
Test all saved weight matrices to find which one produces C++'s actual output.
"""

import numpy as np
import struct

# Load C++ ONet input
cpp_input = np.fromfile('/tmp/cpp_onet_input.bin', dtype=np.float32).reshape((48, 48, 3))

# Load C++ layer 0 output
with open('/tmp/cpp_layer0_after_conv_output.bin', 'rb') as f:
    cpp_num_channels = struct.unpack('<I', f.read(4))[0]
    cpp_height = struct.unpack('<I', f.read(4))[0]
    cpp_width = struct.unpack('<I', f.read(4))[0]
    cpp_layer0 = np.frombuffer(f.read(), dtype=np.float32).reshape(cpp_num_channels, cpp_height, cpp_width)

print("="*80)
print("TESTING ALL WEIGHT MATRICES")
print("="*80)
print(f"Target C++ output[0,0,0]: {cpp_layer0[0,0,0]:.6f}")
print()

# Focus on output pixel [0, 0, 0]
out_ch = 0
out_y = 0
out_x = 0
in_y = 0
in_x = 0

# Extract and flatten 3x3 patches in column-major order
patch = []
for in_ch in range(3):
    channel = cpp_input[in_y:in_y+3, in_x:in_x+3, in_ch]
    for xx in range(3):
        for yy in range(3):
            patch.append(channel[yy, xx])
patch.append(1.0)  # bias term
patch = np.array(patch, dtype=np.float32)

# Try all weight matrices
for i in range(10):
    try:
        with open(f'/tmp/cpp_conv{i}_weight.bin', 'rb') as f:
            rows = struct.unpack('<I', f.read(4))[0]
            cols = struct.unpack('<I', f.read(4))[0]
            weight = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)

        # Check if dimensions match (should be 28 rows for 3x3x3 + bias)
        if rows != 28:
            print(f"conv{i}: SKIP (rows={rows}, expected 28)")
            continue

        # Check if it has 32 output channels
        if cols != 32:
            print(f"conv{i}: SKIP (cols={cols}, expected 32)")
            continue

        # Compute output
        result = patch @ weight[:, out_ch]

        diff = abs(result - cpp_layer0[out_ch, out_y, out_x])
        if diff < 1e-4:
            print(f"conv{i}: ✅ MATCH! result={result:.6f}, diff={diff:.6e}")
        else:
            print(f"conv{i}: result={result:.6f}, diff={diff:.6e}")

    except FileNotFoundError:
        break
    except Exception as e:
        print(f"conv{i}: ERROR - {e}")
