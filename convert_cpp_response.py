#!/usr/bin/env python3
"""
Convert C++ binary response map to numpy format.
"""
import numpy as np
import struct

# Read C++ binary response map
with open('/tmp/cpp_response_map_lm36_iter0.bin', 'rb') as f:
    rows = struct.unpack('i', f.read(4))[0]
    cols = struct.unpack('i', f.read(4))[0]
    data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
    response_map = data.reshape((rows, cols))

print(f"C++ response map loaded:")
print(f"  Shape: {response_map.shape}")
print(f"  min={response_map.min():.6f}, max={response_map.max():.6f}, mean={response_map.mean():.6f}")

# Find peak
peak_y, peak_x = np.unravel_index(response_map.argmax(), response_map.shape)
print(f"  Peak: ({peak_x}, {peak_y}) = {response_map[peak_y, peak_x]:.6f}")

# Save as numpy
np.save('/tmp/cpp_response_lm36_iter1.npy', response_map)
print(f"\nSaved to /tmp/cpp_response_lm36_iter1.npy")
