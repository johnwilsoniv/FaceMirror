#!/usr/bin/env python3
"""
Inspect C++ binary .dat file format to understand the exact structure.
"""

import struct
import os

model_path = os.path.expanduser("~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/face_detection/mtcnn/convert_to_cpp/PNet.dat")

print(f"Inspecting: {model_path}")
print(f"File size: {os.path.getsize(model_path)} bytes")
print("="*80)

with open(model_path, 'rb') as f:
    # Read network depth
    network_depth = struct.unpack('<i', f.read(4))[0]
    print(f"Network depth: {network_depth}")
    print()

    for layer_idx in range(network_depth):
        print(f"Layer {layer_idx}:")
        layer_type = struct.unpack('<i', f.read(4))[0]
        layer_names = ["Conv", "MaxPool", "FC", "PReLU", "Sigmoid"]
        print(f"  Type: {layer_type} ({layer_names[layer_type] if layer_type < len(layer_names) else 'Unknown'})")

        if layer_type == 0:  # Conv
            num_in_maps = struct.unpack('<i', f.read(4))[0]
            num_kernels = struct.unpack('<i', f.read(4))[0]
            print(f"  num_in_maps: {num_in_maps}")
            print(f"  num_kernels: {num_kernels}")

            # Read biases
            print(f"  Reading {num_kernels} biases...")
            biases = struct.unpack(f'<{num_kernels}f', f.read(4 * num_kernels))
            print(f"  Biases (first 3): {biases[:3]}")

            # Read kernels
            print(f"  Reading kernels...")
            for i in range(num_in_maps):
                for k in range(num_kernels):
                    h = struct.unpack('<i', f.read(4))[0]
                    w = struct.unpack('<i', f.read(4))[0]
                    print(f"    Kernel[{i}][{k}]: {h}x{w} = {h*w} floats")

                    if h < 0 or w < 0 or h > 100 or w > 100:
                        print(f"    ⚠ WARNING: Suspicious dimensions!")
                        print(f"    File position: {f.tell()}")
                        print(f"    Next 32 bytes (hex): {f.read(32).hex()}")
                        f.seek(-32, 1)  # Go back
                        break

                    # Read kernel data
                    kernel_data = struct.unpack(f'<{h*w}f', f.read(4 * h * w))
                    if i == 0 and k == 0:
                        print(f"    First kernel values (first 5): {kernel_data[:5]}")

        elif layer_type == 1:  # MaxPool
            kernel_size = struct.unpack('<i', f.read(4))[0]
            stride = struct.unpack('<i', f.read(4))[0]
            print(f"  kernel_size: {kernel_size}")
            print(f"  stride: {stride}")

        elif layer_type == 2:  # FC
            input_size = struct.unpack('<i', f.read(4))[0]
            output_size = struct.unpack('<i', f.read(4))[0]
            print(f"  input_size: {input_size}")
            print(f"  output_size: {output_size}")

            # Read weights and biases
            weights = struct.unpack(f'<{output_size * input_size}f', f.read(4 * output_size * input_size))
            biases = struct.unpack(f'<{output_size}f', f.read(4 * output_size))
            print(f"  Weights shape: ({output_size}, {input_size})")
            print(f"  Biases shape: ({output_size},)")

        elif layer_type == 3:  # PReLU
            num_channels = struct.unpack('<i', f.read(4))[0]
            print(f"  num_channels: {num_channels}")

            slopes = struct.unpack(f'<{num_channels}f', f.read(4 * num_channels))
            print(f"  Slopes (first 3): {slopes[:3]}")

        elif layer_type == 4:  # Sigmoid
            print(f"  (no parameters)")

        print()

    print(f"File position after reading: {f.tell()}/{os.path.getsize(model_path)}")
