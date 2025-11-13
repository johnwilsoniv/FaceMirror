#!/usr/bin/env python3
"""
Extract C++ OpenFace MTCNN Weights from Binary .dat Files

Parses the custom binary format used by C++ OpenFace and exports to
numpy arrays compatible with Python MTCNN implementations.

Binary format (from Write_CNN_to_binary.m):
- Little-endian
- Layer types: 0=conv, 1=max_pooling, 2=fc, 3=prelu, 4=sigmoid
- Conv layer: type(uint32), num_in_map(uint32), num_out_kerns(uint32), biases(float32[]), kernels(float32[])
- FC layer: type(uint32), bias(float32[]), weights(float32[])
- MaxPool layer: type(uint32), kernel_x(uint32), kernel_y(uint32), stride_x(uint32), stride_y(uint32)
- PReLU layer: type(uint32), weights(float32[])
"""

import struct
import numpy as np
from pathlib import Path
import json


def read_uint32(f):
    """Read a single uint32 (4 bytes, little-endian)."""
    return struct.unpack('<I', f.read(4))[0]


def read_float32(f):
    """Read a single float32 (4 bytes, little-endian)."""
    return struct.unpack('<f', f.read(4))[0]


def read_matrix(f):
    """
    Read a matrix in OpenFace binary format.

    Format (from writeMatrixBin):
    - rows (uint32)
    - cols (uint32)
    - type (uint32): 0=uint8, 1=int8, 2=uint16, 3=int16, 4=int, 5=float32, 6=float64
    - data (column-major order, transposed)

    Note: Matrix is stored transposed (column-major) and needs to be transposed back.
    """
    rows = read_uint32(f)
    cols = read_uint32(f)
    dtype_code = read_uint32(f)

    # Map type code to numpy dtype
    dtype_map = {
        0: np.uint8,
        1: np.int8,
        2: np.uint16,
        3: np.int16,
        4: np.int32,
        5: np.float32,
        6: np.float64
    }

    if dtype_code not in dtype_map:
        dtype = np.float32
    else:
        dtype = dtype_map[dtype_code]

    # Read matrix data (stored in row-major order, same as C++ cv::Mat)
    data = []
    for i in range(rows * cols):
        if dtype == np.float32:
            data.append(read_float32(f))
        else:
            # For now, only support float32
            raise ValueError(f"Unsupported dtype: {dtype}")

    # Reshape to (rows, cols) - NO transpose needed!
    # C++ ReadMatBin reads directly into cv::Mat without transposing
    matrix = np.array(data, dtype=dtype).reshape(rows, cols)
    return matrix


def parse_conv_layer(f):
    """
    Parse convolutional layer.

    Format:
    - num_in_map (uint32): number of input channels
    - num_out_kerns (uint32): number of output channels (kernels)
    - biases (float32[num_out_kerns])
    - kernels: for each input map, for each output kernel, read kernel matrix
    """
    num_in_map = read_uint32(f)
    num_out_kerns = read_uint32(f)

    print(f"      Conv: {num_in_map} input maps, {num_out_kerns} output kernels")

    # Read biases
    biases = []
    for k in range(num_out_kerns):
        biases.append(read_float32(f))
    biases = np.array(biases, dtype=np.float32)

    # Read kernels
    # MTCNN uses small kernels (typically 3x3 or 2x2)
    kernels = []
    for in_ch in range(num_in_map):
        for out_ch in range(num_out_kerns):
            kernel = read_matrix(f)
            kernels.append(kernel)

    # Reshape kernels: (out_channels, in_channels, height, width)
    # From C++: for k=1:num_in_map, for k2=1:num_out_kerns, W = squeeze(weights(:,:,k,k2))
    # This means weights are stored as [kernel_h, kernel_w, in_ch, out_ch]
    # We need [out_ch, in_ch, kernel_h, kernel_w] for PyTorch/ONNX

    kernel_h, kernel_w = kernels[0].shape
    weights = np.zeros((num_out_kerns, num_in_map, kernel_h, kernel_w), dtype=np.float32)

    idx = 0
    for in_ch in range(num_in_map):
        for out_ch in range(num_out_kerns):
            # CRITICAL: C++ builds im2col weight matrix by:
            # 1. Transposing each kernel (line 437)
            # 2. Flattening in column-major order
            # PyTorch needs the INVERSE of this transformation:
            # 1. Reshape with column-major order (order='F')
            # 2. Transpose to get correct orientation
            kernel_flat = kernels[idx].flatten()
            weights[out_ch, in_ch, :, :] = kernel_flat.reshape(kernel_h, kernel_w, order='F').T
            idx += 1

    return {
        'type': 'conv',
        'weights': weights,
        'biases': biases,
        'in_channels': num_in_map,
        'out_channels': num_out_kerns,
        'kernel_size': (kernel_h, kernel_w)
    }


def parse_fc_layer(f):
    """
    Parse fully connected layer.

    Format:
    - bias (matrix)
    - weights (matrix)

    Note: C++ stores FC weights as (in_features, out_features) for computation: out = W @ x
    PyTorch expects (out_features, in_features) for computation: out = x @ W.T
    So we need to transpose for PyTorch.
    """
    bias = read_matrix(f)
    weights = read_matrix(f)

    # Transpose for PyTorch: (in, out) -> (out, in)
    weights = weights.T

    print(f"      FC: {weights.shape[1]} → {weights.shape[0]}")

    return {
        'type': 'fc',
        'weights': weights,
        'bias': bias.flatten()
    }


def parse_maxpool_layer(f):
    """
    Parse max pooling layer.

    Format:
    - kernel_size_x (uint32)
    - kernel_size_y (uint32)
    - stride_x (uint32)
    - stride_y (uint32)
    """
    kernel_x = read_uint32(f)
    kernel_y = read_uint32(f)
    stride_x = read_uint32(f)
    stride_y = read_uint32(f)

    print(f"      MaxPool: kernel={kernel_x}x{kernel_y}, stride={stride_x}x{stride_y}")

    return {
        'type': 'maxpool',
        'kernel_size': (kernel_x, kernel_y),
        'stride': (stride_x, stride_y)
    }


def parse_prelu_layer(f):
    """
    Parse parametric ReLU layer.

    Format:
    - weights (matrix)
    """
    weights = read_matrix(f)

    print(f"      PReLU: {weights.shape}")

    return {
        'type': 'prelu',
        'weights': weights.flatten()
    }


def parse_mtcnn_network(dat_path):
    """
    Parse an MTCNN network from .dat file.

    Returns:
        dict: Network structure with layers
    """
    print(f"\n  Parsing {dat_path.name}...")

    with open(dat_path, 'rb') as f:
        # Read number of layers
        num_layers = read_uint32(f)
        print(f"    Layers: {num_layers}")

        layers = []
        for i in range(num_layers):
            layer_type = read_uint32(f)

            print(f"    Layer {i+1}/{num_layers}:", end=" ")

            if layer_type == 0:  # Convolutional
                layer = parse_conv_layer(f)
            elif layer_type == 1:  # Max pooling
                layer = parse_maxpool_layer(f)
            elif layer_type == 2:  # Fully connected
                layer = parse_fc_layer(f)
            elif layer_type == 3:  # PReLU
                layer = parse_prelu_layer(f)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            layers.append(layer)

    return {
        'num_layers': num_layers,
        'layers': layers
    }


def main():
    """Extract MTCNN weights from C++ OpenFace."""
    print("="*80)
    print("C++ OPENFACE MTCNN WEIGHT EXTRACTION")
    print("="*80)

    # Locate .dat files
    cpp_openface_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace"
    mtcnn_dir = cpp_openface_path / "lib/local/LandmarkDetector/model/mtcnn_detector"

    dat_files = {
        'PNet': mtcnn_dir / 'PNet.dat',
        'RNet': mtcnn_dir / 'RNet.dat',
        'ONet': mtcnn_dir / 'ONet.dat'
    }

    # Verify files exist
    for name, path in dat_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        print(f"✓ Found {name}: {path}")

    # Parse each network
    networks = {}
    for name, path in dat_files.items():
        networks[name] = parse_mtcnn_network(path)

    # Save as numpy arrays
    output_dir = Path("cpp_mtcnn_weights")
    output_dir.mkdir(exist_ok=True)

    print(f"\n" + "="*80)
    print("SAVING WEIGHTS")
    print("="*80)

    for net_name, network in networks.items():
        print(f"\n  {net_name}:")

        net_dir = output_dir / net_name.lower()
        net_dir.mkdir(exist_ok=True)

        # Save each layer
        for i, layer in enumerate(network['layers']):
            layer_prefix = f"{net_name.lower()}_layer{i:02d}_{layer['type']}"

            if layer['type'] == 'conv':
                weights_path = net_dir / f"{layer_prefix}_weights.npy"
                biases_path = net_dir / f"{layer_prefix}_biases.npy"
                np.save(weights_path, layer['weights'])
                np.save(biases_path, layer['biases'])
                print(f"    Layer {i}: Conv {layer['weights'].shape} + bias {layer['biases'].shape}")

            elif layer['type'] == 'fc':
                weights_path = net_dir / f"{layer_prefix}_weights.npy"
                bias_path = net_dir / f"{layer_prefix}_bias.npy"
                np.save(weights_path, layer['weights'])
                np.save(bias_path, layer['bias'])
                print(f"    Layer {i}: FC {layer['weights'].shape} + bias {layer['bias'].shape}")

            elif layer['type'] == 'prelu':
                weights_path = net_dir / f"{layer_prefix}_weights.npy"
                np.save(weights_path, layer['weights'])
                print(f"    Layer {i}: PReLU {layer['weights'].shape}")

            elif layer['type'] == 'maxpool':
                # Save metadata as JSON
                meta_path = net_dir / f"{layer_prefix}_params.json"
                with open(meta_path, 'w') as f:
                    json.dump({
                        'kernel_size': layer['kernel_size'],
                        'stride': layer['stride']
                    }, f, indent=2)
                print(f"    Layer {i}: MaxPool {layer['kernel_size']} stride {layer['stride']}")

        # Save network structure
        structure = {
            'num_layers': network['num_layers'],
            'layers': [
                {k: v for k, v in layer.items() if k != 'weights' and k != 'biases' and k != 'bias'}
                for layer in network['layers']
            ]
        }
        structure_path = net_dir / 'structure.json'
        with open(structure_path, 'w') as f:
            json.dump(structure, f, indent=2)
        print(f"    ✓ Saved structure: {structure_path}")

    print(f"\n" + "="*80)
    print("✅ EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nWeights saved to: {output_dir.absolute()}")
    print(f"\nNext step: Create Python MTCNN detector using these weights")


if __name__ == "__main__":
    main()
