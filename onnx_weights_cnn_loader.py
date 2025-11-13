#!/usr/bin/env python3
"""
Pure Python CNN loader for ONNX-extracted weights.
Loads from cpp_mtcnn_weights/ directory (numpy format).
"""

import numpy as np
import json
import os
from typing import List
from cpp_cnn_loader import ConvLayer, PReLULayer, MaxPoolLayer, FullyConnectedLayer


class ONNXWeightsCNN:
    """
    Pure Python CNN that loads weights extracted from ONNX models.
    This ensures exact parity with the ONNX detector.
    """

    def __init__(self, weights_dir: str):
        """
        Load CNN from ONNX-extracted weights directory.

        Args:
            weights_dir: Path to directory containing structure.json and weight .npy files
        """
        self.weights_dir = weights_dir

        # Load structure
        with open(os.path.join(weights_dir, 'structure.json'), 'r') as f:
            self.structure = json.load(f)

        print(f"Loading CNN from {weights_dir}")
        print(f"Network depth: {self.structure['num_layers']} layers")

        # Build layers
        self.layers = []
        layer_idx = 0

        for layer_info in self.structure['layers']:
            layer_type = layer_info['type']

            if layer_type == 'conv':
                # Load conv weights and biases
                weights = np.load(os.path.join(
                    weights_dir,
                    f"{os.path.basename(weights_dir)}_layer{layer_idx:02d}_conv_weights.npy"
                ))
                biases = np.load(os.path.join(
                    weights_dir,
                    f"{os.path.basename(weights_dir)}_layer{layer_idx:02d}_conv_biases.npy"
                ))

                in_ch = layer_info['in_channels']
                out_ch = layer_info['out_channels']
                k_h, k_w = layer_info['kernel_size']

                print(f"  Layer {layer_idx}: Conv ({in_ch}→{out_ch}, {k_h}x{k_w})")
                print(f"    Weights shape: {weights.shape}")
                print(f"    Biases shape: {biases.shape}")

                # ONNX format: (out_ch, in_ch, kh, kw)
                # ConvLayer expects: (num_in_maps, num_kernels, kernel_h, kernel_w, kernels, biases)
                layer = ConvLayer(in_ch, out_ch, k_h, k_w, weights, biases)
                self.layers.append(layer)

            elif layer_type == 'prelu':
                # Load PReLU slopes
                slopes = np.load(os.path.join(
                    weights_dir,
                    f"{os.path.basename(weights_dir)}_layer{layer_idx:02d}_prelu_weights.npy"
                ))

                print(f"  Layer {layer_idx}: PReLU ({len(slopes)} channels)")

                layer = PReLULayer(slopes)
                self.layers.append(layer)

            elif layer_type == 'maxpool':
                # Load maxpool params
                with open(os.path.join(
                    weights_dir,
                    f"{os.path.basename(weights_dir)}_layer{layer_idx:02d}_maxpool_params.json"
                ), 'r') as f:
                    params = json.load(f)

                k_h, k_w = params['kernel_size']
                s_h, s_w = params['stride']

                print(f"  Layer {layer_idx}: MaxPool ({k_h}x{k_w}, stride={s_h})")

                layer = MaxPoolLayer(kernel_size=k_h, stride=s_h)
                self.layers.append(layer)

            elif layer_type == 'fc':
                # Load FC weights and biases
                weights = np.load(os.path.join(
                    weights_dir,
                    f"{os.path.basename(weights_dir)}_layer{layer_idx:02d}_fc_weights.npy"
                ))
                biases = np.load(os.path.join(
                    weights_dir,
                    f"{os.path.basename(weights_dir)}_layer{layer_idx:02d}_fc_bias.npy"
                ))

                print(f"  Layer {layer_idx}: FC ({weights.shape[1]}→{weights.shape[0]})")
                print(f"    Weights shape: {weights.shape}")
                print(f"    Biases shape: {biases.shape}")

                # ONNX FC weights are (out, in), need to transpose for matrix multiply
                layer = FullyConnectedLayer(weights.T, biases)
                self.layers.append(layer)

            layer_idx += 1

        print(f"Successfully loaded {len(self.layers)} layers!")

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Forward pass through the network.

        Args:
            x: Input (3, H, W) for conv networks or (features,) for FC input

        Returns:
            List of outputs at each FC/Sigmoid layer
        """
        outputs = []
        current = x

        for i, layer in enumerate(self.layers):
            current = layer.forward(current)

            # MTCNN networks output intermediate results
            if isinstance(layer, FullyConnectedLayer):
                outputs.append(current.copy())

        return outputs if outputs else [current]

    def __call__(self, x: np.ndarray) -> List[np.ndarray]:
        """Callable interface"""
        return self.forward(x)


if __name__ == "__main__":
    # Test loading
    print("=" * 80)
    print("TESTING ONNX WEIGHTS CNN LOADER")
    print("=" * 80)

    rnet = ONNXWeightsCNN("cpp_mtcnn_weights/rnet")

    # Test with dummy input
    test_input = np.random.randn(3, 24, 24).astype(np.float32)
    print(f"\nTest input shape: {test_input.shape}")

    outputs = rnet(test_input)
    print(f"\nNumber of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")
